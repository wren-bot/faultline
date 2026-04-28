package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// Agent is the autonomous agent that runs in a continuous loop.
type Agent struct {
	cfg      *Config
	llm      *LLMClient
	memory   *MemoryStore
	index    *SearchIndex
	tools    *ToolExecutor
	telegram *Telegram
	sandbox  *Sandbox
	logger   *slog.Logger
}

// NewAgent creates and initializes a new agent.
func NewAgent(cfg *Config, telegram *Telegram, logger *slog.Logger) (*Agent, error) {
	memory, err := NewMemoryStore(cfg.Agent.MemoryDir)
	if err != nil {
		return nil, fmt.Errorf("init memory store: %w", err)
	}

	index := NewSearchIndex()

	// Build initial search index from existing memory files
	docs, err := memory.AllFiles()
	if err != nil {
		logger.Warn("failed to load memory files for indexing", "error", err)
	} else if len(docs) > 0 {
		index.Build(docs)
		logger.Info("search index built", "documents", len(docs))
	}

	// Set up sandbox if enabled
	var sandbox *Sandbox
	if cfg.Sandbox.Enabled {
		// Determine working directory for resolving relative paths
		workDir, err := os.Getwd()
		if err != nil {
			return nil, fmt.Errorf("get working directory: %w", err)
		}
		sandbox, err = NewSandbox(cfg.Sandbox, workDir, cfg.Log.Dir, logger)
		if err != nil {
			return nil, fmt.Errorf("init sandbox: %w", err)
		}
		logger.Info("sandbox enabled", "dir", cfg.Sandbox.Dir, "image", cfg.Sandbox.Image)
	} else {
		logger.Info("sandbox not configured, Python execution disabled")
	}

	llm := NewLLMClient(cfg.API.URL, cfg.API.Key, cfg.API.Model, logger)
	executor := NewToolExecutor(memory, index, telegram, sandbox, logger, cfg.Agent.MaxTokens)

	return &Agent{
		cfg:      cfg,
		llm:      llm,
		memory:   memory,
		index:    index,
		tools:    executor,
		telegram: telegram,
		sandbox:  sandbox,
		logger:   logger,
	}, nil
}

// Close releases resources held by the agent.
func (a *Agent) Close() {
	a.tools.Close()
	if a.sandbox != nil {
		a.sandbox.Close()
	}
}

// errShutdown is a sentinel error indicating a graceful shutdown was completed.
var errShutdown = fmt.Errorf("graceful shutdown completed")

// Run starts the agent's continuous operation loop.
// The agent runs indefinitely in a single conversation. When context reaches
// the compaction threshold, the agent is asked to save state and produce a
// summary, then the context is rebuilt and operation continues seamlessly.
// ctx is cancelled only on forced exit (second SIGINT).
// shutdownCh is closed on first SIGINT to trigger graceful save.
func (a *Agent) Run(ctx context.Context, shutdownCh <-chan struct{}) error {
	a.logger.Info("=== agent starting continuous operation ===")

	// Build initial context
	messages, prompts, err := a.initializeContext()
	if err != nil {
		return err
	}

	toolDefs := a.tools.ToolDefs()

	for {
		// Check for shutdown
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-shutdownCh:
			return a.handleShutdown(ctx, messages, toolDefs, prompts)
		default:
		}

		// Inject any pending operator messages before each turn
		messages, _ = a.injectPendingMessages(messages)
		a.drainWakeup()

		// Check if compaction is needed
		tokenEst := EstimateMessagesTokens(messages)
		if tokenEst > a.cfg.Agent.CompactionThreshold {
			a.logger.Warn("context at compaction threshold, compacting",
				"tokens_est", tokenEst, "threshold", a.cfg.Agent.CompactionThreshold)
			messages, prompts, err = a.compactContext(ctx, messages, toolDefs, prompts)
			if err != nil {
				return err
			}
			continue
		}

		// Hard safety limit - force compaction even if threshold wasn't hit
		if tokenEst > int(float64(a.cfg.Agent.MaxTokens)*0.95) {
			a.logger.Warn("context at hard limit, forcing compaction",
				"tokens_est", tokenEst, "max", a.cfg.Agent.MaxTokens)
			messages, prompts, err = a.compactContext(ctx, messages, toolDefs, prompts)
			if err != nil {
				return err
			}
			continue
		}

		// Create a per-turn context. If an operator message arrives while
		// the LLM is generating, we cancel this context to abort the
		// in-flight request so the agent sees the message immediately.
		turnCtx, turnCancel := context.WithCancel(ctx)
		if a.telegram != nil {
			go func() {
				select {
				case <-turnCtx.Done():
				case <-a.telegram.WakeupChan():
					turnCancel()
				}
			}()
		}

		// Send to LLM
		resp, err := a.llm.Chat(turnCtx, ChatRequest{
			Messages:    messages,
			Tools:       toolDefs,
			Temperature: a.cfg.Agent.Temperature,
			MaxTokens:   a.cfg.Agent.MaxRespTokens,
		})
		turnCancel()

		if err != nil {
			// Distinguish operator-message interrupt from real errors
			if turnCtx.Err() != nil && ctx.Err() == nil {
				a.logger.Info("turn interrupted by incoming operator message")
				continue
			}
			return fmt.Errorf("llm chat: %w", err)
		}

		msg := resp.Choices[0].Message
		messages = append(messages, msg)

		if msg.Content != "" {
			a.logThought(msg.Content)
		}

		// Handle tool calls
		if len(msg.ToolCalls) > 0 {
			a.tools.SetContextInfo(EstimateMessagesTokens(messages))

			for _, tc := range msg.ToolCalls {
				result := a.tools.Execute(ctx, tc)
				timestamped := fmt.Sprintf("[%s]\n%s", time.Now().Format(time.RFC1123), result)
				messages = append(messages, openai.ChatCompletionMessage{
					Role:       openai.ChatMessageRoleTool,
					Content:    timestamped,
					ToolCallID: tc.ID,
				})
			}
		} else {
			// Text-only response - inject continue prompt to keep going
			messages = append(messages, openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleUser,
				Content: RenderPrompt(prompts["continue"], time.Now()),
			})
		}

		a.logger.Debug("turn complete",
			"messages", len(messages),
			"tokens_est", EstimateMessagesTokens(messages))
	}
}

// initializeContext builds the initial conversation context.
func (a *Agent) initializeContext() ([]openai.ChatCompletionMessage, map[string]string, error) {
	// Build search index
	docs, err := a.memory.AllFiles()
	if err == nil && len(docs) > 0 {
		a.index.Build(docs)
		a.logger.Info("search index built", "documents", len(docs))
	}

	// Load all mutable prompts from disk (seeding defaults on first run)
	prompts, err := LoadAllPrompts(a.memory)
	if err != nil {
		return nil, nil, fmt.Errorf("load prompts: %w", err)
	}

	// Gather context memories
	memories := a.gatherContextMemories()

	// Build system prompt
	now := time.Now()
	fullSystemPrompt := BuildCycleContext(prompts["system"], memories, now)

	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: fullSystemPrompt,
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: RenderPrompt(prompts["cycle_start"], now),
		},
	}

	a.logger.Info("context initialized",
		"messages", len(messages),
		"tokens_est", EstimateMessagesTokens(messages))

	return messages, prompts, nil
}

// compactContext performs context compaction: asks the agent to save its state
// and produce a summary, then rebuilds the context with that summary.
func (a *Agent) compactContext(ctx context.Context, messages []openai.ChatCompletionMessage, toolDefs []openai.Tool, prompts map[string]string) ([]openai.ChatCompletionMessage, map[string]string, error) {
	a.logger.Info("starting context compaction")

	// Inject compaction prompt
	messages = append(messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: RenderPrompt(prompts["compaction"], time.Now()),
	})

	var summary string
	const maxCompactionTurns = 10

	for i := 0; i < maxCompactionTurns; i++ {
		// Safety: don't exceed hard token limit during compaction
		tokenEst := EstimateMessagesTokens(messages)
		if tokenEst > int(float64(a.cfg.Agent.MaxTokens)*0.98) {
			a.logger.Warn("approaching hard limit during compaction, using best available summary")
			break
		}

		resp, err := a.llm.Chat(ctx, ChatRequest{
			Messages:    messages,
			Tools:       toolDefs,
			Temperature: a.cfg.Agent.Temperature,
			MaxTokens:   a.cfg.Agent.MaxRespTokens,
		})
		if err != nil {
			return nil, nil, fmt.Errorf("llm chat during compaction: %w", err)
		}

		msg := resp.Choices[0].Message
		messages = append(messages, msg)

		if msg.Content != "" {
			summary = msg.Content
			a.logThought(msg.Content)
		}

		if len(msg.ToolCalls) > 0 {
			a.tools.SetContextInfo(EstimateMessagesTokens(messages))

			for _, tc := range msg.ToolCalls {
				result := a.tools.Execute(ctx, tc)
				timestamped := fmt.Sprintf("[%s]\n%s", time.Now().Format(time.RFC1123), result)
				messages = append(messages, openai.ChatCompletionMessage{
					Role:       openai.ChatMessageRoleTool,
					Content:    timestamped,
					ToolCallID: tc.ID,
				})
			}
		} else {
			// Text-only response - agent is done saving state
			break
		}
	}

	a.logger.Info("compaction complete, rebuilding context", "summary_len", len(summary))
	return a.rebuildContext(summary)
}

// rebuildContext creates a fresh conversation with the system prompt,
// memories, and an optional summary from compaction.
func (a *Agent) rebuildContext(summary string) ([]openai.ChatCompletionMessage, map[string]string, error) {
	// Rebuild search index in case files changed
	docs, err := a.memory.AllFiles()
	if err == nil && len(docs) > 0 {
		a.index.Build(docs)
	}

	// Reload prompts (agent may have modified them)
	prompts, err := LoadAllPrompts(a.memory)
	if err != nil {
		return nil, nil, fmt.Errorf("load prompts: %w", err)
	}

	// Load fresh memories
	memories := a.gatherContextMemories()
	now := time.Now()
	fullSystemPrompt := BuildCycleContext(prompts["system"], memories, now)

	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: fullSystemPrompt},
	}

	if summary != "" {
		messages = append(messages, openai.ChatCompletionMessage{
			Role: openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("[Context Compacted - %s]\n\nYour context was compacted. Here is your summary from before compaction:\n\n%s",
				now.Format(time.RFC1123), summary),
		})
	} else {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: RenderPrompt(prompts["continue"], now),
		})
	}

	a.logger.Info("context rebuilt",
		"messages", len(messages),
		"tokens_est", EstimateMessagesTokens(messages))

	return messages, prompts, nil
}

// isOperationalFile returns true for files that are loaded separately
// (prompts, trash) and should not be surfaced as memories.
func isOperationalFile(path string) bool {
	if strings.HasPrefix(path, "prompts/") {
		return true
	}
	if isTrashPath(path) {
		return true
	}
	return false
}

// gatherContextMemories finds relevant memories to include in context.
// Returns the most recently modified non-operational files.
func (a *Agent) gatherContextMemories() []SearchResult {
	var results []SearchResult

	// Get recent files (most recently modified)
	recent, err := a.memory.RecentFiles(5)
	if err == nil {
		for _, r := range recent {
			if !isOperationalFile(r.Path) {
				results = append(results, r)
			}
		}
	}

	// Cap total context memories to avoid blowing up the system prompt
	if len(results) > 5 {
		results = results[:5]
	}

	return results
}

// drainWakeup consumes any pending wakeup signal so the interrupt goroutine
// doesn't fire on stale signals from messages we already processed.
func (a *Agent) drainWakeup() {
	if a.telegram == nil {
		return
	}
	select {
	case <-a.telegram.WakeupChan():
	default:
	}
}

// wakeupChan returns a channel that signals when an operator message arrives.
// Returns a nil channel (blocks forever) if Telegram is not configured,
// so the select in the sleep loop falls through to the timer.
func (a *Agent) wakeupChan() <-chan struct{} {
	if a.telegram != nil {
		return a.telegram.WakeupChan()
	}
	return nil
}

// injectPendingMessages checks for queued operator messages and appends them
// to the conversation. Returns the updated messages and whether any were injected.
func (a *Agent) injectPendingMessages(messages []openai.ChatCompletionMessage) ([]openai.ChatCompletionMessage, bool) {
	if a.telegram == nil {
		return messages, false
	}

	pending := a.telegram.Pending()
	if len(pending) == 0 {
		return messages, false
	}

	for _, text := range pending {
		a.logger.Info("injecting operator message into conversation", "text", text)
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("[OPERATOR MESSAGE - RESPOND TO THIS BEFORE CONTINUING OTHER WORK]\n\nYour operator says: %s\n\nPlease acknowledge and respond to this message. Use send_message to reply to your operator, then you may continue what you were doing.", text),
		})
	}

	return messages, true
}

// handleShutdown wraps gracefulSave and translates errShutdown to nil.
func (a *Agent) handleShutdown(ctx context.Context, messages []openai.ChatCompletionMessage, toolDefs []openai.Tool, prompts map[string]string) error {
	err := a.gracefulSave(ctx, messages, toolDefs, prompts)
	if err == errShutdown {
		return nil
	}
	return err
}

// gracefulSave gives the agent a limited number of turns to save its state
// before the process exits. Uses a 2-minute timeout.
func (a *Agent) gracefulSave(ctx context.Context, messages []openai.ChatCompletionMessage, toolDefs []openai.Tool, prompts map[string]string) error {
	a.logger.Info("graceful shutdown: asking agent to save state")

	saveCtx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()

	// Load the mutable shutdown prompt
	shutdownPrompt := prompts["shutdown"]
	messages = append(messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: RenderPrompt(shutdownPrompt, time.Now()),
	})

	const maxSaveTurns = 10
	for i := 0; i < maxSaveTurns; i++ {
		resp, err := a.llm.Chat(saveCtx, ChatRequest{
			Messages:    messages,
			Tools:       toolDefs,
			Temperature: a.cfg.Agent.Temperature,
			MaxTokens:   a.cfg.Agent.MaxRespTokens,
		})
		if err != nil {
			a.logger.Error("LLM call failed during save", "error", err)
			return errShutdown
		}

		msg := resp.Choices[0].Message
		messages = append(messages, msg)

		if msg.Content != "" {
			a.logThought(msg.Content)
		}

		if len(msg.ToolCalls) > 0 {
			a.tools.SetContextInfo(EstimateMessagesTokens(messages))
			for _, tc := range msg.ToolCalls {
				result := a.tools.Execute(saveCtx, tc)
				timestamped := fmt.Sprintf("[%s]\n%s", time.Now().Format(time.RFC1123), result)
				messages = append(messages, openai.ChatCompletionMessage{
					Role:       openai.ChatMessageRoleTool,
					Content:    timestamped,
					ToolCallID: tc.ID,
				})
			}
		} else {
			// No tool calls - agent is done saving
			a.logger.Info("agent finished saving state (no more tool calls)")
			return errShutdown
		}
	}

	a.logger.Info("save turn limit reached, shutting down")
	return errShutdown
}

// logThought prints the agent's thought with some formatting.
func (a *Agent) logThought(content string) {
	// Show a preview in structured log
	preview := content
	if len(preview) > 300 {
		preview = preview[:300] + "..."
	}
	a.logger.Info("agent", "thought", preview)

	// Also print the full thought to stdout for live monitoring
	fmt.Println()
	fmt.Println(content)
	fmt.Println()
}
