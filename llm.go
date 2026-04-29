package main

import (
	"context"
	"fmt"
	"log/slog"

	openai "github.com/sashabaranov/go-openai"
)

// LLMClient wraps the OpenAI-compatible API client.
type LLMClient struct {
	client *openai.Client
	model  string
	logger *slog.Logger

	// lastLoggedAt is the index of the first message that has not yet been
	// debug-logged. On each Chat() call we only log messages with index >=
	// lastLoggedAt, then advance it to len(messages). This avoids re-logging
	// the entire conversation on every turn (the message list grows
	// monotonically within a single context lifetime).
	//
	// Assumption: the message slice grows append-only between calls. When
	// the agent rebuilds context (compaction, restart) the new slice is
	// shorter than lastLoggedAt; we detect that and reset to log the full
	// new context once. Any other shrinkage will also trigger a full
	// re-log on the next call, which is cosmetic noise but not incorrect.
	lastLoggedAt int
}

// NewLLMClient creates a new LLM client configured for the given endpoint.
func NewLLMClient(apiURL, apiKey, model string, logger *slog.Logger) *LLMClient {
	if apiKey == "" {
		apiKey = "not-needed"
	}

	config := openai.DefaultConfig(apiKey)
	config.BaseURL = apiURL

	return &LLMClient{
		client: openai.NewClientWithConfig(config),
		model:  model,
		logger: logger,
	}
}

// ChatRequest holds the parameters for a chat completion request.
type ChatRequest struct {
	Messages    []openai.ChatCompletionMessage
	Tools       []openai.Tool
	Temperature float32
	MaxTokens   int
}

// Chat sends a chat completion request and returns the response.
func (l *LLMClient) Chat(ctx context.Context, req ChatRequest) (*openai.ChatCompletionResponse, error) {
	ccr := openai.ChatCompletionRequest{
		Model:       l.model,
		Messages:    req.Messages,
		Temperature: req.Temperature,
	}

	if len(req.Tools) > 0 {
		ccr.Tools = req.Tools
	}

	if req.MaxTokens > 0 {
		ccr.MaxTokens = req.MaxTokens
	}

	l.logger.Debug("sending chat request",
		"messages", len(req.Messages),
		"tools", len(req.Tools),
		"model", l.model,
	)

	// Only log NEW messages since the last request (avoid re-logging the full context).
	// See the doc comment on lastLoggedAt for the invariant being maintained here.
	start := l.lastLoggedAt
	if len(req.Messages) < start {
		// Message list shrank: context was rebuilt (compaction or fresh
		// run). Log the entire new context once.
		start = 0
	}
	for i := start; i < len(req.Messages); i++ {
		m := req.Messages[i]
		l.logger.Debug(">>> message",
			"index", i,
			"role", m.Role,
			"content", m.Content,
			"tool_call_id", m.ToolCallID,
		)
		for _, tc := range m.ToolCalls {
			l.logger.Debug(">>> tool_call",
				"index", i,
				"id", tc.ID,
				"function", tc.Function.Name,
				"arguments", tc.Function.Arguments,
			)
		}
	}
	l.lastLoggedAt = len(req.Messages)

	resp, err := l.client.CreateChatCompletion(ctx, ccr)
	if err != nil {
		return nil, fmt.Errorf("chat completion: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	msg := resp.Choices[0].Message

	l.logger.Debug("<<< response",
		"finish_reason", resp.Choices[0].FinishReason,
		"content", msg.Content,
	)
	for _, tc := range msg.ToolCalls {
		l.logger.Debug("<<< tool_call",
			"id", tc.ID,
			"function", tc.Function.Name,
			"arguments", tc.Function.Arguments,
		)
	}

	return &resp, nil
}

// EstimateTokens provides a rough token count for a string.
// Uses the approximation of ~4 characters per token for English text.
func EstimateTokens(text string) int {
	if len(text) == 0 {
		return 0
	}
	return len(text) / 4
}

// EstimateMessagesTokens estimates total tokens across all messages.
func EstimateMessagesTokens(messages []openai.ChatCompletionMessage) int {
	total := 0
	for _, m := range messages {
		total += EstimateTokens(m.Content)
		// Account for role and message overhead
		total += 4
		for _, tc := range m.ToolCalls {
			total += EstimateTokens(tc.Function.Name)
			total += EstimateTokens(tc.Function.Arguments)
			total += 4
		}
	}
	return total
}
