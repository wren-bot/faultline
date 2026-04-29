package main

import (
	_ "embed"
	"fmt"
	"strings"
	"time"
)

// Embedded default prompt contents, compiled into the binary from prompts/*.md.
var (
	//go:embed prompts/system.md
	defaultSystemPrompt string

	//go:embed prompts/compaction.md
	defaultCompactionPrompt string

	//go:embed prompts/cycle_start.md
	defaultCycleStartPrompt string

	//go:embed prompts/continue.md
	defaultContinuePrompt string

	//go:embed prompts/shutdown.md
	defaultShutdownPrompt string
)

// promptFile defines a mutable prompt file with its default seed content.
type promptFile struct {
	path         string
	defaultValue string
}

// promptFiles maps prompt names to their memory paths and embedded defaults.
// Initialized in init() after embed variables are populated.
var promptFiles map[string]promptFile

func init() {
	promptFiles = map[string]promptFile{
		"system": {
			path:         "prompts/system.md",
			defaultValue: defaultSystemPrompt,
		},
		"compaction": {
			path:         "prompts/compaction.md",
			defaultValue: defaultCompactionPrompt,
		},
		"cycle_start": {
			path:         "prompts/cycle_start.md",
			defaultValue: defaultCycleStartPrompt,
		},
		"continue": {
			path:         "prompts/continue.md",
			defaultValue: defaultContinuePrompt,
		},
		"shutdown": {
			path:         "prompts/shutdown.md",
			defaultValue: defaultShutdownPrompt,
		},
	}
}

// LoadPrompt loads a prompt from disk, seeding the default if it doesn't exist yet.
func LoadPrompt(memory *MemoryStore, name string) (string, error) {
	pf, ok := promptFiles[name]
	if !ok {
		return "", fmt.Errorf("unknown prompt: %s", name)
	}

	content, err := memory.Read(pf.path)
	if err == nil && content != "" {
		return content, nil
	}

	// First run - seed the default
	if err := memory.Write(pf.path, pf.defaultValue); err != nil {
		return "", fmt.Errorf("write default prompt %s: %w", name, err)
	}
	return pf.defaultValue, nil
}

// LoadAllPrompts loads all prompt files, seeding defaults as needed.
// Returns a map of prompt name -> content.
func LoadAllPrompts(memory *MemoryStore) (map[string]string, error) {
	prompts := make(map[string]string)
	for name := range promptFiles {
		content, err := LoadPrompt(memory, name)
		if err != nil {
			return nil, err
		}
		prompts[name] = content
	}
	return prompts, nil
}

// BuildCycleContext assembles the full system message with recent memories.
// memoryCharLimit caps the per-entry content size; when exceeded, a
// retrieval hint is appended pointing the agent at memory_read so it can
// load the rest of the file. A non-positive limit disables the cap.
func BuildCycleContext(systemPrompt string, memories []SearchResult, now time.Time, memoryCharLimit int) string {
	var sb strings.Builder

	sb.WriteString(systemPrompt)
	sb.WriteString("\n\n---\n\n")
	fmt.Fprintf(&sb, "**Current Time**: %s\n\n", now.Format(time.RFC1123))

	if len(memories) > 0 {
		sb.WriteString("## Recent Memories\n\n")
		for _, m := range memories {
			fmt.Fprintf(&sb, "### %s\n", m.Path)
			content := m.Content
			total := len(content)
			if memoryCharLimit > 0 && total > memoryCharLimit {
				content = content[:memoryCharLimit]
				sb.WriteString(content)
				fmt.Fprintf(&sb,
					"\n\n*[truncated: showing first %d of %d chars; call `memory_read` with path=%q to read the full file, or with offset=%d (line-based) to continue from where this preview ends]*",
					memoryCharLimit, total, m.Path, lineCountFor(content)+1)
			} else {
				sb.WriteString(content)
			}
			sb.WriteString("\n\n")
		}
	}

	return sb.String()
}

// lineCountFor returns the number of newline-delimited lines in s.
// A trailing newline does not add an extra line. Used to build retrieval
// hints that tell the agent where to resume reading after a truncated
// preview.
func lineCountFor(s string) int {
	if s == "" {
		return 0
	}
	n := strings.Count(s, "\n")
	if !strings.HasSuffix(s, "\n") {
		n++
	}
	return n
}

// RenderPrompt takes a prompt string and substitutes known placeholders.
func RenderPrompt(template string, now time.Time) string {
	result := template
	result = strings.ReplaceAll(result, "{{TIME}}", now.Format(time.RFC1123))
	return result
}
