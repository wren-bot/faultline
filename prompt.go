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
func BuildCycleContext(systemPrompt string, memories []SearchResult, now time.Time) string {
	var sb strings.Builder

	sb.WriteString(systemPrompt)
	sb.WriteString("\n\n---\n\n")
	fmt.Fprintf(&sb, "**Current Time**: %s\n\n", now.Format(time.RFC1123))

	if len(memories) > 0 {
		sb.WriteString("## Recent Memories\n\n")
		for _, m := range memories {
			fmt.Fprintf(&sb, "### %s\n", m.Path)
			content := m.Content
			if len(content) > 2000 {
				content = content[:2000] + "\n\n*[truncated]*"
			}
			sb.WriteString(content)
			sb.WriteString("\n\n")
		}
	}

	return sb.String()
}

// RenderPrompt takes a prompt string and substitutes known placeholders.
func RenderPrompt(template string, now time.Time) string {
	result := template
	result = strings.ReplaceAll(result, "{{TIME}}", now.Format(time.RFC1123))
	return result
}
