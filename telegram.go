package main

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"unicode/utf8"

	tgmd "github.com/Mad-Pixels/goldmark-tgmd"
	tgbotapi "github.com/go-telegram-bot-api/telegram-bot-api/v5"
)

// Telegram manages bidirectional communication with the operator.
type Telegram struct {
	bot    *tgbotapi.BotAPI
	chatID int64
	logger *slog.Logger

	mu      sync.Mutex
	pending []string
	wakeup  chan struct{}
}

// NewTelegram creates a new Telegram bot connection.
// Clears any existing webhook so long polling works.
func NewTelegram(token string, chatID int64, logger *slog.Logger) (*Telegram, error) {
	bot, err := tgbotapi.NewBotAPI(token)
	if err != nil {
		return nil, fmt.Errorf("create telegram bot: %w", err)
	}

	logger.Info("telegram bot connected", "username", bot.Self.UserName)

	// Delete any existing webhook - webhooks block getUpdates (long polling).
	// This is the most common reason for the bot silently not receiving messages.
	deleteWebhook := tgbotapi.DeleteWebhookConfig{DropPendingUpdates: false}
	if _, err := bot.Request(deleteWebhook); err != nil {
		logger.Warn("failed to delete webhook (may not matter)", "error", err)
	} else {
		logger.Debug("cleared any existing webhook")
	}

	return &Telegram{
		bot:    bot,
		chatID: chatID,
		logger: logger,
		wakeup: make(chan struct{}, 1),
	}, nil
}

// Start begins listening for incoming messages.
// It blocks until the context is cancelled.
func (t *Telegram) Start(ctx context.Context) {
	u := tgbotapi.NewUpdate(0)
	u.Timeout = 30

	updates := t.bot.GetUpdatesChan(u)
	t.logger.Info("telegram listener started, waiting for messages")

	go func() {
		<-ctx.Done()
		t.bot.StopReceivingUpdates()
	}()

	for update := range updates {
		if ctx.Err() != nil {
			return
		}

		t.logger.Debug("telegram update received",
			"update_id", update.UpdateID,
			"has_message", update.Message != nil,
		)

		if update.Message == nil {
			continue
		}

		t.logger.Debug("telegram message",
			"chat_id", update.Message.Chat.ID,
			"from", update.Message.From.UserName,
			"text_len", len(update.Message.Text),
		)

		// Only accept messages from the configured chat
		if update.Message.Chat.ID != t.chatID {
			t.logger.Warn("ignoring message from unknown chat",
				"chat_id", update.Message.Chat.ID,
				"expected_chat_id", t.chatID,
				"username", update.Message.From.UserName,
			)
			continue
		}

		text := update.Message.Text
		if text == "" {
			continue
		}

		t.logger.Info("received message from operator", "text", text)

		t.mu.Lock()
		t.pending = append(t.pending, text)
		t.mu.Unlock()

		// Signal the agent to wake up if it's sleeping
		select {
		case t.wakeup <- struct{}{}:
		default:
			// Already signaled, don't block
		}
	}

	t.logger.Info("telegram listener stopped")
}

// mdConverter is the goldmark instance configured for Telegram MarkdownV2 output.
var mdConverter = tgmd.TGMD()

// listBullets are the Unicode bullet characters used by goldmark-tgmd.
// Used for post-processing to fix the extra newline after bullets.
var listBullets = []string{"•", "‣", "⁃"}

// toTelegramMarkdown converts standard markdown to Telegram MarkdownV2 format.
// Returns the converted text and true on success, or the original text and false on failure.
func toTelegramMarkdown(text string) (string, bool) {
	var buf bytes.Buffer
	if err := mdConverter.Convert([]byte(text), &buf); err != nil {
		return text, false
	}
	result := buf.String()
	if result == "" {
		return text, false
	}

	// Fix goldmark-tgmd bug: paragraph nodes inside list items emit an
	// extra newline after the bullet, producing "• \ntext" instead of "• text".
	for _, bullet := range listBullets {
		result = strings.ReplaceAll(result, bullet+" \n", bullet+" ")
	}

	return result, true
}

// Send sends a text message to the operator.
func (t *Telegram) Send(text string) error {
	// Telegram has a 4096 character limit per message.
	// Split long messages.
	const maxLen = 4000

	for len(text) > 0 {
		chunk := text
		if len(chunk) > maxLen {
			// Find a safe UTF-8 boundary at or before maxLen
			cut := maxLen
			for cut > 0 && !utf8.RuneStart(text[cut]) {
				cut--
			}
			// Try to split at a newline within the last 500 bytes
			for i := cut; i > cut-500 && i > 0; i-- {
				if text[i] == '\n' {
					cut = i + 1
					break
				}
			}
			chunk = text[:cut]
			text = text[cut:]
		} else {
			text = ""
		}

		// Convert markdown to Telegram MarkdownV2
		converted, ok := toTelegramMarkdown(chunk)
		if ok {
			msg := tgbotapi.NewMessage(t.chatID, converted)
			msg.ParseMode = tgbotapi.ModeMarkdownV2
			if _, err := t.bot.Send(msg); err != nil {
				// MarkdownV2 send failed -- fall back to plain text
				t.logger.Debug("markdownV2 send failed, retrying as plain text", "error", err)
				msg = tgbotapi.NewMessage(t.chatID, chunk)
				if _, err := t.bot.Send(msg); err != nil {
					return fmt.Errorf("send telegram message: %w", err)
				}
			}
		} else {
			// Conversion failed -- send as plain text
			msg := tgbotapi.NewMessage(t.chatID, chunk)
			if _, err := t.bot.Send(msg); err != nil {
				return fmt.Errorf("send telegram message: %w", err)
			}
		}
	}

	return nil
}

// Pending drains and returns all queued incoming messages.
func (t *Telegram) Pending() []string {
	t.mu.Lock()
	defer t.mu.Unlock()

	if len(t.pending) == 0 {
		return nil
	}

	msgs := t.pending
	t.pending = nil
	return msgs
}

// WakeupChan returns a channel that signals when a new message arrives.
func (t *Telegram) WakeupChan() <-chan struct{} {
	return t.wakeup
}
