//! Completion response types and streaming.

use super::common::{ContentBlock, Provider, StopReason, ToolCall, Usage};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// The unified response format from all providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Unique identifier for this completion.
    pub id: String,

    /// Provider that generated this response.
    pub provider: Provider,

    /// Model that generated this response.
    pub model: String,

    /// Generated content.
    pub content: Vec<ContentBlock>,

    /// Why generation stopped.
    pub stop_reason: StopReason,

    /// Token usage information.
    pub usage: Usage,

    /// Tool calls made by the model.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tool_calls: Vec<ToolCall>,

    /// Timestamp when response was created.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,

    /// Provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl CompletionResponse {
    /// Returns the concatenated text content from the response.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| {
                if matches!(b.content_type, Some(super::common::ContentType::Text)) {
                    b.text.as_deref()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Returns true if the response contains tool calls.
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

/// The type of a streaming event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamEventType {
    /// Placeholder used only by `StreamEvent::default()` as a base for struct-update syntax.
    /// A `StreamEvent` with this type is never yielded to callers.
    #[serde(skip)]
    Unset,
    /// Stream started.
    Start,
    /// Text content chunk.
    ContentDelta,
    /// Tool call started.
    ToolCallStart,
    /// Tool call input chunk.
    ToolCallDelta,
    /// Tool call finished.
    ToolCallEnd,
    /// Stream completed.
    Done,
    /// Error occurred.
    Error,
}

/// A single event in a streaming response.
#[derive(Debug, Clone)]
pub struct StreamEvent {
    /// Type of this event.
    pub event_type: StreamEventType,

    /// Content delta (for ContentDelta events).
    pub delta: Option<ContentBlock>,

    /// Index of the content block being updated.
    pub index: usize,

    /// Tool call information (for ToolCall* events).
    pub tool_call: Option<ToolCall>,

    /// Partial tool input JSON (for ToolCallDelta).
    pub tool_input_delta: Option<String>,

    /// Error information (for Error events).
    pub error: Option<crate::errors::RouterError>,

    /// Final usage stats (for Done events).
    pub usage: Option<Usage>,

    /// Stop reason (for Done events).
    pub stop_reason: Option<StopReason>,

    /// Response ID (for Start/Done events).
    pub response_id: Option<String>,

    /// Model (for Start events).
    pub model: Option<String>,
}

impl StreamEvent {
    pub fn start(response_id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            event_type: StreamEventType::Start,
            response_id: Some(response_id.into()),
            model: Some(model.into()),
            ..Default::default()
        }
    }

    pub fn content_delta(text: impl Into<String>, index: usize) -> Self {
        Self {
            event_type: StreamEventType::ContentDelta,
            delta: Some(ContentBlock::text(text)),
            index,
            ..Default::default()
        }
    }

    pub fn tool_call_start(tool_call: ToolCall) -> Self {
        Self {
            event_type: StreamEventType::ToolCallStart,
            tool_call: Some(tool_call),
            ..Default::default()
        }
    }

    pub fn tool_call_delta(delta: impl Into<String>, index: usize) -> Self {
        Self {
            event_type: StreamEventType::ToolCallDelta,
            tool_input_delta: Some(delta.into()),
            index,
            ..Default::default()
        }
    }

    pub fn tool_call_end(tool_call: ToolCall) -> Self {
        Self {
            event_type: StreamEventType::ToolCallEnd,
            tool_call: Some(tool_call),
            ..Default::default()
        }
    }

    pub fn done(
        usage: Option<Usage>,
        stop_reason: Option<StopReason>,
        response_id: Option<String>,
    ) -> Self {
        Self {
            event_type: StreamEventType::Done,
            usage,
            stop_reason,
            response_id,
            ..Default::default()
        }
    }

    pub fn error(err: crate::errors::RouterError) -> Self {
        Self {
            event_type: StreamEventType::Error,
            error: Some(err),
            ..Default::default()
        }
    }
}

impl Default for StreamEvent {
    /// Returns an "unset" base event used only with struct-update syntax (`..Default::default()`).
    /// Direct callers should use the named constructors (`StreamEvent::start`, `StreamEvent::done`, etc.).
    fn default() -> Self {
        Self {
            event_type: StreamEventType::Unset,
            delta: None,
            index: 0,
            tool_call: None,
            tool_input_delta: None,
            error: None,
            usage: None,
            stop_reason: None,
            response_id: None,
            model: None,
        }
    }
}
