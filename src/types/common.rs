//! Common types shared across all providers.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported LLM providers.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    #[serde(rename = "openai")]
    OpenAI,
    #[serde(rename = "anthropic")]
    Anthropic,
    #[serde(rename = "google")]
    Google,
    #[serde(rename = "vertex")]
    Vertex,
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provider::OpenAI => write!(f, "openai"),
            Provider::Anthropic => write!(f, "anthropic"),
            Provider::Google => write!(f, "google"),
            Provider::Vertex => write!(f, "vertex"),
        }
    }
}

/// Message roles in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

/// The type of content in a content block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentType {
    Text,
    Image,
    ToolUse,
    ToolResult,
}

/// A piece of content (text, image, tool use, etc.).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub content_type: Option<ContentType>,

    // For text content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    // For image content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,

    // For tool use (assistant calling a tool)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_input: Option<serde_json::Value>,

    // For tool result (user providing tool output)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_result_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

impl ContentBlock {
    /// Create a text content block.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content_type: Some(ContentType::Text),
            text: Some(text.into()),
            ..Default::default()
        }
    }

    /// Create an image content block with a URL.
    pub fn image_url(url: impl Into<String>) -> Self {
        Self {
            content_type: Some(ContentType::Image),
            image_url: Some(url.into()),
            ..Default::default()
        }
    }

    /// Create an image content block with base64 data.
    pub fn image_base64(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self {
            content_type: Some(ContentType::Image),
            image_base64: Some(data.into()),
            media_type: Some(media_type.into()),
            ..Default::default()
        }
    }

    /// Create a tool use content block.
    pub fn tool_use(
        id: impl Into<String>,
        name: impl Into<String>,
        input: serde_json::Value,
    ) -> Self {
        Self {
            content_type: Some(ContentType::ToolUse),
            tool_use_id: Some(id.into()),
            tool_name: Some(name.into()),
            tool_input: Some(input),
            ..Default::default()
        }
    }

    /// Create a tool result content block.
    pub fn tool_result(
        tool_result_id: impl Into<String>,
        result: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self {
            content_type: Some(ContentType::ToolResult),
            tool_result_id: Some(tool_result_id.into()),
            text: Some(result.into()),
            is_error: if is_error { Some(true) } else { None },
            ..Default::default()
        }
    }
}

/// A conversation message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Create a simple text message.
    pub fn new_text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentBlock::text(text)],
        }
    }

    /// Create a tool result message.
    pub fn new_tool_result(
        tool_use_id: impl Into<String>,
        result: impl Into<String>,
        is_error: bool,
    ) -> Self {
        let id = tool_use_id.into();
        Self {
            role: Role::Tool,
            content: vec![ContentBlock::tool_result(id, result, is_error)],
        }
    }
}

/// A function/tool that the model can use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: JsonSchema,
}

/// A tool invocation by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// A JSON Schema definition (unified format translated to provider-specific formats).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JsonSchema {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub schema_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, JsonSchema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<JsonSchema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<serde_json::Value>>,
    #[serde(rename = "const", skip_serializing_if = "Option::is_none")]
    pub const_value: Option<serde_json::Value>,
    #[serde(
        rename = "additionalProperties",
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_properties: Option<bool>,
    #[serde(rename = "minItems", skip_serializing_if = "Option::is_none")]
    pub min_items: Option<i64>,
    #[serde(rename = "maxItems", skip_serializing_if = "Option::is_none")]
    pub max_items: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,
    #[serde(rename = "minLength", skip_serializing_if = "Option::is_none")]
    pub min_length: Option<i64>,
    #[serde(rename = "maxLength", skip_serializing_if = "Option::is_none")]
    pub max_length: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    #[serde(rename = "anyOf", skip_serializing_if = "Option::is_none")]
    pub any_of: Option<Vec<JsonSchema>>,
    #[serde(rename = "oneOf", skip_serializing_if = "Option::is_none")]
    pub one_of: Option<Vec<JsonSchema>>,
    #[serde(rename = "allOf", skip_serializing_if = "Option::is_none")]
    pub all_of: Option<Vec<JsonSchema>>,
    #[serde(rename = "$ref", skip_serializing_if = "Option::is_none")]
    pub ref_: Option<String>,
    #[serde(rename = "$defs", skip_serializing_if = "Option::is_none")]
    pub defs: Option<HashMap<String, JsonSchema>>,
}

impl JsonSchema {
    /// Convert JSONSchema to a serde_json::Value map.
    pub fn to_map(&self) -> serde_json::Map<String, serde_json::Value> {
        let v = serde_json::to_value(self).unwrap_or(serde_json::Value::Object(Default::default()));
        match v {
            serde_json::Value::Object(m) => m,
            _ => Default::default(),
        }
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    #[default]
    End,
    MaxTokens,
    ToolUse,
    StopSequence,
    ContentFilter,
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopReason::End => write!(f, "end"),
            StopReason::MaxTokens => write!(f, "max_tokens"),
            StopReason::ToolUse => write!(f, "tool_use"),
            StopReason::StopSequence => write!(f, "stop_sequence"),
            StopReason::ContentFilter => write!(f, "content_filter"),
        }
    }
}

/// Token usage information.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub total_tokens: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<i64>,
}

/// Provider capabilities.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Feature {
    Streaming,
    StructuredOutput,
    Tools,
    Vision,
    Batch,
    Json,
}

impl std::fmt::Display for Feature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Feature::Streaming => write!(f, "streaming"),
            Feature::StructuredOutput => write!(f, "structured_output"),
            Feature::Tools => write!(f, "tools"),
            Feature::Vision => write!(f, "vision"),
            Feature::Batch => write!(f, "batch"),
            Feature::Json => write!(f, "json_mode"),
        }
    }
}
