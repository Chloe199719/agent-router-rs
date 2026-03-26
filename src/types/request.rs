//! Completion request types.

use std::collections::HashMap;

use super::common::{JsonSchema, Message, Provider, Tool};
use serde::{Deserialize, Serialize};

/// The unified request format for all providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Provider to use for this request.
    pub provider: Provider,

    /// Model identifier (provider-specific).
    pub model: String,

    /// Messages in the conversation.
    pub messages: Vec<Message>,

    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,

    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Top-p nucleus sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Top-k sampling (Anthropic/Google only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,

    /// Stop sequences.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub stop_sequences: Vec<String>,

    /// Structured output configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// Tool/function calling definitions.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tools: Vec<Tool>,

    /// Tool choice configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Enable streaming (set by Router::stream).
    #[serde(default)]
    pub stream: bool,

    /// Provider-specific options (passed through without modification).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,

    /// Optional string key–value metadata for tracing, billing labels, or dashboards.
    ///
    /// Not merged with [`Self::extra`]. Mapping is provider-specific: OpenAI forwards
    /// as chat `metadata`; Vertex merges into Gemini `labels`; Google (AI Studio) ignores
    /// it; Anthropic only forwards the `user_id` key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

impl CompletionRequest {
    /// Create a new completion request.
    pub fn new(provider: Provider, model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            provider,
            model: model.into(),
            messages,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: Vec::new(),
            response_format: None,
            tools: Vec::new(),
            tool_choice: None,
            stream: false,
            extra: None,
            metadata: None,
        }
    }

    /// Set request metadata (replaces any previous map).
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = if metadata.is_empty() {
            None
        } else {
            Some(metadata)
        };
        self
    }

    /// Set max tokens.
    pub fn with_max_tokens(mut self, n: i32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Add tools to the request.
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Add a single tool.
    pub fn with_tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    /// Set JSON schema response format.
    pub fn with_json_schema(mut self, name: impl Into<String>, schema: JsonSchema) -> Self {
        self.response_format = Some(ResponseFormat {
            format_type: "json_schema".to_string(),
            schema: Some(schema),
            name: Some(name.into()),
            description: None,
            strict: Some(true),
        });
        self
    }

    /// Enable streaming.
    pub fn with_stream(mut self) -> Self {
        self.stream = true;
        self
    }
}

/// Configures structured output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    /// Type: "text", "json", or "json_schema"
    #[serde(rename = "type")]
    pub format_type: String,

    /// Schema for structured output (when type is "json_schema").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<JsonSchema>,

    /// Name for the schema (required by some providers).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Description of what the schema represents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Strict mode (OpenAI).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// How the model should use tools.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceType {
    /// Model decides whether to use tools.
    Auto,
    /// Model must use at least one tool.
    Required,
    /// Model cannot use tools.
    None,
    /// Model must use a specific tool.
    Tool,
}

/// Controls how the model uses tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoice {
    /// Type of tool choice.
    #[serde(rename = "type")]
    pub choice_type: ToolChoiceType,

    /// Name of specific tool (when type is Tool).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Prevent parallel tool use (Anthropic).
    #[serde(default)]
    pub disable_parallel_tool_use: bool,
}

impl ToolChoice {
    pub fn auto() -> Self {
        Self {
            choice_type: ToolChoiceType::Auto,
            name: None,
            disable_parallel_tool_use: false,
        }
    }

    pub fn required() -> Self {
        Self {
            choice_type: ToolChoiceType::Required,
            name: None,
            disable_parallel_tool_use: false,
        }
    }

    pub fn none() -> Self {
        Self {
            choice_type: ToolChoiceType::None,
            name: None,
            disable_parallel_tool_use: false,
        }
    }

    pub fn tool(name: impl Into<String>) -> Self {
        Self {
            choice_type: ToolChoiceType::Tool,
            name: Some(name.into()),
            disable_parallel_tool_use: false,
        }
    }
}
