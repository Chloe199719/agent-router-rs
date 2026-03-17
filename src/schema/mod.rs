//! JSON Schema translation between providers.

use crate::types::{JsonSchema, ResponseFormat, Tool};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Translates unified JSONSchema to provider-specific formats.
pub struct Translator;

impl Translator {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Translator {
    fn default() -> Self {
        Self::new()
    }
}

// ---- OpenAI Format ----

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<OpenAIJsonSchema>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIJsonSchema {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub schema: serde_json::Map<String, serde_json::Value>,
    pub strict: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAITool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAIFunctionTool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIFunctionTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Map<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl Translator {
    /// Convert unified schema to OpenAI format.
    pub fn to_openai(&self, rf: &ResponseFormat) -> Option<OpenAIResponseFormat> {
        match rf.format_type.as_str() {
            "json" => Some(OpenAIResponseFormat {
                format_type: "json_object".to_string(),
                json_schema: None,
            }),
            "json_schema" => {
                let schema = rf.schema.as_ref().map(|s| {
                    let mut m = s.to_map();
                    add_additional_properties_false(&mut m);
                    m
                })?;
                let strict = rf.strict.unwrap_or(true);
                Some(OpenAIResponseFormat {
                    format_type: "json_schema".to_string(),
                    json_schema: Some(OpenAIJsonSchema {
                        name: rf.name.clone().unwrap_or_default(),
                        description: rf.description.clone(),
                        schema,
                        strict,
                    }),
                })
            }
            _ => Some(OpenAIResponseFormat {
                format_type: "text".to_string(),
                json_schema: None,
            }),
        }
    }

    /// Convert unified tools to OpenAI format (non-strict).
    pub fn tools_to_openai(&self, tools: &[Tool]) -> Vec<OpenAITool> {
        tools
            .iter()
            .map(|tool| {
                let mut params = tool.parameters.to_map();
                add_additional_properties_false(&mut params);
                OpenAITool {
                    tool_type: "function".to_string(),
                    function: OpenAIFunctionTool {
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        parameters: params,
                        strict: Some(false),
                    },
                }
            })
            .collect()
    }

    /// Convert unified tools to OpenAI format with strict mode.
    pub fn tools_to_openai_strict(&self, tools: &[Tool]) -> Vec<OpenAITool> {
        tools
            .iter()
            .map(|tool| {
                let mut params = tool.parameters.to_map();
                add_additional_properties_false(&mut params);
                OpenAITool {
                    tool_type: "function".to_string(),
                    function: OpenAIFunctionTool {
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        parameters: params,
                        strict: Some(true),
                    },
                }
            })
            .collect()
    }
}

// ---- Anthropic Format ----

#[derive(Debug, Serialize, Deserialize)]
pub struct AnthropicOutputConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<AnthropicFormat>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AnthropicFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Map<String, serde_json::Value>,
}

impl Translator {
    /// Convert unified schema to Anthropic format.
    pub fn to_anthropic(&self, rf: &ResponseFormat) -> Option<AnthropicOutputConfig> {
        if rf.format_type == "text" {
            return None;
        }
        if rf.format_type == "json" {
            return None;
        }
        if rf.format_type == "json_schema" {
            if let Some(schema) = &rf.schema {
                let mut m = schema.to_map();
                add_additional_properties_false(&mut m);
                return Some(AnthropicOutputConfig {
                    format: Some(AnthropicFormat {
                        format_type: "json_schema".to_string(),
                        schema: Some(m),
                    }),
                });
            }
        }
        None
    }

    /// Convert unified tools to Anthropic format.
    pub fn tools_to_anthropic(&self, tools: &[Tool]) -> Vec<AnthropicTool> {
        tools
            .iter()
            .map(|tool| AnthropicTool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                input_schema: tool.parameters.to_map(),
            })
            .collect()
    }
}

// ---- Google/Gemini Format ----

#[derive(Debug, Serialize, Deserialize)]
pub struct GoogleGenerationConfig {
    #[serde(rename = "responseMimeType", skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    #[serde(rename = "responseSchema", skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<GoogleSchema>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(rename = "topP", skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(rename = "topK", skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(rename = "maxOutputTokens", skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i32>,
    #[serde(rename = "stopSequences", skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleSchema {
    #[serde(rename = "type")]
    pub schema_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, GoogleSchema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<GoogleSchema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nullable: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GoogleTool {
    #[serde(rename = "functionDeclarations", skip_serializing_if = "Vec::is_empty")]
    pub function_declarations: Vec<GoogleFunctionDeclaration>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GoogleFunctionDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<GoogleSchema>,
}

impl Translator {
    /// Convert unified schema to Google format.
    pub fn to_google(&self, rf: &ResponseFormat) -> Option<GoogleGenerationConfig> {
        if rf.format_type == "text" {
            return None;
        }
        if rf.format_type == "json" {
            return Some(GoogleGenerationConfig {
                response_mime_type: Some("application/json".to_string()),
                response_schema: None,
                temperature: None,
                top_p: None,
                top_k: None,
                max_output_tokens: None,
                stop_sequences: None,
            });
        }
        if rf.format_type == "json_schema" {
            if let Some(schema) = &rf.schema {
                let google_schema = self.convert_to_google_schema(schema);
                return Some(GoogleGenerationConfig {
                    response_mime_type: Some("application/json".to_string()),
                    response_schema: Some(google_schema),
                    temperature: None,
                    top_p: None,
                    top_k: None,
                    max_output_tokens: None,
                    stop_sequences: None,
                });
            }
        }
        None
    }

    /// Convert unified tools to Google format.
    pub fn tools_to_google(&self, tools: &[Tool]) -> Option<GoogleTool> {
        if tools.is_empty() {
            return None;
        }
        let declarations = tools
            .iter()
            .map(|tool| GoogleFunctionDeclaration {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: Some(self.convert_to_google_schema(&tool.parameters)),
            })
            .collect();
        Some(GoogleTool {
            function_declarations: declarations,
        })
    }

    /// Convert JSON Schema to Google's schema format.
    pub fn convert_to_google_schema(&self, s: &JsonSchema) -> GoogleSchema {
        let mut gs = GoogleSchema {
            schema_type: map_type_to_google(s.schema_type.as_deref().unwrap_or("")),
            description: s.description.clone(),
            required: s.required.clone(),
            enum_values: None,
            properties: None,
            items: None,
            nullable: None,
        };

        // Convert enum (Google only supports string enums)
        if let Some(ref enum_vals) = s.enum_values {
            gs.enum_values = Some(
                enum_vals
                    .iter()
                    .map(|v| match v {
                        serde_json::Value::String(s) => s.clone(),
                        _ => v.to_string(),
                    })
                    .collect(),
            );
        }

        // Convert properties
        if let Some(ref props) = s.properties {
            gs.properties = Some(
                props
                    .iter()
                    .map(|(k, v)| (k.clone(), self.convert_to_google_schema(v)))
                    .collect(),
            );
        }

        // Convert items
        if let Some(ref items) = s.items {
            gs.items = Some(Box::new(self.convert_to_google_schema(items)));
        }

        gs
    }
}

fn map_type_to_google(json_type: &str) -> String {
    match json_type {
        "integer" => "INTEGER".to_string(),
        "number" => "NUMBER".to_string(),
        "string" => "STRING".to_string(),
        "boolean" => "BOOLEAN".to_string(),
        "array" => "ARRAY".to_string(),
        "object" => "OBJECT".to_string(),
        _ => "STRING".to_string(),
    }
}

/// Recursively add `additionalProperties: false` to all object schemas.
pub fn add_additional_properties_false(schema: &mut serde_json::Map<String, serde_json::Value>) {
    if let Some(serde_json::Value::String(t)) = schema.get("type") {
        if t == "object" {
            schema.insert(
                "additionalProperties".to_string(),
                serde_json::Value::Bool(false),
            );
        }
    }

    // Recurse into properties
    if let Some(serde_json::Value::Object(props)) = schema.get_mut("properties") {
        let keys: Vec<String> = props.keys().cloned().collect();
        for key in keys {
            if let Some(serde_json::Value::Object(prop)) = props.get_mut(&key) {
                add_additional_properties_false(prop);
            }
        }
    }

    // Recurse into items
    if let Some(serde_json::Value::Object(items)) = schema.get_mut("items") {
        add_additional_properties_false(items);
    }

    // Recurse into anyOf, oneOf, allOf
    for key in &["anyOf", "oneOf", "allOf"] {
        if let Some(serde_json::Value::Array(arr)) = schema.get_mut(*key) {
            for item in arr.iter_mut() {
                if let serde_json::Value::Object(obj) = item {
                    add_additional_properties_false(obj);
                }
            }
        }
    }

    // Recurse into $defs
    if let Some(serde_json::Value::Object(defs)) = schema.get_mut("$defs") {
        let keys: Vec<String> = defs.keys().cloned().collect();
        for key in keys {
            if let Some(serde_json::Value::Object(def)) = defs.get_mut(&key) {
                add_additional_properties_false(def);
            }
        }
    }
}
