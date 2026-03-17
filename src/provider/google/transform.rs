//! Google Gemini request/response transformation.

use serde_json::Value;
use std::collections::HashMap;

use super::types::*;
use crate::schema::{GoogleSchema, Translator};
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentType, Message, Provider, Role,
    StopReason, StreamEvent, StreamEventType, Tool, ToolCall, ToolChoice, ToolChoiceType, Usage,
};

/// Transformer handles conversion between unified and Google formats.
pub struct Transformer {
    schema_translator: Translator,
}

impl Transformer {
    pub fn new() -> Self {
        Self {
            schema_translator: Translator::new(),
        }
    }

    /// Convert a unified request to Google format.
    pub fn transform_request(&self, req: &CompletionRequest) -> GenerateContentRequest {
        let (contents, system_instruction) = self.transform_messages(&req.messages);

        let mut gen_config = GenerationConfig {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            max_output_tokens: req.max_tokens,
            stop_sequences: req.stop_sequences.clone(),
            ..Default::default()
        };

        if let Some(rf) = &req.response_format {
            self.apply_response_format(&mut gen_config, rf);
        }

        let tools = if !req.tools.is_empty() {
            self.transform_tools(&req.tools)
        } else {
            vec![]
        };

        let tool_config = req
            .tool_choice
            .as_ref()
            .map(|tc| self.transform_tool_choice(tc));

        GenerateContentRequest {
            contents,
            system_instruction,
            generation_config: Some(gen_config),
            safety_settings: vec![],
            tools,
            tool_config,
            labels: None,
        }
    }

    /// Convert unified messages to Google format, separating system instruction.
    fn transform_messages(&self, messages: &[Message]) -> (Vec<Content>, Option<Content>) {
        let mut contents = Vec::new();
        let mut system_instruction: Option<Content> = None;

        for msg in messages {
            if msg.role == Role::System {
                let parts: Vec<Part> = msg
                    .content
                    .iter()
                    .filter(|b| b.content_type == Some(ContentType::Text))
                    .filter_map(|b| b.text.as_ref())
                    .map(|t| Part {
                        text: Some(t.clone()),
                        ..Default::default()
                    })
                    .collect();
                if !parts.is_empty() {
                    system_instruction = Some(Content { role: None, parts });
                }
                continue;
            }

            let content = Content {
                role: Some(self.map_role(&msg.role).to_string()),
                parts: self.transform_parts(&msg.content),
            };
            contents.push(content);
        }

        (contents, system_instruction)
    }

    fn map_role(&self, role: &Role) -> &'static str {
        match role {
            Role::User | Role::Tool => "user",
            Role::Assistant => "model",
            Role::System => "user",
        }
    }

    fn transform_parts(&self, blocks: &[ContentBlock]) -> Vec<Part> {
        let mut parts = Vec::new();

        for block in blocks {
            match &block.content_type {
                Some(ContentType::Text) => {
                    parts.push(Part {
                        text: block.text.clone(),
                        ..Default::default()
                    });
                }
                Some(ContentType::Image) => {
                    if let Some(b64) = &block.image_base64 {
                        if !b64.is_empty() {
                            parts.push(Part {
                                inline_data: Some(InlineData {
                                    mime_type: block.media_type.clone().unwrap_or_default(),
                                    data: b64.clone(),
                                }),
                                ..Default::default()
                            });
                            continue;
                        }
                    }
                    if let Some(url) = &block.image_url {
                        if !url.is_empty() {
                            parts.push(Part {
                                file_data: Some(FileData {
                                    mime_type: block.media_type.clone().unwrap_or_default(),
                                    file_uri: url.clone(),
                                }),
                                ..Default::default()
                            });
                        }
                    }
                }
                Some(ContentType::ToolUse) => {
                    let args = block
                        .tool_input
                        .clone()
                        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
                    parts.push(Part {
                        function_call: Some(FunctionCall {
                            name: block.tool_name.clone().unwrap_or_default(),
                            args,
                        }),
                        ..Default::default()
                    });
                }
                Some(ContentType::ToolResult) => {
                    let text = block.text.as_deref().unwrap_or("");
                    let response: Value = serde_json::from_str(text).unwrap_or_else(|_| {
                        let mut m = serde_json::Map::new();
                        m.insert("result".to_string(), Value::String(text.to_string()));
                        Value::Object(m)
                    });
                    parts.push(Part {
                        function_response: Some(FunctionResponse {
                            name: block.tool_name.clone().unwrap_or_default(),
                            response,
                        }),
                        ..Default::default()
                    });
                }
                None => {}
            }
        }

        parts
    }

    fn apply_response_format(
        &self,
        config: &mut GenerationConfig,
        rf: &crate::types::ResponseFormat,
    ) {
        if let Some(google_config) = self.schema_translator.to_google(rf) {
            config.response_mime_type = google_config.response_mime_type;
            if let Some(gs) = google_config.response_schema {
                config.response_schema = Some(self.convert_google_schema(&gs));
            }
        }
    }

    fn convert_google_schema(&self, s: &GoogleSchema) -> Schema {
        let properties = s.properties.as_ref().map(|props| {
            props
                .iter()
                .map(|(k, v)| (k.clone(), Box::new(self.convert_google_schema(v))))
                .collect::<HashMap<String, Box<Schema>>>()
        });

        Schema {
            schema_type: s.schema_type.clone(),
            description: s.description.clone(),
            enum_values: s.enum_values.clone(),
            properties,
            required: s.required.clone(),
            items: s
                .items
                .as_ref()
                .map(|i| Box::new(self.convert_google_schema(i))),
            nullable: s.nullable,
        }
    }

    fn transform_tools(&self, tools: &[Tool]) -> Vec<super::types::Tool> {
        match self.schema_translator.tools_to_google(tools) {
            None => vec![],
            Some(gt) => {
                let declarations: Vec<FunctionDeclaration> = gt
                    .function_declarations
                    .into_iter()
                    .map(|decl| FunctionDeclaration {
                        name: decl.name,
                        description: decl.description,
                        parameters: decl.parameters.map(|p| self.convert_google_schema(&p)),
                    })
                    .collect();
                vec![super::types::Tool {
                    function_declarations: declarations,
                }]
            }
        }
    }

    fn transform_tool_choice(&self, tc: &ToolChoice) -> ToolConfig {
        let (mode, allowed_names) = match tc.choice_type {
            ToolChoiceType::Auto => ("AUTO".to_string(), vec![]),
            ToolChoiceType::Required => ("ANY".to_string(), vec![]),
            ToolChoiceType::None => ("NONE".to_string(), vec![]),
            ToolChoiceType::Tool => (
                "ANY".to_string(),
                tc.name
                    .as_ref()
                    .map(|n| vec![n.clone()])
                    .unwrap_or_default(),
            ),
        };

        ToolConfig {
            function_calling_config: Some(FunctionCallingConfig {
                mode,
                allowed_function_names: allowed_names,
            }),
        }
    }

    /// Convert Google response to unified format.
    pub fn transform_response(&self, resp: &GenerateContentResponse) -> Option<CompletionResponse> {
        if resp.candidates.is_empty() {
            return None;
        }

        let candidate = &resp.candidates[0];
        let content = self.transform_response_content(candidate.content.as_ref());
        let tool_calls = self.extract_tool_calls(candidate.content.as_ref());
        let stop_reason = self.transform_stop_reason(&candidate.finish_reason);

        let mut result = CompletionResponse {
            id: String::new(),
            provider: Provider::Google,
            model: String::new(),
            content,
            stop_reason,
            tool_calls,
            usage: Usage::default(),
            created_at: None,
            metadata: None,
        };

        if let Some(meta) = &resp.usage_metadata {
            result.usage = Usage {
                input_tokens: meta.prompt_token_count as i64,
                output_tokens: meta.candidates_token_count as i64,
                total_tokens: meta.total_token_count as i64,
                cached_tokens: None,
                reasoning_tokens: None,
            };
        }

        Some(result)
    }

    fn transform_response_content(&self, content: Option<&Content>) -> Vec<ContentBlock> {
        let content = match content {
            None => return vec![],
            Some(c) => c,
        };

        let mut blocks = Vec::new();
        for part in &content.parts {
            if let Some(text) = &part.text {
                if !text.is_empty() {
                    blocks.push(ContentBlock {
                        content_type: Some(ContentType::Text),
                        text: Some(text.clone()),
                        ..Default::default()
                    });
                }
            }
            if let Some(fc) = &part.function_call {
                // Mirror the synthesised ID used in extract_tool_calls so that
                // ContentBlock.tool_use_id and ToolCall.id are consistent.
                blocks.push(ContentBlock {
                    content_type: Some(ContentType::ToolUse),
                    tool_use_id: Some(format!("call_{}", fc.name)),
                    tool_name: Some(fc.name.clone()),
                    tool_input: Some(fc.args.clone()),
                    ..Default::default()
                });
            }
        }
        blocks
    }

    fn extract_tool_calls(&self, content: Option<&Content>) -> Vec<ToolCall> {
        let content = match content {
            None => return vec![],
            Some(c) => c,
        };

        content
            .parts
            .iter()
            .filter_map(|part| part.function_call.as_ref())
            .map(|fc| ToolCall {
                // Google's API does not return tool call IDs. We synthesise one
                // from the function name so that callers can round-trip tool
                // results back via `Message::new_tool_result`. A stable,
                // name-based ID is preferable to an empty string, which would
                // cause silent failures when correlating results.
                id: format!("call_{}", fc.name),
                name: fc.name.clone(),
                input: fc.args.clone(),
            })
            .collect()
    }

    /// Convert Google finish reason to unified StopReason.
    pub fn transform_stop_reason(&self, reason: &str) -> StopReason {
        match reason {
            "STOP" | "OTHER" => StopReason::End,
            "MAX_TOKENS" => StopReason::MaxTokens,
            "SAFETY" | "RECITATION" => StopReason::ContentFilter,
            _ => StopReason::End,
        }
    }

    /// Process a streaming chunk into a StreamEvent.
    pub fn process_chunk(
        &self,
        chunk: &StreamChunk,
        acc_content: &mut Vec<ContentBlock>,
        acc_tool_calls: &mut Vec<ToolCall>,
        usage: &mut Option<Usage>,
        stop_reason: &mut StopReason,
    ) -> Option<StreamEvent> {
        if chunk.candidates.is_empty() {
            return None;
        }

        let candidate = &chunk.candidates[0];

        if !candidate.finish_reason.is_empty() {
            *stop_reason = self.transform_stop_reason(&candidate.finish_reason);
        }

        if let Some(meta) = &chunk.usage_metadata {
            *usage = Some(Usage {
                input_tokens: meta.prompt_token_count as i64,
                output_tokens: meta.candidates_token_count as i64,
                total_tokens: meta.total_token_count as i64,
                cached_tokens: None,
                reasoning_tokens: None,
            });
        }

        let cand_content = candidate.content.as_ref()?;

        for part in &cand_content.parts {
            if let Some(text) = &part.text {
                if !text.is_empty() {
                    // Accumulate text
                    if acc_content
                        .last()
                        .map(|b| b.content_type == Some(ContentType::Text))
                        .unwrap_or(false)
                    {
                        if let Some(last) = acc_content.last_mut() {
                            if let Some(t) = last.text.as_mut() {
                                t.push_str(text);
                            }
                        }
                    } else {
                        acc_content.push(ContentBlock {
                            content_type: Some(ContentType::Text),
                            text: Some(text.clone()),
                            ..Default::default()
                        });
                    }

                    return Some(StreamEvent {
                        event_type: StreamEventType::ContentDelta,
                        delta: Some(ContentBlock {
                            content_type: Some(ContentType::Text),
                            text: Some(text.clone()),
                            ..Default::default()
                        }),
                        ..Default::default()
                    });
                }
            }

            if let Some(fc) = &part.function_call {
                let tc = ToolCall {
                    id: String::new(),
                    name: fc.name.clone(),
                    input: fc.args.clone(),
                };
                acc_tool_calls.push(tc.clone());
                acc_content.push(ContentBlock {
                    content_type: Some(ContentType::ToolUse),
                    tool_name: Some(fc.name.clone()),
                    tool_input: Some(fc.args.clone()),
                    ..Default::default()
                });

                return Some(StreamEvent {
                    event_type: StreamEventType::ToolCallStart,
                    tool_call: Some(tc),
                    ..Default::default()
                });
            }
        }

        None
    }
}

impl Default for Transformer {
    fn default() -> Self {
        Self::new()
    }
}
