//! OpenAI request/response transformation.

use super::types::*;
use crate::schema::Translator;
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentType, Message, Provider, Role,
    StopReason, ToolChoice, ToolChoiceType, Usage as UnifiedUsage,
};

pub struct Transformer {
    schema_translator: Translator,
}

impl Transformer {
    pub fn new() -> Self {
        Self {
            schema_translator: Translator::new(),
        }
    }

    /// Transform a unified request to OpenAI format.
    pub fn transform_request(&self, req: &CompletionRequest) -> ChatCompletionRequest {
        let mut oai_req = ChatCompletionRequest {
            model: req.model.clone(),
            messages: self.transform_messages(&req.messages),
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            stream: None,
            stream_options: None,
            stop: if req.stop_sequences.is_empty() {
                None
            } else {
                Some(req.stop_sequences.clone())
            },
            response_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            metadata: req
                .metadata
                .as_ref()
                .filter(|m| !m.is_empty())
                .cloned(),
        };

        // Transform response format
        if let Some(ref rf) = req.response_format {
            oai_req.response_format =
                self.schema_translator
                    .to_openai(rf)
                    .map(|oai_rf| ResponseFormat {
                        format_type: oai_rf.format_type,
                        json_schema: oai_rf.json_schema.map(|js| JsonSchema {
                            name: js.name,
                            description: js.description,
                            schema: js.schema,
                            strict: js.strict,
                        }),
                    });
        }

        // Transform tools
        if !req.tools.is_empty() {
            let oai_tools = self.schema_translator.tools_to_openai(&req.tools);
            oai_req.tools = Some(
                oai_tools
                    .into_iter()
                    .map(|t| Tool {
                        tool_type: t.tool_type,
                        function: Function {
                            name: t.function.name,
                            description: t.function.description,
                            parameters: t.function.parameters,
                            strict: t.function.strict,
                        },
                    })
                    .collect(),
            );
        }

        // Transform tool choice
        if let Some(ref tc) = req.tool_choice {
            oai_req.tool_choice = Some(self.transform_tool_choice(tc));
        }

        oai_req
    }

    fn transform_messages(&self, messages: &[Message]) -> Vec<ChatMessage> {
        let mut result = Vec::new();

        for msg in messages {
            if msg.role == Role::Tool {
                for block in &msg.content {
                    if block.content_type == Some(ContentType::ToolResult) {
                        result.push(ChatMessage {
                            role: "tool".to_string(),
                            content: Some(serde_json::Value::String(
                                block.text.clone().unwrap_or_default(),
                            )),
                            name: None,
                            tool_calls: None,
                            tool_call_id: block.tool_result_id.clone(),
                        });
                    }
                }
                continue;
            }

            let has_images = msg
                .content
                .iter()
                .any(|b| b.content_type == Some(ContentType::Image));
            let has_tool_calls = msg
                .content
                .iter()
                .any(|b| b.content_type == Some(ContentType::ToolUse));
            let has_multiple = msg.content.len() > 1;

            if has_tool_calls && msg.role == Role::Assistant {
                let mut text_content = String::new();
                let mut tool_calls = Vec::new();

                for block in &msg.content {
                    match block.content_type {
                        Some(ContentType::Text) => {
                            if let Some(ref t) = block.text {
                                text_content.push_str(t);
                            }
                        }
                        Some(ContentType::ToolUse) => {
                            let args = block
                                .tool_input
                                .as_ref()
                                .map(|v| serde_json::to_string(v).unwrap_or_default())
                                .unwrap_or_default();
                            tool_calls.push(ToolCall {
                                id: block.tool_use_id.clone().unwrap_or_default(),
                                call_type: "function".to_string(),
                                function: FunctionCall {
                                    name: block.tool_name.clone().unwrap_or_default(),
                                    arguments: args,
                                },
                                index: None,
                            });
                        }
                        _ => {}
                    }
                }

                result.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: if text_content.is_empty() {
                        None
                    } else {
                        Some(serde_json::Value::String(text_content))
                    },
                    name: None,
                    tool_calls: Some(tool_calls),
                    tool_call_id: None,
                });
            } else if has_images || has_multiple {
                let parts: Vec<serde_json::Value> = msg
                    .content
                    .iter()
                    .filter_map(|block| match block.content_type {
                        Some(ContentType::Text) => Some(serde_json::json!({
                            "type": "text",
                            "text": block.text.as_deref().unwrap_or("")
                        })),
                        Some(ContentType::Image) => {
                            let url = if let Some(ref u) = block.image_url {
                                u.clone()
                            } else if let Some(ref b64) = block.image_base64 {
                                let mt = block.media_type.as_deref().unwrap_or("image/jpeg");
                                format!("data:{};base64,{}", mt, b64)
                            } else {
                                return None;
                            };
                            Some(serde_json::json!({
                                "type": "image_url",
                                "image_url": { "url": url }
                            }))
                        }
                        _ => None,
                    })
                    .collect();

                result.push(ChatMessage {
                    role: msg.role.to_string(),
                    content: Some(serde_json::Value::Array(parts)),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
            } else {
                let text = msg
                    .content
                    .iter()
                    .filter(|b| b.content_type == Some(ContentType::Text))
                    .filter_map(|b| b.text.as_deref())
                    .collect::<Vec<_>>()
                    .join("");
                result.push(ChatMessage {
                    role: msg.role.to_string(),
                    content: Some(serde_json::Value::String(text)),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
        }

        result
    }

    fn transform_tool_choice(&self, tc: &ToolChoice) -> serde_json::Value {
        match tc.choice_type {
            ToolChoiceType::Auto => serde_json::json!("auto"),
            ToolChoiceType::Required => serde_json::json!("required"),
            ToolChoiceType::None => serde_json::json!("none"),
            ToolChoiceType::Tool => serde_json::json!({
                "type": "function",
                "function": { "name": tc.name.as_deref().unwrap_or("") }
            }),
        }
    }

    /// Transform OpenAI response to unified format.
    pub fn transform_response(&self, resp: &ChatCompletionResponse) -> Option<CompletionResponse> {
        if resp.choices.is_empty() {
            return None;
        }
        let choice = &resp.choices[0];
        let content = self.transform_content(&choice.message);
        let tool_calls = self.extract_tool_calls(&choice.message);
        let stop_reason = self.transform_stop_reason(choice.finish_reason.as_deref().unwrap_or(""));

        let mut usage = UnifiedUsage::default();
        if let Some(ref u) = resp.usage {
            usage.input_tokens = u.prompt_tokens;
            usage.output_tokens = u.completion_tokens;
            usage.total_tokens = u.total_tokens;
            if let Some(ref pd) = u.prompt_tokens_details {
                usage.cached_tokens = Some(pd.cached_tokens);
            }
            if let Some(ref cd) = u.completion_tokens_details {
                usage.reasoning_tokens = Some(cd.reasoning_tokens);
            }
        }

        Some(CompletionResponse {
            id: resp.id.clone(),
            provider: Provider::OpenAI,
            model: resp.model.clone(),
            content,
            stop_reason,
            usage,
            tool_calls,
            created_at: chrono::DateTime::from_timestamp(resp.created, 0)
                .or_else(|| Some(chrono::Utc::now())),
            metadata: None,
        })
    }

    fn transform_content(&self, msg: &ChatMessage) -> Vec<ContentBlock> {
        let mut blocks = Vec::new();

        if let Some(ref content) = msg.content {
            match content {
                serde_json::Value::String(s) if !s.is_empty() => {
                    blocks.push(ContentBlock::text(s.clone()));
                }
                serde_json::Value::Array(parts) => {
                    for part in parts {
                        if let Some(t) = part.get("type").and_then(|v| v.as_str()) {
                            if t == "text" {
                                if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                                    blocks.push(ContentBlock::text(text));
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Handle tool calls
        if let Some(ref tcs) = msg.tool_calls {
            for tc in tcs {
                let input: serde_json::Value =
                    serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::Value::Null);
                blocks.push(ContentBlock::tool_use(&tc.id, &tc.function.name, input));
            }
        }

        blocks
    }

    fn extract_tool_calls(&self, msg: &ChatMessage) -> Vec<crate::types::ToolCall> {
        msg.tool_calls
            .as_ref()
            .map(|tcs| {
                tcs.iter()
                    .map(|tc| {
                        let input: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::Value::Null);
                        crate::types::ToolCall {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            input,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn transform_stop_reason(&self, reason: &str) -> StopReason {
        match reason {
            "stop" => StopReason::End,
            "length" => StopReason::MaxTokens,
            "tool_calls" => StopReason::ToolUse,
            "content_filter" => StopReason::ContentFilter,
            _ => StopReason::End,
        }
    }
}

impl Default for Transformer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::types::{CompletionRequest, Message, Provider, Role};

    #[test]
    fn transform_includes_metadata_in_json() {
        let t = Transformer::new();
        let mut m = HashMap::new();
        m.insert("trace_id".to_string(), "abc".to_string());
        let req = CompletionRequest::new(
            Provider::OpenAI,
            "gpt-4o-mini",
            vec![Message::new_text(Role::User, "x")],
        )
        .with_metadata(m);
        let oai = t.transform_request(&req);
        let v = serde_json::to_value(&oai).unwrap();
        assert!(v.get("metadata").is_some());
        assert_eq!(v["metadata"]["trace_id"], "abc");
    }
}
