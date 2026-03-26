//! Anthropic request/response transformation.

use super::types::*;
use crate::schema::Translator;
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock as UnifiedContentBlock, ContentType,
    Message as UnifiedMessage, Provider, Role, StopReason, ToolChoice as UnifiedToolChoice,
    ToolChoiceType, Usage as UnifiedUsage,
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

    pub fn transform_request(&self, req: &CompletionRequest) -> MessagesRequest {
        let (messages, system) = self.transform_messages(&req.messages);

        let metadata = req
            .metadata
            .as_ref()
            .and_then(|m| m.get("user_id"))
            .map(|uid| super::types::MessagesMetadata {
                user_id: Some(uid.clone()),
            });

        let mut anth_req = MessagesRequest {
            model: req.model.clone(),
            messages,
            max_tokens: req.max_tokens.unwrap_or(8192),
            system: system.map(serde_json::Value::String),
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            stop_sequences: if req.stop_sequences.is_empty() {
                None
            } else {
                Some(req.stop_sequences.clone())
            },
            stream: None,
            tools: None,
            tool_choice: None,
            output_config: None,
            metadata,
        };

        if let Some(ref rf) = req.response_format {
            if let Some(anth_cfg) = self.schema_translator.to_anthropic(rf) {
                anth_req.output_config = Some(OutputConfig {
                    format: anth_cfg.format.map(|f| OutputFormat {
                        format_type: f.format_type,
                        schema: f.schema,
                    }),
                });
            }
        }

        if !req.tools.is_empty() {
            let anth_tools = self.schema_translator.tools_to_anthropic(&req.tools);
            anth_req.tools = Some(
                anth_tools
                    .into_iter()
                    .map(|t| Tool {
                        name: t.name,
                        description: t.description,
                        input_schema: t.input_schema,
                    })
                    .collect(),
            );
        }

        if let Some(ref tc) = req.tool_choice {
            anth_req.tool_choice = Some(self.transform_tool_choice(tc));
        }

        anth_req
    }

    fn transform_messages(
        &self,
        messages: &[UnifiedMessage],
    ) -> (Vec<super::types::Message>, Option<String>) {
        let mut result = Vec::new();
        let mut system = None;

        for msg in messages {
            if msg.role == Role::System {
                let text = msg
                    .content
                    .iter()
                    .filter(|b| b.content_type == Some(ContentType::Text))
                    .filter_map(|b| b.text.as_deref())
                    .collect::<Vec<_>>()
                    .join("\n");
                system = Some(text);
                continue;
            }

            let role = self.map_role(&msg.role);

            // Check if we can use simple string content
            if msg.content.len() == 1 && msg.content[0].content_type == Some(ContentType::Text) {
                result.push(super::types::Message {
                    role,
                    content: serde_json::Value::String(
                        msg.content[0].text.clone().unwrap_or_default(),
                    ),
                });
            } else {
                let blocks = self.transform_content_blocks(&msg.content);
                result.push(super::types::Message {
                    role,
                    content: serde_json::json!(blocks),
                });
            }
        }

        (result, system)
    }

    fn map_role(&self, role: &crate::types::Role) -> String {
        match role {
            Role::User | Role::Tool => "user".to_string(),
            Role::Assistant => "assistant".to_string(),
            // Role::System is filtered out before this function is called.
            // Using an explicit arm (rather than `_`) ensures the compiler will
            // warn us if a new Role variant is added and we forget to handle it.
            Role::System => unreachable!(
                "system role is extracted as a top-level field before map_role is called"
            ),
        }
    }

    fn transform_content_blocks(&self, blocks: &[UnifiedContentBlock]) -> Vec<serde_json::Value> {
        let mut result = Vec::new();

        for block in blocks {
            match block.content_type {
                Some(ContentType::Text) => {
                    result.push(serde_json::json!({
                        "type": "text",
                        "text": block.text.as_deref().unwrap_or("")
                    }));
                }
                Some(ContentType::Image) => {
                    if let Some(ref b64) = block.image_base64 {
                        result.push(serde_json::json!({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block.media_type.as_deref().unwrap_or("image/jpeg"),
                                "data": b64
                            }
                        }));
                    } else if let Some(ref url) = block.image_url {
                        result.push(serde_json::json!({
                            "type": "image",
                            "source": { "type": "url", "url": url }
                        }));
                    }
                }
                Some(ContentType::ToolUse) => {
                    result.push(serde_json::json!({
                        "type": "tool_use",
                        "id": block.tool_use_id.as_deref().unwrap_or(""),
                        "name": block.tool_name.as_deref().unwrap_or(""),
                        "input": block.tool_input.as_ref().unwrap_or(&serde_json::Value::Null)
                    }));
                }
                Some(ContentType::ToolResult) => {
                    result.push(serde_json::json!({
                        "type": "tool_result",
                        "tool_use_id": block.tool_result_id.as_deref().unwrap_or(""),
                        "content": block.text.as_deref().unwrap_or(""),
                        "is_error": block.is_error.unwrap_or(false)
                    }));
                }
                _ => {}
            }
        }

        result
    }

    fn transform_tool_choice(&self, tc: &UnifiedToolChoice) -> super::types::ToolChoice {
        let choice_type = match tc.choice_type {
            ToolChoiceType::Auto => "auto",
            ToolChoiceType::Required => "any",
            ToolChoiceType::None => "none",
            ToolChoiceType::Tool => "tool",
        };
        super::types::ToolChoice {
            choice_type: choice_type.to_string(),
            name: tc.name.clone(),
            disable_parallel_tool_use: if tc.disable_parallel_tool_use {
                Some(true)
            } else {
                None
            },
        }
    }

    pub fn transform_response(&self, resp: &MessagesResponse) -> CompletionResponse {
        let content = self.transform_response_content(&resp.content);
        let tool_calls = self.extract_tool_calls(&resp.content);

        CompletionResponse {
            id: resp.id.clone(),
            provider: Provider::Anthropic,
            model: resp.model.clone(),
            content,
            stop_reason: self.transform_stop_reason(&resp.stop_reason),
            usage: UnifiedUsage {
                input_tokens: resp.usage.input_tokens,
                output_tokens: resp.usage.output_tokens,
                total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
                cached_tokens: Some(resp.usage.cache_read_input_tokens),
                reasoning_tokens: None,
            },
            tool_calls,
            created_at: Some(chrono::Utc::now()),
            metadata: None,
        }
    }

    fn transform_response_content(
        &self,
        blocks: &[ContentBlock],
    ) -> Vec<crate::types::ContentBlock> {
        let mut result = Vec::new();
        for block in blocks {
            match block.block_type.as_str() {
                "text" => {
                    if let Some(ref text) = block.text {
                        result.push(crate::types::ContentBlock::text(text.clone()));
                    }
                }
                "tool_use" => {
                    result.push(crate::types::ContentBlock::tool_use(
                        block.id.clone().unwrap_or_default(),
                        block.name.clone().unwrap_or_default(),
                        block.input.clone().unwrap_or(serde_json::Value::Null),
                    ));
                }
                _ => {}
            }
        }
        result
    }

    fn extract_tool_calls(&self, blocks: &[ContentBlock]) -> Vec<crate::types::ToolCall> {
        blocks
            .iter()
            .filter_map(|block| {
                if block.block_type == "tool_use" {
                    Some(crate::types::ToolCall {
                        id: block.id.clone().unwrap_or_default(),
                        name: block.name.clone().unwrap_or_default(),
                        input: block.input.clone().unwrap_or(serde_json::Value::Null),
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn transform_stop_reason(&self, reason: &str) -> StopReason {
        match reason {
            "end_turn" => StopReason::End,
            "max_tokens" => StopReason::MaxTokens,
            "tool_use" => StopReason::ToolUse,
            "stop_sequence" => StopReason::StopSequence,
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
    fn forwards_only_user_id_in_metadata() {
        let t = Transformer::new();
        let mut m = HashMap::new();
        m.insert("user_id".to_string(), "u1".to_string());
        m.insert("ignored".to_string(), "x".to_string());
        let req = CompletionRequest::new(
            Provider::Anthropic,
            "claude-3-haiku-20240307",
            vec![Message::new_text(Role::User, "x")],
        )
        .with_metadata(m);
        let anth = t.transform_request(&req);
        let v = serde_json::to_value(&anth).unwrap();
        let meta = v.get("metadata").expect("metadata");
        assert_eq!(meta["user_id"], "u1");
        assert!(meta.get("ignored").is_none());
    }
}
