//! Anthropic API client implementation.

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use std::sync::Arc;

use super::transform::Transformer;
use super::types::*;
use crate::errors::*;
use crate::provider::{ProviderClient, StreamResponse};
use crate::types::*;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
pub const DEFAULT_VERSION: &str = "2023-06-01";
pub const BETA_HEADER: &str =
    "prompt-caching-2024-07-31,output-128k-2025-02-19,message-batches-2024-09-24";

pub struct Client {
    pub config: Arc<crate::provider::ProviderConfig>,
    pub http: reqwest::Client,
    pub base_url: String,
    pub transformer: Transformer,
}

impl Client {
    pub fn new(opts: Vec<crate::provider::ProviderOption>) -> Result<Self, RouterError> {
        let mut cfg = crate::provider::ProviderConfig::default_with_timeout();
        crate::provider::apply_options(&mut cfg, opts);

        let base_url = cfg
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let timeout = cfg.timeout.unwrap_or(120);

        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout))
            .build()
            .map_err(|e| err_invalid_request(format!("failed to build HTTP client: {}", e)))?;

        Ok(Self {
            config: Arc::new(cfg),
            http,
            base_url,
            transformer: Transformer::new(),
        })
    }

    fn set_headers(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        // Do not set Content-Type here — reqwest's .json() already sets it and
        // adding it again after .json() causes reqwest to discard the body.
        builder
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", DEFAULT_VERSION)
            .header("anthropic-beta", BETA_HEADER)
    }

    pub async fn handle_error_response(&self, resp: reqwest::Response) -> RouterError {
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();

        if let Ok(err_resp) = serde_json::from_str::<ErrorResponse>(&body) {
            if let Some(api_err) = err_resp.error {
                return self.map_api_error(&api_err, status);
            }
        }

        err_server_error(Provider::Anthropic, body).with_status_code(status)
    }

    fn map_api_error(&self, api_err: &APIError, status_code: u16) -> RouterError {
        match status_code {
            401 => err_invalid_api_key(Provider::Anthropic).with_status_code(status_code),
            429 => {
                err_rate_limit(Provider::Anthropic, &api_err.message).with_status_code(status_code)
            }
            404 => err_model_not_found(Provider::Anthropic, &api_err.message)
                .with_status_code(status_code),
            400 => {
                if api_err.message.contains("context") || api_err.message.contains("token") {
                    err_context_length(Provider::Anthropic, &api_err.message)
                        .with_status_code(status_code)
                } else {
                    err_invalid_request(&api_err.message)
                        .with_provider(Provider::Anthropic)
                        .with_status_code(status_code)
                }
            }
            _ => err_server_error(Provider::Anthropic, &api_err.message)
                .with_status_code(status_code),
        }
    }
}

#[async_trait]
impl ProviderClient for Client {
    fn name(&self) -> Provider {
        Provider::Anthropic
    }

    fn supports_feature(&self, feature: &Feature) -> bool {
        matches!(
            feature,
            Feature::Streaming
                | Feature::StructuredOutput
                | Feature::Tools
                | Feature::Vision
                | Feature::Batch
        )
    }

    fn models(&self) -> Vec<String> {
        vec![
            "claude-sonnet-4-20250514".to_string(),
            "claude-opus-4-20250514".to_string(),
            "claude-3-5-sonnet-20241022".to_string(),
            "claude-3-5-haiku-20241022".to_string(),
            "claude-3-opus-20240229".to_string(),
            "claude-3-sonnet-20240229".to_string(),
            "claude-3-haiku-20240307".to_string(),
        ]
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, RouterError> {
        let mut anth_req = self.transformer.transform_request(req);
        anth_req.stream = Some(false);

        let builder = self
            .http
            .post(format!("{}/v1/messages", self.base_url))
            .json(&anth_req);
        let builder = self.set_headers(builder);

        let resp = builder.send().await.map_err(|e| {
            err_provider_unavailable(Provider::Anthropic, format!("request failed: {}", e))
        })?;

        if !resp.status().is_success() {
            return Err(self.handle_error_response(resp).await);
        }

        let anth_resp: MessagesResponse = resp.json().await.map_err(|e| {
            err_server_error(Provider::Anthropic, format!("decode response: {}", e))
        })?;

        Ok(self.transformer.transform_response(&anth_resp))
    }

    async fn stream(&self, req: &CompletionRequest) -> Result<StreamResponse, RouterError> {
        let mut anth_req = self.transformer.transform_request(req);
        anth_req.stream = Some(true);

        let builder = self
            .http
            .post(format!("{}/v1/messages", self.base_url))
            .json(&anth_req);
        let builder = self.set_headers(builder);

        let resp = builder.send().await.map_err(|e| {
            err_provider_unavailable(Provider::Anthropic, format!("request failed: {}", e))
        })?;

        if !resp.status().is_success() {
            return Err(self.handle_error_response(resp).await);
        }

        let transformer = Arc::new(Transformer::new());
        let byte_stream = resp.bytes_stream();
        let event_stream = anthropic_sse_stream(byte_stream, transformer);

        Ok(Box::pin(event_stream))
    }
}

/// Parse Anthropic SSE stream into StreamEvents.
fn anthropic_sse_stream(
    byte_stream: impl futures::Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    transformer: Arc<Transformer>,
) -> impl futures::Stream<Item = Result<StreamEvent, RouterError>> + Send + 'static {
    use std::collections::HashMap;

    async_stream::stream! {
        let mut buffer = String::new();
        let mut stream = Box::pin(byte_stream);

        let mut response_id = String::new();
        let mut model = String::new();
        let mut content_blocks: Vec<crate::types::ContentBlock> = Vec::new();
        let mut tool_partial_json: HashMap<usize, String> = HashMap::new();
        let mut stop_reason = StopReason::End;
        let mut output_tokens: i64 = 0;

        while let Some(chunk_result) = stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    yield Err(err_provider_unavailable(Provider::Anthropic, format!("stream error: {}", e)));
                    return;
                }
            };

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process SSE format: "event: xxx\ndata: yyy\n\n"
            // We track how many bytes have been consumed and drain once per chunk
            // to avoid re-allocating the buffer on every event.
            loop {
                let event_prefix = "event: ";
                let data_prefix = "data: ";

                if let Some(event_pos) = buffer.find(event_prefix) {
                    let after_event = &buffer[event_pos + event_prefix.len()..];
                    if let Some(newline_pos) = after_event.find('\n') {
                        let event_type = after_event[..newline_pos].trim().to_string();
                        let remaining = &after_event[newline_pos + 1..];

                        if let Some(data_pos) = remaining.find(data_prefix) {
                            let after_data = &remaining[data_pos + data_prefix.len()..];
                            if let Some(data_newline) = after_data.find('\n') {
                                let data = after_data[..data_newline].trim().to_string();
                                // Compute how many bytes were consumed and drain them.
                                let consumed_len = event_pos + event_prefix.len() + newline_pos + 1
                                    + data_pos + data_prefix.len() + data_newline + 1;
                                buffer.drain(..consumed_len);

                                // Process the event
                                match event_type.as_str() {
                                    "message_start" => {
                                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) {
                                            if let Some(msg) = v.get("message") {
                                                response_id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                                model = msg.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                            }
                                        }
                                        yield Ok(StreamEvent::start(response_id.clone(), model.clone()));
                                    }
                                    "content_block_start" => {
                                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) {
                                            let idx = v.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                                            if let Some(cb) = v.get("content_block") {
                                                let cb_type = cb.get("type").and_then(|v| v.as_str()).unwrap_or("");
                                                while content_blocks.len() <= idx {
                                                    content_blocks.push(crate::types::ContentBlock::default());
                                                }
                                                if cb_type == "tool_use" {
                                                    let tool_id = cb.get("id").and_then(|v| v.as_str()).unwrap_or("");
                                                    let tool_name = cb.get("name").and_then(|v| v.as_str()).unwrap_or("");
                                                    content_blocks[idx] = crate::types::ContentBlock::tool_use(tool_id, tool_name, serde_json::Value::Null);
                                                    tool_partial_json.insert(idx, String::new());
                                                    yield Ok(StreamEvent::tool_call_start(crate::types::ToolCall {
                                                        id: tool_id.to_string(),
                                                        name: tool_name.to_string(),
                                                        input: serde_json::Value::Null,
                                                    }));
                                                } else {
                                                    content_blocks[idx] = crate::types::ContentBlock::text("");
                                                }
                                            }
                                        }
                                    }
                                    "content_block_delta" => {
                                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) {
                                            let idx = v.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                                            if let Some(delta) = v.get("delta") {
                                                if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                                                    if idx < content_blocks.len() {
                                                        let existing = content_blocks[idx].text.get_or_insert_with(String::new);
                                                        existing.push_str(text);
                                                    }
                                                    yield Ok(StreamEvent::content_delta(text, idx));
                                                } else if let Some(partial) = delta.get("partial_json").and_then(|v| v.as_str()) {
                                                    if let Some(builder) = tool_partial_json.get_mut(&idx) {
                                                        builder.push_str(partial);
                                                    }
                                                    yield Ok(StreamEvent::tool_call_delta(partial, idx));
                                                }
                                            }
                                        }
                                    }
                                    "content_block_stop" => {
                                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) {
                                            let idx = v.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                                            if idx < content_blocks.len() {
                                                let block = &content_blocks[idx];
                                                if block.content_type == Some(ContentType::ToolUse) {
                                                    let input_str = tool_partial_json.get(&idx).cloned().unwrap_or_default();
                                                    let input: serde_json::Value = serde_json::from_str(&input_str).unwrap_or(serde_json::Value::Null);
                                                    let tc = crate::types::ToolCall {
                                                        id: block.tool_use_id.clone().unwrap_or_default(),
                                                        name: block.tool_name.clone().unwrap_or_default(),
                                                        input,
                                                    };
                                                    yield Ok(StreamEvent::tool_call_end(tc));
                                                }
                                            }
                                        }
                                    }
                                    "message_delta" => {
                                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) {
                                            if let Some(delta) = v.get("delta") {
                                                if let Some(sr) = delta.get("stop_reason").and_then(|v| v.as_str()) {
                                                    stop_reason = transformer.transform_stop_reason(sr);
                                                }
                                            }
                                            if let Some(usage) = v.get("usage") {
                                                output_tokens = usage.get("output_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
                                            }
                                        }
                                    }
                                    "message_stop" => {
                                        let usage = crate::types::Usage {
                                            output_tokens,
                                            ..Default::default()
                                        };
                                        yield Ok(StreamEvent::done(Some(usage), Some(stop_reason.clone()), Some(response_id.clone())));
                                        return;
                                    }
                                    "error" => {
                                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) {
                                            let msg = v.get("error").and_then(|e| e.get("message"))
                                                .and_then(|m| m.as_str()).unwrap_or("unknown error");
                                            yield Err(err_server_error(Provider::Anthropic, msg));
                                        }
                                        return;
                                    }
                                    _ => {}
                                }
                                continue;
                            }
                        }
                    }
                }
                break;
            }
        }
    }
}
