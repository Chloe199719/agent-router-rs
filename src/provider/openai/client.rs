//! OpenAI API client implementation.

use std::sync::Arc;
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use std::collections::HashMap;
use bytes::Bytes;

use crate::types::{
    CompletionRequest, CompletionResponse, Feature, Provider,
    StreamEvent, StopReason,
    Usage as UnifiedUsage,
};
use crate::errors::*;
use crate::provider::{ProviderClient, StreamResponse};
use super::transform::Transformer;
use super::types::*;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

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

        let base_url = cfg.base_url.clone().unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
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
            .header("Authorization", format!("Bearer {}", self.config.api_key))
    }

    pub async fn handle_error_response(&self, resp: reqwest::Response) -> RouterError {
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();

        if let Ok(err_resp) = serde_json::from_str::<ErrorResponse>(&body) {
            if let Some(api_err) = err_resp.error {
                return self.map_api_error(&api_err, status);
            }
        }

        err_server_error(Provider::OpenAI, body).with_status_code(status)
    }

    fn map_api_error(&self, api_err: &APIError, status_code: u16) -> RouterError {
        match status_code {
            401 => err_invalid_api_key(Provider::OpenAI).with_status_code(status_code),
            429 => err_rate_limit(Provider::OpenAI, &api_err.message).with_status_code(status_code),
            404 => err_model_not_found(Provider::OpenAI, &api_err.message).with_status_code(status_code),
            400 => {
                if api_err.message.contains("context_length") {
                    err_context_length(Provider::OpenAI, &api_err.message).with_status_code(status_code)
                } else {
                    err_invalid_request(&api_err.message)
                        .with_provider(Provider::OpenAI)
                        .with_status_code(status_code)
                }
            }
            _ => err_server_error(Provider::OpenAI, &api_err.message).with_status_code(status_code),
        }
    }
}

#[async_trait]
impl ProviderClient for Client {
    fn name(&self) -> Provider {
        Provider::OpenAI
    }

    fn supports_feature(&self, feature: &Feature) -> bool {
        matches!(
            feature,
            Feature::Streaming | Feature::StructuredOutput | Feature::Tools
            | Feature::Vision | Feature::Batch | Feature::Json
        )
    }

    async fn models(&self) -> Result<Vec<String>, RouterError> {
        let url = format!("{}/models", self.base_url.trim_end_matches('/'));
        let builder = self.http.get(&url);
        let builder = self.set_headers(builder);

        let resp = builder.send().await.map_err(|e| {
            err_provider_unavailable(Provider::OpenAI, format!("request failed: {}", e))
        })?;

        if !resp.status().is_success() {
            return Err(self.handle_error_response(resp).await);
        }

        let body: ListModelsResponse = resp.json().await.map_err(|e| {
            err_server_error(Provider::OpenAI, format!("failed to decode models list: {}", e))
        })?;

        let mut ids: Vec<String> = body.data.into_iter().map(|m| m.id).collect();
        ids.sort();
        ids.dedup();
        Ok(ids)
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, RouterError> {
        let mut oai_req = self.transformer.transform_request(req);
        oai_req.stream = Some(false);

        let builder = self.http.post(format!("{}/chat/completions", self.base_url))
            .json(&oai_req);
        let builder = self.set_headers(builder);

        let resp = builder.send().await.map_err(|e| {
            err_provider_unavailable(Provider::OpenAI, format!("request failed: {}", e))
        })?;

        if !resp.status().is_success() {
            return Err(self.handle_error_response(resp).await);
        }

        let oai_resp: ChatCompletionResponse = resp.json().await.map_err(|e| {
            err_server_error(Provider::OpenAI, format!("failed to decode response: {}", e))
        })?;

        self.transformer.transform_response(&oai_resp)
            .ok_or_else(|| err_server_error(Provider::OpenAI, "empty response"))
    }

    async fn stream(&self, req: &CompletionRequest) -> Result<StreamResponse, RouterError> {
        let mut oai_req = self.transformer.transform_request(req);
        oai_req.stream = Some(true);
        oai_req.stream_options = Some(StreamOptions { include_usage: true });

        let builder = self.http.post(format!("{}/chat/completions", self.base_url))
            .json(&oai_req);
        let builder = self.set_headers(builder);

        let resp = builder.send().await.map_err(|e| {
            err_provider_unavailable(Provider::OpenAI, format!("request failed: {}", e))
        })?;

        if !resp.status().is_success() {
            return Err(self.handle_error_response(resp).await);
        }

        let transformer = Arc::new(Transformer::new());
        let byte_stream = resp.bytes_stream();
        let event_stream = openai_sse_stream(byte_stream, transformer);

        Ok(Box::pin(event_stream))
    }
}

/// Parse OpenAI SSE stream into StreamEvents.
fn openai_sse_stream(
    byte_stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    transformer: Arc<Transformer>,
) -> impl Stream<Item = Result<StreamEvent, RouterError>> + Send + 'static {
    let mut buffer = String::new();
    let mut id = String::new();
    let mut model = String::new();
    let mut content = String::new();
    let mut tool_calls: HashMap<usize, (String, String, String)> = HashMap::new(); // idx -> (id, name, args)
    let mut usage: Option<UnifiedUsage> = None;
    let mut stop_reason = StopReason::End;
    let mut started = false;

    async_stream::stream! {
        let mut stream = Box::pin(byte_stream);

        while let Some(chunk_result) = stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    yield Err(err_provider_unavailable(Provider::OpenAI, format!("stream error: {}", e)));
                    return;
                }
            };

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete lines without re-allocating the buffer on every line.
            // We advance a cursor and drain everything consumed at once.
            let mut start = 0;
            while let Some(rel) = buffer[start..].find('\n') {
                let end = start + rel;
                let line = buffer[start..end].trim().to_string();
                start = end + 1;

                if line.is_empty() {
                    continue;
                }

                if !line.starts_with("data: ") {
                    continue;
                }

                let data = &line["data: ".len()..];
                if data == "[DONE]" {
                    if !started {
                        yield Ok(StreamEvent::start(id.clone(), model.clone()));
                    }
                    yield Ok(StreamEvent::done(usage.clone(), Some(stop_reason.clone()), Some(id.clone())));
                    return;
                }

                let chunk: StreamChunk = match serde_json::from_str(data) {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                if id.is_empty() { id = chunk.id.clone(); }
                if model.is_empty() { model = chunk.model.clone(); }

                if !started {
                    started = true;
                    yield Ok(StreamEvent::start(id.clone(), model.clone()));
                }

                if let Some(ref u) = chunk.usage {
                    usage = Some(UnifiedUsage {
                        input_tokens: u.prompt_tokens,
                        output_tokens: u.completion_tokens,
                        total_tokens: u.total_tokens,
                        cached_tokens: u.prompt_tokens_details.as_ref().map(|d| d.cached_tokens),
                        reasoning_tokens: u.completion_tokens_details.as_ref().map(|d| d.reasoning_tokens),
                    });
                }

                if chunk.choices.is_empty() {
                    continue;
                }

                let choice = &chunk.choices[0];
                let delta = &choice.delta;

                if let Some(ref fr) = choice.finish_reason {
                    stop_reason = transformer.transform_stop_reason(fr);
                }

                if let Some(ref text) = delta.content {
                    if !text.is_empty() {
                        content.push_str(text);
                        yield Ok(StreamEvent::content_delta(text.clone(), 0));
                    }
                }

                if let Some(ref tcs) = delta.tool_calls {
                    for tc in tcs {
                        let idx = tc.index.unwrap_or(0) as usize;
                        if !tc.id.is_empty() {
                            tool_calls.insert(idx, (tc.id.clone(), tc.function.name.clone(), String::new()));
                            yield Ok(StreamEvent::tool_call_start(crate::types::ToolCall {
                                id: tc.id.clone(),
                                name: tc.function.name.clone(),
                                input: serde_json::Value::Null,
                            }));
                        } else if !tc.function.arguments.is_empty() {
                            if let Some(entry) = tool_calls.get_mut(&idx) {
                                entry.2.push_str(&tc.function.arguments);
                            }
                            yield Ok(StreamEvent::tool_call_delta(tc.function.arguments.clone(), idx));
                        }
                    }
                }
            }
            // Discard all bytes that were consumed above in one allocation-free drain.
            buffer.drain(..start);
        }

        // Finalize remaining tool calls
        for (_, (tc_id, name, args)) in &tool_calls {
            let input: serde_json::Value = serde_json::from_str(args).unwrap_or(serde_json::Value::Null);
            yield Ok(StreamEvent::tool_call_end(crate::types::ToolCall {
                id: tc_id.clone(),
                name: name.clone(),
                input,
            }));
        }

        yield Ok(StreamEvent::done(usage, Some(stop_reason), Some(id)));
    }
}
