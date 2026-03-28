//! Google Gemini API client implementation.

use std::sync::Arc;
use async_trait::async_trait;

use crate::errors::*;
use crate::provider::{ProviderClient, ProviderConfig, StreamResponse, apply_options, ProviderOption};
use crate::types::{CompletionRequest, CompletionResponse, Feature, Provider, StopReason, Usage, ContentBlock, ToolCall};
use super::transform::Transformer;
use super::types::*;

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

pub struct Client {
    pub config: Arc<ProviderConfig>,
    pub http: reqwest::Client,
    pub base_url: String,
    pub transformer: Transformer,
}

impl Client {
    pub fn new(opts: Vec<ProviderOption>) -> Result<Self, RouterError> {
        let mut cfg = ProviderConfig::default_with_timeout();
        apply_options(&mut cfg, opts);

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

    fn build_url(&self, model: &str, stream: bool) -> String {
        let action = if stream { "streamGenerateContent" } else { "generateContent" };
        format!("{}/models/{}:{}?key={}", self.base_url, model, action, self.config.api_key)
    }

    pub(crate) fn handle_error_response_sync(
        &self,
        status: u16,
        body: &[u8],
    ) -> RouterError {
        if let Ok(err_resp) = serde_json::from_slice::<ErrorResponse>(body) {
            if let Some(api_err) = err_resp.error {
                return self.map_api_error(&api_err, status);
            }
        }
        let msg = String::from_utf8_lossy(body).to_string();
        err_server_error(Provider::Google, &msg).with_status_code(status)
    }

    fn map_api_error(&self, api_err: &APIError, status_code: u16) -> RouterError {
        match status_code {
            401 => err_invalid_api_key(Provider::Google).with_status_code(status_code),
            429 => err_rate_limit(Provider::Google, &api_err.message).with_status_code(status_code),
            404 => err_model_not_found(Provider::Google, &api_err.message).with_status_code(status_code),
            400 => {
                let msg = &api_err.message;
                if msg.contains("context") || msg.contains("token") {
                    err_context_length(Provider::Google, msg).with_status_code(status_code)
                } else {
                    err_invalid_request(msg).with_provider(Provider::Google).with_status_code(status_code)
                }
            }
            _ => err_server_error(Provider::Google, &api_err.message).with_status_code(status_code),
        }
    }
}

#[async_trait]
impl ProviderClient for Client {
    fn name(&self) -> Provider {
        Provider::Google
    }

    fn supports_feature(&self, feature: &Feature) -> bool {
        matches!(
            feature,
            Feature::Streaming
                | Feature::StructuredOutput
                | Feature::Tools
                | Feature::Vision
                | Feature::Json
                | Feature::Batch
        )
    }

    async fn models(&self) -> Result<Vec<String>, RouterError> {
        let mut collected = Vec::new();
        let mut page_token: Option<String> = None;

        loop {
            let mut req = self
                .http
                .get(format!("{}/models", self.base_url.trim_end_matches('/')))
                .query(&[("key", self.config.api_key.as_str())]);
            if let Some(ref t) = page_token {
                req = req.query(&[("pageToken", t.as_str())]);
            }

            let resp = req
                .send()
                .await
                .map_err(|e| err_provider_unavailable(Provider::Google, &e.to_string()))?;

            let status = resp.status().as_u16();
            let body = resp
                .bytes()
                .await
                .map_err(|e| err_server_error(Provider::Google, &e.to_string()))?;

            if status != 200 {
                return Err(self.handle_error_response_sync(status, &body));
            }

            let page: GeminiModelsListResponse = serde_json::from_slice(&body).map_err(|e| {
                err_server_error(Provider::Google, &format!("failed to decode models list: {}", e))
            })?;

            for m in page.models {
                let id = m
                    .name
                    .strip_prefix("models/")
                    .unwrap_or(&m.name)
                    .to_string();
                let include = if m.supported_generation_methods.is_empty() {
                    !id.to_ascii_lowercase().contains("embedding")
                } else {
                    m.supported_generation_methods
                        .iter()
                        .any(|x| x == "generateContent")
                };
                if include {
                    collected.push(id);
                }
            }

            page_token = page.next_page_token;
            if page_token.is_none() {
                break;
            }
        }

        collected.sort();
        collected.dedup();
        Ok(collected)
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, RouterError> {
        let g_req = self.transformer.transform_request(req);
        let url = self.build_url(&req.model, false);

        let resp = self.http
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&g_req)
            .send()
            .await
            .map_err(|e| err_provider_unavailable(Provider::Google, &e.to_string()))?;

        let status = resp.status().as_u16();
        let body = resp.bytes().await
            .map_err(|e| err_server_error(Provider::Google, &e.to_string()))?;

        if status != 200 {
            return Err(self.handle_error_response_sync(status, &body));
        }

        let g_resp: GenerateContentResponse = serde_json::from_slice(&body)
            .map_err(|e| err_server_error(Provider::Google, &format!("failed to decode response: {}", e)))?;

        let mut result = self.transformer.transform_response(&g_resp)
            .ok_or_else(|| err_server_error(Provider::Google, "empty response"))?;
        result.model = req.model.clone();
        Ok(result)
    }

    /// Returns a stream of `StreamEvent`s for the given request.
    ///
    /// # Note on streaming behaviour
    /// Google's Gemini streaming endpoint (`streamGenerateContent`) returns a
    /// JSON array rather than true SSE chunks, so the entire response body is
    /// buffered before any events are yielded. The stream is therefore
    /// simulated: all events are emitted synchronously once the body arrives.
    /// True token-by-token incremental delivery is not supported by this
    /// provider at the transport level.
    async fn stream(&self, req: &CompletionRequest) -> Result<StreamResponse, RouterError> {
        let g_req = self.transformer.transform_request(req);
        let url = self.build_url(&req.model, true);
        let model = req.model.clone();
        let transformer = Transformer::new();

        let resp = self.http
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&g_req)
            .send()
            .await
            .map_err(|e| err_provider_unavailable(Provider::Google, &e.to_string()))?;

        let status = resp.status().as_u16();
        if status != 200 {
            let body = resp.bytes().await
                .map_err(|e| err_server_error(Provider::Google, &e.to_string()))?;
            return Err(self.handle_error_response_sync(status, &body));
        }

        let body_bytes = resp.bytes().await
            .map_err(|e| err_server_error(Provider::Google, &e.to_string()))?;

        let stream = async_stream::stream! {
            use crate::types::{StreamEvent, StreamEventType};

            // Emit start event
            yield Ok(StreamEvent {
                event_type: StreamEventType::Start,
                model: Some(model.clone()),
                ..Default::default()
            });

            // Parse as JSON array of StreamChunks
            let mut acc_content: Vec<ContentBlock> = Vec::new();
            let mut acc_tool_calls: Vec<ToolCall> = Vec::new();
            let mut usage: Option<Usage> = None;
            let mut stop_reason: StopReason = StopReason::End;

            let chunks: Vec<StreamChunk> = match serde_json::from_slice(&body_bytes) {
                Ok(v) => v,
                Err(e) => {
                    yield Err(err_server_error(Provider::Google, &format!("failed to parse stream: {}", e)));
                    return;
                }
            };

            for chunk in &chunks {
                if let Some(event) = transformer.process_chunk(
                    chunk,
                    &mut acc_content,
                    &mut acc_tool_calls,
                    &mut usage,
                    &mut stop_reason,
                ) {
                    yield Ok(event);
                }
            }

            yield Ok(StreamEvent {
                event_type: StreamEventType::Done,
                usage,
                stop_reason: Some(stop_reason),
                ..Default::default()
            });
        };

        Ok(Box::pin(stream))
    }
}
