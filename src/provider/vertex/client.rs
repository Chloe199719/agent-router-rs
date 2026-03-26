//! Google Vertex AI client implementation.
//!
//! Vertex AI uses the same Gemini API request/response format as the standard
//! Google Gemini API, but with a different base URL pattern and authentication
//! mechanism (OAuth2 Bearer token or API key).
//!
//! URL pattern:
//!   https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL}:{ACTION}
//!
//! For "global" location:
//!   https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/global/publishers/google/models/{MODEL}:{ACTION}

use std::sync::Arc;
use async_trait::async_trait;

use crate::errors::*;
use crate::provider::{ProviderClient, ProviderConfig, StreamResponse, apply_options, ProviderOption};
use crate::provider::google::{self, Transformer, types::{GenerateContentResponse, ErrorResponse, APIError}};
use crate::types::{CompletionRequest, CompletionResponse, ContentBlock, Feature, Provider, StopReason, ToolCall, Usage};

pub struct Client {
    pub config: Arc<ProviderConfig>,
    pub http: reqwest::Client,
    pub project_id: String,
    pub location: String,
    pub base_url: String,
    pub transformer: Transformer,
}

impl Client {
    pub fn new(project_id: impl Into<String>, location: impl Into<String>, opts: Vec<ProviderOption>) -> Result<Self, RouterError> {
        let mut cfg = ProviderConfig::default_with_timeout();
        apply_options(&mut cfg, opts);

        let project_id = {
            let p = project_id.into();
            if p.is_empty() { cfg.project_id.clone().unwrap_or_default() } else { p }
        };
        let location = {
            let l = location.into();
            if l.is_empty() { cfg.location.clone().unwrap_or_else(|| "us-central1".to_string()) } else { l }
        };

        let base_url = cfg.base_url.clone().unwrap_or_else(|| {
            if location == "global" {
                "https://aiplatform.googleapis.com/v1".to_string()
            } else {
                format!("https://{}-aiplatform.googleapis.com/v1", location)
            }
        });

        let timeout = cfg.timeout.unwrap_or(120);
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout))
            .build()
            .map_err(|e| err_invalid_request(format!("failed to build HTTP client: {}", e)))?;

        Ok(Self {
            config: Arc::new(cfg),
            http,
            project_id,
            location,
            base_url,
            transformer: Transformer::new(),
        })
    }

    pub(crate) fn build_url(&self, model: &str, action: &str) -> String {
        let mut url = format!(
            "{}/projects/{}/locations/{}/publishers/google/models/{}:{}",
            self.base_url, self.project_id, self.location, model, action
        );

        if self.config.access_token.as_deref().map(|t| t.is_empty()).unwrap_or(true) {
            if !self.config.api_key.is_empty() {
                url.push_str(&format!("?key={}", self.config.api_key));
            }
        }

        url
    }

    pub(crate) fn set_auth_header(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(token) = &self.config.access_token {
            if !token.is_empty() {
                return builder.header("Authorization", format!("Bearer {}", token));
            }
        }
        builder
    }

    pub(crate) fn handle_error_response_sync(&self, status: u16, body: &[u8]) -> RouterError {
        if let Ok(err_resp) = serde_json::from_slice::<ErrorResponse>(body) {
            if let Some(api_err) = err_resp.error {
                return self.map_api_error(&api_err, status);
            }
        }
        let msg = String::from_utf8_lossy(body).to_string();
        err_server_error(Provider::Vertex, &msg).with_status_code(status)
    }

    fn map_api_error(&self, api_err: &APIError, status_code: u16) -> RouterError {
        match status_code {
            401 => err_invalid_api_key(Provider::Vertex).with_status_code(status_code),
            403 => err_authentication(Provider::Vertex, &api_err.message).with_status_code(status_code),
            429 => err_rate_limit(Provider::Vertex, &api_err.message).with_status_code(status_code),
            404 => err_model_not_found(Provider::Vertex, &api_err.message).with_status_code(status_code),
            400 => {
                let msg = &api_err.message;
                if msg.contains("context") || msg.contains("token") {
                    err_context_length(Provider::Vertex, msg).with_status_code(status_code)
                } else {
                    err_invalid_request(msg).with_provider(Provider::Vertex).with_status_code(status_code)
                }
            }
            _ => err_server_error(Provider::Vertex, &api_err.message).with_status_code(status_code),
        }
    }
}

#[async_trait]
impl ProviderClient for Client {
    fn name(&self) -> Provider {
        Provider::Vertex
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

    fn models(&self) -> Vec<String> {
        vec![
            "gemini-2.0-flash".to_string(),
            "gemini-2.0-flash-lite".to_string(),
            "gemini-1.5-pro".to_string(),
            "gemini-1.5-flash".to_string(),
            "gemini-1.5-flash-8b".to_string(),
            "gemini-1.0-pro".to_string(),
        ]
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, RouterError> {
        let mut g_req = self.transformer.transform_request(req);
        google::transform::apply_metadata_as_labels(&mut g_req, req.metadata.as_ref());
        let url = self.build_url(&req.model, "generateContent");

        let builder = self.http
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&g_req);
        let builder = self.set_auth_header(builder);

        let resp = builder.send().await
            .map_err(|e| err_provider_unavailable(Provider::Vertex, &e.to_string()))?;

        let status = resp.status().as_u16();
        let body = resp.bytes().await
            .map_err(|e| err_server_error(Provider::Vertex, &e.to_string()))?;

        if status != 200 {
            return Err(self.handle_error_response_sync(status, &body));
        }

        let g_resp: GenerateContentResponse = serde_json::from_slice(&body)
            .map_err(|e| err_server_error(Provider::Vertex, &format!("failed to decode response: {}", e)))?;

        let mut result = self.transformer.transform_response(&g_resp)
            .ok_or_else(|| err_server_error(Provider::Vertex, "empty response"))?;
        result.provider = Provider::Vertex;
        result.model = req.model.clone();
        Ok(result)
    }

    /// Returns a stream of `StreamEvent`s for the given request.
    ///
    /// # Note on streaming behaviour
    /// Vertex AI's streaming endpoint returns a JSON array rather than true SSE
    /// chunks, so the entire response body is buffered before any events are
    /// yielded. The stream is therefore simulated: all events are emitted
    /// synchronously once the body arrives.
    async fn stream(&self, req: &CompletionRequest) -> Result<StreamResponse, RouterError> {
        let mut g_req = self.transformer.transform_request(req);
        google::transform::apply_metadata_as_labels(&mut g_req, req.metadata.as_ref());
        let url = self.build_url(&req.model, "streamGenerateContent");
        let model = req.model.clone();
        let transformer = Transformer::new();

        let builder = self.http
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&g_req);
        let builder = self.set_auth_header(builder);

        let resp = builder.send().await
            .map_err(|e| err_provider_unavailable(Provider::Vertex, &e.to_string()))?;

        let status = resp.status().as_u16();
        if status != 200 {
            let body = resp.bytes().await
                .map_err(|e| err_server_error(Provider::Vertex, &e.to_string()))?;
            return Err(self.handle_error_response_sync(status, &body));
        }

        let body_bytes = resp.bytes().await
            .map_err(|e| err_server_error(Provider::Vertex, &e.to_string()))?;

        use crate::provider::google::types::StreamChunk;

        let stream = async_stream::stream! {
            use crate::types::{StreamEvent, StreamEventType};

            yield Ok(StreamEvent {
                event_type: StreamEventType::Start,
                model: Some(model.clone()),
                ..Default::default()
            });

            let mut acc_content: Vec<ContentBlock> = Vec::new();
            let mut acc_tool_calls: Vec<ToolCall> = Vec::new();
            let mut usage: Option<Usage> = None;
            let mut stop_reason: StopReason = StopReason::End;

            let chunks: Vec<StreamChunk> = match serde_json::from_slice(&body_bytes) {
                Ok(v) => v,
                Err(e) => {
                    yield Err(err_server_error(Provider::Vertex, &format!("failed to parse stream: {}", e)));
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
