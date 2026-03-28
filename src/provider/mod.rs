//! Provider traits and common configuration.

use async_trait::async_trait;
use std::collections::HashMap;
use std::pin::Pin;
use futures::Stream;
use crate::types::{Provider, Feature, CompletionRequest, CompletionResponse, StreamEvent};
use crate::errors::RouterError;

/// A boxed stream of stream events.
pub type StreamResponse = Pin<Box<dyn Stream<Item = Result<StreamEvent, RouterError>> + Send>>;

/// The interface that all LLM providers must implement.
#[async_trait]
pub trait ProviderClient: Send + Sync {
    /// Returns the provider identifier.
    fn name(&self) -> Provider;

    /// Sends a completion request and returns the response.
    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, RouterError>;

    /// Sends a streaming completion request and returns a stream.
    async fn stream(&self, req: &CompletionRequest) -> Result<StreamResponse, RouterError>;

    /// Checks if the provider supports a specific feature.
    fn supports_feature(&self, feature: &Feature) -> bool;

    /// Lists model identifiers by calling the provider's models API (current catalog, not a static list).
    async fn models(&self) -> Result<Vec<String>, RouterError>;
}

/// Optional interface for providers that support batch processing.
#[async_trait]
pub trait BatchProviderClient: ProviderClient {
    /// Creates a new batch job.
    async fn create_batch(&self, requests: Vec<BatchRequest>) -> Result<BatchJob, RouterError>;

    /// Retrieves the status of a batch job.
    async fn get_batch(&self, batch_id: &str) -> Result<BatchJob, RouterError>;

    /// Retrieves the results of a completed batch job.
    async fn get_batch_results(&self, batch_id: &str) -> Result<Vec<BatchResult>, RouterError>;

    /// Cancels a batch job.
    async fn cancel_batch(&self, batch_id: &str) -> Result<(), RouterError>;

    /// Lists all batch jobs.
    async fn list_batches(&self, opts: Option<ListBatchOptions>) -> Result<Vec<BatchJob>, RouterError>;
}

/// Wraps a completion request with a custom ID for batch processing.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    /// Developer-provided ID for matching results to requests.
    pub custom_id: String,
    /// The completion request to process.
    pub request: CompletionRequest,
}

/// A batch processing job.
#[derive(Debug, Clone)]
pub struct BatchJob {
    /// Unique identifier for this batch.
    pub id: String,
    /// Provider processing this batch.
    pub provider: Provider,
    /// Status of the batch job.
    pub status: BatchStatus,
    /// When the batch was created (Unix timestamp).
    pub created_at: i64,
    /// When the batch completed (Unix timestamp).
    pub completed_at: i64,
    /// When the batch will expire (Unix timestamp).
    pub expires_at: i64,
    /// Progress counts.
    pub request_counts: RequestCounts,
    /// Provider-specific metadata.
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

/// Batch job status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchStatus {
    Pending,
    Validating,
    InProgress,
    Finalizing,
    Completed,
    Failed,
    Cancelled,
    Expired,
}

impl std::fmt::Display for BatchStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchStatus::Pending => write!(f, "pending"),
            BatchStatus::Validating => write!(f, "validating"),
            BatchStatus::InProgress => write!(f, "in_progress"),
            BatchStatus::Finalizing => write!(f, "finalizing"),
            BatchStatus::Completed => write!(f, "completed"),
            BatchStatus::Failed => write!(f, "failed"),
            BatchStatus::Cancelled => write!(f, "cancelled"),
            BatchStatus::Expired => write!(f, "expired"),
        }
    }
}

impl BatchStatus {
    /// Returns true if the batch is in a terminal state.
    pub fn is_done(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled | Self::Expired)
    }
}

/// Tracks batch request progress.
#[derive(Debug, Clone, Default)]
pub struct RequestCounts {
    pub total: i32,
    pub completed: i32,
    pub failed: i32,
}

/// A single result from a batch job.
#[derive(Debug)]
pub struct BatchResult {
    /// Matches the request's custom_id.
    pub custom_id: String,
    /// Echoed Gemini `labels` from the batch output line (`request.labels`), when present.
    pub request_labels: Option<HashMap<String, String>>,
    /// The completion response (if successful).
    pub response: Option<CompletionResponse>,
    /// The error that occurred (if failed).
    pub error: Option<RouterError>,
}

/// Options for listing batches.
#[derive(Debug, Clone, Default)]
pub struct ListBatchOptions {
    /// Maximum number of batches to return.
    pub limit: Option<i32>,
    /// Cursor for pagination.
    pub after: Option<String>,
}

/// Common configuration for providers.
#[derive(Debug, Clone, Default)]
pub struct ProviderConfig {
    /// API key for authentication.
    pub api_key: String,
    /// Custom base URL.
    pub base_url: Option<String>,
    /// Request timeout in seconds.
    pub timeout: Option<u64>,
    /// Maximum number of retries.
    pub max_retries: Option<u32>,
    /// Debug logging.
    pub debug: bool,
    /// Google Cloud project ID (Vertex AI).
    pub project_id: Option<String>,
    /// Google Cloud region (Vertex AI).
    pub location: Option<String>,
    /// OAuth2 access token (Vertex AI).
    pub access_token: Option<String>,
    /// GCS bucket for Vertex AI batch staging.
    pub batch_bucket: Option<String>,
}

impl ProviderConfig {
    pub fn default_with_timeout() -> Self {
        Self {
            timeout: Some(120),
            max_retries: Some(3),
            ..Default::default()
        }
    }
}

/// A function that configures a provider. Uses `FnOnce` so captured values are
/// moved rather than cloned on every call — options are applied exactly once.
pub type ProviderOption = Box<dyn FnOnce(&mut ProviderConfig) + Send>;

/// Set the API key.
pub fn with_api_key(key: impl Into<String>) -> ProviderOption {
    let key = key.into();
    Box::new(move |c: &mut ProviderConfig| { c.api_key = key; })
}

/// Set a custom base URL.
pub fn with_base_url(url: impl Into<String>) -> ProviderOption {
    let url = url.into();
    Box::new(move |c: &mut ProviderConfig| { c.base_url = Some(url); })
}

/// Set the request timeout in seconds.
pub fn with_timeout(secs: u64) -> ProviderOption {
    Box::new(move |c: &mut ProviderConfig| { c.timeout = Some(secs); })
}

/// Set the maximum number of retries.
pub fn with_max_retries(n: u32) -> ProviderOption {
    Box::new(move |c: &mut ProviderConfig| { c.max_retries = Some(n); })
}

/// Enable debug logging.
pub fn with_debug(debug: bool) -> ProviderOption {
    Box::new(move |c: &mut ProviderConfig| { c.debug = debug; })
}

/// Set the Google Cloud project ID.
pub fn with_project_id(id: impl Into<String>) -> ProviderOption {
    let id = id.into();
    Box::new(move |c: &mut ProviderConfig| { c.project_id = Some(id); })
}

/// Set the Google Cloud region.
pub fn with_location(location: impl Into<String>) -> ProviderOption {
    let location = location.into();
    Box::new(move |c: &mut ProviderConfig| { c.location = Some(location); })
}

/// Set the OAuth2 access token.
pub fn with_access_token(token: impl Into<String>) -> ProviderOption {
    let token = token.into();
    Box::new(move |c: &mut ProviderConfig| { c.access_token = Some(token); })
}

/// Set the GCS bucket for Vertex AI batch staging.
pub fn with_batch_bucket(bucket: impl Into<String>) -> ProviderOption {
    let bucket = bucket.into();
    Box::new(move |c: &mut ProviderConfig| { c.batch_bucket = Some(bucket); })
}

/// Apply options to a config. Each option is consumed exactly once.
pub fn apply_options(cfg: &mut ProviderConfig, opts: Vec<ProviderOption>) {
    for opt in opts {
        opt(cfg);
    }
}

pub mod openai;
pub mod anthropic;
pub mod google;
pub mod vertex;
