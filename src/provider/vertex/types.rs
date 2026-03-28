//! Vertex AI batch prediction API types.

use crate::provider::google::types::GenerateContentResponse;
use serde::{Deserialize, Serialize};

/// Request body for creating a batch prediction job.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexBatchPredictionJobRequest {
    pub display_name: String,
    pub model: String,
    pub input_config: VertexBatchInputConfig,
    pub output_config: VertexBatchOutputConfig,
}

/// Specifies the input source for a batch job.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexBatchInputConfig {
    pub instances_format: String,
    pub gcs_source: GcsSource,
}

/// Specifies GCS URIs as input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcsSource {
    pub uris: Vec<String>,
}

/// Specifies the output destination for a batch job.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexBatchOutputConfig {
    pub predictions_format: String,
    pub gcs_destination: GcsDestination,
}

/// Specifies a GCS URI prefix for output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GcsDestination {
    pub output_uri_prefix: String,
}

/// Response from creating or getting a batch prediction job.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexBatchPredictionJob {
    pub name: String,
    #[serde(default)]
    pub display_name: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_config: Option<VertexBatchInputConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<VertexBatchOutputConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_info: Option<VertexBatchOutputInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<VertexRpcStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub create_time: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_time: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub update_time: Option<String>,
}

/// Contains the output information after job completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexBatchOutputInfo {
    #[serde(default)]
    pub gcs_output_directory: String,
}

/// A gRPC status error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexRpcStatus {
    pub code: i32,
    pub message: String,
}

/// Response from listing batch prediction jobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexBatchPredictionJobList {
    #[serde(default)]
    pub batch_prediction_jobs: Vec<VertexBatchPredictionJob>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_page_token: Option<String>,
}

/// A single line in the JSONL input file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexBatchInputLine {
    pub request: serde_json::Value,
}

/// A single line in the JSONL output file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexBatchOutputLine {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<VertexBatchOutputRequest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<GenerateContentResponse>,
    #[serde(default)]
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processed_time: Option<String>,
}

/// The echoed request in the batch output (we only need labels).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexBatchOutputRequest {
    #[serde(default)]
    pub labels: std::collections::HashMap<String, String>,
}

/// Response from listing Vertex publisher models.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct VertexModelsListResponse {
    #[serde(default)]
    pub models: Vec<VertexListedModel>,
    #[serde(default)]
    pub next_page_token: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct VertexListedModel {
    pub name: String,
    #[serde(default)]
    pub supported_generation_methods: Vec<String>,
}
