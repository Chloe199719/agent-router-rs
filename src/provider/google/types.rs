//! Google Gemini API types.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// GenerateContentRequest is the Google Gemini API request.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentRequest {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub safety_settings: Vec<SafetySetting>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<std::collections::HashMap<String, String>>,
}

/// Content is a content message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    pub parts: Vec<Part>,
}

/// Part is a content part.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inline_data: Option<InlineData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_data: Option<FileData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_response: Option<FunctionResponse>,
}

/// InlineData is inline binary data (images, etc).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InlineData {
    pub mime_type: String,
    pub data: String, // base64 encoded
}

/// FileData is a reference to a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileData {
    pub mime_type: String,
    pub file_uri: String,
}

/// FunctionCall is a function call from the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub args: Value,
}

/// FunctionResponse is a function response from the user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResponse {
    pub name: String,
    pub response: Value,
}

/// GenerationConfig configures generation parameters.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub stop_sequences: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<Schema>,
}

/// Schema is Google's schema format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    #[serde(rename = "type")]
    pub schema_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<std::collections::HashMap<String, Box<Schema>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<Schema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nullable: Option<bool>,
}

/// SafetySetting configures safety thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetySetting {
    pub category: String,
    pub threshold: String,
}

/// Tool is a Google tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub function_declarations: Vec<FunctionDeclaration>,
}

/// FunctionDeclaration declares a function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Schema>,
}

/// ToolConfig configures tool usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_calling_config: Option<FunctionCallingConfig>,
}

/// FunctionCallingConfig configures function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {
    pub mode: String, // "AUTO", "ANY", "NONE"
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub allowed_function_names: Vec<String>,
}

/// GenerateContentResponse is the response from generateContent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_feedback: Option<PromptFeedback>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<UsageMetadata>,
}

/// Candidate is a response candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub content: Option<Content>,
    #[serde(default)]
    pub finish_reason: String,
    #[serde(default)]
    pub index: i32,
    #[serde(default)]
    pub safety_ratings: Vec<SafetyRating>,
}

/// SafetyRating is a safety rating for content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyRating {
    pub category: String,
    pub probability: String,
}

/// PromptFeedback is feedback about the prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptFeedback {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_reason: Option<String>,
    #[serde(default)]
    pub safety_ratings: Vec<SafetyRating>,
}

/// UsageMetadata contains usage information.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    #[serde(default)]
    pub prompt_token_count: i32,
    #[serde(default)]
    pub candidates_token_count: i32,
    #[serde(default)]
    pub total_token_count: i32,
}

/// StreamChunk is a streaming response chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamChunk {
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<UsageMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_feedback: Option<PromptFeedback>,
}

/// ErrorResponse is a Google API error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: Option<APIError>,
}

/// APIError is a Google API error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIError {
    pub code: i32,
    pub message: String,
    pub status: String,
}

// ---- Batch API types ----

/// BatchGenerateContentRequest is the request to create a batch job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchGenerateContentRequest {
    pub batch: BatchConfig,
}

/// BatchConfig configures a batch job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    pub input_config: InputConfig,
}

/// InputConfig specifies the input for a batch job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requests: Option<RequestsInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_name: Option<String>,
}

/// RequestsInput contains inline requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestsInput {
    pub requests: Vec<BatchRequestItem>,
}

/// BatchRequestItem is a single request in a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequestItem {
    pub request: GenerateContentRequest,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<RequestMetadata>,
}

/// RequestMetadata contains metadata for a batch request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetadata {
    pub key: String,
}

/// BatchJob represents a batch job (long-running operation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchJob {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<BatchMetadata>,
    #[serde(default)]
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<StatusError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<BatchResponse>,
}

/// BatchMetadata contains batch job metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchMetadata {
    #[serde(rename = "@type", skip_serializing_if = "Option::is_none")]
    pub type_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(default)]
    pub state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub create_time: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub update_time: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<BatchOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_stats: Option<BatchStats>,
}

/// BatchOutput contains batch output in metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchOutput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inlined_responses: Option<InlinedResponsesWrapper>,
}

/// BatchStats contains batch job statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchStats {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_count: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub successful_request_count: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_request_count: Option<String>,
}

/// StatusError is an error status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusError {
    pub code: i32,
    pub message: String,
}

/// BatchResponse is the response from a completed batch job.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchResponse {
    #[serde(rename = "@type", skip_serializing_if = "Option::is_none")]
    pub type_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inlined_responses: Option<InlinedResponsesWrapper>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub responses_file: Option<String>,
}

/// InlinedResponsesWrapper wraps the array of inlined responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InlinedResponsesWrapper {
    pub inlined_responses: Vec<InlinedResponse>,
}

/// InlinedResponse is an inline response from a batch job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InlinedResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<GenerateContentResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ResponseMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<StatusError>,
}

/// ResponseMetadata contains metadata for a batch response item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
}

/// BatchListResponse is the response from listing batches.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchListResponse {
    #[serde(default)]
    pub batches: Vec<BatchJob>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_page_token: Option<String>,
}

/// FileUploadResponse is the response from uploading a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileUploadResponse {
    pub file: Option<UploadedFile>,
}

/// UploadedFile represents an uploaded file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UploadedFile {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub create_time: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
}
