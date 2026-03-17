//! OpenAI batch processing implementation.

use async_trait::async_trait;
use crate::errors::*;
use crate::provider::{BatchProviderClient, BatchRequest, BatchJob, BatchResult, BatchStatus, ListBatchOptions, RequestCounts};
use crate::types::Provider;
use super::client::Client;
use super::types::*;
use std::collections::HashMap;

#[async_trait]
impl BatchProviderClient for Client {
    async fn create_batch(&self, requests: Vec<BatchRequest>) -> Result<BatchJob, RouterError> {
        // Step 1: Build JSONL content
        let mut jsonl = String::new();
        for req in &requests {
            let mut oai_req = self.transformer.transform_request(&req.request);
            oai_req.stream = Some(false);

            let body = serde_json::to_value(&oai_req)
                .map_err(|e| err_invalid_request(format!("marshal request: {}", e)))?;

            let line = BatchInputLine {
                custom_id: req.custom_id.clone(),
                method: "POST".to_string(),
                url: "/v1/chat/completions".to_string(),
                body,
            };
            let encoded = serde_json::to_string(&line)
                .map_err(|e| err_invalid_request(format!("encode batch line: {}", e)))?;
            jsonl.push_str(&encoded);
            jsonl.push('\n');
        }

        // Step 2: Upload JSONL file
        let file_id = self.upload_batch_file(jsonl.into_bytes()).await?;

        // Step 3: Create batch
        let create_req = serde_json::json!({
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h"
        });

        let resp = self.http
            .post(format!("{}/batches", self.base_url))
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .json(&create_req)
            .send()
            .await
            .map_err(|e| err_provider_unavailable(Provider::OpenAI, format!("request failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(self.handle_error_response_pub(resp).await);
        }

        let batch: BatchObject = resp.json().await
            .map_err(|e| err_server_error(Provider::OpenAI, format!("decode response: {}", e)))?;

        Ok(self.convert_batch_job(&batch))
    }

    async fn get_batch(&self, batch_id: &str) -> Result<BatchJob, RouterError> {
        let resp = self.http
            .get(format!("{}/batches/{}", self.base_url, batch_id))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .send()
            .await
            .map_err(|e| err_provider_unavailable(Provider::OpenAI, format!("request failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(self.handle_error_response_pub(resp).await);
        }

        let batch: BatchObject = resp.json().await
            .map_err(|e| err_server_error(Provider::OpenAI, format!("decode response: {}", e)))?;

        Ok(self.convert_batch_job(&batch))
    }

    async fn get_batch_results(&self, batch_id: &str) -> Result<Vec<BatchResult>, RouterError> {
        let job = self.get_batch(batch_id).await?;

        let output_file_id = job.metadata.get("output_file_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| err_invalid_request("batch has no output file").with_provider(Provider::OpenAI))?;

        let resp = self.http
            .get(format!("{}/files/{}/content", self.base_url, output_file_id))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .send()
            .await
            .map_err(|e| err_provider_unavailable(Provider::OpenAI, format!("request failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(self.handle_error_response_pub(resp).await);
        }

        let content = resp.text().await
            .map_err(|e| err_server_error(Provider::OpenAI, format!("read response: {}", e)))?;

        let mut results = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() { continue; }
            if let Ok(output) = serde_json::from_str::<BatchOutputLine>(line) {
                let result = if let Some(err) = output.error {
                    BatchResult {
                        custom_id: output.custom_id,
                        response: None,
                        error: Some(err_server_error(Provider::OpenAI, err.message)),
                    }
                } else if let Some(resp_data) = output.response {
                    BatchResult {
                        custom_id: output.custom_id,
                        response: self.transformer.transform_response(&resp_data.body),
                        error: None,
                    }
                } else {
                    continue;
                };
                results.push(result);
            }
        }

        Ok(results)
    }

    async fn cancel_batch(&self, batch_id: &str) -> Result<(), RouterError> {
        let resp = self.http
            .post(format!("{}/batches/{}/cancel", self.base_url, batch_id))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .send()
            .await
            .map_err(|e| err_provider_unavailable(Provider::OpenAI, format!("request failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(self.handle_error_response_pub(resp).await);
        }

        Ok(())
    }

    async fn list_batches(&self, opts: Option<ListBatchOptions>) -> Result<Vec<BatchJob>, RouterError> {
        let mut url = format!("{}/batches", self.base_url);
        if let Some(ref opts) = opts {
            let mut params = Vec::new();
            if let Some(limit) = opts.limit {
                params.push(format!("limit={}", limit));
            }
            if let Some(ref after) = opts.after {
                params.push(format!("after={}", after));
            }
            if !params.is_empty() {
                url.push('?');
                url.push_str(&params.join("&"));
            }
        }

        let resp = self.http.get(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .send()
            .await
            .map_err(|e| err_provider_unavailable(Provider::OpenAI, format!("request failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(self.handle_error_response_pub(resp).await);
        }

        let list: BatchList = resp.json().await
            .map_err(|e| err_server_error(Provider::OpenAI, format!("decode response: {}", e)))?;

        Ok(list.data.iter().map(|b| self.convert_batch_job(b)).collect())
    }
}

impl Client {
    async fn upload_batch_file(&self, content: Vec<u8>) -> Result<String, RouterError> {
        let file_part = reqwest::multipart::Part::bytes(content)
            .file_name("batch_input.jsonl")
            .mime_str("application/jsonl")
            .map_err(|e| err_invalid_request(format!("invalid MIME type: {}", e)))?;

        let form = reqwest::multipart::Form::new()
            .text("purpose", "batch")
            .part("file", file_part);

        let resp = self.http
            .post(format!("{}/files", self.base_url))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .multipart(form)
            .send()
            .await
            .map_err(|e| err_provider_unavailable(Provider::OpenAI, format!("upload failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(self.handle_error_response_pub(resp).await);
        }

        let file_resp: FileUploadResponse = resp.json().await
            .map_err(|e| err_server_error(Provider::OpenAI, format!("decode upload response: {}", e)))?;

        Ok(file_resp.id)
    }

    pub async fn handle_error_response_pub(&self, resp: reqwest::Response) -> RouterError {
        self.handle_error_response(resp).await
    }

    fn convert_batch_job(&self, batch: &BatchObject) -> BatchJob {
        let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
        metadata.insert("input_file_id".to_string(), serde_json::json!(batch.input_file_id));
        if let Some(ref oid) = batch.output_file_id {
            metadata.insert("output_file_id".to_string(), serde_json::json!(oid));
        }
        if let Some(ref eid) = batch.error_file_id {
            metadata.insert("error_file_id".to_string(), serde_json::json!(eid));
        }
        metadata.insert("endpoint".to_string(), serde_json::json!(batch.endpoint));

        BatchJob {
            id: batch.id.clone(),
            provider: Provider::OpenAI,
            status: self.convert_batch_status(&batch.status),
            created_at: batch.created_at,
            completed_at: batch.completed_at,
            expires_at: batch.expires_at,
            request_counts: batch.request_counts.as_ref().map(|rc| RequestCounts {
                total: rc.total,
                completed: rc.completed,
                failed: rc.failed,
            }).unwrap_or_default(),
            metadata,
        }
    }

    fn convert_batch_status(&self, status: &str) -> BatchStatus {
        match status {
            "validating" => BatchStatus::Validating,
            "in_progress" => BatchStatus::InProgress,
            "finalizing" => BatchStatus::Finalizing,
            "completed" => BatchStatus::Completed,
            "failed" => BatchStatus::Failed,
            "expired" => BatchStatus::Expired,
            "cancelling" => BatchStatus::InProgress,
            "cancelled" => BatchStatus::Cancelled,
            _ => BatchStatus::Pending,
        }
    }
}
