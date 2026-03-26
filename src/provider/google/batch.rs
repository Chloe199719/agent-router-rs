//! Google Gemini batch processing implementation.

use async_trait::async_trait;
use chrono::DateTime;

use crate::errors::*;
use crate::provider::{BatchProviderClient, BatchRequest, BatchJob as ProviderBatchJob, BatchResult, BatchStatus, ListBatchOptions, RequestCounts};
use crate::types::Provider;
use super::client::Client;
use super::types::{
    BatchConfig, BatchGenerateContentRequest, BatchJob, BatchListResponse,
    InputConfig, InlinedResponse, RequestMetadata, RequestsInput, BatchRequestItem,
};

#[async_trait]
impl BatchProviderClient for Client {
    async fn create_batch(&self, requests: Vec<BatchRequest>) -> Result<ProviderBatchJob, RouterError> {
        if requests.is_empty() {
            return Err(err_invalid_request("no requests provided").with_provider(Provider::Google));
        }

        let model = requests[0].request.model.clone();
        let model = if model.is_empty() { "gemini-2.0-flash".to_string() } else { model };

        let batch_items: Vec<BatchRequestItem> = requests.iter().map(|req| {
            let g_req = self.transformer.transform_request(&req.request);
            BatchRequestItem {
                request: g_req,
                metadata: Some(RequestMetadata { key: req.custom_id.clone() }),
            }
        }).collect();

        let batch_req = BatchGenerateContentRequest {
            batch: BatchConfig {
                display_name: Some(format!("batch-{}", chrono::Utc::now().timestamp())),
                input_config: InputConfig {
                    requests: Some(RequestsInput { requests: batch_items }),
                    file_name: None,
                },
            },
        };

        let url = format!("{}/models/{}:batchGenerateContent?key={}",
            self.base_url, model, self.config.api_key);

        let resp = self.http
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&batch_req)
            .send()
            .await
            .map_err(|e| err_provider_unavailable(Provider::Google, &e.to_string()))?;

        let status = resp.status().as_u16();
        let body = resp.bytes().await
            .map_err(|e| err_server_error(Provider::Google, &e.to_string()))?;

        if status != 200 {
            return Err(self.handle_error_response_sync(status, &body));
        }

        let batch_job: BatchJob = serde_json::from_slice(&body)
            .map_err(|e| err_server_error(Provider::Google, &format!("failed to decode response: {}", e)))?;

        Ok(self.convert_batch_job(&batch_job, &model))
    }

    async fn get_batch(&self, batch_id: &str) -> Result<ProviderBatchJob, RouterError> {
        let batch_name = if batch_id.starts_with("batches/") {
            batch_id.to_string()
        } else {
            format!("batches/{}", batch_id)
        };

        let url = format!("{}/{}?key={}", self.base_url, batch_name, self.config.api_key);

        let resp = self.http
            .get(&url)
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| err_provider_unavailable(Provider::Google, &e.to_string()))?;

        let status = resp.status().as_u16();
        let body = resp.bytes().await
            .map_err(|e| err_server_error(Provider::Google, &e.to_string()))?;

        if status != 200 {
            return Err(self.handle_error_response_sync(status, &body));
        }

        let batch_job: BatchJob = serde_json::from_slice(&body)
            .map_err(|e| err_server_error(Provider::Google, &format!("failed to decode response: {}", e)))?;

        Ok(self.convert_batch_job(&batch_job, ""))
    }

    async fn get_batch_results(&self, batch_id: &str) -> Result<Vec<BatchResult>, RouterError> {
        let job = self.get_batch(batch_id).await?;

        if job.status != BatchStatus::Completed {
            return Err(err_invalid_request(&format!("batch job is not complete, status: {}", job.status))
                .with_provider(Provider::Google));
        }

        // Re-fetch to get full response
        let batch_name = if batch_id.starts_with("batches/") {
            batch_id.to_string()
        } else {
            format!("batches/{}", batch_id)
        };

        let url = format!("{}/{}?key={}", self.base_url, batch_name, self.config.api_key);

        let resp = self.http
            .get(&url)
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| err_provider_unavailable(Provider::Google, &e.to_string()))?;

        let status = resp.status().as_u16();
        let body = resp.bytes().await
            .map_err(|e| err_server_error(Provider::Google, &e.to_string()))?;

        if status != 200 {
            return Err(self.handle_error_response_sync(status, &body));
        }

        let batch_job: BatchJob = serde_json::from_slice(&body)
            .map_err(|e| err_server_error(Provider::Google, &format!("failed to decode response: {}", e)))?;

        // Check for inline responses
        if let Some(response) = &batch_job.response {
            if let Some(wrapper) = &response.inlined_responses {
                if !wrapper.inlined_responses.is_empty() {
                    return Ok(self.convert_inlined_responses(&wrapper.inlined_responses));
                }
            }
            // Check for file-based responses
            if let Some(responses_file) = &response.responses_file {
                return self.download_batch_results(responses_file).await;
            }
        }

        Err(err_server_error(Provider::Google, "no results found in batch response"))
    }

    async fn cancel_batch(&self, batch_id: &str) -> Result<(), RouterError> {
        let batch_name = if batch_id.starts_with("batches/") {
            batch_id.to_string()
        } else {
            format!("batches/{}", batch_id)
        };

        let url = format!("{}/{}:cancel?key={}", self.base_url, batch_name, self.config.api_key);

        let resp = self.http
            .post(&url)
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| err_provider_unavailable(Provider::Google, &e.to_string()))?;

        let status = resp.status().as_u16();
        if status != 200 {
            let body = resp.bytes().await
                .map_err(|e| err_server_error(Provider::Google, &e.to_string()))?;
            return Err(self.handle_error_response_sync(status, &body));
        }

        Ok(())
    }

    async fn list_batches(&self, opts: Option<ListBatchOptions>) -> Result<Vec<ProviderBatchJob>, RouterError> {
        let mut url = format!("{}/batches?key={}", self.base_url, self.config.api_key);

        if let Some(opts) = &opts {
            if let Some(limit) = opts.limit {
                url.push_str(&format!("&pageSize={}", limit));
            }
            if let Some(after) = &opts.after {
                url.push_str(&format!("&pageToken={}", after));
            }
        }

        let resp = self.http
            .get(&url)
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| err_provider_unavailable(Provider::Google, &e.to_string()))?;

        let status = resp.status().as_u16();
        let body = resp.bytes().await
            .map_err(|e| err_server_error(Provider::Google, &e.to_string()))?;

        if status != 200 {
            return Err(self.handle_error_response_sync(status, &body));
        }

        let list_resp: BatchListResponse = serde_json::from_slice(&body)
            .map_err(|e| err_server_error(Provider::Google, &format!("failed to decode response: {}", e)))?;

        let jobs = list_resp.batches.iter()
            .map(|b| self.convert_batch_job(b, ""))
            .collect();

        Ok(jobs)
    }
}

impl Client {
    async fn download_batch_results(&self, file_name: &str) -> Result<Vec<BatchResult>, RouterError> {
        let url = format!(
            "https://generativelanguage.googleapis.com/download/v1beta/{}:download?alt=media&key={}",
            file_name, self.config.api_key
        );

        let resp = self.http
            .get(&url)
            .send()
            .await
            .map_err(|e| err_provider_unavailable(Provider::Google, &e.to_string()))?;

        let status = resp.status().as_u16();
        let body = resp.bytes().await
            .map_err(|e| err_server_error(Provider::Google, &e.to_string()))?;

        if status != 200 {
            return Err(self.handle_error_response_sync(status, &body));
        }

        let mut results = Vec::new();
        let mut deserializer = serde_json::Deserializer::from_slice(&body).into_iter::<InlinedResponse>();

        while let Some(Ok(line)) = deserializer.next() {
            let mut result = BatchResult {
                custom_id: String::new(),
                request_labels: None,
                response: None,
                error: None,
            };

            if let Some(meta) = &line.metadata {
                if let Some(key) = &meta.key {
                    result.custom_id = key.clone();
                }
            }

            if let Some(err) = &line.error {
                result.error = Some(err_server_error(Provider::Google, &err.message));
            } else if let Some(resp) = &line.response {
                result.response = self.transformer.transform_response(resp);
            }

            results.push(result);
        }

        Ok(results)
    }

    fn convert_inlined_responses(&self, responses: &[InlinedResponse]) -> Vec<BatchResult> {
        responses.iter().map(|resp| {
            let mut result = BatchResult {
                custom_id: String::new(),
                request_labels: None,
                response: None,
                error: None,
            };

            if let Some(meta) = &resp.metadata {
                if let Some(key) = &meta.key {
                    result.custom_id = key.clone();
                }
            }

            if let Some(err) = &resp.error {
                result.error = Some(err_server_error(Provider::Google, &err.message));
            } else if let Some(r) = &resp.response {
                result.response = self.transformer.transform_response(r);
            }

            result
        }).collect()
    }

    fn convert_batch_job(&self, batch: &BatchJob, model: &str) -> ProviderBatchJob {
        let mut job = ProviderBatchJob {
            id: batch.name.clone(),
            provider: Provider::Google,
            status: self.convert_batch_status(batch),
            created_at: 0,
            completed_at: 0,
            expires_at: 0,
            request_counts: RequestCounts::default(),
            metadata: std::collections::HashMap::new(),
        };

        if let Some(meta) = &batch.metadata {
            if let Some(dn) = &meta.display_name {
                job.metadata.insert("display_name".to_string(), serde_json::Value::String(dn.clone()));
            }
            job.metadata.insert("state".to_string(), serde_json::Value::String(meta.state.clone()));

            if let Some(ct) = &meta.create_time {
                if let Ok(t) = DateTime::parse_from_rfc3339(ct) {
                    job.created_at = t.timestamp();
                }
            }
        }

        if !model.is_empty() {
            job.metadata.insert("model".to_string(), serde_json::Value::String(model.to_string()));
        }

        if let Some(response) = &batch.response {
            if let Some(rf) = &response.responses_file {
                job.metadata.insert("responses_file".to_string(), serde_json::Value::String(rf.clone()));
            }
            if let Some(wrapper) = &response.inlined_responses {
                let count = wrapper.inlined_responses.len() as i32;
                job.request_counts.total = count;
                job.request_counts.completed = count;
            }
        }

        job
    }

    fn convert_batch_status(&self, batch: &BatchJob) -> BatchStatus {
        if batch.done {
            if batch.error.is_some() {
                return BatchStatus::Failed;
            }
            return BatchStatus::Completed;
        }

        if let Some(meta) = &batch.metadata {
            match meta.state.as_str() {
                "JOB_STATE_PENDING" | "BATCH_STATE_PENDING" => return BatchStatus::Pending,
                "JOB_STATE_RUNNING" | "BATCH_STATE_RUNNING" => return BatchStatus::InProgress,
                "JOB_STATE_SUCCEEDED" | "BATCH_STATE_SUCCEEDED" => return BatchStatus::Completed,
                "JOB_STATE_FAILED" | "BATCH_STATE_FAILED" => return BatchStatus::Failed,
                "JOB_STATE_CANCELLED" | "BATCH_STATE_CANCELLED" => return BatchStatus::Cancelled,
                "JOB_STATE_EXPIRED" | "BATCH_STATE_EXPIRED" => return BatchStatus::Expired,
                _ => {}
            }
        }

        BatchStatus::Pending
    }
}
