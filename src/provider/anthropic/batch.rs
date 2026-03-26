//! Anthropic batch processing implementation.

use async_trait::async_trait;
use std::collections::HashMap;
use chrono::DateTime;

use crate::errors::*;
use crate::provider::{BatchProviderClient, BatchRequest, BatchJob, BatchResult, BatchStatus, ListBatchOptions, RequestCounts};
use crate::types::Provider;
use super::client::Client;
use super::types::*;

#[async_trait]
impl BatchProviderClient for Client {
    async fn create_batch(&self, requests: Vec<BatchRequest>) -> Result<BatchJob, RouterError> {
        let items: Vec<BatchRequestItem> = requests.iter().map(|req| {
            let mut anth_req = self.transformer.transform_request(&req.request);
            anth_req.stream = Some(false);
            BatchRequestItem {
                custom_id: req.custom_id.clone(),
                params: anth_req,
            }
        }).collect();

        let batch_req = super::types::BatchRequest { requests: items };

        let builder = self.http
            .post(format!("{}/v1/messages/batches", self.base_url))
            .json(&batch_req);
        let builder = self.set_headers_batch(builder);

        let resp = builder.send().await.map_err(|e| {
            err_provider_unavailable(Provider::Anthropic, format!("request failed: {}", e))
        })?;

        if !resp.status().is_success() {
            return Err(self.handle_error_response(resp).await);
        }

        let batch: BatchResponse = resp.json().await.map_err(|e| {
            err_server_error(Provider::Anthropic, format!("decode response: {}", e))
        })?;

        Ok(self.convert_batch_job(&batch))
    }

    async fn get_batch(&self, batch_id: &str) -> Result<BatchJob, RouterError> {
        let builder = self.http
            .get(format!("{}/v1/messages/batches/{}", self.base_url, batch_id));
        let builder = self.set_headers_batch(builder);

        let resp = builder.send().await.map_err(|e| {
            err_provider_unavailable(Provider::Anthropic, format!("request failed: {}", e))
        })?;

        if !resp.status().is_success() {
            return Err(self.handle_error_response(resp).await);
        }

        let batch: BatchResponse = resp.json().await.map_err(|e| {
            err_server_error(Provider::Anthropic, format!("decode response: {}", e))
        })?;

        Ok(self.convert_batch_job(&batch))
    }

    async fn get_batch_results(&self, batch_id: &str) -> Result<Vec<BatchResult>, RouterError> {
        let job = self.get_batch(batch_id).await?;

        let results_url = job.metadata.get("results_url")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| err_invalid_request("batch has no results URL").with_provider(Provider::Anthropic))?;

        let builder = self.http.get(&results_url);
        let builder = self.set_headers_batch(builder);

        let resp = builder.send().await.map_err(|e| {
            err_provider_unavailable(Provider::Anthropic, format!("request failed: {}", e))
        })?;

        if !resp.status().is_success() {
            return Err(self.handle_error_response(resp).await);
        }

        let content = resp.text().await.map_err(|e| {
            err_server_error(Provider::Anthropic, format!("read response: {}", e))
        })?;

        let mut results = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() { continue; }
            if let Ok(item) = serde_json::from_str::<BatchResultItem>(line) {
                let result = if item.result.result_type == "succeeded" {
                    if let Some(msg) = item.result.message {
                        BatchResult {
                            custom_id: item.custom_id,
                            request_labels: None,
                            response: Some(self.transformer.transform_response(&msg)),
                            error: None,
                        }
                    } else { continue; }
                } else if let Some(err) = item.result.error {
                    BatchResult {
                        custom_id: item.custom_id,
                        request_labels: None,
                        response: None,
                        error: Some(err_server_error(Provider::Anthropic, err.message)),
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
        let builder = self.http
            .post(format!("{}/v1/messages/batches/{}/cancel", self.base_url, batch_id));
        let builder = self.set_headers_batch(builder);

        let resp = builder.send().await.map_err(|e| {
            err_provider_unavailable(Provider::Anthropic, format!("request failed: {}", e))
        })?;

        if !resp.status().is_success() {
            return Err(self.handle_error_response(resp).await);
        }

        Ok(())
    }

    async fn list_batches(&self, opts: Option<ListBatchOptions>) -> Result<Vec<BatchJob>, RouterError> {
        let mut url = format!("{}/v1/messages/batches", self.base_url);
        if let Some(ref opts) = opts {
            let mut params = Vec::new();
            if let Some(limit) = opts.limit {
                params.push(format!("limit={}", limit));
            }
            if let Some(ref after) = opts.after {
                params.push(format!("after_id={}", after));
            }
            if !params.is_empty() {
                url.push('?');
                url.push_str(&params.join("&"));
            }
        }

        let builder = self.http.get(&url);
        let builder = self.set_headers_batch(builder);

        let resp = builder.send().await.map_err(|e| {
            err_provider_unavailable(Provider::Anthropic, format!("request failed: {}", e))
        })?;

        if !resp.status().is_success() {
            return Err(self.handle_error_response(resp).await);
        }

        let list: BatchListResponse = resp.json().await.map_err(|e| {
            err_server_error(Provider::Anthropic, format!("decode response: {}", e))
        })?;

        Ok(list.data.iter().map(|b| self.convert_batch_job(b)).collect())
    }
}

impl Client {
    fn set_headers_batch(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        use super::client::{DEFAULT_VERSION, BETA_HEADER};
        builder
            .header("Content-Type", "application/json")
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", DEFAULT_VERSION)
            .header("anthropic-beta", BETA_HEADER)
    }

    fn convert_batch_job(&self, batch: &BatchResponse) -> BatchJob {
        let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
        if let Some(ref url) = batch.results_url {
            metadata.insert("results_url".to_string(), serde_json::json!(url));
        }

        let created_at = DateTime::parse_from_rfc3339(&batch.created_at)
            .map(|d| d.timestamp()).unwrap_or(0);
        let ended_at = batch.ended_at.as_ref()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.timestamp()).unwrap_or(0);
        let expires_at = batch.expires_at.as_ref()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.timestamp()).unwrap_or(0);

        let total = batch.request_counts.processing + batch.request_counts.succeeded
            + batch.request_counts.errored + batch.request_counts.canceled + batch.request_counts.expired;

        BatchJob {
            id: batch.id.clone(),
            provider: Provider::Anthropic,
            status: self.convert_batch_status(&batch.processing_status),
            created_at,
            completed_at: ended_at,
            expires_at,
            request_counts: RequestCounts {
                total,
                completed: batch.request_counts.succeeded,
                failed: batch.request_counts.errored + batch.request_counts.expired,
            },
            metadata,
        }
    }

    fn convert_batch_status(&self, status: &str) -> BatchStatus {
        match status {
            "in_progress" | "canceling" => BatchStatus::InProgress,
            "ended" => BatchStatus::Completed,
            _ => BatchStatus::Pending,
        }
    }
}


