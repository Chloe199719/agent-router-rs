//! Vertex AI batch prediction implementation.

use async_trait::async_trait;
use chrono::DateTime;

use crate::errors::*;
use crate::provider::{BatchProviderClient, BatchRequest, BatchJob, BatchResult, BatchStatus, ListBatchOptions, RequestCounts};
use crate::types::Provider;
use super::client::Client;
use super::types::*;

#[async_trait]
impl BatchProviderClient for Client {
    async fn create_batch(&self, requests: Vec<BatchRequest>) -> Result<BatchJob, RouterError> {
        if requests.is_empty() {
            return Err(err_invalid_request("no requests provided").with_provider(Provider::Vertex));
        }

        if self.config.batch_bucket.as_deref().map(|b| b.is_empty()).unwrap_or(true) {
            return Err(err_invalid_request(
                "batch bucket is required for Vertex AI batch operations; use with_batch_bucket()"
            ).with_provider(Provider::Vertex));
        }

        let model = {
            let m = requests[0].request.model.clone();
            if m.is_empty() { "gemini-2.0-flash".to_string() } else { m }
        };

        // Build JSONL content from requests, embedding custom_id in labels
        let mut buf = Vec::new();
        for req in &requests {
            let mut g_req = self.transformer.transform_request(&req.request);
            if !req.custom_id.is_empty() {
                g_req.labels.get_or_insert_with(Default::default)
                    .insert("custom_id".to_string(), req.custom_id.clone());
            }
            let line = VertexBatchInputLine {
                request: serde_json::to_value(&g_req)
                    .map_err(|e| err_invalid_request(format!("failed to marshal request: {}", e)))?,
            };
            serde_json::to_writer(&mut buf, &line)
                .map_err(|e| err_invalid_request(format!("failed to marshal batch line: {}", e)))?;
            buf.push(b'\n');
        }

        // Upload JSONL to GCS
        let batch_id = format!("batch-{}", chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default());
        let batch_bucket = self.config.batch_bucket.as_deref().unwrap_or_default();
        let (bucket, prefix) = parse_bucket_path(batch_bucket);
        let input_path = format!("{}{}/input.jsonl", prefix, batch_id);
        let input_uri = format!("gs://{}/{}", bucket, input_path);
        let output_uri_prefix = format!("gs://{}/{}{}/output/", bucket, prefix, batch_id);

        self.upload_to_gcs(&bucket, &input_path, &buf).await
            .map_err(|e| err_server_error(Provider::Vertex, format!("failed to upload batch input to GCS: {}", e)))?;

        // Create batch prediction job
        let model_path = format!("publishers/google/models/{}", model);
        let job_req = VertexBatchPredictionJobRequest {
            display_name: batch_id.clone(),
            model: model_path,
            input_config: VertexBatchInputConfig {
                instances_format: "jsonl".to_string(),
                gcs_source: GcsSource { uris: vec![input_uri] },
            },
            output_config: VertexBatchOutputConfig {
                predictions_format: "jsonl".to_string(),
                gcs_destination: GcsDestination { output_uri_prefix },
            },
        };

        let url = self.batch_jobs_url();
        let builder = self.http
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&job_req);
        let builder = self.set_auth_header(builder);

        let resp = builder.send().await
            .map_err(|e| err_provider_unavailable(Provider::Vertex, &e.to_string()))?;

        let status = resp.status().as_u16();
        let body = resp.bytes().await
            .map_err(|e| err_server_error(Provider::Vertex, &e.to_string()))?;

        if status != 200 {
            return Err(self.handle_error_response_sync(status, &body));
        }

        let job: VertexBatchPredictionJob = serde_json::from_slice(&body)
            .map_err(|e| err_server_error(Provider::Vertex, format!("failed to decode response: {}", e)))?;

        Ok(self.convert_vertex_batch_job(&job, &model))
    }

    async fn get_batch(&self, batch_id: &str) -> Result<BatchJob, RouterError> {
        let batch_name = if batch_id.starts_with("projects/") {
            batch_id.to_string()
        } else {
            format!("projects/{}/locations/{}/batchPredictionJobs/{}", self.project_id, self.location, batch_id)
        };

        let mut url = format!("{}/{}", self.base_url, batch_name);
        if self.config.access_token.as_deref().map(|t| t.is_empty()).unwrap_or(true) && !self.config.api_key.is_empty() {
            url.push_str(&format!("?key={}", self.config.api_key));
        }

        let builder = self.http.get(&url);
        let builder = self.set_auth_header(builder);

        let resp = builder.send().await
            .map_err(|e| err_provider_unavailable(Provider::Vertex, &e.to_string()))?;

        let status = resp.status().as_u16();
        let body = resp.bytes().await
            .map_err(|e| err_server_error(Provider::Vertex, &e.to_string()))?;

        if status != 200 {
            return Err(self.handle_error_response_sync(status, &body));
        }

        let job: VertexBatchPredictionJob = serde_json::from_slice(&body)
            .map_err(|e| err_server_error(Provider::Vertex, format!("failed to decode response: {}", e)))?;

        Ok(self.convert_vertex_batch_job(&job, ""))
    }

    async fn get_batch_results(&self, batch_id: &str) -> Result<Vec<BatchResult>, RouterError> {
        let job = self.get_batch(batch_id).await?;

        if job.status != BatchStatus::Completed {
            return Err(err_invalid_request(format!("batch job is not complete, status: {}", job.status))
                .with_provider(Provider::Vertex));
        }

        let output_dir = job.metadata.get("gcs_output_directory")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if output_dir.is_empty() {
            return Err(err_server_error(Provider::Vertex, "no output directory found in batch job"));
        }

        self.download_batch_results(&output_dir).await
    }

    async fn cancel_batch(&self, batch_id: &str) -> Result<(), RouterError> {
        let batch_name = if batch_id.starts_with("projects/") {
            batch_id.to_string()
        } else {
            format!("projects/{}/locations/{}/batchPredictionJobs/{}", self.project_id, self.location, batch_id)
        };

        let mut url = format!("{}/{}:cancel", self.base_url, batch_name);
        if self.config.access_token.as_deref().map(|t| t.is_empty()).unwrap_or(true) && !self.config.api_key.is_empty() {
            url.push_str(&format!("?key={}", self.config.api_key));
        }

        // POST with an empty JSON body so that Content-Length: 0 is sent.
        // Without a body reqwest omits Content-Length, which causes a 411 from the API.
        let builder = self.http.post(&url)
            .header("Content-Type", "application/json")
            .body("{}");
        let builder = self.set_auth_header(builder);

        let resp = builder.send().await
            .map_err(|e| err_provider_unavailable(Provider::Vertex, &e.to_string()))?;

        let status = resp.status().as_u16();
        if status != 200 {
            let body = resp.bytes().await
                .map_err(|e| err_server_error(Provider::Vertex, &e.to_string()))?;
            return Err(self.handle_error_response_sync(status, &body));
        }

        Ok(())
    }

    async fn list_batches(&self, opts: Option<ListBatchOptions>) -> Result<Vec<BatchJob>, RouterError> {
        let base_url = format!("{}/projects/{}/locations/{}/batchPredictionJobs",
            self.base_url, self.project_id, self.location);

        // Collect query parameters and let reqwest encode + join them.
        let mut query: Vec<(&str, String)> = Vec::new();
        if self.config.access_token.as_deref().map(|t| t.is_empty()).unwrap_or(true) && !self.config.api_key.is_empty() {
            query.push(("key", self.config.api_key.clone()));
        }
        if let Some(ref opts) = opts {
            if let Some(limit) = opts.limit {
                query.push(("pageSize", limit.to_string()));
            }
            if let Some(ref after) = opts.after {
                query.push(("pageToken", after.clone()));
            }
        }

        let builder = self.http.get(&base_url).query(&query);
        let builder = self.set_auth_header(builder);

        let resp = builder.send().await
            .map_err(|e| err_provider_unavailable(Provider::Vertex, &e.to_string()))?;

        let status = resp.status().as_u16();
        let body = resp.bytes().await
            .map_err(|e| err_server_error(Provider::Vertex, &e.to_string()))?;

        if status != 200 {
            return Err(self.handle_error_response_sync(status, &body));
        }

        let list_resp: VertexBatchPredictionJobList = serde_json::from_slice(&body)
            .map_err(|e| err_server_error(Provider::Vertex, format!("failed to decode response: {}", e)))?;

        let jobs = list_resp.batch_prediction_jobs.iter()
            .map(|j| self.convert_vertex_batch_job(j, ""))
            .collect();

        Ok(jobs)
    }
}

impl Client {
    fn batch_jobs_url(&self) -> String {
        let base = format!("{}/projects/{}/locations/{}/batchPredictionJobs",
            self.base_url, self.project_id, self.location);
        // Append the API key as a query parameter when not using OAuth2.
        if self.config.access_token.as_deref().map(|t| t.is_empty()).unwrap_or(true) && !self.config.api_key.is_empty() {
            format!("{}?key={}", base, self.config.api_key)
        } else {
            base
        }
    }

    async fn upload_to_gcs(&self, bucket: &str, object_path: &str, data: &[u8]) -> Result<(), String> {
        let url = format!(
            "https://storage.googleapis.com/upload/storage/v1/b/{}/o?uploadType=media&name={}",
            bucket, object_path
        );

        let builder = self.http
            .post(&url)
            .header("Content-Type", "application/jsonl")
            .body(data.to_vec());
        let builder = if let Some(token) = &self.config.access_token {
            if !token.is_empty() {
                builder.header("Authorization", format!("Bearer {}", token))
            } else {
                builder
            }
        } else {
            builder
        };

        let resp = builder.send().await.map_err(|e| format!("upload request: {}", e))?;
        if resp.status() != 200 {
            let status = resp.status().as_u16();
            let body = resp.bytes().await.unwrap_or_default();
            return Err(format!("GCS upload failed with status {}: {}", status, String::from_utf8_lossy(&body)));
        }
        Ok(())
    }

    async fn download_from_gcs(&self, bucket: &str, object_path: &str) -> Result<Vec<u8>, String> {
        let encoded_path = urlencoding_simple(object_path);
        let url = format!(
            "https://storage.googleapis.com/storage/v1/b/{}/o/{}?alt=media",
            bucket, encoded_path
        );

        let builder = self.http.get(&url);
        let builder = if let Some(token) = &self.config.access_token {
            if !token.is_empty() {
                builder.header("Authorization", format!("Bearer {}", token))
            } else {
                builder
            }
        } else {
            builder
        };

        let resp = builder.send().await.map_err(|e| format!("download request: {}", e))?;
        if resp.status() != 200 {
            let status = resp.status().as_u16();
            let body = resp.bytes().await.unwrap_or_default();
            return Err(format!("GCS download failed with status {}: {}", status, String::from_utf8_lossy(&body)));
        }
        let body = resp.bytes().await.map_err(|e| format!("read response: {}", e))?;
        Ok(body.to_vec())
    }

    async fn find_batch_output_file(&self, bucket: &str, prefix: &str) -> Result<String, String> {
        let list_url = format!(
            "https://storage.googleapis.com/storage/v1/b/{}/o?prefix={}",
            bucket, urlencoding_simple(prefix)
        );

        let builder = self.http.get(&list_url);
        let builder = if let Some(token) = &self.config.access_token {
            if !token.is_empty() {
                builder.header("Authorization", format!("Bearer {}", token))
            } else {
                builder
            }
        } else {
            builder
        };

        let resp = builder.send().await.map_err(|e| format!("create list request: {}", e))?;
        if resp.status() != 200 {
            let status = resp.status().as_u16();
            let body = resp.bytes().await.unwrap_or_default();
            return Err(format!("GCS list failed with status {}: {}", status, String::from_utf8_lossy(&body)));
        }

        let body = resp.bytes().await.map_err(|e| format!("decode list response: {}", e))?;

        #[derive(serde::Deserialize)]
        struct ListResponse {
            #[serde(default)]
            items: Vec<ListItem>,
        }
        #[derive(serde::Deserialize)]
        struct ListItem {
            name: String,
        }

        let list_resp: ListResponse = serde_json::from_slice(&body)
            .map_err(|e| format!("decode list response: {}", e))?;

        // Look for a prediction output file
        for item in &list_resp.items {
            let base = item.name.rsplit('/').next().unwrap_or(&item.name);
            if base.starts_with("prediction") {
                return Ok(item.name.clone());
            }
        }

        // Fall back to first file
        if let Some(first) = list_resp.items.first() {
            return Ok(first.name.clone());
        }

        Err(format!("no output files found in GCS directory: gs://{}/{}", bucket, prefix))
    }

    async fn download_batch_results(&self, gcs_output_dir: &str) -> Result<Vec<BatchResult>, RouterError> {
        let dir = gcs_output_dir.trim_end_matches('/').to_string() + "/";
        let (bucket, prefix) = parse_gcs_uri(&dir);
        if bucket.is_empty() {
            return Err(err_server_error(Provider::Vertex, format!("invalid GCS output URI: {}", gcs_output_dir)));
        }

        let object_path = self.find_batch_output_file(&bucket, &prefix).await
            .map_err(|e| err_server_error(Provider::Vertex, format!("failed to find batch output file in GCS: {}", e)))?;

        let content = self.download_from_gcs(&bucket, &object_path).await
            .map_err(|e| err_server_error(Provider::Vertex, format!("failed to download batch results from GCS: {}", e)))?;

        let mut results = Vec::new();
        let mut deserializer = serde_json::Deserializer::from_slice(&content).into_iter::<VertexBatchOutputLine>();

        while let Some(Ok(line)) = deserializer.next() {
            let mut result = BatchResult {
                custom_id: String::new(),
                response: None,
                error: None,
            };

            if let Some(req) = &line.request {
                if let Some(custom_id) = req.labels.get("custom_id") {
                    result.custom_id = custom_id.clone();
                }
            }

            if let Some(resp) = &line.response {
                if let Some(mut r) = self.transformer.transform_response(resp) {
                    r.provider = Provider::Vertex;
                    result.response = Some(r);
                }
            }

            if !line.status.is_empty() {
                result.error = Some(err_server_error(Provider::Vertex, &line.status));
            }

            results.push(result);
        }

        Ok(results)
    }

    fn convert_vertex_batch_job(&self, job: &VertexBatchPredictionJob, model: &str) -> BatchJob {
        let mut result = BatchJob {
            id: job.name.clone(),
            provider: Provider::Vertex,
            status: self.convert_vertex_job_state(&job.state),
            created_at: 0,
            completed_at: 0,
            expires_at: 0,
            request_counts: RequestCounts::default(),
            metadata: std::collections::HashMap::new(),
        };

        if !job.display_name.is_empty() {
            result.metadata.insert("display_name".to_string(), serde_json::Value::String(job.display_name.clone()));
        }
        if !job.state.is_empty() {
            result.metadata.insert("state".to_string(), serde_json::Value::String(job.state.clone()));
        }
        if !model.is_empty() {
            result.metadata.insert("model".to_string(), serde_json::Value::String(model.to_string()));
        } else if !job.model.is_empty() {
            result.metadata.insert("model".to_string(), serde_json::Value::String(job.model.clone()));
        }

        if let Some(output_info) = &job.output_info {
            if !output_info.gcs_output_directory.is_empty() {
                result.metadata.insert(
                    "gcs_output_directory".to_string(),
                    serde_json::Value::String(output_info.gcs_output_directory.clone()),
                );
            }
        }

        if let Some(ct) = &job.create_time {
            if let Ok(t) = DateTime::parse_from_rfc3339(ct) {
                result.created_at = t.timestamp();
            }
        }
        if let Some(et) = &job.end_time {
            if let Ok(t) = DateTime::parse_from_rfc3339(et) {
                result.completed_at = t.timestamp();
            }
        }

        if let Some(err) = &job.error {
            result.metadata.insert("error_code".to_string(), serde_json::Value::Number(err.code.into()));
            result.metadata.insert("error_message".to_string(), serde_json::Value::String(err.message.clone()));
        }

        result
    }

    fn convert_vertex_job_state(&self, state: &str) -> BatchStatus {
        match state {
            "JOB_STATE_PENDING" | "JOB_STATE_QUEUED" => BatchStatus::Pending,
            "JOB_STATE_RUNNING" | "JOB_STATE_UPDATING" => BatchStatus::InProgress,
            "JOB_STATE_SUCCEEDED" => BatchStatus::Completed,
            "JOB_STATE_FAILED" | "JOB_STATE_PARTIALLY_SUCCEEDED" => BatchStatus::Failed,
            "JOB_STATE_CANCELLED" | "JOB_STATE_CANCELLING" => BatchStatus::Cancelled,
            "JOB_STATE_EXPIRED" => BatchStatus::Expired,
            _ => BatchStatus::Pending,
        }
    }
}

/// Split "my-bucket/path/prefix" or "gs://my-bucket/path/prefix" into (bucket, prefix).
fn parse_bucket_path(bucket_path: &str) -> (String, String) {
    let path = bucket_path.trim_start_matches("gs://");
    let mut parts = path.splitn(2, '/');
    let bucket = parts.next().unwrap_or("").to_string();
    let mut prefix = parts.next().unwrap_or("").to_string();
    if !prefix.is_empty() && !prefix.ends_with('/') {
        prefix.push('/');
    }
    (bucket, prefix)
}

/// Parse "gs://bucket/path" into (bucket, path).
fn parse_gcs_uri(uri: &str) -> (String, String) {
    if !uri.starts_with("gs://") {
        return (String::new(), String::new());
    }
    let path = &uri["gs://".len()..];
    let mut parts = path.splitn(2, '/');
    let bucket = parts.next().unwrap_or("").to_string();
    let object = parts.next().unwrap_or("").to_string();
    (bucket, object)
}

/// Percent-encode a path string, preserving `/` as a delimiter.
///
/// Characters that are unreserved in RFC 3986 (`A-Z a-z 0-9 - _ . ~`) and
/// forward-slash are passed through unchanged; everything else is
/// percent-encoded as UTF-8 bytes.
///
/// Avoids allocating a temporary `String` per character: multi-byte chars are
/// encoded directly from their `encode_utf8` buffer.
fn urlencoding_simple(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut buf = [0u8; 4];
    for c in s.chars() {
        match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' | '/' => result.push(c),
            _ => {
                let encoded = c.encode_utf8(&mut buf);
                for &byte in encoded.as_bytes() {
                    result.push('%');
                    result.push(char::from_digit((byte >> 4) as u32, 16).unwrap().to_ascii_uppercase());
                    result.push(char::from_digit((byte & 0xF) as u32, 16).unwrap().to_ascii_uppercase());
                }
            }
        }
    }
    result
}
