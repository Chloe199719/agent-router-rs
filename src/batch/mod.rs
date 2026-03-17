//! Unified batch processing interface across providers.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::time::{interval, Duration};

use crate::errors::*;
use crate::provider::{BatchProviderClient, BatchRequest, BatchJob, BatchResult, ListBatchOptions};
use crate::types::Provider;

/// Re-export `BatchStatus` as `batch::Status` — there is a single canonical enum.
pub use crate::provider::BatchStatus as Status;

/// A batch processing request.
#[derive(Debug, Clone)]
pub struct Request {
    /// Developer-provided ID for matching results.
    pub custom_id: String,
    /// The completion request to process.
    pub request: crate::types::CompletionRequest,
}

/// A batch processing job.
#[derive(Debug, Clone)]
pub struct Job {
    pub id: String,
    pub provider: Provider,
    pub status: Status,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    pub counts: Counts,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Tracks batch request progress.
#[derive(Debug, Clone, Default)]
pub struct Counts {
    pub total: i32,
    pub completed: i32,
    pub failed: i32,
}

/// A single result from a batch job.
#[derive(Debug)]
pub struct Result {
    pub custom_id: String,
    pub response: Option<crate::types::CompletionResponse>,
    pub error: Option<RouterError>,
}

/// Options for listing batches.
#[derive(Debug, Clone, Default)]
pub struct ListOptions {
    pub limit: Option<i32>,
    pub after: Option<String>,
}

/// Provides a unified interface for batch processing across providers.
pub struct Manager {
    providers: HashMap<Provider, Arc<dyn BatchProviderClient>>,
}

impl Manager {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    /// Register a batch-capable provider.
    pub fn register_provider(&mut self, provider: Arc<dyn BatchProviderClient>) {
        let name = provider.name();
        self.providers.insert(name, provider);
    }

    /// Create a new batch job.
    pub async fn create(&self, provider_name: Provider, requests: Vec<Request>) -> std::result::Result<Job, RouterError> {
        let p = self.providers.get(&provider_name)
            .ok_or_else(|| err_provider_unavailable(provider_name.clone(), "provider not registered or does not support batch"))?;

        let batch_reqs: Vec<BatchRequest> = requests.into_iter().map(|r| BatchRequest {
            custom_id: r.custom_id,
            request: r.request,
        }).collect();

        let job = p.create_batch(batch_reqs).await?;
        Ok(convert_job(job))
    }

    /// Retrieve the status of a batch job.
    pub async fn get(&self, provider_name: Provider, batch_id: &str) -> std::result::Result<Job, RouterError> {
        let p = self.providers.get(&provider_name)
            .ok_or_else(|| err_provider_unavailable(provider_name.clone(), "provider not registered or does not support batch"))?;

        let job = p.get_batch(batch_id).await?;
        Ok(convert_job(job))
    }

    /// Retrieve the results of a completed batch job.
    pub async fn get_results(&self, provider_name: Provider, batch_id: &str) -> std::result::Result<Vec<Result>, RouterError> {
        let p = self.providers.get(&provider_name)
            .ok_or_else(|| err_provider_unavailable(provider_name.clone(), "provider not registered or does not support batch"))?;

        let results = p.get_batch_results(batch_id).await?;
        Ok(convert_results(results))
    }

    /// Cancel a batch job.
    pub async fn cancel(&self, provider_name: Provider, batch_id: &str) -> std::result::Result<(), RouterError> {
        let p = self.providers.get(&provider_name)
            .ok_or_else(|| err_provider_unavailable(provider_name.clone(), "provider not registered or does not support batch"))?;

        p.cancel_batch(batch_id).await
    }

    /// List batch jobs for a provider.
    pub async fn list(&self, provider_name: Provider, opts: Option<ListOptions>) -> std::result::Result<Vec<Job>, RouterError> {
        let p = self.providers.get(&provider_name)
            .ok_or_else(|| err_provider_unavailable(provider_name.clone(), "provider not registered or does not support batch"))?;

        let list_opts = opts.map(|o| ListBatchOptions {
            limit: o.limit,
            after: o.after,
        });

        let jobs = p.list_batches(list_opts).await?;
        Ok(jobs.into_iter().map(convert_job).collect())
    }

    /// Wait for a batch to complete, polling at the given interval.
    pub async fn wait(&self, provider_name: Provider, batch_id: &str, poll_interval: Duration) -> std::result::Result<Job, RouterError> {
        let mut ticker = interval(poll_interval);
        loop {
            ticker.tick().await;
            let job = self.get(provider_name.clone(), batch_id).await?;
            if job.status.is_done() {
                return Ok(job);
            }
        }
    }
}

impl Default for Manager {
    fn default() -> Self {
        Self::new()
    }
}

fn convert_job(j: BatchJob) -> Job {
    use chrono::{TimeZone, Utc};

    Job {
        id: j.id,
        provider: j.provider,
        // BatchStatus and batch::Status are the same type via re-export — no conversion needed.
        status: j.status,
        created_at: if j.created_at > 0 {
            Utc.timestamp_opt(j.created_at, 0).single().unwrap_or_else(Utc::now)
        } else {
            Utc::now()
        },
        completed_at: if j.completed_at > 0 {
            Utc.timestamp_opt(j.completed_at, 0).single()
        } else {
            None
        },
        expires_at: if j.expires_at > 0 {
            Utc.timestamp_opt(j.expires_at, 0).single()
        } else {
            None
        },
        counts: Counts {
            total: j.request_counts.total,
            completed: j.request_counts.completed,
            failed: j.request_counts.failed,
        },
        metadata: j.metadata,
    }
}

fn convert_results(results: Vec<BatchResult>) -> Vec<Result> {
    results.into_iter().map(|r| Result {
        custom_id: r.custom_id,
        response: r.response,
        error: r.error,
    }).collect()
}
