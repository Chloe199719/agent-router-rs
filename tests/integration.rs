//! Integration tests — gated by the `integration` feature flag.
//! Run with: cargo test --features integration
//!
//! Optional: `VERTEX_BATCH_WAIT_RESULTS=1` enables the long-running Vertex batch test
//! [`test_vertex_batch_metadata_get_results_echoed_labels`] that asserts echoed `request_labels`.

#[cfg(feature = "integration")]
mod tests {
    use std::collections::HashMap;

    use agent_router::{with_anthropic, with_google, with_openai, with_vertex, provider, Router};
    use agent_router::types::{CompletionRequest, Message, Provider, Role, StreamEventType};
    use agent_router::batch::{Manager as BatchManager, Request as BatchRequest, Status as BatchStatus};
    use futures::StreamExt;

    fn setup() {
        let _ = dotenvy::dotenv();
    }

    #[tokio::test]
    async fn test_openai_basic_completion() {
        setup();
        let key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
        let router = Router::new(vec![with_openai(&key, vec![])]).unwrap();

        let req = CompletionRequest::new(
            Provider::OpenAI,
            "gpt-4o-mini",
            vec![Message::new_text(Role::User, "Say 'hello' only")],
        )
        .with_max_tokens(10);

        let resp = router.complete(&req).await.expect("openai completion failed");
        assert!(!resp.text().is_empty(), "response should not be empty");
        assert_eq!(resp.provider, Provider::OpenAI);
    }

    #[tokio::test]
    async fn test_openai_completion_with_metadata() {
        setup();
        let key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
        let router = Router::new(vec![with_openai(&key, vec![])]).unwrap();

        let mut meta = HashMap::new();
        meta.insert("trace_id".to_string(), "router-integration".to_string());

        let req = CompletionRequest::new(
            Provider::OpenAI,
            "gpt-4o-mini",
            vec![Message::new_text(Role::User, "Say 'ok' only")],
        )
        .with_max_tokens(10)
        .with_metadata(meta);

        let resp = router.complete(&req).await.expect("openai completion with metadata failed");
        assert!(!resp.text().is_empty());
        assert_eq!(resp.provider, Provider::OpenAI);
    }

    #[tokio::test]
    async fn test_anthropic_basic_completion() {
        setup();
        let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
        let router = Router::new(vec![with_anthropic(&key, vec![])]).unwrap();

        let req = CompletionRequest::new(
            Provider::Anthropic,
            "claude-3-haiku-20240307",
            vec![Message::new_text(Role::User, "Say 'hello' only")],
        )
        .with_max_tokens(10);

        let resp = router.complete(&req).await.expect("anthropic completion failed");
        assert!(!resp.text().is_empty(), "response should not be empty");
        assert_eq!(resp.provider, Provider::Anthropic);
    }

    #[tokio::test]
    async fn test_anthropic_completion_with_metadata_user_id() {
        setup();
        let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
        let router = Router::new(vec![with_anthropic(&key, vec![])]).unwrap();

        let mut meta = HashMap::new();
        meta.insert("user_id".to_string(), "integration-user".to_string());
        meta.insert("other_key".to_string(), "ignored".to_string());

        let req = CompletionRequest::new(
            Provider::Anthropic,
            "claude-3-haiku-20240307",
            vec![Message::new_text(Role::User, "Say 'ok' only")],
        )
        .with_max_tokens(10)
        .with_metadata(meta);

        let resp = router.complete(&req).await.expect("anthropic completion with metadata failed");
        assert!(!resp.text().is_empty());
        assert_eq!(resp.provider, Provider::Anthropic);
    }

    #[tokio::test]
    async fn test_google_basic_completion() {
        setup();
        let key = std::env::var("GOOGLE_API_KEY").expect("GOOGLE_API_KEY not set");
        let router = Router::new(vec![with_google(&key, vec![])]).unwrap();

        let req = CompletionRequest::new(
            Provider::Google,
            "gemini-2.0-flash",
            vec![Message::new_text(Role::User, "Say 'hello' only")],
        )
        .with_max_tokens(10);

        let resp = router.complete(&req).await.expect("google completion failed");
        assert!(!resp.text().is_empty(), "response should not be empty");
        assert_eq!(resp.provider, Provider::Google);
    }

    #[tokio::test]
    async fn test_google_completion_with_metadata_not_forwarded() {
        setup();
        let key = std::env::var("GOOGLE_API_KEY").expect("GOOGLE_API_KEY not set");
        let router = Router::new(vec![with_google(&key, vec![])]).unwrap();

        let mut meta = HashMap::new();
        meta.insert("would_break_if_sent_as_labels".to_string(), "x".to_string());

        let req = CompletionRequest::new(
            Provider::Google,
            "gemini-2.0-flash",
            vec![Message::new_text(Role::User, "Say 'ok' only")],
        )
        .with_max_tokens(10)
        .with_metadata(meta);

        let resp = router.complete(&req).await.expect("google should accept request; metadata ignored");
        assert!(!resp.text().is_empty());
        assert_eq!(resp.provider, Provider::Google);
    }

    #[tokio::test]
    async fn test_openai_streaming() {
        setup();
        let key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
        let router = Router::new(vec![with_openai(&key, vec![])]).unwrap();

        let req = CompletionRequest::new(
            Provider::OpenAI,
            "gpt-4o-mini",
            vec![Message::new_text(Role::User, "Say 'hello' only")],
        )
        .with_max_tokens(10);

        let mut stream = router.stream(&req).await.expect("stream failed");
        let mut got_content = false;

        while let Some(result) = stream.next().await {
            let event = result.expect("stream event error");
            if event.event_type == StreamEventType::ContentDelta {
                got_content = true;
            }
        }

        assert!(got_content, "should have received content delta");
    }

    // ---- Vertex AI ----

    fn vertex_router() -> Router {
        setup();
        let project_id = std::env::var("VERTEX_PROJECT_ID").expect("VERTEX_PROJECT_ID not set");
        let location   = std::env::var("VERTEX_LOCATION").expect("VERTEX_LOCATION not set");
        let token      = std::env::var("VERTEX_ACCESS_TOKEN").expect("VERTEX_ACCESS_TOKEN not set");

        Router::new(vec![
            with_vertex(project_id, location, vec![provider::with_access_token(token)]),
        ])
        .expect("failed to build Vertex router")
    }

    #[tokio::test]
    async fn test_vertex_basic_completion() {
        let router = vertex_router();

        let req = CompletionRequest::new(
            Provider::Vertex,
            "gemini-3.1-pro-preview",
            vec![Message::new_text(Role::User, "Say 'hello' only")],
        )
        .with_max_tokens(10);

        let resp = router.complete(&req).await;
        match resp {
            Ok(resp) => {
                assert!(!resp.text().is_empty(), "response should not be empty");
                assert_eq!(resp.provider, Provider::Vertex);
            }
            Err(e) if e.code == "model_not_found" => {
                println!("SKIP: Vertex model not available on this project: {}", e.message);
            }
            Err(e) => panic!("vertex completion failed: {}", e),
        }
    }

    #[tokio::test]
    async fn test_vertex_completion_with_metadata() {
        let router = vertex_router();

        let mut meta = HashMap::new();
        meta.insert("team".to_string(), "integration".to_string());

        let req = CompletionRequest::new(
            Provider::Vertex,
            "gemini-3.1-pro-preview",
            vec![Message::new_text(Role::User, "Say 'hello' only")],
        )
        .with_max_tokens(10)
        .with_metadata(meta);

        let resp = router.complete(&req).await;
        match resp {
            Ok(resp) => {
                assert!(!resp.text().is_empty());
                assert_eq!(resp.provider, Provider::Vertex);
            }
            Err(e) if e.code == "model_not_found" => {
                println!("SKIP: Vertex model not available on this project: {}", e.message);
            }
            Err(e) => panic!("vertex completion with metadata failed: {}", e),
        }
    }

    #[tokio::test]
    async fn test_vertex_streaming() {
        let router = vertex_router();

        let req = CompletionRequest::new(
            Provider::Vertex,
            "gemini-3.1-pro-preview",
            vec![Message::new_text(Role::User, "Say 'hello' only")],
        )
        .with_max_tokens(10);

        let stream = router.stream(&req).await;
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) if e.code == "model_not_found"
                   || (e.code == "server_error" && e.message.contains("NOT_FOUND")) => {
                println!("SKIP: Vertex model not available on this project: {}", e.message);
                return;
            }
            Err(e) if e.code == "rate_limit"
                   || (e.code == "server_error" && e.message.contains("RESOURCE_EXHAUSTED")) => {
                println!("SKIP: Vertex rate limit hit: {}", e.message);
                return;
            }
            Err(e) => panic!("vertex stream failed: {}", e),
        };

        let mut got_start = false;
        let mut got_content = false;
        let mut got_done = false;

        while let Some(result) = stream.next().await {
            let event = result.expect("vertex stream event error");
            match event.event_type {
                StreamEventType::Start => got_start = true,
                StreamEventType::ContentDelta => got_content = true,
                StreamEventType::Done => got_done = true,
                _ => {}
            }
        }

        assert!(got_start,   "should have received Start event");
        assert!(got_content, "should have received ContentDelta event");
        assert!(got_done,    "should have received Done event");
    }

    fn vertex_batch_manager() -> BatchManager {
        setup();
        use std::sync::Arc;
        let project_id = std::env::var("VERTEX_PROJECT_ID").expect("VERTEX_PROJECT_ID not set");
        let location   = std::env::var("VERTEX_LOCATION").expect("VERTEX_LOCATION not set");
        let token      = std::env::var("VERTEX_ACCESS_TOKEN").expect("VERTEX_ACCESS_TOKEN not set");
        let bucket     = std::env::var("VERTEX_BATCH_BUCKET").expect("VERTEX_BATCH_BUCKET not set");

        let client = Arc::new(
            agent_router::provider::vertex::Client::new(
                project_id,
                location,
                vec![
                    provider::with_access_token(token),
                    provider::with_batch_bucket(bucket),
                ],
            )
            .expect("failed to build vertex client"),
        );

        let mut manager = BatchManager::new();
        manager.register_provider(client);
        manager
    }

    fn vertex_batch_requests(n: usize) -> Vec<BatchRequest> {
        (1..=n).map(|i| BatchRequest {
            custom_id: format!("req-{}", i),
            request: CompletionRequest::new(
                Provider::Vertex,
                "gemini-3.1-pro-preview",
                vec![Message::new_text(Role::User, "Reply with one word: hello")],
            )
            .with_max_tokens(10),
        }).collect()
    }

    #[tokio::test]
    async fn test_vertex_batch_create_and_cancel() {
        let manager = vertex_batch_manager();

        let requests = vertex_batch_requests(1);

        // Create the batch job
        let job = match manager.create(Provider::Vertex, requests).await {
            Ok(j) => j,
            Err(e) if e.code == "model_not_found" => {
                println!("SKIP: Vertex model not available on this project: {}", e.message);
                return;
            }
            Err(e) => panic!("failed to create vertex batch job: {}", e),
        };

        assert!(!job.id.is_empty(), "job id should not be empty");
        assert_eq!(job.provider, Provider::Vertex);
        assert!(
            matches!(job.status, BatchStatus::Pending | BatchStatus::Validating | BatchStatus::InProgress),
            "unexpected initial status: {:?}", job.status
        );

        // Verify get_batch works
        let fetched = manager
            .get(Provider::Vertex, &job.id)
            .await
            .expect("failed to fetch vertex batch job");
        assert_eq!(fetched.id, job.id);

        // Cancel so we don't waste quota
        manager
            .cancel(Provider::Vertex, &job.id)
            .await
            .expect("failed to cancel vertex batch job");

        // Confirm the job is either already terminal or on its way to cancellation.
        // Vertex cancel is asynchronous — the job may still be Pending/InProgress
        // for a short window before transitioning to Cancelled.
        let after_cancel = manager
            .get(Provider::Vertex, &job.id)
            .await
            .expect("failed to fetch job after cancel");
        assert!(
            after_cancel.status.is_done()
                || matches!(after_cancel.status, BatchStatus::Pending | BatchStatus::InProgress | BatchStatus::Validating),
            "unexpected job status after cancel: {:?}", after_cancel.status
        );
    }

    #[tokio::test]
    async fn test_vertex_batch_list() {
        let manager = vertex_batch_manager();

        // list returns successfully (may be empty if no prior jobs)
        let jobs = manager
            .list(Provider::Vertex, Some(agent_router::batch::ListOptions { limit: Some(5), after: None }))
            .await
            .expect("failed to list vertex batch jobs");

        // Each returned job must have a non-empty id
        for job in &jobs {
            assert!(!job.id.is_empty(), "listed job id should not be empty");
        }
    }

    /// End-to-end batch test: create → wait for completion → read results.
    ///
    /// This test is intentionally long-running — Vertex batch jobs typically
    /// take 5–20 minutes. It polls every 30 seconds with no upper timeout so
    /// that CI runners with generous timeouts can exercise the full path.
    #[tokio::test]
    async fn test_vertex_batch_e2e_create_wait_results() {
        use tokio::time::Duration;

        let manager = vertex_batch_manager();

        // Submit 2 requests so we can validate per-custom-id result matching.
        let requests = vertex_batch_requests(2);
        let expected_ids: std::collections::HashSet<String> =
            requests.iter().map(|r| r.custom_id.clone()).collect();

        let job = match manager.create(Provider::Vertex, requests).await {
            Ok(j) => j,
            Err(e) if e.code == "model_not_found" => {
                println!("SKIP: Vertex model not available on this project: {}", e.message);
                return;
            }
            Err(e) => panic!("failed to create vertex batch job: {}", e),
        };

        println!("batch job created: {} (status: {})", job.id, job.status);
        assert!(!job.id.is_empty());
        assert_eq!(job.provider, Provider::Vertex);
        // counts.total is 0 at creation time; Vertex populates it after validation.

        // Poll until the job reaches a terminal state.
        let completed = manager
            .wait(Provider::Vertex, &job.id, Duration::from_secs(15))
            .await
            .expect("failed while waiting for vertex batch job");

        println!("batch job finished: {} (status: {})", completed.id, completed.status);
        assert_eq!(completed.status, BatchStatus::Completed,
            "expected Completed but got {:?}", completed.status);
        assert_eq!(completed.counts.total, 2);
        assert_eq!(completed.counts.failed, 0);
        assert_eq!(completed.counts.completed, 2);

        // Fetch and validate results.
        let results = manager
            .get_results(Provider::Vertex, &completed.id)
            .await
            .expect("failed to get vertex batch results");

        assert_eq!(results.len(), 2, "expected 2 results, got {}", results.len());

        let mut seen_ids = std::collections::HashSet::new();
        for result in &results {
            assert!(
                expected_ids.contains(&result.custom_id),
                "unexpected custom_id in results: {:?}", result.custom_id
            );
            assert!(result.error.is_none(), "result {:?} has error: {:?}", result.custom_id, result.error);
            let resp = result.response.as_ref()
                .expect(&format!("result {:?} has no response", result.custom_id));
            assert!(!resp.text().is_empty(), "result {:?} has empty text", result.custom_id);
            seen_ids.insert(result.custom_id.clone());
        }

        assert_eq!(seen_ids, expected_ids, "not all custom_ids appeared in results");
    }

    /// Requires `VERTEX_BATCH_WAIT_RESULTS=1` plus the same Vertex batch env as other batch tests.
    #[tokio::test]
    async fn test_vertex_batch_metadata_get_results_echoed_labels() {
        if std::env::var("VERTEX_BATCH_WAIT_RESULTS").ok().as_deref() != Some("1") {
            println!("SKIP: set VERTEX_BATCH_WAIT_RESULTS=1 to run this long-running batch test");
            return;
        }

        use tokio::time::Duration;

        let manager = vertex_batch_manager();

        let requests: Vec<BatchRequest> = (1..=2)
            .map(|i| {
                let mut meta = HashMap::new();
                meta.insert("team".to_string(), "qa".to_string());
                BatchRequest {
                    custom_id: format!("meta-req-{}", i),
                    request: CompletionRequest::new(
                        Provider::Vertex,
                        "gemini-3.1-pro-preview",
                        vec![Message::new_text(Role::User, "Reply with one word: hello")],
                    )
                    .with_max_tokens(10)
                    .with_metadata(meta),
                }
            })
            .collect();

        let job = match manager.create(Provider::Vertex, requests).await {
            Ok(j) => j,
            Err(e) if e.code == "model_not_found" => {
                println!("SKIP: Vertex model not available: {}", e.message);
                return;
            }
            Err(e) => panic!("failed to create vertex batch job: {}", e),
        };

        let completed = manager
            .wait(Provider::Vertex, &job.id, Duration::from_secs(15))
            .await
            .expect("wait for vertex batch");

        if completed.status != BatchStatus::Completed {
            println!(
                "SKIP: batch ended with status {:?} (not Completed)",
                completed.status
            );
            return;
        }

        let results = manager
            .get_results(Provider::Vertex, &completed.id)
            .await
            .expect("get_results");

        for r in &results {
            let labels = r
                .request_labels
                .as_ref()
                .expect("expected echoed request_labels from Vertex");
            assert_eq!(labels.get("team"), Some(&"qa".to_string()));
            assert!(
                r.custom_id.starts_with("meta-req-"),
                "custom_id mismatch: {}",
                r.custom_id
            );
            assert_eq!(labels.get("custom_id"), Some(&r.custom_id));
        }
    }

    #[tokio::test]
    async fn test_multiple_providers() {
        setup();
        let openai_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        let anthropic_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();

        let mut opts = Vec::new();
        if !openai_key.is_empty() {
            opts.push(with_openai(&openai_key, vec![]));
        }
        if !anthropic_key.is_empty() {
            opts.push(with_anthropic(&anthropic_key, vec![]));
        }

        if opts.is_empty() {
            println!("skipping test: no API keys set");
            return;
        }

        let router = Router::new(opts).unwrap();
        let providers = router.providers();
        assert!(!providers.is_empty());
    }
}
