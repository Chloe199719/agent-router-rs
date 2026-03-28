//! agent-router-rs — a unified Rust library for making LLM inference requests
//! across multiple providers (OpenAI, Anthropic, Google/Gemini, Vertex AI).
//!
//! # Example
//!
//! ```rust,no_run
//! use agent_router::{Router, with_openai, with_anthropic};
//! use agent_router::types::{CompletionRequest, Message, Provider, Role};
//!
//! #[tokio::main]
//! async fn main() {
//!     let router = Router::new(vec![
//!         with_openai("sk-...", vec![]),
//!         with_anthropic("sk-ant-...", vec![]),
//!     ]).unwrap();
//!
//!     let req = CompletionRequest::new(
//!         Provider::OpenAI,
//!         "gpt-4o",
//!         vec![Message::new_text(Role::User, "Hello!")],
//!     );
//!
//!     let resp = router.complete(&req).await.unwrap();
//!     println!("{}", resp.text());
//! }
//! ```

pub mod types;
pub mod errors;
pub mod schema;
pub mod provider;
pub mod batch;

use std::collections::HashMap;
use std::sync::Arc;

use errors::*;
use provider::{BatchProviderClient, ProviderClient, ProviderOption, StreamResponse};
use types::{CompletionRequest, CompletionResponse, Feature, Provider};

/// Controls behavior when a provider doesn't support a requested feature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnsupportedFeaturePolicy {
    /// Return an error.
    Error,
    /// Log a warning and continue without the feature.
    Warn,
    /// Silently ignore.
    Ignore,
}

impl Default for UnsupportedFeaturePolicy {
    fn default() -> Self {
        Self::Error
    }
}

/// Router configuration.
#[derive(Debug, Clone, Default)]
pub struct RouterConfig {
    pub on_unsupported_feature: UnsupportedFeaturePolicy,
    pub debug: bool,
}

/// A function that adds providers or changes config on the router builder.
/// Returns `Result` so that provider construction errors (e.g. bad TLS config)
/// are propagated to the caller of `Router::new` instead of panicking.
pub struct RouterOption(Box<dyn FnOnce(&mut RouterBuilder) -> std::result::Result<(), RouterError> + Send>);

impl RouterOption {
    fn apply(self, builder: &mut RouterBuilder) -> std::result::Result<(), RouterError> {
        (self.0)(builder)
    }
}

/// Builder for constructing a Router.
pub struct RouterBuilder {
    providers: HashMap<Provider, Arc<dyn ProviderClient>>,
    batch_providers: Vec<(Provider, Arc<dyn BatchProviderClient>)>,
    config: RouterConfig,
}

impl RouterBuilder {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            batch_providers: Vec::new(),
            config: RouterConfig::default(),
        }
    }
}

impl Default for RouterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// The unified LLM router.
pub struct Router {
    providers: HashMap<Provider, Arc<dyn ProviderClient>>,
    batch: batch::Manager,
    config: RouterConfig,
}

impl Router {
    /// Create a new Router with the given options.
    pub fn new(opts: Vec<RouterOption>) -> std::result::Result<Self, RouterError> {
        let mut builder = RouterBuilder::new();
        for opt in opts {
            opt.apply(&mut builder)?;
        }

        if builder.providers.is_empty() {
            return Err(err_invalid_request("at least one provider must be configured"));
        }

        let mut batch_manager = batch::Manager::new();
        for (_, bp) in builder.batch_providers {
            batch_manager.register_provider(bp);
        }

        Ok(Self {
            providers: builder.providers,
            batch: batch_manager,
            config: builder.config,
        })
    }

    /// Send a completion request to the specified provider.
    pub async fn complete(&self, req: &CompletionRequest) -> std::result::Result<CompletionResponse, RouterError> {
        let p = self.get_provider(&req.provider)?;
        self.check_feature_support(p.as_ref(), req)?;
        p.complete(req).await
    }

    /// Send a streaming completion request.
    pub async fn stream(&self, req: &CompletionRequest) -> std::result::Result<StreamResponse, RouterError> {
        let p = self.get_provider(&req.provider)?;

        if !p.supports_feature(&Feature::Streaming) {
            return Err(err_unsupported_feature(req.provider.clone(), Feature::Streaming));
        }

        self.check_feature_support(p.as_ref(), req)?;
        p.stream(req).await
    }

    /// Returns the batch manager.
    pub fn batch(&self) -> &batch::Manager {
        &self.batch
    }

    /// Returns a specific provider for direct access.
    pub fn provider(&self, name: &Provider) -> std::result::Result<Arc<dyn ProviderClient>, RouterError> {
        self.providers.get(name)
            .cloned()
            .ok_or_else(|| err_provider_unavailable(name.clone(), "provider not configured"))
    }

    /// Returns all configured providers.
    pub fn providers(&self) -> Vec<Provider> {
        self.providers.keys().cloned().collect()
    }

    /// Checks if a provider supports a specific feature.
    pub fn supports_feature(&self, provider_name: &Provider, feature: &Feature) -> bool {
        self.providers.get(provider_name)
            .map(|p| p.supports_feature(feature))
            .unwrap_or(false)
    }

    /// Lists model identifiers for a provider by querying the provider's API.
    pub async fn models(&self, provider_name: &Provider) -> std::result::Result<Vec<String>, RouterError> {
        let p = self.get_provider(provider_name)?;
        p.models().await
    }

    fn get_provider(&self, name: &Provider) -> std::result::Result<Arc<dyn ProviderClient>, RouterError> {
        self.providers.get(name)
            .cloned()
            .ok_or_else(|| err_provider_unavailable(name.clone(), "provider not configured"))
    }

    fn check_feature_support(&self, p: &dyn ProviderClient, req: &CompletionRequest) -> std::result::Result<(), RouterError> {
        // Check structured output
        if let Some(rf) = &req.response_format {
            if rf.format_type == "json_schema" && !p.supports_feature(&Feature::StructuredOutput) {
                return self.handle_unsupported_feature(p.name(), Feature::StructuredOutput);
            }
            if rf.format_type == "json" && !p.supports_feature(&Feature::Json) {
                return self.handle_unsupported_feature(p.name(), Feature::Json);
            }
        }

        // Check tools
        if !req.tools.is_empty() && !p.supports_feature(&Feature::Tools) {
            return self.handle_unsupported_feature(p.name(), Feature::Tools);
        }

        // Check vision (images in messages)
        'outer: for msg in &req.messages {
            for block in &msg.content {
                if block.content_type == Some(types::ContentType::Image) && !p.supports_feature(&Feature::Vision) {
                    self.handle_unsupported_feature(p.name(), Feature::Vision)?;
                    break 'outer;
                }
            }
        }

        Ok(())
    }

    fn handle_unsupported_feature(&self, provider: Provider, feature: Feature) -> std::result::Result<(), RouterError> {
        match self.config.on_unsupported_feature {
            UnsupportedFeaturePolicy::Error => Err(err_unsupported_feature(provider, feature)),
            UnsupportedFeaturePolicy::Warn => {
                eprintln!(
                    "[agent-router] warning: provider '{}' does not support feature '{}'; continuing without it",
                    provider, feature
                );
                Ok(())
            }
            UnsupportedFeaturePolicy::Ignore => Ok(()),
        }
    }
}

// ---- Router option constructors ----

/// Add OpenAI as a provider.
pub fn with_openai(api_key: impl Into<String>, mut opts: Vec<ProviderOption>) -> RouterOption {
    let api_key = api_key.into();
    RouterOption(Box::new(move |builder: &mut RouterBuilder| {
        let mut all_opts = vec![provider::with_api_key(api_key)];
        all_opts.append(&mut opts);
        let client = Arc::new(provider::openai::Client::new(all_opts)?);
        builder.providers.insert(Provider::OpenAI, client.clone());
        builder.batch_providers.push((Provider::OpenAI, client));
        Ok(())
    }))
}

/// Add Anthropic as a provider.
pub fn with_anthropic(api_key: impl Into<String>, mut opts: Vec<ProviderOption>) -> RouterOption {
    let api_key = api_key.into();
    RouterOption(Box::new(move |builder: &mut RouterBuilder| {
        let mut all_opts = vec![provider::with_api_key(api_key)];
        all_opts.append(&mut opts);
        let client = Arc::new(provider::anthropic::Client::new(all_opts)?);
        builder.providers.insert(Provider::Anthropic, client.clone());
        builder.batch_providers.push((Provider::Anthropic, client));
        Ok(())
    }))
}

/// Add Google (Gemini) as a provider.
pub fn with_google(api_key: impl Into<String>, mut opts: Vec<ProviderOption>) -> RouterOption {
    let api_key = api_key.into();
    RouterOption(Box::new(move |builder: &mut RouterBuilder| {
        let mut all_opts = vec![provider::with_api_key(api_key)];
        all_opts.append(&mut opts);
        let client = Arc::new(provider::google::Client::new(all_opts)?);
        builder.providers.insert(Provider::Google, client.clone());
        builder.batch_providers.push((Provider::Google, client));
        Ok(())
    }))
}

/// Add Google Vertex AI as a provider.
///
/// Authentication via `provider::with_access_token()` (OAuth2) or
/// `provider::with_api_key()` in opts.
pub fn with_vertex(project_id: impl Into<String>, location: impl Into<String>, opts: Vec<ProviderOption>) -> RouterOption {
    let project_id = project_id.into();
    let location = location.into();
    RouterOption(Box::new(move |builder: &mut RouterBuilder| {
        let client = Arc::new(provider::vertex::Client::new(project_id, location, opts)?);
        builder.providers.insert(Provider::Vertex, client.clone());
        builder.batch_providers.push((Provider::Vertex, client));
        Ok(())
    }))
}

/// Set the unsupported feature policy.
pub fn with_unsupported_feature_policy(policy: UnsupportedFeaturePolicy) -> RouterOption {
    RouterOption(Box::new(move |builder: &mut RouterBuilder| {
        builder.config.on_unsupported_feature = policy;
        Ok(())
    }))
}

/// Enable or disable debug logging.
pub fn with_debug(debug: bool) -> RouterOption {
    RouterOption(Box::new(move |builder: &mut RouterBuilder| {
        builder.config.debug = debug;
        Ok(())
    }))
}
