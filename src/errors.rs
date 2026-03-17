//! Unified error types for the agent router.

use crate::types::common::{Feature, Provider};
use thiserror::Error;

/// Error codes for programmatic handling.
pub const ERR_CODE_INVALID_REQUEST: &str = "invalid_request";
pub const ERR_CODE_AUTHENTICATION: &str = "authentication_error";
pub const ERR_CODE_RATE_LIMIT: &str = "rate_limit";
pub const ERR_CODE_SERVER_ERROR: &str = "server_error";
pub const ERR_CODE_UNSUPPORTED_FEATURE: &str = "unsupported_feature";
pub const ERR_CODE_PROVIDER_UNAVAILABLE: &str = "provider_unavailable";
pub const ERR_CODE_TIMEOUT: &str = "timeout";
pub const ERR_CODE_CONTENT_FILTER: &str = "content_filter";
pub const ERR_CODE_INVALID_API_KEY: &str = "invalid_api_key";
pub const ERR_CODE_MODEL_NOT_FOUND: &str = "model_not_found";
pub const ERR_CODE_CONTEXT_LENGTH: &str = "context_length_exceeded";

/// The base error type for all router errors.
#[derive(Debug, Clone, Error)]
#[error("{}", self.format_error())]
pub struct RouterError {
    /// Error code for programmatic handling.
    pub code: String,

    /// Human-readable error message.
    pub message: String,

    /// Provider that generated the error (if applicable).
    pub provider: Option<Provider>,

    /// HTTP status code from provider (if applicable).
    pub status_code: Option<u16>,

    /// Additional details.
    pub details: Option<serde_json::Value>,
}

impl RouterError {
    fn format_error(&self) -> String {
        let base = if let Some(ref p) = self.provider {
            format!("[{}] {}: {}", p, self.code, self.message)
        } else {
            format!("{}: {}", self.code, self.message)
        };
        base
    }

    /// Create a new RouterError.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            provider: None,
            status_code: None,
            details: None,
        }
    }

    /// Add provider information.
    pub fn with_provider(mut self, provider: Provider) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Add HTTP status code.
    pub fn with_status_code(mut self, code: u16) -> Self {
        self.status_code = Some(code);
        self
    }

    /// Add additional details.
    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }
}

// ---- Common error constructors ----

/// Create an invalid request error.
pub fn err_invalid_request(message: impl Into<String>) -> RouterError {
    RouterError::new(ERR_CODE_INVALID_REQUEST, message)
}

/// Create an authentication error.
pub fn err_authentication(provider: Provider, message: impl Into<String>) -> RouterError {
    RouterError::new(ERR_CODE_AUTHENTICATION, message).with_provider(provider)
}

/// Create a rate limit error.
pub fn err_rate_limit(provider: Provider, message: impl Into<String>) -> RouterError {
    RouterError::new(ERR_CODE_RATE_LIMIT, message)
        .with_provider(provider)
        .with_status_code(429)
}

/// Create a server error.
pub fn err_server_error(provider: Provider, message: impl Into<String>) -> RouterError {
    RouterError::new(ERR_CODE_SERVER_ERROR, message)
        .with_provider(provider)
        .with_status_code(500)
}

/// Create an unsupported feature error.
pub fn err_unsupported_feature(provider: Provider, feature: Feature) -> RouterError {
    RouterError::new(
        ERR_CODE_UNSUPPORTED_FEATURE,
        format!(
            "provider {} does not support feature: {}",
            provider, feature
        ),
    )
    .with_provider(provider)
}

/// Create a provider unavailable error.
pub fn err_provider_unavailable(provider: Provider, message: impl Into<String>) -> RouterError {
    RouterError::new(ERR_CODE_PROVIDER_UNAVAILABLE, message).with_provider(provider)
}

/// Create a timeout error.
pub fn err_timeout(provider: Provider) -> RouterError {
    RouterError::new(ERR_CODE_TIMEOUT, "request timed out").with_provider(provider)
}

/// Create an invalid API key error.
pub fn err_invalid_api_key(provider: Provider) -> RouterError {
    RouterError::new(ERR_CODE_INVALID_API_KEY, "invalid or missing API key")
        .with_provider(provider)
        .with_status_code(401)
}

/// Create a model not found error.
pub fn err_model_not_found(provider: Provider, model: impl Into<String>) -> RouterError {
    RouterError::new(
        ERR_CODE_MODEL_NOT_FOUND,
        format!("model not found: {}", model.into()),
    )
    .with_provider(provider)
    .with_status_code(404)
}

/// Create a context length exceeded error.
pub fn err_context_length(provider: Provider, message: impl Into<String>) -> RouterError {
    RouterError::new(ERR_CODE_CONTEXT_LENGTH, message)
        .with_provider(provider)
        .with_status_code(400)
}

/// Returns true if the error is potentially retryable.
pub fn is_retryable(err: &RouterError) -> bool {
    matches!(
        err.code.as_str(),
        ERR_CODE_RATE_LIMIT | ERR_CODE_SERVER_ERROR | ERR_CODE_TIMEOUT
    )
}

/// Returns true if the error is an authentication error.
pub fn is_auth_error(err: &RouterError) -> bool {
    matches!(
        err.code.as_str(),
        ERR_CODE_AUTHENTICATION | ERR_CODE_INVALID_API_KEY
    )
}
