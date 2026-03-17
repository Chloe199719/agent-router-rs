//! Basic usage example demonstrating completions across multiple providers.

use agent_router::{with_anthropic, with_google, with_openai, Router};
use agent_router::types::{CompletionRequest, JsonSchema, Message, Provider, Role};

#[tokio::main]
async fn main() {
    // Load .env if present
    let _ = dotenvy::dotenv();

    let openai_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    let anthropic_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
    let google_key = std::env::var("GOOGLE_API_KEY").unwrap_or_default();

    let router = Router::new(vec![
        with_openai(&openai_key, vec![]),
        with_anthropic(&anthropic_key, vec![]),
        with_google(&google_key, vec![]),
    ])
    .expect("failed to create router");

    // Example 1: OpenAI completion
    println!("=== OpenAI Completion ===");
    let req = CompletionRequest::new(
        Provider::OpenAI,
        "gpt-4o-mini",
        vec![Message::new_text(Role::User, "Say hello in French")],
    );
    match router.complete(&req).await {
        Ok(resp) => {
            println!("Response: {}", resp.text());
            println!("Tokens: {} input, {} output", resp.usage.input_tokens, resp.usage.output_tokens);
        }
        Err(e) => eprintln!("OpenAI error: {}", e),
    }

    // Example 2: Anthropic completion
    println!("\n=== Anthropic Completion ===");
    let req = CompletionRequest::new(
        Provider::Anthropic,
        "claude-3-haiku-20240307",
        vec![Message::new_text(Role::User, "Say hello in French")],
    )
    .with_max_tokens(100);
    match router.complete(&req).await {
        Ok(resp) => {
            println!("Response: {}", resp.text());
            println!("Tokens: {} input, {} output", resp.usage.input_tokens, resp.usage.output_tokens);
        }
        Err(e) => eprintln!("Anthropic error: {}", e),
    }

    // Example 3: Google completion
    println!("\n=== Google Completion ===");
    let req = CompletionRequest::new(
        Provider::Google,
        "gemini-2.0-flash",
        vec![Message::new_text(Role::User, "Say hello in French")],
    );
    match router.complete(&req).await {
        Ok(resp) => {
            println!("Response: {}", resp.text());
            println!("Tokens: {} input, {} output", resp.usage.input_tokens, resp.usage.output_tokens);
        }
        Err(e) => eprintln!("Google error: {}", e),
    }

    // Example 4: Structured output (OpenAI)
    println!("\n=== Structured Output (OpenAI) ===");
    let mut properties = std::collections::HashMap::new();
    properties.insert(
        "greeting".to_string(),
        JsonSchema {
            schema_type: Some("string".to_string()),
            description: Some("The greeting in French".to_string()),
            ..Default::default()
        },
    );
    properties.insert(
        "language".to_string(),
        JsonSchema {
            schema_type: Some("string".to_string()),
            description: Some("The language code".to_string()),
            ..Default::default()
        },
    );

    let schema = JsonSchema {
        schema_type: Some("object".to_string()),
        properties: Some(properties),
        required: Some(vec!["greeting".to_string(), "language".to_string()]),
        ..Default::default()
    };

    let req = CompletionRequest::new(
        Provider::OpenAI,
        "gpt-4o-mini",
        vec![Message::new_text(Role::User, "Give me a French greeting")],
    )
    .with_json_schema("greeting_response", schema);

    match router.complete(&req).await {
        Ok(resp) => println!("Structured response: {}", resp.text()),
        Err(e) => eprintln!("Structured output error: {}", e),
    }
}
