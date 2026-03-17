//! Streaming example demonstrating real-time token streaming.

use futures::StreamExt;
use agent_router::{with_openai, Router};
use agent_router::types::{CompletionRequest, Message, Provider, Role, StreamEventType};

#[tokio::main]
async fn main() {
    let _ = dotenvy::dotenv();

    let openai_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();

    let router = Router::new(vec![with_openai(&openai_key, vec![])])
        .expect("failed to create router");

    println!("=== Streaming Response ===");

    let req = CompletionRequest::new(
        Provider::OpenAI,
        "gpt-4o-mini",
        vec![Message::new_text(
            Role::User,
            "Tell me a short story about a robot in 3 sentences.",
        )],
    );

    let mut stream = router.stream(&req).await.expect("failed to start stream");

    while let Some(result) = stream.next().await {
        match result {
            Ok(event) => match event.event_type {
                StreamEventType::Start => {
                    if let Some(model) = &event.model {
                        println!("[Started, model: {}]", model);
                    }
                }
                StreamEventType::ContentDelta => {
                    if let Some(delta) = &event.delta {
                        if let Some(text) = &delta.text {
                            print!("{}", text);
                        }
                    }
                }
                StreamEventType::ToolCallStart => {
                    if let Some(tc) = &event.tool_call {
                        println!("\n[Tool call: {}]", tc.name);
                    }
                }
                StreamEventType::Done => {
                    println!();
                    if let Some(stop) = &event.stop_reason {
                        println!("[Done, stop reason: {}]", stop);
                    }
                    if let Some(usage) = &event.usage {
                        println!("[Tokens: {} input, {} output]", usage.input_tokens, usage.output_tokens);
                    }
                }
                StreamEventType::Error => {
                    if let Some(err) = &event.error {
                        eprintln!("\n[Error: {}]", err);
                    }
                }
                _ => {}
            },
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }
}
