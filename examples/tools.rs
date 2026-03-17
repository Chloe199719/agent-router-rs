//! Tool calling example demonstrating function/tool invocation.

use agent_router::{with_openai, Router};
use agent_router::types::{
    CompletionRequest, ContentBlock, JsonSchema, Message, Provider, Role, Tool,
};

#[tokio::main]
async fn main() {
    let _ = dotenvy::dotenv();

    let openai_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();

    let router = Router::new(vec![with_openai(&openai_key, vec![])])
        .expect("failed to create router");

    // Define tools
    let mut weather_props = std::collections::HashMap::new();
    weather_props.insert(
        "location".to_string(),
        JsonSchema {
            schema_type: Some("string".to_string()),
            description: Some("The city and country, e.g., 'Paris, France'".to_string()),
            ..Default::default()
        },
    );
    weather_props.insert(
        "unit".to_string(),
        JsonSchema {
            schema_type: Some("string".to_string()),
            enum_values: Some(vec![
                serde_json::Value::String("celsius".to_string()),
                serde_json::Value::String("fahrenheit".to_string()),
            ]),
            description: Some("Temperature unit".to_string()),
            ..Default::default()
        },
    );

    let mut search_props = std::collections::HashMap::new();
    search_props.insert(
        "query".to_string(),
        JsonSchema {
            schema_type: Some("string".to_string()),
            description: Some("The search query".to_string()),
            ..Default::default()
        },
    );

    let tools = vec![
        Tool {
            name: "get_weather".to_string(),
            description: Some("Get the current weather for a location".to_string()),
            parameters: JsonSchema {
                schema_type: Some("object".to_string()),
                properties: Some(weather_props),
                required: Some(vec!["location".to_string()]),
                ..Default::default()
            },
        },
        Tool {
            name: "search_web".to_string(),
            description: Some("Search the web for information".to_string()),
            parameters: JsonSchema {
                schema_type: Some("object".to_string()),
                properties: Some(search_props),
                required: Some(vec!["query".to_string()]),
                ..Default::default()
            },
        },
    ];

    println!("=== Tool Calling Example ===");
    let req = CompletionRequest::new(
        Provider::OpenAI,
        "gpt-4o-mini",
        vec![Message::new_text(Role::User, "What's the weather like in Tokyo?")],
    )
    .with_tools(tools.clone());

    let resp = router.complete(&req).await.expect("completion failed");

    if resp.has_tool_calls() {
        println!("Model wants to use tools:");
        for tc in &resp.tool_calls {
            let input_json = serde_json::to_string_pretty(&tc.input).unwrap_or_default();
            println!("  - {}({})", tc.name, input_json);
        }

        // Build conversation with tool results
        let mut messages = vec![
            Message::new_text(Role::User, "What's the weather like in Tokyo?"),
        ];

        // Add assistant's response (with tool calls in content)
        messages.push(Message {
            role: Role::Assistant,
            content: resp.content.clone(),
        });

        // Add tool results
        for tc in &resp.tool_calls {
            let result = match tc.name.as_str() {
                "get_weather" => r#"{"temperature": 22, "condition": "Partly cloudy", "humidity": 65}"#,
                "search_web" => r#"{"results": [{"title": "Example result", "url": "https://example.com"}]}"#,
                _ => r#"{"error": "Unknown tool"}"#,
            };
            messages.push(Message::new_tool_result(&tc.id, result, false));
        }

        println!("\n=== Continuing with tool results ===");
        let req2 = CompletionRequest::new(Provider::OpenAI, "gpt-4o-mini", messages)
            .with_tools(tools);

        match router.complete(&req2).await {
            Ok(resp) => println!("Final response: {}", resp.text()),
            Err(e) => eprintln!("Error: {}", e),
        }
    } else {
        println!("Response (no tool calls): {}", resp.text());
    }
}
