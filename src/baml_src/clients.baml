// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview


client<llm> LocalQwen {
  provider ollama
  options {
    // base_url "http://localhost:11434/v1"
    model "qwen3:latest"
    temperature 0.0
  }
}

client<llm> QwenLocal {
  provider "openai-generic"
  retry_policy Exponential
  options {
    base_url "http://127.0.0.1:8000/v1"
    model "qwen3-1.7b-local"
    default_role "user" // vLLM-specific quirk
    temperature 0.0
    // chat_template_kwargs { "enable_thinking" false }
    // top_p 1.0
    logprobs null
  }
}

client<llm> MyClient {
  provider "openai"
  retry_policy Constant
  options {
    api_key env.OPENAI_API_KEY
    model "gpt-4.1-nano-2025-04-14"
    temperature 0.0
  }
}


// https://docs.boundaryml.com/docs/snippets/clients/fallback
client<llm> OpenaiFallback {
  provider fallback
  options {
    // This will try the clients in order until one succeeds
    strategy [QwenLocal, QwenLocal]
  }
}

// https://docs.boundaryml.com/docs/snippets/clients/retry
retry_policy Constant {
  max_retries 3
  // Strategy is optional
  strategy {
    type constant_delay
    delay_ms 200
  }
}

retry_policy Exponential {
  max_retries 3
  // Strategy is optional
  strategy {
    type exponential_backoff
    delay_ms 300
    multiplier 1.5
    max_delay_ms 10000
  }
}