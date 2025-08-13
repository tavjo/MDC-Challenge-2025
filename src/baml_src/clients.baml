// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview


client<llm> LocalQwen {
  provider ollama
  options {
    // base_url "http://localhost:11434/v1"
    model "qwen3:latest"
  }
}

client<llm> QwenLocal {
  provider "openai-generic"
  options {
    base_url "http://127.0.0.1:8000/v1"
    model "qwen3-8b-awq-local"
    default_role "user" // vLLM-specific quirk
  }
}


// https://docs.boundaryml.com/docs/snippets/clients/fallback
client<llm> OpenaiFallback {
  provider fallback
  options {
    // This will try the clients in order until one succeeds
    strategy [LocalQwen, LocalQwen]
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
  max_retries 2
  // Strategy is optional
  strategy {
    type exponential_backoff
    delay_ms 300
    multiplier 1.5
    max_delay_ms 10000
  }
}