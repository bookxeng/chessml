# The LLM explains engine facts; it never analyses chess

Explanations come from an LLM, but the LLM is given only facts that Search produced (candidate moves, expected scores, main lines, captures, checks, material) and is told to use nothing else. A guardrail rejects any explanation that names a move not in those facts, retries once with the reason, and then falls back to a plain template built from the facts. Letting the LLM look at the position and reason about it directly would be simpler, but general LLMs are weak at chess: tokenization splits moves into fragments, and they invent threats and illegal moves fluently. Grounding keeps explanations truthful even with a small local model (Qwen3-1.7B).

## Consequences

- The guardrail only checks moves that cannot be mistaken for a square name (piece moves, captures, castling, promotions). A plain pawn push like "e4" looks exactly like a square, so it is not checked.
- The quality of explanations is limited by how rich the facts are. Adding facts (for example threats, or pieces left undefended) is the way to improve explanations, not prompting the LLM to analyse more.
- LLM backends sit behind one `LLMBackend` interface: any OpenAI-compatible server (Ollama by default, or OpenAI itself) or Hugging Face transformers in-process. Tests use a fake backend and need no model.
