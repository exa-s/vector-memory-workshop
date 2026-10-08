# Vector Memory Workshop

A small retrieval demo for the ShopSmart store assistant. It chunks `kb.txt`, embeds each chunk, stores the vectors in a plain Python list (`memory.json`), and answers questions from the top cosine matches. There is no vector database.

## Setup

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Get an [OpenRouter](https://openrouter.ai/) API key and export it. Do not put the key in code or commit it.

```bash
export OPENROUTER_API_KEY="your-key"
```

Optional model overrides (both default to free OpenRouter models):

```bash
export OPENROUTER_EMBEDDING_MODEL="nvidia/nemotron-3-embed-1b:free"
export OPENROUTER_CHAT_MODEL="nvidia/nemotron-3.5-lightning:free"
```

## Run

```bash
python memory_agent.py
```

The first run indexes `kb.txt` and writes `memory.json`. Later runs reuse that file when the knowledge base and embedding model are unchanged.

```text
You: What is the return window for unused items?
```

Type `exit` to quit.

| Flag | Effect |
| --- | --- |
| `--reindex` | Rebuild `memory.json` from `kb.txt` |
| `--top-k N` | How many chunks to retrieve (default `3`) |

## How a question is answered

1. `kb.txt` is split into overlapping character chunks (500 characters, 60 overlap).
2. Each chunk is embedded and saved in `memory.json` with its text and chunk id.
3. The question is embedded and scored against every stored vector with cosine similarity.
4. The top matches are printed, then sent to the chat model as the only allowed context.
5. The model must cite chunk ids such as `[chunk-001]`, or say it could not find the answer in the knowledge base.

Try a question that is not in `kb.txt` to see the grounding behavior. `kb.txt` is fictional workshop data.
