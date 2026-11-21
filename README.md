# systemone-bifrost-shim

An OpenAI-compatible front for decision models served by `llama-server`, so they can be registered as a custom OpenAI provider in the [bifrost](https://github.com/maximhq/bifrost) gateway.

Decision models such as [LiquidAI d1](https://huggingface.co/LiquidAI/d1-2B-GGUF) don't generate text. They read a state and typed questions (`noul`, `choice`, `score`) and return calibrated answers in one forward pass, through llama-server's `/v1/systemone` endpoint (llama.cpp b11960 and newer). Gateways and OpenAI clients only speak chat completions, so this shim carries the decision request inside one.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/v1/models` | the configured models |
| `POST` | `/v1/chat/completions` | a decision request in a chat completion (below) |
| `POST` | `/v1/systemone` | the native body, passed through; pick the model with a `model` field |
| `POST` | `/v1/embeddings` | passed through to an `embeddings` model |
| `GET` | `/health` | liveness |

## Asking a decision model through chat completions

Send the decision request as the last user message, as JSON:

```json
{
  "model": "d1-2b",
  "response_format": {"type": "json_object"},
  "messages": [{"role": "user", "content": "{\"state\": \"The prisoner spits and calls the guard a coward.\", \"questions\": {\"speaker\": {\"type\": \"choice\", \"instructions\": \"Who speaks next?\", \"criteria\": {\"Inquisitor\": \"\", \"Servitor\": \"\", \"Guard\": \"\"}}}}"}]
}
```

You can put it in a top-level `systemone` object instead of the message. If the object has no `state`, the text of the system and earlier messages becomes the state.

The assistant message content is the model's answer as JSON, ready for `json.loads`:

```json
{"model": "d1-2b", "answers": {"speaker": {"type": "choice", "choice": "Guard", "probabilities": {"Inquisitor": 0.23, "Servitor": 0.07, "Guard": 0.70}}}}
```

`stream: true` returns the same answer as one chunk. A plain-text message gets a 400 that explains the format.

## Configuration

`config/models.json` maps each public model ID to one llama-server (each llama-server serves one model):

```json
{
  "models": {
    "d1-2b": {"upstream": "http://127.0.0.1:8080"},
    "nomic-embed-text-v1.5": {"upstream": "http://127.0.0.1:8081", "kind": "embeddings"}
  }
}
```

`kind` is `decision` (the default) or `embeddings`.

Environment:

| Variable | Default |
|---|---|
| `SHIM_ADDR` | `0.0.0.0:8190` |
| `SHIM_CONFIG` | `./config/models.json` |

## Running the models

```bash
llama-server -m d1-2B-Q8_0.gguf --mmproj mmproj-d1-2B-Q8_0.gguf -ngl 99 -c 4096 -b 4096 -ub 4096 --port 8080
llama-server -m nomic-embed-text-v1.5.Q8_0.gguf --embedding --pooling mean -ngl 99 -c 8192 -b 2048 -ub 2048 --port 8081
```

A question is read in one batch, so `-b` and `-ub` must hold the whole state. On a 4 GB GTX 970 (Vulkan build) the two together use about 3.2 GB.

## Build

Standard library only. A version tag builds a static linux/amd64 binary and attaches it to the release (`.github/workflows/release.yml`).

```bash
CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o systemone-bifrost-shim .
```

## License

AGPL-3.0-or-later.
