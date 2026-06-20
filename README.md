# Super STT — Mistral backend

[![coverage](https://img.shields.io/endpoint?url=https://jorge-menjivar.github.io/super-stt-mistral/coverage.json)](https://jorge-menjivar.github.io/super-stt-mistral/)

A speech-to-text backend for **[Super STT](https://github.com/jorge-menjivar/super-stt)**.
It proxies audio to [Mistral](https://mistral.ai/)'s hosted transcription API, so
transcription runs in the cloud rather than on your machine. It supports both **batch**
transcription and **realtime** (streaming) transcription over a WebSocket.

Super STT is an on-device speech-to-text engine. It doesn't ship any models of its own —
it loads **backends** like this one at runtime. This repo packages Mistral as one of those
backends, shipped as a sandboxed **WASM component** (a `wasi:http` proxy that also speaks
the `super-stt:realtime` WebSocket world).

## Using it

You don't run this directly. Super STT discovers it through its backend registry,
downloads the prebuilt `.wasm` from this repo's GitHub release, and runs it in-process in a
WASM sandbox whose only network egress is the allowlisted Mistral API. To use it, install
Super STT, enable Mistral from the app, and add your **Mistral API key** — see the
[Super STT docs](https://github.com/jorge-menjivar/super-stt).

## Models

Chosen by `name` when Super STT loads the backend. These are **online** models: they send
audio to Mistral and need a Mistral API key (set in the app); no local GPU or weights are
involved.

| Model (`name`)                          | Provider | Type             | Languages | Requires        |
| --------------------------------------- | -------- | ---------------- | --------- | --------------- |
| `voxtral-mini-latest`                   | mistral  | online           | en        | Mistral API key |
| `voxtral-mini-transcribe-realtime-2602` | mistral  | online, realtime | en        | Mistral API key |

The realtime model streams audio over a WebSocket to
`wss://api.mistral.ai/v1/audio/transcriptions/realtime`.

## What's in here

A small, self-contained Rust component (`src/`) that speaks the Super STT backend protocol
(the `/v1` contract) plus the `super-stt:realtime/ws-server` interface, forwarding audio to
Mistral over `wasi:http/outgoing-handler` (batch) and the daemon-provided
`super-stt:realtime/ws` (realtime). It shares no code with the Super STT project. The pure
audio/parsing/payload helpers are unit-tested natively; the component as a whole is
exercised by a wasmtime harness under `tests/` that loads the built `.wasm` and drives the
batch `/v1` contract against a mock upstream and the realtime path against a mock WebSocket.

## Building from source

Most people never need to — Super STT downloads prebuilt releases. For development (requires
[`just`](https://github.com/casey/just) and the `wasm32-wasip2` target):

```bash
rustup target add wasm32-wasip2
just build-component   # builds target/wasm32-wasip2/release/super_stt_backend_mistral.wasm
just ci                # format, lint, build, and test
```

## License

GPL-3.0-only.
