# Super Voice — Mistral backend

[![coverage](https://img.shields.io/endpoint?url=https://super-libre.github.io/super-voice-mistral/coverage.json)](https://super-libre.github.io/super-voice-mistral/)

[Mistral](https://docs.mistral.ai/capabilities/audio_transcription)'s hosted
speech-to-text, Voxtral Transcribe 3 (`voxtral-mini-transcribe-3`,
`voxtral-small-transcribe-3` and the streaming
`voxtral-mini-transcribe-realtime-3`), as a backend for
[Super Voice](https://github.com/jorge-menjivar/super-voice). The audio goes to
Mistral, so it needs an API key and a connection rather than a GPU. It hears
your dictation context: your vocabulary and prompt go to the model with every
recording.

These models replace `voxtral-mini-latest` and
`voxtral-mini-transcribe-realtime-2602`, which Mistral has deprecated and
retires on April 30, 2027.

## What it is

The Super Voice daemon does not compile model inference in-tree. It discovers
backends and drives each one over a `/v1` HTTP contract. This is one such
backend: a sandboxed WebAssembly component that answers `POST /v1/transcribe`
and streams realtime sessions over a WebSocket, with no network access but
`api.mistral.ai`.

## License

GPL-3.0-only. See `LICENSE`.
