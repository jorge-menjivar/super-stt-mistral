# Super Voice — Mistral backend

[![coverage](https://img.shields.io/endpoint?url=https://super-libre.github.io/super-voice-mistral/coverage.json)](https://super-libre.github.io/super-voice-mistral/)

Mistral's hosted speech models as a backend for
[Super Voice](https://github.com/jorge-menjivar/super-voice):

- [**Voxtral Transcribe 3**](https://docs.mistral.ai/studio/audio/speech-to-text/overview)
  to type what you say: `voxtral-mini-transcribe-3`,
  `voxtral-small-transcribe-3` and the streaming
  `voxtral-mini-transcribe-realtime-3`. It hears your dictation context: your
  vocabulary and prompt go to the model with every recording.
- [**Voxtral TTS 3**](https://docs.mistral.ai/studio/audio/text-to-speech/overview)
  to read text aloud: `voxtral-mini-tts-3`, with Mistral's 36 preset voices in
  22 languages. Or **in your own voice**: record a few seconds, and it speaks
  like you. The audio streams back as Mistral makes it.

Everything goes to Mistral, so it needs an API key and a connection rather
than a GPU.

The transcription models replace `voxtral-mini-latest` and
`voxtral-mini-transcribe-realtime-2602`, which Mistral has deprecated and
retires on April 30, 2027.

## What it is

The Super Voice daemon does not compile model inference in-tree. It discovers
backends and drives each one over a `/v1` HTTP contract. This is one such
backend: a sandboxed WebAssembly component that answers `POST /v1/transcribe`
and `POST /v1/synthesize` and streams realtime sessions over a WebSocket, with
no network access but `api.mistral.ai`.

## Requirements

A Mistral API key, and Super Voice newer than 0.3.0-beta.4.

## License

GPL-3.0-only. See `LICENSE`.
