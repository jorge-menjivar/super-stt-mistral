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
  22 languages, or in a [voice cloned](#cloned-voices) from your own
  recording. The audio streams back as Mistral makes it.

Everything goes to Mistral, so it needs an API key and a connection rather
than a GPU.

The transcription models replace `voxtral-mini-latest` and
`voxtral-mini-transcribe-realtime-2602`, which Mistral has deprecated and
retires on April 30, 2027.

## Cloned voices

Add a voice in Super Voice from a recording of your own, then pick it for
`voxtral-mini-tts-3`. Mistral clones it from the recording on the spot: there
is nothing to train and nothing to upload ahead of time.

Where your recording goes:

- **Your machine.** Super Voice keeps the recording, in its voice library, until
  you delete it.
- **This backend, while the model is loaded.** The first time you speak in the
  voice, Super Voice hands the backend up to 15 s of the recording. The backend
  keeps it in a private folder that no other backend sees, which is in memory
  on Linux, and which is deleted when the model unloads.
- **Mistral, with every sentence in that voice.** The clip goes along as the
  reference audio. Mistral clones from it for that sentence and does not save
  it as a voice: none is created in your Mistral account. Mistral's terms
  govern what it keeps of a request otherwise.

Only the audio goes to Mistral. Its API takes no transcript, so one stored
with the recording stays on your machine.

## What it is

The Super Voice daemon does not compile model inference in-tree. It discovers
backends and drives each one over a `/v1` HTTP contract. This is one such
backend: a sandboxed WebAssembly component that answers `POST /v1/transcribe`
and `POST /v1/synthesize` and streams realtime sessions over a WebSocket, with
no network access but `api.mistral.ai`.

## Requirements

A Mistral API key, and Super Voice newer than 0.3.0-beta.4. This version keeps
cloned voices in the private folder that Super Voice gives a WebAssembly
backend, and a Super Voice without that folder does not load the backend at
all.

## License

GPL-3.0-only. See `LICENSE`.
