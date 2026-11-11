// SPDX-License-Identifier: GPL-3.0-only
//! `POST /v1/synthesize`: text to Mistral's speech API, audio back as frames.
//!
//! The response head goes out once Mistral has accepted the request, and the
//! audio follows as Mistral streams it, so the daemon starts playing while the
//! rest is still being made. A refusal before that is the JSON envelope the
//! contract asks for. A failure after it is an `error` frame, since the status
//! line has been sent by then.
//!
//! The pure half (the request, the event stream, the frames) is
//! [`crate::speech`].

use super::wasi::http::types::{
    Fields, IncomingRequest, IncomingResponse, OutgoingBody, OutgoingResponse, ResponseOutparam,
};
use super::wasi::io::streams::{OutputStream, StreamError};
use crate::speech::{self, SpeechEvent, SseReader, Voice};

/// How much of Mistral's stream is read at once.
const READ_CHUNK: u64 = 64 * 1024;

/// Serve one synthesis.
pub(super) fn handle(request: &IncomingRequest, outparam: ResponseOutparam) {
    match start(request) {
        Ok(upstream) => respond_with_audio(&upstream, outparam),
        Err((status, body)) => super::send_response(outparam, status, &body),
    }
}

/// Everything before the audio: the key, the request, and Mistral taking it.
/// A refusal is the `(status, body)` to answer with.
fn start(request: &IncomingRequest) -> Result<IncomingResponse, (u16, Vec<u8>)> {
    let entries = request.headers().entries();
    let Some(api_key) = crate::header(&entries, "x-voice-secret-mistral_api_key") else {
        return Err(super::missing_key());
    };
    let base_url = super::base_url(&entries);
    let model = crate::header(&entries, "x-voice-model")
        .unwrap_or_else(|| speech::DEFAULT_MODEL.to_string());

    let body = request.consume().map_err(|()| super::err(400, "no_body"))?;
    let raw = super::read_all(body).map_err(|e| super::err(500, &e))?;
    let wanted = speech::parse_synthesize(&raw).map_err(|code| super::err(400, code))?;
    // A cloned voice is spoken from the clip registered for it; Mistral is
    // not asked when there is none.
    let clip;
    let voice = match wanted.voice.as_deref() {
        Some(id) if id.starts_with("voice:") => {
            clip = super::voices::clip(id).ok_or_else(|| {
                refusal(
                    400,
                    "unknown_voice",
                    "This cloned voice is not registered with the loaded model.",
                )
            })?;
            Voice::Cloned(&clip)
        }
        Some(id) => Voice::Preset(id),
        None => Voice::Preset(speech::DEFAULT_VOICE),
    };

    let payload = speech::speech_request_json(&model, &wanted.text, &voice);
    let response = super::post(
        &base_url,
        &api_key,
        "/v1/audio/speech",
        "application/json",
        &payload,
    )
    .map_err(|e| {
        refusal(
            500,
            "inference_failed",
            &format!("Mistral is unreachable: {e}"),
        )
    })?;

    let status = response.status();
    if status != 200 {
        let body = response
            .consume()
            .ok()
            .and_then(|b| super::read_all(b).ok())
            .unwrap_or_default();
        let (status, code, detail) = speech::upstream_refusal(status, &body);
        return Err(refusal(status, code, &detail));
    }
    Ok(response)
}

/// The JSON envelope for a refusal, with the `detail` the daemon shows.
fn refusal(status: u16, code: &str, detail: &str) -> (u16, Vec<u8>) {
    (
        status,
        super::to_vec(&serde_json::json!({
            "status": "error", "message": code, "detail": detail
        })),
    )
}

/// Answer `200` with the audio frames, as Mistral streams them, and end the
/// stream with `done` or, if it failed part way, `error`.
fn respond_with_audio(upstream: &IncomingResponse, outparam: ResponseOutparam) {
    let headers = Fields::new();
    let sample_rate = speech::SAMPLE_RATE.to_string();
    let channels = speech::CHANNELS.to_string();
    for (name, value) in [
        ("content-type", speech::FRAMES_CONTENT_TYPE),
        ("x-voice-sample-rate", sample_rate.as_str()),
        ("x-voice-channels", channels.as_str()),
        ("x-voice-format", speech::FORMAT),
    ] {
        let _ = headers.append(name, value.as_bytes());
    }
    let response = OutgoingResponse::new(headers);
    let _ = response.set_status_code(200);
    let Ok(body) = response.body() else {
        ResponseOutparam::set(outparam, Ok(response));
        return;
    };
    ResponseOutparam::set(outparam, Ok(response));
    let Ok(sink) = body.write() else {
        let _ = OutgoingBody::finish(body, None);
        return;
    };

    let last = match relay(upstream, &sink) {
        Ok(()) => speech::frame(speech::FRAME_DONE, &[]),
        Err(message) => speech::error_frame(&message),
    };
    // When the daemon has stopped reading (a cancelled utterance), this
    // fails too, and there is no one left to tell.
    let _ = write(&sink, &last);
    drop(sink);
    let _ = OutgoingBody::finish(body, None);
}

/// Pass Mistral's audio on until it says it is done. An error is the message
/// for the `error` frame.
fn relay(upstream: &IncomingResponse, sink: &OutputStream) -> Result<(), String> {
    let body = upstream
        .consume()
        .map_err(|()| "Mistral's answer has no body".to_string())?;
    let input = body
        .stream()
        .map_err(|()| "Mistral's answer has no body".to_string())?;
    let mut reader = SseReader::default();
    loop {
        let chunk = match input.blocking_read(READ_CHUNK) {
            Ok(chunk) => chunk,
            Err(StreamError::Closed) => {
                return Err("Mistral's speech ended before it was done".to_string());
            }
            Err(StreamError::LastOperationFailed(e)) => {
                return Err(format!(
                    "reading Mistral's speech failed: {}",
                    e.to_debug_string()
                ));
            }
        };
        for event in reader.push(&chunk) {
            match speech::classify_speech_event(&event) {
                SpeechEvent::Audio(pcm) => {
                    for frame in speech::audio_frames(&pcm) {
                        write(sink, &frame)?;
                    }
                }
                SpeechEvent::Done => return Ok(()),
                SpeechEvent::Error(message) => return Err(format!("Mistral: {message}")),
                SpeechEvent::Ignore => {}
            }
        }
    }
}

/// Write all of `bytes`, in the 4096-byte steps `blocking-write-and-flush`
/// allows.
fn write(sink: &OutputStream, bytes: &[u8]) -> Result<(), String> {
    for chunk in bytes.chunks(4096) {
        sink.blocking_write_and_flush(chunk)
            .map_err(|_| "the daemon stopped reading the speech".to_string())?;
    }
    Ok(())
}
