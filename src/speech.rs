// SPDX-License-Identifier: GPL-3.0-only
//! Text to speech: the pure half of `POST /v1/synthesize`.
//!
//! The component asks Mistral's `POST /v1/audio/speech` for a stream: `pcm`
//! audio, which is 24 kHz mono `f32le`, sent as server-sent events, each a
//! `speech.audio.delta` carrying a base64 chunk, then one `speech.audio.done`.
//! It answers the daemon with the framed body the contract defines, passing
//! the audio on as it arrives.
//!
//! Everything here compiles and is unit-tested on the host. The wasm-only
//! `component::synthesize` wires it to `wasi:http`.

use base64::Engine as _;
use serde_json::{Value, json};

/// The model a request naming none is synthesized with.
pub const DEFAULT_MODEL: &str = "voxtral-mini-tts-3";

/// The voice a request naming none is spoken in. The manifest's
/// `default_voice`, which the daemon sends when the user picked none, so this
/// is only reached by a caller that is not the daemon.
pub const DEFAULT_VOICE: &str = "voice_478";

/// What Mistral's `pcm` is: 24 kHz, mono, little-endian 32-bit float.
pub const SAMPLE_RATE: u32 = 24_000;
pub const CHANNELS: u16 = 1;
pub const FORMAT: &str = "f32le";

/// The content type of the framed body.
pub const FRAMES_CONTENT_TYPE: &str = "application/vnd.super-voice.frames";

/// The most audio one frame carries. The daemon refuses a frame over 1 MiB,
/// and Mistral sends a long input's audio in deltas of 2 MB and more, so a
/// delta is passed on as several frames.
pub const MAX_FRAME_AUDIO: usize = 64 * 1024;

/// Frame kinds, as the contract numbers them.
pub const FRAME_AUDIO: u8 = 0x01;
pub const FRAME_DONE: u8 = 0x03;
pub const FRAME_ERROR: u8 = 0x04;

/// What the daemon asked to have spoken.
#[derive(Debug, PartialEq, Eq)]
pub struct SynthesizeRequest {
    pub text: String,
    /// A preset voice id. The model declares no other kind, so the daemon
    /// never sends a `voice:` or `desc:` one.
    pub voice: Option<String>,
}

/// Read a `/v1/synthesize` body.
///
/// Only `text` and `voice` are used. Mistral takes no language, since the
/// voice decides it, and no rate, and it refuses a request carrying either.
/// `instructions` it refuses for this model. All three are left unsent, as
/// the contract asks of a backend that cannot honor them.
///
/// # Errors
/// `invalid_text` when the body is not JSON or the text is missing or blank.
pub fn parse_synthesize(body: &[u8]) -> Result<SynthesizeRequest, &'static str> {
    let v: Value = serde_json::from_slice(body).map_err(|_| "invalid_text")?;
    let text = v
        .get("text")
        .and_then(Value::as_str)
        .filter(|t| !t.trim().is_empty())
        .ok_or("invalid_text")?;
    let voice = v
        .get("voice")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    Ok(SynthesizeRequest {
        text: text.to_string(),
        voice,
    })
}

/// The body of a streamed `POST /v1/audio/speech`.
#[must_use]
pub fn speech_request_json(model: &str, text: &str, voice: &str) -> Vec<u8> {
    json!({
        "model": model,
        "input": text,
        "voice_id": voice,
        "response_format": "pcm",
        "stream": true,
    })
    .to_string()
    .into_bytes()
}

/// One frame: `[kind][u32 length, little-endian][payload]`.
///
/// # Panics
/// If `payload` is over 4 GiB. No caller comes near: audio goes out in
/// [`MAX_FRAME_AUDIO`] pieces, and an error is one message.
#[must_use]
pub fn frame(kind: u8, payload: &[u8]) -> Vec<u8> {
    let len = u32::try_from(payload.len()).expect("a frame payload fits in a u32");
    let mut out = Vec::with_capacity(5 + payload.len());
    out.push(kind);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Audio frames for one chunk of PCM, none over [`MAX_FRAME_AUDIO`].
pub fn audio_frames(pcm: &[u8]) -> impl Iterator<Item = Vec<u8>> + '_ {
    pcm.chunks(MAX_FRAME_AUDIO)
        .map(|chunk| frame(FRAME_AUDIO, chunk))
}

/// An `error` frame, which ends the stream.
#[must_use]
pub fn error_frame(message: &str) -> Vec<u8> {
    frame(
        FRAME_ERROR,
        json!({ "message": message }).to_string().as_bytes(),
    )
}

/// One server-sent event.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct SseEvent {
    /// The `event:` field, when the server named one.
    pub event: Option<String>,
    /// The `data:` lines, joined with `\n`.
    pub data: String,
}

/// An incremental server-sent-events reader: bytes go in as the network
/// delivers them, and whole events come out.
///
/// Mistral's lines are long: a delta's base64 runs to megabytes for a long
/// input. The search for a line's end resumes where the last one stopped, so
/// a line arriving in many reads is scanned once.
#[derive(Debug, Default)]
pub struct SseReader {
    buf: Vec<u8>,
    /// How far into `buf` holds no line end.
    scanned: usize,
    event: Option<String>,
    data: Vec<String>,
}

impl SseReader {
    /// Feed bytes and take the events they complete.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        self.buf.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some(offset) = self.buf[self.scanned..].iter().position(|&b| b == b'\n') {
            let end = self.scanned + offset;
            let mut line: Vec<u8> = self.buf.drain(..=end).collect();
            line.pop(); // the `\n`
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            self.scanned = 0;
            if let Some(event) = self.line(&String::from_utf8_lossy(&line)) {
                events.push(event);
            }
        }
        self.scanned = self.buf.len();
        events
    }

    /// Take one line; a blank one dispatches the event it ends.
    fn line(&mut self, line: &str) -> Option<SseEvent> {
        if line.is_empty() {
            if self.data.is_empty() {
                self.event = None;
                return None;
            }
            return Some(SseEvent {
                event: self.event.take(),
                data: std::mem::take(&mut self.data).join("\n"),
            });
        }
        if line.starts_with(':') {
            return None; // a comment
        }
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "event" => self.event = Some(value.to_string()),
            "data" => self.data.push(value.to_string()),
            _ => {} // `id`, `retry`, and anything later
        }
        None
    }
}

/// What one event of Mistral's speech stream means.
#[derive(Debug, PartialEq, Eq)]
pub enum SpeechEvent {
    /// PCM to pass on.
    Audio(Vec<u8>),
    /// The audio is complete.
    Done,
    /// The stream failed, with the best message it gave.
    Error(String),
    /// Anything else. Mistral may add event types, and an older backend
    /// must keep speaking, so an unknown one is skipped.
    Ignore,
}

/// Classify one event of the speech stream, by the `type` in its data, or by
/// its event name when the data has none.
#[must_use]
pub fn classify_speech_event(event: &SseEvent) -> SpeechEvent {
    let Ok(v) = serde_json::from_str::<Value>(&event.data) else {
        return SpeechEvent::Ignore;
    };
    let kind = v
        .get("type")
        .and_then(Value::as_str)
        .or(event.event.as_deref())
        .unwrap_or("");
    match kind {
        "speech.audio.delta" => {
            let Some(b64) = v.get("audio_data").and_then(Value::as_str) else {
                return SpeechEvent::Ignore;
            };
            match base64::engine::general_purpose::STANDARD.decode(b64) {
                Ok(pcm) => SpeechEvent::Audio(pcm),
                Err(_) => SpeechEvent::Error("Mistral sent audio that is not base64".to_string()),
            }
        }
        "speech.audio.done" => SpeechEvent::Done,
        "error" => SpeechEvent::Error(error_message(&v).unwrap_or_else(|| "speech failed".into())),
        _ => SpeechEvent::Ignore,
    }
}

/// The message in a Mistral error body. Its `message` is a string, or a
/// validation report whose entries each carry a `msg`, and some errors nest
/// it under `error`.
fn error_message(v: &Value) -> Option<String> {
    let v = v.get("error").filter(|e| e.is_object()).unwrap_or(v);
    match v.get("message")? {
        Value::String(s) => Some(s.clone()),
        other => {
            let msgs: Vec<&str> = other
                .get("detail")?
                .as_array()?
                .iter()
                .filter_map(|d| d.get("msg").and_then(Value::as_str))
                .collect();
            (!msgs.is_empty()).then(|| msgs.join("; "))
        }
    }
}

/// How a refused speech request is answered, before any audio: the status,
/// the contract's `message` code, and a `detail` for the user.
///
/// Mistral's `404 invalid_voice` is a voice it does not know, which the
/// contract calls `unknown_voice`. Anything else is a failed synthesis.
#[must_use]
pub fn upstream_refusal(status: u16, body: &[u8]) -> (u16, &'static str, String) {
    let v = serde_json::from_slice::<Value>(body).ok();
    let message = v
        .as_ref()
        .and_then(error_message)
        .unwrap_or_else(|| String::from_utf8_lossy(body).trim().to_string());
    let detail = format!("Mistral answered {status}: {message}");
    let is_unknown_voice =
        status == 404 && v.as_ref().and_then(|v| v.get("type")?.as_str()) == Some("invalid_voice");
    if is_unknown_voice {
        (400, "unknown_voice", detail)
    } else {
        (500, "inference_failed", detail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_needs_text_and_may_name_a_voice() {
        assert_eq!(
            parse_synthesize(br#"{"text":"Hi.","voice":"voice_3150","language":"en","speed":1.2}"#),
            Ok(SynthesizeRequest {
                text: "Hi.".to_string(),
                voice: Some("voice_3150".to_string()),
            })
        );
        assert_eq!(parse_synthesize(br#"{"text":"Hi."}"#).unwrap().voice, None);
        for bad in [&br#"{"text":"   "}"#[..], br#"{"voice":"x"}"#, b"not json"] {
            assert_eq!(parse_synthesize(bad), Err("invalid_text"));
        }
    }

    /// Mistral refuses any field it does not know, so the request carries
    /// exactly these and never a language or a rate.
    #[test]
    fn the_speech_request_asks_for_streamed_pcm() {
        let v: Value = serde_json::from_slice(&speech_request_json(
            "voxtral-mini-tts-3",
            "Hi.",
            "voice_478",
        ))
        .unwrap();
        assert_eq!(
            v,
            json!({
                "model": "voxtral-mini-tts-3",
                "input": "Hi.",
                "voice_id": "voice_478",
                "response_format": "pcm",
                "stream": true,
            })
        );
    }

    #[test]
    fn a_frame_is_kind_length_payload() {
        assert_eq!(frame(FRAME_DONE, &[]), [3, 0, 0, 0, 0]);
        assert_eq!(frame(FRAME_AUDIO, &[9, 8]), [1, 2, 0, 0, 0, 9, 8]);
        let error = error_frame("boom");
        assert_eq!(error[0], FRAME_ERROR);
        assert_eq!(&error[5..], br#"{"message":"boom"}"#);
    }

    #[test]
    fn a_long_delta_is_split_into_frames_the_daemon_takes() {
        let pcm = vec![7u8; MAX_FRAME_AUDIO * 2 + 10];
        let frames: Vec<Vec<u8>> = audio_frames(&pcm).collect();
        assert_eq!(frames.len(), 3);
        assert!(frames.iter().all(|f| f.len() <= 5 + MAX_FRAME_AUDIO));
        let rejoined: Vec<u8> = frames.iter().flat_map(|f| f[5..].to_vec()).collect();
        assert_eq!(rejoined, pcm);
    }

    #[test]
    fn events_come_out_whole_however_the_bytes_are_cut() {
        let stream = "event: speech.audio.delta\r\ndata: {\"a\":1}\r\n\r\n: comment\n\
                      event: speech.audio.done\ndata: one\ndata: two\n\n";
        let whole = SseReader::default().push(stream.as_bytes());
        assert_eq!(
            whole,
            [
                SseEvent {
                    event: Some("speech.audio.delta".into()),
                    data: "{\"a\":1}".into()
                },
                SseEvent {
                    event: Some("speech.audio.done".into()),
                    data: "one\ntwo".into()
                },
            ]
        );
        let mut reader = SseReader::default();
        let mut bytewise = Vec::new();
        for b in stream.as_bytes() {
            bytewise.extend(reader.push(std::slice::from_ref(b)));
        }
        assert_eq!(bytewise, whole);
    }

    #[test]
    fn an_event_without_data_is_not_dispatched() {
        let mut reader = SseReader::default();
        assert_eq!(reader.push(b"event: ping\n\n"), Vec::<SseEvent>::new());
        assert_eq!(
            reader.push(b"data: x\n\n"),
            [SseEvent {
                event: None,
                data: "x".into()
            }]
        );
    }

    #[test]
    fn speech_events_classify_by_type() {
        let ev = |data: &str| SseEvent {
            event: None,
            data: data.to_string(),
        };
        assert_eq!(
            classify_speech_event(&ev(
                r#"{"type":"speech.audio.delta","audio_data":"AAECAw=="}"#
            )),
            SpeechEvent::Audio(vec![0, 1, 2, 3])
        );
        assert_eq!(
            classify_speech_event(&ev(r#"{"type":"speech.audio.done","usage":{}}"#)),
            SpeechEvent::Done
        );
        assert_eq!(
            classify_speech_event(&ev(r#"{"type":"error","error":{"message":"overloaded"}}"#)),
            SpeechEvent::Error("overloaded".into())
        );
        assert!(matches!(
            classify_speech_event(&ev(r#"{"type":"speech.audio.delta","audio_data":"!!"}"#)),
            SpeechEvent::Error(_)
        ));
        assert_eq!(
            classify_speech_event(&ev(r#"{"type":"speech.something.new"}"#)),
            SpeechEvent::Ignore
        );
        assert_eq!(classify_speech_event(&ev("not json")), SpeechEvent::Ignore);
        // The event name stands in for a missing `type`.
        let named = SseEvent {
            event: Some("speech.audio.done".into()),
            data: "{}".into(),
        };
        assert_eq!(classify_speech_event(&named), SpeechEvent::Done);
    }

    #[test]
    fn a_refusal_names_the_voice_or_the_failure() {
        let (status, code, detail) = upstream_refusal(
            404,
            br#"{"object":"error","message":"Voice 'nope' not found.","type":"invalid_voice"}"#,
        );
        assert_eq!((status, code), (400, "unknown_voice"));
        assert_eq!(detail, "Mistral answered 404: Voice 'nope' not found.");

        let (status, code, detail) = upstream_refusal(
            422,
            br#"{"object":"error","message":{"detail":[{"type":"extra_forbidden","msg":"Extra inputs are not permitted"}]}}"#,
        );
        assert_eq!((status, code), (500, "inference_failed"));
        assert_eq!(
            detail,
            "Mistral answered 422: Extra inputs are not permitted"
        );

        let (_, code, detail) = upstream_refusal(502, b"Bad Gateway");
        assert_eq!(code, "inference_failed");
        assert_eq!(detail, "Mistral answered 502: Bad Gateway");
    }
}
