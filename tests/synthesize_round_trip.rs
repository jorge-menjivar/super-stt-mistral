// SPDX-License-Identifier: GPL-3.0-only
//! `POST /v1/synthesize` end to end: the real component in wasmtime, against a
//! `wiremock` Mistral that answers `POST /v1/audio/speech` with server-sent
//! events, the way the live API does. The tests read the framed body the way
//! the daemon reads it.
#![allow(clippy::doc_markdown)]

mod common;

use base64::Engine as _;
use common::WasmBackend;
use serde_json::{Value, json};
use wiremock::matchers::{header, method, path};
use wiremock::{Match, Mock, MockServer, Request, ResponseTemplate};

const AUDIO: u8 = 0x01;
const DONE: u8 = 0x03;
const ERROR: u8 = 0x04;

macro_rules! component_or_skip {
    () => {
        match common::component_path() {
            Some(p) => p,
            None => {
                eprintln!("skipping: component not built (run `just build-component`)");
                return;
            }
        }
    };
}

/// Matches a request whose JSON body is exactly `want`.
struct JsonBody(Value);
impl Match for JsonBody {
    fn matches(&self, req: &Request) -> bool {
        serde_json::from_slice::<Value>(&req.body).is_ok_and(|v| v == self.0)
    }
}

/// One `speech.audio.delta` event carrying `pcm`.
fn delta(pcm: &[u8]) -> String {
    let b64 = base64::engine::general_purpose::STANDARD.encode(pcm);
    format!(
        "event: speech.audio.delta\ndata: {{\"type\":\"speech.audio.delta\",\"audio_data\":\"{b64}\"}}\n\n"
    )
}

/// The `speech.audio.done` event that ends Mistral's stream.
const DONE_EVENT: &str = "event: speech.audio.done\ndata: {\"type\":\"speech.audio.done\",\"usage\":{\"completion_tokens\":4}}\n\n";

fn sse(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(body.into_bytes(), "text/event-stream")
}

/// The backend, its upstream pointed at `server`. `key` is the API key, when
/// there is one.
fn backend(server: &MockServer, key: Option<&str>) -> Option<WasmBackend> {
    let path = common::component_path()?;
    let authority = server.address().to_string();
    let mut headers = vec![(
        "x-voice-option-base_url".to_string(),
        format!("http://{authority}"),
    )];
    if let Some(key) = key {
        headers.push((
            "x-voice-secret-mistral_api_key".to_string(),
            key.to_string(),
        ));
    }
    Some(
        WasmBackend::new_realtime(
            &path,
            vec![authority],
            "voxtral-mini-tts-3".to_string(),
            headers,
        )
        .expect("load backend")
        .permit_loopback_egress(),
    )
}

/// Mistral's audio comes back as frames, as it streams, with headers saying
/// what it is. The request asks for streamed PCM in the chosen voice and
/// carries nothing Mistral would refuse: no language and no rate.
#[tokio::test]
async fn speech_streams_back_as_audio_frames() {
    let _path = component_or_skip!();
    let server = MockServer::start().await;
    let first: Vec<u8> = (0..32u8).collect();
    let second: Vec<u8> = (100..140u8).collect();
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .and(header("authorization", "Bearer test-key"))
        .and(JsonBody(json!({
            "model": "voxtral-mini-tts-3",
            "input": "The kettle is boiling.",
            "voice_id": "voice_3150",
            "response_format": "pcm",
            "stream": true,
        })))
        .respond_with(sse(format!(
            "{}: keep-alive\n\n{}{DONE_EVENT}",
            delta(&first),
            delta(&second)
        )))
        .expect(1)
        .mount(&server)
        .await;

    let answer = backend(&server, Some("test-key"))
        .unwrap()
        .synthesize(&json!({
            "text": "The kettle is boiling.",
            "voice": "voice_3150",
            "language": "en",
            "speed": 1.25,
            "instructions": "Read this calmly.",
        }))
        .await
        .expect("synthesis runs");

    assert_eq!(answer.status, 200);
    assert_eq!(
        answer.header("content-type"),
        Some("application/vnd.super-voice.frames")
    );
    assert_eq!(answer.header("x-voice-sample-rate"), Some("24000"));
    assert_eq!(answer.header("x-voice-channels"), Some("1"));
    assert_eq!(answer.header("x-voice-format"), Some("f32le"));
    assert_eq!(
        answer.frames(),
        [(AUDIO, first), (AUDIO, second), (DONE, Vec::new())]
    );
}

/// A request naming no voice is spoken in the manifest's default.
#[tokio::test]
async fn a_request_without_a_voice_gets_the_default() {
    let _path = component_or_skip!();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .and(JsonBody(json!({
            "model": "voxtral-mini-tts-3",
            "input": "Hello.",
            "voice_id": "voice_478",
            "response_format": "pcm",
            "stream": true,
        })))
        .respond_with(sse(format!("{}{DONE_EVENT}", delta(&[1, 2, 3, 4]))))
        .expect(1)
        .mount(&server)
        .await;

    let answer = backend(&server, Some("test-key"))
        .unwrap()
        .synthesize(&json!({ "text": "Hello." }))
        .await
        .unwrap();
    assert_eq!(answer.status, 200);
    assert_eq!(answer.frames().last(), Some(&(DONE, Vec::new())));
}

/// A long input's audio arrives in deltas of megabytes. The daemon refuses a
/// frame over 1 MiB, so one delta goes out as several frames.
#[tokio::test]
async fn a_large_delta_goes_out_in_frames_the_daemon_takes() {
    let _path = component_or_skip!();
    let server = MockServer::start().await;
    let pcm: Vec<u8> = (0..1_200_000u32).map(|i| (i % 251) as u8).collect();
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(sse(format!("{}{DONE_EVENT}", delta(&pcm))))
        .mount(&server)
        .await;

    let answer = backend(&server, Some("test-key"))
        .unwrap()
        .synthesize(&json!({ "text": "A long passage." }))
        .await
        .unwrap();
    let frames = answer.frames();
    let (audio, rest) = frames.split_at(frames.len() - 1);
    assert_eq!(rest, [(DONE, Vec::new())]);
    assert!(audio.len() > 1, "one delta, several frames");
    assert!(
        audio
            .iter()
            .all(|(kind, payload)| *kind == AUDIO && payload.len() <= 64 * 1024)
    );
    let rejoined: Vec<u8> = audio.iter().flat_map(|(_, p)| p.clone()).collect();
    assert_eq!(rejoined, pcm);
}

/// A voice Mistral does not know is the contract's `unknown_voice`, refused
/// before any audio, with Mistral's own words for the user.
#[tokio::test]
async fn an_unknown_voice_is_refused_before_any_audio() {
    let _path = component_or_skip!();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "object": "error",
            "message": "Voice 'nope' not found.",
            "type": "invalid_voice",
            "code": "1902",
        })))
        .mount(&server)
        .await;

    let answer = backend(&server, Some("test-key"))
        .unwrap()
        .synthesize(&json!({ "text": "Hello.", "voice": "nope" }))
        .await
        .unwrap();
    assert_eq!(answer.status, 400);
    let body = answer.json();
    assert_eq!(body["status"], "error");
    assert_eq!(body["message"], "unknown_voice");
    assert_eq!(
        body["detail"],
        "Mistral answered 404: Voice 'nope' not found."
    );
}

/// Any other refusal is a failed synthesis, still before any audio.
#[tokio::test]
async fn a_refused_request_is_a_failed_synthesis() {
    let _path = component_or_skip!();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "object": "error",
            "message": "Unauthorized",
            "type": "invalid_request_error",
        })))
        .mount(&server)
        .await;

    let answer = backend(&server, Some("bad-key"))
        .unwrap()
        .synthesize(&json!({ "text": "Hello." }))
        .await
        .unwrap();
    assert_eq!(answer.status, 500);
    assert_eq!(answer.json()["message"], "inference_failed");
    assert_eq!(
        answer.json()["detail"],
        "Mistral answered 401: Unauthorized"
    );
}

/// A stream that stops without `speech.audio.done` is not a finished
/// utterance. The audio that came is kept, and an `error` frame says the rest
/// did not, since the `200` has been sent by then.
#[tokio::test]
async fn speech_that_stops_early_ends_with_an_error_frame() {
    let _path = component_or_skip!();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(sse(delta(&[9, 9, 9, 9])))
        .mount(&server)
        .await;

    let answer = backend(&server, Some("test-key"))
        .unwrap()
        .synthesize(&json!({ "text": "Hello." }))
        .await
        .unwrap();
    assert_eq!(answer.status, 200);
    let frames = answer.frames();
    assert_eq!(frames[0], (AUDIO, vec![9, 9, 9, 9]));
    let (kind, payload) = &frames[1];
    assert_eq!(*kind, ERROR);
    let message: Value = serde_json::from_slice(payload).unwrap();
    assert_eq!(
        message["message"],
        "Mistral's speech ended before it was done"
    );
    assert_eq!(frames.len(), 2, "nothing after the error frame");
}

/// An `error` event in the stream ends it the same way, with its message.
#[tokio::test]
async fn an_error_event_ends_the_stream_with_its_message() {
    let _path = component_or_skip!();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(sse(format!(
            "{}event: error\ndata: {{\"type\":\"error\",\"error\":{{\"message\":\"overloaded\"}}}}\n\n",
            delta(&[1, 1, 1, 1])
        )))
        .mount(&server)
        .await;

    let answer = backend(&server, Some("test-key"))
        .unwrap()
        .synthesize(&json!({ "text": "Hello." }))
        .await
        .unwrap();
    let frames = answer.frames();
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[1].0, ERROR);
    let message: Value = serde_json::from_slice(&frames[1].1).unwrap();
    assert_eq!(message["message"], "Mistral: overloaded");
}

/// Blank text is the contract's `invalid_text`, and Mistral is never asked.
#[tokio::test]
async fn blank_text_is_refused_without_asking_mistral() {
    let _path = component_or_skip!();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&server)
        .await;

    let answer = backend(&server, Some("test-key"))
        .unwrap()
        .synthesize(&json!({ "text": "   " }))
        .await
        .unwrap();
    assert_eq!(answer.status, 400);
    assert_eq!(answer.json()["message"], "invalid_text");
}

/// Without a key the user is told where to set it, and Mistral is never asked.
#[tokio::test]
async fn a_missing_key_is_reported_without_asking_mistral() {
    let _path = component_or_skip!();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&server)
        .await;

    let answer = backend(&server, None)
        .unwrap()
        .synthesize(&json!({ "text": "Hello." }))
        .await
        .unwrap();
    assert_eq!(answer.status, 400);
    assert_eq!(answer.json()["message"], "missing_secret_mistral_api_key");
}
