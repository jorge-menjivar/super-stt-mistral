// SPDX-License-Identifier: GPL-3.0-only
//! The wasm-only `wasi:http` / `super-engine:realtime` component. Compiled for
//! `wasm32-wasip2` only (the `wit-bindgen` bindings don't build for the host);
//! the pure helpers it calls live in the crate root and are unit-tested there.

// wit-bindgen's generated `Vec::from_raw_parts` glue trips this pedantic lint.
#![allow(clippy::same_length_and_capacity)]

wit_bindgen::generate!({
    path: "wit",
    world: "realtime-backend",
    generate_all,
    // The bundled WASI 0.2.0 dep WITs gate a handful of interfaces behind
    // `@unstable` feature flags (e.g. `wasi:clocks/timezone`). wit-bindgen
    // parses every file in `wit/`, so those gates must be enabled for the
    // directory to resolve even though `realtime-backend` does not use them.
    features: [
        "clocks-timezone",
    ],
});

use exports::wasi::http::incoming_handler::Guest;
use wasi::http::types::{
    Fields, IncomingBody, IncomingRequest, IncomingResponse, Method, OutgoingBody, OutgoingRequest,
    OutgoingResponse, ResponseOutparam, Scheme,
};
use wasi::io::streams::StreamError;

mod realtime;
mod synthesize;
mod voices;

/// Mistral's API, unless a test points the component elsewhere.
const DEFAULT_BASE_URL: &str = "https://api.mistral.ai";

struct Component;

impl Guest for Component {
    fn handle(request: IncomingRequest, outparam: ResponseOutparam) {
        // Speech is the one route whose answer is a stream rather than one
        // JSON body, so it writes its own response.
        if matches!(request.method(), Method::Post) && path(&request) == "/v1/synthesize" {
            synthesize::handle(&request, outparam);
            return;
        }
        let (status, body) = route(&request);
        send_response(outparam, status, &body);
    }
}

export!(Component);

/// The request's path, without its query.
fn path(request: &IncomingRequest) -> String {
    let full = request.path_with_query().unwrap_or_default();
    full.split('?').next().unwrap_or("").to_string()
}

/// Dispatch a `/v1` request to its handler, returning `(status, json_bytes)`.
fn route(request: &IncomingRequest) -> (u16, Vec<u8>) {
    let method = request.method();
    let path = path(request);

    match (&method, path.as_str()) {
        (Method::Get, "/v1/ping") => ok(&serde_json::json!({
            "status": "success", "message": "pong"
        })),
        (Method::Get, "/v1/status") => ok(&serde_json::json!({
            "status": "success", "state": "ready", "device": "cpu"
        })),
        (Method::Post, "/v1/load") => (
            202,
            to_vec(&serde_json::json!({ "status": "success", "message": "Loading started" })),
        ),
        (Method::Post, "/v1/cancel") => ok(&serde_json::json!({
            "status": "success", "message": "Cancelled"
        })),
        (Method::Post, "/v1/transcribe") => transcribe(request),
        (Method::Post, "/v1/voices") => voices::register(request),
        (Method::Delete, p) if p.starts_with("/v1/voices/") => {
            voices::unregister(&p["/v1/voices/".len()..])
        }
        _ => err(404, "not_found"),
    }
}

/// Handle `POST /v1/transcribe`: read the injected secret/option headers and
/// the audio body, forward to Mistral, and return the transcription.
fn transcribe(request: &IncomingRequest) -> (u16, Vec<u8>) {
    let entries = request.headers().entries();
    let Some(api_key) = crate::header(&entries, "x-voice-secret-mistral_api_key") else {
        return missing_key();
    };
    let base_url = base_url(&entries);
    let model = crate::header(&entries, "x-voice-model")
        .unwrap_or_else(|| "voxtral-mini-transcribe-3".to_string());

    let Ok(body) = request.consume() else {
        return err(400, "no_body");
    };
    let raw = match read_all(body) {
        Ok(r) => r,
        Err(e) => return err(500, &e),
    };
    let req: serde_json::Value = match serde_json::from_slice(&raw) {
        Ok(v) => v,
        Err(_) => return err(400, "invalid_json"),
    };

    let Some(audio) = req.get("audio_data").and_then(|v| v.as_array()) else {
        return err(400, "invalid_audio");
    };
    let audio: Vec<f32> = audio
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
        .collect();
    let sample_rate = u32::try_from(
        req.get("sample_rate")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(16000),
    )
    .unwrap_or(16000);
    // Mistral's transcription API auto-detects when `language` is omitted, so the
    // reserved `auto` (and a missing field) map to "no language"; a specific code
    // is forwarded.
    let language = match req.get("language").and_then(|v| v.as_str()) {
        Some("auto") | None => None,
        Some(code) => Some(code),
    };

    // The user's dictation context, as `context_bias` entries. Read from the
    // headers rather than the body: it is the daemon's, not the caller's.
    let context_bias = crate::context_bias(
        crate::header(&entries, "x-voice-prompt").as_deref(),
        crate::header(&entries, "x-voice-vocabulary").as_deref(),
    );

    match call_mistral(
        &base_url,
        &api_key,
        &model,
        language,
        &context_bias,
        &audio,
        sample_rate,
    ) {
        Ok(text) => (
            200,
            to_vec(&serde_json::json!({ "status": "success", "transcription": text })),
        ),
        Err(detail) => (
            502,
            to_vec(&serde_json::json!({
                "status": "error", "message": "upstream_error", "detail": detail
            })),
        ),
    }
}

/// Send the audio to Mistral's transcription API and return the text.
///
/// A request carrying a context that Mistral answers with `400` is sent again
/// without it. Mistral refuses the whole request over its context, when the
/// audio and the context together run past the model's window, for one, and a
/// context it will not take should cost the context, not the recording.
fn call_mistral(
    base_url: &str,
    api_key: &str,
    model: &str,
    language: Option<&str>,
    context_bias: &[String],
    audio: &[f32],
    sample_rate: u32,
) -> Result<String, String> {
    let wav = crate::encode_wav(audio, sample_rate);
    let send = |context_bias: &[String]| {
        post_transcription(base_url, api_key, model, language, context_bias, &wav)
    };
    let (mut status, mut bytes) = send(context_bias)?;
    if status == 400 && !context_bias.is_empty() {
        (status, bytes) = send(&[])?;
    }
    if !(200..300).contains(&status) {
        return Err(format!(
            "status {status}: {}",
            String::from_utf8_lossy(&bytes)
        ));
    }
    crate::parse_transcript(&bytes)
}

/// The answer to a request arriving without the API key, with a
/// human-readable `detail` that the daemon surfaces to the user.
fn missing_key() -> (u16, Vec<u8>) {
    (
        400,
        to_vec(&serde_json::json!({
            "status": "error",
            "message": "missing_secret_mistral_api_key",
            "detail": "Mistral API key not set. Add it in Settings \u{2192} Models \u{2192} Mistral.",
        })),
    )
}

/// Where Mistral is.
///
/// Internal test seam: `base_url` is not a declared manifest option, so the
/// daemon never sends this header in production — the default is always used.
/// The round-trip tests inject it to reach a mock upstream.
fn base_url(entries: &[(String, Vec<u8>)]) -> String {
    crate::header(entries, "x-voice-option-base_url")
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string())
}

/// One `POST /v1/audio/transcriptions`, answered with its status and body.
fn post_transcription(
    base_url: &str,
    api_key: &str,
    model: &str,
    language: Option<&str>,
    context_bias: &[String],
    wav: &[u8],
) -> Result<(u16, Vec<u8>), String> {
    let boundary = "----superstt7MA4YWxkTrZu0gW";
    let multipart = crate::build_multipart(boundary, model, language, context_bias, wav);
    let response = post(
        base_url,
        api_key,
        "/v1/audio/transcriptions",
        &format!("multipart/form-data; boundary={boundary}"),
        &multipart,
    )?;
    let status = response.status();
    let body = response.consume().map_err(|()| "response_consume")?;
    Ok((status, read_all(body)?))
}

/// Send one `POST` to Mistral and wait for the head of its answer. The body
/// is the caller's to read, all at once or as it streams.
fn post(
    base_url: &str,
    api_key: &str,
    path: &str,
    content_type: &str,
    body: &[u8],
) -> Result<IncomingResponse, String> {
    let (is_https, authority) = crate::parse_base(base_url);
    let scheme = if is_https {
        Scheme::Https
    } else {
        Scheme::Http
    };

    let headers = Fields::new();
    headers
        .append("authorization", format!("Bearer {api_key}").as_bytes())
        .map_err(|e| format!("header: {e:?}"))?;
    headers
        .append("content-type", content_type.as_bytes())
        .map_err(|e| format!("header: {e:?}"))?;

    let request = OutgoingRequest::new(headers);
    request
        .set_method(&Method::Post)
        .map_err(|()| "set_method")?;
    request
        .set_scheme(Some(&scheme))
        .map_err(|()| "set_scheme")?;
    request
        .set_authority(Some(&authority))
        .map_err(|()| "set_authority")?;
    request
        .set_path_with_query(Some(path))
        .map_err(|()| "set_path")?;

    // Obtain the body handle, start the request, then stream the body — the
    // canonical wasi:http outbound order.
    let out_body = request.body().map_err(|()| "request_body")?;
    let future = wasi::http::outgoing_handler::handle(request, None)
        .map_err(|e| format!("handle: {e:?}"))?;
    write_all(&out_body, body)?;
    OutgoingBody::finish(out_body, None).map_err(|e| format!("finish: {e:?}"))?;

    let pollable = future.subscribe();
    pollable.block();
    future
        .get()
        .ok_or("no_response")?
        .map_err(|()| "future_taken")?
        .map_err(|e| format!("http: {e:?}"))
}

// ── helpers ───────────────────────────────────────────────────────────────

fn ok(value: &serde_json::Value) -> (u16, Vec<u8>) {
    (200, to_vec(value))
}

fn err(status: u16, message: &str) -> (u16, Vec<u8>) {
    (
        status,
        to_vec(&serde_json::json!({ "status": "error", "message": message })),
    )
}

fn to_vec(value: &serde_json::Value) -> Vec<u8> {
    serde_json::to_vec(value).unwrap_or_default()
}

/// Drain an incoming body to bytes.
fn read_all(body: IncomingBody) -> Result<Vec<u8>, String> {
    let stream = body.stream().map_err(|()| "no_stream".to_string())?;
    let mut out = Vec::new();
    loop {
        match stream.blocking_read(65536) {
            Ok(chunk) => out.extend_from_slice(&chunk),
            Err(StreamError::Closed) => break,
            Err(StreamError::LastOperationFailed(_)) => return Err("read_failed".to_string()),
        }
    }
    drop(stream);
    let _ = IncomingBody::finish(body);
    Ok(out)
}

/// Write all bytes to an outgoing body in ≤4096-byte flushes.
fn write_all(body: &OutgoingBody, data: &[u8]) -> Result<(), String> {
    let stream = body.write().map_err(|()| "write_stream".to_string())?;
    for chunk in data.chunks(4096) {
        stream
            .blocking_write_and_flush(chunk)
            .map_err(|_| "write_failed".to_string())?;
    }
    drop(stream);
    Ok(())
}

/// Build the response and hand it to the outparam.
fn send_response(outparam: ResponseOutparam, status: u16, body_bytes: &[u8]) {
    let headers = Fields::new();
    let _ = headers.append("content-type", b"application/json");
    let response = OutgoingResponse::new(headers);
    let _ = response.set_status_code(status);
    let Ok(body) = response.body() else {
        ResponseOutparam::set(outparam, Ok(response));
        return;
    };
    ResponseOutparam::set(outparam, Ok(response));
    if let Ok(stream) = body.write() {
        for chunk in body_bytes.chunks(4096) {
            let _ = stream.blocking_write_and_flush(chunk);
        }
        drop(stream);
    }
    let _ = OutgoingBody::finish(body, None);
}
