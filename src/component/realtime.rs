// SPDX-License-Identifier: GPL-3.0-only
//! Mistral realtime WebSocket transcription bridge.
//!
//! Bridges a consumer WebSocket session to Mistral's realtime transcription
//! API (`wss://api.mistral.ai/v1/audio/transcriptions/realtime`).
//!
//! ## Half-duplex limitation
//! The host does not yet implement `wasi:io/poll` for the WS resources
//! (`subscribe` traps), so this guest cannot wait on the consumer and the
//! upstream at the same time. It therefore runs half-duplex: it forwards ALL
//! consumer audio to the upstream first, then drains the upstream's
//! transcript events. Mistral's incremental `delta` events buffer host-side
//! during the audio phase and are delivered to the consumer in the finalize
//! phase (so previews arrive in a burst near the end rather than live).
//! Implementing host-side `subscribe` would restore true streaming.
//!
//! The pure frame-parsing and payload-building helpers (`parse_start`,
//! `is_stop`, `ws_url`, `audio_append_json`,
//! `preview_json`/`done_json`/`error_json`, `header`) live in the crate root so
//! they compile and unit-test on the host; this module wires them to the
//! wasm-only `super-stt:realtime` resources.

use serde_json::Value;

use super::exports::super_stt::realtime::ws_server::Guest as WsServerGuest;
use super::super_stt::realtime::ws::{self, ConsumerStream, WsError, WsFrame, WsStream};

const DEFAULT_BASE_URL: &str = "https://api.mistral.ai";
const DEFAULT_MODEL: &str = "voxtral-mini-transcribe-realtime-2602";
const INPUT_AUDIO_FLUSH: &str = r#"{"type":"input_audio.flush"}"#;
const INPUT_AUDIO_END: &str = r#"{"type":"input_audio.end"}"#;

impl WsServerGuest for super::Component {
    fn handle(headers: Vec<(String, Vec<u8>)>, consumer: ConsumerStream) -> Result<(), WsError> {
        run(&headers, &consumer)
    }
}

fn run(headers: &[(String, Vec<u8>)], consumer: &ConsumerStream) -> Result<(), WsError> {
    let Some(api_key) = crate::header(headers, "x-stt-secret-mistral_api_key") else {
        let _ = consumer.send_text(&crate::error_json("missing mistral api key"));
        return Ok(());
    };
    let base_url = crate::header(headers, "x-stt-option-base_url")
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
    let model = crate::header(headers, "x-stt-model").unwrap_or_else(|| DEFAULT_MODEL.to_string());

    // 1. Read and validate the consumer's `start` frame. Mistral needs no
    //    session config — the model is in the URL and PCM s16le/16 kHz is
    //    assumed — so the frame's fields are not forwarded upstream.
    match consumer.recv()? {
        WsFrame::Text(s) if crate::parse_start(&s).is_ok() => {}
        WsFrame::Text(_) => {
            let _ = consumer.send_text(&crate::error_json("invalid start frame"));
            return Ok(());
        }
        WsFrame::Close(_) => return Ok(()), // consumer hung up before starting
        WsFrame::Binary(_) => {
            let _ = consumer.send_text(&crate::error_json("audio before start frame"));
            return Ok(());
        }
    }

    // 2. Open the upstream WS and wait for `session.created` before streaming.
    let url = crate::ws_url(&base_url, &model);
    let upstream = match ws::connect(
        &url,
        &[(
            "authorization".to_string(),
            format!("Bearer {api_key}").into_bytes(),
        )],
    ) {
        Ok(u) => u,
        Err(e) => {
            let _ = consumer.send_text(&crate::error_json(&format!(
                "upstream connect failed: {e:?}"
            )));
            return Ok(());
        }
    };
    if !await_session_created(&upstream, consumer) {
        return Ok(());
    }

    // 3. PHASE 1 — forward all consumer audio to the upstream.
    loop {
        match consumer.recv()? {
            WsFrame::Binary(pcm) => {
                if let Err(e) = upstream.send_text(&crate::audio_append_json(&pcm)) {
                    let _ = consumer
                        .send_text(&crate::error_json(&format!("upstream send failed: {e:?}")));
                    return Ok(());
                }
            }
            WsFrame::Text(s) if crate::is_stop(&s) => break,
            WsFrame::Text(_) => {}      // ignore unknown control frames
            WsFrame::Close(_) => break, // consumer done; finalize what we have
        }
    }

    // 4. Flush + end the input, then PHASE 2 — drain transcript events.
    for msg in [INPUT_AUDIO_FLUSH, INPUT_AUDIO_END] {
        if let Err(e) = upstream.send_text(msg) {
            let _ = consumer.send_text(&crate::error_json(&format!("flush/end failed: {e:?}")));
            return Ok(());
        }
    }
    drain_upstream(&upstream, consumer);
    let _ = consumer.close();
    Ok(())
}

/// Read upstream until Mistral's `session.created` handshake event. Returns
/// `false` (after notifying the consumer) if the upstream errors or closes
/// before the session is ready.
fn await_session_created(upstream: &WsStream, consumer: &ConsumerStream) -> bool {
    loop {
        match upstream.recv() {
            Ok(WsFrame::Text(s)) => match event_type(&s).as_deref() {
                Some("session.created") => return true,
                Some("error") => {
                    let _ = consumer.send_text(&crate::error_json(&format!("upstream error: {s}")));
                    return false;
                }
                _ => {} // ignore other handshake chatter
            },
            Ok(WsFrame::Binary(_)) => {}
            Ok(WsFrame::Close(_)) | Err(_) => {
                let _ = consumer.send_text(&crate::error_json("upstream closed during handshake"));
                return false;
            }
        }
    }
}

/// The `type` field of a JSON event, if present.
fn event_type(s: &str) -> Option<String> {
    serde_json::from_str::<Value>(s)
        .ok()
        .and_then(|v| v.get("type").and_then(Value::as_str).map(str::to_string))
}

/// PHASE 2 — read upstream transcript events until completion/close.
fn drain_upstream(upstream: &WsStream, consumer: &ConsumerStream) {
    let mut accumulated = String::new();
    loop {
        match upstream.recv() {
            Ok(WsFrame::Text(s)) => {
                if handle_upstream_event(&s, consumer, &mut accumulated) {
                    break; // completed
                }
            }
            Ok(WsFrame::Binary(_)) => {} // Mistral sends JSON text; ignore binary
            Ok(WsFrame::Close(_)) | Err(WsError::Closed) => {
                // Upstream closed without a completed event: emit what we have.
                let _ = consumer.send_text(&crate::done_json(accumulated.trim()));
                break;
            }
            Err(e) => {
                let _ =
                    consumer.send_text(&crate::error_json(&format!("upstream recv failed: {e:?}")));
                break;
            }
        }
    }
}

/// Handle one upstream JSON event. Returns `true` when the session is complete
/// (a `completed`/`error` event), `false` to keep draining.
fn handle_upstream_event(s: &str, consumer: &ConsumerStream, accumulated: &mut String) -> bool {
    let Ok(v) = serde_json::from_str::<Value>(s) else {
        return false; // ignore non-JSON frames
    };
    let kind = v.get("type").and_then(Value::as_str).unwrap_or("");

    if kind == "error" {
        let msg = v
            .get("error")
            .and_then(|e| e.get("message"))
            .or_else(|| v.get("message"))
            .and_then(Value::as_str)
            .unwrap_or("upstream error");
        let _ = consumer.send_text(&crate::error_json(msg));
        return true;
    }

    if kind == "transcription.done" {
        let transcript = v
            .get("text")
            .and_then(Value::as_str)
            .map_or_else(|| accumulated.trim().to_string(), str::to_string);
        let _ = consumer.send_text(&crate::done_json(&transcript));
        return true;
    }

    if kind == "transcription.text.delta" {
        if let Some(delta) = v.get("text").and_then(Value::as_str) {
            accumulated.push_str(delta);
            let _ = consumer.send_text(&crate::preview_json(accumulated.trim()));
        }
        return false;
    }

    false // unknown event: ignore
}
