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
//! `is_stop`, `ws_url`, `session_update_json`, `audio_append_json`,
//! `preview_json`/`done_json`/`error_json`, `header`) live in the crate root so
//! they compile and unit-test on the host; this module wires them to the
//! wasm-only `super-stt:realtime` resources.

use serde_json::Value;

use super::exports::super_stt::realtime::ws_server::Guest as WsServerGuest;
use super::super_stt::realtime::ws::{self, ConsumerStream, WsError, WsFrame, WsStream};

const DEFAULT_BASE_URL: &str = "https://api.mistral.ai";
const DEFAULT_MODEL: &str = "voxtral-mini-transcribe-realtime-2602";
const COMMIT_JSON: &str = r#"{"type":"input_audio_buffer.commit"}"#;

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

    // 1. Read the consumer's `start` frame (sample_rate + optional language).
    let (sample_rate, language) = match consumer.recv()? {
        WsFrame::Text(s) => {
            let Ok(parsed) = crate::parse_start(&s) else {
                let _ = consumer.send_text(&crate::error_json("invalid start frame"));
                return Ok(());
            };
            parsed
        }
        WsFrame::Close(_) => return Ok(()), // consumer hung up before starting
        WsFrame::Binary(_) => {
            let _ = consumer.send_text(&crate::error_json("audio before start frame"));
            return Ok(());
        }
    };

    // 2. Open the upstream WS and configure the session.
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
    if let Err(e) = upstream.send_text(&crate::session_update_json(
        sample_rate,
        language.as_deref(),
    )) {
        let _ = consumer.send_text(&crate::error_json(&format!("session.update failed: {e:?}")));
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

    // 4. Commit and PHASE 2 — drain upstream transcript events.
    if let Err(e) = upstream.send_text(COMMIT_JSON) {
        let _ = consumer.send_text(&crate::error_json(&format!("commit failed: {e:?}")));
        return Ok(());
    }
    drain_upstream(&upstream, consumer);
    let _ = consumer.close();
    Ok(())
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

    if kind.ends_with("completed") {
        let transcript = v
            .get("transcript")
            .and_then(Value::as_str)
            .map_or_else(|| accumulated.trim().to_string(), str::to_string);
        let _ = consumer.send_text(&crate::done_json(&transcript));
        return true;
    }

    if kind.ends_with("delta") {
        if let Some(delta) = v.get("delta").and_then(Value::as_str) {
            accumulated.push_str(delta);
            let _ = consumer.send_text(&crate::preview_json(accumulated.trim()));
        }
        return false;
    }

    false // unknown event: ignore
}
