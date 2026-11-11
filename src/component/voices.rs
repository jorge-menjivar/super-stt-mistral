// SPDX-License-Identifier: GPL-3.0-only
//! Cloned voices: `POST /v1/voices` and `DELETE /v1/voices/{voice}`.
//!
//! Mistral clones from a reference clip sent with each request and keeps
//! nothing of it, so the backend keeps the clip, from its registration to the
//! syntheses in that voice. The component is instantiated afresh for every
//! request, so it keeps the clip on disk: in its private `/tmp`, which only it
//! sees, which is memory-backed, and which the daemon removes when the model
//! unloads. The daemon registers each voice again after every load.

use std::io::ErrorKind;

use super::wasi::http::types::IncomingRequest;
use crate::speech;

/// `POST /v1/voices`: keep a cloned voice's clip for the syntheses in it.
pub(super) fn register(request: &IncomingRequest) -> (u16, Vec<u8>) {
    let Ok(body) = request.consume() else {
        return super::err(400, "no_body");
    };
    let raw = match super::read_all(body) {
        Ok(raw) => raw,
        Err(e) => return super::err(500, &e),
    };
    let registration = match speech::parse_registration(&raw) {
        Ok(registration) => registration,
        Err(code) => return super::err(400, code),
    };
    match keep(&registration) {
        Ok(()) => success(),
        Err(e) => failure(&format!("Keeping the voice's clip failed: {e}")),
    }
}

/// Write the clip whole or not at all: to a temporary name, then renamed, so a
/// synthesis never reads half a clip.
fn keep(registration: &speech::Registration) -> std::io::Result<()> {
    std::fs::create_dir_all(speech::VOICES_DIR)?;
    let part = format!("{}.part", registration.file);
    std::fs::write(&part, &registration.wav)?;
    std::fs::rename(&part, &registration.file)
}

/// `DELETE /v1/voices/{voice}`, the id percent-encoded: forget a voice. A
/// voice never registered is a `404`, which the daemon takes as done.
pub(super) fn unregister(encoded: &str) -> (u16, Vec<u8>) {
    let Some(file) = speech::percent_decode(encoded)
        .as_deref()
        .and_then(speech::clip_file)
    else {
        return super::err(400, "invalid_voice");
    };
    match std::fs::remove_file(&file) {
        Ok(()) => success(),
        Err(e) if e.kind() == ErrorKind::NotFound => super::err(404, "unknown_voice"),
        Err(e) => failure(&format!("Forgetting the voice's clip failed: {e}")),
    }
}

/// The clip of a registered cloned voice, or `None` for one this loaded model
/// was never given.
pub(super) fn clip(voice: &str) -> Option<Vec<u8>> {
    std::fs::read(speech::clip_file(voice)?).ok()
}

fn success() -> (u16, Vec<u8>) {
    (
        200,
        super::to_vec(&serde_json::json!({ "status": "success" })),
    )
}

fn failure(detail: &str) -> (u16, Vec<u8>) {
    (
        500,
        super::to_vec(&serde_json::json!({
            "status": "error", "message": "voice_store_failed", "detail": detail
        })),
    )
}
