// SPDX-License-Identifier: GPL-3.0-only
//! CI-runnable protocol checks against the real component, no upstream needed:
//! `GET /v1/ping`, `GET /v1/status`, and the no-key `/v1/transcribe` error.
//! These exercise the batch `/v1` surface the realtime component still exports.
#![allow(clippy::doc_markdown)]

mod common;

use common::WasmBackend;

const SECRET: &str = "x-stt-secret-mistral_api_key";

/// Yield the component path or skip (CI builds it via `just build-component`).
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

/// `ping` → pong and `status` → ready (a cloud backend is ready immediately).
#[tokio::test]
async fn ping_and_status() {
    let backend = WasmBackend::new_realtime(
        &component_or_skip!(),
        Vec::new(),
        "voxtral-mini-latest".to_string(),
        Vec::new(),
    )
    .expect("load backend");

    let ping = backend.ping().await.expect("ping");
    assert_eq!(ping["status"], "success");
    assert_eq!(ping["message"], "pong");

    let status = backend.status().await.expect("status");
    assert_eq!(status["status"], "success");
    assert_eq!(status["state"], "ready");
}

/// A transcribe with no API key returns the structured `missing_secret_*` error,
/// surfaced to the daemon as a human-readable detail — no outbound call is made.
#[tokio::test]
async fn missing_secret_is_reported() {
    let mut backend = WasmBackend::new_realtime(
        &component_or_skip!(),
        Vec::new(),
        "voxtral-mini-latest".to_string(),
        // No SECRET header injected.
        vec![],
    )
    .expect("load backend");
    let _ = SECRET; // documents the header the daemon would normally inject

    let err = backend
        .transcribe_audio(&[0.0_f32; 100], 16000)
        .await
        .expect_err("a transcribe with no API key must fail");
    assert!(
        err.to_string().contains("API key not set"),
        "expected the missing-key detail; got: {err}"
    );
}
