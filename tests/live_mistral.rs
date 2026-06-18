// SPDX-License-Identifier: GPL-3.0-only
//! Optional live test against the real Mistral API. Gated behind
//! `SUPER_STT_TEST_MISTRAL=1` + `MISTRAL_API_KEY` + a WAV at
//! `SUPER_STT_TEST_AUDIO`; self-skips otherwise (so it never runs in CI). Run:
//!   SUPER_STT_TEST_MISTRAL=1 MISTRAL_API_KEY=... \
//!   SUPER_STT_TEST_AUDIO=/path/to/mono.wav just test -- --nocapture
//! Optionally set `SUPER_STT_TEST_EXPECT` to a phrase the result must contain.
#![allow(clippy::doc_markdown)]

mod common;

use common::WasmBackend;

const SECRET: &str = "x-stt-secret-mistral_api_key";

#[tokio::test]
async fn live_mistral() {
    if std::env::var("SUPER_STT_TEST_MISTRAL").is_err() {
        return;
    }
    let Some(path) = common::component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };
    let key = std::env::var("MISTRAL_API_KEY").expect("MISTRAL_API_KEY must be set for live test");
    let audio_path =
        std::env::var("SUPER_STT_TEST_AUDIO").expect("SUPER_STT_TEST_AUDIO must point to a WAV");
    let (samples, sample_rate) = read_wav_mono_f32(&audio_path);

    let mut backend = WasmBackend::new_realtime(
        &path,
        vec!["api.mistral.ai".to_string()],
        "voxtral-mini-latest".to_string(),
        vec![(SECRET.to_string(), key)],
    )
    .expect("load backend");

    let text = backend
        .transcribe_audio(&samples, sample_rate)
        .await
        .expect("live transcription should succeed");
    println!("\n=== LIVE MISTRAL TRANSCRIPTION ===\n{text}\n==================================\n");
    assert!(
        !text.trim().is_empty(),
        "expected a non-empty transcription"
    );
    if let Ok(expect) = std::env::var("SUPER_STT_TEST_EXPECT") {
        assert!(
            text.to_lowercase().contains(&expect.to_lowercase()),
            "transcription {text:?} did not contain {expect:?}"
        );
    }
}

/// Decode a mono WAV file to f32 samples (test helper).
fn read_wav_mono_f32(path: &str) -> (Vec<f32>, u32) {
    let mut reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| f32::from(s.expect("sample")) / f32::from(i16::MAX))
            .collect(),
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.expect("sample"))
            .collect(),
    };
    (samples, spec.sample_rate)
}
