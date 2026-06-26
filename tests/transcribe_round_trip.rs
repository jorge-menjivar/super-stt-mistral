// SPDX-License-Identifier: GPL-3.0-only
//! Batch `/v1/transcribe` against a `wiremock` mock of the Mistral upstream:
//! proves request shaping (bearer auth + multipart model/file), egress through
//! the host allowlist, and response parsing — plus the allowlist/SSRF guards.
#![allow(clippy::doc_markdown)]

mod common;

use common::WasmBackend;
use wiremock::matchers::{header, method, path};
use wiremock::{Match, Mock, MockServer, Request, ResponseTemplate};

const SECRET: &str = "x-stt-secret-mistral_api_key";
const BASE_URL: &str = "x-stt-option-base_url";

/// A `wiremock` matcher asserting the request body contains `needle`. Decodes
/// lossily because the multipart body carries binary WAV bytes (invalid UTF-8);
/// the `language` field is plain ASCII and precedes the audio, so it survives.
struct BodyContains(&'static str);
impl Match for BodyContains {
    fn matches(&self, req: &Request) -> bool {
        String::from_utf8_lossy(&req.body).contains(self.0)
    }
}

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

/// Happy path: the component shapes the Mistral request (bearer auth + multipart
/// model/file), the host permits the allowlisted upstream, and the transcription
/// comes back.
#[tokio::test]
async fn transcribe_round_trip() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        // Proves the component forwarded the injected x-stt-secret-mistral_api_key
        // as the upstream bearer token.
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "text": "hello world"
        })))
        .mount(&server)
        .await;

    let authority = server.address().to_string();
    let mut backend = WasmBackend::new_realtime(
        &component_or_skip!(),
        vec![authority.clone()],
        "voxtral-mini-latest".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), format!("http://{authority}")),
        ],
    )
    .expect("load backend")
    // Mock upstream on loopback; the SSRF guard blocks loopback otherwise.
    .permit_loopback_egress();

    let audio = vec![0.0_f32; 1600];
    let text = backend
        .transcribe_audio(&audio, 16000)
        .await
        .expect("transcription should succeed");
    assert_eq!(text, "hello world");
}

/// A chosen `language` is forwarded to Mistral as a `language` multipart field.
#[tokio::test]
async fn language_forwarded_in_multipart() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        // The component adds a `language` form field carrying the chosen code.
        .and(BodyContains("name=\"language\"\r\n\r\nes\r\n"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "text": "hola mundo"
        })))
        .mount(&server)
        .await;

    let authority = server.address().to_string();
    let mut backend = WasmBackend::new_realtime(
        &component_or_skip!(),
        vec![authority.clone()],
        "voxtral-mini-latest".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), format!("http://{authority}")),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    let audio = vec![0.0_f32; 1600];
    let text = backend
        .transcribe_with_language(&audio, 16000, Some("es"))
        .await
        .expect("transcription should succeed");
    assert_eq!(text, "hola mundo");
}

/// The host allowlist blocks egress to a host the configuration does not permit,
/// even though a server is listening there.
#[tokio::test]
async fn allowlist_blocks_disallowed_host() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "text": "should never be reached"
        })))
        .mount(&server)
        .await;

    let mut backend = WasmBackend::new_realtime(
        &component_or_skip!(),
        // Allowlist a different host than the mock is listening on.
        vec!["api.mistral.ai".to_string()],
        "voxtral-mini-latest".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), server.uri()),
        ],
    )
    .expect("load backend");

    let result = backend.transcribe_audio(&vec![0.0_f32; 100], 16000).await;
    assert!(
        result.is_err(),
        "outbound call to a non-allowlisted host must be blocked"
    );
}

/// SSRF guard: an allowlisted *hostname* that resolves to a loopback address is
/// blocked, even though the host string itself is on the allowlist.
#[tokio::test]
async fn ssrf_blocks_hostname_resolving_to_loopback() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "text": "should never be reached"
        })))
        .mount(&server)
        .await;

    let port = server.address().port();
    let mut backend = WasmBackend::new_realtime(
        &component_or_skip!(),
        // `localhost` is allowlisted by name, but resolves to 127.0.0.1 / ::1.
        vec!["localhost".to_string()],
        "voxtral-mini-latest".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), format!("http://localhost:{port}")),
        ],
    )
    .expect("load backend");

    let result = backend.transcribe_audio(&vec![0.0_f32; 100], 16000).await;
    assert!(
        result.is_err(),
        "a hostname resolving to loopback must be blocked by the SSRF guard"
    );
}
