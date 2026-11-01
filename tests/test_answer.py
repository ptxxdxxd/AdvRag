import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from advrag.answer import build_request, extractive_answer, parse_response
from advrag.corpus import Chunk

CHUNKS = [
    Chunk("pep-0618#specification", "0618", "Add Optional Length-Checking To zip", "Specification",
          "When zip is called with strict=True, it raises a ValueError if the inputs have "
          "different lengths. This is checked lazily."),
    Chunk("pep-0008#indentation", "0008", "Style Guide for Python Code", "Indentation",
          "Use 4 spaces per indentation level. Continuation lines should align wrapped elements."),
]


def test_extractive_answer_cites_the_chunk_it_quotes():
    answer = extractive_answer("How do I make zip check lengths with strict?", CHUNKS)
    assert "strict=True" in answer.text
    assert answer.citations[0].chunk_id == "pep-0618#specification"
    assert "[1]" in answer.text


def test_extractive_answer_abstains_without_overlap():
    answer = extractive_answer("capital of France", CHUNKS)
    assert answer.citations == []


def test_build_request_sends_one_cited_document_per_chunk():
    req = build_request("q", CHUNKS)
    docs = req["messages"][0]["content"][:-1]
    assert [d["title"] for d in docs] == [c.id for c in CHUNKS]
    assert all(d["citations"] == {"enabled": True} for d in docs)


def _block(text, citations=None):
    return SimpleNamespace(type="text", text=text, citations=citations)


def test_parse_response_maps_citations_to_chunks():
    cite = SimpleNamespace(document_index=0, cited_text="raises a ValueError")
    bogus = SimpleNamespace(document_index=7, cited_text="not ours")
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[_block("Pass strict=True.", [cite, bogus]), _block(" That's it.")],
    )
    answer = parse_response(response, CHUNKS)
    assert answer.text == "Pass strict=True. [1] That's it."
    assert [c.chunk_id for c in answer.citations] == ["pep-0618#specification"]


def test_parse_response_handles_refusal():
    response = SimpleNamespace(stop_reason="refusal", content=[])
    assert parse_response(response, CHUNKS).citations == []


def test_request_serializes_through_the_sdk():
    """Send the request to a local fake server: checks the SDK accepts our fields.

    This does not show that the real API accepts them; nothing here calls it.
    """
    anthropic = pytest.importorskip("anthropic")
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["beta"] = self.headers.get("anthropic-beta")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps({
                "id": "msg_test", "type": "message", "role": "assistant",
                "model": "claude-opus-5-5", "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "content": [{"type": "text", "text": "Use strict=True.", "citations": [{
                    "type": "char_location", "cited_text": "strict=True", "document_index": 0,
                    "document_title": "pep-0618#specification",
                    "start_char_index": 0, "end_char_index": 11,
                }]}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = anthropic.Anthropic(
            api_key="test", base_url=f"http://127.0.0.1:{server.server_port}", max_retries=0
        )
        from advrag.answer import claude_answer

        answer = claude_answer("How do I check zip lengths?", CHUNKS, client=client)
    finally:
        server.shutdown()
    assert seen["path"].startswith("/v1/messages")
    assert "server-side-fallback-2026-07-01" in seen["beta"]
    assert seen["body"]["fallbacks"] == "default"
    assert answer.citations[0].chunk_id == "pep-0618#specification"
