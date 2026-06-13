from __future__ import annotations

from cosmos_vlm.predict25.schemas import extract_task_id, validate_result_payload


def test_extract_task_id() -> None:
    payload = {"data": {"id": "abc123"}}
    assert extract_task_id(payload) == "abc123"


def test_validate_result_payload_with_string_output() -> None:
    payload = {
        "data": {
            "id": "abc123",
            "status": "completed",
            "outputs": "https://example.com/out.mp4",
            "error": "",
            "model": "wavespeed-ai/cosmos-predict-2.5/image-to-video",
        }
    }
    parsed = validate_result_payload(payload)
    assert parsed["status"] == "completed"
    assert parsed["outputs"] == ["https://example.com/out.mp4"]

