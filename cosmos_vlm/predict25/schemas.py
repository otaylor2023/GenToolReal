from __future__ import annotations

from typing import Any


def extract_task_id(response_json: dict[str, Any]) -> str:
    data = response_json.get("data")
    if not isinstance(data, dict):
        raise ValueError("response missing data object")
    task_id = data.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("response missing task id at data.id")
    return task_id


def validate_result_payload(response_json: dict[str, Any]) -> dict[str, Any]:
    data = response_json.get("data")
    if not isinstance(data, dict):
        raise ValueError("result response missing data object")

    status = data.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("result response missing data.status")

    outputs = data.get("outputs")
    if outputs is None:
        outputs = []
    elif isinstance(outputs, str):
        outputs = [outputs]
    elif not isinstance(outputs, list):
        raise ValueError("data.outputs must be list, string, or null")

    return {
        "id": data.get("id"),
        "status": status,
        "outputs": outputs,
        "error": data.get("error", ""),
        "model": data.get("model", ""),
        "created_at": data.get("created_at", ""),
    }

