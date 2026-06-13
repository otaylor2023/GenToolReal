"""Chunked mega-payload instruction expansion for dataset shards.

Pipeline:
1) Export datapoints from scene shards into chunked request files
2) Call Gemini per chunk with retry/backoff (optional recursive split on repeated failure)
3) Validate responses and merge instructions back into scene shards
4) Emit per-chunk merge reports + retry queue files
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from google import genai
from google.genai import errors as genai_errors
from google.genai import types


SCHEMA_VERSION = "instruction_batch_uid_v1"
DEFAULT_INSTRUCTION_MODEL = "gemini-2.5-flash"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_RUN_MAX_WORKERS = 8


@dataclass
class RetryConfig:
    max_attempts: int = 6
    base_seconds: float = 1.5
    max_seconds: float = 60.0
    jitter_seconds: float = 0.6


def _load_api_key_from_repo_env() -> str:
    repo_root = Path(__file__).resolve().parent.parent.parent
    sidecar_dir = repo_root / "vlm_sidecar"
    if str(sidecar_dir) not in sys.path:
        sys.path.insert(0, str(sidecar_dir))
    from gemini_backend import _hydrate_gemini_env_from_dotenv, resolve_gemini_api_key

    _hydrate_gemini_env_from_dotenv()
    key = resolve_gemini_api_key()
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY/GOOGLE_API_KEY in .env")
    return key


def _extract_json_blob(text: str) -> Dict[str, Any]:
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    start = s.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model response")
    for end in range(len(s), start, -1):
        try:
            obj = json.loads(s[start:end])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("Failed to parse JSON from model response")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_object_name(name: str) -> str:
    s = str(name).strip().lower().replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _object_name_from_object_id(object_id: str) -> str:
    s = str(object_id).strip().lower()
    if not s:
        return ""
    s = re.sub(r"[_\-\s]*\d+$", "", s)
    return _normalize_object_name(s)


def _request_prompt(chunk: Dict[str, Any]) -> str:
    items = [
        {
            "datapoint_uid": d["datapoint_uid"],
            "relation_string": d["relation_string"],
        }
        for d in chunk["datapoints"]
    ]
    return f"""Generate instruction variants for each input item.

Return ONLY valid JSON in this exact format:
{{
  "schema_version": "{SCHEMA_VERSION}",
  "results": [
    {{
      "datapoint_uid": "scene_0001_dp_000123",
      "instructions": ["...", "...", "...", "..."]
    }}
  ]
}}

Rules:
- Produce exactly one result for each input item.
- Preserve and echo datapoint_uid exactly.
- Each instructions array must have exactly 4 concise imperative instructions.
- Keep meaning consistent with relation_string.
- Do not introduce new objects, targets, or relation changes.
- Use moderate wording variation and avoid near-duplicates.
- Prefer varying the leading verb and sentence opening across results for different datapoints.
- Avoid overly repetitive templates (for example, do not start most instructions with the same verb).
- Do not include any fields other than datapoint_uid and instructions in results.

Input items JSON:
{json.dumps({"schema_version": SCHEMA_VERSION, "items": items}, indent=2)}
"""


def export_chunks(
    *,
    dataset_dir: Path,
    output_root: Path,
    run_id: str,
    chunk_size: int,
    instruction_count_per_datapoint: int = 4,
    style: str = "moderate",
) -> List[Path]:
    requests_dir = output_root / "requests"
    responses_dir = output_root / "responses"
    logs_dir = output_root / "logs"
    for d in (requests_dir, responses_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(dataset_dir.glob("*_position_dataset_v0_1.json"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in {dataset_dir}")

    all_items: List[Dict[str, Any]] = []
    for shard_path in shard_paths:
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        scene_id = str(shard["scene_id"])
        keypoint_map = shard.get("keypoints", {})

        def phrase_for_kp(kp_id: str) -> str:
            kp = keypoint_map.get(kp_id, {})
            label = str(kp.get("label", kp_id)).strip()
            obj_name = str(kp.get("object_name", "")).strip()
            if not obj_name:
                obj_name = _object_name_from_object_id(str(kp.get("object_id", "")))
            if obj_name:
                return f"{label} of {_normalize_object_name(obj_name)}"
            return label

        for idx, dp in enumerate(shard.get("datapoints", [])):
            tool_id = str(dp.get("tool_keypoint_id", ""))
            ref_ids = [str(x) for x in dp.get("ref_keypoint_ids", [])]
            movement = str(dp.get("movement_token", "")).strip()
            parts = []
            labels = []
            if tool_id:
                tool_phrase = phrase_for_kp(tool_id)
                parts.append(f"[{tool_phrase}]")
                labels.append(tool_phrase)
            if movement:
                parts.append(f"[{movement}]")
            for rid in ref_ids:
                ref_phrase = phrase_for_kp(rid)
                parts.append(f"[{ref_phrase}]")
                labels.append(ref_phrase)
            rel = " ".join(parts).strip()
            if not rel:
                rel = str(dp.get("relation_string", "")).strip()
                if rel:
                    tokens = [s.strip() for s in re.findall(r"\[([^\]]+)\]", rel)]
                    mv = movement.lower()
                    labels = [t for t in tokens if t.lower() != mv]
            uid = f"{scene_id}_dp_{idx:06d}"
            all_items.append(
                {
                    "index": -1,  # set per chunk
                    "datapoint_uid": uid,
                    "scene_id": scene_id,
                    "movement_token": str(dp.get("movement_token", "")),
                    "relation_string": rel,
                    "labels": labels,
                    "source_shard": str(shard_path),
                    "source_datapoint_index": idx,
                }
            )

    req_paths: List[Path] = []
    for chunk_i, start in enumerate(range(0, len(all_items), chunk_size), start=1):
        items = all_items[start : start + chunk_size]
        for i, it in enumerate(items):
            it["index"] = i
        chunk_id = f"chunk_{chunk_i:04d}"
        request_obj = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "chunk_id": chunk_id,
            "chunk_index": chunk_i,
            "chunk_size": len(items),
            "instruction_count_per_datapoint": instruction_count_per_datapoint,
            "style": style,
            "constraints": {
                "imperative": True,
                "preserve_entities": True,
                "no_new_objects": True,
                "no_relation_change": True,
            },
            "datapoints": items,
            "items": [
                {
                    "datapoint_uid": it["datapoint_uid"],
                    "relation_string": it["relation_string"],
                }
                for it in items
            ],
        }
        out = requests_dir / f"{chunk_id}.json"
        out.write_text(json.dumps(request_obj, indent=2), encoding="utf-8")
        req_paths.append(out)
    return req_paths


def _call_chunk_once(client: genai.Client, model: str, chunk: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _request_prompt(chunk)
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_text(text=prompt)],
        config=types.GenerateContentConfig(
            temperature=0.4,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
        ),
    )
    parsed = _extract_json_blob(resp.text or "")
    return parsed


def _is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, genai_errors.ClientError):
        txt = str(exc).lower()
        return "429" in txt or "resource_exhausted" in txt or "quota" in txt
    return False


def _retry_delay_seconds(exc: Exception, attempt: int, cfg: RetryConfig) -> float:
    txt = str(exc)
    m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", txt.lower())
    if m:
        return min(float(m.group(1)) + random.uniform(0, cfg.jitter_seconds), cfg.max_seconds)
    base = min(cfg.base_seconds * (2 ** max(0, attempt - 1)), cfg.max_seconds)
    return base + random.uniform(0, cfg.jitter_seconds)


def call_chunk_with_retry(
    *,
    client: genai.Client,
    model: str,
    chunk: Dict[str, Any],
    retry: RetryConfig,
) -> Dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, retry.max_attempts + 1):
        try:
            return _call_chunk_once(client, model, chunk)
        except Exception as exc:
            last_exc = exc
            if not _is_rate_limit_error(exc):
                raise
            if attempt >= retry.max_attempts:
                break
            time.sleep(_retry_delay_seconds(exc, attempt, retry))
    assert last_exc is not None
    raise last_exc


def _split_chunk_obj(chunk: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dps = chunk["datapoints"]
    mid = len(dps) // 2
    left_items = dps[:mid]
    right_items = dps[mid:]
    if not left_items or not right_items:
        raise ValueError("Cannot split chunk with <2 datapoints")

    def mk(base_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = dict(chunk)
        out["chunk_id"] = base_id
        out["chunk_size"] = len(items)
        out["datapoints"] = []
        for i, it in enumerate(items):
            nit = dict(it)
            nit["index"] = i
            out["datapoints"].append(nit)
        return out

    left = mk(f"{chunk['chunk_id']}_a", left_items)
    right = mk(f"{chunk['chunk_id']}_b", right_items)
    return left, right


def _write_response(path: Path, response_obj: Dict[str, Any], raw_error: str | None = None) -> None:
    payload = {"received_at": _now_iso(), "response": response_obj}
    if raw_error is not None:
        payload["error"] = raw_error
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def process_request_file(
    *,
    request_file: Path,
    output_root: Path,
    model: str,
    retry: RetryConfig,
    split_on_exhaust: bool = True,
) -> List[Path]:
    client = genai.Client(api_key=_load_api_key_from_repo_env())
    req = json.loads(request_file.read_text(encoding="utf-8"))
    responses_dir = output_root / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    def run_chunk_obj(chunk: Dict[str, Any]) -> List[Path]:
        resp_path = responses_dir / f"{chunk['chunk_id']}.response.json"
        try:
            result = call_chunk_with_retry(client=client, model=model, chunk=chunk, retry=retry)
            _write_response(resp_path, result)
            return [resp_path]
        except Exception as exc:
            if split_on_exhaust and len(chunk["datapoints"]) > 1 and _is_rate_limit_error(exc):
                left, right = _split_chunk_obj(chunk)
                return run_chunk_obj(left) + run_chunk_obj(right)
            _write_response(resp_path, {}, raw_error=str(exc))
            return [resp_path]

    return run_chunk_obj(req)


def validate_and_merge(
    *,
    request_file: Path,
    response_files: Iterable[Path],
    output_root: Path,
) -> Dict[str, Any]:
    req = json.loads(request_file.read_text(encoding="utf-8"))
    # Support both legacy request shape (datapoints with merge metadata)
    # and minimal wire request shape (items only).
    if "datapoints" not in req:
        # Reconstruct merge metadata from source dataset shards for uid lookup.
        base = request_file.resolve()
        dataset_dir = base.parents[2] / "mini_dataset"
        shard_paths = sorted(dataset_dir.glob("*_position_dataset_v0_1.json"))
        uid_map: Dict[str, Dict[str, Any]] = {}
        for shard_path in shard_paths:
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            scene_id = str(shard["scene_id"])
            for i, _dp in enumerate(shard.get("datapoints", [])):
                uid = f"{scene_id}_dp_{i:06d}"
                uid_map[uid] = {
                    "index": -1,
                    "datapoint_uid": uid,
                    "scene_id": scene_id,
                    "source_shard": str(shard_path),
                    "source_datapoint_index": i,
                }
        items = req.get("items", [])
        dps = []
        for i, it in enumerate(items):
            uid = str(it.get("datapoint_uid", ""))
            if uid in uid_map:
                r = dict(uid_map[uid])
                r["index"] = i
                dps.append(r)
        req["datapoints"] = dps
        req["chunk_id"] = str(req.get("chunk_id", request_file.stem))

    req_map: Dict[str, Dict[str, Any]] = {d["datapoint_uid"]: d for d in req["datapoints"]}
    req_by_scene_index: Dict[Tuple[str, int], Dict[str, Any]] = {
        (str(d["scene_id"]), int(d["index"])): d for d in req["datapoints"]
    }
    by_chunk = {req["chunk_id"]: req}

    # Include split chunks if present (loaded from response names)
    for rf in response_files:
        cid = rf.name.replace(".response.json", "")
        if cid == req["chunk_id"]:
            continue
        if cid.startswith(req["chunk_id"] + "_"):
            # Reconstruct expected chunk map entries by splitting datapoints logically.
            # For validation we only need uid->request item, already covered by req_map.
            by_chunk[cid] = {"chunk_id": cid}

    merged_by_shard: Dict[str, Dict[str, Any]] = {}
    ok = 0
    failed = 0
    retried: List[Dict[str, Any]] = []
    errors: List[str] = []

    for rf in response_files:
        obj = json.loads(rf.read_text(encoding="utf-8"))
        resp = obj.get("response", {})
        if not isinstance(resp, dict) or not resp:
            parsed = obj.get("parsed")
            if isinstance(parsed, dict):
                resp = parsed
        results = resp.get("results", [])
        if not isinstance(results, list):
            errors.append(f"{rf.name}: missing/invalid results")
            continue
        for item in results:
            try:
                uid = str(item["datapoint_uid"]) if "datapoint_uid" in item else ""
                instructions = item["instructions"]
                req_item = None
                if uid and uid in req_map:
                    req_item = req_map[uid]
                if req_item is None:
                    raise ValueError("could not map response item to request by datapoint_uid")
                if not isinstance(instructions, list) or len(instructions) != 4:
                    raise ValueError(f"instruction count invalid for {uid}")
                cleaned = [str(x).strip() for x in instructions if str(x).strip()]
                if len(cleaned) != 4 or len({c.lower() for c in cleaned}) != 4:
                    raise ValueError(f"instructions invalid/duplicate for {uid}")

                shard_path = str(req_item["source_shard"])
                if shard_path not in merged_by_shard:
                    merged_by_shard[shard_path] = json.loads(
                        Path(shard_path).read_text(encoding="utf-8")
                    )
                shard = merged_by_shard[shard_path]
                dp_idx = int(req_item["source_datapoint_index"])
                dp = shard["datapoints"][dp_idx]
                dp["instructions"] = cleaned
                dp["instruction_status"] = "ok"
                dp["instruction_chunk_id"] = str(resp.get("chunk_id", rf.stem))
                dp["instruction_generated_at"] = _now_iso()
                ok += 1
            except Exception as exc:
                failed += 1
                retried.append(item if isinstance(item, dict) else {"raw_item": item})
                errors.append(f"{rf.name}: {exc}")

    # Any missing uids from responses are retried too.
    seen_uids = set()
    for rf in response_files:
        obj = json.loads(rf.read_text(encoding="utf-8"))
        resp = obj.get("response", {})
        if not isinstance(resp, dict) or not resp:
            parsed = obj.get("parsed")
            if isinstance(parsed, dict):
                resp = parsed
        for item in resp.get("results", []) if isinstance(resp.get("results", []), list) else []:
            if isinstance(item, dict) and "datapoint_uid" in item:
                seen_uids.add(str(item["datapoint_uid"]))
    for uid, req_item in req_map.items():
        if uid not in seen_uids:
            failed += 1
            retried.append(req_item)
            errors.append(f"missing result for uid={uid}")

    for shard_path, shard in merged_by_shard.items():
        Path(shard_path).write_text(json.dumps(shard, indent=2), encoding="utf-8")

    requests_dir = output_root / "requests"
    logs_dir = output_root / "logs"
    requests_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    retry_file = requests_dir / f"{req['chunk_id']}.retry.json"
    if retried:
        retry_obj = dict(req)
        retry_obj["chunk_id"] = f"{req['chunk_id']}.retry"
        retry_obj["datapoints"] = []
        for i, item in enumerate(retried):
            if isinstance(item, dict) and "datapoint_uid" in item and item["datapoint_uid"] in req_map:
                r = dict(req_map[item["datapoint_uid"]])
            elif isinstance(item, dict) and "datapoint_uid" in item and item["datapoint_uid"] in req_map:
                r = dict(req_map[item["datapoint_uid"]])
            elif isinstance(item, dict) and "source_shard" in item:
                r = dict(item)
            else:
                continue
            r["index"] = i
            retry_obj["datapoints"].append(r)
        retry_obj["chunk_size"] = len(retry_obj["datapoints"])
        retry_file.write_text(json.dumps(retry_obj, indent=2), encoding="utf-8")

    report = {
        "schema_version": SCHEMA_VERSION,
        "chunk_id": req["chunk_id"],
        "ok": ok,
        "failed": failed,
        "retried": len(retried),
        "response_files": [str(p) for p in response_files],
        "retry_file": str(retry_file) if retried else None,
        "errors": errors[:200],
    }
    report_file = logs_dir / f"{req['chunk_id']}.merge_report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _timed_process_request_file(
    *,
    request_file: Path,
    output_root: Path,
    model: str,
    retry: RetryConfig,
) -> Tuple[Path, List[Path], float]:
    t0 = time.perf_counter()
    response_files = process_request_file(
        request_file=request_file,
        output_root=output_root,
        model=model,
        retry=retry,
        split_on_exhaust=True,
    )
    elapsed = time.perf_counter() - t0
    return request_file, response_files, elapsed


def _extract_response_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    resp = obj.get("response", {})
    if not isinstance(resp, dict) or not resp:
        parsed = obj.get("parsed")
        if isinstance(parsed, dict):
            resp = parsed
    return resp if isinstance(resp, dict) else {}


def _is_chunk_complete(request_obj: Dict[str, Any], response_files: List[Path]) -> bool:
    req_uids = {str(d.get("datapoint_uid", "")) for d in request_obj.get("datapoints", [])}
    req_uids.discard("")
    if not req_uids:
        return False
    seen: Dict[str, List[str]] = {}
    for rf in response_files:
        try:
            obj = json.loads(rf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if obj.get("error"):
            continue
        resp = _extract_response_payload(obj)
        results = resp.get("results", [])
        if not isinstance(results, list):
            continue
        for item in results:
            if not isinstance(item, dict):
                continue
            uid = str(item.get("datapoint_uid", ""))
            instructions = item.get("instructions", [])
            if uid not in req_uids or not isinstance(instructions, list):
                continue
            cleaned = [str(x).strip() for x in instructions if str(x).strip()]
            if len(cleaned) != 4 or len({c.lower() for c in cleaned}) != 4:
                continue
            seen[uid] = cleaned
    return len(seen) == len(req_uids)


def run_resume(
    *,
    output_root: Path,
    model: str,
    retry: RetryConfig,
    max_workers: int = DEFAULT_RUN_MAX_WORKERS,
) -> Dict[str, Any]:
    wall_t0 = time.perf_counter()
    requests_dir = output_root / "requests"
    req_files = sorted(requests_dir.glob("chunk_[0-9][0-9][0-9][0-9].json"))
    if not req_files:
        raise FileNotFoundError(f"No top-level chunk request files found in {requests_dir}")

    responses_by_request: Dict[Path, List[Path]] = {}
    call_timings: List[Dict[str, Any]] = []
    already_complete = 0
    rerun_files: List[Path] = []

    for req_file in req_files:
        req_obj = json.loads(req_file.read_text(encoding="utf-8"))
        chunk_id = str(req_obj.get("chunk_id", req_file.stem))
        existing = sorted((output_root / "responses").glob(f"{chunk_id}*.response.json"))
        if _is_chunk_complete(req_obj, existing):
            already_complete += 1
            responses_by_request[req_file] = existing
        else:
            rerun_files.append(req_file)

    call_phase_t0 = time.perf_counter()
    workers = max(1, int(max_workers))
    if workers == 1:
        for req_file in rerun_files:
            rf, rpaths, elapsed = _timed_process_request_file(
                request_file=req_file,
                output_root=output_root,
                model=model,
                retry=retry,
            )
            responses_by_request[rf] = sorted((output_root / "responses").glob(f"{rf.stem}*.response.json"))
            req_obj = json.loads(rf.read_text(encoding="utf-8"))
            call_timings.append(
                {
                    "chunk_id": str(req_obj.get("chunk_id", rf.stem)),
                    "chunk_index": int(req_obj.get("chunk_index", 0)),
                    "seconds": round(elapsed, 3),
                }
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    _timed_process_request_file,
                    request_file=rf,
                    output_root=output_root,
                    model=model,
                    retry=retry,
                ): rf
                for rf in rerun_files
            }
            for fut in as_completed(futures):
                req_file, _rpaths, elapsed = fut.result()
                req_obj = json.loads(req_file.read_text(encoding="utf-8"))
                chunk_id = str(req_obj.get("chunk_id", req_file.stem))
                responses_by_request[req_file] = sorted(
                    (output_root / "responses").glob(f"{chunk_id}*.response.json")
                )
                call_timings.append(
                    {
                        "chunk_id": chunk_id,
                        "chunk_index": int(req_obj.get("chunk_index", 0)),
                        "seconds": round(elapsed, 3),
                    }
                )
        call_timings.sort(key=lambda row: row["chunk_index"])
    parallel_calls_seconds = time.perf_counter() - call_phase_t0

    merge_phase_t0 = time.perf_counter()
    merge_timings: List[Dict[str, Any]] = []
    reports = []
    for req_file in req_files:
        if req_file not in responses_by_request:
            req_obj = json.loads(req_file.read_text(encoding="utf-8"))
            chunk_id = str(req_obj.get("chunk_id", req_file.stem))
            responses_by_request[req_file] = sorted(
                (output_root / "responses").glob(f"{chunk_id}*.response.json")
            )
        t_m0 = time.perf_counter()
        report = validate_and_merge(
            request_file=req_file,
            response_files=responses_by_request[req_file],
            output_root=output_root,
        )
        merge_elapsed = time.perf_counter() - t_m0
        merge_timings.append(
            {"chunk_id": report.get("chunk_id", req_file.stem), "seconds": round(merge_elapsed, 3)}
        )
        reports.append(report)
    merge_phase_seconds = time.perf_counter() - merge_phase_t0

    wall_seconds = time.perf_counter() - wall_t0
    summary = {
        "mode": "resume",
        "requests": len(req_files),
        "already_complete": already_complete,
        "rerun_needed": len(rerun_files),
        "ok": int(sum(r["ok"] for r in reports)),
        "failed": int(sum(r["failed"] for r in reports)),
        "retried": int(sum(r["retried"] for r in reports)),
        "reports": reports,
        "generated_at": _now_iso(),
        "max_workers": workers,
        "timing_seconds": {
            "parallel_calls_wall": round(parallel_calls_seconds, 3),
            "merge_phase_wall": round(merge_phase_seconds, 3),
            "wall_total": round(wall_seconds, 3),
            "per_chunk_gemini_call": call_timings,
            "per_chunk_merge": merge_timings,
        },
    }
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "resume_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_all(
    *,
    dataset_dir: Path,
    output_root: Path,
    run_id: str,
    chunk_size: int,
    model: str,
    retry: RetryConfig,
    max_workers: int = DEFAULT_RUN_MAX_WORKERS,
) -> Dict[str, Any]:
    wall_t0 = time.perf_counter()
    t_exp0 = time.perf_counter()
    req_files = export_chunks(
        dataset_dir=dataset_dir,
        output_root=output_root,
        run_id=run_id,
        chunk_size=chunk_size,
    )
    export_seconds = time.perf_counter() - t_exp0

    call_phase_t0 = time.perf_counter()
    call_timings: List[Dict[str, Any]] = []
    responses_by_request: Dict[Path, List[Path]] = {}

    workers = max(1, int(max_workers))
    if workers == 1:
        for req_file in req_files:
            rf, rpaths, elapsed = _timed_process_request_file(
                request_file=req_file,
                output_root=output_root,
                model=model,
                retry=retry,
            )
            responses_by_request[rf] = rpaths
            req_obj = json.loads(rf.read_text(encoding="utf-8"))
            call_timings.append(
                {
                    "chunk_id": str(req_obj.get("chunk_id", rf.stem)),
                    "chunk_index": int(req_obj.get("chunk_index", 0)),
                    "seconds": round(elapsed, 3),
                }
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    _timed_process_request_file,
                    request_file=rf,
                    output_root=output_root,
                    model=model,
                    retry=retry,
                ): rf
                for rf in req_files
            }
            for fut in as_completed(futures):
                req_file, rpaths, elapsed = fut.result()
                responses_by_request[req_file] = rpaths
                req_obj = json.loads(req_file.read_text(encoding="utf-8"))
                call_timings.append(
                    {
                        "chunk_id": str(req_obj.get("chunk_id", req_file.stem)),
                        "chunk_index": int(req_obj.get("chunk_index", 0)),
                        "seconds": round(elapsed, 3),
                    }
                )
        call_timings.sort(key=lambda row: row["chunk_index"])
    parallel_calls_seconds = time.perf_counter() - call_phase_t0

    merge_phase_t0 = time.perf_counter()
    merge_timings: List[Dict[str, Any]] = []
    reports = []
    for req_file in req_files:
        response_files = responses_by_request[req_file]
        t_m0 = time.perf_counter()
        report = validate_and_merge(
            request_file=req_file,
            response_files=response_files,
            output_root=output_root,
        )
        merge_elapsed = time.perf_counter() - t_m0
        merge_timings.append(
            {"chunk_id": report.get("chunk_id", req_file.stem), "seconds": round(merge_elapsed, 3)}
        )
        reports.append(report)
    merge_phase_seconds = time.perf_counter() - merge_phase_t0

    wall_seconds = time.perf_counter() - wall_t0

    summary = {
        "run_id": run_id,
        "requests": len(req_files),
        "ok": int(sum(r["ok"] for r in reports)),
        "failed": int(sum(r["failed"] for r in reports)),
        "retried": int(sum(r["retried"] for r in reports)),
        "reports": reports,
        "generated_at": _now_iso(),
        "max_workers": workers,
        "timing_seconds": {
            "export": round(export_seconds, 3),
            "parallel_calls_wall": round(parallel_calls_seconds, 3),
            "merge_phase_wall": round(merge_phase_seconds, 3),
            "wall_total": round(wall_seconds, 3),
            "per_chunk_gemini_call": call_timings,
            "per_chunk_merge": merge_timings,
        },
    }
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def finalize_dataset_rows(
    *,
    dataset_dir: Path,
    output_path: Path,
    output_format: str = "jsonl",
    include_tool_dot_path: bool = False,
) -> Dict[str, Any]:
    """Flatten per-scene shard files into one long unsplit dataset."""
    shard_paths = sorted(dataset_dir.glob("*_position_dataset_v0_1.json"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in {dataset_dir}")

    rows: List[Dict[str, Any]] = []
    missing_instructions = 0
    for shard_path in shard_paths:
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        scene_id = str(shard.get("scene_id", ""))
        image = str(shard.get("image", ""))
        depth = str(shard.get("depth", ""))
        cam = shard.get("camera", {})
        intr = cam.get("intrinsics", {})
        world_from_camera = cam.get("world_from_camera")
        keypoints = shard.get("keypoints", {})

        for i, dp in enumerate(shard.get("datapoints", [])):
            uid = f"{scene_id}_dp_{i:06d}"
            tool_id = str(dp.get("tool_keypoint_id", ""))
            tool_kp = keypoints.get(tool_id, {})
            tool_label = f"{tool_kp.get('label', tool_id)}"
            tool_obj = str(tool_kp.get("object_name", "")).strip()
            if tool_obj:
                tool_label = f"{tool_label} of {tool_obj}"

            all_kps = []
            for k in keypoints.values():
                if not k.get("valid", False):
                    continue
                lbl = str(k.get("label", "")).strip()
                obj_name = str(k.get("object_name", "")).strip()
                if obj_name:
                    lbl = f"{lbl} of {obj_name}"
                all_kps.append(
                    {
                        "label": lbl,
                        "xyz_world": k.get("xyz_world"),
                    }
                )

            instructions = dp.get("instructions", [])
            if not isinstance(instructions, list) or len(instructions) == 0:
                missing_instructions += 1

            if not isinstance(instructions, list) or not instructions:
                # Preserve a single row even if missing instructions.
                instructions = [""]

            for ins_idx, instruction in enumerate(instructions):
                row = {
                    "datapoint_uid": uid,
                    "instruction_uid": f"{uid}_ins_{ins_idx:02d}",
                    "scene_id": scene_id,
                    "input": {
                        "rgb_path": image,
                        "depth_path": depth,
                        "instruction": str(instruction),
                        "tool_keypoint_label": tool_label,
                        "tool_keypoint_xyz_world": tool_kp.get("xyz_world"),
                        "keypoints": all_kps,
                    },
                    "target": {
                        "goal_tool_keypoint_xyz_world": dp.get("goal_tool_keypoint_xyz_world"),
                    },
                    "metadata": {
                        "movement_token": dp.get("movement_token"),
                        "relation_string": dp.get("relation_string"),
                        "instruction_index": ins_idx,
                        "camera": {
                            "fx": intr.get("fx"),
                            "fy": intr.get("fy"),
                            "cx": intr.get("cx"),
                            "cy": intr.get("cy"),
                            "width": intr.get("width"),
                            "height": intr.get("height"),
                            "world_from_camera": world_from_camera,
                        },
                        "instruction_source": {
                            "instruction_status": dp.get("instruction_status"),
                            "instruction_chunk_id": dp.get("instruction_chunk_id"),
                            "instruction_generated_at": dp.get("instruction_generated_at"),
                        },
                    },
                }
                if include_tool_dot_path:
                    row["input"]["rgb_with_tool_dot_path"] = (
                        f"{Path(image).with_suffix('')}_tooldot.png"
                    )
                rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    elif output_format == "json":
        output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    elif output_format == "parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("parquet output requires pandas") from exc
        # Keep nested fields as JSON strings for compatibility.
        flat = []
        for r in rows:
            flat.append(
                {
                    "datapoint_uid": r["datapoint_uid"],
                    "scene_id": r["scene_id"],
                    "input_json": json.dumps(r["input"]),
                    "target_json": json.dumps(r["target"]),
                    "metadata_json": json.dumps(r["metadata"]),
                }
            )
        pd.DataFrame(flat).to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output_format: {output_format}")

    summary = {
        "rows": len(rows),
        "missing_instructions": missing_instructions,
        "output_path": str(output_path),
        "output_format": output_format,
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chunked mega-payload instruction batching pipeline.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export", help="Export chunked request files from dataset shards.")
    p_export.add_argument("--dataset-dir", type=Path, required=True)
    p_export.add_argument("--output-root", type=Path, required=True)
    p_export.add_argument("--run-id", type=str, required=True)
    p_export.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)

    p_call = sub.add_parser("call", help="Call Gemini for a request file and write response files.")
    p_call.add_argument("--request-file", type=Path, required=True)
    p_call.add_argument("--output-root", type=Path, required=True)
    p_call.add_argument("--model", type=str, default=DEFAULT_INSTRUCTION_MODEL)
    p_call.add_argument("--retry-max-attempts", type=int, default=6)
    p_call.add_argument("--retry-base-seconds", type=float, default=1.5)
    p_call.add_argument("--retry-max-seconds", type=float, default=60.0)

    p_merge = sub.add_parser("merge", help="Validate and merge one request + response set.")
    p_merge.add_argument("--request-file", type=Path, required=True)
    p_merge.add_argument("--response-files", type=Path, nargs="+", required=True)
    p_merge.add_argument("--output-root", type=Path, required=True)

    p_run = sub.add_parser("run", help="Run full export->call->merge pipeline.")
    p_run.add_argument("--dataset-dir", type=Path, required=True)
    p_run.add_argument("--output-root", type=Path, required=True)
    p_run.add_argument("--run-id", type=str, required=True)
    p_run.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p_run.add_argument("--model", type=str, default=DEFAULT_INSTRUCTION_MODEL)
    p_run.add_argument("--retry-max-attempts", type=int, default=6)
    p_run.add_argument("--retry-base-seconds", type=float, default=1.5)
    p_run.add_argument("--retry-max-seconds", type=float, default=60.0)
    p_run.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_RUN_MAX_WORKERS,
        help="Parallel Gemini calls (one client per thread). Merges stay sequential to avoid shard races.",
    )

    p_resume = sub.add_parser(
        "resume",
        help="Resume from existing requests/responses; only incomplete chunks are re-called.",
    )
    p_resume.add_argument("--output-root", type=Path, required=True)
    p_resume.add_argument("--model", type=str, default=DEFAULT_INSTRUCTION_MODEL)
    p_resume.add_argument("--retry-max-attempts", type=int, default=6)
    p_resume.add_argument("--retry-base-seconds", type=float, default=1.5)
    p_resume.add_argument("--retry-max-seconds", type=float, default=60.0)
    p_resume.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_RUN_MAX_WORKERS,
        help="Parallel Gemini calls for incomplete chunks.",
    )

    p_finalize = sub.add_parser("finalize", help="Flatten shard files into one unsplit long dataset.")
    p_finalize.add_argument("--dataset-dir", type=Path, required=True)
    p_finalize.add_argument("--output-path", type=Path, required=True)
    p_finalize.add_argument("--format", type=str, choices=["jsonl", "json", "parquet"], default="jsonl")
    p_finalize.add_argument("--include-tool-dot-path", action="store_true")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.cmd == "export":
        files = export_chunks(
            dataset_dir=args.dataset_dir,
            output_root=args.output_root,
            run_id=args.run_id,
            chunk_size=args.chunk_size,
        )
        print(f"Exported {len(files)} request chunks to {args.output_root / 'requests'}")
        return

    if args.cmd == "call":
        retry = RetryConfig(
            max_attempts=args.retry_max_attempts,
            base_seconds=args.retry_base_seconds,
            max_seconds=args.retry_max_seconds,
        )
        outs = process_request_file(
            request_file=args.request_file,
            output_root=args.output_root,
            model=args.model,
            retry=retry,
            split_on_exhaust=True,
        )
        print(f"Wrote {len(outs)} response file(s)")
        return

    if args.cmd == "merge":
        report = validate_and_merge(
            request_file=args.request_file,
            response_files=args.response_files,
            output_root=args.output_root,
        )
        print(json.dumps(report, indent=2))
        return

    if args.cmd == "finalize":
        summary = finalize_dataset_rows(
            dataset_dir=args.dataset_dir,
            output_path=args.output_path,
            output_format=args.format,
            include_tool_dot_path=bool(args.include_tool_dot_path),
        )
        print(json.dumps(summary, indent=2))
        return

    if args.cmd == "resume":
        retry = RetryConfig(
            max_attempts=args.retry_max_attempts,
            base_seconds=args.retry_base_seconds,
            max_seconds=args.retry_max_seconds,
        )
        summary = run_resume(
            output_root=args.output_root,
            model=args.model,
            retry=retry,
            max_workers=int(args.max_workers),
        )
        print(json.dumps(summary, indent=2))
        return

    retry = RetryConfig(
        max_attempts=args.retry_max_attempts,
        base_seconds=args.retry_base_seconds,
        max_seconds=args.retry_max_seconds,
    )
    summary = run_all(
        dataset_dir=args.dataset_dir,
        output_root=args.output_root,
        run_id=args.run_id,
        chunk_size=args.chunk_size,
        model=args.model,
        retry=retry,
        max_workers=int(args.max_workers),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

