from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from cosmos_vlm.artifacts.io import read_json, write_json
from cosmos_vlm.artifacts.layout import (
    ensure_run_dir,
    inputs_dir,
    latest_predict_dir,
    next_predict_dir,
    reason_dir,
    stage_dir,
)
from cosmos_vlm.config import load_config
from cosmos_vlm.importers.local_file import import_local_image
from cosmos_vlm.importers.sim_runs import import_latest_sim_image, import_sim_image
from cosmos_vlm.predict25.client import download_url_to_file, get_prediction_result, submit_image_to_video
from cosmos_vlm.predict25.debug_prompting import build_predict25_self_planned_debug_prompt
from cosmos_vlm.predict25.prompting import build_predict25_prompt_from_trajectory
from cosmos_vlm.predict25.schemas import extract_task_id, validate_result_payload
from cosmos_vlm.predict_veo.client import (
    download_veo_outputs,
    poll_veo_operation,
    submit_veo_image_to_video,
)
from cosmos_vlm.reason2.inference import run_reason_plan


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    cfg = load_config()
    run_id = getattr(args, "run_id", None)
    run_dir_arg = getattr(args, "run_dir", None)
    if run_dir_arg:
        run_dir = Path(run_dir_arg).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    return ensure_run_dir(cfg.cosmos_runs_dir, run_id=run_id).resolve()


def _resolve_existing_run_dir(run_dir: str | None, run_id: str | None) -> Path:
    cfg = load_config()
    if run_dir:
        p = Path(run_dir).resolve()
    elif run_id:
        p = (cfg.cosmos_runs_dir / run_id).resolve()
    else:
        raise ValueError("provide --run-id or --run-dir")
    if not p.is_dir():
        raise FileNotFoundError(f"run directory not found: {p}")
    return p


def cmd_import_image(args: argparse.Namespace) -> int:
    run_dir = _resolve_run_dir(args)
    dst = inputs_dir(run_dir) / "start_image.png"
    stage = stage_dir(run_dir, "stage1_import")

    if args.source_file:
        metadata = import_local_image(Path(args.source_file), dst)
    elif args.sim_runs_root:
        metadata = import_latest_sim_image(Path(args.sim_runs_root), dst, step=args.step)
    else:
        sim_session_dir = Path(args.sim_session_dir)
        metadata = import_sim_image(sim_session_dir, dst, step=args.step)

    write_json(stage / "metadata.json", metadata)
    print(str(run_dir))
    return 0


def cmd_reason_plan(args: argparse.Namespace) -> int:
    cfg = load_config()
    # Reason runs are always isolated in a fresh run directory.
    run_dir = ensure_run_dir(cfg.cosmos_runs_dir, run_id=None).resolve()
    image_path = Path(args.image_path) if args.image_path else (inputs_dir(run_dir) / "start_image.png")
    if not image_path.is_file():
        # Auto-import latest sim 00_main.png when reason is run directly.
        metadata = import_latest_sim_image(Path("/home/ubuntu/Generative_STR/simtoolreal/runs"), image_path)
        write_json(stage_dir(run_dir, "stage1_import") / "metadata.json", metadata)

    reason = reason_dir(run_dir)
    raw_text, plan_json, meta, prompt_text = run_reason_plan(
        image_path=image_path,
        model_id_or_path=args.model_id or cfg.cosmos_reason2_model_id,
        device=args.device or cfg.cosmos_device,
        dtype_name=args.dtype or cfg.cosmos_dtype,
        max_new_tokens=args.max_new_tokens or cfg.cosmos_max_new_tokens,
        task_description=args.task_description,
    )
    (reason / "raw.txt").write_text(raw_text, encoding="utf-8")
    (reason / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    write_json(reason / "plan.json", plan_json)
    write_json(reason / "meta.json", meta)
    print(str(run_dir))
    return 0


def _resolve_predict_dir(run_dir: Path, predict_dir_arg: str | None, *, create_if_missing: bool) -> Path:
    if predict_dir_arg:
        p = Path(predict_dir_arg).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    latest = latest_predict_dir(run_dir)
    if latest is not None:
        return latest
    if create_if_missing:
        return next_predict_dir(run_dir)
    raise FileNotFoundError(f"no predict directories found under {run_dir}")


def cmd_submit_video(args: argparse.Namespace) -> int:
    cfg = load_config()
    run_dir = _resolve_existing_run_dir(args.run_dir, args.run_id)
    predict = _resolve_predict_dir(run_dir, args.predict_dir, create_if_missing=True)
    provider = (args.video_provider or cfg.video_provider).strip().lower()

    image_value = args.image
    if not image_value:
        default_image = inputs_dir(run_dir) / "start_image.png"
        if not default_image.is_file():
            raise FileNotFoundError(
                f"no --image provided and default start image not found: {default_image}"
            )
        raise ValueError(
            "submit-video requires --image as a publicly accessible URL; "
            f"local default image is available at {default_image.resolve()}"
        )

    if provider == "veo":
        veo_seed = int(args.veo_seed) if args.veo_seed is not None else int(cfg.veo_seed)
        veo_duration = (
            int(args.veo_duration_seconds)
            if args.veo_duration_seconds is not None
            else int(cfg.veo_duration_seconds)
        )
        veo_aspect = (
            str(args.veo_aspect_ratio).strip()
            if args.veo_aspect_ratio is not None
            else str(cfg.veo_aspect_ratio).strip()
        )
        response_json = submit_veo_image_to_video(
            prompt=args.prompt,
            image=image_value,
            model=args.veo_model or cfg.veo_model_id,
            output_gcs_uri=args.veo_output_gcs_uri or cfg.veo_output_gcs_uri,
            duration_seconds=veo_duration,
            aspect_ratio=veo_aspect,
            seed=veo_seed,
            negative_prompt=str(args.veo_negative_prompt or "").strip(),
        )
        task_id = response_json.get("operation_name", "")
    else:
        response_json = submit_image_to_video(
            base_url=cfg.wavespeed_base_url,
            api_key=cfg.wavespeed_api_key,
            prompt=args.prompt,
            image=image_value,
        )
        task_id = extract_task_id(response_json)

    write_json(
        predict / "request.json",
        {
            "prompt": args.prompt,
            "image": image_value,
            "base_url": cfg.wavespeed_base_url,
        },
    )
    metadata_path = predict / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    metadata["submit_video"] = {
        "provider": provider,
        "prompt_provided_to_model": args.prompt,
        "image": image_value,
    }
    if provider == "wavespeed":
        metadata["submit_video"]["base_url"] = cfg.wavespeed_base_url
    if provider == "veo":
        metadata["submit_video"]["veo_model"] = args.veo_model or cfg.veo_model_id
        metadata["submit_video"]["veo_output_gcs_uri"] = (
            args.veo_output_gcs_uri or cfg.veo_output_gcs_uri
        )
        metadata["submit_video"]["veo_seed"] = int(response_json.get("seed", 0) or 0)
        metadata["submit_video"]["veo_duration_seconds"] = int(
            response_json.get("duration_seconds", 0) or 0
        )
        metadata["submit_video"]["veo_aspect_ratio"] = str(response_json.get("aspect_ratio", "") or "")
        if response_json.get("negative_prompt"):
            metadata["submit_video"]["veo_negative_prompt"] = str(response_json.get("negative_prompt") or "")
    write_json(metadata_path, metadata)
    write_json(predict / "response.json", response_json)
    print(task_id)
    return 0


def cmd_draft_video_prompt(args: argparse.Namespace) -> int:
    run_dir = _resolve_existing_run_dir(args.run_dir, args.run_id)
    trajectory_steps: list[str] = []
    motion_prompt_for_video = ""
    if args.prompt_mode == "self_planned_debug":
        prompt = build_predict25_self_planned_debug_prompt()
    else:
        reason = reason_dir(run_dir)
        plan = read_json(reason / "plan.json")
        trajectory_steps = plan.get("trajectory_steps", [])
        motion_prompt_for_video = plan.get("motion_prompt_for_video", "")
        if not trajectory_steps:
            raw_path = reason / "raw.txt"
            if raw_path.is_file():
                raw = raw_path.read_text(encoding="utf-8")
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
                if not match:
                    match = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(1))
                        trajectory_steps = parsed.get("trajectory_steps", [])
                        motion_prompt_for_video = parsed.get("motion_prompt_for_video", motion_prompt_for_video)
                    except json.JSONDecodeError:
                        trajectory_steps = []
                if not trajectory_steps:
                    # Fallback for truncated JSON: recover quoted step lines.
                    step_lines = re.findall(r'^\s*"([^"\n]+)"\s*,?\s*$', raw, flags=re.MULTILINE)
                    if step_lines:
                        trajectory_steps = [s for s in step_lines if "scene_summary" not in s]
        prompt = build_predict25_prompt_from_trajectory(
            trajectory_steps,
            motion_prompt_for_video=motion_prompt_for_video,
        )
    if args.predict_dir:
        predict = Path(args.predict_dir).resolve()
        predict.mkdir(parents=True, exist_ok=True)
    else:
        predict = next_predict_dir(run_dir)
    prompt_path = predict / "prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    write_json(
        predict / "metadata.json",
        {
            "predict_prompt": {
                "prompt_provided_to_model": prompt,
                "prompt_mode": args.prompt_mode,
                "motion_prompt_for_video": motion_prompt_for_video,
                "trajectory_steps": trajectory_steps,
            }
        },
    )
    print(str(prompt_path))
    return 0


def cmd_get_video_result(args: argparse.Namespace) -> int:
    cfg = load_config()
    run_dir = _resolve_existing_run_dir(args.run_dir, args.run_id)
    predict = _resolve_predict_dir(run_dir, args.predict_dir, create_if_missing=False)
    metadata_path = predict / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    provider = (
        args.video_provider
        or metadata.get("submit_video", {}).get("provider")
        or cfg.video_provider
    )
    provider = str(provider).strip().lower()

    task_id = args.task_id
    if not task_id:
        prior = read_json(predict / "response.json")
        if provider == "veo":
            task_id = str(prior.get("operation_name") or "")
            if not task_id:
                raise ValueError("VEO response missing operation_name")
        else:
            task_id = extract_task_id(prior)

    if provider == "veo":
        model = metadata.get("submit_video", {}).get("veo_model") or cfg.veo_model_id
        start = time.time()
        while True:
            polled = poll_veo_operation(operation_name=task_id, model=model)
            status = polled.get("status", "")
            if not args.wait or status in {"completed", "failed"}:
                break
            if (time.time() - start) >= args.max_wait_seconds:
                break
            time.sleep(args.poll_interval_seconds)

        write_json(
            predict / "result.json",
            {
                "provider": "veo",
                "operation_name": polled.get("operation_name", ""),
                "status": polled.get("status", ""),
                "outputs": polled.get("outputs", []),
                "model": polled.get("model", ""),
            },
        )
        downloaded_files: list[str] = []
        if polled.get("status") == "completed":
            downloaded_files = download_veo_outputs(
                operation_obj=polled.get("operation_obj"),
                output_dir=predict / "downloads",
            )
        summary = {
            "provider": "veo",
            "id": polled.get("operation_name", ""),
            "status": polled.get("status", ""),
            "outputs": polled.get("outputs", []),
            "model": polled.get("model", ""),
            "downloaded_files": downloaded_files,
            "error": "",
        }
        write_json(predict / "summary.json", summary)
        print(summary["status"])
        return 0
    else:
        start = time.time()
        while True:
            response_json = get_prediction_result(
                base_url=cfg.wavespeed_base_url,
                api_key=cfg.wavespeed_api_key,
                task_id=task_id,
            )
            validated = validate_result_payload(response_json)
            status = validated.get("status", "")
            if not args.wait or status in {"completed", "failed"}:
                break
            if (time.time() - start) >= args.max_wait_seconds:
                break
            time.sleep(args.poll_interval_seconds)

        write_json(predict / "result.json", response_json)
        downloaded_files: list[str] = []
        if validated.get("status") == "completed":
            downloads_dir = predict / "downloads"
            outputs = validated.get("outputs", [])
            for idx, out_url in enumerate(outputs):
                if not isinstance(out_url, str) or not out_url:
                    continue
                parsed = urlparse(out_url)
                suffix = Path(parsed.path).suffix or ".mp4"
                out_path = downloads_dir / f"output_{idx:04d}{suffix}"
                download_url_to_file(out_url, out_path)
                downloaded_files.append(str(out_path.resolve()))
        validated["downloaded_files"] = downloaded_files
        write_json(predict / "summary.json", validated)
        print(validated["status"])
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cosmos-vlm")
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import-image", help="Stage 1: import start image")
    group = p_import.add_mutually_exclusive_group(required=False)
    group.add_argument("--source-file", help="Path to local source image")
    group.add_argument("--sim-session-dir", help="Path to sim session directory containing log.jsonl")
    group.add_argument(
        "--sim-runs-root",
        default=None,
        help="Root with sim sessions (defaults to simtoolreal/runs); picks latest vlm_* session",
    )
    p_import.add_argument("--step", type=int, default=None, help="Specific sim step; default latest")
    p_import.add_argument("--run-dir", default=None, help="Existing/new run directory")
    p_import.add_argument("--run-id", default=None, help="Run id under COSMOS_RUNS_DIR")
    p_import.set_defaults(func=cmd_import_image)

    p_reason = sub.add_parser("reason-plan", help="Stage 2: run local Cosmos Reason2 plan")
    p_reason.add_argument("--image-path", default=None, help="Override image path")
    p_reason.add_argument("--task-description", default=None, help="Planning task text")
    p_reason.add_argument("--model-id", default=None, help="HF id or local model path")
    p_reason.add_argument("--device", default=None, help="Device override")
    p_reason.add_argument("--dtype", default=None, help="Dtype override (bfloat16/float16)")
    p_reason.add_argument("--max-new-tokens", type=int, default=None, help="Generation token cap")
    p_reason.set_defaults(func=cmd_reason_plan)

    p_submit = sub.add_parser("submit-video", help="Stage 3: submit Wavespeed image-to-video request")
    p_submit.add_argument("--run-dir", default=None, help="Run directory")
    p_submit.add_argument("--run-id", default=None, help="Run id under COSMOS_RUNS_DIR")
    p_submit.add_argument(
        "--video-provider",
        choices=["wavespeed", "veo"],
        default=None,
        help="Video generation provider (default from VIDEO_PROVIDER env)",
    )
    p_submit.add_argument("--veo-model", default=None, help="VEO model id override")
    p_submit.add_argument(
        "--veo-output-gcs-uri",
        default=None,
        help="Vertex GCS output prefix for downloadable VEO outputs (e.g. gs://bucket/prefix/)",
    )
    p_submit.add_argument(
        "--veo-seed",
        type=int,
        default=None,
        help="VEO generation seed (default from VEO_SEED env / CosmosConfig)",
    )
    p_submit.add_argument(
        "--veo-duration-seconds",
        type=int,
        default=None,
        help="VEO clip duration in seconds (default from VEO_DURATION_SECONDS env / CosmosConfig)",
    )
    p_submit.add_argument(
        "--veo-aspect-ratio",
        default=None,
        help='VEO aspect ratio string like "16:9" (default from VEO_ASPECT_RATIO env / CosmosConfig)',
    )
    p_submit.add_argument(
        "--veo-negative-prompt",
        default=(
            "claw gripper, pincer, mechanical hand, extra fingers, deformed hand, "
            "tool swap, different brush, short handle, missing handle, bent handle, "
            "morphing brush, melting plastic, teleporting objects, camera move, zoom, pan, tilt, crop change"
        ),
        help="VEO negative prompt (pass empty string to disable)",
    )
    p_submit.add_argument("--prompt", required=True, help="Motion prompt for video generation")
    p_submit.add_argument(
        "--image",
        default=None,
        help="Image URL or image identifier for API payload; defaults to run inputs/start_image.png",
    )
    p_submit.add_argument("--predict-dir", default=None, help="Predict directory; default latest or next")
    p_submit.set_defaults(func=cmd_submit_video)

    p_draft = sub.add_parser(
        "predict",
        help="Create next predict_00XX prompt from reason/trajectory_steps (no API call)",
    )
    p_draft.add_argument("--run-dir", default=None, help="Run directory")
    p_draft.add_argument("--run-id", default=None, help="Run id under COSMOS_RUNS_DIR")
    p_draft.add_argument("--predict-dir", default=None, help="Predict directory; default latest or next")
    p_draft.add_argument(
        "--prompt-mode",
        choices=["trajectory_from_reason", "self_planned_debug"],
        default="trajectory_from_reason",
        help="Prompt construction mode for video generation",
    )
    p_draft.set_defaults(func=cmd_draft_video_prompt)

    p_result = sub.add_parser("get-video-result", help="Stage 4: poll Wavespeed result")
    p_result.add_argument("--run-dir", default=None, help="Run directory")
    p_result.add_argument("--run-id", default=None, help="Run id under COSMOS_RUNS_DIR")
    p_result.add_argument(
        "--video-provider",
        choices=["wavespeed", "veo"],
        default=None,
        help="Provider override; defaults from submission metadata",
    )
    p_result.add_argument("--predict-dir", default=None, help="Predict directory; default latest")
    p_result.add_argument("--task-id", default=None, help="Task id; defaults from <predict_dir>/response.json")
    p_result.add_argument(
        "--wait",
        action="store_true",
        default=True,
        help="Poll until completed/failed (default true)",
    )
    p_result.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="Do a single status fetch without polling",
    )
    p_result.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=5,
        help="Polling interval while waiting",
    )
    p_result.add_argument(
        "--max-wait-seconds",
        type=int,
        default=600,
        help="Maximum wait time before returning current status",
    )
    p_result.set_defaults(func=cmd_get_video_result)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "command", None) == "import-image" and not (
        args.source_file or args.sim_session_dir or args.sim_runs_root
    ):
        args.sim_runs_root = "/home/ubuntu/Generative_STR/simtoolreal/runs"
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

