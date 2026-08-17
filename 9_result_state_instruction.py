













from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image
from volcenginesdkarkruntime import Ark

from agent_process import (
    API_KEY as DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MODEL_NAME,
    anchor_frame_disk_path,
    extract_video_id_from_url,
    image_to_base64,
    set_model_name,
)
import importlib

_align_type = importlib.import_module("8_align_type")
_normalize_string_list = _align_type._normalize_string_list
extract_output_text = _align_type.extract_output_text
parse_json_from_text = _align_type.parse_json_from_text
_resolve_clip_path_base = _align_type.resolve_clip_path
from media_specs import (
    add_upload_tier_argument,
    fit_image_to_frame,
    normalize_upload_tier,
    upload_video_file,
)
from pipeline_paths import DATASET_DIR, PROMPT_DIR

DEFAULT_DATASET_DIR = DATASET_DIR
DEFAULT_INPUT_PARQUET = os.path.join(DEFAULT_DATASET_DIR, "8_align_type.parquet")
DEFAULT_OUTPUT_PARQUET = os.path.join(
    DEFAULT_DATASET_DIR, "9_result_state_instruction.parquet"
)
DEFAULT_FRAMES_DIR = os.path.join(DEFAULT_DATASET_DIR, "stage3_frames")
DEFAULT_VIDEO_CLIP_DIR = os.path.join(DEFAULT_DATASET_DIR, "video_clip")
DEFAULT_VIDEO_FPS = 2.0

ANCHOR_IMAGE_PLACEHOLDER = "(attached below as the first image)"
OUTCOME_VIDEO_PLACEHOLDER = "(attached below as the outcome video clip)"

RESULT_STATE_OUTPUT_COLUMNS = [
    "video_description_en",
    "video_description_zh",
    "reference_image_description_en",
    "reference_image_description_zh",
    "preserve_ignore_en",
    "preserve_ignore_zh",
]


def resolve_preserved_attributes(row: pd.Series) -> List[str]:
    for col in (
        "align_type_preserved_attributes",
        "preserved_attributes",
    ):
        if col in row.index:
            return _normalize_string_list(row.get(col))
    return []


def resolve_row_text_fields(row: pd.Series) -> Dict[str, str]:
    return {
        "video_instruction": str(row.get("video_instruction", "") or "").strip(),
        "video_instruction_zh": str(
            row.get("video_instruction_zh", "") or ""
        ).strip(),
        "goal": str(row.get("goal", "") or "").strip(),
        "align_type_label": str(row.get("align_type_label", "") or "").strip(),
    }


def row_has_any_instruction(row: pd.Series) -> bool:
    fields = resolve_row_text_fields(row)
    return bool(fields["video_instruction"] or fields["video_instruction_zh"])


def render_prompt(
    prompt_template: str,
    *,
    video_instruction: str,
    video_instruction_zh: str,
    goal: str,
    align_type_label: str,
    preserved_attributes: List[str],
) -> str:
    attrs_json = json.dumps(preserved_attributes, ensure_ascii=False)
    text = prompt_template
    text = text.replace("{video_instruction}", video_instruction)
    text = text.replace("{video_instruction_zh}", video_instruction_zh)
    text = text.replace("{goal}", goal)
    text = text.replace("{align_type_label}", align_type_label)
    text = text.replace("{align_type_preserved_attributes}", attrs_json)
    text = text.replace("{anchor_image}", ANCHOR_IMAGE_PLACEHOLDER)
    text = text.replace("{outcome_video}", OUTCOME_VIDEO_PLACEHOLDER)
    return text


def _strip_json_str_field(data: Dict[str, Any], key: str) -> str:
    val = data.get(key)
    return val.strip() if isinstance(val, str) else ""


def parse_result_state_response(text: str) -> Dict[str, str]:
    empty = {col: "" for col in RESULT_STATE_OUTPUT_COLUMNS}
    data = parse_json_from_text(text)
    if not data:
        return empty
    return {
        col: _strip_json_str_field(data, col) for col in RESULT_STATE_OUTPUT_COLUMNS
    }


def _empty_restate_fields() -> Dict[str, Any]:
    fields: Dict[str, Any] = {col: "" for col in RESULT_STATE_OUTPUT_COLUMNS}
    fields["result_state_raw_response"] = ""
    fields["response_id_restate"] = ""
    return fields


def row_has_restate_result(row: Dict[str, Any]) -> bool:
    return any(
        str(row.get(col, "") or "").strip() for col in RESULT_STATE_OUTPUT_COLUMNS
    )


def load_image_for_upload(path: str, upload_tier: str) -> Image.Image:
    return fit_image_to_frame(Image.open(path), upload_tier)


def read_parquet_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        import polars as pl

        return pl.read_parquet(path).to_pandas()


def resolve_clip_path(row: pd.Series, video_clip_dir: str) -> str:







    clip_frame = str(row.get("clip_frame", "") or "").strip()
    path = _resolve_clip_path_base(clip_frame, video_clip_dir)
    if path and os.path.isfile(path):
        return path

    video_id = extract_video_id_from_url(str(row.get("youtube_url", "") or ""))
    clip_start = str(row.get("clip_start", "") or "").strip()
    clip_end = str(row.get("clip_end", "") or "").strip()
    if not video_id or not clip_start or not clip_end:
        return ""

    name = (
        f"{video_id}_{clip_start.replace(':', '_')}_"
        f"{clip_end.replace(':', '_')}.mp4"
    )
    candidate = os.path.join(os.path.abspath(video_clip_dir), name)
    return candidate if os.path.isfile(candidate) else ""


def resolve_anchor_frame_path(
    frames_dir: str,
    youtube_url: str,
    anchor_timestamp: str,
) -> Optional[str]:
    video_id = extract_video_id_from_url(youtube_url)
    if not video_id:
        return None
    ref_ts = str(anchor_timestamp or "").strip()
    if not ref_ts:
        return None
    ref_path = anchor_frame_disk_path(frames_dir, video_id, ref_ts)
    return ref_path if os.path.isfile(ref_path) else None


def load_parquet(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"未找到输入文件: {input_path}")
    df = read_parquet_safe(input_path)
    required = ["youtube_url", "anchor_timestamp", "clip_frame", "clip_start", "clip_end"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"数据中缺少列 {missing}，请使用含 anchor 与 outcome clip 的上游 parquet"
        )
    return df


def load_checkpoint(output_path: str, n_process: int) -> List[Dict[str, Any]]:
    if not os.path.exists(output_path):
        return []
    try:
        ckpt = read_parquet_safe(output_path)
    except Exception as e:
        print(f"  警告: 无法读取已有输出 {output_path}: {e}")
        return []
    records = ckpt.to_dict(orient="records")
    if len(records) > n_process:
        records = records[:n_process]
    return records


def save_checkpoint(output_path: str, results: List[Dict[str, Any]]) -> None:
    if not results:
        return
    out_df = pd.DataFrame(results)
    tmp_path = f"{output_path}.tmp"
    out_df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, output_path)


def _store_result(
    results: List[Dict[str, Any]], idx: int, out_row: Dict[str, Any]
) -> None:
    if idx < len(results):
        results[idx] = out_row
    elif idx == len(results):
        results.append(out_row)
    else:
        raise IndexError(
            f"结果列表出现空洞: idx={idx}, len(results)={len(results)}"
        )


def print_restate_progress(
    idx: int,
    total: int,
    *,
    status: str = "",
    instruction: str = "",
    result_en: str = "",
    result_zh: str = "",
    extra: str = "",
) -> None:
    inst_preview = (instruction[:60] + "…") if len(instruction) > 60 else instruction
    en_preview = (result_en[:40] + "…") if len(result_en) > 40 else result_en
    zh_preview = (result_zh[:40] + "…") if len(result_zh) > 40 else result_zh
    msg = f"[{idx}/{total}]"
    if inst_preview:
        msg += f" in={inst_preview!r}"
    if en_preview:
        msg += f" en={en_preview!r}"
    if zh_preview:
        msg += f" zh={zh_preview!r}"
    if status:
        msg += f" | {status}"
    if extra:
        msg += f" | {extra}"
    print(msg, flush=True)


RESULT_STATE_PRINT_SECTIONS_ZH = [
    ("video_description_zh", "video_description_zh"),
    ("reference_image_description_zh", "reference_image_description_zh"),
    ("preserve_ignore_zh", "preserve_ignore_zh"),
]


def print_result_state_detail(
    idx: int,
    total: int,
    parsed: Dict[str, str],
    *,
    status: str = "OK",
    youtube_url: str = "",
    video_instruction: str = "",
    video_instruction_zh: str = "",
    goal: str = "",
    align_type_label: str = "",
    preserved_attributes: Optional[List[str]] = None,
    raw_response: str = "",
) -> None:

    header = f"[{idx}/{total}] result_state | {status}"
    if youtube_url:
        header += f" | {youtube_url}"
    print(header, flush=True)
    if video_instruction_zh:
        print(f"video_instruction_zh: {video_instruction_zh}", flush=True)
    if goal:
        print(f"goal: {goal}", flush=True)
    if align_type_label:
        print(f"align_type_label: {align_type_label}", flush=True)
    if preserved_attributes:
        attrs = json.dumps(preserved_attributes, ensure_ascii=False)
        print(f"preserved_attributes: {attrs}", flush=True)
    for key, label in RESULT_STATE_PRINT_SECTIONS_ZH:
        value = (parsed.get(key, "") or "").strip()
        print(f"--- {label} ---", flush=True)
        print(value if value else "(empty)", flush=True)
    if status == "PARSE_FAIL" and raw_response:
        print("--- raw_response ---", flush=True)
        print(raw_response, flush=True)
    print("=" * 60, flush=True)


def call_restate_vlm(
    client: Ark,
    user_text: str,
    ref_b64: str,
    video_file_id: str,
) -> tuple[str, str]:
    resp = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{ref_b64}",
                    },
                    {
                        "type": "input_video",
                        "file_id": video_file_id,
                    },
                ],
            }
        ],
        thinking={"type": "enabled"},
    )
    output_text = extract_output_text(resp)
    rid = str(getattr(resp, "id", None) or "")
    return output_text, rid


def process_parquet(
    *,
    input_path: str,
    output_path: str,
    prompt_template: str,
    client: Ark,
    frames_dir: str,
    video_clip_dir: str,
    upload_tier: str,
    video_fps: float,
    limit: Optional[int],
    resume: bool,
) -> None:
    df = load_parquet(input_path)
    total = len(df)
    n_process = min(total, limit if limit is not None else total)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    if resume:
        results = load_checkpoint(output_path, n_process)
    else:
        results = []
        if os.path.exists(output_path):
            os.remove(output_path)

    done_before = sum(1 for r in results if row_has_restate_result(r))
    print(
        f"\n处理: {input_path}\n"
        f"  输出: {output_path}\n"
        f"  条数: {n_process}/{total}\n"
        f"  上传规格: --upload-tier={upload_tier}, video_fps={video_fps}\n"
        f"  frames_dir: {frames_dir}\n"
        f"  video_clip_dir: {video_clip_dir}\n"
        f"  断点续跑: {'开启' if resume else '关闭'}"
        + (f" (已有 {len(results)} 条, 已完成 {done_before} 条)" if results else ""),
        flush=True,
    )

    for idx in range(n_process):
        row = df.iloc[idx]
        youtube_url = str(row.get("youtube_url", "") or "").strip()
        fields = resolve_row_text_fields(row)

        if idx < len(results) and row_has_restate_result(results[idx]):
            prev = results[idx]
            print_result_state_detail(
                idx + 1,
                n_process,
                {col: str(prev.get(col, "") or "") for col in RESULT_STATE_OUTPUT_COLUMNS},
                status="SKIP done",
                youtube_url=youtube_url,
                video_instruction=fields["video_instruction"],
                video_instruction_zh=fields["video_instruction_zh"],
                goal=fields["goal"],
                align_type_label=fields["align_type_label"],
                preserved_attributes=resolve_preserved_attributes(row),
            )
            continue

        preserved = resolve_preserved_attributes(row)
        out_row: Dict[str, Any] = row.to_dict()
        out_row.update(_empty_restate_fields())

        if not row_has_any_instruction(row):
            print_restate_progress(
                idx + 1,
                n_process,
                status="SKIP missing instruction",
                extra=youtube_url,
            )
            _store_result(results, idx, out_row)
            save_checkpoint(output_path, results)
            continue

        clip_path = resolve_clip_path(row, video_clip_dir)
        if not clip_path or not os.path.isfile(clip_path):
            print_restate_progress(
                idx + 1,
                n_process,
                status="SKIP missing clip",
                extra=youtube_url,
            )
            _store_result(results, idx, out_row)
            save_checkpoint(output_path, results)
            continue

        ref_path = resolve_anchor_frame_path(
            frames_dir,
            youtube_url,
            str(row.get("anchor_timestamp", "") or ""),
        )
        if not ref_path:
            print_restate_progress(
                idx + 1,
                n_process,
                status="SKIP missing anchor frame",
                extra=youtube_url,
            )
            _store_result(results, idx, out_row)
            save_checkpoint(output_path, results)
            continue

        print(
            f"[{idx + 1}/{n_process}] PROCESSING | {youtube_url}",
            flush=True,
        )

        try:
            video_file_id = upload_video_file(
                client,
                clip_path,
                upload_tier,
                fps=video_fps,
            )
            user_text = render_prompt(
                prompt_template,
                video_instruction=fields["video_instruction"],
                video_instruction_zh=fields["video_instruction_zh"],
                goal=fields["goal"],
                align_type_label=fields["align_type_label"],
                preserved_attributes=preserved,
            )
            ref_im = load_image_for_upload(ref_path, upload_tier)
            ref_b64 = image_to_base64(ref_im)
            output_text, rid = call_restate_vlm(
                client, user_text, ref_b64, video_file_id
            )
            parsed = parse_result_state_response(output_text)
            for col in RESULT_STATE_OUTPUT_COLUMNS:
                out_row[col] = parsed[col]
            out_row["result_state_raw_response"] = output_text
            out_row["response_id_restate"] = rid
            status = "OK" if row_has_restate_result(out_row) else "PARSE_FAIL"
            print_result_state_detail(
                idx + 1,
                n_process,
                parsed,
                status=status,
                youtube_url=youtube_url,
                video_instruction=fields["video_instruction"],
                video_instruction_zh=fields["video_instruction_zh"],
                goal=fields["goal"],
                align_type_label=fields["align_type_label"],
                preserved_attributes=preserved,
                raw_response=output_text,
            )
        except Exception as e:
            print(
                f"[{idx + 1}/{n_process}] ERROR | {youtube_url} | {e}",
                flush=True,
            )
            import traceback

            traceback.print_exc()

        _store_result(results, idx, out_row)
        save_checkpoint(output_path, results)

    if not results:
        print(f"完成: 无记录 -> {output_path}")
        return

    out_df = pd.DataFrame(results)
    ok = out_df.apply(
        lambda r: row_has_restate_result(r.to_dict()), axis=1
    ).sum()
    print(
        f"完成: 共 {len(out_df)} 条, 已改写 {ok} 条 -> {output_path}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "result_state_instruction: 基于 anchor 图、outcome 视频与对齐信息, "
            "生成 video_description / reference_image_description / "
            "preserve_ignore 各 en/zh 共 6 列"
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT_PARQUET,
        help=f"上游 parquet 路径 (默认: {DEFAULT_INPUT_PARQUET})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_PARQUET,
        help=f"输出路径 (默认: {DEFAULT_OUTPUT_PARQUET})",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=PROMPT_DIR,
        help=f"包含 9_result_state_instruction.txt 的目录 (默认: {PROMPT_DIR})",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=DEFAULT_FRAMES_DIR,
        help=f"anchor 关键帧目录 (默认: {DEFAULT_FRAMES_DIR})",
    )
    parser.add_argument(
        "--video_clip_dir",
        type=str,
        default=DEFAULT_VIDEO_CLIP_DIR,
        help=f"outcome 短视频目录 (默认: {DEFAULT_VIDEO_CLIP_DIR})",
    )
    add_upload_tier_argument(parser, default="low")
    parser.add_argument(
        "--video-fps",
        type=float,
        default=DEFAULT_VIDEO_FPS,
        help=f"上传视频预处理 fps (默认: {DEFAULT_VIDEO_FPS})",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=DEFAULT_API_KEY,
        help="VLM API Key（默认读取 VLM_API_KEY）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"VLM 模型名称 (默认: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理条数 (用于测试), 默认全部",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="忽略已有输出 parquet，从头处理并覆盖",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_model_name(args.model)

    upload_tier = normalize_upload_tier(args.upload_tier)
    video_fps = float(args.video_fps)

    prompt_file = os.path.join(args.prompt_dir, "9_result_state_instruction.txt")
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt 文件未找到: {prompt_file}") from None

    client = Ark(
        base_url=DEFAULT_BASE_URL,
        api_key=args.api_key,
    )
    resume = not args.no_resume
    frames_dir = os.path.abspath(args.frames_dir)
    video_clip_dir = os.path.abspath(args.video_clip_dir)

    process_parquet(
        input_path=os.path.abspath(args.input),
        output_path=os.path.abspath(args.output),
        prompt_template=prompt_template,
        client=client,
        frames_dir=frames_dir,
        video_clip_dir=video_clip_dir,
        upload_tier=upload_tier,
        video_fps=video_fps,
        limit=args.limit,
        resume=resume,
    )


if __name__ == "__main__":
    main()
