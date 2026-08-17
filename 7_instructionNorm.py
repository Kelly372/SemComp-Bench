








from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

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

from media_specs import (
    VIDEO_UPLOAD_FPS,
    add_upload_tier_argument,
    fit_image_to_frame,
    frame_size,
    normalize_upload_tier,
    upload_video_file,
)

from pipeline_paths import PROMPT_DIR, RESTATE_DIR


INSTRUCTION_NORM_MODES = ("short", "long")


ANCHOR_IMAGE_PLACEHOLDER = "(attached below as the first image)"
VIDEO_CLIP_PLACEHOLDER = "(attached below as the video clip)"


KEPT_NORM_COLS = (
    "video_instruction",
    "video_instruction_zh",
    "alignment_score",
)


def select_kept_parquet_columns(
    frame: pd.DataFrame,
    input_columns: List[str],
) -> pd.DataFrame:

    ordered = [c for c in input_columns if c in frame.columns]
    for c in KEPT_NORM_COLS:
        if c in frame.columns and c not in ordered:
            ordered.append(c)
    return frame.loc[:, ordered]


def paths_for_mode(output_base: str, mode: str) -> Tuple[str, str, str]:



    base = os.path.abspath(output_base)
    if base.endswith(".parquet"):
        out_dir = os.path.dirname(base) or "."
    elif os.path.isdir(base):
        out_dir = base
    else:
        out_dir = os.path.dirname(base) or "."

    out_full = os.path.join(out_dir, f"7_instructionNorm_output_{mode}.parquet")
    kept = os.path.join(out_dir, f"7_instructionNorm_{mode}.parquet")
    exclude = os.path.join(out_dir, f"7_instructionNorm_exclude_{mode}.parquet")
    return out_full, kept, exclude


def input_path_for_mode(input_base: str, mode: str) -> str:

    base = os.path.abspath(input_base)
    if base.endswith(".parquet"):
        out_dir = os.path.dirname(base) or "."
    elif os.path.isdir(base):
        out_dir = base
    else:
        out_dir = os.path.dirname(base) or "."
    return os.path.join(out_dir, f"6_videoClip_{mode}.parquet")


def _safe_model_dump(obj: Any) -> Any:
    if obj is None:
        return None
    for fn in ("model_dump", "dict"):
        f = getattr(obj, fn, None)
        if callable(f):
            try:
                return f()
            except Exception:
                pass
    return obj


def extract_output_text(resp: Any) -> str:
    r = _safe_model_dump(resp)
    if isinstance(r, dict):
        output = r.get("output")
        if isinstance(output, list):
            texts: List[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and isinstance(c.get("text"), str):
                            texts.append(c["text"])
                elif isinstance(item.get("text"), str):
                    texts.append(item["text"])
            if texts:
                return "\n".join(texts).strip()
        for k in ("output_text", "text", "content"):
            v = r.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

    output = getattr(resp, "output", None)
    if isinstance(output, list):
        texts2: List[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        texts2.append(t)
            t2 = getattr(item, "text", None)
            if isinstance(t2, str):
                texts2.append(t2)
        if texts2:
            return "\n".join(texts2).strip()

    try:
        return str(resp).strip()
    except Exception:
        return ""


def parse_json_from_text(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        return {}
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE
    )
    if fence_match:
        try:
            data = json.loads(fence_match.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    obj_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if obj_match:
        try:
            data = json.loads(obj_match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _normalize_string_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if isinstance(raw, list):
        out: List[str] = []
        for x in raw:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out
    return []


def _parse_alignment_score(raw: Any) -> int:
    if raw is None:
        return 0
    try:
        score = int(raw)
    except (TypeError, ValueError):
        return 0
    if score < 1 or score > 3:
        return 0
    return score


def normalize_result(parsed: Dict[str, Any]) -> Dict[str, Any]:

    video_instruction = str(
        parsed.get("adjusted_instruction_en")
        or parsed.get("video_instruction")
        or ""
    ).strip()
    video_instruction_zh = str(
        parsed.get("adjusted_instruction_zh")
        or parsed.get("video_instruction_zh")
        or ""
    ).strip()
    return {
        "grounding_elements": _normalize_string_list(parsed.get("grounding_elements")),
        "verb": str(parsed.get("verb", "")).strip(),
        "result": str(parsed.get("result", "")).strip(),
        "video_instruction": video_instruction,
        "video_instruction_zh": video_instruction_zh,
        "alignment_score": _parse_alignment_score(parsed.get("alignment_score")),
        "reason": str(parsed.get("reason", "")).strip(),
    }


def _norm_succeeded(normalized: Dict[str, Any]) -> bool:
    return bool(
        normalized.get("video_instruction", "").strip()
        or normalized.get("video_instruction_zh", "").strip()
    )


def render_prompt(
    prompt_template: str,
    instruction_en: str,
    instruction_zh: str,
) -> str:
    text = prompt_template
    text = text.replace("{anchor_image}", ANCHOR_IMAGE_PLACEHOLDER)
    text = text.replace("{video_clip}", VIDEO_CLIP_PLACEHOLDER)
    text = text.replace("{instruction_en}", instruction_en)
    text = text.replace("{instruction_zh}", instruction_zh)
    if "{instruction}" in text:
        text = text.replace("{instruction}", instruction_en)
    return text


def load_image_for_upload(path: str, upload_tier: str) -> Image.Image:
    return fit_image_to_frame(Image.open(path), upload_tier)


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
    df = pd.read_parquet(input_path)
    required_cols = [
        "youtube_url",
        "title",
        "domain",
        "category",
        "goal",
        "anchor_timestamp",
        "outcome_timestamp",
        "file_id",
        "response_id_3",
        "response_id_4",
        "instruction_zh",
        "instruction_en",
        "response_id_5",
        "clip_frame",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(
                f"数据中未找到 '{col}' 列, 请使用 6_videoClip 保留集作为输入"
            )
    return df


def _empty_norm_fields() -> Dict[str, Any]:
    return {
        "grounding_elements": [],
        "verb": "",
        "result": "",
        "video_instruction": "",
        "video_instruction_zh": "",
        "alignment_score": 0,
        "reason": "",
        "instruction_norm_raw_response": "",
        "instruction_norm_parse_ok": False,
        "response_id_7": "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage7_instructionNorm: 基于 6_videoClip 保留集、anchor 帧与缩放 video clip, "
            "对 instruction 做 VLM 归一化 (--mode short/long 决定读写文件名)"
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        default=os.path.join(RESTATE_DIR, "6_videoClip.parquet"),
        help=(
            "输入锚点(默认 pipeline/ReState/6_videoClip.parquet 所在目录); "
            "实际读取 6_videoClip_{mode}.parquet"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(RESTATE_DIR, "7_instructionNorm.parquet"),
        help=(
            "输出锚点(默认 pipeline/ReState/7_instructionNorm.parquet 所在目录); "
            "实际写入 7_instructionNorm_output_{mode}.parquet / "
            "7_instructionNorm_{mode}.parquet / 7_instructionNorm_exclude_{mode}.parquet"
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(INSTRUCTION_NORM_MODES),
        required=True,
        help="short 或 long, 与 6_videoClip 一致",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=PROMPT_DIR,
        help=f"包含 7_instructionNorm.txt 的目录 (默认: {PROMPT_DIR})",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=None,
        help="关键帧目录; 未指定时使用输出目录下的 stage3_frames",
    )
    add_upload_tier_argument(parser, default="low")
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="上传视频预处理 fps; 默认随 --upload-tier (low/mid=0.5, high=1.0)",
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
        help=f"VLM 模型名称, 默认: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理条数 (用于测试), 默认全部",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode
    set_model_name(args.model)

    upload_tier = normalize_upload_tier(args.upload_tier)
    video_fps = (
        args.video_fps
        if args.video_fps is not None
        else VIDEO_UPLOAD_FPS[upload_tier]
    )
    upload_w, upload_h = frame_size(upload_tier)

    input_path = input_path_for_mode(args.input, mode)
    out_full_path, kept_path, exclude_path = paths_for_mode(args.output, mode)

    df = load_parquet(input_path)
    input_columns = list(df.columns)

    prompt_file = os.path.join(args.prompt_dir, "7_instructionNorm.txt")
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt 文件未找到: {prompt_file}") from None

    frames_dir = args.frames_dir or os.path.join(
        os.path.dirname(os.path.abspath(kept_path)) or ".", "stage3_frames"
    )
    frames_dir = os.path.abspath(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_full_path) or ".", exist_ok=True)

    client = Ark(
        base_url=DEFAULT_BASE_URL,
        api_key=args.api_key,
    )

    total = len(df)
    limit = args.limit if args.limit is not None else total
    n_process = min(total, limit)
    results: list[dict[str, Any]] = []

    print(
        f"[{mode}] 输入: {input_path}\n"
        f"  上传规格: --upload-tier={upload_tier} → {upload_w}×{upload_h} (16:9), "
        f"video_fps={video_fps}\n"
        f"  输出: {kept_path}"
    )

    for idx in range(n_process):
        row = df.iloc[idx]
        youtube_url = str(row.get("youtube_url", ""))
        title = str(row.get("title", ""))
        instruction_en = str(row.get("instruction_en", "") or "").strip()
        instruction_zh = str(row.get("instruction_zh", "") or "").strip()
        clip_path = str(row.get("clip_frame", "") or "").strip()

        print(f"\n[{idx + 1}/{n_process}] [{mode}] instruction_norm: {title}")

        out_row: dict[str, Any] = row.to_dict()
        out_row["instruction_norm_error"] = ""

        if not instruction_en and not instruction_zh:
            out_row.update(_empty_norm_fields())
            out_row["instruction_norm_error"] = "missing instruction_en and instruction_zh"
            results.append(out_row)
            print("  跳过: missing instruction")
            continue

        if not clip_path or not os.path.isfile(clip_path):
            out_row.update(_empty_norm_fields())
            out_row["instruction_norm_error"] = (
                f"missing or invalid clip_frame: {clip_path!r}"
            )
            results.append(out_row)
            print(f"  跳过: {out_row['instruction_norm_error']}")
            continue

        ref_path = resolve_anchor_frame_path(
            frames_dir,
            youtube_url,
            str(row.get("anchor_timestamp", "") or ""),
        )
        if not ref_path:
            out_row.update(_empty_norm_fields())
            out_row["instruction_norm_error"] = (
                f"missing anchor frame under {frames_dir}"
            )
            results.append(out_row)
            print(f"  跳过: {out_row['instruction_norm_error']}")
            continue

        try:
            video_file_id = upload_video_file(
                client,
                clip_path,
                upload_tier,
                fps=video_fps,
            )
            user_text = render_prompt(
                prompt_template, instruction_en, instruction_zh
            )
            ref_im = load_image_for_upload(ref_path, upload_tier)
            ref_b64 = image_to_base64(ref_im)

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
            parsed_json = parse_json_from_text(output_text)
            normalized = normalize_result(parsed_json)
            parse_ok = bool(parsed_json)

            rid = getattr(resp, "id", None) or ""
            out_row["grounding_elements"] = normalized["grounding_elements"]
            out_row["verb"] = normalized["verb"]
            out_row["result"] = normalized["result"]
            out_row["video_instruction"] = normalized["video_instruction"]
            out_row["video_instruction_zh"] = normalized["video_instruction_zh"]
            out_row["alignment_score"] = normalized["alignment_score"]
            out_row["reason"] = normalized["reason"]
            out_row["instruction_norm_raw_response"] = output_text
            out_row["instruction_norm_parse_ok"] = parse_ok and _norm_succeeded(normalized)
            out_row["response_id_7"] = str(rid)
            if not _norm_succeeded(normalized):
                out_row["instruction_norm_error"] = (
                    "parse_failed_or_empty_adjusted_instruction"
                )
            print(
                f"  parse_ok={parse_ok} alignment_score="
                f"{normalized.get('alignment_score')} video_instruction="
                f"{normalized.get('video_instruction', '')!r}"
            )
            results.append(out_row)
        except Exception as e:
            print(f"  错误: instruction_norm 处理该视频时出错: {e}")
            import traceback

            traceback.print_exc()
            out_row.update(_empty_norm_fields())
            out_row["instruction_norm_error"] = str(e)
            results.append(out_row)

    if not results:
        print(f"\n7_instructionNorm [{mode}] 完成: 无记录被处理")
        return

    out_df = pd.DataFrame(results)
    out_df["instruction_norm_error"] = out_df.get(
        "instruction_norm_error", pd.Series([""] * len(out_df))
    ).fillna("").astype(str)

    kept_mask = out_df["instruction_norm_error"].str.strip() == ""
    kept_df = out_df.loc[kept_mask].reset_index(drop=True)
    exclude_df = out_df.loc[~kept_mask].reset_index(drop=True)

    for p in (out_full_path, kept_path, exclude_path):
        if os.path.exists(p):
            os.remove(p)

    out_df.to_parquet(out_full_path, index=False)
    select_kept_parquet_columns(kept_df, input_columns).to_parquet(
        kept_path, index=False
    )
    exclude_df.to_parquet(exclude_path, index=False)

    kept_cols = list(select_kept_parquet_columns(kept_df, input_columns).columns)
    print(
        f"\n7_instructionNorm [{mode}] 完成: 全量 {len(out_df)} 条, "
        f"保留 {len(kept_df)} 条, 排除 {len(exclude_df)} 条\n"
        f"  全量(VLM 全列) -> {out_full_path}\n"
        f"  保留(输入+{list(KEPT_NORM_COLS)}) -> {kept_path} "
        f"({len(kept_cols)} 列)\n"
        f"  排除 -> {exclude_path}"
    )


if __name__ == "__main__":
    main()
