









from __future__ import annotations

import argparse
import json
import os
import re
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
from media_specs import (
    VIDEO_UPLOAD_FPS,
    add_upload_tier_argument,
    fit_image_to_frame,
    normalize_upload_tier,
    upload_video_file,
)
from pipeline_paths import PROMPT_DIR, RESTATE_DIR

DEFAULT_VIDEO_CLIP_DIR = os.path.join(RESTATE_DIR, "video_clip")
DEFAULT_FRAMES_DIR = os.path.join(RESTATE_DIR, "stage3_frames")
DEFAULT_INPUT_PARQUET = os.path.join(
    RESTATE_DIR, "7_instructionNorm_short.parquet"
)
DEFAULT_OUTPUT_PARQUET = os.path.join(RESTATE_DIR, "8_align_type.parquet")

ANCHOR_IMAGE_PLACEHOLDER = "(attached below as the first image)"
OUTCOME_VIDEO_PLACEHOLDER = "(attached below as the outcome video clip)"

TYPE_BY_LETTER: Dict[str, str] = {
    "A": "Object Element",
    "B": "Identity",
    "C": "Scene",
    "D": "Object Appearance",
}


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
    cleaned = (text or "").strip()
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
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    elif isinstance(raw, tuple):
        raw = list(raw)
    if isinstance(raw, list):
        out: List[str] = []
        for x in raw:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out
    return []


def _normalize_answer(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw).strip().upper()
    if not s:
        return ""
    letter = s[0]
    return letter if letter in TYPE_BY_LETTER else ""


def parse_align_type_response(text: str) -> Dict[str, Any]:

    empty: Dict[str, Any] = {
        "align_type_answer": "",
        "align_type_label": "",
        "align_type_preserved_attributes": [],
    }
    data = parse_json_from_text(text)
    if not data:
        return empty

    answer = _normalize_answer(data.get("answer"))
    attrs = _normalize_string_list(data.get("preserved_attributes"))
    if answer:
        label = TYPE_BY_LETTER.get(answer, "")
    else:
        label = str(data.get("type") or "").strip()

    return {
        "align_type_answer": answer,
        "align_type_label": label,
        "align_type_preserved_attributes": attrs,
    }


def print_align_progress(
    idx: int,
    total: int,
    *,
    status: str = "",
    answer: str = "",
    align_type: str = "",
    attributes: Optional[List[str]] = None,
    extra: str = "",
) -> None:
    attrs = _normalize_string_list(attributes)
    attrs_str = ", ".join(attrs) if attrs else "-"
    type_part = f"{answer}({align_type})" if answer or align_type else "-"
    msg = f"[{idx}/{total}] align_type={type_part} | attributes=[{attrs_str}]"
    if status:
        msg += f" | {status}"
    if extra:
        msg += f" | {extra}"
    print(msg, flush=True)


def resolve_instruction(row: pd.Series) -> str:
    for col in (
        "video_instruction",
        "instruction_en",
        "instruction",
        "video_instruction_zh",
        "instruction_zh",
    ):
        val = str(row.get(col, "") or "").strip()
        if val:
            return val
    return ""


def render_prompt(prompt_template: str, instruction: str) -> str:
    text = prompt_template
    text = text.replace("{instruction}", instruction)
    text = text.replace("{anchor_image}", ANCHOR_IMAGE_PLACEHOLDER)
    text = text.replace("{outcome video}", OUTCOME_VIDEO_PLACEHOLDER)
    text = text.replace("{outcome_video}", OUTCOME_VIDEO_PLACEHOLDER)
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
    required = ["youtube_url", "anchor_timestamp", "clip_frame"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"数据中缺少列 {missing}，请使用含 anchor 与 outcome clip 的上游 parquet"
        )
    return df


def _empty_align_fields() -> Dict[str, Any]:
    return {
        "align_type_answer": "",
        "align_type_label": "",
        "align_type_preserved_attributes": [],
        "align_type_raw_response": "",
        "response_id_align": "",
    }


def row_has_align_result(row: Dict[str, Any]) -> bool:
    return bool(str(row.get("align_type_answer", "") or "").strip())


def load_checkpoint(output_path: str, n_process: int) -> List[Dict[str, Any]]:
    if not os.path.exists(output_path):
        return []
    try:
        ckpt = pd.read_parquet(output_path)
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


def default_output_path(input_path: str) -> str:
    d = os.path.dirname(os.path.abspath(input_path)) or "."
    return os.path.join(d, "8_align_type.parquet")


def resolve_clip_path(clip_frame: str, video_clip_dir: str) -> str:




    raw = str(clip_frame or "").strip()
    if not raw:
        return ""
    if os.path.isfile(raw):
        return raw

    clip_root = os.path.abspath(video_clip_dir)
    base = os.path.basename(raw)

    for candidate in (
        os.path.join(clip_root, "short", base),
        os.path.join(clip_root, base),
    ):
        if os.path.isfile(candidate):
            return candidate

    norm = raw.replace("\\", "/")
    marker = "video_clip/"
    if marker in norm:
        suffix = norm.split(marker, 1)[1]
        candidate = os.path.join(clip_root, suffix)
        if os.path.isfile(candidate):
            return candidate

    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "align_type: 基于上游 parquet、anchor 帧与 outcome video clip, "
            "对 preservation type 做 VLM 分类 (A/B/C/D)"
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
        help=f"包含 8_align_type.txt 的目录 (默认: {PROMPT_DIR})",
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
        help=(
            f"短视频 clip 根目录, 实际文件在 <dir>/short/ 下 "
            f"(默认: {DEFAULT_VIDEO_CLIP_DIR})"
        ),
    )
    add_upload_tier_argument(parser, default="low")
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="上传视频预处理 fps; 默认随 --upload-tier",
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
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="忽略已有 8_align_type.parquet，从头处理并覆盖",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_model_name(args.model)

    upload_tier = normalize_upload_tier(args.upload_tier)
    video_fps = (
        args.video_fps
        if args.video_fps is not None
        else VIDEO_UPLOAD_FPS[upload_tier]
    )

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(
        args.output or default_output_path(input_path)
    )

    df = load_parquet(input_path)

    prompt_file = os.path.join(args.prompt_dir, "8_align_type.txt")
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt 文件未找到: {prompt_file}") from None

    video_clip_dir = os.path.abspath(args.video_clip_dir)
    frames_dir = os.path.abspath(args.frames_dir)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    client = Ark(
        base_url=DEFAULT_BASE_URL,
        api_key=args.api_key,
    )

    total = len(df)
    limit = args.limit if args.limit is not None else total
    n_process = min(total, limit)
    resume = not args.no_resume

    if resume:
        results = load_checkpoint(output_path, n_process)
    else:
        results = []
        if os.path.exists(output_path):
            os.remove(output_path)

    done_before = sum(1 for r in results if row_has_align_result(r))
    print(
        f"输入: {input_path}\n"
        f"  上传规格: --upload-tier={upload_tier}, video_fps={video_fps}\n"
        f"  video_clip_dir: {video_clip_dir}\n"
        f"  frames_dir: {frames_dir}\n"
        f"  输出: {output_path}\n"
        f"  断点续跑: {'开启' if resume else '关闭'}"
        + (f" (已有 {len(results)} 条, 已分类 {done_before} 条)" if results else ""),
        flush=True,
    )

    for idx in range(n_process):
        row = df.iloc[idx]
        youtube_url = str(row.get("youtube_url", "") or "").strip()

        if idx < len(results) and row_has_align_result(results[idx]):
            prev = results[idx]
            print_align_progress(
                idx + 1,
                n_process,
                status="SKIP done",
                answer=str(prev.get("align_type_answer", "") or ""),
                align_type=str(prev.get("align_type_label", "") or ""),
                attributes=prev.get("align_type_preserved_attributes"),
                extra=youtube_url,
            )
            continue
        instruction = resolve_instruction(row)
        clip_path = resolve_clip_path(
            str(row.get("clip_frame", "") or ""),
            video_clip_dir,
        )

        out_row: Dict[str, Any] = row.to_dict()
        out_row.update(_empty_align_fields())

        if not instruction:
            print_align_progress(
                idx + 1,
                n_process,
                status="SKIP missing instruction",
                extra=youtube_url,
            )
            _store_result(results, idx, out_row)
            save_checkpoint(output_path, results)
            continue

        if not clip_path or not os.path.isfile(clip_path):
            print_align_progress(
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
            print_align_progress(
                idx + 1,
                n_process,
                status="SKIP missing anchor frame",
                extra=youtube_url,
            )
            _store_result(results, idx, out_row)
            save_checkpoint(output_path, results)
            continue

        try:
            video_file_id = upload_video_file(
                client,
                clip_path,
                upload_tier,
                fps=video_fps,
            )
            user_text = render_prompt(prompt_template, instruction)
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
            parsed = parse_align_type_response(output_text)
            rid = getattr(resp, "id", None) or ""
            out_row["align_type_answer"] = parsed["align_type_answer"]
            out_row["align_type_label"] = parsed["align_type_label"]
            out_row["align_type_preserved_attributes"] = parsed[
                "align_type_preserved_attributes"
            ]
            out_row["align_type_raw_response"] = output_text
            out_row["response_id_align"] = str(rid)
            status = "OK" if parsed.get("align_type_answer") else "PARSE_FAIL"
            print_align_progress(
                idx + 1,
                n_process,
                status=status,
                answer=str(parsed.get("align_type_answer", "") or ""),
                align_type=str(parsed.get("align_type_label", "") or ""),
                attributes=parsed.get("align_type_preserved_attributes"),
                extra=youtube_url,
            )
            _store_result(results, idx, out_row)
            save_checkpoint(output_path, results)
        except Exception as e:
            print_align_progress(
                idx + 1,
                n_process,
                status=f"ERROR {e}",
                extra=youtube_url,
            )
            import traceback

            traceback.print_exc()
            _store_result(results, idx, out_row)
            save_checkpoint(output_path, results)

    if not results:
        print("\nalign_type 完成: 无记录被处理")
        return

    out_df = pd.DataFrame(results)
    classified = out_df["align_type_answer"].astype(str).str.strip().ne("").sum()
    with_attrs = out_df["align_type_preserved_attributes"].apply(
        lambda x: isinstance(x, list) and len(x) > 0
    ).sum()
    print(
        f"\nalign_type 完成: 共 {len(out_df)} 条, "
        f"已分类 {classified} 条, "
        f"含 preserved_attributes {with_attrs} 条 -> {output_path}",
        flush=True,
    )


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


if __name__ == "__main__":
    main()
