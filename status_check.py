







from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from pipeline_paths import PROMPT_DIR, RESTATE_DIR

CHECK_IMAGE_WIDTH = 281
CHECK_IMAGE_HEIGHT = 160

DEFAULT_INPUT_JSON = os.path.join(RESTATE_DIR, "video_clips_short_rewrite_result.json")
DEFAULT_IMAGES_DIR = os.path.join(RESTATE_DIR, "stage3_frames")
DEFAULT_OUTPUT_JSON = os.path.join(RESTATE_DIR, "video_clips_status_check_result.json")
DEFAULT_PROMPT_PATH = os.path.join(PROMPT_DIR, "status_check_prompt.txt")


NEUTRAL_FILLER_IMAGE_A = (
    "corresponding to the first image attached after this text (Image A)"
)
NEUTRAL_FILLER_IMAGE_B = (
    "corresponding to the second image attached after this text (Image B)"
)


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


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON: {path}")
    return data


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def output_json_path_with_idx_range(
    output_json: str,
    start_idx: int,
    end_idx: Optional[int],
    total_in_file: int,
) -> str:
    path = os.path.abspath(output_json)
    dirname, basename = os.path.dirname(path), os.path.basename(path)
    stem, ext = os.path.splitext(basename)
    if ext.lower() != ".json":
        stem, ext = basename, ".json"
    s_lab = max(0, int(start_idx))
    if end_idx is None:
        e_lab = (total_in_file - 1) if total_in_file > 0 else 0
    else:
        e_lab = int(end_idx)
    new_name = f"{stem}_{s_lab}_{e_lab}{ext}"
    return os.path.join(dirname, new_name) if dirname else new_name


def render_status_check_prompt(
    prompt_template: str,
    image_a_filler: str,
    image_b_filler: str,
    instruction: str,
) -> str:




    n_brace = prompt_template.count("{}")
    if n_brace < 2:
        raise ValueError(
            f"status_check_prompt.txt 中 `{{}}` 至少需要 2 个（当前 {n_brace} 个），"
            "用于 Image A 与 Image B。"
        )
    s = prompt_template.replace("{}", image_a_filler, 1)
    s = s.replace("{}", image_b_filler, 1)
    if "{instruction}" not in s:
        raise ValueError("status_check_prompt.txt 缺少 `{instruction}` 占位符")
    s = s.replace("{instruction}", instruction)
    return s


def resolve_reference_image_path(images_dir: str, uid: Any) -> Optional[str]:
    uid_s = str(uid).strip()
    if not uid_s or not os.path.isdir(images_dir):
        return None
    for pat in (
        os.path.join(images_dir, f"{uid_s}_*_anchor.*"),
        os.path.join(images_dir, f"{uid_s}_anchor.*"),
        os.path.join(images_dir, f"{uid_s}_*_reference.*"),
        os.path.join(images_dir, f"{uid_s}_reference.*"),
    ):
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"):
        p = os.path.join(images_dir, f"{uid_s}_reference{ext}")
        if os.path.isfile(p):
            return p
    return None


def resolve_outcome_image_path(images_dir: str, uid: Any) -> Optional[str]:
    uid_s = str(uid).strip()
    if not uid_s or not os.path.isdir(images_dir):
        return None
    for pat in (
        os.path.join(images_dir, f"{uid_s}_*_outcome.*"),
        os.path.join(images_dir, f"{uid_s}_outcome.*"),
    ):
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"):
        p = os.path.join(images_dir, f"{uid_s}_outcome{ext}")
        if os.path.isfile(p):
            return p
    return None


def load_image_for_check(path: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS  # type: ignore[attr-defined]
    return im.resize((CHECK_IMAGE_WIDTH, CHECK_IMAGE_HEIGHT), resample)


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


def _slot_from_label(label: str) -> Optional[str]:




    s = label.strip().lower()
    if not s:
        return None
    if re.search(r"\bimage\s*a\b|^a$|^image\s+a\b|first|第一张|首图", s):
        return "A"
    if re.search(r"\bimage\s*b\b|^b$|^image\s+b\b|second|第二张|次图", s):
        return "B"

    if s in ("a", "image a"):
        return "A"
    if s in ("b", "image b"):
        return "B"
    if "reference" in s and "outcome" not in s:

        if "a" in s and "b" not in s:
            return "A"
        if "b" in s and "a" not in s:
            return "B"
    if "outcome" in s and "reference" not in s:
        if "b" in s and "a" not in s:
            return "B"
        if "a" in s and "b" not in s:
            return "A"
    return None


def parse_confidence(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        if 1 <= raw <= 3:
            return raw
        return None
    if isinstance(raw, float):
        i = int(round(raw))
        if 1 <= i <= 3:
            return i
        return None
    if isinstance(raw, str):
        m = re.search(r"[123]", raw.strip())
        if m:
            return int(m.group(0))
    return None


def local_verify_status(
    parsed: Dict[str, Any],
    image_a_is_reference: bool,
) -> Tuple[str, bool]:





    ref_label = str(parsed.get("reference_image", "")).strip()
    out_label = str(parsed.get("outcome_image", "")).strip()
    ref_slot = _slot_from_label(ref_label)
    out_slot = _slot_from_label(out_label)
    if ref_slot is None or out_slot is None:
        return "unknown", False
    if image_a_is_reference:
        ok = ref_slot == "A" and out_slot == "B"
    else:
        ok = ref_slot == "B" and out_slot == "A"
    return ("right" if ok else "wrong"), True


def _import_vlm_runtime():
    try:
        from agent_process import (  # type: ignore
            API_KEY,
            DEFAULT_BASE_URL,
            MODEL_NAME,
            image_to_base64,
        )
    except Exception as e:
        raise RuntimeError("无法从本目录的 agent_process.py 加载 VLM 配置。") from e
    return image_to_base64, API_KEY, DEFAULT_BASE_URL, MODEL_NAME


def instruction_for_item(item: Dict[str, Any]) -> str:

    return str(item.get("rewritten_instruction", "")).strip()


def run_status_check(
    input_json: str,
    prompt_path: str,
    output_json: str,
    images_dir: str,
    limit: Optional[int] = None,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    minimal_output: bool = False,
    rng_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    try:
        from volcenginesdkarkruntime import Ark  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 volcenginesdkarkruntime,无法调用模型。") from e

    image_to_base64, api_key, base_url, model_name = _import_vlm_runtime()
    client = Ark(
        base_url=base_url,
        api_key=api_key,
    )

    data = load_json(input_json)
    total_in_file = len(data)
    output_json = output_json_path_with_idx_range(
        output_json, start_idx, end_idx, total_in_file
    )
    slice_start = max(0, int(start_idx))
    if end_idx is None:
        slice_end_excl = total_in_file
    else:
        end_i = int(end_idx)
        if end_i < slice_start:
            raise ValueError(
                f"end_idx ({end_i}) 不能小于 start_idx ({slice_start})（区间为闭区间 [start_idx, end_idx]）。"
            )
        slice_end_excl = min(total_in_file, end_i + 1)
    data = data[slice_start:slice_end_excl]
    if limit is not None and limit > 0:
        data = data[:limit]

    end_desc = "EOF" if end_idx is None else str(end_idx)
    print(
        f"Slice: start_idx={slice_start}, end_idx (inclusive)={end_desc}, "
        f"after slice+limit: n={len(data)} (full file n={total_in_file})"
    )
    print(f"Output file: {output_json}")

    images_dir = os.path.abspath(images_dir)
    prompt_template = load_prompt(prompt_path)
    rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
    results: List[Dict[str, Any]] = []

    for idx, item in enumerate(data):
        file_idx = slice_start + idx
        instruction = instruction_for_item(item)
        base_out: Dict[str, Any] = {
            "uid": item.get("uid"),
            "youtube_url": item.get("youtube_url"),
            "status_check": "unknown",
            "QA_confidence": None,
        }
        if minimal_output:
            row: Dict[str, Any] = dict(base_out)
        else:
            row = {**item, **base_out}

        if not instruction:
            row["status_check_error"] = "missing rewritten_instruction"
            if not minimal_output:
                row["status_raw_response"] = ""
                row["status_parse_ok"] = False
            results.append(row)
            print(
                f"[batch {idx + 1}/{len(data)} | file_idx {file_idx}] uid={item.get('uid')} | missing instruction"
            )
            continue

        reference_path = resolve_reference_image_path(images_dir, item.get("uid"))
        outcome_path = resolve_outcome_image_path(images_dir, item.get("uid"))
        if not reference_path or not os.path.isfile(reference_path):
            row["status_check_error"] = "missing reference image"
            if not minimal_output:
                row["status_raw_response"] = ""
                row["status_parse_ok"] = False
            results.append(row)
            print(
                f"[batch {idx + 1}/{len(data)} | file_idx {file_idx}] uid={item.get('uid')} | missing reference"
            )
            continue
        if not outcome_path or not os.path.isfile(outcome_path):
            row["status_check_error"] = "missing outcome image"
            if not minimal_output:
                row["status_raw_response"] = ""
                row["status_parse_ok"] = False
            results.append(row)
            print(
                f"[batch {idx + 1}/{len(data)} | file_idx {file_idx}] uid={item.get('uid')} | missing outcome"
            )
            continue

        image_a_is_reference = rng.choice([True, False])
        if image_a_is_reference:
            first_path, second_path = reference_path, outcome_path
        else:
            first_path, second_path = outcome_path, reference_path

        user_text = render_status_check_prompt(
            prompt_template,
            NEUTRAL_FILLER_IMAGE_A,
            NEUTRAL_FILLER_IMAGE_B,
            instruction,
        )
        im_a = load_image_for_check(first_path)
        im_b = load_image_for_check(second_path)
        b64_a = image_to_base64(im_a)
        b64_b = image_to_base64(im_b)

        resp = client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{b64_a}",
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{b64_b}",
                        },
                    ],
                }
            ],
        )

        output_text = extract_output_text(resp)
        parsed_json = parse_json_from_text(output_text)
        conf = parse_confidence(parsed_json.get("confidence"))
        status, verified = local_verify_status(parsed_json, image_a_is_reference)

        row["QA_confidence"] = conf
        row["status_check"] = status
        if minimal_output:
            row.pop("status_check_error", None)
        else:
            row["image_a_is_reference"] = image_a_is_reference
            row["status_raw_response"] = output_text
            row["status_parse_ok"] = bool(parsed_json) and verified
            row["status_model_reference_image"] = str(
                parsed_json.get("reference_image", "")
            ).strip()
            row["status_model_outcome_image"] = str(
                parsed_json.get("outcome_image", "")
            ).strip()
            row["status_model_reason"] = str(parsed_json.get("reason", "")).strip()
            row.pop("status_check_error", None)

        results.append(row)
        print(
            f"[batch {idx + 1}/{len(data)} | file_idx {file_idx}] uid={item.get('uid')} | "
            f"status_check={status} QA_confidence={conf}"
        )

    output_dir = os.path.dirname(os.path.abspath(output_json))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_json}, total={len(results)}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "VLM status QA：Image A/B 与 reference/outcome 每条随机对应，"
            "不在 prompt 中泄露 reference；本地按随机位校验。"
        )
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default=DEFAULT_INPUT_JSON,
        help=f"输入 JSON（默认 {DEFAULT_INPUT_JSON}）",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=DEFAULT_PROMPT_PATH,
        help=f"Prompt 模板（默认 {DEFAULT_PROMPT_PATH}）",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=DEFAULT_OUTPUT_JSON,
        help=f"输出 JSON 基底路径（默认 {DEFAULT_OUTPUT_JSON}），会插入 _start_end",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=DEFAULT_IMAGES_DIR,
        help=f"图像目录（默认 {DEFAULT_IMAGES_DIR}）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="切片后再只处理前 N 条（调试）",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="起始下标（含）",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="结束下标（含）；不传则到文件末尾",
    )
    parser.add_argument(
        "--minimal-output",
        action="store_true",
        help="每条仅保留 uid、youtube_url、status_check（right/wrong/unknown）、QA_confidence；失败行另含 status_check_error",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机数种子（Image A 是否为 reference）；不传则每条独立随机",
    )
    args = parser.parse_args()
    run_status_check(
        input_json=args.input_json,
        prompt_path=args.prompt_path,
        output_json=args.output_json,
        images_dir=args.images_dir,
        limit=args.limit,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        minimal_output=args.minimal_output,
        rng_seed=args.seed,
    )
