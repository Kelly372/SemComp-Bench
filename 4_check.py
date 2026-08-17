














from __future__ import annotations

import argparse
import glob
import json
import os
import random
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
    extract_frame_at_timestamp,
    extract_video_id_from_url,
    image_to_base64,
    sanitize_timestamp_for_filename,
    set_model_name,
)
from media_specs import (
    add_upload_tier_argument,
    fit_image_to_frame,
    normalize_upload_tier,
    save_disk_frame_jpg,
)

from pipeline_paths import (
    HIGHRES_VIDEO_DIR,
    PROMPT_DIR,
    RESTATE_DIR,
    parquet_error_path,
    parquet_triplet,
)


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
    start = cleaned.find("{")
    if start != -1:
        brace_count = 0
        end = start
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                brace_count += 1
            elif cleaned[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i
                    break
        if brace_count == 0:
            try:
                data = json.loads(cleaned[start : end + 1])
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


def render_4_check_prompt(
    prompt_template: str,
    image_a_filler: str,
    image_b_filler: str,
    goal: str,
) -> str:

    s = prompt_template.replace("{image_a}", image_a_filler)
    s = s.replace("{image_b}", image_b_filler)
    s = s.replace("{goal}", goal)
    return s


def _as_bool(v: Any) -> bool:

    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(int(v)) if int(v) in (0, 1) else bool(v)
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("true", "1", "yes"):
            return True
        if t in ("false", "0", "no", ""):
            return False
    return False


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
    if ("anchor" in s or "reference" in s) and "outcome" not in s:
        if "a" in s and "b" not in s:
            return "A"
        if "b" in s and "a" not in s:
            return "B"
    if "outcome" in s and "anchor" not in s and "reference" not in s:
        if "b" in s and "a" not in s:
            return "B"
        if "a" in s and "b" not in s:
            return "A"
    return None


def _slot_from_schema_image_label(raw: Any) -> Optional[str]:




    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        sl = s.lower()
        if sl == "image a":
            return "A"
        if sl == "image b":
            return "B"
        return _slot_from_label(s)
    return None


def parse_confidence_4check(raw: Any) -> Optional[int]:

    if raw is None or isinstance(raw, bool):
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


def _json_null_like(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        t = v.strip().lower()
        return t in ("", "null", "none")
    return False



_EXCLUDE_COLUMNS = [
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
    "anchor_quality",
    "outcome_quality",
    "check_status",
]

_ERROR_COLUMNS = [
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
    "error_kind",
    "message",
]



_OUTPUT_SNAPSHOT_META = [
    "response_id_4",
    "detail",
    "image_quality_a",
    "image_quality_b",
    "predict_outcome",
    "predict_confidence",
    "shuffle",
    "pipeline_status",
]


_OUTPUT_META_WITH_SPLIT = [
    "response_id_4",
    "split_tag",
    "detail",
    "image_quality_a",
    "image_quality_b",
    "predict_outcome",
    "predict_confidence",
    "shuffle",
    "pipeline_status",
]


def _predict_outcome_ab(raw: Any) -> Optional[str]:

    slot = _slot_from_schema_image_label(raw)
    if slot == "A":
        return "a"
    if slot == "B":
        return "b"
    return None


def _parse_quality_check_images(parsed: Dict[str, Any]) -> Tuple[Optional[bool], Optional[bool]]:

    qc = parsed.get("quality_check")
    if not isinstance(qc, dict):
        return None, None
    if "image_a" not in qc or "image_b" not in qc:
        return None, None
    if qc.get("image_a") is None or qc.get("image_b") is None:
        return None, None
    return _as_bool(qc["image_a"]), _as_bool(qc["image_b"])


def derive_anchor_outcome_check(
    shuffle: bool,
    image_quality_a: bool,
    image_quality_b: bool,
    predict_outcome: str,
) -> Tuple[bool, bool, bool]:






    aq = image_quality_a if shuffle else image_quality_b
    oq = image_quality_b if shuffle else image_quality_a
    po = predict_outcome.strip().lower()
    correct = "b" if shuffle else "a"
    ok = po == correct
    return bool(aq), bool(oq), bool(ok)


def _output_snapshot_row(
    row: pd.Series,
    input_columns: List[str],
    response_id_4: str,
    detail: str = "",
    pipeline_status: str = "vlm",
    *,
    image_quality_a: Any = pd.NA,
    image_quality_b: Any = pd.NA,
    predict_outcome: Any = "",
    predict_confidence: Any = pd.NA,
    shuffle: Any = pd.NA,
) -> Dict[str, Any]:

    out: Dict[str, Any] = {c: row.get(c) for c in input_columns}
    out["response_id_4"] = response_id_4
    out["detail"] = detail
    out["image_quality_a"] = image_quality_a
    out["image_quality_b"] = image_quality_b
    out["predict_outcome"] = predict_outcome
    out["predict_confidence"] = (
        int(predict_confidence)
        if predict_confidence in (1, 2, 3)
        else predict_confidence
    )
    out["shuffle"] = shuffle
    out["pipeline_status"] = pipeline_status
    return out


def assign_split_tags(out_df: pd.DataFrame) -> pd.DataFrame:




    df = out_df.copy()
    tags: List[str] = []
    for _, r in df.iterrows():
        if str(r.get("pipeline_status", "vlm")) == "exception":
            tags.append("exception")
            continue
        if pd.isna(r.get("shuffle")):
            tags.append("parse_error")
            continue
        shuffle = bool(r["shuffle"])
        iqa, iqb = r.get("image_quality_a"), r.get("image_quality_b")
        if pd.isna(iqa) or pd.isna(iqb):
            tags.append("parse_error")
            continue
        po = str(r.get("predict_outcome", "") or "").strip().lower()
        if po not in ("a", "b"):
            tags.append("parse_error")
            continue
        conf_i = parse_confidence_4check(r.get("predict_confidence"))
        if conf_i is None:
            tags.append("parse_error")
            continue
        aq, oq, ck = derive_anchor_outcome_check(
            shuffle, bool(iqa), bool(iqb), po
        )
        if not aq or not oq:
            tags.append("quality_fail")
        elif not ck:
            tags.append("wrong_predict")
        else:
            tags.append("kept")
    df["split_tag"] = tags
    return df


def split_unified_output_to_files(
    out_df: pd.DataFrame,
    input_columns: List[str],
    kept_path: str,
    exclude_path: str,
    error_path: str,
) -> None:

    os.makedirs(os.path.dirname(kept_path) or ".", exist_ok=True)

    kept_cols = list(input_columns) + ["check_confidence", "response_id_4", "shuffle"]
    if len(out_df) == 0:
        pd.DataFrame(columns=kept_cols).to_parquet(kept_path, index=False)
        pd.DataFrame(columns=_EXCLUDE_COLUMNS).to_parquet(exclude_path, index=False)
        pd.DataFrame(columns=_ERROR_COLUMNS).to_parquet(error_path, index=False)
        print(f"无处理记录, 已写入空表: {kept_path}, {exclude_path}, {error_path}")
        return

    km = out_df["split_tag"] == "kept"
    if km.any():
        kept_part = out_df.loc[km].copy()
        kept_part["check_confidence"] = kept_part["predict_confidence"]
        kept_part = kept_part[[c for c in kept_cols if c in kept_part.columns]]
        kept_part.to_parquet(kept_path, index=False)
        print(
            f"\n4_check 完成, 通过校验 {int(km.sum())} 条, 已写入: {kept_path}"
        )
    else:
        pd.DataFrame(columns=kept_cols).to_parquet(kept_path, index=False)
        print(f"\n4_check 完成: 无通过校验记录, 已写入空表: {kept_path}")

    ex_mask = out_df["split_tag"].isin(["quality_fail", "wrong_predict"])
    if ex_mask.any():
        ex_df = out_df.loc[ex_mask]
        aq_list: List[bool] = []
        oq_list: List[bool] = []
        ck_list: List[bool] = []
        for _, r in ex_df.iterrows():
            try:
                sh = bool(r["shuffle"])
                iqa = bool(r["image_quality_a"])
                iqb = bool(r["image_quality_b"])
                po = str(r.get("predict_outcome", "") or "")
                aq, oq, ck = derive_anchor_outcome_check(sh, iqa, iqb, po)
            except Exception:
                aq, oq, ck = False, False, False
            aq_list.append(aq)
            oq_list.append(oq)
            ck_list.append(ck)
        ex_out = pd.DataFrame(
            {
                "youtube_url": ex_df["youtube_url"],
                "title": ex_df["title"],
                "domain": ex_df["domain"],
                "category": ex_df["category"],
                "goal": ex_df["goal"],
                "anchor_timestamp": ex_df["anchor_timestamp"],
                "outcome_timestamp": ex_df["outcome_timestamp"],
                "file_id": ex_df["file_id"],
                "response_id_3": ex_df["response_id_3"],
                "response_id_4": ex_df["response_id_4"],
                "anchor_quality": aq_list,
                "outcome_quality": oq_list,
                "check_status": ck_list,
            }
        )
        ex_out.to_parquet(exclude_path, index=False)
        print(f"未通过样本已写入: {exclude_path} ({len(ex_out)} 条)")
    else:
        pd.DataFrame(columns=_EXCLUDE_COLUMNS).to_parquet(exclude_path, index=False)
        print(f"无未通过样本, 已写入空表: {exclude_path}")

    err_mask = out_df["split_tag"].isin(["parse_error", "exception"])
    if err_mask.any():
        er_df = out_df.loc[err_mask]
        err_out = pd.DataFrame(
            {
                "youtube_url": er_df["youtube_url"],
                "title": er_df["title"],
                "domain": er_df["domain"],
                "category": er_df["category"],
                "goal": er_df["goal"],
                "anchor_timestamp": er_df["anchor_timestamp"],
                "outcome_timestamp": er_df["outcome_timestamp"],
                "file_id": er_df["file_id"],
                "response_id_3": er_df["response_id_3"],
                "response_id_4": er_df["response_id_4"],
                "error_kind": er_df["split_tag"],
                "message": er_df["detail"].fillna("").astype(str),
            }
        )
        err_out.to_parquet(error_path, index=False)
        print(f"技术异常样本已写入: {error_path} ({len(err_out)} 条)")
    else:
        pd.DataFrame(columns=_ERROR_COLUMNS).to_parquet(error_path, index=False)
        print(f"无技术异常样本, 已写入空表: {error_path}")


def load_parquet(input_path: str) -> pd.DataFrame:





    if os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "*.parquet"))
        if not files:
            raise FileNotFoundError(f"在目录 '{input_path}' 下未找到任何 parquet 文件")
        print(f"发现 {len(files)} 个 parquet 文件, 正在加载...")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"未找到文件: {input_path}")
        print(f"正在读取 parquet 文件: {input_path}")
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
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"数据中未找到 '{col}' 列, 请使用 3_extract 输出作为输入")

    print(f"加载完成, 共 {len(df)} 行")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取 3_extract.parquet, 按 prompt/4_check.txt 统一画质与 outcome 判定, "
            "再分类写入 4_check / 4_check_exclude / 4_check_error"
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        default=os.path.join(RESTATE_DIR, "3_extract.parquet"),
        help="输入 parquet 文件或目录路径 (默认: pipeline/ReState/3_extract.parquet, 须为 3_extract 保留集)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(RESTATE_DIR, "4_check.parquet"),
        help=(
            "通过样本输出路径 (默认: pipeline/ReState/4_check.parquet); "
            "同目录先写 4_check_output.parquet(全量快照,无 split_tag),读回后计算 split_tag 再拆分写入本路径、*_exclude、*_error"
        ),
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=PROMPT_DIR,
        help=f"包含 4_check.txt 的目录 (默认: {PROMPT_DIR})",
    )
    parser.add_argument(
        "--highres_video_path",
        type=str,
        default=HIGHRES_VIDEO_DIR,
        help="高清视频根目录（通过 --highres_video_path 或 HIGHRES_VIDEO_DIR 指定）",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=None,
        help="关键帧保存目录; 未指定时使用输出文件同目录下的 stage3_frames",
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
        "--seed",
        type=int,
        default=None,
        help="随机数种子 (Image A 是否对应 anchor 帧); 不传则每次运行非固定随机",
    )
    add_upload_tier_argument(parser, default="low")
    return parser.parse_args()


def main():
    args = parse_args()
    upload_tier = normalize_upload_tier(args.upload_tier)

    set_model_name(args.model)

    df = load_parquet(args.input)

    image_check_prompt_file = os.path.join(args.prompt_dir, "4_check.txt")
    try:
        with open(image_check_prompt_file, "r", encoding="utf-8") as f:
            image_check_prompt_text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt 文件未找到: {image_check_prompt_file}")
    except Exception as e:
        raise Exception(f"读取 image_check Prompt 文件时出错: {e}")

    out_full_path, kept_path, exclude_path = parquet_triplet(args.output)
    error_path = parquet_error_path(kept_path)
    frames_dir = args.frames_dir or os.path.join(os.path.dirname(kept_path) or ".", "stage3_frames")
    os.makedirs(frames_dir, exist_ok=True)

    client = Ark(
        base_url=DEFAULT_BASE_URL,
        api_key=args.api_key,
    )

    for p in (out_full_path, kept_path, exclude_path, error_path):
        if os.path.exists(p):
            print(f"输出文件已存在, 将先删除: {p}")
            os.remove(p)

    total = len(df)
    limit = args.limit if args.limit is not None else total
    n_process = min(total, limit)
    print(f"4_check 上传规格: --upload-tier={upload_tier}")
    output_rows: List[dict] = []
    input_cols = list(df.columns)
    rng = random.Random(args.seed) if args.seed is not None else random.Random()

    for idx in range(n_process):
        row = df.iloc[idx]
        response_id_4 = ""
        youtube_url = str(row.get("youtube_url", ""))
        title = str(row.get("title", ""))
        domain = str(row.get("domain", ""))
        category = str(row.get("category", "Uncertain"))
        goal = str(row.get("goal", ""))
        anchor_timestamp = str(row.get("anchor_timestamp", ""))
        outcome_timestamp = str(row.get("outcome_timestamp", ""))
        file_id = str(row.get("file_id", ""))

        shuffle_val: Any = pd.NA

        print(f"\n[{idx + 1}/{n_process}] 4_check: {title}")
        print(f"  URL: {youtube_url}")
        print(f"  Anchor: {anchor_timestamp}, Outcome: {outcome_timestamp}")

        try:
            if not youtube_url or not anchor_timestamp:
                raise ValueError("缺少 youtube_url 或 anchor_timestamp")

            video_id = extract_video_id_from_url(youtube_url)
            if not video_id:
                raise ValueError(f"无法从 URL 提取视频 ID: {youtube_url}")


            anchor_frame_path = anchor_frame_disk_path(
                frames_dir, video_id, anchor_timestamp
            )

            if not os.path.exists(anchor_frame_path):
                highres_pattern = os.path.join(
                    args.highres_video_path, f"{video_id}_*.mp4"
                )
                highres_files = glob.glob(highres_pattern)
                if not highres_files:
                    raise FileNotFoundError(
                        f"anchor 帧不存在且未找到高清视频文件: {anchor_frame_path} / {highres_pattern}"
                    )

                highres_video_path = highres_files[0]
                ref_img = extract_frame_at_timestamp(
                    highres_video_path, anchor_timestamp
                )
                anchor_frame_path = anchor_frame_disk_path(
                    frames_dir, video_id, anchor_timestamp, for_write=True
                )
                save_disk_frame_jpg(anchor_frame_path, ref_img)

            try:
                anchor_img = Image.open(anchor_frame_path)
            except Exception as e:
                raise RuntimeError(f"加载 anchor 帧失败: {e}") from e

            anchor_b64 = image_to_base64(
                fit_image_to_frame(anchor_img, upload_tier)
            )


            if not outcome_timestamp:
                raise ValueError("缺少 outcome_timestamp")

            out_ts_safe = sanitize_timestamp_for_filename(outcome_timestamp)
            outcome_frame_filename = f"{video_id}_{out_ts_safe}_outcome.jpg"
            outcome_frame_path = os.path.join(frames_dir, outcome_frame_filename)

            if not os.path.exists(outcome_frame_path):
                highres_pattern = os.path.join(
                    args.highres_video_path, f"{video_id}_*.mp4"
                )
                highres_files = glob.glob(highres_pattern)
                if not highres_files:
                    raise FileNotFoundError(
                        f"outcome 帧不存在且未找到高清视频文件: {outcome_frame_path} / {highres_pattern}"
                    )

                highres_video_path = highres_files[0]
                out_img = extract_frame_at_timestamp(
                    highres_video_path, outcome_timestamp
                )
                save_disk_frame_jpg(outcome_frame_path, out_img)

            try:
                outcome_img = Image.open(outcome_frame_path)
            except Exception as e:
                raise RuntimeError(f"加载 outcome 帧失败: {e}")

            outcome_b64 = image_to_base64(
                fit_image_to_frame(outcome_img, upload_tier)
            )

            image_a_is_anchor = rng.choice([True, False])
            shuffle_val = bool(image_a_is_anchor)
            if image_a_is_anchor:
                first_b64, second_b64 = anchor_b64, outcome_b64
            else:
                first_b64, second_b64 = outcome_b64, anchor_b64

            user_text = render_4_check_prompt(
                image_check_prompt_text,
                NEUTRAL_FILLER_IMAGE_A,
                NEUTRAL_FILLER_IMAGE_B,
                goal,
            )
            quality_user_content = [
                {"type": "input_text", "text": user_text},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{first_b64}",
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{second_b64}",
                },
            ]
            quality_input_data = [
                {
                    "role": "user",
                    "content": quality_user_content,
                },
            ]
            quality_response = client.responses.create(
                model=MODEL_NAME,
                input=quality_input_data,
                thinking={"type": "enabled"},
            )
            response_id_4 = getattr(quality_response, "id", "") or ""
            from_text = extract_output_text(quality_response)
            if not from_text.strip():
                raise RuntimeError("4_check VLM 响应为空")

            parsed = parse_json_from_text(from_text)
            if not parsed:
                print("  4_check: 无法解析 JSON (快照已记, split_tag 将在读回 _output 后计算)")
                output_rows.append(
                    _output_snapshot_row(
                        row,
                        input_cols,
                        response_id_4,
                        detail=from_text[:2000],
                        shuffle=shuffle_val,
                    )
                )
                continue

            iqa, iqb = _parse_quality_check_images(parsed)
            if iqa is None or iqb is None:
                print(
                    "  4_check: quality_check 缺失或非法 (快照已记, split_tag 在读回 _output 后计算)"
                )
                output_rows.append(
                    _output_snapshot_row(
                        row,
                        input_cols,
                        response_id_4,
                        detail=from_text[:2000],
                        shuffle=shuffle_val,
                    )
                )
                continue

            out_raw = parsed.get("outcome_image")
            if _json_null_like(out_raw):
                print(
                    "  4_check: outcome_image 为空 (快照已记, split_tag 在读回 _output 后计算)"
                )
                output_rows.append(
                    _output_snapshot_row(
                        row,
                        input_cols,
                        response_id_4,
                        detail=from_text[:2000],
                        shuffle=shuffle_val,
                        image_quality_a=iqa,
                        image_quality_b=iqb,
                    )
                )
                continue

            predict_ab = _predict_outcome_ab(out_raw)
            if predict_ab is None:
                print(
                    "  4_check: outcome_image 须为 Image A / Image B (快照已记, split_tag 在读回 _output 后计算)"
                )
                output_rows.append(
                    _output_snapshot_row(
                        row,
                        input_cols,
                        response_id_4,
                        detail=from_text[:2000],
                        shuffle=shuffle_val,
                        image_quality_a=iqa,
                        image_quality_b=iqb,
                    )
                )
                continue

            reason = str(parsed.get("reason", "") or "").strip()
            conf_i = parse_confidence_4check(parsed.get("confidence"))
            if conf_i is None:
                print(
                    "  4_check: confidence 须为 1/2/3 (快照已记, split_tag 在读回 _output 后计算)"
                )
                output_rows.append(
                    _output_snapshot_row(
                        row,
                        input_cols,
                        response_id_4,
                        detail=reason or from_text[:2000],
                        image_quality_a=iqa,
                        image_quality_b=iqb,
                        predict_outcome=predict_ab,
                        shuffle=shuffle_val,
                    )
                )
                continue

            shuffle = bool(shuffle_val)
            anchor_quality, outcome_quality, check_status = derive_anchor_outcome_check(
                shuffle, bool(iqa), bool(iqb), predict_ab
            )

            base_meta = dict(
                image_quality_a=iqa,
                image_quality_b=iqb,
                predict_outcome=predict_ab,
                predict_confidence=conf_i,
                shuffle=shuffle,
            )

            if not anchor_quality or not outcome_quality:
                print(
                    f"  4_check: anchor/outcome 画质未通过 (anchor={anchor_quality}, outcome={outcome_quality}), 快照已记"
                )
                output_rows.append(
                    _output_snapshot_row(
                        row,
                        input_cols,
                        response_id_4,
                        detail=reason,
                        **base_meta,
                    )
                )
                continue

            if not check_status:
                print(
                    f"  4_check: outcome 预测与真值不符 (shuffle={shuffle}, predict={predict_ab}), 快照已记"
                )
                note = reason or f"shuffle={shuffle}, predict_outcome={predict_ab}"
                output_rows.append(
                    _output_snapshot_row(
                        row,
                        input_cols,
                        response_id_4,
                        detail=note,
                        **base_meta,
                    )
                )
                continue

            output_rows.append(
                _output_snapshot_row(
                    row,
                    input_cols,
                    response_id_4,
                    detail="",
                    **base_meta,
                )
            )

        except Exception as e:
            print(f"  错误: 4_check 阶段处理该视频时出错: {e}")
            import traceback

            traceback.print_exc()
            output_rows.append(
                _output_snapshot_row(
                    row,
                    input_cols,
                    response_id_4,
                    detail=str(e),
                    shuffle=shuffle_val,
                    pipeline_status="exception",
                )
            )

    snapshot_cols = list(input_cols) + _OUTPUT_SNAPSHOT_META
    if output_rows:
        out_df = pd.DataFrame(output_rows)
        out_df = out_df[[c for c in snapshot_cols if c in out_df.columns]]
    else:
        out_df = pd.DataFrame(columns=snapshot_cols)

    os.makedirs(os.path.dirname(out_full_path) or ".", exist_ok=True)
    out_df.to_parquet(out_full_path, index=False)
    print(f"\n全量快照(无 split_tag)已写入: {out_full_path} ({len(out_df)} 条)")

    out_df_reload = pd.read_parquet(out_full_path)
    out_classified = assign_split_tags(out_df_reload)
    final_cols = list(input_cols) + _OUTPUT_META_WITH_SPLIT
    out_classified = out_classified[
        [c for c in final_cols if c in out_classified.columns]
    ]
    out_classified.to_parquet(out_full_path, index=False)
    print(
        f"已读回并计算 split_tag, 写回: {out_full_path} ({len(out_classified)} 条)"
    )

    split_unified_output_to_files(
        out_classified, input_cols, kept_path, exclude_path, error_path
    )


if __name__ == "__main__":
    main()

