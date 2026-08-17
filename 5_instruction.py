







import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Tuple

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

from pipeline_paths import HIGHRES_VIDEO_DIR, PROMPT_DIR, RESTATE_DIR, parquet_triplet


KEYWORDS_SUBSTRING = (
    "first impression",
    "review",
    "introduce",
    "explain",
    "optimize",
    "refine",
    "improve",
    "enhance",
    "demonstrate",
    "experience",
    "evaluate",
    "compare",
    "performance",
    "discuss",
    "showcase",
)
_TEST_RE = re.compile(r"\btest\b", re.IGNORECASE)


INSTRUCTION_OUTPUT_COLS = ("instruction_zh", "instruction_en", "response_id_5")


def instruction_matches_keywords(text: str | None) -> bool:

    if not text or not isinstance(text, str):
        return False
    lowered = text.lower()
    if any(kw.lower() in lowered for kw in KEYWORDS_SUBSTRING):
        return True
    return bool(_TEST_RE.search(text))


def _extract_vlm_response_text(response: Any) -> str:

    response_text = None
    if hasattr(response, "output") and isinstance(response.output, list):
        for item in response.output:
            if hasattr(item, "content") and isinstance(item.content, list):
                for content_item in item.content:
                    if hasattr(content_item, "text"):
                        response_text = content_item.text
                        break
                if response_text:
                    break
    elif hasattr(response, "output") and hasattr(response.output, "choices"):
        response_text = response.output.choices[0].message.content
    elif hasattr(response, "choices"):
        response_text = response.choices[0].message.content
    return response_text or ""


def _ark_response_id(response: Any) -> str:
    if hasattr(response, "id") and getattr(response, "id", None):
        return str(response.id)
    if hasattr(response, "response_id") and getattr(response, "response_id", None):
        return str(response.response_id)
    out = getattr(response, "output", None)
    if out is not None and hasattr(out, "id") and getattr(out, "id", None):
        return str(out.id)
    return ""


def _response_id_nonempty(rid: str | None) -> bool:
    return bool(rid and str(rid).strip())


def ensure_instruction_output_columns(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()
    for col in INSTRUCTION_OUTPUT_COLS:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    return out


def _parse_shuffle_cell(val: Any) -> bool:

    if isinstance(val, bool):
        return val
    if val is None:
        return True
    try:
        if pd.isna(val):
            return True
    except TypeError:
        pass
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return bool(int(val))
    if isinstance(val, str):
        t = val.strip().lower()
        if t in ("true", "1", "yes"):
            return True
        if t in ("false", "0", "no"):
            return False
    return bool(val)


def shuffle_image_mapping_explanation(shuffle: bool) -> str:

    if shuffle:
        return (
            "Image order mapping (shuffle=true): "
            "Image A — the first image in this conversation — is the ANCHOR (State A); "
            "Image B — the second image — is the OUTCOME (State B)."
        )
    return (
        "Image order mapping (shuffle=false): "
        "Image A — the first image in this conversation — is the OUTCOME (State B); "
        "Image B — the second image — is the ANCHOR (State A)."
    )


def load_anchor_outcome_b64_pair(
    *,
    frames_dir: str,
    highres_video_path: str,
    youtube_url: str,
    anchor_timestamp: str,
    outcome_timestamp: str,
    upload_tier: str,
) -> Tuple[str, str]:

    if not youtube_url or not anchor_timestamp or not outcome_timestamp:
        raise ValueError("缺少 youtube_url、anchor_timestamp 或 outcome_timestamp")

    video_id = extract_video_id_from_url(youtube_url)
    if not video_id:
        raise ValueError(f"无法从 URL 提取视频 ID: {youtube_url}")

    anchor_frame_path = anchor_frame_disk_path(frames_dir, video_id, anchor_timestamp)
    if not os.path.exists(anchor_frame_path):
        highres_pattern = os.path.join(highres_video_path, f"{video_id}_*.mp4")
        highres_files = glob.glob(highres_pattern)
        if not highres_files:
            raise FileNotFoundError(
                f"anchor 帧不存在且未找到高清视频: {anchor_frame_path} / {highres_pattern}"
            )
        ref_img = extract_frame_at_timestamp(highres_files[0], anchor_timestamp)
        anchor_frame_path = anchor_frame_disk_path(
            frames_dir, video_id, anchor_timestamp, for_write=True
        )
        save_disk_frame_jpg(anchor_frame_path, ref_img)

    outcome_ts_safe = sanitize_timestamp_for_filename(outcome_timestamp)
    outcome_frame_path = os.path.join(
        frames_dir, f"{video_id}_{outcome_ts_safe}_outcome.jpg"
    )
    if not os.path.exists(outcome_frame_path):
        highres_pattern = os.path.join(highres_video_path, f"{video_id}_*.mp4")
        highres_files = glob.glob(highres_pattern)
        if not highres_files:
            raise FileNotFoundError(
                f"outcome 帧不存在且未找到高清视频: {outcome_frame_path} / {highres_pattern}"
            )
        out_img = extract_frame_at_timestamp(highres_files[0], outcome_timestamp)
        save_disk_frame_jpg(outcome_frame_path, out_img)

    try:
        anchor_img = Image.open(anchor_frame_path)
    except Exception as e:
        raise RuntimeError(f"加载 anchor 帧失败: {e}") from e
    try:
        outcome_img = Image.open(outcome_frame_path)
    except Exception as e:
        raise RuntimeError(f"加载 outcome 帧失败: {e}") from e

    return (
        image_to_base64(fit_image_to_frame(anchor_img, upload_tier)),
        image_to_base64(fit_image_to_frame(outcome_img, upload_tier)),
    )


def upload_instruction_context_frames(
    client: Ark,
    *,
    anchor_b64: str,
    outcome_b64: str,
    shuffle: bool,
    goal: str,
) -> str:



    if shuffle:
        first_b64, second_b64 = anchor_b64, outcome_b64
    else:
        first_b64, second_b64 = outcome_b64, anchor_b64

    intro = (
        "Two images are attached below in order as Image A (first), then Image B (second). "
        f"Recorded shuffle flag for this sample: {str(shuffle).lower()}.\n"
        + shuffle_image_mapping_explanation(shuffle)
    )
    if goal:
        intro += f"\n\nGoal summary: {goal}"

    user_content: list[dict[str, Any]] = [
        {"type": "input_text", "text": intro},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{first_b64}"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{second_b64}"},
    ]
    response = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": user_content}],
        thinking={"type": "enabled"},
    )
    rid = _ark_response_id(response)
    if not rid:
        raise RuntimeError("上传关键帧后未获得有效的 response id,无法继续生成 instruction")
    return rid


def Instruction(
    goal: str,
    *,
    client: Ark,
    instruction_prompt_text: str,
    previous_response_id: str,
    shuffle: bool,
    title: str = "",
) -> Dict[str, str]:








    formatted_prompt = instruction_prompt_text
    if "{Goal}" in formatted_prompt:
        formatted_prompt = formatted_prompt.replace("{Goal}", goal or "")
    if "{goal}" in formatted_prompt:
        formatted_prompt = formatted_prompt.replace("{goal}", goal or "")
    if "{title}" in formatted_prompt:
        formatted_prompt = formatted_prompt.replace("{title}", title or "")
    if "{Title}" in formatted_prompt:
        formatted_prompt = formatted_prompt.replace("{Title}", title or "")

    mapping = shuffle_image_mapping_explanation(shuffle)

    user_content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": formatted_prompt,
        },
        {"type": "input_text", "text": mapping},
    ]


    if goal:
        user_content.append(
            {
                "type": "input_text",
                "text": f"Goal summary: {goal}",
            }
        )
    user_content.append(
        {
            "type": "input_text",
            "text": (
                "Please strictly answer with exactly one JSON object containing "
                "instruction_zh and instruction_en string fields only, without any "
                "Markdown code fences."
            ),
        }
    )

    user_message = {
        "role": "user",
        "content": user_content,
    }

    input_data = [user_message]

    response = client.responses.create(
        model=MODEL_NAME,
        input=input_data,
        previous_response_id=previous_response_id,
        thinking={"type": "enabled"},
    )

    response_id_5 = _ark_response_id(response) or getattr(response, "id", "") or ""
    if isinstance(response_id_5, str):
        response_id_5 = response_id_5.strip()
    else:
        response_id_5 = str(response_id_5 or "").strip()
    if not response_id_5:
        print("警告: Instruction 调用未解析到 response id, response_id_5 将为空")

    response_text = _extract_vlm_response_text(response)
    if not response_text:
        print("警告: Instruction 步骤 VLM 响应为空")
        return {
            "instruction_zh": "",
            "instruction_en": "",
            "response_id_5": response_id_5,
        }

    cleaned = (
        response_text.replace("```json", "")
        .replace("```", "")
        .strip()
    )

    instruction_zh = ""
    instruction_en = ""

    try:
        start_idx = cleaned.find("{")
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i in range(start_idx, len(cleaned)):
                if cleaned[i] == "{":
                    brace_count += 1
                elif cleaned[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
            if brace_count == 0:
                cleaned_json = cleaned[start_idx : end_idx + 1]
            else:
                cleaned_json = cleaned
        else:
            cleaned_json = cleaned

        result = json.loads(cleaned_json)
        instruction_zh = result.get("instruction_zh", "").strip()
        instruction_en = result.get("instruction_en", "").strip()
    except json.JSONDecodeError as e:
        print(f"警告: Instruction JSON 解析失败: {e}, 尝试使用正则作为备选方案")

        m_zh = re.search(
            r"instruction_zh\s*:\s*\"([^\"]+)\"",
            cleaned,
            re.IGNORECASE,
        )
        if not m_zh:
            m_zh = re.search(
                r"instruction_zh\s*:\s*([^,\n\}]+)",
                cleaned,
                re.IGNORECASE,
            )
        if m_zh:
            instruction_zh = m_zh.group(1).strip().strip('"')

        m_en = re.search(
            r"instruction_en\s*:\s*\"([^\"]+)\"",
            cleaned,
            re.IGNORECASE,
        )
        if not m_en:
            m_en = re.search(
                r"instruction_en\s*:\s*([^,\n\}]+)",
                cleaned,
                re.IGNORECASE,
            )
        if m_en:
            instruction_en = m_en.group(1).strip().strip('"')

    return {
        "instruction_zh": instruction_zh,
        "instruction_en": instruction_en,
        "response_id_5": response_id_5,
    }


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
        "shuffle",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"数据中未找到 '{col}' 列, 请使用 4_check 输出作为输入")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "5_instruction: 基于 4_check.parquet 与 Goal 生成中英文高层指令, "
            "并写入 instruction_zh / instruction_en / response_id_5"
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        default=os.path.join(RESTATE_DIR, "4_check.parquet"),
        help="输入 parquet 文件路径 (默认: pipeline/ReState/4_check.parquet, 须为 4_check 保留集)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(RESTATE_DIR, "5_instruction.parquet"),
        help="输出 parquet 路径 (默认: pipeline/ReState/5_instruction.parquet); 使用 --filter_keywords 时同目录写入 *_output / *_exclude",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=PROMPT_DIR,
        help=f"包含 5_instruction.txt 的目录 (默认: {PROMPT_DIR})",
    )
    parser.add_argument(
        "--highres_video_path",
        type=str,
        default=HIGHRES_VIDEO_DIR,
        help="高清视频根目录(当 response_id_4 为空需补抽帧时使用; 与 4_check 默认一致)",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=None,
        help="关键帧目录; 未指定时使用输出 parquet 同目录下的 stage3_frames (与 4_check 一致)",
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
        "--filter_keywords",
        action="store_true",
        help=(
            "开启后写出初稿 *_output.parquet,再按 instruction_en 是否命中关键词拆分为 "
            "保留集与 *_exclude.parquet(规则同 filter_video_clips_keywords)"
        ),
    )
    add_upload_tier_argument(parser, default="low")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upload_tier = normalize_upload_tier(args.upload_tier)

    set_model_name(args.model)

    df = load_parquet(args.input)

    prompt_file = os.path.join(args.prompt_dir, "5_instruction.txt")
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            instruction_prompt_text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt 文件未找到: {prompt_file}")
    except Exception as e:
        raise Exception(f"读取 5_instruction Prompt 文件时出错: {e}")

    client = Ark(
        base_url=DEFAULT_BASE_URL,
        api_key=args.api_key,
    )

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    frames_dir = args.frames_dir or os.path.join(out_dir, "stage3_frames")
    frames_dir = os.path.abspath(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    total = len(df)
    limit = args.limit if args.limit is not None else total
    n_process = min(total, limit)
    print(f"5_instruction 上传规格: --upload-tier={upload_tier}")

    results: list[dict[str, Any]] = []

    for idx in range(n_process):
        row = df.iloc[idx]
        youtube_url = str(row.get("youtube_url", ""))
        title = str(row.get("title", ""))
        goal = str(row.get("goal", ""))
        response_id_4 = str(row.get("response_id_4", "") or "").strip()
        shuffle = _parse_shuffle_cell(row.get("shuffle"))
        anchor_timestamp = str(row.get("anchor_timestamp", "") or "")
        outcome_timestamp = str(row.get("outcome_timestamp", "") or "")

        print(f"\n[{idx + 1}/{n_process}] 5_instruction: {title}")
        print(f"  URL: {youtube_url}")

        try:
            if _response_id_nonempty(response_id_4):
                previous_id = response_id_4
                print(f"  会话: 复用 response_id_4, shuffle={shuffle}")
            else:
                print(f"  会话: response_id_4 为空,按 shuffle={shuffle} 重新上传关键帧并建立对话")
                anchor_b64, out_b64 = load_anchor_outcome_b64_pair(
                    frames_dir=frames_dir,
                    highres_video_path=args.highres_video_path,
                    youtube_url=youtube_url,
                    anchor_timestamp=anchor_timestamp,
                    outcome_timestamp=outcome_timestamp,
                    upload_tier=upload_tier,
                )
                previous_id = upload_instruction_context_frames(
                    client,
                    anchor_b64=anchor_b64,
                    outcome_b64=out_b64,
                    shuffle=shuffle,
                    goal=goal,
                )

            analysis = Instruction(
                goal=goal,
                client=client,
                instruction_prompt_text=instruction_prompt_text,
                previous_response_id=previous_id,
                shuffle=shuffle,
                title=title,
            )
            response_id_5 = str(analysis.get("response_id_5", "") or "").strip()
            print(analysis.get("instruction_zh", ""), analysis.get("instruction_en", ""))
            print(f"  response_id_5: {response_id_5 or '(empty)'}")


            out_row: dict[str, Any] = row.to_dict()
            out_row["youtube_url"] = youtube_url
            out_row["title"] = title
            out_row["goal"] = goal
            out_row["response_id_4"] = response_id_4
            if not _response_id_nonempty(response_id_4) and _response_id_nonempty(previous_id):

                out_row["response_id_4"] = previous_id
            out_row["instruction_zh"] = analysis.get("instruction_zh", "")
            out_row["instruction_en"] = analysis.get("instruction_en", "")
            out_row["response_id_5"] = response_id_5
            results.append(out_row)
        except Exception as e:
            print(f"  错误: 5_instruction 处理该视频时出错: {e}")
            import traceback

            traceback.print_exc()
            out_row = row.to_dict()
            out_row["youtube_url"] = youtube_url
            out_row["title"] = title
            out_row["goal"] = goal
            out_row["response_id_4"] = response_id_4
            out_row["instruction_zh"] = ""
            out_row["instruction_en"] = ""
            out_row["response_id_5"] = ""
            out_row["error"] = str(e)
            results.append(out_row)

    if results:
        out_df = ensure_instruction_output_columns(pd.DataFrame(results))
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        if args.filter_keywords:
            out_full_path, kept_path, exclude_path = parquet_triplet(args.output)
            for p in (out_full_path, kept_path, exclude_path):
                if os.path.exists(p):
                    os.remove(p)
            out_df.to_parquet(out_full_path, index=False)
            print(f"\n5_instruction 初稿已写入: {out_full_path} ({len(out_df)} 条)")
            en_col = out_df.get("instruction_en")
            if en_col is None:
                print("警告: --filter_keywords 已开启但结果中无 instruction_en 列,跳过关键词拆分")
                out_df.iloc[0:0].to_parquet(exclude_path, index=False)
                out_df.to_parquet(kept_path, index=False)
            else:
                mask = en_col.map(instruction_matches_keywords)
                matched_df = out_df[mask]
                kept_df = out_df[~mask]
                matched_df.to_parquet(exclude_path, index=False)
                kept_df.to_parquet(kept_path, index=False)
                print(
                    f"关键词拆分: 命中 -> {os.path.basename(exclude_path)} ({len(matched_df)}), "
                    f"保留 -> {os.path.basename(kept_path)} ({len(kept_df)})"
                )
        else:
            out_path = Path(args.output)
            if out_path.exists():
                out_path.unlink()
            out_df.to_parquet(str(out_path), index=False)
            print(
                f"\n5_instruction 完成, 共处理样本 {len(results)} 条, 结果已写入: {out_path}"
            )
    else:
        print("\n5_instruction 完成: 无记录被处理, 未生成输出 parquet")


if __name__ == "__main__":
    main()

