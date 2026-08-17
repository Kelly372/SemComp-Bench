







import argparse
import glob
import json
import os
import re
from typing import List

import pandas as pd
from volcenginesdkarkruntime import Ark

from agent_process import (
    API_KEY as DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MODEL_NAME,
    anchor_frame_disk_path,
    extract_frame_at_timestamp,
    extract_video_id_from_url,
    sanitize_timestamp_for_filename,
    set_model_name,
)
from media_specs import (
    TIER_VIDEO_EXTRACT,
    VIDEO_UPLOAD_FPS,
    save_disk_frame_jpg,
    upload_video_file,
)

from pipeline_paths import (
    HIGHRES_VIDEO_DIR,
    LOWRES_VIDEO_DIR,
    PROMPT_DIR,
    RESTATE_DIR,
    parquet_triplet,
)





def load_stage2_extract_prompt(stage2_extract_prompt_file: str) -> str:

    try:
        with open(stage2_extract_prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt文件未找到: {stage2_extract_prompt_file}")
    except Exception as e:
        raise Exception(f"读取 Stage2 提取 Prompt 文件时出错: {e}") from e


def load_category_taxonomy(category_json_file: str) -> dict:

    try:
        with open(category_json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"taxonomy 定义文件未找到: {category_json_file}")
    except Exception as e:
        raise Exception(f"读取 category.json 时出错: {e}") from e


def _strip_optional_code_prefix(label: str) -> str:

    s = str(label or "").strip()
    if not s or "." not in s:
        return s
    head, rest = s.split(".", 1)
    if head.isdigit() or (len(head) == 1 and head.isalpha()):
        return rest.strip()
    return s


def get_anchor_outcome_definitions(
    category_taxonomy: dict,
    domain: str,
    category: str,
) -> tuple[str, str]:





    domain_key = _strip_optional_code_prefix(domain)
    category_key = _strip_optional_code_prefix(category)

    domain_dict = category_taxonomy.get(domain_key, {})
    category_dict = domain_dict.get(category_key, {})

    anchor_def = category_dict.get("Anchor Frame", "")
    outcome_def = category_dict.get("Outcome Frame", "")
    return (anchor_def or "Not defined", outcome_def or "Not defined")


# extract the image according to the timestamp




def validate_extract_result(
    goal: str,
    anchor_timestamp: str,
    outcome_timestamp: str,
    anchor_frame_path: str,
    outcome_frame_path: str,
) -> tuple[bool, str]:

    if not (goal or "").strip():
        return False, "empty_goal"
    if not (anchor_timestamp or "").strip():
        return False, "empty_anchor_timestamp"
    if not (outcome_timestamp or "").strip():
        return False, "empty_outcome_timestamp"
    if not anchor_frame_path or not os.path.isfile(anchor_frame_path):
        return False, "missing_anchor_frame"
    if not outcome_frame_path or not os.path.isfile(outcome_frame_path):
        return False, "missing_outcome_frame"
    return True, ""





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

    if "title" not in df.columns:
        raise KeyError("数据中未找到 'title' 列")
    if "youtube_url" not in df.columns and "url" not in df.columns:
        raise KeyError("数据中未找到 'youtube_url' 或 'url' 列")
    if "domain" not in df.columns or "category" not in df.columns:
        raise KeyError("数据中未找到 'domain' 或 'category' 列 (建议使用 2_classify 的输出作为输入)")

    print(f"加载完成, 共 {len(df)} 行")
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="对 parquet 中每条视频提取 Goal 与时间戳并保存关键帧; 输入需含 youtube_url/title/domain/category (domain=旧粗类 category, category=旧 sub_category)"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=os.path.join(RESTATE_DIR, "2_classify.parquet"),
        help="输入 parquet 文件或目录路径 (默认: pipeline/ReState/2_classify.parquet)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(RESTATE_DIR, "3_extract.parquet"),
        help="保留样本输出路径 (默认: pipeline/ReState/3_extract.parquet); 同目录写入 *_output / *_exclude",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=PROMPT_DIR,
        help=f"包含 3_extract.txt 与 category.json 的目录 (默认: {PROMPT_DIR})",
    )
    parser.add_argument(
        "--lowres_video_path",
        type=str,
        default=LOWRES_VIDEO_DIR,
        help="低清视频根目录（通过 --lowres_video_path 或 LOWRES_VIDEO_DIR 指定）",
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
        "--fps",
        type=float,
        default=VIDEO_UPLOAD_FPS[TIER_VIDEO_EXTRACT],
        help=f"上传视频时的 fps 预处理 (默认 {VIDEO_UPLOAD_FPS[TIER_VIDEO_EXTRACT]}, 与 low tier 一致)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=DEFAULT_API_KEY,
        help="VLM API Key（默认读取 VLM_API_KEY）",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"VLM 模型名称, 默认: {DEFAULT_MODEL}")
    parser.add_argument("--limit", type=int, default=None, help="限制处理条数 (用于测试), 默认全部")
    return parser.parse_args()


def main():
    args = parse_args()

    set_model_name(args.model)

    df = load_parquet(args.input)

    out_full_path, kept_path, exclude_path = parquet_triplet(args.output)
    frames_dir = args.frames_dir or os.path.join(os.path.dirname(kept_path) or ".", "stage3_frames")
    stage2_extract_prompt_file = os.path.join(args.prompt_dir, "3_extract.txt")
    category_json_file = os.path.join(args.prompt_dir, "category.json")
    stage2_extract_prompt_text = load_stage2_extract_prompt(stage2_extract_prompt_file)
    category_taxonomy = load_category_taxonomy(category_json_file)
    os.makedirs(frames_dir, exist_ok=True)

    client = Ark(
        base_url=DEFAULT_BASE_URL,
        api_key=args.api_key,
    )

    for p in (out_full_path, kept_path, exclude_path):
        if os.path.exists(p):
            print(f"输出文件已存在, 将先删除: {p}")
            os.remove(p)

    total = len(df)
    limit = args.limit if args.limit is not None else total
    n_process = min(total, limit)
    passed_rows: List[dict] = []
    error_rows: List[dict] = []

    for idx in range(n_process):
        row = df.iloc[idx]
        youtube_url = str(row.get("youtube_url", row.get("url", "")))
        title = str(row["title"])
        domain = str(row.get("domain", "Uncertain"))
        category = str(row.get("category", ""))

        print(f"\n[{idx + 1}/{n_process}] 处理视频: {title}")
        print(f"  URL: {youtube_url}")
        print(f"  Domain: {domain}, Category: {category}")

        try:

            video_id = extract_video_id_from_url(youtube_url)
            if not video_id:
                raise ValueError(f"无法从 URL 提取视频 ID: {youtube_url}")


            lowres_pattern = os.path.join(args.lowres_video_path, f"{video_id}_*.mp4")
            lowres_files = glob.glob(lowres_pattern)
            if not lowres_files:
                raise FileNotFoundError(f"未找到低清视频文件: {lowres_pattern}")
            lowres_video_path = lowres_files[0]


            anchor_definition, outcome_definition = get_anchor_outcome_definitions(
                category_taxonomy,
                domain,
                category,
            )


            file_id = upload_video_file(
                client,
                lowres_video_path,
                TIER_VIDEO_EXTRACT,
                fps=args.fps,
            )


            formatted_prompt = (
                stage2_extract_prompt_text.replace("{file_id}", file_id)
                .replace("{title}", title or "")
                .replace("{Anchor Frame}", anchor_definition or "Not defined")
                .replace("{Outcome Frame}", outcome_definition or "Not defined")
            )
            

            response = client.responses.create(
                model=MODEL_NAME,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": formatted_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_video", "file_id": file_id},
                        ],
                    },
                ],
                thinking={"type": "enabled"},
                # caching={"type": "enabled", "prefix": True},
            )


            output = response.output[1].content[0].text


            cleaned = (
                output.replace("```json", "")
                .replace("```", "")
                .strip()
            )

            goal = ""
            anchor_timestamp = ""
            outcome_timestamp = ""
            try:
                data = json.loads(cleaned)
                goal = str(data.get("Goal", "")).strip()
                anchor_timestamp = str(data.get("anchor_timestamp", "")).strip()
                outcome_timestamp = str(
                    data.get("outcome_timestamp", "")
                ).strip()
            except json.JSONDecodeError as e:
                print(f"警告: Stage2_extract JSON 解析失败: {e}, 尝试使用正则表达式解析")
                m_anchor_ts = re.search(
                    r'"anchor_timestamp"\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE
                )
                if m_anchor_ts:
                    anchor_timestamp = m_anchor_ts.group(1).strip()
                m_out_ts = re.search(
                    r'"outcome_timestamp"\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE
                )
                if m_out_ts:
                    outcome_timestamp = m_out_ts.group(1).strip()


            anchor_frame_path = ""
            outcome_frame_path = ""
            if anchor_timestamp and outcome_timestamp:
                highres_pattern = os.path.join(
                    args.highres_video_path, f"{video_id}_*.mp4"
                )
                highres_files = glob.glob(highres_pattern)
                if highres_files:
                    highres_video_path = highres_files[0]

                    ref_img = extract_frame_at_timestamp(
                        highres_video_path, anchor_timestamp
                    )
                    anchor_frame_path = anchor_frame_disk_path(
                        frames_dir, video_id, anchor_timestamp, for_write=True
                    )
                    if not os.path.exists(anchor_frame_path):
                        save_disk_frame_jpg(anchor_frame_path, ref_img)

                    out_img = extract_frame_at_timestamp(
                        highres_video_path, outcome_timestamp
                    )
                    out_ts_safe = sanitize_timestamp_for_filename(outcome_timestamp)
                    outcome_frame_filename = f"{video_id}_{out_ts_safe}_outcome.jpg"
                    outcome_frame_path = os.path.join(
                        frames_dir, outcome_frame_filename
                    )
                    if not os.path.exists(outcome_frame_path):
                        save_disk_frame_jpg(outcome_frame_path, out_img)
                else:
                    print(f"  警告: 未找到高清视频文件: {highres_pattern}, 跳过帧提取")

            ok, reject_reason = validate_extract_result(
                goal,
                anchor_timestamp,
                outcome_timestamp,
                anchor_frame_path,
                outcome_frame_path,
            )
            base_row = {
                **row.to_dict(),
                "youtube_url": youtube_url,
                "title": title,
                "domain": domain,
                "category": category,
                "goal": goal,
                "anchor_timestamp": anchor_timestamp,
                "outcome_timestamp": outcome_timestamp,
                "file_id": file_id,
                "response_id_3": response.id,
            }
            if not ok:
                print(f"  剔除: {reject_reason}")
                error_rows.append({**base_row, "error_reason": reject_reason})
                continue

            passed_rows.append(base_row)

        except Exception as e:
            print(f"  错误: 处理该视频时出错: {e}")
            import traceback

            traceback.print_exc()
            error_rows.append(
                {
                    **row.to_dict(),
                    "youtube_url": youtube_url,
                    "title": title,
                    "domain": domain,
                    "category": category,
                    "error_reason": str(e),
                }
            )

    os.makedirs(os.path.dirname(kept_path) or ".", exist_ok=True)
    if passed_rows:
        out_df = pd.DataFrame(passed_rows)
        out_df.to_parquet(kept_path, index=False)
        print(f"\n3_extract 完成, 成功 {len(passed_rows)} 条, 结果已写入: {kept_path}")
    else:
        z = df.iloc[0:0].copy()
        for col in (
            "goal",
            "anchor_timestamp",
            "outcome_timestamp",
            "file_id",
            "response_id_3",
        ):
            if col not in z.columns:
                z[col] = pd.Series(dtype="object")
        z.to_parquet(kept_path, index=False)
        print(f"\n3_extract 完成: 无成功记录, 已写入空表: {kept_path}")

    if error_rows:
        err_df = pd.DataFrame(error_rows)
        err_df.to_parquet(exclude_path, index=False)
        print(f"失败样本已写入: {exclude_path} ({len(err_df)} 条)")
    else:
        _exclude_cols = list(
            dict.fromkeys(
                list(df.columns)
                + ["youtube_url", "title", "domain", "category", "error_reason"]
            )
        )
        pd.DataFrame(columns=_exclude_cols).to_parquet(exclude_path, index=False)
        print(f"无失败样本, 已写入空表: {exclude_path}")

    parts_out = []
    if passed_rows:
        parts_out.append(pd.DataFrame(passed_rows))
    if error_rows:
        parts_out.append(pd.DataFrame(error_rows))
    if parts_out:
        pd.concat(parts_out, ignore_index=True, sort=False).to_parquet(out_full_path, index=False)
        print(f"初稿(成功 ∪ 失败)已写入: {out_full_path}")
    else:
        _full_cols = list(
            dict.fromkeys(
                list(df.columns)
                + [
                    "goal",
                    "anchor_timestamp",
                    "outcome_timestamp",
                    "file_id",
                    "response_id_3",
                    "error_reason",
                ]
            )
        )
        pd.DataFrame(columns=_full_cols).to_parquet(out_full_path, index=False)
        print(f"无处理记录, 已写入空初稿: {out_full_path}")


if __name__ == "__main__":
    main()
