import argparse
import glob
import os
import re
from typing import List

import imageio
import pandas as pd
from PIL import Image  # noqa: F401  
from volcenginesdkarkruntime import Ark

from agent_process import (
    API_KEY as DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MODEL_NAME,
    create_image_grid,
    extract_uniform_frames,
    image_to_base64,
    extract_video_id_from_url,
    set_model_name,
)

from pipeline_paths import (
    LOWRES_VIDEO_DIR,
    PROMPT_DIR,
    RESTATE_DIR,
    parquet_error_path,
    parquet_triplet,
)


def load_parquet(input_path: str) -> pd.DataFrame:






    if os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "*.parquet"))
        if not files:
            raise FileNotFoundError(f"在目录 '{input_path}' 下未找到任何 parquet 文件")
        print(f"发现 {len(files)} 个 parquet 文件,正在加载...")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"未找到文件: {input_path}")
        print(f"正在读取 parquet 文件: {input_path}")
        df = pd.read_parquet(input_path)

    if "title" not in df.columns:
        raise KeyError("数据中未找到 'title' 列, 无法进行分类")

    if "youtube_url" not in df.columns and "url" not in df.columns:
        raise KeyError("数据中未找到 'youtube_url' 或 'url' 列, 无法定位视频")

    print(f"加载完成, 共 {len(df)} 行")
    return df


def load_stage1_prompt(prompt_dir: str) -> str:



    stage1_prompt_file = os.path.join(prompt_dir, "2_classify.txt")
    if not os.path.exists(stage1_prompt_file):
        raise FileNotFoundError(f"未找到 2_classify.txt: {stage1_prompt_file}")
    with open(stage1_prompt_file, "r", encoding="utf-8") as f:
        return f.read()


def parse_stage1_response(response) -> dict:



    response_text = None
    try:
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

        if not response_text:
            return {
                "domain": "Uncertain",
                "category": "",
            }

        cleaned = response_text.replace("```json", "").replace("```", "").strip()


        domain_match = re.search(r"domain\s*:\s*([^\n]+)", cleaned, re.IGNORECASE)
        if domain_match:
            domain = domain_match.group(1).strip()
        else:
            domain = "Uncertain"


        category_match = re.search(r"category\s*:\s*([^\n]+)", cleaned, re.IGNORECASE)
        category = category_match.group(1).strip() if category_match else ""

        return {
            "domain": domain,
            "category": category,
        }

    except Exception as e:
        print(f"处理阶段一分类响应时出错: {e}")
        import traceback

        traceback.print_exc()
        return {
            "domain": "Uncertain",
            "category": "",
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 VLM 对 parquet 中的视频分类; 产出 *_output(全部分类)、2_classify.parquet(非 Uncertain)、*_exclude(Uncertain)、*_error(技术失败,仅原列)"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=os.path.join(RESTATE_DIR, "1_titleFilter.parquet"),
        help="输入 parquet 文件或目录路径 (默认: pipeline/ReState/1_titleFilter.parquet)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(RESTATE_DIR, "2_classify.parquet"),
        help="非 Uncertain 结果路径 (默认: pipeline/ReState/2_classify.parquet); 同目录写入 *_output(全部分类行)、*_exclude(Uncertain)、*_error(技术失败,仅原列)",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=PROMPT_DIR,
        help=f"包含 2_classify.txt 的目录 (默认: {PROMPT_DIR})",
    )
    parser.add_argument(
        "--lowres_video_path",
        type=str,
        default=LOWRES_VIDEO_DIR,
        help="低清视频根目录（通过 --lowres_video_path 或 LOWRES_VIDEO_DIR 指定）",
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
        help="限制处理的样本数量(用于测试),None 表示全部",
    )
    return parser.parse_args()


def main():
    args = parse_args()


    set_model_name(args.model)

    df = load_parquet(args.input)
    stage1_prompt = load_stage1_prompt(args.prompt_dir)


    client = Ark(
        base_url=DEFAULT_BASE_URL,
        api_key=args.api_key,
    )

    out_full_path, kept_path, exclude_path = parquet_triplet(args.output)
    error_path = parquet_error_path(kept_path)
    for p in (out_full_path, kept_path, exclude_path, error_path):
        if os.path.exists(p):
            print(f"输出文件已存在,将先删除: {p}")
            os.remove(p)


    classified_rows: List[dict] = []
    error_rows: List[dict] = []
    total = len(df)

    limit = args.limit if args.limit is not None else total
    n_process = min(total, limit)

    for idx in range(n_process):
        row = df.iloc[idx]
        youtube_url = str(row.get("youtube_url", row.get("url", "")))
        title = str(row["title"])

        print(f"\n[{idx + 1}/{n_process}] 处理视频: {title}")
        print(f"URL: {youtube_url}")


        video_id = extract_video_id_from_url(youtube_url)
        if not video_id:
            print("  [警告] 无法从 URL 提取 video_id, 跳过")
            error_rows.append(row.to_dict())
            continue

        video_pattern = os.path.join(args.lowres_video_path, f"{video_id}_*.mp4")
        video_files = glob.glob(video_pattern)
        if not video_files:
            print(f"  [警告] 未找到低清视频文件: {video_pattern}, 跳过")
            error_rows.append(row.to_dict())
            continue
        video_path = video_files[0]


        try:
            reader = imageio.get_reader(video_path, format="FFMPEG")
            metadata = reader.get_meta_data()
            video_duration = metadata.get("duration", 0)
            if video_duration == 0 or video_duration is None:
                try:
                    fps_val = metadata.get("fps", 30.0)
                    frame_count = reader.count_frames()
                    video_duration = frame_count / fps_val if fps_val > 0 else 0
                except Exception:
                    video_duration = 3600.0
            reader.close()
        except Exception as e:
            print(f"  [错误] 读取视频元数据失败: {e}")
            error_rows.append(row.to_dict())
            continue

        def _get_frame_count_by_duration(duration: float) -> int:

            if duration <= 180:
                return 16
            elif duration <= 300:
                return 25
            else:
                return 36

        num_frames = _get_frame_count_by_duration(video_duration)


        try:
            frames = extract_uniform_frames(video_path, num_frames)
            if not frames:
                print("  [警告] 未能提取任何帧, 跳过")
                error_rows.append(row.to_dict())
                continue
            image_grid = create_image_grid(frames)
            image_grid_base64 = image_to_base64(image_grid)
        except Exception as e:
            print(f"  [错误] 抽帧或拼接 image grid 失败: {e}")
            error_rows.append(row.to_dict())
            continue


        input_text = stage1_prompt.replace("{title}", title)


        try:
            response = client.responses.create(
                model=MODEL_NAME,
                input=[
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": input_text}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{image_grid_base64}",
                            }
                        ],
                    },
                ],
            )
        except Exception as e:
            print(f"  [错误] 调用 VLM 接口失败: {e}")
            error_rows.append(row.to_dict())
            continue


        result = parse_stage1_response(response)
        domain = result.get("domain", "Uncertain")
        category = result.get("category", "")

        print(f"  分类结果 -> domain: {domain}, category: {category}")


        out_row = row.to_dict()
        out_row["youtube_url"] = youtube_url
        out_row["title"] = title
        out_row["domain"] = domain
        out_row["category"] = category

        classified_rows.append(out_row)

    os.makedirs(os.path.dirname(kept_path) or ".", exist_ok=True)

    classified_cols = list(dict.fromkeys(list(df.columns) + ["domain", "category"]))

    if error_rows:
        err_df = pd.DataFrame(error_rows)
        err_df.to_parquet(error_path, index=False)
        print(f"\n技术失败样本已写入: {error_path} ({len(err_df)} 条, 原列)")
    else:
        pd.DataFrame(columns=list(df.columns)).to_parquet(error_path, index=False)
        print(f"\n无技术失败样本, 已写入空表: {error_path}")

    if classified_rows:
        all_classified = pd.DataFrame(classified_rows)
        all_classified.to_parquet(out_full_path, index=False)
        print(f"全部分类结果(含 Uncertain)已写入: {out_full_path} ({len(all_classified)} 条)")

        dom = (
            all_classified["domain"]
            .fillna("")
            .map(lambda x: str(x).strip().lower())
        )
        uncertain_mask = dom.isin(["uncertain", "", "nan"])
        uncertain_df = all_classified[uncertain_mask]
        kept_df = all_classified[~uncertain_mask]

        uncertain_df.to_parquet(exclude_path, index=False)
        print(f"Uncertain 样本已写入: {exclude_path} ({len(uncertain_df)} 条)")

        kept_df.to_parquet(kept_path, index=False)
        print(f"非 Uncertain 样本已写入: {kept_path} ({len(kept_df)} 条)")
    else:
        empty_cls = pd.DataFrame(columns=classified_cols)
        empty_cls.to_parquet(out_full_path, index=False)
        empty_cls.to_parquet(exclude_path, index=False)
        empty_cls.to_parquet(kept_path, index=False)
        print(
            f"\n无 VLM 分类行: 已写入空表 {out_full_path}、{exclude_path}、{kept_path}"
        )

if __name__ == "__main__":
    main()
