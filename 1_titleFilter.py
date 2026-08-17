import argparse
import glob
import json
import os
from typing import List

import pandas as pd

from pipeline_paths import PROMPT_DIR, RESTATE_DIR, parquet_triplet


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
        raise KeyError("数据中未找到 'title' 列, 无法按标题筛选")

    print(f"加载完成, 共 {len(df)} 行")
    return df

def load_stage1_keywords(
    json_path: str | None = None,
) -> List[str]:









    if json_path is None:
        json_path = os.path.join(PROMPT_DIR, "1_titleFilter.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到关键词配置文件: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    keywords: List[str] = []
    for v in data.values():
        if isinstance(v, list):
            keywords.extend([str(x).strip() for x in v if str(x).strip()])

    print(f"从 '{json_path}' 加载到 {len(keywords)} 个关键词")
    return keywords

def parse_args():
    parser = argparse.ArgumentParser(
        description="根据关键词筛选 parquet 中的 title 并输出新的 parquet 文件"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="输入 parquet 文件或目录路径",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(RESTATE_DIR, "1_titleFilter.parquet"),
        help="保留样本输出路径 (默认: pipeline/ReState/1_titleFilter.parquet); 同目录写入 *_output / *_exclude",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理的行数(用于测试), 不传则处理全部",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    df = load_parquet(args.input)
    keywords = load_stage1_keywords()

    out_full_path, kept_path, exclude_path = parquet_triplet(args.output)
    for p in (out_full_path, kept_path, exclude_path):
        if os.path.exists(p):
            print(f"输出文件已存在,将先删除: {p}")
            os.remove(p)


    n_total = len(df)
    n_process = min(n_total, args.limit) if args.limit is not None else n_total
    if args.limit is not None:
        print(f"限制处理前 {n_process} 行 (--limit={args.limit})")


    df_proc = df.iloc[:n_process].copy()
    mask_keep = []
    for i in range(n_process):
        title = str(df_proc.iloc[i]["title"])

        keep = not any(kw in title for kw in keywords)
        mask_keep.append(keep)

    keep_idx = [i for i, keep in enumerate(mask_keep) if keep]
    drop_idx = [i for i, keep in enumerate(mask_keep) if not keep]
    df_kept_head = df_proc.iloc[keep_idx].copy()
    df_excluded_head = df_proc.iloc[drop_idx].copy()


    for _df in (df_kept_head, df_excluded_head):
        if "youtube_url" not in _df.columns and "url" in _df.columns:
            _df["youtube_url"] = _df["url"]


    if n_process < n_total:
        df_tail = df.iloc[n_process:].copy()
        if "youtube_url" not in df_tail.columns and "url" in df_tail.columns:
            df_tail["youtube_url"] = df_tail["url"]
        df_out = pd.concat([df_kept_head, df_tail], ignore_index=True)
    else:
        df_out = df_kept_head.reset_index(drop=True)

    if "youtube_url" not in df_out.columns and "url" in df_out.columns:
        df_out["youtube_url"] = df_out["url"]

    os.makedirs(os.path.dirname(kept_path) or ".", exist_ok=True)

    df_proc_out = df_proc.copy()
    if "youtube_url" not in df_proc_out.columns and "url" in df_proc_out.columns:
        df_proc_out["youtube_url"] = df_proc_out["url"]
    df_proc_out.to_parquet(out_full_path, index=False)
    print(f"初稿(处理前缀全量)已写入: {out_full_path} ({len(df_proc_out)} 行)")

    if len(df_excluded_head) > 0:
        df_excluded_head.to_parquet(exclude_path, index=False)
        print(f"标题命中关键词已写入: {exclude_path} ({len(df_excluded_head)} 行)")
    else:
        df_proc.iloc[0:0].to_parquet(exclude_path, index=False)
        print(f"无剔除行, 已写入空表: {exclude_path}")

    if len(df_out) > 0:
        df_out.to_parquet(kept_path, index=False)
        print(
            f"过滤后剩余 {len(df_out)} 行(前缀内已删除命中关键词行), 已保存到: {kept_path}"
        )
    else:
        z = df.iloc[0:0].copy()
        if "youtube_url" not in z.columns and "url" in z.columns:
            z["youtube_url"] = pd.Series(dtype="object")
        z.to_parquet(kept_path, index=False)
        print(f"所有标题都命中关键词, 已写入空保留表: {kept_path}")


if __name__ == "__main__":
    main()
