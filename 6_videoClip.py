

















from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from pipeline_paths import (
    HIGHRES_VIDEO_DIR,
    HIGHRES_VIDEO_LIST,
    RESTATE_DIR,
    SPLITTING_DIR,
)

from panda_clips import (
    build_video_filename_to_events,
    extract_youtube_id,
    find_event_containing_timestamp,
    parse_timestamp_to_seconds,
    resolve_clip_range_for_outcome,
    seconds_to_timestamp,
)


DURATION_TOLERANCE_SEC = 0.25

CLIP_COL_START = "clip_start"
CLIP_COL_END = "clip_end"
CLIP_COL_FRAME = "clip_frame"
CLIP_COL_DURATION = "clip_duration"
CLIP_COL_ERROR = "clip_error"
CLIP_DATA_COLS = (
    CLIP_COL_START,
    CLIP_COL_END,
    CLIP_COL_FRAME,
    CLIP_COL_DURATION,
)

CLIP_MODES: Dict[str, Dict[str, Any]] = {
    "short": {
        "min_duration": 3.0,
        "max_duration": 4.0,
        "subdir": "short",
    },
    "long": {
        "min_duration": 3.0,
        "max_duration": 10.0,
        "subdir": "long",
    },
}


def paths_for_mode(output_base: str, mode: str) -> Tuple[str, str, str]:




    base = os.path.abspath(output_base)
    if base.endswith(".parquet"):
        out_dir = os.path.dirname(base) or "."
    elif os.path.isdir(base):
        out_dir = base
    else:
        out_dir = os.path.dirname(base) or "."

    out_full = os.path.join(out_dir, f"6_videoClip_output_{mode}.parquet")
    kept = os.path.join(out_dir, f"6_videoClip_{mode}.parquet")
    exclude = os.path.join(out_dir, f"6_videoClip_exclude_{mode}.parquet")
    return out_full, kept, exclude


def _map_video_paths_to_youtube_ids(
    video_paths: List[str], expected_ids: set[str]
) -> Dict[str, str]:

    id_to_path: Dict[str, str] = {}
    for path in sorted(video_paths):
        if not os.path.isfile(path) or os.path.splitext(path)[1].lower() != ".mp4":
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        for youtube_id in expected_ids:
            suffix = stem[len(youtube_id) :] if stem.startswith(youtube_id) else None
            if suffix == "" or (suffix and suffix[0] in "_.-"):
                id_to_path.setdefault(youtube_id, os.path.abspath(path))
                break
    return id_to_path


def build_youtube_id_to_video_path_from_dir(
    highres_video_path: str, expected_ids: set[str]
) -> Dict[str, str]:
    if not os.path.isdir(highres_video_path):
        return {}
    paths = [os.path.join(highres_video_path, name) for name in os.listdir(highres_video_path)]
    return _map_video_paths_to_youtube_ids(paths, expected_ids)


def build_youtube_id_to_video_path_from_list(
    video_list_path: str, expected_ids: set[str]
) -> Dict[str, str]:
    if not os.path.isfile(video_list_path):
        raise FileNotFoundError(f"未找到视频清单: {video_list_path}")
    list_dir = os.path.dirname(os.path.abspath(video_list_path))
    paths: List[str] = []
    with open(video_list_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            path = line.strip()
            if not path:
                continue
            if not os.path.isabs(path):
                path = os.path.join(list_dir, path)
            paths.append(os.path.abspath(path))
    return _map_video_paths_to_youtube_ids(paths, expected_ids)


def run_splitting_pipeline_batched(
    splitting_dir: str,
    video_list_path: str,
    event_timecode_path: str,
    *,
    device: str = "",
    frame_read_workers: int = 8,
    rerun_cutscene: bool = False,
) -> None:

    from panda_clips import run_panda_splitting_pipeline

    required_files = (
        "cutscene_detect.py",
        "event_stitching.py",
        os.path.join("ImageBind", "models", "imagebind_model.py"),
    )
    missing = [
        rel
        for rel in required_files
        if not os.path.isfile(os.path.join(os.path.abspath(splitting_dir), rel))
    ]
    if missing:
        raise FileNotFoundError(
            f"splitting 目录不完整 ({splitting_dir}): 缺少 {', '.join(missing)}"
        )

    cutscene_frame_idx_path = os.path.join(
        os.path.abspath(splitting_dir), "cutscene_frame_idx.json"
    )
    run_panda_splitting_pipeline(
        splitting_dir,
        os.path.abspath(video_list_path),
        os.path.abspath(event_timecode_path),
        cutscene_frameidx_path=cutscene_frame_idx_path,
        device=device or None,
        frame_read_workers=frame_read_workers,
        rerun_cutscene=rerun_cutscene,
    )


def _run_ffmpeg_checked(cmd: List[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "ffmpeg 失败").strip()
        raise RuntimeError(msg)


def probe_video_duration_sec(video_path: str) -> Optional[float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(r.stdout.strip())
    except Exception:
        return None


def clip_duration_in_range(
    actual_sec: float,
    min_duration: float,
    max_duration: float,
) -> bool:
    lo = min_duration - DURATION_TOLERANCE_SEC
    hi = max_duration + DURATION_TOLERANCE_SEC
    return lo <= actual_sec <= hi


def run_ffmpeg_clip_strict(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
) -> None:

    from media_specs import clip_save_size

    if end_sec <= start_sec:
        raise ValueError(
            f"end_sec ({end_sec}) 必须大于 start_sec ({start_sec})"
        )
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    w, h = clip_save_size()
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-ss",
        f"{start_sec:.6f}",
        "-to",
        f"{end_sec:.6f}",
        "-map",
        "0:v:0",
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        output_path,
    ]
    _run_ffmpeg_checked(cmd)


def event_timecode_needs_refresh(
    video_paths: set,
    event_timecode_path: str,
) -> bool:
    return json_index_needs_refresh(video_paths, event_timecode_path)


def json_index_needs_refresh(video_paths: set[str], index_path: str) -> bool:

    if not os.path.isfile(index_path):
        return True
    try:
        with open(index_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return True
    if not isinstance(data, dict):
        return True
    return any(os.path.basename(path) not in data for path in video_paths)


def filename_safe_timestamp(timestamp: str) -> str:

    return timestamp.replace(":", "-")


def write_parquet_atomic(df: pd.DataFrame, output_path: str) -> None:

    tmp_path = f"{output_path}.tmp"
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def compute_clip_for_row(
    row: pd.Series,
    *,
    id_to_path: Dict[str, str],
    filename_to_events: Dict[str, List[Tuple[float, float]]],
    mode: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    spec = CLIP_MODES[mode]
    youtube_url = str(row.get("youtube_url", "") or "")
    outcome_ts = str(row.get("outcome_timestamp", "") or "")
    if not youtube_url:
        return None, "missing_youtube_url"
    if not outcome_ts:
        return None, "missing_outcome_timestamp"

    try:
        youtube_id = extract_youtube_id(youtube_url)
    except Exception:
        return None, "invalid_youtube_url"

    video_path = id_to_path.get(youtube_id)
    if not video_path or not os.path.exists(video_path):
        return None, "video_not_found"

    video_filename = os.path.basename(video_path)
    events = filename_to_events.get(video_filename)
    if not events:
        return None, "no_events_in_timecode"

    try:
        outcome_sec = parse_timestamp_to_seconds(outcome_ts)
    except Exception:
        return None, "invalid_outcome_timestamp"

    event = find_event_containing_timestamp(events, outcome_sec)
    if event is None:
        return None, "no_event_for_outcome"

    video_duration = probe_video_duration_sec(video_path)
    resolved = resolve_clip_range_for_outcome(
        outcome_sec,
        event,
        min_duration=spec["min_duration"],
        max_duration=spec["max_duration"],
        video_duration=video_duration,
    )
    if resolved is None:
        return None, "clip_range_compute_failed"

    clip_start_sec, clip_end_sec, clip_source = resolved
    if not (clip_start_sec <= outcome_sec <= clip_end_sec):
        return None, "outcome_not_in_clip_range"

    return {
        "youtube_id": youtube_id,
        "video_path": video_path,
        "clip_start_sec": clip_start_sec,
        "clip_end_sec": clip_end_sec,
        "clip_start_timestamp": seconds_to_timestamp(clip_start_sec),
        "clip_end_timestamp": seconds_to_timestamp(clip_end_sec),
        "planned_duration_sec": clip_end_sec - clip_start_sec,
        "clip_source": clip_source,
    }, ""


def _remove_file_quiet(path: str) -> None:
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "6_videoClip: panda splitting + outcome event 裁切 short(3-4s) 或 long(3-10s), "
            "ffprobe 校验时长后写出 output/kept/exclude 三份 parquet"
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        default=os.path.join(RESTATE_DIR, "5_instruction.parquet"),
        help="输入 parquet (默认: pipeline/ReState/5_instruction.parquet)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(RESTATE_DIR, "6_videoClip.parquet"),
        help=(
            "输出目录锚点(默认 pipeline/ReState/6_videoClip.parquet 所在目录); "
            "实际文件为 6_videoClip_output_{mode}.parquet / 6_videoClip_{mode}.parquet / "
            "6_videoClip_exclude_{mode}.parquet"
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=sorted(CLIP_MODES.keys()),
        required=True,
        help="short(3-4s) 或 long(3-10s)",
    )
    parser.add_argument(
        "--highres_video_list",
        type=str,
        default=HIGHRES_VIDEO_LIST,
        help="视频路径清单（通过 --highres_video_list 或 HIGHRES_VIDEO_LIST 指定）",
    )
    parser.add_argument(
        "--highres_video_path",
        type=str,
        default=HIGHRES_VIDEO_DIR,
        help="高清视频根目录（通过 --highres_video_path 或 HIGHRES_VIDEO_DIR 指定）",
    )
    parser.add_argument(
        "--event_timecode",
        type=str,
        default=os.path.join(RESTATE_DIR, "event_timecode.json"),
        help="event_timecode.json (splitting 输出)",
    )
    parser.add_argument(
        "--splitting_dir",
        type=str,
        default=SPLITTING_DIR,
        help="切分脚本目录（通过 --splitting_dir 或 SPLITTING_DIR 指定）",
    )
    parser.add_argument(
        "--run_splitting",
        action="store_true",
        help="强制重新运行 splitting (cutscene + ImageBind)",
    )
    parser.add_argument(
        "--skip_splitting",
        action="store_true",
        help="不运行 splitting, 仅使用已有 event_timecode.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="event_stitching 设备: cuda / cpu / cuda:0; 默认自动",
    )
    parser.add_argument(
        "--frame_read_workers",
        type=int,
        default=8,
        help="读帧线程数 (不用多进程, 避免破坏 CUDA)",
    )
    parser.add_argument(
        "--output_clip_dir",
        type=str,
        default=os.path.join(RESTATE_DIR, "video_clip"),
        help="切分 mp4 根目录; short/long 子目录",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 条(测试用)",
    )
    return parser.parse_args()


def load_parquet(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"未找到输入文件: {input_path}")
    df = pd.read_parquet(input_path)

    required_cols = [
        "youtube_url",
        "anchor_timestamp",
        "outcome_timestamp",
        "instruction_zh",
        "instruction_en",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"数据中未找到 '{col}' 列, 请使用 5_instruction 输出作为输入")
    return df


def resolve_id_to_path(
    df: pd.DataFrame,
    *,
    highres_video_list: str,
    highres_video_path: str,
) -> Dict[str, str]:
    expected_ids: set[str] = set()
    for youtube_url in df.get("youtube_url", pd.Series(dtype=str)).fillna("").astype(str):
        try:
            expected_ids.add(extract_youtube_id(youtube_url))
        except ValueError:
            continue

    if highres_video_list:
        id_to_path = build_youtube_id_to_video_path_from_list(
            highres_video_list, expected_ids
        )
    else:
        id_to_path = build_youtube_id_to_video_path_from_dir(
            highres_video_path, expected_ids
        )

    if not id_to_path:
        raise FileNotFoundError(
            "未找到任何视频: 请提供 --highres_video_list 或有效的 --highres_video_path"
        )
    return id_to_path


def collect_needed_video_paths(
    df: pd.DataFrame,
    id_to_path: Dict[str, str],
) -> set:
    paths = set()
    for _, row in df.iterrows():
        youtube_url = str(row.get("youtube_url", "") or "")
        if not youtube_url:
            continue
        try:
            youtube_id = extract_youtube_id(youtube_url)
        except Exception:
            continue
        video_path = id_to_path.get(youtube_id)
        if video_path and os.path.exists(video_path):
            paths.add(video_path)
    return paths


def ensure_event_timecodes(
    needed_video_paths: set,
    *,
    event_timecode_path: str,
    splitting_dir: str,
    run_splitting: bool,
    skip_splitting: bool,
    device: str = "",
    frame_read_workers: int = 8,
) -> None:
    if skip_splitting:
        if not os.path.isfile(event_timecode_path):
            raise FileNotFoundError(
                f"--skip_splitting 但未找到 {event_timecode_path}, 请先运行 splitting"
            )
        return

    if not needed_video_paths:
        raise FileNotFoundError("输入中无可用视频路径")

    cutscene_path = os.path.join(
        os.path.abspath(splitting_dir), "cutscene_frame_idx.json"
    )
    need_run = (
        run_splitting
        or event_timecode_needs_refresh(needed_video_paths, event_timecode_path)
    )
    rerun_cutscene = run_splitting or json_index_needs_refresh(
        needed_video_paths, cutscene_path
    )
    if not need_run:
        print(f"[INFO] 复用已有 event_timecode: {event_timecode_path}")
        return

    import tempfile

    os.makedirs(os.path.dirname(os.path.abspath(event_timecode_path)) or ".", exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8", newline="\n"
    )
    tmp_path = tmp.name
    with tmp:
        for p in sorted(needed_video_paths):
            tmp.write(f"{os.path.abspath(p)}\n")
    try:
        run_splitting_pipeline_batched(
            splitting_dir=splitting_dir,
            video_list_path=tmp_path,
            event_timecode_path=event_timecode_path,
            device=device,
            frame_read_workers=frame_read_workers,
            rerun_cutscene=rerun_cutscene,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def main() -> None:
    args = parse_args()
    if args.run_splitting and args.skip_splitting:
        raise ValueError("--run_splitting 与 --skip_splitting 不能同时使用")
    if args.frame_read_workers < 1:
        raise ValueError("--frame_read_workers 必须大于或等于 1")
    mode = args.mode
    spec = CLIP_MODES[mode]
    min_dur = spec["min_duration"]
    max_dur = spec["max_duration"]
    clip_subdir = os.path.join(args.output_clip_dir, spec["subdir"])

    out_full_path, kept_path, exclude_path = paths_for_mode(args.output, mode)
    os.makedirs(os.path.dirname(out_full_path) or ".", exist_ok=True)

    df = load_parquet(args.input)
    if args.limit is not None and args.limit > 0:
        df = df.iloc[: args.limit].copy()

    id_to_path = resolve_id_to_path(
        df,
        highres_video_list=args.highres_video_list,
        highres_video_path=args.highres_video_path,
    )
    needed_video_paths = collect_needed_video_paths(df, id_to_path)
    ensure_event_timecodes(
        needed_video_paths,
        event_timecode_path=args.event_timecode,
        splitting_dir=args.splitting_dir,
        run_splitting=args.run_splitting,
        skip_splitting=args.skip_splitting,
        device=args.device,
        frame_read_workers=args.frame_read_workers,
    )
    filename_to_events = build_video_filename_to_events(args.event_timecode)

    os.makedirs(clip_subdir, exist_ok=True)

    start_list: List[str] = []
    end_list: List[str] = []
    frame_list: List[str] = []
    dur_list: List[Any] = []
    err_list: List[str] = []

    n_ok = 0
    n_fail = 0

    for idx, row in df.iterrows():
        title = str(row.get("title", "") or "")
        err_msg = ""
        start_ts = ""
        end_ts = ""
        out_path = ""
        actual_dur: Any = pd.NA

        res, compute_err = compute_clip_for_row(
            row,
            id_to_path=id_to_path,
            filename_to_events=filename_to_events,
            mode=mode,
        )
        if res is None:
            err_msg = compute_err or "clip_range_compute_failed"
        else:
            youtube_id = res["youtube_id"]
            video_path = res["video_path"]
            clip_start_sec = res["clip_start_sec"]
            clip_end_sec = res["clip_end_sec"]
            start_ts = res["clip_start_timestamp"]
            end_ts = res["clip_end_timestamp"]
            planned = res["planned_duration_sec"]
            out_filename = (
                f"{youtube_id}_{filename_safe_timestamp(start_ts)}_"
                f"{filename_safe_timestamp(end_ts)}.mp4"
            )
            out_path = os.path.join(clip_subdir, out_filename)

            try:
                run_ffmpeg_clip_strict(
                    video_path=video_path,
                    start_sec=clip_start_sec,
                    end_sec=clip_end_sec,
                    output_path=out_path,
                )
            except Exception as e:
                err_msg = f"ffmpeg_failed: {e}"
                _remove_file_quiet(out_path)
                out_path = ""

            if not err_msg and out_path:
                probed = probe_video_duration_sec(out_path)
                if probed is None:
                    err_msg = "ffprobe_duration_failed"
                    _remove_file_quiet(out_path)
                    out_path = ""
                else:
                    actual_dur = float(probed)
                    if not clip_duration_in_range(actual_dur, min_dur, max_dur):
                        err_msg = (
                            f"duration_out_of_range: actual={actual_dur:.3f}s, "
                            f"required=[{min_dur}, {max_dur}]s "
                            f"(±{DURATION_TOLERANCE_SEC}s), planned={planned:.3f}s"
                        )
                        _remove_file_quiet(out_path)
                        out_path = ""
                        start_ts = ""
                        end_ts = ""

        ok = not err_msg and bool(out_path)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            if not err_msg:
                err_msg = "unknown_failure"

        start_list.append(start_ts)
        end_list.append(end_ts)
        frame_list.append(out_path)
        dur_list.append(actual_dur if ok else pd.NA)
        err_list.append("" if ok else err_msg)

        status = "OK" if ok else f"SKIP ({err_msg})"
        dur_print = (
            f", duration={float(actual_dur):.3f}s" if ok and actual_dur is not pd.NA else ""
        )
        print(f"[{idx + 1}/{len(df)}] [{mode}] {title} -> {status}{dur_print}")

    df_out = df.copy()
    df_out[CLIP_COL_START] = start_list
    df_out[CLIP_COL_END] = end_list
    df_out[CLIP_COL_FRAME] = frame_list
    df_out[CLIP_COL_DURATION] = dur_list
    df_out[CLIP_COL_ERROR] = err_list

    kept_mask = df_out[CLIP_COL_ERROR].fillna("").astype(str).str.strip() == ""
    kept_df = (
        df_out.loc[kept_mask]
        .drop(columns=[CLIP_COL_ERROR], errors="ignore")
        .reset_index(drop=True)
    )
    exclude_df = df_out[~kept_mask].reset_index(drop=True)

    write_parquet_atomic(df_out, out_full_path)
    write_parquet_atomic(kept_df, kept_path)
    write_parquet_atomic(exclude_df, exclude_path)

    print(
        f"\n6_videoClip [{mode}] 完成: 成功 {n_ok} 条, 失败 {n_fail} 条\n"
        f"  全量 -> {out_full_path} ({len(df_out)})\n"
        f"  保留 -> {kept_path} ({len(kept_df)})\n"
        f"  排除 -> {exclude_path} ({len(exclude_df)})"
    )


if __name__ == "__main__":
    main()
