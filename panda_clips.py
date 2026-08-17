import os
import re
import json
import argparse
import importlib.util
import subprocess
import sys
import tempfile
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta

from pipeline_paths import HIGHRES_VIDEO_LIST, RESTATE_DIR, SPLITTING_DIR, configured_path


DEFAULT_PANDA_INPUT_JSON = configured_path(
    "PANDA_INPUT_JSON", os.path.join(RESTATE_DIR, "stage3_panda_clips.json")
)
DEFAULT_EVENT_TIMECODE = configured_path(
    "EVENT_TIMECODE_PATH", os.path.join(RESTATE_DIR, "event_timecode.json")
)
DEFAULT_PANDA_OUTPUT_DIR = configured_path(
    "PANDA_OUTPUT_DIR", os.path.join(RESTATE_DIR, "panda_clips")
)


def parse_timestamp_to_seconds(ts: str) -> float:










    ts = ts.strip()
    if not ts:
        raise ValueError("Empty timestamp string")


    if ts.count(":") == 2 and "." in ts:
        try:
            dt = datetime.strptime(ts, "%H:%M:%S.%f")
            return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1000000.0
        except ValueError:
            pass

    parts = ts.split(":")

    if len(parts) == 1:
        return float(parts[0])

    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds

    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    raise ValueError(f"Unrecognized timestamp format: {ts}")


def seconds_to_timestamp(seconds: float) -> str:








    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:06.3f}"


def build_youtube_id_to_video_path(video_list_path: str) -> Dict[str, str]:






    id_to_path: Dict[str, str] = {}
    with open(video_list_path, "r") as f:
        for line in f:
            path = line.strip()
            if not path:
                continue
            base = os.path.basename(path)

            if "_" in base:
                youtube_id = base.split("_", 1)[0]
            else:
                youtube_id = os.path.splitext(base)[0]
            if youtube_id:
                id_to_path[youtube_id] = path
    return id_to_path


def build_video_filename_to_events(event_timecode_path: str) -> Dict[str, List[Tuple[float, float]]]:







    with open(event_timecode_path, "r") as f:
        data = json.load(f)
    
    filename_to_events: Dict[str, List[Tuple[float, float]]] = {}
    for filename, events in data.items():
        event_list = []
        for event in events:
            start_ts, end_ts = event[0], event[1]
            start_sec = parse_timestamp_to_seconds(start_ts)
            end_sec = parse_timestamp_to_seconds(end_ts)
            event_list.append((start_sec, end_sec))
        filename_to_events[filename] = event_list
    return filename_to_events


def find_event_containing_timestamp(
    events: List[Tuple[float, float]], 
    target_sec: float
) -> Optional[Tuple[float, float]]:




    for start_sec, end_sec in events:

        if start_sec <= target_sec <= end_sec:
            return (start_sec, end_sec)
    return None


def _shift_window_to_bounds(
    start: float,
    end: float,
    outcome_sec: float,
    bounds_lo: float,
    bounds_hi: float,
) -> Tuple[float, float]:




    lo = float(bounds_lo)
    hi = float(bounds_hi)
    start, end = float(start), float(end)

    if end > hi:
        shift = end - hi
        start -= shift
        end = hi
    if start < lo:
        shift = lo - start
        start = lo
        end += shift
    if end > hi:
        shift = end - hi
        start -= shift
        end = hi
    if start < lo:
        start = lo
        end = min(hi, end)

    if not (start <= outcome_sec <= end):
        o = float(outcome_sec)
        dur = end - start
        if dur <= 0:
            return start, end
        start = max(lo, min(o, hi - dur))
        end = start + dur
        if end > hi:
            end = hi
            start = max(lo, end - dur)

    return start, end


def outcome_centered_clip_range(
    outcome_sec: float,
    target_duration: float,
    *,
    min_duration: float,
    max_duration: float,
    bounds_lo: float = 0.0,
    bounds_hi: Optional[float] = None,
) -> Optional[Tuple[float, float]]:



    o = float(outcome_sec)
    target = max(float(min_duration), min(float(max_duration), float(target_duration)))
    start = o - target / 2.0
    end = o + target / 2.0

    hi = float(bounds_hi) if bounds_hi is not None and bounds_hi > 0 else None
    if hi is not None:
        start, end = _shift_window_to_bounds(start, end, o, bounds_lo, hi)

    dur = end - start
    if dur <= 0 or dur < min_duration - 1e-3:
        return None
    if dur > max_duration + 1e-3:
        start = o - max_duration / 2.0
        end = o + max_duration / 2.0
        if hi is not None:
            start, end = _shift_window_to_bounds(start, end, o, bounds_lo, hi)
        dur = end - start
        if dur > max_duration + 1e-3 or dur < min_duration - 1e-3:
            return None

    if not (start <= o <= end):
        return None
    return float(start), float(end)


def resolve_clip_range_for_outcome(
    outcome_sec: float,
    event: Optional[Tuple[float, float]],
    *,
    min_duration: float,
    max_duration: float,
    video_duration: Optional[float] = None,
) -> Optional[Tuple[float, float, str]]:










    if event is None:
        return None

    es, ee = float(event[0]), float(event[1])
    o = float(outcome_sec)
    if not (es <= o <= ee):
        return None

    bounds_hi = float(video_duration) if video_duration and video_duration > 0 else None
    event_dur = ee - es

    if event_dur >= min_duration - 1e-3 and event_dur <= max_duration + 1e-3:
        return es, ee, "event_fit"

    target = max_duration if event_dur > max_duration + 1e-3 else min_duration
    clipped = outcome_centered_clip_range(
        o,
        target,
        min_duration=min_duration,
        max_duration=max_duration,
        bounds_lo=0.0,
        bounds_hi=bounds_hi,
    )
    if clipped is None:
        return None
    return clipped[0], clipped[1], "outcome_centered"


def extract_youtube_id(youtube_url: str) -> str:






    m = re.search(r"[?&]v=([A-Za-z0-9_-]+)", youtube_url)
    if m:
        return m.group(1)

    m = re.search(r"youtu\.be/([A-Za-z0-9_-]+)", youtube_url)
    if m:
        return m.group(1)

    m = re.search(r"/([A-Za-z0-9_-]+)$", youtube_url)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot extract youtube id from url: {youtube_url}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_ffmpeg_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
) -> None:




    if end_sec <= start_sec:
        raise ValueError(
            f"end_sec ({end_sec}) must be greater than start_sec ({start_sec})"
        )

    # ``output_path`` may be a bare filename in the current directory.
    ensure_dir(os.path.dirname(output_path) or ".")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",


        "-i",
        video_path,
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-y",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def run_panda_splitting_pipeline(
    splitting_dir: str,
    video_list_path: str,
    event_timecode_path: str,
    *,
    cutscene_frameidx_path: str = "",
    device: Optional[str] = None,
    frame_read_workers: int = 8,
    rerun_cutscene: bool = False,
) -> None:








    splitting_dir = os.path.abspath(splitting_dir)
    if not cutscene_frameidx_path:
        cutscene_frameidx_path = os.path.join(splitting_dir, "cutscene_frame_idx.json")

    module_path = os.path.join(splitting_dir, "event_stitching.py")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"Missing splitting entry point: {module_path}")

    inserted_path = splitting_dir not in sys.path
    try:
        if inserted_path:
            sys.path.insert(0, splitting_dir)
        module_name = f"_pipeline_event_stitching_{abs(hash(module_path))}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load splitting entry point: {module_path}")
        es_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(es_mod)

        es_mod.run_panda_splitting_pipeline(
            os.path.abspath(video_list_path),
            os.path.abspath(cutscene_frameidx_path),
            os.path.abspath(event_timecode_path),
            device=device,
            frame_read_workers=frame_read_workers,
            rerun_cutscene=rerun_cutscene,
        )
    finally:
        if inserted_path:
            try:
                sys.path.remove(splitting_dir)
            except ValueError:
                pass


def run_splitting_pipeline(
    splitting_dir: str,
    video_list_path: str,
    event_timecode_path: str,
) -> None:

    run_panda_splitting_pipeline(
        splitting_dir,
        video_list_path,
        event_timecode_path,
        rerun_cutscene=True,
    )


def process_stage3_panda_clips(
    json_path: str,
    video_list_path: str,
    event_timecode_path: str,
    output_clip_dir: str,
    splitting_dir: str,
    run_splitting: bool = False,
    limit: Optional[int] = None,
) -> None:










    if not os.path.exists(json_path):
        print(f"[INFO] {json_path} 不存在，尝试从 stage3_results.json 或 stage3_no_video.json 创建...")
        

        json_dir = os.path.dirname(json_path)
        stage3_results_json = os.path.join(json_dir, "stage3_results.json")
        stage3_no_video_json = os.path.join(json_dir, "stage3_no_video.json")
        
        source_json = None
        if os.path.exists(stage3_results_json):
            source_json = stage3_results_json
            print(f"[INFO] 找到 {stage3_results_json}，将从此文件创建")
        elif os.path.exists(stage3_no_video_json):
            source_json = stage3_no_video_json
            print(f"[INFO] 找到 {stage3_no_video_json}，将从此文件创建")
        else:
            raise FileNotFoundError(
                f"无法创建 {json_path}：找不到源文件 "
                f"({stage3_results_json} 或 {stage3_no_video_json})"
            )
        

        with open(source_json, "r", encoding="utf-8") as f:
            source_data = json.load(f)
        

        for item in source_data:
            if "panda_clip" not in item:
                item["panda_clip"] = None
        

        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(source_data, f, ensure_ascii=False, indent=2)
        
        print(f"[INFO] 已创建 {json_path}，包含 {len(source_data)} 条记录")
    

    with open(json_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)


    if limit is not None and limit > 0:
        data = data[:limit]
        print(f"[INFO] Limiting processing to first {limit} items for testing.")

    id_to_path = build_youtube_id_to_video_path(video_list_path)
    

    if run_splitting or not os.path.exists(event_timecode_path):

        needed_video_paths = set()
        for item in data:
            youtube_url = item.get("youtube_url")
            if not youtube_url:
                continue
            try:
                youtube_id = extract_youtube_id(youtube_url)
            except ValueError:
                continue
            video_path = id_to_path.get(youtube_id)
            if video_path and os.path.exists(video_path):
                needed_video_paths.add(video_path)
        
        if not needed_video_paths:
            print("[WARN] No valid video paths found in JSON, skipping splitting.")
        else:

            temp_video_list = tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.txt', 
                delete=False,
                dir=splitting_dir
            )
            temp_video_list_path = temp_video_list.name
            for video_path in sorted(needed_video_paths):
                temp_video_list.write(f"{video_path}\n")
            temp_video_list.close()
            
            print(f"[INFO] Running splitting for {len(needed_video_paths)} videos from JSON...")
            try:
                run_splitting_pipeline(splitting_dir, temp_video_list_path, event_timecode_path)
            finally:

                if os.path.exists(temp_video_list_path):
                    os.remove(temp_video_list_path)
    filename_to_events = build_video_filename_to_events(event_timecode_path)


    total_items = len(data)
    processed_count = 0
    skipped_count = 0
    already_has_clip_count = 0
    
    print(f"[INFO] Processing {total_items} items...")
    
    for idx, item in enumerate(data):


        existing_clip = item.get("panda_clip")
        if existing_clip and isinstance(existing_clip, dict) and "clip_start_timestamp" in existing_clip:
            already_has_clip_count += 1
            continue

        youtube_url = item.get("youtube_url")
        if not youtube_url:
            print(f"[WARN] item {idx} missing youtube_url, skip.")
            skipped_count += 1
            continue

        try:
            youtube_id = extract_youtube_id(youtube_url)
        except ValueError as e:
            print(f"[WARN] item {idx} cannot extract youtube id: {e}")
            skipped_count += 1
            continue

        video_path = id_to_path.get(youtube_id)
        if not video_path or not os.path.exists(video_path):
            print(
                f"[WARN] item {idx} youtube_id={youtube_id} has no local video, skip."
            )
            skipped_count += 1
            continue


        video_filename = os.path.basename(video_path)
        events = filename_to_events.get(video_filename)
        if not events:
            print(
                f"[WARN] item {idx} video_filename={video_filename} has no events in event_timecode.json, skip."
            )
            skipped_count += 1
            continue

        end_ts = item.get("end_timestamp")
        if not end_ts:
            print(f"[WARN] item {idx} missing end_timestamp, skip.")
            skipped_count += 1
            continue

        try:
            end_sec = parse_timestamp_to_seconds(end_ts)
        except Exception as e:
            print(f"[WARN] item {idx} cannot parse end_timestamp={end_ts}: {e}")
            skipped_count += 1
            continue


        event = find_event_containing_timestamp(events, end_sec)
        if not event:
            print(
                f"[WARN] item {idx} end_timestamp={end_sec:.3f}s not found in any event for {video_filename}, skip."
            )
            skipped_count += 1
            continue

        start_sec, event_end_sec = event


        clip_filename = f"{youtube_id}_panda_clip.mp4"
        output_path = os.path.join(output_clip_dir, clip_filename)

        print(
            f"[INFO] item {idx}: id={youtube_id}, "
            f"video={video_path}, event=[{start_sec:.3f}, {event_end_sec:.3f}], "
            f"end_timestamp={end_sec:.3f}, clip={output_path}"
        )

        try:
            run_ffmpeg_clip(video_path, start_sec, event_end_sec, output_path)
            processed_count += 1
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] ffmpeg failed for item {idx}, id={youtube_id}: {e}")
            skipped_count += 1
            continue
        except Exception as e:
            print(f"[ERROR] Unexpected error for item {idx}, id={youtube_id}: {e}")
            skipped_count += 1
            continue


        item["panda_clip"] = {
            "clip_path": output_path,
            "clip_start_timestamp": seconds_to_timestamp(start_sec),
            "clip_end_timestamp": seconds_to_timestamp(event_end_sec),
            "clip_start_seconds": start_sec,
            "clip_end_seconds": event_end_sec,
        }



    if limit is not None and limit > 0:

        with open(json_path, "r") as f:
            full_data = json.load(f)

        for i in range(min(limit, len(full_data))):
            if i < len(data):
                full_data[i] = data[i]

        with open(json_path, "w") as f:
            json.dump(full_data, f, ensure_ascii=False, indent=2)
    else:

        with open(json_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[INFO] Processing completed:")
    print(f"  - Processed: {processed_count}")
    print(f"  - Skipped: {skipped_count}")
    print(f"  - Already had clips: {already_has_clip_count}")
    print(f"  - Total: {total_items}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate panda clips based on stage3_panda_clips.json using splitting pipeline."
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=DEFAULT_PANDA_INPUT_JSON,
        help="Path to stage3_panda_clips.json",
    )
    parser.add_argument(
        "--video-list",
        type=str,
        default=HIGHRES_VIDEO_LIST,
        help="Video list path; set --video-list or HIGHRES_VIDEO_LIST",
    )
    parser.add_argument(
        "--event-timecode",
        type=str,
        default=DEFAULT_EVENT_TIMECODE,
        help="Path to event_timecode.json (output of splitting pipeline)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_PANDA_OUTPUT_DIR,
        help="Directory to save generated panda clips",
    )
    parser.add_argument(
        "--splitting-dir",
        type=str,
        default=SPLITTING_DIR,
        help="Directory containing splitting scripts (cutscene_detect.py, event_stitching.py)",
    )
    parser.add_argument(
        "--run-splitting",
        action="store_true",
        help="Force run splitting pipeline even if event_timecode.json exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of items to process (for testing). If not specified, process all items.",
    )
    args = parser.parse_args()

    if not args.video_list:
        parser.error("set --video-list or HIGHRES_VIDEO_LIST")

    process_stage3_panda_clips(
        json_path=args.json_path,
        video_list_path=args.video_list,
        event_timecode_path=args.event_timecode,
        output_clip_dir=args.output_dir,
        splitting_dir=args.splitting_dir,
        run_splitting=args.run_splitting,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
