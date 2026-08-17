from scenedetect import detect, ContentDetector
from tqdm import tqdm
import cv2
import os, re, json, argparse


def cutscene_detection(video_path, cutscene_threshold=27, max_cutscene_len=10):
    scene_list = detect(video_path, ContentDetector(threshold=cutscene_threshold, min_scene_len=15), start_in_scene=True)
    end_frame_idx = [0]

    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if not fps or fps <= 0:
        raise ValueError(f"无法读取有效帧率: {video_path}")

    for scene in scene_list:
        new_end_frame_idx = scene[1].get_frames()
        while (new_end_frame_idx-end_frame_idx[-1]) > (max_cutscene_len+2)*fps: # if no cutscene at min_scene_len+2, then cut at min_scene_len
            end_frame_idx.append(end_frame_idx[-1] + int(max_cutscene_len*fps))
        end_frame_idx.append(new_end_frame_idx)
    
    cutscenes =[]
    for i in range(len(end_frame_idx)-1):
        cutscenes.append([end_frame_idx[i], end_frame_idx[i+1]])

    return cutscenes


def run_cutscene_detection(
    video_list_path: str,
    output_json_file: str,
    *,
    cutscene_threshold: int = 25,
    max_cutscene_len: int = 5,
) -> dict:




    with open(video_list_path, "r", encoding="utf-8") as f:
        video_paths = [ln.strip() for ln in f if ln.strip()]

    video_cutscenes = {}
    for video_path in tqdm(video_paths, desc="cutscene_detect"):
        try:
            cutscenes_raw = cutscene_detection(
                video_path,
                cutscene_threshold=cutscene_threshold,
                max_cutscene_len=max_cutscene_len,
            )
        except Exception as exc:
            print(f"[WARN] 镜头检测失败: {os.path.basename(video_path)} ({exc})")
            cutscenes_raw = []
        video_cutscenes[os.path.basename(video_path)] = cutscenes_raw

    os.makedirs(os.path.dirname(os.path.abspath(output_json_file)) or ".", exist_ok=True)
    write_json_file(video_cutscenes, output_json_file)
    return video_cutscenes


def write_json_file(data, output_file):
    data = json.dumps(data, indent = 4)
    def repl_func(match: re.Match):
        return " ".join(match.group().split())
    data = re.sub(r"(?<=\[)[^\[\]]+(?=])", repl_func, data)
    data = re.sub(r'\[\s+', '[', data)
    data = re.sub(r'],\s+\[', '], [', data)
    data = re.sub(r'\s+\]', ']', data)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cutscene Detection")
    parser.add_argument("--video-list", type=str, required=True)
    parser.add_argument("--output-json-file", type=str, default="cutscene_frame_idx.json")
    args = parser.parse_args()

    run_cutscene_detection(args.video_list, args.output_json_file)
