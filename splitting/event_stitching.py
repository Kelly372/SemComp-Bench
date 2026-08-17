import torch
from torchvision import transforms
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor
import os, sys, re, json, argparse
from typing import List, Optional, Tuple
from datetime import timedelta
from tqdm import tqdm
import numpy as np

_SPLITTING_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGEBIND_DIR = os.path.join(_SPLITTING_DIR, "ImageBind")
if _IMAGEBIND_DIR not in sys.path:
    sys.path.insert(0, _IMAGEBIND_DIR)
from models import imagebind_model
from models.imagebind_model import ModalityType


def resolve_torch_device(device: Optional[str] = None) -> str:

    if device:
        d = device.strip().lower()
        if d.startswith("cuda") and not torch.cuda.is_available():
            print("[WARN] 指定了 CUDA 但不可用, 回退到 cpu")
            return "cpu"
        if d == "cpu":
            return "cpu"
        if d.startswith("cuda"):
            try:
                torch.zeros(1, device=d)
                return d
            except RuntimeError as e:
                print(f"[WARN] CUDA 自检失败 ({e}), 回退到 cpu")
                return "cpu"
        return device

    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda")
        return "cuda"
    except RuntimeError as e:
        print(f"[WARN] CUDA 自检失败 ({e}), 回退到 cpu")
        return "cpu"


def _read_videoframes_parallel(
    frame_idx: List[Tuple[str, int]], max_workers: int = 8
) -> List[Tuple[np.ndarray, bool]]:

    if not frame_idx:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(lambda args: read_videoframe(*args), frame_idx))


def read_videoframe(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        res, frame = cap.read()
        if res:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        else:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        return frame, res
    finally:
        cap.release()


def transfer_timecode(frameidx, fps):
    if not fps or fps <= 0:
        raise ValueError(f"无效视频帧率: {fps}")
    timecode = []
    for (start_idx, end_idx) in frameidx:
        s = str(timedelta(seconds=start_idx/fps, microseconds=1))[:-3]
        e = str(timedelta(seconds=end_idx/fps, microseconds=1))[:-3]
        timecode.append([s, e])
    return timecode


def load_imagebind_model(device: Optional[str] = None):

    device_str = resolve_torch_device(device)
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device_str)
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    return model, image_transform, device_str


def extract_cutscene_feature(
    video_path,
    cutscenes,
    model,
    device,
    image_transform,
    *,
    frame_read_workers: int = 8,
    inference_batch_size: int = 128,
):
    features = torch.empty((0, 1024))
    res: List[bool] = []
    use_cuda = str(device).startswith("cuda")
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_cuda
        else torch.autocast(device_type="cpu", enabled=False)
    )

    for i in range(0, len(cutscenes), inference_batch_size):
        cutscenes_sub = cutscenes[i : i + inference_batch_size]
        frame_idx = []
        for cutscene in cutscenes_sub:
            start_frame_idx = int(0.95 * cutscene[0] + 0.05 * (cutscene[1] - 1))
            end_frame_idx = int(0.05 * cutscene[0] + 0.95 * (cutscene[1] - 1))
            frame_idx.extend([(video_path, start_frame_idx), (video_path, end_frame_idx)])

        results = _read_videoframes_parallel(
            frame_idx, max_workers=frame_read_workers
        )
        frames = [image_transform(Image.fromarray(i[0])) for i in results]
        res.extend([i[1] for i in results])

        frames_t = torch.stack(frames, dim=0)
        with torch.inference_mode(), autocast_ctx:
            batch_features = model(
                {ModalityType.VISION: frames_t.to(device, non_blocking=use_cuda)}
            )[ModalityType.VISION]
        features = torch.vstack((features, batch_features.detach().cpu()))

    return features, res


def verify_cutscene(cutscenes, cutscene_feature, cutscene_status, transition_threshold=0.8):
    cutscenes_new = []
    cutscene_feature_new = []
    for i, cutscene in enumerate(cutscenes):
        start_frame_fet, end_frame_fet = cutscene_feature[2*i], cutscene_feature[2*i+1]
        start_frame_res, end_frame_res = cutscene_status[2*i], cutscene_status[2*i+1]
        diff = (start_frame_fet - end_frame_fet).pow(2).sum().sqrt()

        # Remove condition 1: start_frame or end_frame cannot be loaded
        if not (start_frame_res and end_frame_res):
            continue
        # Remove condition 2: cutscene include scene transition effect
        if diff > transition_threshold:
            continue

        cutscenes_new.append(cutscene)
        cutscene_feature_new.append([start_frame_fet, end_frame_fet])
    return cutscenes_new, cutscene_feature_new


def cutscene_stitching(cutscenes, cutscene_feature, eventcut_threshold=0.6):
    assert len(cutscenes) == len(cutscene_feature)
    num_cutscenes = len(cutscenes)
    if num_cutscenes == 0:
        return [], []

    events = []
    event_feature = []
    for i in range(num_cutscenes):
        # The first cutscene or the cutscene is discontinuous from the previous one => start a new event
        if i == 0 or cutscenes[i][0] != events[-1][-1]:
            events.append(cutscenes[i])
            event_feature.append(cutscene_feature[i])
            continue

        diff = (event_feature[-1][-1] - cutscene_feature[i][0]).pow(2).sum().sqrt()
        # The difference between the cutscene and the previous one is large => start a new event
        if diff > eventcut_threshold:
            events.append(cutscenes[i])
            event_feature.append(cutscene_feature[i])
        # Otherwise => extend the previous event
        else:
            events[-1].extend(cutscenes[i])
            event_feature[-1].extend(cutscene_feature[i])

    if len(events[-1]) == 0:
        events.pop(-1)
        event_feature.pop(-1)

    return events, event_feature


def verify_event(events, event_feature, fps, min_event_len=1.5, max_event_len=60, redundant_event_threshold=0.4, trim_begin_last_percent=0.1, still_event_threshold=0.1): # add remove no change event
    assert len(events) == len(event_feature)
    num_events = len(events)
    
    events_final = []
    event_feature_final = torch.empty((0,1024))
    
    min_event_len = min_event_len*fps
    max_event_len = max_event_len*fps

    for i in range(num_events):
        assert len(events[i]) == len(event_feature[i])
        # Remove condition 1: shorter than min_event_len
        if (events[i][-1] - events[i][0]) < min_event_len:
            continue
            
        # Remove condition 2: within-event difference is too small
        diff = (event_feature[i][0] - event_feature[i][-1]).pow(2).sum().sqrt()
        if diff < still_event_threshold:
            continue
        
        avg_feature = torch.stack(event_feature[i]).mean(axis=0)
        # Remove condition 3: too similar to the previous events
        diff = (event_feature_final - avg_feature).pow(2).sum(axis=1).sqrt()
        if torch.any(diff < redundant_event_threshold):
            continue

        # Trim the event if it is too long
        events[i][-1] = events[i][0] + min(int(max_event_len), (events[i][-1]-events[i][0]))
        
        trim_len = int(trim_begin_last_percent*(events[i][-1]-events[i][0]))
        events_final.append([events[i][0]+trim_len, events[i][-1]-trim_len])
        event_feature_final = torch.vstack((event_feature_final, avg_feature))
        
    return events_final, event_feature_final


def write_json_file(data, output_file):
    data = json.dumps(data, indent = 4)
    def repl_func(match: re.Match):
        return " ".join(match.group().split())
    data = re.sub(r"(?<=\[)[^\[\]]+(?=])", repl_func, data)
    data = re.sub(r'\[\s+', '[', data)
    data = re.sub(r'],\s+\[', '], [', data)
    data = re.sub(r'\s+\]', ']', data)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(data)


def run_stitching(
    video_list_path: str,
    cutscene_frameidx_path: str,
    output_json_file: str,
    *,
    device: Optional[str] = None,
    frame_read_workers: int = 8,
    model=None,
    image_transform=None,
    device_str: Optional[str] = None,
) -> None:






    owns_model = model is None
    if owns_model:
        model, image_transform, device_str = load_imagebind_model(device)
    else:
        device_str = device_str or resolve_torch_device(device)

    print(f"[INFO] event_stitching device={device_str}")

    with open(video_list_path, "r", encoding="utf-8") as f:
        video_paths = [ln.strip() for ln in f if ln.strip()]
    with open(cutscene_frameidx_path, encoding="utf-8") as f:
        video_cutscenes = json.load(f)

    video_events = {}
    try:
        for video_path in tqdm(video_paths, desc="event_stitching"):
            video_key = os.path.basename(video_path)
            cutscene = video_cutscenes.get(video_key)
            if not cutscene:
                print(f"[WARN] 无 cutscene: {video_key}, 跳过")
                continue

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if not fps or fps <= 0:
                print(f"[WARN] 无法读取有效帧率: {video_key}, 记录为空事件")
                video_events[video_key] = []
                continue

            cutscene_raw_feature, cutscene_raw_status = extract_cutscene_feature(
                video_path,
                cutscene,
                model,
                device_str,
                image_transform,
                frame_read_workers=frame_read_workers,
            )
            cutscenes, cutscene_feature = verify_cutscene(
                cutscene,
                cutscene_raw_feature,
                cutscene_raw_status,
                transition_threshold=1.0,
            )
            events_raw, event_feature_raw = cutscene_stitching(
                cutscenes, cutscene_feature, eventcut_threshold=0.6
            )
            events, _event_feature = verify_event(
                events_raw,
                event_feature_raw,
                fps,
                min_event_len=2.0,
                max_event_len=1200,
                redundant_event_threshold=0.0,
                trim_begin_last_percent=0.0,
                still_event_threshold=0.15,
            )
            video_events[video_key] = transfer_timecode(events, fps)

            if str(device_str).startswith("cuda"):
                torch.cuda.synchronize()
    finally:
        if owns_model:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_json_file(video_events, output_json_file)


def run_panda_splitting_pipeline(
    video_list_path: str,
    cutscene_frameidx_path: str,
    event_timecode_path: str,
    *,
    device: Optional[str] = None,
    frame_read_workers: int = 8,
    rerun_cutscene: bool = False,
) -> None:









    video_list_path = os.path.abspath(video_list_path)
    cutscene_frameidx_path = os.path.abspath(cutscene_frameidx_path)
    event_timecode_path = os.path.abspath(event_timecode_path)

    need_cutscene = rerun_cutscene or not os.path.isfile(cutscene_frameidx_path)
    if need_cutscene:
        print("[INFO] Phase 1/2: cutscene_detect (CPU)...")
        from cutscene_detect import run_cutscene_detection

        run_cutscene_detection(video_list_path, cutscene_frameidx_path)
    else:
        print(f"[INFO] 复用 cutscene: {cutscene_frameidx_path}")

    print("[INFO] Phase 2/2: 加载 ImageBind (一次) + event_stitching...")
    model, image_transform, device_str = load_imagebind_model(device)
    try:
        run_stitching(
            video_list_path,
            cutscene_frameidx_path,
            event_timecode_path,
            device=device_str,
            frame_read_workers=frame_read_workers,
            model=model,
            image_transform=image_transform,
            device_str=device_str,
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[INFO] panda splitting 完成 -> {event_timecode_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event Stitching / Panda Pipeline")
    parser.add_argument("--video-list", type=str, required=True)
    parser.add_argument("--cutscene-frameidx", type=str, default="cutscene_frame_idx.json")
    parser.add_argument("--output-json-file", type=str, default="event_timecode.json")
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="运行 cutscene_detect + event_stitching (CUDA 稳定单进程)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="cuda / cpu / cuda:0; 默认自动选择",
    )
    parser.add_argument("--frame-read-workers", type=int, default=8)
    parser.add_argument("--rerun-cutscene", action="store_true")
    args = parser.parse_args()

    if args.full_pipeline:
        run_panda_splitting_pipeline(
            args.video_list,
            args.cutscene_frameidx,
            args.output_json_file,
            device=args.device or None,
            frame_read_workers=args.frame_read_workers,
            rerun_cutscene=args.rerun_cutscene,
        )
    else:
        run_stitching(
            args.video_list,
            args.cutscene_frameidx,
            args.output_json_file,
            device=args.device or None,
            frame_read_workers=args.frame_read_workers,
        )
