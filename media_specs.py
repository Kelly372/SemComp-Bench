








from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from typing import Literal, Tuple

from PIL import Image

UploadTier = Literal["low", "mid", "high"]

UPLOAD_TIERS: tuple[str, ...] = ("low", "mid", "high")

ASPECT_W = 16
ASPECT_H = 9


GRID_CELL_MAX_EDGE = 256


DISK_FRAME_MAX_EDGE = 1280


CLIP_SAVE_WIDTH = 1280
CLIP_SAVE_HEIGHT = 720

UPLOAD_FRAME_SIZE: dict[str, Tuple[int, int]] = {
    "low": (256, 144),
    "mid": (512, 288),
    "high": (1280, 720),
}

VIDEO_UPLOAD_FPS: dict[str, float] = {
    "low": 0.5,
    "mid": 0.5,
    "high": 1.0,
}

JPEG_QUALITY_DISK = 95

DEFAULT_UPLOAD_TIER = "mid"
TIER_VIDEO_EXTRACT = "low"


def normalize_upload_tier(tier: str) -> str:
    t = str(tier or "").strip().lower()
    if t not in UPLOAD_TIERS:
        raise ValueError(f"upload_tier 须为 {UPLOAD_TIERS}, 收到: {tier!r}")
    return t


def add_upload_tier_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str = DEFAULT_UPLOAD_TIER,
    dest: str = "upload_tier",
) -> None:
    parser.add_argument(
        "--upload-tier",
        type=str,
        choices=list(UPLOAD_TIERS),
        default=default,
        dest=dest,
        help=(
            "上传 VLM 的图片/视频尺寸: low=256×144, mid=512×288, high=1280×720 "
            f"(16:9, 默认 {default})"
        ),
    )


def frame_size(tier: str) -> Tuple[int, int]:
    t = normalize_upload_tier(tier)
    return UPLOAD_FRAME_SIZE[t]


def disk_frame_size() -> Tuple[int, int]:
    e = DISK_FRAME_MAX_EDGE
    w = max(2, e - (e % 2))
    h = max(2, (e * ASPECT_H // ASPECT_W) - ((e * ASPECT_H // ASPECT_W) % 2))
    return w, h


def clip_save_size() -> Tuple[int, int]:
    return CLIP_SAVE_WIDTH, CLIP_SAVE_HEIGHT


def resize_image_max_edge(image: Image.Image, max_edge: int) -> Image.Image:

    width, height = image.size
    max_dim = max(width, height)
    if max_dim <= 0:
        raise ValueError(f"无效图像尺寸: {width}x{height}")
    if max_dim == max_edge:
        return image.convert("RGB") if image.mode != "RGB" else image
    scale = max_edge / max_dim
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS  # type: ignore[attr-defined]
    return image.convert("RGB").resize((new_width, new_height), resample)


def fit_image_to_box(image: Image.Image, width: int, height: int) -> Image.Image:

    tw, th = width, height
    im = image.convert("RGB")
    w, h = im.size
    if w <= 0 or h <= 0:
        raise ValueError(f"无效图像尺寸: {w}x{h}")
    scale = min(tw / w, th / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS  # type: ignore[attr-defined]
    resized = im.resize((nw, nh), resample)
    canvas = Image.new("RGB", (tw, th), (0, 0, 0))
    canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return canvas


def fit_image_to_frame(image: Image.Image, tier: str) -> Image.Image:
    tw, th = frame_size(tier)
    return fit_image_to_box(image, tw, th)


def fit_image_to_disk_frame(image: Image.Image) -> Image.Image:
    w, h = disk_frame_size()
    return fit_image_to_box(image, w, h)


def save_disk_frame_jpg(path: str, image: Image.Image) -> None:

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fit_image_to_disk_frame(image).save(path, quality=JPEG_QUALITY_DISK)


def scale_video_for_upload(
    src_path: str,
    dst_path: str,
    width: int,
    height: int,
) -> None:
    out_dir = os.path.dirname(dst_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        src_path,
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        dst_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "ffmpeg scale 失败").strip()
        raise RuntimeError(msg)


def upload_video_file(
    client,
    video_path: str,
    tier: str,
    *,
    fps: float | None = None,
) -> str:
    t = normalize_upload_tier(tier)
    if fps is None:
        fps = VIDEO_UPLOAD_FPS[t]
    w, h = frame_size(t)
    fd, scaled_path = tempfile.mkstemp(suffix=".mp4", prefix=f"upload_{t}_")
    os.close(fd)
    try:
        scale_video_for_upload(video_path, scaled_path, w, h)
        with open(scaled_path, "rb") as f:
            file = client.files.create(
                file=f,
                purpose="user_data",
                preprocess_configs={"video": {"fps": fps}},
            )
        client.files.wait_for_processing(file.id)
        return str(file.id)
    finally:
        try:
            if os.path.isfile(scaled_path):
                os.remove(scaled_path)
        except OSError:
            pass
