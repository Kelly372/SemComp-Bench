#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HIGHRES_VIDEO_DIR:?Set HIGHRES_VIDEO_DIR to <HIGHRES_VIDEO_DIR>}"
VIDEO_LIST_PATH="${VIDEO_LIST_PATH:-${SCRIPT_DIR}/video_list.txt}"
CUTSCENE_INDEX_PATH="${CUTSCENE_INDEX_PATH:-${SCRIPT_DIR}/cutscene_frame_idx.json}"
EVENT_TIMECODE_PATH="${EVENT_TIMECODE_PATH:-${SCRIPT_DIR}/event_timecode.json}"

echo "Step 0: 生成完整 video_list.txt ..."
find "${HIGHRES_VIDEO_DIR}" -maxdepth 1 -type f -iname '*.mp4' | sort > "${VIDEO_LIST_PATH}"

echo "Step 1-2: 镜头切分与事件合并 ..."
python "${SCRIPT_DIR}/event_stitching.py" \
  --full-pipeline \
  --video-list "${VIDEO_LIST_PATH}" \
  --cutscene-frameidx "${CUTSCENE_INDEX_PATH}" \
  --output-json-file "${EVENT_TIMECODE_PATH}" \
  --rerun-cutscene

echo "全部完成，已生成 ${CUTSCENE_INDEX_PATH} 与 ${EVENT_TIMECODE_PATH}。"

