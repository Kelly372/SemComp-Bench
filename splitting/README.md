# Stage 6 video splitting

This directory contains the shot-boundary and ImageBind event-stitching code
used by `../6_videoClip.py`. Install dependencies from the repository-level
`requirements.txt`; `ffmpeg` and `ffprobe` must also be available on `PATH`.

> **License notice:** this component is adapted from Panda-70M and ImageBind
> and is restricted to non-commercial use under their respective upstream
> terms. It is not covered by the repository's Apache License 2.0. See
> `../THIRD_PARTY_NOTICES.md` before use or redistribution.

The normal entry point is Stage 6 itself:

```bash
python 6_videoClip.py --mode short --highres_video_path <HIGHRES_VIDEO_DIR>
```

Stage 6 creates a temporary video list, maintains `cutscene_frame_idx.json` and
`event_timecode.json`, selects the event containing `outcome_timestamp`, and
then performs the final duration-constrained clip extraction. Generated index
files and downloaded ImageBind weights are ignored by version control.

For a standalone full-index run on Linux/macOS, set `HIGHRES_VIDEO_DIR` and run
`bash splitting/run_splitting.sh`. The lower-level equivalent is:

```bash
python splitting/event_stitching.py \
  --full-pipeline \
  --video-list <VIDEO_LIST_PATH> \
  --cutscene-frameidx <CUTSCENE_INDEX_PATH> \
  --output-json-file <EVENT_TIMECODE_PATH>
```

The retained `video_splitting.py` and ImageBind `demo.py`, `main.py`, and
`test.py` files are upstream reference utilities; Stage 6 does not call them.

## Acknowledgements

The splitting implementation is derived from
[Panda-70M](https://github.com/snap-research/Panda-70M),
[PySceneDetect](https://github.com/Breakthrough/PySceneDetect), and
[ImageBind](https://github.com/facebookresearch/ImageBind).
