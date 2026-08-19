# SemComp-Data Construction Pipeline

This repository provides the code and prompts used to construct
**SemComp-Data** for *SemComp-Bench: Benchmarking Semantic Task Completion in
Video Generation*.

The pipeline starts from video metadata and locally available source videos.
It filters titles, classifies tasks, grounds reference and outcome states,
checks visual consistency, generates and normalizes instructions, extracts
outcome-centric clips, labels alignment types, and describes result states.

Source videos, generated datasets, model weights, service credentials, and
runtime outputs are intentionally not included.

> [!IMPORTANT]
> The `splitting/` component includes code adapted from Panda-70M and
> ImageBind. These components have non-commercial licensing restrictions.
> Review [Third-party licenses](#third-party-licenses) before using or
> redistributing the repository.

## Pipeline overview

| Stage | Script | Purpose | Default kept output |
| --- | --- | --- | --- |
| 1 | `1_titleFilter.py` | Filter metadata by title keywords | `ReState/1_titleFilter.parquet` |
| 2 | `2_classify.py` | Classify task domain and category | `ReState/2_classify.parquet` |
| 3 | `3_extract.py` | Ground reference/outcome timestamps and extract frames | `ReState/3_extract.parquet` |
| 4 | `4_check.py` | Validate frame quality and state ordering | `ReState/4_check.parquet` |
| 5 | `5_instruction.py` | Generate detailed bilingual instructions | `ReState/5_instruction.parquet` |
| 6 | `6_videoClip.py` | Extract outcome-centric video clips | `ReState/6_videoClip_<mode>.parquet` |
| 7 | `7_instructionNorm.py` | Normalize instructions for the selected clip mode | `ReState/7_instructionNorm_<mode>.parquet` |
| 8 | `8_align_type.py` | Annotate semantic alignment types | `ReState/8_align_type.parquet` |
| 9 | `9_result_state_instruction.py` | Describe result states and preservation guidance | `ReState/9_result_state_instruction.parquet` |

Stages 1–7 usually also write a full snapshot (`*_output*.parquet`) and an
excluded set (`*_exclude*.parquet`). Stages that distinguish technical
failures may additionally write `*_error.parquet`.

## Repository layout

```text
.
|-- 1_titleFilter.py
|-- 2_classify.py
|-- 3_extract.py
|-- 4_check.py
|-- 5_instruction.py
|-- 6_videoClip.py
|-- 7_instructionNorm.py
|-- 8_align_type.py
|-- 9_result_state_instruction.py
|-- prompt/                       # stage prompts and task taxonomy
|-- splitting/                    # shot/event splitting used by Stage 6
|-- tests/                        # offline regression tests
|-- agent_process.py              # shared VLM and media helpers
|-- media_specs.py                # upload and frame specifications
|-- panda_clips.py                # Stage 6 orchestration helpers
|-- pipeline_paths.py             # runtime path configuration
|-- requirements.txt
|-- THIRD_PARTY_NOTICES.md
`-- LICENSE
```

## Requirements

- Python 3.10 or newer
- `ffmpeg` and `ffprobe` available on `PATH`
- Access to a configured multimodal model service for Stages 2–5 and 7–9
- A CUDA-capable environment is recommended for Stage 6 ImageBind inference

Create and activate an isolated Conda environment:

```bash
conda create -n semcomp python=3.10 -y
conda activate semcomp
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch installation depends on the operating system and accelerator. If the
default packages do not match the local CUDA runtime, install the appropriate
PyTorch build first, then install `requirements.txt`.

Verify the media tools before running the pipeline:

```bash
ffmpeg -version
ffprobe -version
```

## Input data

Stage 1 accepts either one Parquet file or a directory containing Parquet
files. The initial metadata must contain:

| Column | Requirement | Description |
| --- | --- | --- |
| `title` | required | Source-video title used by the early stages |
| `youtube_url` or `url` | required from Stage 2 | URL containing an 11-character video identifier |

The upstream videos are not downloaded by this repository. Low- and
high-resolution video directories must be prepared separately. Video file
names should begin with the identifier extracted from the metadata URL:

```text
<video-id>_<optional-name>.mp4
```

You are responsible for obtaining and processing media in accordance with its
license, terms, privacy requirements, and applicable law.

## Runtime configuration

Keep all credentials and machine-specific paths outside version control. The
supported environment-variable names are listed in `.env.example`, with empty
values by design. The code does not load `.env.example` automatically.

| Variable | Purpose | Default |
| --- | --- | --- |
| `VLM_API_KEY` | Service credential for VLM stages | empty; required |
| `VLM_BASE_URL` | Service endpoint for VLM stages | empty; required |
| `VLM_MODEL_NAME` | Model identifier for VLM stages | empty; required |
| `DATASET_DIR` | Dataset workspace used by Stage 9 | local `ReState/` |
| `LOWRES_VIDEO_DIR` | Low-resolution source videos | empty |
| `HIGHRES_VIDEO_DIR` | High-resolution source videos | empty |
| `HIGHRES_VIDEO_LIST` | Optional text file containing one source path per line | empty |
| `SPLITTING_DIR` | Stage 6 splitting implementation | local `splitting/` |
| `IMAGEBIND_CHECKPOINT_PATH` | Optional local ImageBind checkpoint | ignored local checkpoint location |
| `CJK_FONT_PATH` | Optional font path or font name | platform lookup |

The VLM settings can also be supplied with stage-specific command-line flags
where available. Never commit real values to configuration files, scripts,
notebooks, examples, logs, or issue reports.

## Quick start

Run commands from the repository root. Replace the uppercase relative
placeholders with local inputs. The defaults connect each stage through the
ignored `ReState/` directory.

```bash
# 1. Metadata filtering
python 1_titleFilter.py --input INPUT_METADATA.parquet

# 2. Task classification
python 2_classify.py

# 3. Timestamp grounding and frame extraction
python 3_extract.py

# 4. Visual consistency checks
python 4_check.py --seed 0

# 5. Detailed instruction generation
python 5_instruction.py

# 6. Clip extraction; select either short or long
python 6_videoClip.py --mode short --highres_video_path VIDEO_DIRECTORY

# 7. Instruction normalization; use the same mode as Stage 6
python 7_instructionNorm.py --mode short

# 8. Alignment-type annotation (defaults to the short-mode Stage 7 output)
python 8_align_type.py

# 9. Result-state descriptions
python 9_result_state_instruction.py
```

Use `python <script>.py --help` to inspect all options. Most stages support
`--limit` for a small smoke run before processing the complete dataset.

### Stage 6 splitting and clip modes

Stage 6 supports two duration policies:

- `short`: clips between 3 and 4 seconds
- `long`: clips between 3 and 10 seconds

For each mode it writes:

```text
ReState/6_videoClip_output_<mode>.parquet
ReState/6_videoClip_<mode>.parquet
ReState/6_videoClip_exclude_<mode>.parquet
ReState/video_clip/<mode>/
```

Stage 6 reuses a complete `event_timecode.json` when available and otherwise
runs shot/event splitting. Use `--run_splitting` to force rebuilding or
`--skip_splitting` to require an existing index. The first ImageBind run may
need to obtain a checkpoint unless `IMAGEBIND_CHECKPOINT_PATH` identifies a
local copy.

Stage 7 must use the same `--mode` and produces the analogous
`7_instructionNorm_*_<mode>.parquet` files. Stage 8 defaults to
`7_instructionNorm_short.parquet`; pass `--input` explicitly when continuing
from long mode.

### Resume behavior

Stages 8 and 9 reuse their existing output Parquet files as checkpoints. Pass
`--no-resume` to discard the corresponding checkpoint and process from the
beginning. Write outputs to durable local storage and keep intermediate files
until the run has been verified.

## License

Original SemComp pipeline code is provided under the Apache License 2.0; see
`LICENSE`. Files derived from third-party projects remain under their upstream
terms and are excluded from the Apache grant.

### Third-party licenses

- `splitting/cutscene_detect.py`, `splitting/event_stitching.py`,
  `splitting/video_splitting.py`, and related utilities are adapted from
  Panda-70M and are limited to non-commercial research use.
- `splitting/ImageBind/` is adapted from Meta's ImageBind and is licensed under
  CC BY-NC-SA 4.0.
- PySceneDetect is used as a dependency under the BSD 3-Clause License.

See `THIRD_PARTY_NOTICES.md` for provenance, modification notes, and upstream
notices. Because some bundled components are non-commercial, the repository as
a whole must not be described as commercially permissive.

## Contributing and security

See `CONTRIBUTING.md` for contribution guidance and `SECURITY.md` for private
vulnerability reporting guidance.
