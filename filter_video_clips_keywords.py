








from __future__ import annotations

import json
import re
from pathlib import Path

from pipeline_paths import RESTATE_DIR

INPUT_PATH = Path(RESTATE_DIR) / "video_clips_short.json"
OUT_KEYWORDS = Path(RESTATE_DIR) / "video_clips_short_keywords.json"
OUT_KEEP = Path(RESTATE_DIR) / "video_clips_short_keep.json"


KEYWORDS_SUBSTRING = (
    "first impression",
    "review",
    "introduce",
    "explain",
    "optimize",
    "refine",
    "improve",
    "enhance",
    "demonstrate",
    "experience",
    "evaluate",
    "compare",
    "performance",
    "discuss",
    "showcase"
)


_TEST_RE = re.compile(r"\btest\b", re.IGNORECASE)


def instruction_matches(text: str | None) -> bool:
    if not text or not isinstance(text, str):
        return False
    lowered = text.lower()
    if any(kw.lower() in lowered for kw in KEYWORDS_SUBSTRING):
        return True
    return bool(_TEST_RE.search(text))


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"期望 JSON 数组，得到 {type(data).__name__}")

    matched: list[dict] = []
    kept: list[dict] = []

    for item in data:
        if not isinstance(item, dict):
            kept.append(item)
            continue
        en = item.get("image_level_instruction_en")
        if instruction_matches(en):
            matched.append(item)
        else:
            kept.append(item)

    with OUT_KEYWORDS.open("w", encoding="utf-8") as f:
        json.dump(matched, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with OUT_KEEP.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"总计: {len(data)}")
    print(f"含关键词 -> {OUT_KEYWORDS.name}: {len(matched)}")
    print(f"保留 -> {OUT_KEEP.name}: {len(kept)}")


if __name__ == "__main__":
    main()
