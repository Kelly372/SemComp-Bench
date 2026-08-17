from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import panda_clips
import pipeline_paths


class PipelinePathTests(unittest.TestCase):
    def test_parquet_triplet_normalizes_suffixes(self) -> None:
        full, kept, excluded = pipeline_paths.parquet_triplet(
            os.path.join("work", "6_videoClip_output.parquet")
        )
        self.assertEqual(full, os.path.join("work", "6_videoClip_output.parquet"))
        self.assertEqual(kept, os.path.join("work", "6_videoClip.parquet"))
        self.assertEqual(
            excluded, os.path.join("work", "6_videoClip_exclude.parquet")
        )

    def test_parquet_error_path_normalizes_suffixes(self) -> None:
        path = pipeline_paths.parquet_error_path(
            os.path.join("work", "4_check_exclude.parquet")
        )
        self.assertEqual(path, os.path.join("work", "4_check_error.parquet"))


class ClipUtilityTests(unittest.TestCase):
    def test_timestamp_formats(self) -> None:
        cases = {
            "5.0": 5.0,
            "00:05.0": 5.0,
            "03:31.0": 211.0,
            "00:01:36.0": 96.0,
            "0:02:42.000": 162.0,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertAlmostEqual(
                    panda_clips.parse_timestamp_to_seconds(text), expected
                )

        self.assertEqual(panda_clips.seconds_to_timestamp(96.0), "0:01:36.000")

    def test_event_lookup_includes_boundaries(self) -> None:
        events = [(0.0, 4.0), (4.0, 8.0)]
        self.assertEqual(
            panda_clips.find_event_containing_timestamp(events, 4.0),
            (0.0, 4.0),
        )
        self.assertIsNone(panda_clips.find_event_containing_timestamp(events, 9.0))

    def test_clip_range_uses_valid_event_directly(self) -> None:
        result = panda_clips.resolve_clip_range_for_outcome(
            4.0,
            (2.0, 6.0),
            min_duration=3.0,
            max_duration=4.0,
            video_duration=10.0,
        )
        self.assertEqual(result, (2.0, 6.0, "event_fit"))

    def test_short_event_is_shifted_inside_video_bounds(self) -> None:
        result = panda_clips.resolve_clip_range_for_outcome(
            0.5,
            (0.0, 1.0),
            min_duration=3.0,
            max_duration=4.0,
            video_duration=10.0,
        )
        self.assertIsNotNone(result)
        start, end, source = result  # type: ignore[misc]
        self.assertAlmostEqual(start, 0.0)
        self.assertAlmostEqual(end, 3.0)
        self.assertEqual(source, "outcome_centered")

    def test_ffmpeg_output_may_be_a_bare_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(panda_clips.subprocess, "run") as run:
                previous = os.getcwd()
                try:
                    os.chdir(temp_dir)
                    panda_clips.run_ffmpeg_clip("input.mp4", 1.0, 2.0, "clip.mp4")
                finally:
                    os.chdir(previous)
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
