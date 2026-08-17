from __future__ import annotations

import ast
import io
import pathlib
import re
import tokenize
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


class RepositoryTests(unittest.TestCase):
    def test_all_python_files_parse(self) -> None:
        failures: list[str] = []
        for path in sorted(ROOT.rglob("*.py")):
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                failures.append(f"{path.relative_to(ROOT)}: {exc}")
        self.assertEqual(failures, [])

    def test_no_cjk_comments_or_docstrings(self) -> None:
        failures: list[str] = []
        for path in sorted(ROOT.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type == tokenize.COMMENT and CJK.search(token.string):
                    failures.append(f"{path.relative_to(ROOT)}:{token.start[0]}")
            tree = ast.parse(text, filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(
                    node,
                    (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                ) or not node.body:
                    continue
                first = node.body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                    and CJK.search(first.value.value)
                ):
                    failures.append(f"{path.relative_to(ROOT)}:{first.lineno}")

        for pattern in ("*.sh", "*.yml", "*.yaml"):
            for path in sorted(ROOT.rglob(pattern)):
                for line_number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    marker = line.find("#")
                    if marker >= 0 and CJK.search(line[marker:]):
                        failures.append(f"{path.relative_to(ROOT)}:{line_number}")
        self.assertEqual(failures, [])

    def test_pipeline_stages_have_entry_points(self) -> None:
        for stage in range(1, 10):
            matches = list(ROOT.glob(f"{stage}_*.py"))
            self.assertEqual(len(matches), 1, f"expected exactly one Stage {stage} script")
            text = matches[0].read_text(encoding="utf-8")
            self.assertIn('if __name__ == "__main__":', text)

    def test_required_prompt_files_are_present(self) -> None:
        expected = {
            "1_titleFilter.json",
            "2_classify.txt",
            "3_extract.txt",
            "4_check.txt",
            "5_instruction.txt",
            "7_instructionNorm.txt",
            "8_align_type.txt",
            "9_result_state_instruction.txt",
            "category.json",
        }
        present = {path.name for path in (ROOT / "prompt").iterdir() if path.is_file()}
        self.assertTrue(expected <= present, sorted(expected - present))

    def test_no_machine_specific_absolute_paths_in_python(self) -> None:
        # The negative lookbehind avoids treating URL schemes as drive names.
        windows_path = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
        unix_home = re.compile(r"/(?:home|Users)/[^/\s]+/")
        failures: list[str] = []
        for path in sorted(ROOT.rglob("*.py")):
            if "tests" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            if windows_path.search(text) or unix_home.search(text):
                failures.append(str(path.relative_to(ROOT)))
        self.assertEqual(failures, [])

    def test_no_obvious_live_credentials(self) -> None:
        patterns = (
            re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
            re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        )
        failures: list[str] = []
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {
                ".py",
                ".json",
                ".md",
                ".txt",
                ".yml",
                ".yaml",
                ".sh",
            }:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(pattern.search(text) for pattern in patterns):
                failures.append(str(path.relative_to(ROOT)))
        self.assertEqual(failures, [])

    def test_no_committed_service_endpoint_or_model_identifier(self) -> None:
        banned_literals = (
            "ark" + ".cn-beijing",
            "dou" + "bao-seed",
        )
        failures: list[str] = []
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {
                ".py",
                ".json",
                ".md",
                ".txt",
                ".yml",
                ".yaml",
                ".sh",
            }:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").casefold()
            if any(literal.casefold() in text for literal in banned_literals):
                failures.append(str(path.relative_to(ROOT)))
        self.assertEqual(failures, [])

    def test_no_conference_or_review_metadata(self) -> None:
        # Split literals so this test does not flag its own source text.
        banned_terms = (
            "AA" + "AI",
            "20" + "27",
            "under" + " review",
            "under" + " submission",
            "anonymous" + " submission",
            "anonymized" + " submission",
            "double" + " blind",
            "camera" + " ready",
            "review" + " status",
            "submission" + " status",
            "venue" + " metadata",
        )
        failures: list[str] = []
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file() or path.name == "LICENSE":
                continue
            if path.suffix.lower() not in {
                ".py",
                ".json",
                ".md",
                ".txt",
                ".yml",
                ".yaml",
                ".sh",
            }:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            folded = text.casefold().replace("-", " ").replace("_", " ")
            if any(term.casefold() in folded for term in banned_terms):
                failures.append(str(path.relative_to(ROOT)))
        self.assertEqual(failures, [])

    def test_license_boundaries_are_documented(self) -> None:
        self.assertTrue((ROOT / "LICENSE").is_file())
        self.assertTrue((ROOT / "THIRD_PARTY_NOTICES.md").is_file())
        self.assertTrue((ROOT / "splitting" / "ImageBind" / "NOTICE.md").is_file())


if __name__ == "__main__":
    unittest.main()
