from __future__ import annotations

import pathlib
import tomllib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PackagingConfigTest(unittest.TestCase):
    def test_skill_markdown_files_are_declared_for_build_artifacts(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
        data_files = pyproject["tool"]["setuptools"]["data-files"]

        self.assertIn("SKILL.md", data_files["share/arrowspace-skills"])
        self.assertIn("HYPERPARAMETERS.md", data_files["share/arrowspace-skills"])
        self.assertEqual(data_files["share/arrowspace-skills/skills"], ["skills/*.md"])

        manifest = (ROOT / "MANIFEST.in").read_text()
        self.assertIn("include SKILL.md HYPERPARAMETERS.md", manifest)
        self.assertIn("recursive-include skills *.md", manifest)


if __name__ == "__main__":
    unittest.main()
