from __future__ import annotations

import pathlib
import tomllib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PackagingConfigTest(unittest.TestCase):
    def test_skill_markdown_files_are_bundled_via_package_data(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
        pkg_data = pyproject["tool"]["setuptools"]["package-data"]

        self.assertIn("SKILL.md", pkg_data["arrowspace_skills"])
        self.assertIn("HYPERPARAMETERS.md", pkg_data["arrowspace_skills"])
        self.assertIn("skills/*.md", pkg_data["arrowspace_skills"])

    def test_bundled_files_exist_on_disk(self) -> None:
        pkg_dir = ROOT / "arrowspace_skills"
        self.assertTrue((pkg_dir / "SKILL.md").is_file())
        self.assertTrue((pkg_dir / "HYPERPARAMETERS.md").is_file())
        self.assertTrue((pkg_dir / "skills" / "arrowspace-core.md").is_file())
        self.assertTrue((pkg_dir / "skills" / "arrowspace-search.md").is_file())
        self.assertTrue((pkg_dir / "skills" / "arrowspace-spectral.md").is_file())

    def test_manifest_in_includes_package_markdown(self) -> None:
        manifest = (ROOT / "MANIFEST.in").read_text()
        self.assertIn("recursive-include arrowspace_skills *.md", manifest)


if __name__ == "__main__":
    unittest.main()
