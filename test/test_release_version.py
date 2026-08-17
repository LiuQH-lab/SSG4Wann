import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import runpy


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = REPOSITORY_ROOT / "scripts" / "check_release_version.py"
VERSION_MODULE = REPOSITORY_ROOT / "src" / "ssg4wann" / "version.py"


class RuntimeVersionTests(unittest.TestCase):
    @patch("importlib.metadata.version", return_value="9.8.7")
    def test_runtime_version_comes_from_distribution_metadata(self, metadata_version):
        namespace = runpy.run_path(VERSION_MODULE)

        metadata_version.assert_called_once_with("ssg4wann")
        self.assertEqual(namespace["__version__"], "9.8.7")


class ReleaseVersionCheckTests(unittest.TestCase):
    def run_check(
        self, project_version: str, tag: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "pyproject.toml").write_text(
                f'[project]\nversion = "{project_version}"\n', encoding="utf-8"
            )

            command = [sys.executable, str(CHECK_SCRIPT), "--root", str(root)]
            if tag is not None:
                command.extend(["--tag", tag])
            return subprocess.run(command, capture_output=True, text=True, check=False)

    def test_matching_versions_and_tag_pass(self):
        result = self.run_check("1.0.1", "v1.0.1")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Version check passed", result.stdout)

    def test_tag_version_mismatch_fails(self):
        result = self.run_check("1.0.1", "v1.0.0")

        self.assertEqual(result.returncode, 1)
        self.assertIn("version mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
