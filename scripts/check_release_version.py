#!/usr/bin/env python3
"""Check that project metadata, built artifacts, and the release tag agree."""

from __future__ import annotations

import argparse
from email.parser import Parser
from pathlib import Path
import sys
import tarfile
import tomllib
import zipfile


class VersionCheckError(RuntimeError):
    """Raised when release version metadata is invalid or inconsistent."""


def read_project_version(root: Path) -> str:
    pyproject_path = root / "pyproject.toml"

    with pyproject_path.open("rb") as handle:
        project_version = tomllib.load(handle).get("project", {}).get("version")
    if not isinstance(project_version, str) or not project_version:
        raise VersionCheckError(
            f"{pyproject_path}: project.version must be a non-empty string"
        )

    return project_version


def version_from_tag(tag: str) -> str:
    if not tag.startswith("v") or len(tag) == 1:
        raise VersionCheckError(
            f"release tag {tag!r} must have the form 'v<project-version>'"
        )
    return tag[1:]


def check_tag(tag: str, expected_version: str) -> None:
    tag_version = version_from_tag(tag)
    if tag_version != expected_version:
        raise VersionCheckError(
            f"version mismatch: tag {tag!r} declares {tag_version!r}, "
            f"but the project declares {expected_version!r}"
        )


def check_wheel(path: Path, expected_version: str) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise VersionCheckError(
                f"{path}: expected one wheel METADATA file, found {len(metadata_names)}"
            )
        metadata = Parser().parsestr(
            archive.read(metadata_names[0]).decode("utf-8", errors="strict")
        )
        artifact_version = metadata.get("Version")
        if artifact_version != expected_version:
            raise VersionCheckError(
                f"{path}: wheel metadata has version {artifact_version!r}, "
                f"expected {expected_version!r}"
            )

        version_name = "ssg4wann/version.py"
        if version_name not in names:
            raise VersionCheckError(f"{path}: missing {version_name}")


def check_sdist(path: Path, expected_version: str) -> None:
    with tarfile.open(path, "r:gz") as archive:
        files = {member.name: member for member in archive.getmembers() if member.isfile()}
        pyproject_names = [name for name in files if name.endswith("/pyproject.toml")]
        version_names = [name for name in files if name.endswith("/src/ssg4wann/version.py")]
        if len(pyproject_names) != 1 or len(version_names) != 1:
            raise VersionCheckError(
                f"{path}: expected one pyproject.toml and one src/ssg4wann/version.py"
            )

        pyproject_handle = archive.extractfile(files[pyproject_names[0]])
        if pyproject_handle is None:
            raise VersionCheckError(f"{path}: could not read packaged version files")

        project_version = tomllib.loads(
            pyproject_handle.read().decode("utf-8", errors="strict")
        ).get("project", {}).get("version")
        if project_version != expected_version:
            raise VersionCheckError(
                f"{path}: sdist project version is {project_version!r}; "
                f"expected {expected_version!r}"
            )


def check_artifacts(paths: list[Path], expected_version: str) -> None:
    wheels = [path for path in paths if path.suffix == ".whl"]
    sdists = [path for path in paths if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1 or len(paths) != 2:
        raise VersionCheckError(
            "expected exactly one wheel and one .tar.gz source distribution; "
            f"received {[path.name for path in paths]}"
        )
    check_wheel(wheels[0], expected_version)
    check_sdist(sdists[0], expected_version)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="project root (defaults to the repository containing this script)",
    )
    parser.add_argument("--tag", help="release tag, for example v1.0.1")
    parser.add_argument(
        "--artifacts",
        nargs="+",
        type=Path,
        help="built wheel and source distribution to inspect",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        project_version = read_project_version(args.root.resolve())
        if args.tag:
            check_tag(args.tag, project_version)
        if args.artifacts:
            check_artifacts(args.artifacts, project_version)
    except (OSError, SyntaxError, ValueError, VersionCheckError) as error:
        print(f"::error title=Release version check failed::{error}", file=sys.stderr)
        return 1

    checked = [f"project={project_version}"]
    if args.tag:
        checked.append(f"tag={args.tag}")
    if args.artifacts:
        checked.append("wheel/sdist=verified")
    print("Version check passed: " + ", ".join(checked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
