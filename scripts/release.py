"""
Make a release: set the version and date, commit, tag and push; CI does the rest.

    uv run python scripts/release.py 0.5.0           # asks before pushing
    uv run python scripts/release.py 0.5.0 --dry-run # show the changes only

Run on an up-to-date, clean main. The script sets the version in pyproject.toml and
rust/Cargo.toml, dates CHANGELOG.md's heading for the version
("## [x.y.z] - unreleased" -> "## [x.y.z] - YYYY-MM-DD") and CITATION.cff, commits
"Release vx.y.z", tags vx.y.z and pushes both. The pushed tag starts the release in
GitHub Actions (tests, wheels, GitHub release, PyPI after approval).
"""

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def replace(path: Path, pattern: str, new: str, what: str) -> None:
    text = path.read_text()
    updated, count = re.subn(pattern, new, text, count=1, flags=re.MULTILINE)
    if count != 1:
        sys.exit(f"{path.name}: {what} not found")
    path.write_text(updated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("version", help="the new version, e.g. 0.5.0")
    parser.add_argument("--dry-run", action="store_true", help="show changes only")
    parser.add_argument("--yes", action="store_true", help="do not ask before pushing")
    args = parser.parse_args()
    version, tag = args.version, f"v{args.version}"
    if not re.fullmatch(r"\d+\.\d+\.\d+([.-]?[a-z0-9]+)?", version):
        sys.exit(f"not a version: {version}")

    # Releases come from an up-to-date, clean main, and each version only once.
    if git("rev-parse", "--abbrev-ref", "HEAD") != "main":
        sys.exit("check out main first")
    if git("status", "--porcelain"):
        sys.exit("the working tree has uncommitted changes")
    git("fetch", "origin", "main", "--tags")
    if git("rev-parse", "HEAD") != git("rev-parse", "origin/main"):
        sys.exit("main differs from origin/main: pull (or push) first")
    if git("tag", "--list", tag):
        sys.exit(f"tag {tag} already exists")

    today = datetime.date.today().isoformat()
    escaped = re.escape(version)
    replace(
        ROOT / "CHANGELOG.md",
        rf"^## \[{escaped}\].*$",
        f"## [{version}] - {today}",
        f"a heading '## [{version}] - unreleased'",
    )
    replace(
        ROOT / "pyproject.toml",
        r'^version = ".*"$',
        f'version = "{version}"',
        "the version",
    )
    replace(
        ROOT / "rust" / "Cargo.toml",
        r'^version = ".*"$',
        f'version = "{version}"',
        "the version",
    )
    replace(ROOT / "CITATION.cff", r"^version: .*$", f"version: {version}", "version")
    replace(
        ROOT / "CITATION.cff",
        r"^date-released: .*$",
        f"date-released: {today}",
        "date-released",
    )
    print(git("diff", "--stat"))
    print(git("diff"))

    if args.dry_run:
        git("checkout", "--", ".")
        print("\nDry run: changes undone.")
        return
    if not args.yes and input(f"\nCommit, tag and push {tag}? [y/N] ") != "y":
        git("checkout", "--", ".")
        sys.exit("Cancelled: changes undone.")
    git("commit", "-am", f"Release {tag}")
    git("tag", "-a", tag, "-m", f"Release {tag}")
    git("push", "--atomic", "origin", "main", f"refs/tags/{tag}")
    print(f"\nPushed {tag}. GitHub Actions now tests, builds and creates the release;")
    print("approve the PyPI upload under 'Review deployments' in the run.")


if __name__ == "__main__":
    main()
