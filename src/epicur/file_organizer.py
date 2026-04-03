from __future__ import annotations

import datetime
import logging
import re
import shutil
from pathlib import Path

from .models import EpisodeMatch, OrganizationResult, TVH_SUFFIX_RE

logger = logging.getLogger(__name__)


def _base_stem(source_file: Path) -> str:
    """Return the file stem with the TVH duplicate suffix (-N) stripped."""
    name = source_file.name
    m = TVH_SUFFIX_RE.match(name)
    if m:
        return m.group(1)
    return source_file.stem


def compute_target_path(series_dir: Path, match: EpisodeMatch, source_file: Path) -> Path:
    """Compute the target path for a matched episode.

    Pattern: <series_dir>/Season <N>/<basename> S<NN>E<NN>.<ext>
    The basename is the original filename without the TVH -N suffix.
    """
    ext = source_file.suffix
    stem = _base_stem(source_file)
    season_folder = f"Season {match.season_number}"
    filename = f"{stem} S{match.season_number:02d}E{match.episode_number:02d}{ext}"
    target = series_dir / season_folder / filename
    if not target.resolve().is_relative_to(series_dir.resolve()):
        raise ValueError(f"Path traversal detected: {target} escapes {series_dir}")
    return target


def compute_duplicate_path(series_dir: Path, match: EpisodeMatch, source_file: Path) -> Path:
    """Compute the path for a duplicate file.

    Pattern: <series_dir>/duplicates/<YYYYMMDD_HHMMSS>_<basename> S<NN>E<NN>.<ext>
    The timestamp is taken from the source file's modification time.
    """
    ext = source_file.suffix
    stem = _base_stem(source_file)
    mtime = datetime.datetime.fromtimestamp(source_file.stat().st_mtime)
    timestamp = mtime.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{stem} S{match.season_number:02d}E{match.episode_number:02d}{ext}"
    target = series_dir / "duplicates" / filename
    if not target.resolve().is_relative_to(series_dir.resolve()):
        raise ValueError(f"Path traversal detected: {target} escapes {series_dir}")
    return target


def organize_file(
    series_dir: Path,
    source_file: Path,
    match: EpisodeMatch | None,
    *,
    dry_run: bool = False,
) -> OrganizationResult:
    """Move or rename a recording file based on its episode match.

    - If *match* is None the file is left untouched (action="skipped").
    - If the target path already exists the file is treated as a duplicate.
    - In dry-run mode no files are moved; the intended action is still returned.
    """
    # Unmatched → leave in place
    if match is None:
        logger.info("Skipping unmatched file: %s", source_file.name)
        return OrganizationResult(
            source_path=source_file,
            target_path=None,
            action="skipped",
        )

    target = compute_target_path(series_dir, match, source_file)

    # Check for duplicate
    is_duplicate = target.exists()
    if is_duplicate:
        target = compute_duplicate_path(series_dir, match, source_file)
        action = "duplicate"
        logger.info("Duplicate detected – will move to: %s", target)
    else:
        action = "moved"

    # Skip the actual move if source == target
    if source_file.resolve() == target.resolve():
        logger.info("File already at target location: %s", target)
        return OrganizationResult(
            source_path=source_file,
            target_path=target,
            action="skipped",
            episode_match=match,
        )

    if dry_run:
        logger.info("[DRY RUN] Would %s: %s -> %s", action, source_file, target)
        return OrganizationResult(
            source_path=source_file,
            target_path=target,
            action=action,
            episode_match=match,
        )

    # Execute the move
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_file), str(target))
        logger.info("Moved: %s -> %s", source_file, target)
        # Handle .meta sidecar file
        meta_file = source_file.with_suffix(".meta")
        if meta_file.is_file():
            if action == "moved":
                meta_file.unlink()
                logger.debug("Removed meta file: %s", meta_file)
            elif action == "duplicate":
                meta_target = target.with_suffix(".meta")
                meta_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(meta_file), str(meta_target))
                logger.debug("Moved meta file to duplicates: %s", meta_target)
        return OrganizationResult(
            source_path=source_file,
            target_path=target,
            action=action,
            episode_match=match,
        )
    except OSError as exc:
        logger.error("Failed to move %s -> %s: %s", source_file, target, exc)
        return OrganizationResult(
            source_path=source_file,
            target_path=target,
            action="error",
            episode_match=match,
            error_message=str(exc),
        )
