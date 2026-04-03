#!/usr/bin/env python3
"""epicur (Episode Curator) -- Automatically identify and organize TV recordings.

Scans a root directory for series subfolders, extracts metadata from video
files (via ffprobe, TVHeadend EPG data, subtitles), identifies episodes
through free online databases (TVMaze, TMDB), and renames/organizes them
into season folders.

Usage:
    python main.py /home/recordings
    python main.py /home/recordings --dry-run --verbose
    python main.py /home/recordings --tmdb-api-key YOUR_KEY --confidence 0.5
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from .episode_identifier import identify_episode
from .file_organizer import organize_file
from .metadata_extractor import extract_all_metadata, write_meta_file
from .models import OrganizationResult
from .review import review_unmatched
from .tvheadend_client import (
    find_dvr_entry_for_file,
    parse_dvr_log_dir,
    fetch_dvr_entries_api,
)
from . import __version__

logger = logging.getLogger("epicur")

DEFAULT_EXTENSIONS = {".ts", ".mp4", ".mkv"}


def setup_logging(verbose: bool = False, log_file: str | None = None) -> None:
    """Configure logging with optional file output."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def list_video_files(directory: Path, extensions: set[str]) -> list[Path]:
    """List video files in *directory* (non-recursive), sorted by mtime ascending."""
    files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]
    files.sort(key=lambda f: f.stat().st_mtime)
    return files


def process_directory(
    root_dir: Path,
    *,
    extensions: set[str] = DEFAULT_EXTENSIONS,
    min_confidence: float = 0.6,
    tmdb_api_key: str = "",
    use_tvmaze: bool = True,
    language: str = "de-DE",
    dry_run: bool = False,
    tvh_entries: list[dict] | None = None,
    min_age: int = 300,
) -> list[OrganizationResult]:
    """Process all series subdirectories under *root_dir*.

    Each first-level subdirectory is treated as a TV series whose name
    is the directory name.  Video files within are analysed, matched
    against online databases, and moved into season subfolders.
    """
    results: list[OrganizationResult] = []

    if not root_dir.is_dir():
        logger.error("Root directory does not exist: %s", root_dir)
        return results

    series_dirs = sorted(
        [d for d in root_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    if not series_dirs:
        logger.warning("No series subdirectories found in %s", root_dir)
        return results

    logger.info("Found %d series directories in %s", len(series_dirs), root_dir)

    for series_dir in series_dirs:
        # Skip organizational subdirectories
        if series_dir.name.lower() in ("duplicates", "unmatched"):
            continue

        series_title = series_dir.name
        logger.info("=== Processing series: %s ===", series_title)

        video_files = list_video_files(series_dir, extensions)
        if not video_files:
            logger.info("No video files found in %s", series_dir)
            continue

        logger.info("Found %d video files for '%s'", len(video_files), series_title)

        for video_file in video_files:
            logger.info("--- Processing: %s ---", video_file.name)

            # Skip files still being recorded (mtime too recent)
            if min_age > 0:
                file_age = time.time() - video_file.stat().st_mtime
                if file_age < min_age:
                    logger.info(
                        "Skipping %s – file still being recorded (age %.0fs < %ds)",
                        video_file.name, file_age, min_age,
                    )
                    results.append(OrganizationResult(
                        source_path=video_file,
                        target_path=None,
                        action="recording",
                    ))
                    continue

            # Step 0: Find matching TVH DVR entry
            tvh_entry = None
            if tvh_entries:
                tvh_entry = find_dvr_entry_for_file(video_file, tvh_entries)
                if tvh_entry:
                    logger.info("TVH match: subtitle=%r, channel=%s", tvh_entry.get("subtitle", ""), tvh_entry.get("channel", ""))
                else:
                    logger.debug("No TVH entry found for %s", video_file.name)

            # Step 1: Extract metadata
            metadata = extract_all_metadata(video_file, tvh_entry=tvh_entry)

            # Step 2: Identify episode
            match = identify_episode(
                series_title,
                metadata,
                video_file,
                min_confidence=min_confidence,
                tmdb_api_key=tmdb_api_key,
                use_tvmaze=use_tvmaze,
                language=language,
            )

            # Step 2.5: Write .meta sidecar file (includes match even if below threshold)
            try:
                write_meta_file(video_file, metadata, match)
            except Exception as exc:
                logger.warning("Failed to write .meta file for %s: %s", video_file.name, exc)

            # Step 3: Organize file (only if match meets confidence threshold)
            accepted_match = match if match and match.confidence >= min_confidence else None
            if match and not accepted_match:
                logger.info(
                    "Match below threshold (%.3f < %.2f), skipping: %s",
                    match.confidence, min_confidence, video_file.name,
                )
            result = organize_file(series_dir, video_file, accepted_match, dry_run=dry_run)
            results.append(result)

    return results


def print_report(results: list[OrganizationResult]) -> None:
    """Print a summary report of all organization results."""
    if not results:
        logger.info("No files were processed.")
        return

    moved = [r for r in results if r.action == "moved"]
    duplicates = [r for r in results if r.action == "duplicate"]
    skipped = [r for r in results if r.action == "skipped"]
    recording = [r for r in results if r.action == "recording"]
    errors = [r for r in results if r.action == "error"]

    print("\n" + "=" * 70)
    print("REPORT")
    print("=" * 70)
    print(f"  Total files processed : {len(results)}")
    print(f"  Moved (identified)    : {len(moved)}")
    print(f"  Duplicates            : {len(duplicates)}")
    print(f"  Skipped (unmatched)   : {len(skipped)}")
    print(f"  Recording (skipped)   : {len(recording)}")
    print(f"  Errors                : {len(errors)}")
    print("-" * 70)

    for r in moved + duplicates:
        match = r.episode_match
        conf = f" (confidence={match.confidence:.2f}, source={match.source})" if match else ""
        print(f"  [{r.action.upper():9s}] {r.source_path.name}")
        print(f"             -> {r.target_path}{conf}")

    for r in skipped:
        print(f"  [SKIPPED  ] {r.source_path.name}")

    for r in recording:
        print(f"  [RECORDING] {r.source_path.name}")

    for r in errors:
        print(f"  [ERROR    ] {r.source_path.name}: {r.error_message}")

    print("=" * 70 + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Automatically identify and organize TV recordings into season folders.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"epicur {__version__}",
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Root directory containing series subfolders with recordings.",
    )
    parser.add_argument(
        "--tmdb-api-key",
        default=os.environ.get("EPICUR_TMDB_API_KEY", ""),
        help="Optional TMDB API key for richer episode data (or set EPICUR_TMDB_API_KEY).",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Interactively review and manually assign unmatched recordings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without actually moving files.",
    )
    parser.add_argument(
        "--no-tvmaze",
        action="store_true",
        help="Disable TVMaze lookups (use only TMDB and local matching). Implied when --language is not English.",
    )
    parser.add_argument(
        "--language",
        default="de-DE",
        help="TMDB metadata language as BCP-47 tag (default: de-DE). TVMaze is automatically disabled for non-English languages.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Minimum match confidence to accept (0.0–1.0, default: 0.6).",
    )
    parser.add_argument(
        "--tvh-dvr-log",
        type=Path,
        default=Path.home() / ".hts" / "tvheadend" / "dvr" / "log",
        help="Path to TVHeadend DVR log directory (default: ~/.hts/tvheadend/dvr/log/).",
    )
    parser.add_argument(
        "--tvh-url",
        default="",
        help="TVHeadend HTTP API URL (e.g. http://localhost:9981). Used as fallback if DVR logs are unavailable.",
    )
    parser.add_argument(
        "--tvh-user",
        default=os.environ.get("EPICUR_TVH_USER", ""),
        help="TVHeadend API username (or set EPICUR_TVH_USER).",
    )
    parser.add_argument(
        "--tvh-pass",
        default=os.environ.get("EPICUR_TVH_PASS", ""),
        help="TVHeadend API password (or set EPICUR_TVH_PASS).",
    )
    parser.add_argument(
        "--min-age",
        type=int,
        default=300,
        help="Minimum file age in seconds before processing (default: 300). Files modified more recently are assumed to still be recording. Use 0 to disable.",
    )
    parser.add_argument(
        "--extensions",
        default=".ts,.mp4,.mkv",
        help="Comma-separated list of file extensions to process (default: .ts,.mp4,.mkv).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to a log file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    setup_logging(verbose=args.verbose, log_file=args.log_file)

    extensions = {ext.strip() if ext.startswith(".") else f".{ext.strip()}" for ext in args.extensions.split(",")}

    # Interactive review mode
    if args.review:
        # Suppress log output during interactive review
        logging.getLogger().setLevel(logging.WARNING)
        review_unmatched(args.directory, extensions, dry_run=args.dry_run)
        return 0

    # Load TVHeadend DVR entries
    tvh_entries: list[dict] = []
    tvh_dvr_log = args.tvh_dvr_log
    if tvh_dvr_log.is_dir():
        tvh_entries = parse_dvr_log_dir(tvh_dvr_log)
    elif args.tvh_url:
        tvh_entries = fetch_dvr_entries_api(args.tvh_url, args.tvh_user, args.tvh_pass)
    else:
        logger.warning("No TVH data source available (DVR log dir: %s)", tvh_dvr_log)

    logger.info("epicur starting")
    logger.info("Root directory : %s", args.directory)
    logger.info("Extensions     : %s", extensions)
    logger.info("Min confidence : %.2f", args.confidence)
    logger.info("TVH entries    : %d", len(tvh_entries))
    logger.info("Dry run        : %s", args.dry_run)

    # Imply --no-tvmaze for non-English languages (TVMaze is English-only)
    use_tvmaze = not args.no_tvmaze
    if use_tvmaze and not args.language.lower().startswith("en"):
        logger.info("Disabling TVMaze (English-only) for language '%s'", args.language)
        use_tvmaze = False

    results = process_directory(
        args.directory,
        extensions=extensions,
        min_confidence=args.confidence,
        tmdb_api_key=args.tmdb_api_key,
        use_tvmaze=use_tvmaze,
        language=args.language,
        dry_run=args.dry_run,
        tvh_entries=tvh_entries or None,
        min_age=args.min_age,
    )

    print_report(results)

    errors = [r for r in results if r.action == "error"]
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
