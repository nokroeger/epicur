"""Postprocess completed seasons: detect commercials, convert to mp4, move to library."""
from __future__ import annotations

import importlib.resources
import logging
import re
import time
import subprocess
from pathlib import Path

from .episode_identifier import (
    get_all_tmdb_episodes,
    get_tmdb_season_count,
    get_tvmaze_episodes,
    search_tmdb,
    search_tvmaze,
)
from .metadata_extractor import extract_ffprobe_metadata
from .models import PostprocessResult, SeasonInfo

logger = logging.getLogger(__name__)

# Pattern to extract S##E## or S##EXXEYY from organized filenames
_EPISODE_RE = re.compile(r"S(\d{2})E(\d{2})(?:E(\d{2}))?")

# comskip artifact extensions to clean up (relative to ts file stem)
_COMSKIP_ARTIFACTS = (".edl", ".log", ".logo.txt", ".txt")


def _default_comskip_ini() -> Path:
    """Return the path to the bundled comskip.ini."""
    ref = importlib.resources.files("epicur") / "data" / "comskip.ini"
    with importlib.resources.as_file(ref) as p:
        return Path(p)


# ---------------------------------------------------------------------------
# Season completeness
# ---------------------------------------------------------------------------


def _get_episode_count_per_season(
    series_title: str,
    *,
    tmdb_api_key: str = "",
    use_tvmaze: bool = True,
    language: str = "de-DE",
) -> dict[int, int]:
    """Query episode databases for the number of episodes per season.

    Returns a dict mapping season_number → episode_count.
    """
    counts: dict[int, int] = {}

    # Try TMDB first (supports language)
    if tmdb_api_key:
        show_id = search_tmdb(series_title, tmdb_api_key, language)
        if show_id is not None:
            season_count = get_tmdb_season_count(show_id, tmdb_api_key, language)
            for sn in range(1, season_count + 1):
                from .episode_identifier import get_tmdb_episodes
                eps = get_tmdb_episodes(show_id, sn, tmdb_api_key, language)
                if eps:
                    counts[sn] = len(eps)
            if counts:
                return counts

    # Fallback to TVMaze
    if use_tvmaze:
        show_id = search_tvmaze(series_title)
        if show_id is not None:
            episodes = get_tvmaze_episodes(show_id)
            for ep in episodes:
                sn = ep.get("season", 0)
                if sn > 0:
                    counts[sn] = counts.get(sn, 0) + 1
            if counts:
                return counts

    return counts


def _scan_season_dir(season_dir: Path, extensions: set[str]) -> dict[int, Path]:
    """Scan a Season directory and return {episode_number: file_path}. Supports multi-episode files."""
    episodes: dict[int, Path] = {}
    for f in season_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in extensions:
            continue
        m = _EPISODE_RE.search(f.name)
        if m:
            ep_start = int(m.group(2))
            ep_end = int(m.group(3)) if m.group(3) else None
            if ep_end and ep_end > ep_start:
                for ep_num in range(ep_start, ep_end + 1):
                    episodes[ep_num] = f
            else:
                episodes[ep_start] = f
    return episodes


def find_complete_seasons(
    root_dir: Path,
    extensions: set[str],
    *,
    tmdb_api_key: str = "",
    use_tvmaze: bool = True,
    language: str = "de-DE",
) -> list[SeasonInfo]:
    """Find all complete seasons across series directories in *root_dir*."""
    complete: list[SeasonInfo] = []

    if not root_dir.is_dir():
        return complete

    for series_dir in sorted(root_dir.iterdir()):
        if not series_dir.is_dir():
            continue
        if series_dir.name.lower() in ("duplicates", "unmatched"):
            continue

        series_title = series_dir.name
        logger.info("Checking completeness for: %s", series_title)

        episode_counts = _get_episode_count_per_season(
            series_title,
            tmdb_api_key=tmdb_api_key,
            use_tvmaze=use_tvmaze,
            language=language,
        )
        if not episode_counts:
            logger.warning("Could not determine episode counts for '%s', skipping", series_title)
            continue

        # Scan Season N/ subdirectories
        for season_dir in sorted(series_dir.iterdir()):
            if not season_dir.is_dir():
                continue
            m = re.match(r"^Season\s+(\d+)$", season_dir.name)
            if not m:
                continue
            season_num = int(m.group(1))
            expected = episode_counts.get(season_num)
            if not expected:
                logger.debug("No episode count for %s Season %d, skipping", series_title, season_num)
                continue

            present = _scan_season_dir(season_dir, extensions)
            missing = [ep for ep in range(1, expected + 1) if ep not in present]

            info = SeasonInfo(
                series_title=series_title,
                series_dir=series_dir,
                season_number=season_num,
                total_episodes=expected,
                present_episodes=present,
                missing_episodes=missing,
            )

            if info.is_complete:
                logger.info(
                    "Complete: %s Season %d (%d/%d episodes)",
                    series_title, season_num, len(present), expected,
                )
                complete.append(info)
            else:
                logger.info(
                    "Incomplete: %s Season %d (%d/%d, missing: %s)",
                    series_title, season_num, len(present), expected, missing,
                )

    return complete


# ---------------------------------------------------------------------------
# Commercial detection
# ---------------------------------------------------------------------------


def detect_commercials(ts_file: Path, comskip_ini: Path) -> list[tuple[float, float]]:
    """Run comskip on *ts_file* and return list of (start, end) commercial breaks in seconds."""
    try:
        subprocess.run(
            ["comskip", "--ini", str(comskip_ini), str(ts_file)],
            check=False,
            capture_output=True,
            timeout=7200,
        )
    except FileNotFoundError:
        logger.warning("comskip not found – skipping commercial detection for %s", ts_file.name)
        return []
    except subprocess.TimeoutExpired:
        logger.warning("comskip timed out for %s", ts_file.name)
        return []

    edl_file = ts_file.with_suffix(".edl")
    if not edl_file.exists():
        logger.info("No EDL file generated for %s (no commercials found)", ts_file.name)
        return []

    commercials: list[tuple[float, float]] = []
    try:
        for line in edl_file.read_text().splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                start = float(parts[0])
                end = float(parts[1])
                commercials.append((start, end))
    except (ValueError, OSError) as exc:
        logger.warning("Failed to parse EDL file %s: %s", edl_file, exc)
        return []

    logger.info("Detected %d commercial break(s) in %s", len(commercials), ts_file.name)
    return commercials


# ---------------------------------------------------------------------------
# FFmpeg metadata with chapters
# ---------------------------------------------------------------------------


def generate_ffmetadata(
    ts_file: Path,
    commercials: list[tuple[float, float]],
    title: str,
    duration_seconds: float,
) -> Path:
    """Generate an FFmpeg metadata file with chapter markers from commercial breaks.

    Returns the path to the generated .ffmeta file.
    """
    ffmeta_path = ts_file.with_suffix(".ffmeta")

    # Extract existing metadata via ffmpeg
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(ts_file), "-f", "ffmetadata", str(ffmeta_path)],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("Failed to extract ffmetadata from %s: %s", ts_file.name, exc)
        # Create a minimal metadata file
        ffmeta_path.write_text(";FFMETADATA1\n", encoding="utf-8")

    # Read existing content and append title + chapters
    content = ffmeta_path.read_text(encoding="utf-8")
    if not content.endswith("\n"):
        content += "\n"
    content += f"title={title}\n"

    if commercials:
        duration_ms = int(duration_seconds * 1000)
        start = 0
        chapter_num = 0

        for comm_start, comm_end in commercials:
            chapter_num += 1
            comm_start_ms = int(comm_start * 1000)
            comm_end_ms = int(comm_end * 1000)

            # Content chapter before commercial
            content += "\n[CHAPTER]\n"
            content += "TIMEBASE=1/1000\n"
            content += f"START={start}\n"
            content += f"END={comm_start_ms}\n"
            content += f"title=Chapter {chapter_num}\n"

            # Commercial chapter
            content += "\n[CHAPTER]\n"
            content += "TIMEBASE=1/1000\n"
            content += f"START={comm_start_ms}\n"
            content += f"END={comm_end_ms}\n"
            content += f"title=Commercial {chapter_num}\n"

            start = comm_end_ms

        # Final content chapter after last commercial
        if start < duration_ms:
            chapter_num += 1
            content += "\n[CHAPTER]\n"
            content += "TIMEBASE=1/1000\n"
            content += f"START={start}\n"
            content += f"END={duration_ms}\n"
            content += f"title=Chapter {chapter_num}\n"

    ffmeta_path.write_text(content, encoding="utf-8")
    return ffmeta_path


# ---------------------------------------------------------------------------
# TS → MP4 conversion
# ---------------------------------------------------------------------------


def convert_to_mp4(
    ts_file: Path,
    ffmeta_file: Path | None,
    output_path: Path,
    *,
    crf: int = 20,
    preset: str = "slow",
) -> bool:
    """Convert *ts_file* to MP4 with optional chapter metadata. Returns True on success."""
    cmd: list[str] = ["ffmpeg", "-y", "-i", str(ts_file)]

    if ffmeta_file and ffmeta_file.exists():
        cmd.extend(["-i", str(ffmeta_file), "-map_metadata", "1"])

    cmd.extend([
        "-c:v", "libx264",
        "-preset", preset,
        "-tune", "film",
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_path),
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=14400)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.error("FFmpeg conversion failed for %s: %s", ts_file.name, exc)
        # Clean up partial output
        if output_path.exists():
            output_path.unlink()
        return False
    except FileNotFoundError:
        logger.error("ffmpeg not found – cannot convert %s", ts_file.name)
        return False

    # Validate output
    try:
        subprocess.run(
            ["ffprobe", "-v", "error", str(output_path)],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        logger.error("Output validation failed for %s", output_path)
        if output_path.exists():
            output_path.unlink()
        return False

    return True


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup_files(ts_file: Path, ffmeta_file: Path | None) -> None:
    """Remove comskip artifacts, ffmeta file, .meta sidecar, and the source .ts file."""
    stem = ts_file.with_suffix("")
    for ext in _COMSKIP_ARTIFACTS:
        artifact = Path(str(stem) + ext)
        if artifact.exists():
            artifact.unlink()
            logger.debug("Removed artifact: %s", artifact)

    if ffmeta_file and ffmeta_file.exists():
        ffmeta_file.unlink()
        logger.debug("Removed ffmeta: %s", ffmeta_file)

    meta_file = ts_file.with_suffix(".meta")
    if meta_file.exists():
        meta_file.unlink()
        logger.debug("Removed meta: %s", meta_file)

    if ts_file.exists():
        ts_file.unlink()
        logger.info("Removed source: %s", ts_file)


# ---------------------------------------------------------------------------
# Per-episode pipeline
# ---------------------------------------------------------------------------


def postprocess_episode(
    ts_file: Path,
    output_path: Path,
    comskip_ini: Path,
    *,
    crf: int = 20,
    preset: str = "slow",
) -> PostprocessResult:
    """Run the full postprocessing pipeline for a single episode."""
    logger.info("Postprocessing: %s → %s", ts_file.name, output_path)

    # Step 1: Detect commercials
    commercials = detect_commercials(ts_file, comskip_ini)

    # Step 2: Generate chapter metadata
    ffmeta_file: Path | None = None
    if commercials:
        meta = extract_ffprobe_metadata(ts_file)
        ffmeta_file = generate_ffmetadata(
            ts_file, commercials, ts_file.stem, meta.duration_seconds,
        )
    else:
        logger.info("No commercials – converting without chapters")

    # Step 3: Convert to mp4
    success = convert_to_mp4(ts_file, ffmeta_file, output_path, crf=crf, preset=preset)

    if not success:
        # Clean up temp files only, keep source
        if ffmeta_file and ffmeta_file.exists():
            ffmeta_file.unlink()
        # Clean comskip artifacts
        stem = ts_file.with_suffix("")
        for ext in _COMSKIP_ARTIFACTS:
            artifact = Path(str(stem) + ext)
            if artifact.exists():
                artifact.unlink()
        return PostprocessResult(
            source_path=ts_file,
            output_path=None,
            action="error",
            error_message=f"FFmpeg conversion failed for {ts_file.name}",
        )

    # Step 4: Clean up source and temp files
    cleanup_files(ts_file, ffmeta_file)

    return PostprocessResult(
        source_path=ts_file,
        output_path=output_path,
        action="converted",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def postprocess_all(
    root_dir: Path,
    library_dir: Path,
    comskip_ini: Path | None,
    *,
    crf: int = 20,
    preset: str = "slow",
    extensions: set[str] | None = None,
    tmdb_api_key: str = "",
    use_tvmaze: bool = True,
    language: str = "de-DE",
    dry_run: bool = False,
) -> list[PostprocessResult]:
    """Find complete seasons and postprocess all episodes."""
    if extensions is None:
        extensions = {".ts", ".mp4", ".mkv"}

    if comskip_ini is None:
        comskip_ini = _default_comskip_ini()

    complete_seasons = find_complete_seasons(
        root_dir,
        extensions,
        tmdb_api_key=tmdb_api_key,
        use_tvmaze=use_tvmaze,
        language=language,
    )

    if not complete_seasons:
        logger.info("No complete seasons found.")
        return []


    results: list[PostprocessResult] = []

    for season in complete_seasons:
        logger.info(
            "=== Postprocessing: %s Season %d (%d episodes) ===",
            season.series_title, season.season_number, season.total_episodes,
        )

        # Multi-Episoden-Dateien: Jede Datei nur einmal verarbeiten
        file_to_eps = {}
        for ep_num, ts_file in season.present_episodes.items():
            file_to_eps.setdefault(ts_file, []).append(ep_num)

        processed_files = set()
        for ts_file, ep_nums in file_to_eps.items():
            if ts_file in processed_files:
                continue
            processed_files.add(ts_file)

            # Skip non-.ts files (already converted)
            if ts_file.suffix.lower() != ".ts":
                logger.info("Skipping non-ts file: %s", ts_file.name)
                results.append(PostprocessResult(
                    source_path=ts_file, output_path=None, action="skipped",
                ))
                continue

            # Compute output path in library_dir, preserving series/season structure
            try:
                rel = season.series_dir.resolve().relative_to(root_dir.resolve())
            except ValueError:
                rel = Path(season.series_title)
            output_path = library_dir / rel / f"Season {season.season_number}" / (ts_file.stem + ".mp4")

            # Check if already converted
            if output_path.exists():
                logger.info("Already exists in library: %s", output_path)
                results.append(PostprocessResult(
                    source_path=ts_file, output_path=output_path, action="skipped",
                ))
                continue

            if dry_run:
                logger.info("[DRY RUN] Would convert: %s → %s", ts_file, output_path)
                results.append(PostprocessResult(
                    source_path=ts_file, output_path=output_path, action="converted",
                ))
                continue

            result = postprocess_episode(
                ts_file, output_path, comskip_ini, crf=crf, preset=preset,
            )
            results.append(result)

    return results

def postprocess_movies(
    root_dir: Path,
    library_dir: Path,
    comskip_ini: Path | None,
    *,
    crf: int = 20,
    preset: str = "slow",
    extensions: set[str] | None = None,
    dry_run: bool = False,
) -> list[PostprocessResult]:
    """Process movies using the same pipeline as episodes, but without season/episode metadata. Finds all movie files in root_dir."""
    if extensions is None:
        extensions = {".ts", ".mp4", ".mkv"}

    if comskip_ini is None:
        comskip_ini = _default_comskip_ini()

    results: list[PostprocessResult] = []

    # Find all movie files in root_dir (non-recursive)
    movie_files = [f for f in root_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions]
    
    if not movie_files:
        logger.info("No movie files found in %s", root_dir)
        return results

    for movie_file in movie_files:
        logger.info("=== Postprocessing movie: %s ===", movie_file.name)


        processed_files = set()
        processed_files.add(movie_file)

        # Skip mp4 files (already converted)
        if movie_file.suffix.lower() == ".mp4":
            logger.info("Skipping already converted file: %s", movie_file.name)
            results.append(PostprocessResult(
                source_path=movie_file, output_path=None, action="skipped",
            ))
            continue
        
        # Skip files still being written (e.g. .ts files with recent modification time)
        mtime = movie_file.stat().st_mtime
        if time.time() - mtime < 60:  # Skip if modified in the last 60 seconds
            logger.info("Skipping file still being written: %s", movie_file.name)
            results.append(PostprocessResult(
                source_path=movie_file, output_path=None, action="skipped",
            ))
            continue

        # Compute output path in library_dir: flat, just use the filename
        output_path = library_dir / (movie_file.stem + ".mp4")

        # Check if already converted
        if output_path.exists():
            logger.info("Already exists in library: %s", output_path)
            results.append(PostprocessResult(
                source_path=movie_file, output_path=output_path, action="skipped",
            ))
            continue

        if dry_run:
            logger.info("[DRY RUN] Would convert: %s → %s", movie_file, output_path)
            results.append(PostprocessResult(
                source_path=movie_file, output_path=output_path, action="converted",
            ))
            continue

        #post process the movie like a series episode
        result = postprocess_episode(
            movie_file, output_path, comskip_ini, crf=crf, preset=preset,
        )
        results.append(result)

    return results


def print_postprocess_report(results: list[PostprocessResult]) -> None:
    """Print a summary report of postprocessing results."""
    if not results:
        print("\nNo episodes to postprocess.")
        return

    converted = [r for r in results if r.action == "converted"]
    skipped = [r for r in results if r.action == "skipped"]
    errors = [r for r in results if r.action == "error"]

    print("\n" + "=" * 70)
    print("POSTPROCESS REPORT")
    print("=" * 70)
    print(f"  Total episodes      : {len(results)}")
    print(f"  Converted           : {len(converted)}")
    print(f"  Skipped             : {len(skipped)}")
    print(f"  Errors              : {len(errors)}")
    print("-" * 70)

    for r in converted:
        print(f"  [CONVERTED] {r.source_path.name}")
        if r.output_path:
            print(f"              → {r.output_path}")

    for r in skipped:
        print(f"  [SKIPPED  ] {r.source_path.name}")

    for r in errors:
        print(f"  [ERROR    ] {r.source_path.name}: {r.error_message}")

    print("=" * 70 + "\n")
