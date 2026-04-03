from __future__ import annotations

import datetime
import json
import logging
import subprocess
from pathlib import Path

from .models import EpisodeMatch, ExtractedMetadata

logger = logging.getLogger(__name__)

# How many seconds of subtitles to extract from the beginning of the file.
SUBTITLE_EXTRACT_SECONDS = 600  # 10 minutes


def extract_ffprobe_metadata(file_path: Path) -> ExtractedMetadata:
    """Extract metadata from a video file using ffprobe.

    Parses format tags for EPG data commonly embedded in DVB/ATSC recordings
    (title, service_name, description, episode_id, etc.).
    """
    metadata = ExtractedMetadata()

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        logger.warning("ffprobe not found – skipping metadata extraction")
        return metadata
    except subprocess.TimeoutExpired:
        logger.warning("ffprobe timed out for %s", file_path)
        return metadata

    if result.returncode != 0:
        logger.warning("ffprobe failed for %s: %s", file_path, result.stderr.strip())
        return metadata

    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.warning("ffprobe returned invalid JSON for %s", file_path)
        return metadata

    fmt = probe.get("format", {})
    tags = fmt.get("tags", {})

    # Duration
    try:
        metadata.duration_seconds = float(fmt.get("duration", 0))
    except (ValueError, TypeError):
        pass

    # Tags – DVB recordings commonly use these keys (case-insensitive lookup)
    tags_lower = {k.lower(): v for k, v in tags.items()}

    metadata.title = (
        tags_lower.get("title", "")
        or tags_lower.get("service_name", "")
    )
    metadata.description = (
        tags_lower.get("description", "")
        or tags_lower.get("synopsis", "")
        or tags_lower.get("comment", "")
    )
    metadata.channel = tags_lower.get("service_provider", "") or tags_lower.get("service_name", "")
    metadata.embedded_episode_title = tags_lower.get("episode_id", "") or tags_lower.get("episode-id", "")

    # Also check per-stream tags (some muxers put EPG data on the video stream)
    for stream in probe.get("streams", []):
        stags = {k.lower(): v for k, v in stream.get("tags", {}).items()}
        if not metadata.title and stags.get("title"):
            metadata.title = stags["title"]
        if not metadata.description and stags.get("description"):
            metadata.description = stags["description"]

    logger.debug(
        "ffprobe metadata for %s: title=%r, description=%r, duration=%.1fs",
        file_path.name, metadata.title, metadata.description[:80], metadata.duration_seconds,
    )
    return metadata


def extract_subtitle_text(file_path: Path, duration_seconds: float = 0) -> list[str]:
    """Extract subtitle/CC text from the first minutes of a video file.

    Tries embedded subtitle streams first, then DVB teletext data streams.
    Returns a list of subtitle text lines.
    """
    texts: list[str] = []
    extract_duration = min(SUBTITLE_EXTRACT_SECONDS, duration_seconds) if duration_seconds > 0 else SUBTITLE_EXTRACT_SECONDS

    # Try subtitle stream (0:s:0), then data stream (0:d:0)
    for stream_spec in ["0:s:0", "0:d:0"]:
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-i", str(file_path),
                    "-t", str(extract_duration),
                    "-map", stream_spec,
                    "-f", "srt",
                    "pipe:1",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

        if result.returncode != 0 or not result.stdout:
            continue

        # Parse SRT: extract only text lines (skip index and timestamp lines)
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.isdigit():
                continue
            if "-->" in line:
                continue
            texts.append(line)

        if texts:
            logger.debug("Extracted %d subtitle lines from stream %s", len(texts), stream_spec)
            break  # got subtitles, no need to try other streams

    return texts


def extract_all_metadata(file_path: Path, *, tvh_entry: dict | None = None) -> ExtractedMetadata:
    """Run the full metadata extraction pipeline on a video file.

    Combines ffprobe metadata, TVHeadend EPG data, and subtitle text.
    Each step degrades gracefully if tools/data are unavailable.
    """
    logger.info("Extracting metadata from %s", file_path.name)

    # Step 1: ffprobe -- always attempted (most reliable)
    metadata = extract_ffprobe_metadata(file_path)

    # Step 2: TVHeadend DVR data -- enrich with EPG fields
    if tvh_entry:
        metadata.tvh_subtitle = tvh_entry.get("subtitle", "")
        metadata.tvh_description = tvh_entry.get("description", "")
        metadata.tvh_channel = tvh_entry.get("channel", "")
        metadata.tvh_start = tvh_entry.get("start", 0)
        metadata.tvh_stop = tvh_entry.get("stop", 0)
        # Supplement ffprobe fields if they are empty
        if not metadata.channel and metadata.tvh_channel:
            metadata.channel = metadata.tvh_channel
        if not metadata.description and metadata.tvh_description:
            metadata.description = metadata.tvh_description
        logger.debug(
            "TVH data for %s: subtitle=%r, description=%r",
            file_path.name, metadata.tvh_subtitle,
            metadata.tvh_description[:80] if metadata.tvh_description else "",
        )

    # Step 3: Subtitles -- best-effort
    try:
        metadata.subtitle_texts = extract_subtitle_text(file_path, metadata.duration_seconds)
    except Exception as exc:
        logger.warning("Subtitle extraction failed for %s: %s", file_path.name, exc)

    logger.info(
        "Metadata extraction complete for %s: title=%r, tvh_subtitle=%r, subtitle_lines=%d",
        file_path.name, metadata.title, metadata.tvh_subtitle, len(metadata.subtitle_texts),
    )
    return metadata


def write_meta_file(file_path: Path, metadata: ExtractedMetadata, match: EpisodeMatch | None = None) -> Path:
    """Write a .meta sidecar file with all extracted data for later analysis.

    The file is written next to the original video file with the same stem
    and a ``.meta`` extension.  Contains ffprobe metadata, TVHeadend EPG
    data, subtitle text, and (if available) the episode match result.
    """
    meta_path = file_path.with_suffix(".meta")

    sections: list[str] = []

    # Header
    sections.append(f"# Metadata for: {file_path.name}")
    sections.append(f"# Generated: {datetime.datetime.now().isoformat()}")
    sections.append("")

    # ffprobe metadata
    sections.append("[metadata]")
    sections.append(f"title           = {metadata.title}")
    sections.append(f"description     = {metadata.description}")
    sections.append(f"channel         = {metadata.channel}")
    sections.append(f"episode_title   = {metadata.embedded_episode_title}")
    sections.append(f"duration        = {metadata.duration_seconds:.1f}s")
    sections.append("")

    # Episode match (if any)
    if match is not None:
        sections.append("[episode_match]")
        sections.append(f"series          = {match.series_title}")
        sections.append(f"season          = {match.season_number}")
        sections.append(f"episode         = {match.episode_number}")
        sections.append(f"episode_title   = {match.episode_title}")
        sections.append(f"episode_summary = {match.episode_summary}")
        sections.append(f"confidence      = {match.confidence:.3f}")
        sections.append(f"source          = {match.source}")
        sections.append("")
    else:
        sections.append("[episode_match]")
        sections.append("# No episode match found")
        sections.append("")

    # TVHeadend EPG data
    sections.append("[tvheadend]")
    sections.append(f"subtitle        = {metadata.tvh_subtitle}")
    sections.append(f"description     = {metadata.tvh_description}")
    sections.append(f"channel         = {metadata.tvh_channel}")
    if metadata.tvh_start:
        start_dt = datetime.datetime.fromtimestamp(metadata.tvh_start).isoformat()
        stop_dt = datetime.datetime.fromtimestamp(metadata.tvh_stop).isoformat() if metadata.tvh_stop else ""
        sections.append(f"start           = {start_dt}")
        sections.append(f"stop            = {stop_dt}")
    sections.append("")

    # Subtitle text
    sections.append(f"[subtitles] # {len(metadata.subtitle_texts)} lines")
    for line in metadata.subtitle_texts:
        sections.append(line)

    content = "\n".join(sections) + "\n"
    meta_path.write_text(content, encoding="utf-8")
    logger.info("Wrote meta file: %s", meta_path)
    return meta_path
