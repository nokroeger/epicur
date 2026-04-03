"""TVHeadend DVR log parser, API client, and file-to-entry matching.

Reads DVR log JSON files from the TVHeadend data directory, normalises
the language-dict fields into plain strings, and matches recording files
on disk to their DVR log entries via basename comparison.  An optional
HTTP API client can be used as a fallback when log files are unavailable.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any

from .models import TVH_SUFFIX_RE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Language-dict helpers
# ---------------------------------------------------------------------------

def _extract_lang_value(field: Any, first_line_only: bool = False) -> str:
    """Extract a plain string from a TVH language-dict field.

    TVH stores localised text as ``{"ger": "text", "eng": "text"}``.
    Plain strings and ``None`` are handled gracefully.
    Priority: ``ger`` → first available key → empty string.

    When *first_line_only* is True, only the first line is returned.
    This is needed for subtitle fields where EPG providers pack extra
    metadata into newlines (e.g.
    ``"Angst\\nDrama, USA 2022\\nAltersfreigabe: ab 12"``).
    """
    raw = ""
    if isinstance(field, str):
        raw = field.strip()
    elif isinstance(field, dict):
        if "ger" in field:
            raw = str(field["ger"]).strip()
        else:
            for val in field.values():
                raw = str(val).strip()
                break
    if first_line_only and raw:
        raw = raw.split("\n")[0].strip()
    return raw


# ---------------------------------------------------------------------------
# Entry normalisation
# ---------------------------------------------------------------------------

def normalize_entry(raw: dict) -> dict:
    """Convert a raw DVR log/API entry into a flat dictionary.

    Returns a dict with the following keys:
      filename, basename, title, subtitle, description, directory,
      channel, start, stop, duration, file_size
    """
    # filename may be top-level (API) or only inside files[] (DVR log)
    filename = raw.get("filename", "")
    file_size = 0
    files_list = raw.get("files", [])
    if files_list and isinstance(files_list, list):
        first_file = files_list[0]
        if not filename:
            filename = first_file.get("filename", "")
        file_size = int(first_file.get("size", 0))

    start = int(raw.get("start", 0))
    stop = int(raw.get("stop", 0))

    return {
        "filename": filename,
        "basename": Path(filename).name if filename else "",
        "title": _extract_lang_value(raw.get("title", "")),
        "subtitle": _extract_lang_value(raw.get("subtitle", ""), first_line_only=True),
        "description": _extract_lang_value(raw.get("description", "")),
        "directory": raw.get("directory", ""),
        "channel": raw.get("channelname", ""),
        "start": start,
        "stop": stop,
        "duration": float(stop - start) if stop > start else 0.0,
        "file_size": file_size,
        "_raw": raw,
    }


# ---------------------------------------------------------------------------
# DVR log parser
# ---------------------------------------------------------------------------

def parse_dvr_log_dir(log_dir: Path) -> list[dict]:
    """Read and normalise all DVR log JSON files from *log_dir*.

    Each file is a single JSON object representing one recording entry.
    Malformed files are logged and skipped.
    """
    if not log_dir.is_dir():
        logger.warning("DVR log directory does not exist: %s", log_dir)
        return []

    entries: list[dict] = []
    for json_file in sorted(log_dir.iterdir()):
        if not json_file.is_file():
            continue
        try:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                entries.append(normalize_entry(raw))
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Skipping malformed DVR log %s: %s", json_file.name, exc)

    logger.info("Parsed %d DVR log entries from %s", len(entries), log_dir)
    return entries


# ---------------------------------------------------------------------------
# File-to-entry matching
# ---------------------------------------------------------------------------

def _strip_tvh_suffix(filename: str) -> str | None:
    """Strip the TVH duplicate suffix from a filename.

    ``"Star Trek_ Lower Decks-3.ts"`` → ``"Star Trek_ Lower Decks.ts"``
    Returns ``None`` if no suffix is present.
    """
    m = TVH_SUFFIX_RE.match(filename)
    if m:
        return m.group(1) + m.group(3)
    return None


def find_dvr_entry_for_file(
    file_path: Path,
    entries: list[dict],
) -> dict | None:
    """Find the DVR log entry that corresponds to *file_path*.

    Matching strategies (tried in order):
      1. Exact basename match (case-insensitive) for entries that have a filename.
      2. Basename with TVH duplicate suffix stripped (``Title-3.ts`` →
         ``Title.ts``), then pick the entry closest to the file's mtime.
      3. Directory + title match: for entries without a filename, match the
         series directory name against the file's parent, then pick by
         mtime proximity to the recording start timestamp.

    Returns the best matching normalised entry, or ``None``.
    """
    target_name = file_path.name.lower()

    # Strategy 1: exact basename match (for entries that have filenames)
    for entry in entries:
        if entry["basename"] and entry["basename"].lower() == target_name:
            logger.debug("DVR match (exact basename): %s", entry["basename"])
            return entry

    # Strategy 2: strip TVH duplicate suffix, then pick by mtime proximity
    stripped = _strip_tvh_suffix(file_path.name)
    if stripped:
        stripped_lower = stripped.lower()
        candidates = [e for e in entries if e["basename"] and e["basename"].lower() == stripped_lower]
        if candidates:
            return _pick_closest_by_mtime(file_path, candidates)

    # Strategy 3: match via directory/title + mtime proximity
    # The file's parent directory name should match the entry's directory field
    # or the entry's title.
    parent_name = file_path.parent.name
    try:
        file_mtime = file_path.stat().st_mtime
    except OSError:
        return None

    dir_candidates: list[dict] = []
    for entry in entries:
        # Skip entries that already have a (different) file on disk
        if entry["basename"] and entry["basename"].lower() != target_name:
            stripped_entry = _strip_tvh_suffix(entry["basename"])
            if stripped_entry and stripped_entry.lower() != (stripped or "").lower():
                continue

        # Match by directory name or title against the parent folder
        entry_dir = entry["directory"]
        entry_title = entry["title"]
        if entry_dir and _normalize_for_compare(entry_dir) == _normalize_for_compare(parent_name):
            dir_candidates.append(entry)
        elif entry_title and _normalize_for_compare(entry_title) == _normalize_for_compare(parent_name):
            dir_candidates.append(entry)

    if dir_candidates:
        best = _pick_closest_by_mtime(file_path, dir_candidates)
        if best:
            logger.debug(
                "DVR match (directory+mtime): %s → subtitle=%r",
                file_path.name, best["subtitle"][:60] if best["subtitle"] else "",
            )
            return best

    logger.debug("No DVR match found for %s", file_path.name)
    return None


def _normalize_for_compare(text: str) -> str:
    """Normalize a string for fuzzy directory/title comparison."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _pick_closest_by_mtime(file_path: Path, candidates: list[dict]) -> dict | None:
    """From *candidates* pick the entry whose ``start`` timestamp is closest
    to the file's modification time."""
    try:
        file_mtime = file_path.stat().st_mtime
    except OSError:
        return candidates[0] if candidates else None

    if not candidates:
        return None
    return min(candidates, key=lambda e: abs(e["start"] - file_mtime))


# ---------------------------------------------------------------------------
# TVH HTTP API client (optional fallback)
# ---------------------------------------------------------------------------

def fetch_dvr_entries_api(
    base_url: str,
    username: str = "",
    password: str = "",
    timeout: int = 30,
) -> list[dict]:
    """Fetch DVR entries from the TVHeadend HTTP API.

    Calls ``GET /api/dvr/entry/grid?limit=10000`` with optional Basic Auth.
    Returns a list of normalised entries.
    """
    import urllib.request
    import urllib.error

    if base_url.startswith("http://"):
        logger.warning(
            "TVH API URL uses plain HTTP – credentials will be sent unencrypted. "
            "Consider using https:// if the server supports it."
        )

    url = f"{base_url.rstrip('/')}/api/dvr/entry/grid?limit=10000"
    logger.info("Fetching DVR entries from TVH API: %s", url)

    req = urllib.request.Request(url, headers={"User-Agent": "epicur/1.0"})
    if username:
        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        req.add_header("Authorization", f"Basic {credentials}")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        logger.error("TVH API HTTP %d: %s", exc.code, exc.reason)
        return []
    except Exception as exc:
        logger.error("TVH API request failed: %s", exc)
        return []

    raw_entries = data.get("entries", [])
    entries = [normalize_entry(e) for e in raw_entries if isinstance(e, dict)]
    logger.info("Fetched %d DVR entries from TVH API", len(entries))
    return entries
