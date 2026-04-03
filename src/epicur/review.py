"""Interactive review of unmatched recordings.

Reads .meta sidecar files to find recordings that were not automatically
organized (i.e. skipped because their confidence was below threshold or
had no match at all).  Presents each one interactively and lets the user
accept, override, or skip the match.
"""
from __future__ import annotations

import configparser
import io
import logging
import re
import sys
from pathlib import Path

from .file_organizer import organize_file
from .models import EpisodeMatch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Meta file parsing
# ---------------------------------------------------------------------------

def _parse_meta_file(meta_path: Path) -> dict[str, dict[str, str]]:
    """Parse a .meta file into a dict of {section: {key: value}}.

    The .meta files use a simplified INI-like format with ``=`` as
    separator.  Comment lines (# …) at the top are skipped.
    """
    sections: dict[str, dict[str, str]] = {}
    current_section: str | None = None

    for line in meta_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Section header: [name] optionally followed by a comment
        m = re.match(r"^\[(\w+)\]", stripped)
        if m:
            current_section = m.group(1)
            sections.setdefault(current_section, {})
            continue
        if current_section and "=" in stripped:
            key, _, val = stripped.partition("=")
            sections[current_section][key.strip()] = val.strip()

    return sections


def _match_from_meta(sections: dict[str, dict[str, str]]) -> EpisodeMatch | None:
    """Reconstruct an EpisodeMatch from parsed meta sections, or None."""
    ep = sections.get("episode_match", {})
    if not ep or "season" not in ep:
        return None
    try:
        return EpisodeMatch(
            series_title=ep.get("series", ""),
            season_number=int(ep.get("season", 0)),
            episode_number=int(ep.get("episode", 0)),
            episode_title=ep.get("episode_title", ""),
            episode_summary=ep.get("episode_summary", ""),
            confidence=float(ep.get("confidence", 0)),
            source=ep.get("source", ""),
        )
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Discovery: find unmatched .ts+.meta pairs
# ---------------------------------------------------------------------------

def find_unmatched_files(root_dir: Path, extensions: set[str]) -> list[tuple[Path, Path]]:
    """Find video files with .meta sidecars that were NOT organized into season folders.

    Returns a list of (video_path, meta_path) tuples sorted by series then
    filename.  Only files in the *series root* directory (not in Season N/
    or duplicates/) are considered.
    """
    pairs: list[tuple[Path, Path]] = []

    for series_dir in sorted(root_dir.iterdir()):
        if not series_dir.is_dir():
            continue
        if series_dir.name.lower() in ("duplicates", "unmatched", ".trash-1000"):
            continue

        for video_file in sorted(series_dir.iterdir()):
            if not video_file.is_file():
                continue
            if video_file.suffix.lower() not in extensions:
                continue
            meta_file = video_file.with_suffix(".meta")
            if meta_file.is_file():
                pairs.append((video_file, meta_file))

    return pairs


# ---------------------------------------------------------------------------
# Terminal colours (ANSI)
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"


def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Interactive review loop
# ---------------------------------------------------------------------------

def review_unmatched(
    root_dir: Path,
    extensions: set[str],
    *,
    dry_run: bool = False,
) -> None:
    """Interactive review of unmatched recordings.

    For each unmatched video+meta pair the user can:
      [a] Accept the suggested match
      [o] Override with manual season/episode input
      [s] Skip (leave as-is)
      [q] Quit review

    In dry-run mode the moves are simulated but not executed.
    """
    pairs = find_unmatched_files(root_dir, extensions)
    if not pairs:
        print("Keine unbearbeiteten Aufnahmen gefunden.")
        return

    total = len(pairs)
    accepted = 0
    overridden = 0
    skipped = 0

    print(f"\n{_color(f'{total} unbearbeitete Aufnahme(n) gefunden', _BOLD)}\n")

    for idx, (video_path, meta_path) in enumerate(pairs, 1):
        sections = _parse_meta_file(meta_path)
        match = _match_from_meta(sections)

        meta_sec = sections.get("metadata", {})
        tvh_sec = sections.get("tvheadend", {})

        series_dir = video_path.parent
        series_title = series_dir.name

        # ── Display ──────────────────────────────────────────────
        print(_color(f"[{idx}/{total}]", _BOLD) + f"  {_color(series_title, _CYAN)}")
        print(f"  Datei: {video_path.name}")
        print()

        # Title / description from metadata & TVH
        desc = meta_sec.get("description") or tvh_sec.get("description") or ""
        tvh_sub = tvh_sec.get("subtitle", "")
        channel = tvh_sec.get("channel") or meta_sec.get("channel") or ""
        tvh_start = tvh_sec.get("start", "")

        if tvh_sub:
            print(f"  TVH Subtitle : {tvh_sub}")
        if channel:
            print(f"  Sender       : {channel}")
        if tvh_start:
            print(f"  Aufnahmedatum: {tvh_start}")
        if desc:
            # Wrap long descriptions
            max_w = 72
            label = "  Beschreibung : "
            indent = " " * len(label)
            words = desc.split()
            lines: list[str] = []
            cur = label
            for w in words:
                if len(cur) + len(w) + 1 > max_w and cur.strip():
                    lines.append(cur)
                    cur = indent
                cur += w + " "
            if cur.strip():
                lines.append(cur)
            print("\n".join(lines))

        print()

        if match and match.season_number > 0:
            conf_color = _GREEN if match.confidence >= 0.5 else (_YELLOW if match.confidence >= 0.3 else _RED)
            print(f"  {_color('Vorschlag:', _BOLD)}")
            print(f"    S{match.season_number:02d}E{match.episode_number:02d} \"{match.episode_title}\"")
            print(f"    Konfidenz: {_color(f'{match.confidence:.3f}', conf_color)}  (Quelle: {match.source})")
            if match.episode_summary:
                print(f"    Inhalt: {match.episode_summary[:120]}{'…' if len(match.episode_summary) > 120 else ''}")
        else:
            print(f"  {_color('Kein Vorschlag vorhanden', _DIM)}")

        print()

        # ── User input ───────────────────────────────────────────
        while True:
            options = "[a]kzeptieren  " if match and match.season_number > 0 else ""
            options += "[ü]berschreiben  [s]kippen  [q]uit"
            try:
                choice = input(f"  {options} > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n\nAbgebrochen.")
                _print_summary(accepted, overridden, skipped, idx - 1 - accepted - overridden - skipped)
                return

            if choice in ("q", "quit"):
                remaining = total - idx
                _print_summary(accepted, overridden, skipped, remaining + 1)
                return

            if choice in ("s", "skip", ""):
                skipped += 1
                print(f"  → {_color('Übersprungen', _DIM)}\n")
                break

            if choice in ("a", "accept", "akzeptieren") and match and match.season_number > 0:
                result = organize_file(series_dir, video_path, match, dry_run=dry_run)
                if result.action in ("moved", "duplicate"):
                    action_label = "Verschoben" if result.action == "moved" else "Duplikat"
                    print(f"  → {_color(action_label, _GREEN)}: {result.target_path}")
                    accepted += 1
                else:
                    print(f"  → {result.action}: {result.error_message or 'unbekannt'}")
                    skipped += 1
                print()
                break

            if choice in ("ü", "u", "o", "override", "überschreiben"):
                override_match = _prompt_override(series_title, match)
                if override_match is None:
                    # User cancelled the override
                    continue
                result = organize_file(series_dir, video_path, override_match, dry_run=dry_run)
                if result.action in ("moved", "duplicate"):
                    action_label = "Verschoben" if result.action == "moved" else "Duplikat"
                    print(f"  → {_color(action_label, _GREEN)}: {result.target_path}")
                    overridden += 1
                else:
                    print(f"  → {result.action}: {result.error_message or 'unbekannt'}")
                    skipped += 1
                print()
                break

            print(f"  {_color('Ungültige Eingabe.', _RED)}")

    _print_summary(accepted, overridden, skipped, 0)


def _prompt_override(series_title: str, existing_match: EpisodeMatch | None) -> EpisodeMatch | None:
    """Prompt for manual season/episode numbers.  Returns an EpisodeMatch or None on cancel."""
    try:
        raw = input("    Staffel Folge (z.B. '3 12' oder 'S03E12', leer=abbrechen): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if not raw:
        return None

    # Parse "S03E12" or "3 12" or "3/12"
    m = re.match(r"[Ss]?(\d{1,2})\s*[Ee /]?\s*(\d{1,3})", raw)
    if not m:
        print(f"    {_color('Format nicht erkannt. Erwartet: z.B. 3 12 oder S03E12', _RED)}")
        return None

    season = int(m.group(1))
    episode = int(m.group(2))

    ep_title = ""
    if existing_match and existing_match.season_number == season and existing_match.episode_number == episode:
        ep_title = existing_match.episode_title

    return EpisodeMatch(
        series_title=series_title,
        season_number=season,
        episode_number=episode,
        episode_title=ep_title,
        confidence=1.0,
        source="manual",
    )


def _print_summary(accepted: int, overridden: int, skipped: int, remaining: int) -> None:
    """Print a short summary of the review session."""
    print(f"\n{'─' * 50}")
    print(f"  Akzeptiert:    {accepted}")
    print(f"  Überschrieben: {overridden}")
    print(f"  Übersprungen:  {skipped}")
    if remaining > 0:
        print(f"  Verbleibend:   {remaining}")
    print(f"{'─' * 50}\n")
