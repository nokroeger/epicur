from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from pathlib import Path

# Regex for TVH duplicate suffix: "Title-3.ts" → group(1)="Title", group(2)="3", group(3)=".ts"
TVH_SUFFIX_RE = re.compile(r"^(.+)-(\d+)(\.\w+)$")


@dataclass
class ExtractedMetadata:
    title: str = ""
    description: str = ""
    duration_seconds: float = 0.0
    channel: str = ""
    embedded_episode_title: str = ""
    subtitle_texts: list[str] = field(default_factory=list)
    # TVHeadend EPG fields
    tvh_subtitle: str = ""
    tvh_description: str = ""
    tvh_channel: str = ""
    tvh_start: int = 0
    tvh_stop: int = 0

    def has_useful_data(self) -> bool:
        return bool(
            self.title
            or self.description
            or self.embedded_episode_title
            or self.tvh_subtitle
            or self.tvh_description
            or self.subtitle_texts
        )


@dataclass
class EpisodeMatch:
    series_title: str
    season_number: int
    episode_number: int
    episode_title: str = ""
    episode_summary: str = ""
    confidence: float = 0.0
    source: str = ""  # "tvmaze", "tmdb", "metadata", "filename"


@dataclass
class OrganizationResult:
    source_path: Path
    target_path: Path | None
    action: str  # "moved", "duplicate", "skipped", "recording", "error"
    episode_match: EpisodeMatch | None = None
    error_message: str = ""


@dataclass
class SeasonInfo:
    series_title: str
    series_dir: Path
    season_number: int
    total_episodes: int
    present_episodes: dict[int, Path] = field(default_factory=dict)
    missing_episodes: list[int] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return len(self.present_episodes) >= self.total_episodes > 0


@dataclass
class PostprocessResult:
    source_path: Path
    output_path: Path | None
    action: str  # "converted", "skipped", "error"
    error_message: str = ""
