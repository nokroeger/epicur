"""Unit tests for epicur core modules."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from epicur.models import TVH_SUFFIX_RE, EpisodeMatch, ExtractedMetadata, OrganizationResult
from epicur.episode_identifier import (
    _normalize,
    _similarity,
    _keyword_overlap,
    _content_words,
    _prefix_match,
    _compute_idf,
    _idf_keyword_score,
    _fuzzy_match_episode,
    _strip_html,
    match_from_filename,
)
from epicur.tvheadend_client import (
    _strip_tvh_suffix,
    _pick_closest_by_mtime,
    normalize_entry,
    find_dvr_entry_for_file,
    _extract_lang_value,
    _normalize_for_compare,
)
from epicur.file_organizer import (
    _base_stem,
    compute_target_path,
    compute_duplicate_path,
    organize_file,
    check_library_duplicate,
)
from epicur.review import (
    _parse_meta_file,
    _match_from_meta,
    find_unmatched_files,
)
from epicur.metadata_extractor import write_meta_file
from epicur.main import parse_args, list_video_files, process_directory, print_report, main


# -----------------------------------------------------------------------
# models.py
# -----------------------------------------------------------------------

class TestTVHSuffixRE:
    def test_matches_suffix(self):
        m = TVH_SUFFIX_RE.match("Star Trek-3.ts")
        assert m is not None
        assert m.group(1) == "Star Trek"
        assert m.group(2) == "3"
        assert m.group(3) == ".ts"

    def test_no_suffix(self):
        assert TVH_SUFFIX_RE.match("Star Trek.ts") is None

    def test_multi_digit(self):
        m = TVH_SUFFIX_RE.match("Show-12.mkv")
        assert m is not None
        assert m.group(2) == "12"


# -----------------------------------------------------------------------
# episode_identifier.py — pure functions
# -----------------------------------------------------------------------

class TestNormalize:
    def test_basic(self):
        assert _normalize("  Hello, World!  ") == "hello world"

    def test_collapses_whitespace(self):
        assert _normalize("a   b\tc") == "a b c"

    def test_strips_punctuation(self):
        assert _normalize("J.A.G. - Im Auftrag") == "j a g im auftrag"


class TestSimilarity:
    def test_identical(self):
        assert _similarity("hello", "hello") == 1.0

    def test_empty(self):
        assert _similarity("", "hello") == 0.0
        assert _similarity("hello", "") == 0.0

    def test_different(self):
        assert _similarity("abc", "xyz") < 0.5


class TestKeywordOverlap:
    def test_identical(self):
        assert _keyword_overlap("Der schnelle Fuchs", "Der schnelle Fuchs") == 1.0

    def test_no_overlap(self):
        assert _keyword_overlap("Fuchs Igel", "Katze Hund") == 0.0

    def test_stopwords_removed(self):
        # "der" and "die" are stopwords, so only content words matter
        score = _keyword_overlap("der Fuchs", "die Katze")
        assert score == 0.0


class TestStripHtml:
    def test_removes_tags(self):
        assert _strip_html("<p>Hello <b>World</b></p>") == "Hello World"

    def test_plain_text(self):
        assert _strip_html("no tags here") == "no tags here"


class TestMatchFromFilename:
    def test_standard_pattern(self):
        m = match_from_filename(Path("/tmp/Show S03E12.ts"), "Show")
        assert m is not None
        assert m.season_number == 3
        assert m.episode_number == 12
        assert m.source == "filename"

    def test_no_pattern(self):
        assert match_from_filename(Path("/tmp/Show.ts"), "Show") is None


# -----------------------------------------------------------------------
# tvheadend_client.py
# -----------------------------------------------------------------------

class TestStripTvhSuffix:
    def test_strips_suffix(self):
        assert _strip_tvh_suffix("Show-3.ts") == "Show.ts"

    def test_no_suffix(self):
        assert _strip_tvh_suffix("Show.ts") is None


class TestExtractLangValue:
    def test_plain_string(self):
        assert _extract_lang_value("hello") == "hello"

    def test_dict_ger(self):
        assert _extract_lang_value({"ger": "Hallo", "eng": "Hello"}) == "Hallo"

    def test_dict_fallback(self):
        assert _extract_lang_value({"eng": "Hello"}) == "Hello"

    def test_none(self):
        assert _extract_lang_value(None) == ""

    def test_first_line_only(self):
        assert _extract_lang_value("line1\nline2\nline3", first_line_only=True) == "line1"


class TestNormalizeEntry:
    def test_basic(self):
        raw = {
            "filename": "/recordings/Show/ep.ts",
            "title": {"ger": "Show"},
            "subtitle": {"ger": "Episode 1"},
            "description": {"ger": "A description"},
            "channelname": "ZDF",
            "start": 1000,
            "stop": 2000,
        }
        entry = normalize_entry(raw)
        assert entry["basename"] == "ep.ts"
        assert entry["title"] == "Show"
        assert entry["subtitle"] == "Episode 1"
        assert entry["duration"] == 1000.0

    def test_files_list_fallback(self):
        raw = {
            "files": [{"filename": "/path/to/file.ts", "size": 12345}],
            "title": "T",
            "start": 0,
            "stop": 0,
        }
        entry = normalize_entry(raw)
        assert entry["filename"] == "/path/to/file.ts"
        assert entry["file_size"] == 12345


class TestNormalizeForCompare:
    def test_strips_special_chars(self):
        assert _normalize_for_compare("J.A.G. - Test") == "jagtest"


# -----------------------------------------------------------------------
# file_organizer.py
# -----------------------------------------------------------------------

class TestBaseStem:
    def test_no_suffix(self):
        assert _base_stem(Path("/tmp/Show.ts")) == "Show"

    def test_with_suffix(self):
        assert _base_stem(Path("/tmp/Show-3.ts")) == "Show"


class TestComputeTargetPath:
    def test_basic(self):
        series_dir = Path("/recordings/JAG")
        match = EpisodeMatch(series_title="JAG", season_number=2, episode_number=5)
        source = Path("/recordings/JAG/JAG.ts")
        target = compute_target_path(series_dir, match, source)
        assert target == series_dir / "Season 2" / "JAG S02E05.ts"

    def test_path_traversal_blocked(self):
        series_dir = Path("/recordings/JAG")
        # A malicious season_number can't escape because it's formatted as int
        # But a crafted series_title in the stem could try via symlinks
        match = EpisodeMatch(series_title="JAG", season_number=1, episode_number=1)
        source = Path("/recordings/JAG/../../etc/passwd.ts")
        # compute_target_path uses _base_stem which works on the filename only,
        # so the path cannot escape series_dir through the filename stem.
        target = compute_target_path(series_dir, match, source)
        assert target.resolve().is_relative_to(series_dir.resolve())


class TestOrganizeFile:
    def test_skip_unmatched(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.touch()
        result = organize_file(tmp_path, video, None)
        assert result.action == "skipped"

    def test_dry_run(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.touch()
        match = EpisodeMatch(series_title="Show", season_number=1, episode_number=1)
        result = organize_file(tmp_path, video, match, dry_run=True)
        assert result.action == "moved"
        assert result.target_path is not None
        # File should NOT have been moved
        assert video.exists()

    def test_actual_move(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.write_text("content")
        match = EpisodeMatch(series_title="Show", season_number=1, episode_number=1)
        result = organize_file(tmp_path, video, match, dry_run=False)
        assert result.action == "moved"
        assert result.target_path is not None
        assert result.target_path.exists()
        assert not video.exists()

    def test_duplicate_handling(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.write_text("content")
        match = EpisodeMatch(series_title="Show", season_number=1, episode_number=1)
        # Pre-create the target to trigger duplicate logic
        target = compute_target_path(tmp_path, match, video)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("existing")
        result = organize_file(tmp_path, video, match, dry_run=False)
        assert result.action == "duplicate"
        assert "duplicates" in str(result.target_path)

    def test_library_duplicate_detected(self, tmp_path: Path):
        root = tmp_path / "recordings"
        series = root / "Show"
        series.mkdir(parents=True)
        video = series / "show.ts"
        video.write_text("content")
        match = EpisodeMatch(series_title="Show", season_number=1, episode_number=1)

        lib = tmp_path / "library"
        lib_target = lib / "Show" / "Season 1" / "show S01E01.mp4"
        lib_target.parent.mkdir(parents=True)
        lib_target.write_text("converted")

        result = organize_file(series, video, match, dry_run=True, library_dir=lib, root_dir=root)
        assert result.action == "duplicate"

    def test_library_duplicate_not_found(self, tmp_path: Path):
        root = tmp_path / "recordings"
        series = root / "Show"
        series.mkdir(parents=True)
        video = series / "show.ts"
        video.write_text("content")
        match = EpisodeMatch(series_title="Show", season_number=1, episode_number=1)

        lib = tmp_path / "library"
        lib.mkdir()

        result = organize_file(series, video, match, dry_run=True, library_dir=lib, root_dir=root)
        assert result.action == "moved"

    def test_library_dir_none_unchanged(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.write_text("content")
        match = EpisodeMatch(series_title="Show", season_number=1, episode_number=1)
        result = organize_file(tmp_path, video, match, dry_run=True, library_dir=None)
        assert result.action == "moved"


# -----------------------------------------------------------------------
# DVR log parsing integration
# -----------------------------------------------------------------------

class TestParseDvrLogDir:
    def test_reads_json_files(self, tmp_path: Path):
        from epicur.tvheadend_client import parse_dvr_log_dir

        entry = {
            "filename": "/rec/show.ts",
            "title": {"ger": "Show"},
            "subtitle": {"ger": "Ep1"},
            "description": {"ger": "Desc"},
            "channelname": "ARD",
            "start": 1000,
            "stop": 2000,
        }
        (tmp_path / "entry1").write_text(json.dumps(entry), encoding="utf-8")
        # A malformed file should be skipped
        (tmp_path / "entry2").write_text("not json", encoding="utf-8")

        entries = parse_dvr_log_dir(tmp_path)
        assert len(entries) == 1
        assert entries[0]["title"] == "Show"

    def test_missing_dir(self, tmp_path: Path):
        from epicur.tvheadend_client import parse_dvr_log_dir

        entries = parse_dvr_log_dir(tmp_path / "nonexistent")
        assert entries == []


# =======================================================================
# HIGH-VALUE TESTS
# =======================================================================

# -----------------------------------------------------------------------
# episode_identifier.py — _content_words
# -----------------------------------------------------------------------

class TestContentWords:
    def test_removes_stopwords(self):
        words = _content_words("der schnelle Fuchs springt über den Zaun")
        assert "der" not in words
        assert "den" not in words
        assert "über" not in words
        assert "fuchs" in words
        assert "schnelle" in words
        assert "zaun" in words

    def test_empty(self):
        assert _content_words("") == set()

    def test_only_stopwords(self):
        assert _content_words("der die das und oder") == set()


# -----------------------------------------------------------------------
# episode_identifier.py — _prefix_match
# -----------------------------------------------------------------------

class TestPrefixMatch:
    def test_matches_german_inflections(self):
        words_a = {"spirituell", "kurz"}
        words_b = {"spirituellen", "lang"}
        matched = _prefix_match(words_a, words_b, min_prefix=7)
        assert "spirituell" in matched

    def test_no_match_short_words(self):
        words_a = {"kurz", "lang"}
        words_b = {"kurze", "lange"}
        matched = _prefix_match(words_a, words_b, min_prefix=7)
        assert matched == set()

    def test_no_match_different_words(self):
        words_a = {"abcdefgh"}
        words_b = {"xyzabcde"}
        matched = _prefix_match(words_a, words_b, min_prefix=7)
        assert matched == set()

    def test_excludes_exact_matches(self):
        # Exact matches are not returned (handled separately)
        words_a = {"identical"}
        words_b = {"identical"}
        matched = _prefix_match(words_a, words_b, min_prefix=7)
        assert matched == set()


# -----------------------------------------------------------------------
# episode_identifier.py — _compute_idf
# -----------------------------------------------------------------------

class TestComputeIdf:
    def test_empty_documents(self):
        assert _compute_idf([]) == {}

    def test_single_document(self):
        idf = _compute_idf(["Fuchs Igel Hund"])
        # With 1 doc, IDF = log(1/1) = 0 for all words
        for v in idf.values():
            assert v == 0.0

    def test_rare_words_get_higher_weight(self):
        docs = [
            "Fuchs Igel Katze",
            "Fuchs Hund Maus",
            "Fuchs Vogel Fisch",
        ]
        idf = _compute_idf(docs)
        # "fuchs" appears in all 3 docs → lowest IDF
        # "igel" appears in 1 doc → highest IDF
        assert idf["fuchs"] < idf["igel"]
        assert idf["fuchs"] < idf["hund"]


# -----------------------------------------------------------------------
# episode_identifier.py — _idf_keyword_score
# -----------------------------------------------------------------------

class TestIdfKeywordScore:
    def test_identical_texts(self):
        idf = _compute_idf(["Fuchs Igel Katze", "Hund Maus Fisch"])
        score = _idf_keyword_score("Fuchs Igel Katze", "Fuchs Igel Katze", idf)
        assert score == 1.0

    def test_no_overlap(self):
        idf = _compute_idf(["Fuchs Igel", "Katze Hund"])
        score = _idf_keyword_score("Fuchs Igel", "Katze Hund", idf)
        assert score == 0.0

    def test_partial_overlap(self):
        idf = _compute_idf(["Fuchs Igel", "Fuchs Katze", "Hund Maus"])
        score = _idf_keyword_score("Fuchs Igel", "Fuchs Katze", idf)
        assert 0.0 < score < 1.0

    def test_empty_text(self):
        assert _idf_keyword_score("", "Fuchs", {}) == 0.0
        assert _idf_keyword_score("Fuchs", "", {}) == 0.0

    def test_prefix_matching_boosts_score(self):
        idf = _compute_idf(["spirituell meditativ", "actionreich spannend"])
        # "spirituell" and "spirituellen" share ≥7 char prefix
        score_prefix = _idf_keyword_score("spirituell meditativ", "spirituellen meditativ", idf)
        score_none = _idf_keyword_score("spirituell meditativ", "actionreich spannend", idf)
        assert score_prefix > score_none


# -----------------------------------------------------------------------
# episode_identifier.py — _fuzzy_match_episode
# -----------------------------------------------------------------------

class TestFuzzyMatchEpisode:
    def test_exact_title_match(self):
        metadata = ExtractedMetadata(tvh_subtitle="Angst")
        episodes = [
            {"season": 1, "number": 1, "name": "Pilot", "summary": "First episode"},
            {"season": 1, "number": 2, "name": "Angst", "summary": "Second episode"},
            {"season": 1, "number": 3, "name": "Mut", "summary": "Third episode"},
        ]
        result = _fuzzy_match_episode("TestShow", metadata, episodes, source="test")
        assert result is not None
        assert result.episode_number == 2
        assert result.episode_title == "Angst"
        assert result.confidence > 0.8

    def test_description_matching(self):
        metadata = ExtractedMetadata(
            description="Ein Detektiv sucht einen verschwundenen Diamanten in Berlin"
        )
        episodes = [
            {"season": 1, "number": 1, "name": "Ep1",
             "summary": "Ein Detektiv sucht einen verschwundenen Diamanten in Berlin"},
            {"season": 1, "number": 2, "name": "Ep2",
             "summary": "Eine Köchin bereitet ein Festmahl vor"},
        ]
        result = _fuzzy_match_episode("TestShow", metadata, episodes, source="test")
        assert result is not None
        assert result.episode_number == 1

    def test_empty_episodes_list(self):
        metadata = ExtractedMetadata(title="Something")
        result = _fuzzy_match_episode("TestShow", metadata, [], source="test")
        assert result is None

    def test_tvh_subtitle_same_as_series_ignored(self):
        # If tvh_subtitle matches the series title (>85% similar), it's discarded
        metadata = ExtractedMetadata(
            tvh_subtitle="TestShow",
            description="A unique detective story in Paris"
        )
        episodes = [
            {"season": 1, "number": 1, "name": "TestShow",
             "summary": "Irrelevant"},
            {"season": 1, "number": 2, "name": "Other",
             "summary": "A unique detective story in Paris"},
        ]
        result = _fuzzy_match_episode("TestShow", metadata, episodes, source="test")
        assert result is not None
        # Should prefer Ep2 based on description, not Ep1 based on title-repeat
        assert result.episode_number == 2

    def test_returns_best_confidence(self):
        metadata = ExtractedMetadata(tvh_subtitle="Alpha")
        episodes = [
            {"season": 1, "number": 1, "name": "Alpha", "summary": ""},
            {"season": 1, "number": 2, "name": "Alph", "summary": ""},
        ]
        result = _fuzzy_match_episode("Show", metadata, episodes, source="test")
        assert result is not None
        assert result.episode_number == 1  # "Alpha" is a better match than "Alph"


# -----------------------------------------------------------------------
# tvheadend_client.py — find_dvr_entry_for_file
# -----------------------------------------------------------------------

class TestFindDvrEntryForFile:
    def _make_entry(self, filename: str, title: str = "", subtitle: str = "",
                    directory: str = "", start: int = 0) -> dict:
        return normalize_entry({
            "filename": filename,
            "title": title,
            "subtitle": subtitle,
            "description": "",
            "channelname": "ARD",
            "start": start,
            "stop": start + 3600,
        })

    def test_exact_basename_match(self, tmp_path: Path):
        video = tmp_path / "Show" / "episode.ts"
        video.parent.mkdir()
        video.touch()
        entries = [self._make_entry("/rec/Show/episode.ts")]
        result = find_dvr_entry_for_file(video, entries)
        assert result is not None
        assert result["basename"] == "episode.ts"

    def test_exact_match_case_insensitive(self, tmp_path: Path):
        video = tmp_path / "Show" / "Episode.TS"
        video.parent.mkdir()
        video.touch()
        entries = [self._make_entry("/rec/Show/episode.ts")]
        result = find_dvr_entry_for_file(video, entries)
        assert result is not None

    def test_suffix_stripped_match(self, tmp_path: Path):
        video = tmp_path / "Show" / "Show-3.ts"
        video.parent.mkdir()
        video.write_text("x")
        entries = [
            self._make_entry("/rec/Show/Show.ts", start=int(video.stat().st_mtime)),
        ]
        result = find_dvr_entry_for_file(video, entries)
        assert result is not None
        assert result["basename"] == "Show.ts"

    def test_directory_title_match(self, tmp_path: Path):
        series_dir = tmp_path / "JAG"
        series_dir.mkdir()
        video = series_dir / "unknown_file.ts"
        video.write_text("x")
        mtime = int(video.stat().st_mtime)
        entries = [
            self._make_entry("", title="JAG", subtitle="Ep1", start=mtime),
        ]
        result = find_dvr_entry_for_file(video, entries)
        assert result is not None
        assert result["subtitle"] == "Ep1"

    def test_no_match(self, tmp_path: Path):
        video = tmp_path / "Show" / "completely_different.ts"
        video.parent.mkdir()
        video.touch()
        entries = [self._make_entry("/rec/Other/other.ts", title="Other")]
        result = find_dvr_entry_for_file(video, entries)
        assert result is None


# -----------------------------------------------------------------------
# tvheadend_client.py — _pick_closest_by_mtime
# -----------------------------------------------------------------------

class TestPickClosestByMtime:
    def test_picks_closest(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.write_text("x")
        mtime = video.stat().st_mtime
        candidates = [
            {"start": int(mtime) - 10000, "basename": "a"},
            {"start": int(mtime) - 5, "basename": "b"},  # closest
            {"start": int(mtime) + 10000, "basename": "c"},
        ]
        result = _pick_closest_by_mtime(video, candidates)
        assert result is not None
        assert result["basename"] == "b"

    def test_empty_candidates(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.write_text("x")
        assert _pick_closest_by_mtime(video, []) is None

    def test_file_not_found_fallback(self, tmp_path: Path):
        video = tmp_path / "nonexistent.ts"
        candidates = [{"start": 1000, "basename": "a"}]
        result = _pick_closest_by_mtime(video, candidates)
        # Fallback: returns first candidate if stat fails
        assert result is not None
        assert result["basename"] == "a"


# -----------------------------------------------------------------------
# review.py — _parse_meta_file
# -----------------------------------------------------------------------

class TestParseMetaFile:
    def test_parses_sections(self, tmp_path: Path):
        content = (
            "# Comment line\n"
            "\n"
            "[metadata]\n"
            "title           = J.A.G.\n"
            "description     = A military drama\n"
            "\n"
            "[episode_match]\n"
            "series          = JAG\n"
            "season          = 3\n"
            "episode         = 12\n"
            "episode_title   = Angst\n"
            "confidence      = 0.850\n"
            "source          = tmdb\n"
        )
        meta = tmp_path / "test.meta"
        meta.write_text(content, encoding="utf-8")
        sections = _parse_meta_file(meta)
        assert "metadata" in sections
        assert sections["metadata"]["title"] == "J.A.G."
        assert "episode_match" in sections
        assert sections["episode_match"]["season"] == "3"
        assert sections["episode_match"]["confidence"] == "0.850"

    def test_handles_comment_after_section(self, tmp_path: Path):
        content = "[subtitles] # 42 lines\nHello World\n"
        meta = tmp_path / "test.meta"
        meta.write_text(content, encoding="utf-8")
        sections = _parse_meta_file(meta)
        assert "subtitles" in sections

    def test_empty_file(self, tmp_path: Path):
        meta = tmp_path / "empty.meta"
        meta.write_text("", encoding="utf-8")
        sections = _parse_meta_file(meta)
        assert sections == {}

    def test_ignores_lines_before_section(self, tmp_path: Path):
        content = "orphan_key = orphan_value\n[section]\nkey = value\n"
        meta = tmp_path / "test.meta"
        meta.write_text(content, encoding="utf-8")
        sections = _parse_meta_file(meta)
        # orphan_key has no section → ignored (current_section is None)
        assert "section" in sections
        assert sections["section"]["key"] == "value"


# -----------------------------------------------------------------------
# review.py — _match_from_meta
# -----------------------------------------------------------------------

class TestMatchFromMeta:
    def test_valid_match(self):
        sections = {
            "episode_match": {
                "series": "JAG",
                "season": "3",
                "episode": "12",
                "episode_title": "Angst",
                "episode_summary": "A story about fear",
                "confidence": "0.850",
                "source": "tmdb",
            }
        }
        match = _match_from_meta(sections)
        assert match is not None
        assert match.series_title == "JAG"
        assert match.season_number == 3
        assert match.episode_number == 12
        assert match.confidence == 0.85
        assert match.source == "tmdb"

    def test_missing_episode_match_section(self):
        assert _match_from_meta({"metadata": {}}) is None

    def test_missing_season_key(self):
        sections = {"episode_match": {"series": "JAG", "episode": "1"}}
        assert _match_from_meta(sections) is None

    def test_invalid_numeric_values(self):
        sections = {"episode_match": {"season": "abc", "episode": "1"}}
        assert _match_from_meta(sections) is None

    def test_empty_episode_match(self):
        assert _match_from_meta({"episode_match": {}}) is None


# -----------------------------------------------------------------------
# main.py — parse_args
# -----------------------------------------------------------------------

class TestParseArgs:
    def test_minimal_args(self):
        args = parse_args(["recognize", "/tmp/recordings"])
        assert args.mode == "recognize"
        assert args.directory == Path("/tmp/recordings")
        assert args.confidence == 0.6
        assert args.dry_run is False
        assert args.language == "de-DE"

    def test_all_flags(self):
        args = parse_args([
            "recognize",
            "/tmp/rec",
            "--dry-run",
            "--verbose",
            "--confidence", "0.8",
            "--language", "en-US",
            "--no-tvmaze",
            "--extensions", ".ts,.mkv",
        ])
        assert args.dry_run is True
        assert args.verbose is True
        assert args.confidence == 0.8
        assert args.language == "en-US"
        assert args.no_tvmaze is True

    def test_env_var_tmdb_key(self):
        with patch.dict(os.environ, {"EPICUR_TMDB_API_KEY": "test_key_123"}):
            args = parse_args(["recognize", "/tmp/rec"])
            assert args.tmdb_api_key == "test_key_123"

    def test_env_var_tvh_credentials(self):
        with patch.dict(os.environ, {
            "EPICUR_TVH_USER": "admin",
            "EPICUR_TVH_PASS": "secret",
        }):
            args = parse_args(["recognize", "/tmp/rec"])
            assert args.tvh_user == "admin"
            assert args.tvh_pass == "secret"

    def test_cli_overrides_env(self):
        with patch.dict(os.environ, {"EPICUR_TMDB_API_KEY": "env_key"}):
            args = parse_args(["recognize", "/tmp/rec", "--tmdb-api-key", "cli_key"])
            assert args.tmdb_api_key == "cli_key"

    def test_review_mode(self):
        args = parse_args(["review", "/tmp/rec", "--dry-run"])
        assert args.mode == "review"
        assert args.directory == Path("/tmp/rec")
        assert args.dry_run is True

    def test_postprocess_mode(self):
        args = parse_args(["postprocess", "/tmp/rec", "--library-dir", "/media/tv"])
        assert args.mode == "postprocess"
        assert args.library_dir == Path("/media/tv")
        assert args.crf == 20
        assert args.preset == "slow"

    def test_no_subcommand_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            parse_args([])
        assert exc_info.value.code == 2


# -----------------------------------------------------------------------
# models.py — ExtractedMetadata.has_useful_data
# -----------------------------------------------------------------------

class TestHasUsefulData:
    def test_empty_metadata(self):
        m = ExtractedMetadata()
        assert m.has_useful_data() is False

    def test_with_title(self):
        assert ExtractedMetadata(title="Show").has_useful_data() is True

    def test_with_description(self):
        assert ExtractedMetadata(description="Something").has_useful_data() is True

    def test_with_tvh_subtitle(self):
        assert ExtractedMetadata(tvh_subtitle="Ep1").has_useful_data() is True

    def test_with_subtitle_texts(self):
        assert ExtractedMetadata(subtitle_texts=["line1"]).has_useful_data() is True

    def test_only_duration_not_useful(self):
        # Duration alone doesn't count as useful
        assert ExtractedMetadata(duration_seconds=3600.0).has_useful_data() is False


# =======================================================================
# MEDIUM-VALUE TESTS
# =======================================================================

# -----------------------------------------------------------------------
# metadata_extractor.py — write_meta_file
# -----------------------------------------------------------------------

class TestWriteMetaFile:
    def test_writes_file(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.touch()
        metadata = ExtractedMetadata(
            title="JAG",
            description="Military drama",
            channel="ZDF",
            duration_seconds=2700.0,
        )
        meta_path = write_meta_file(video, metadata)
        assert meta_path.exists()
        assert meta_path.suffix == ".meta"
        content = meta_path.read_text(encoding="utf-8")
        assert "[metadata]" in content
        assert "title           = JAG" in content
        assert "duration        = 2700.0s" in content

    def test_includes_episode_match(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.touch()
        metadata = ExtractedMetadata(title="JAG")
        match = EpisodeMatch(
            series_title="JAG",
            season_number=3,
            episode_number=12,
            episode_title="Angst",
            confidence=0.85,
            source="tmdb",
        )
        meta_path = write_meta_file(video, metadata, match)
        content = meta_path.read_text(encoding="utf-8")
        assert "[episode_match]" in content
        assert "season          = 3" in content
        assert "episode         = 12" in content
        assert "confidence      = 0.850" in content

    def test_no_match_section(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.touch()
        metadata = ExtractedMetadata(title="JAG")
        meta_path = write_meta_file(video, metadata, None)
        content = meta_path.read_text(encoding="utf-8")
        assert "# No episode match found" in content

    def test_includes_tvh_data(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.touch()
        metadata = ExtractedMetadata(
            title="Test",
            tvh_subtitle="Episode Name",
            tvh_description="A desc",
            tvh_channel="ARD",
            tvh_start=1700000000,
            tvh_stop=1700003600,
        )
        meta_path = write_meta_file(video, metadata)
        content = meta_path.read_text(encoding="utf-8")
        assert "[tvheadend]" in content
        assert "subtitle        = Episode Name" in content
        assert "start           =" in content

    def test_roundtrip_with_parse_meta(self, tmp_path: Path):
        """Write a meta file and parse it back — the match should survive."""
        video = tmp_path / "show.ts"
        video.touch()
        metadata = ExtractedMetadata(title="JAG", tvh_subtitle="Angst")
        match = EpisodeMatch(
            series_title="JAG", season_number=3, episode_number=12,
            episode_title="Angst", confidence=0.85, source="tmdb",
        )
        meta_path = write_meta_file(video, metadata, match)
        sections = _parse_meta_file(meta_path)
        recovered = _match_from_meta(sections)
        assert recovered is not None
        assert recovered.season_number == 3
        assert recovered.episode_number == 12
        assert recovered.confidence == 0.85


# -----------------------------------------------------------------------
# main.py — list_video_files
# -----------------------------------------------------------------------

class TestListVideoFiles:
    def test_filters_by_extension(self, tmp_path: Path):
        (tmp_path / "a.ts").touch()
        (tmp_path / "b.mp4").touch()
        (tmp_path / "c.txt").touch()
        (tmp_path / "d.meta").touch()
        result = list_video_files(tmp_path, {".ts", ".mp4"})
        names = {f.name for f in result}
        assert names == {"a.ts", "b.mp4"}

    def test_sorted_by_mtime(self, tmp_path: Path):
        # Create files with different mtimes
        f1 = tmp_path / "newer.ts"
        f2 = tmp_path / "older.ts"
        f2.write_text("old")
        time.sleep(0.05)
        f1.write_text("new")
        result = list_video_files(tmp_path, {".ts"})
        assert result[0].name == "older.ts"
        assert result[1].name == "newer.ts"

    def test_ignores_directories(self, tmp_path: Path):
        (tmp_path / "subdir.ts").mkdir()
        (tmp_path / "file.ts").touch()
        result = list_video_files(tmp_path, {".ts"})
        assert len(result) == 1
        assert result[0].name == "file.ts"

    def test_empty_directory(self, tmp_path: Path):
        assert list_video_files(tmp_path, {".ts"}) == []

    def test_case_insensitive_extension(self, tmp_path: Path):
        (tmp_path / "show.TS").touch()
        (tmp_path / "show.Mp4").touch()
        result = list_video_files(tmp_path, {".ts", ".mp4"})
        assert len(result) == 2


# -----------------------------------------------------------------------
# review.py — find_unmatched_files
# -----------------------------------------------------------------------

class TestFindUnmatchedFiles:
    def test_finds_video_meta_pairs(self, tmp_path: Path):
        series = tmp_path / "JAG"
        series.mkdir()
        (series / "ep1.ts").touch()
        (series / "ep1.meta").touch()
        (series / "ep2.ts").touch()
        (series / "ep2.meta").touch()
        pairs = find_unmatched_files(tmp_path, {".ts"})
        assert len(pairs) == 2
        assert pairs[0][0].name == "ep1.ts"
        assert pairs[0][1].name == "ep1.meta"

    def test_skips_video_without_meta(self, tmp_path: Path):
        series = tmp_path / "JAG"
        series.mkdir()
        (series / "ep1.ts").touch()
        # No .meta file — should be skipped
        pairs = find_unmatched_files(tmp_path, {".ts"})
        assert len(pairs) == 0

    def test_skips_special_directories(self, tmp_path: Path):
        for name in ("duplicates", "unmatched", ".trash-1000"):
            d = tmp_path / name
            d.mkdir()
            (d / "file.ts").touch()
            (d / "file.meta").touch()
        pairs = find_unmatched_files(tmp_path, {".ts"})
        assert len(pairs) == 0

    def test_filters_by_extension(self, tmp_path: Path):
        series = tmp_path / "Show"
        series.mkdir()
        (series / "ep.ts").touch()
        (series / "ep.meta").touch()
        (series / "notes.txt").touch()
        (series / "notes.meta").touch()
        pairs = find_unmatched_files(tmp_path, {".ts"})
        assert len(pairs) == 1
        assert pairs[0][0].suffix == ".ts"

    def test_multiple_series(self, tmp_path: Path):
        for name in ("AAA", "BBB"):
            d = tmp_path / name
            d.mkdir()
            (d / "ep.ts").touch()
            (d / "ep.meta").touch()
        pairs = find_unmatched_files(tmp_path, {".ts"})
        assert len(pairs) == 2
        # Should be sorted by series name
        assert pairs[0][0].parent.name == "AAA"
        assert pairs[1][0].parent.name == "BBB"


# -----------------------------------------------------------------------
# file_organizer.py — compute_duplicate_path
# -----------------------------------------------------------------------

class TestComputeDuplicatePath:
    def test_basic(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.write_text("content")
        match = EpisodeMatch(series_title="Show", season_number=2, episode_number=5)
        target = compute_duplicate_path(tmp_path, match, video)
        assert "duplicates" in str(target)
        assert "S02E05" in target.name
        assert target.suffix == ".ts"

    def test_contains_timestamp(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.write_text("content")
        match = EpisodeMatch(series_title="Show", season_number=1, episode_number=1)
        target = compute_duplicate_path(tmp_path, match, video)
        # Filename should start with YYYYMMDD_HHMMSS
        import re
        assert re.match(r"\d{8}_\d{6}_", target.name)

    def test_stays_inside_series_dir(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.write_text("content")
        match = EpisodeMatch(series_title="Show", season_number=1, episode_number=1)
        target = compute_duplicate_path(tmp_path, match, video)
        assert target.resolve().is_relative_to(tmp_path.resolve())


# -----------------------------------------------------------------------
# main.py — print_report
# -----------------------------------------------------------------------


class TestPrintReport:
    def test_empty_results(self, capsys):
        print_report([])
        assert capsys.readouterr().out == ""

    def test_mixed_results(self, capsys, tmp_path: Path):
        match = EpisodeMatch(
            series_title="JAG", season_number=1, episode_number=2,
            episode_title="Angst", confidence=0.85, source="tmdb",
        )
        results = [
            OrganizationResult(
                source_path=tmp_path / "a.ts",
                target_path=tmp_path / "Season 1" / "a S01E02.ts",
                action="moved", episode_match=match,
            ),
            OrganizationResult(
                source_path=tmp_path / "b.ts",
                target_path=None, action="skipped",
            ),
        ]
        print_report(results)
        out = capsys.readouterr().out
        assert "REPORT" in out
        assert "Moved (identified)    : 1" in out
        assert "Skipped (unmatched)   : 1" in out
        assert "[MOVED    ]" in out
        assert "[SKIPPED  ]" in out

    def test_with_errors(self, capsys, tmp_path: Path):
        results = [
            OrganizationResult(
                source_path=tmp_path / "c.ts",
                target_path=tmp_path / "Season 1" / "c.ts",
                action="error", error_message="Permission denied",
            ),
        ]
        print_report(results)
        out = capsys.readouterr().out
        assert "[ERROR    ]" in out
        assert "Permission denied" in out

    def test_duplicate(self, capsys, tmp_path: Path):
        match = EpisodeMatch(
            series_title="Show", season_number=2, episode_number=1,
            confidence=0.9, source="tvmaze",
        )
        results = [
            OrganizationResult(
                source_path=tmp_path / "d.ts",
                target_path=tmp_path / "duplicates" / "d S02E01.ts",
                action="duplicate", episode_match=match,
            ),
        ]
        print_report(results)
        out = capsys.readouterr().out
        assert "Duplicates            : 1" in out
        assert "[DUPLICATE]" in out


# -----------------------------------------------------------------------
# file_organizer.py — organize_file (real move, duplicate, error)
# -----------------------------------------------------------------------


class TestOrganizeFileReal:
    def test_move_with_meta_cleanup(self, tmp_path: Path):
        series = tmp_path / "Show"
        series.mkdir()
        video = series / "Show.ts"
        video.write_text("content")
        meta = series / "Show.meta"
        meta.write_text("meta")
        match = EpisodeMatch(series_title="Show", season_number=1, episode_number=1)
        result = organize_file(series, video, match, dry_run=False)
        assert result.action == "moved"
        assert not video.exists()
        assert not meta.exists()  # meta removed on move
        assert result.target_path.exists()

    def test_duplicate_with_meta_move(self, tmp_path: Path):
        series = tmp_path / "Show"
        season = series / "Season 1"
        season.mkdir(parents=True)
        existing = season / "Show S01E01.ts"
        existing.write_text("original")
        video = series / "Show.ts"
        video.write_text("dup content")
        meta = series / "Show.meta"
        meta.write_text("meta")
        match = EpisodeMatch(series_title="Show", season_number=1, episode_number=1)
        result = organize_file(series, video, match, dry_run=False)
        assert result.action == "duplicate"
        assert not video.exists()
        assert result.target_path.exists()
        assert "duplicates" in str(result.target_path)
        # meta file should be moved alongside duplicate
        assert result.target_path.with_suffix(".meta").exists()

    def test_oserror_returns_error(self, tmp_path: Path):
        series = tmp_path / "Show"
        series.mkdir()
        video = series / "Show.ts"
        video.write_text("content")
        match = EpisodeMatch(series_title="Show", season_number=1, episode_number=1)
        with patch("epicur.file_organizer.shutil.move", side_effect=OSError("disk full")):
            result = organize_file(series, video, match, dry_run=False)
            assert result.action == "error"
            assert "disk full" in result.error_message

    def test_source_equals_target_skipped(self, tmp_path: Path):
        series = tmp_path / "Show"
        season = series / "Season 1"
        season.mkdir(parents=True)
        # _base_stem("Show.ts") → "Show", target → "Season 1/Show S01E01.ts"
        # Place a file at exactly that computed target path
        video = season / "Show S01E01.ts"
        video.write_text("content")
        # To make source == target, use a file named "Show-1.ts" (TVH suffix)
        # or just directly create the scenario via a symlink.
        # Simplest: source file IS the target (same path). That needs
        # _base_stem to produce exactly "Show S01E01" already — but then target
        # becomes "Show S01E01 S01E01.ts". So instead, mock compute_target_path.
        match = EpisodeMatch(series_title="Show", season_number=1, episode_number=1)
        with patch("epicur.file_organizer.compute_target_path", return_value=video), \
             patch("epicur.file_organizer.compute_duplicate_path", return_value=video):
            result = organize_file(series, video, match, dry_run=False)
            assert result.action == "skipped"
            assert video.exists()


# -----------------------------------------------------------------------
# episode_identifier.py — _rate_limit, _http_get_json
# -----------------------------------------------------------------------


class TestRateLimit:
    def test_under_limit_no_sleep(self):
        from epicur.episode_identifier import _rate_limit, _last_request_times
        _last_request_times.clear()
        with patch("epicur.episode_identifier.time.sleep") as mock_sleep:
            _rate_limit()
            mock_sleep.assert_not_called()
        _last_request_times.clear()

    def test_over_limit_sleeps(self):
        from epicur.episode_identifier import (
            _rate_limit, _last_request_times,
            RATE_LIMIT_MAX, RATE_LIMIT_WINDOW,
        )
        _last_request_times.clear()
        now = time.monotonic()
        # Fill with RATE_LIMIT_MAX timestamps all within the window
        _last_request_times.extend(now - 1.0 for _ in range(RATE_LIMIT_MAX))
        with patch("epicur.episode_identifier.time.sleep") as mock_sleep:
            _rate_limit()
            mock_sleep.assert_called_once()
            # Sleep time should be roughly WINDOW - 1.0
            assert mock_sleep.call_args[0][0] > 0
        _last_request_times.clear()


class TestHttpGetJson:
    def test_success(self):
        from epicur.episode_identifier import _http_get_json, _last_request_times
        _last_request_times.clear()
        mock_resp = patch("urllib.request.urlopen").__enter__()
        mock_resp.return_value.__enter__ = lambda s: s
        mock_resp.return_value.__exit__ = lambda s, *a: None
        mock_resp.return_value.read.return_value = b'{"id": 1}'
        try:
            result = _http_get_json("https://example.com/api")
            assert result == {"id": 1}
        finally:
            patch.stopall()
            _last_request_times.clear()

    def test_http_error_returns_none(self):
        import urllib.error
        from epicur.episode_identifier import _http_get_json, _last_request_times
        _last_request_times.clear()
        with patch("urllib.request.urlopen",
                    side_effect=urllib.error.HTTPError("url", 404, "Not Found", {}, None)):
            result = _http_get_json("https://example.com/api")
            assert result is None
        _last_request_times.clear()

    def test_connection_error_returns_none(self):
        from epicur.episode_identifier import _http_get_json, _last_request_times
        _last_request_times.clear()
        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            result = _http_get_json("https://example.com/api")
            assert result is None
        _last_request_times.clear()

    def test_params_appended(self):
        from epicur.episode_identifier import _http_get_json, _last_request_times
        _last_request_times.clear()
        mock_resp = patch("urllib.request.urlopen").__enter__()
        mock_resp.return_value.__enter__ = lambda s: s
        mock_resp.return_value.__exit__ = lambda s, *a: None
        mock_resp.return_value.read.return_value = b'{"ok": true}'
        try:
            result = _http_get_json("https://example.com/api", params={"q": "test show"})
            assert result == {"ok": True}
            # Verify the URL was constructed with params
            req = mock_resp.call_args[0][0]
            assert "q=test+show" in req.full_url
        finally:
            patch.stopall()
            _last_request_times.clear()


# ---------------------------------------------------------------------------
# Min-age recording filter
# ---------------------------------------------------------------------------
class TestMinAgeFilter:
    """Tests for the min_age parameter that skips still-recording files."""

    def _make_series(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create a root with one series folder containing one video file."""
        root = tmp_path / "recordings"
        series = root / "TestShow"
        series.mkdir(parents=True)
        video = series / "episode.ts"
        video.write_text("data")
        return root, video

    def test_recent_file_skipped_as_recording(self, tmp_path: Path):
        """A file with mtime = now should be skipped when min_age > 0."""
        root, video = self._make_series(tmp_path)
        # File was just created → age ≈ 0s, well below 300s
        results = process_directory(root, min_age=300, dry_run=True)
        assert len(results) == 1
        assert results[0].action == "recording"
        assert results[0].source_path == video

    def test_old_file_processed(self, tmp_path: Path):
        """A file older than min_age should be processed normally."""
        import os, time as _time
        root, video = self._make_series(tmp_path)
        # Backdate mtime by 10 minutes
        old_time = _time.time() - 600
        os.utime(video, (old_time, old_time))
        with patch("epicur.main.extract_all_metadata") as mock_meta, \
             patch("epicur.main.identify_episode", return_value=None), \
             patch("epicur.main.write_meta_file"), \
             patch("epicur.main.organize_file") as mock_org:
            mock_meta.return_value = MagicMock()
            mock_org.return_value = OrganizationResult(
                source_path=video, target_path=video, action="skipped",
            )
            results = process_directory(root, min_age=300, dry_run=True)
        assert len(results) == 1
        assert results[0].action != "recording"

    def test_min_age_zero_disables_check(self, tmp_path: Path):
        """min_age=0 should disable the recording check entirely."""
        root, video = self._make_series(tmp_path)
        # File is brand new, but min_age=0 disables the check
        with patch("epicur.main.extract_all_metadata") as mock_meta, \
             patch("epicur.main.identify_episode", return_value=None), \
             patch("epicur.main.write_meta_file"), \
             patch("epicur.main.organize_file") as mock_org:
            mock_meta.return_value = MagicMock()
            mock_org.return_value = OrganizationResult(
                source_path=video, target_path=video, action="skipped",
            )
            results = process_directory(root, min_age=0, dry_run=True)
        assert len(results) == 1
        assert results[0].action != "recording"

    def test_report_shows_recording_count(self, capsys):
        """print_report should display recording count."""
        results = [
            OrganizationResult(source_path=Path("a.ts"), target_path=None, action="recording"),
            OrganizationResult(source_path=Path("b.ts"), target_path=None, action="recording"),
        ]
        print_report(results)
        out = capsys.readouterr().out
        assert "Recording (skipped)   : 2" in out
        assert "[RECORDING]" in out
        assert "a.ts" in out
