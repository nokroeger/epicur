"""Integration tests for epicur — mock-based tests for network, subprocess, and interactive functions."""
from __future__ import annotations

import json
import subprocess
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from epicur.models import EpisodeMatch, ExtractedMetadata
from epicur.metadata_extractor import (
    extract_all_metadata,
    extract_ffprobe_metadata,
    extract_subtitle_text,
)

# ═══════════════════════════════════════════════════════════════════════
# Fixtures: clear module-level caches between tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _clear_caches():
    """Reset all module-level caches before every test."""
    from epicur import episode_identifier as ei

    ei._tvmaze_cache.clear()
    ei._tvmaze_show_id_cache.clear()
    ei._tmdb_show_cache.clear()
    ei._tmdb_episode_cache.clear()
    ei._last_request_times.clear()
    yield


# ═══════════════════════════════════════════════════════════════════════
# Mock data constants
# ═══════════════════════════════════════════════════════════════════════

# -- TVMaze --
TVMAZE_SHOW = {"id": 123, "name": "J.A.G."}
TVMAZE_EPISODES = [
    {"season": 1, "number": 1, "name": "Pilot", "summary": "<p>The beginning.</p>", "runtime": 45},
    {"season": 1, "number": 2, "name": "Angst", "summary": "<p>Fear takes over.</p>", "runtime": 45},
    {"season": 2, "number": 1, "name": "Rückkehr", "summary": "<p>They return.</p>", "runtime": 45},
]
TVMAZE_FUZZY = [
    {"score": 15.2, "show": {"id": 123, "name": "J.A.G."}},
    {"score": 8.1, "show": {"id": 999, "name": "Something Else"}},
]

# -- TMDB --
TMDB_SEARCH = {"results": [{"id": 456, "name": "J.A.G."}]}
TMDB_DETAIL = {"number_of_seasons": 2}
TMDB_S1 = {"episodes": [
    {"episode_number": 1, "name": "Pilot", "overview": "The beginning", "runtime": 45},
    {"episode_number": 2, "name": "Angst", "overview": "Fear takes over", "runtime": 45},
]}
TMDB_S2 = {"episodes": [
    {"episode_number": 1, "name": "Rückkehr", "overview": "They return", "runtime": 45},
]}

# -- ffprobe --
FFPROBE_JSON = json.dumps({
    "format": {
        "duration": "2700.5",
        "tags": {
            "title": "J.A.G. - Im Auftrag der Ehre",
            "DESCRIPTION": "Ein Militärdrama",
            "service_provider": "ZDF",
            "episode_id": "Angst",
        },
    },
    "streams": [
        {"codec_type": "video", "tags": {}},
        {"codec_type": "audio", "tags": {"language": "ger"}},
    ],
})

SRT_OUTPUT = (
    "1\n"
    "00:00:01,000 --> 00:00:03,000\n"
    "Willkommen bei JAG.\n"
    "\n"
    "2\n"
    "00:00:05,000 --> 00:00:08,000\n"
    "Ein Militärdrama.\n"
    "\n"
)


# ═══════════════════════════════════════════════════════════════════════
# 1. API CLIENT TESTS
# ═══════════════════════════════════════════════════════════════════════

# -----------------------------------------------------------------------
# TVMaze
# -----------------------------------------------------------------------


class TestSearchTvmaze:
    def test_exact_match(self):
        from epicur.episode_identifier import search_tvmaze

        with patch("epicur.episode_identifier._http_get_json") as m:
            m.return_value = TVMAZE_SHOW
            assert search_tvmaze("J.A.G.") == 123
            m.assert_called_once()

    def test_fuzzy_fallback(self):
        from epicur.episode_identifier import search_tvmaze

        with patch("epicur.episode_identifier._http_get_json") as m:
            m.side_effect = [None, TVMAZE_FUZZY]
            assert search_tvmaze("JAG Im Auftrag der Ehre") == 123
            assert m.call_count == 2

    def test_not_found_cached(self):
        from epicur.episode_identifier import search_tvmaze

        with patch("epicur.episode_identifier._http_get_json") as m:
            m.return_value = None
            assert search_tvmaze("Nonexistent") is None
            assert search_tvmaze("Nonexistent") is None
            # Only 2 calls for the first attempt (single + fuzzy), none for cache hit
            assert m.call_count == 2


class TestGetTvmazeEpisodes:
    def test_fetches_and_caches(self):
        from epicur.episode_identifier import get_tvmaze_episodes

        with patch("epicur.episode_identifier._http_get_json") as m:
            m.return_value = TVMAZE_EPISODES
            assert len(get_tvmaze_episodes(123)) == 3
            assert len(get_tvmaze_episodes(123)) == 3
            m.assert_called_once()

    def test_non_list_response(self):
        from epicur.episode_identifier import get_tvmaze_episodes

        with patch("epicur.episode_identifier._http_get_json") as m:
            m.return_value = {"error": "not found"}
            assert get_tvmaze_episodes(999) == []


class TestMatchEpisodeTvmaze:
    def test_end_to_end(self):
        from epicur.episode_identifier import match_episode_tvmaze

        meta = ExtractedMetadata(tvh_subtitle="Angst", description="Fear takes over")
        with patch("epicur.episode_identifier._http_get_json") as m:
            m.side_effect = [TVMAZE_SHOW, TVMAZE_EPISODES]
            result = match_episode_tvmaze("J.A.G.", meta)
            assert result is not None
            assert result.episode_title == "Angst"
            assert result.season_number == 1
            assert result.episode_number == 2
            assert result.source == "tvmaze"

    def test_show_not_found(self):
        from epicur.episode_identifier import match_episode_tvmaze

        meta = ExtractedMetadata(tvh_subtitle="Angst")
        with patch("epicur.episode_identifier._http_get_json") as m:
            m.return_value = None
            assert match_episode_tvmaze("Unknown", meta) is None


# -----------------------------------------------------------------------
# TMDB
# -----------------------------------------------------------------------


class TestSearchTmdb:
    def test_found(self):
        from epicur.episode_identifier import search_tmdb

        with patch("epicur.episode_identifier._http_get_json") as m:
            m.return_value = TMDB_SEARCH
            assert search_tmdb("J.A.G.", "key") == 456

    def test_not_found(self):
        from epicur.episode_identifier import search_tmdb

        with patch("epicur.episode_identifier._http_get_json") as m:
            m.return_value = {"results": []}
            assert search_tmdb("Unknown", "key") is None

    def test_caches_result(self):
        from epicur.episode_identifier import search_tmdb

        with patch("epicur.episode_identifier._http_get_json") as m:
            m.return_value = TMDB_SEARCH
            search_tmdb("J.A.G.", "key")
            search_tmdb("J.A.G.", "key")
            m.assert_called_once()


class TestGetAllTmdbEpisodes:
    def test_fetches_all_seasons(self):
        from epicur.episode_identifier import get_all_tmdb_episodes

        with patch("epicur.episode_identifier._http_get_json") as m:
            m.side_effect = [TMDB_DETAIL, TMDB_S1, TMDB_S2]
            eps = get_all_tmdb_episodes(456, "key")
            assert len(eps) == 3
            assert eps[0]["_season"] == 1
            assert eps[2]["_season"] == 2


class TestMatchEpisodeTmdb:
    def test_end_to_end(self):
        from epicur.episode_identifier import match_episode_tmdb

        meta = ExtractedMetadata(tvh_subtitle="Angst", description="Fear takes over")
        with patch("epicur.episode_identifier._http_get_json") as m:
            m.side_effect = [TMDB_SEARCH, TMDB_DETAIL, TMDB_S1, TMDB_S2]
            result = match_episode_tmdb("J.A.G.", meta, "key")
            assert result is not None
            assert result.episode_title == "Angst"
            assert result.source == "tmdb"


# -----------------------------------------------------------------------
# TVH direct match
# -----------------------------------------------------------------------


class TestTvhDirectMatch:
    def test_tvmaze_path(self):
        from epicur.episode_identifier import _tvh_direct_match

        with patch("epicur.episode_identifier._http_get_json") as m:
            m.side_effect = [TVMAZE_SHOW, TVMAZE_EPISODES]
            result = _tvh_direct_match("J.A.G.", "Angst", use_tvmaze=True)
            assert result is not None
            assert result.source == "tvh+tvmaze"
            assert result.confidence >= 0.80

    def test_tmdb_path(self):
        from epicur.episode_identifier import _tvh_direct_match

        with patch("epicur.episode_identifier._http_get_json") as m:
            m.side_effect = [TMDB_SEARCH, TMDB_DETAIL, TMDB_S1, TMDB_S2]
            result = _tvh_direct_match("J.A.G.", "Angst", tmdb_api_key="key", use_tvmaze=False)
            assert result is not None
            assert result.source == "tvh+tmdb"

    def test_no_match(self):
        from epicur.episode_identifier import _tvh_direct_match

        with patch("epicur.episode_identifier._http_get_json") as m:
            m.side_effect = [TVMAZE_SHOW, TVMAZE_EPISODES]
            assert _tvh_direct_match("J.A.G.", "Completely Unknown Episode", use_tvmaze=True) is None


# -----------------------------------------------------------------------
# identify_episode — full pipeline
# -----------------------------------------------------------------------


class TestIdentifyEpisode:
    def test_filename_fallback(self):
        """When no APIs are used, filename pattern is the fallback."""
        from epicur.episode_identifier import identify_episode

        meta = ExtractedMetadata()
        result = identify_episode("Show", meta, Path("/rec/Show/Show S03E12.ts"), use_tvmaze=False)
        assert result is not None
        assert result.source == "filename"
        assert result.season_number == 3
        assert result.episode_number == 12

    def test_no_match_at_all(self):
        from epicur.episode_identifier import identify_episode

        result = identify_episode("Show", ExtractedMetadata(), Path("/rec/Show/unknown.ts"), use_tvmaze=False)
        assert result is None

    def test_api_exception_falls_through(self):
        """If TVMaze raises, filename fallback should still work."""
        from epicur.episode_identifier import identify_episode

        with patch("epicur.episode_identifier.match_episode_tvmaze", side_effect=RuntimeError("conn")):
            result = identify_episode("Show", ExtractedMetadata(), Path("/rec/Show/Show S01E01.ts"), use_tvmaze=True)
            assert result is not None
            assert result.source == "filename"

    def test_tvmaze_wins_over_filename(self):
        """A high-confidence API match should be returned over filename."""
        from epicur.episode_identifier import identify_episode

        high_conf = EpisodeMatch(
            series_title="J.A.G.", season_number=1, episode_number=2,
            episode_title="Angst", confidence=0.85, source="tvmaze",
        )
        with patch("epicur.episode_identifier.match_episode_tvmaze", return_value=high_conf):
            result = identify_episode(
                "J.A.G.", ExtractedMetadata(),
                Path("/rec/JAG/JAG S01E02.ts"), use_tvmaze=True,
            )
            assert result is not None
            assert result.source == "tvmaze"
            assert result.confidence == 0.85


# -----------------------------------------------------------------------
# fetch_dvr_entries_api
# -----------------------------------------------------------------------


class TestFetchDvrEntriesApi:
    def _mock_response(self, data: bytes):
        resp = MagicMock()
        resp.read.return_value = data
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def test_success(self):
        from epicur.tvheadend_client import fetch_dvr_entries_api

        payload = json.dumps({"entries": [
            {"filename": "/rec/show.ts", "title": "Show", "subtitle": "Ep1",
             "description": "", "channelname": "ARD", "start": 1000, "stop": 2000},
        ]}).encode()
        with patch("urllib.request.urlopen", return_value=self._mock_response(payload)):
            entries = fetch_dvr_entries_api("http://localhost:9981")
            assert len(entries) == 1
            assert entries[0]["title"] == "Show"

    def test_basic_auth(self):
        from epicur.tvheadend_client import fetch_dvr_entries_api

        payload = json.dumps({"entries": []}).encode()
        with patch("urllib.request.urlopen", return_value=self._mock_response(payload)) as mock_open:
            fetch_dvr_entries_api("http://localhost:9981", "admin", "secret")
            req = mock_open.call_args[0][0]
            assert req.get_header("Authorization").startswith("Basic ")

    def test_http_error(self):
        import urllib.error
        from epicur.tvheadend_client import fetch_dvr_entries_api

        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("http://x", 401, "Unauth", {}, None)):
            assert fetch_dvr_entries_api("http://localhost:9981", "bad", "creds") == []

    def test_connection_error(self):
        from epicur.tvheadend_client import fetch_dvr_entries_api

        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            assert fetch_dvr_entries_api("http://localhost:9981") == []


# ═══════════════════════════════════════════════════════════════════════
# 2. FFPROBE / FFMPEG EXTRACTION TESTS
# ═══════════════════════════════════════════════════════════════════════


def _ffprobe_ok(stdout: str = FFPROBE_JSON) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _run_fail() -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error")


class TestExtractFfprobeMetadata:
    def test_successful_extraction(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.touch()
        with patch("epicur.metadata_extractor.subprocess.run", return_value=_ffprobe_ok()):
            meta = extract_ffprobe_metadata(video)
            assert meta.title == "J.A.G. - Im Auftrag der Ehre"
            assert meta.description == "Ein Militärdrama"
            assert meta.channel == "ZDF"
            assert meta.embedded_episode_title == "Angst"
            assert meta.duration_seconds == pytest.approx(2700.5, abs=0.1)

    def test_not_found(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.touch()
        with patch("epicur.metadata_extractor.subprocess.run", side_effect=FileNotFoundError):
            meta = extract_ffprobe_metadata(video)
            assert meta.title == ""
            assert meta.duration_seconds == 0.0

    def test_timeout(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.touch()
        with patch("epicur.metadata_extractor.subprocess.run",
                    side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=60)):
            assert extract_ffprobe_metadata(video).title == ""

    def test_nonzero_exit(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.touch()
        with patch("epicur.metadata_extractor.subprocess.run", return_value=_run_fail()):
            assert extract_ffprobe_metadata(video).title == ""

    def test_invalid_json(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.touch()
        with patch("epicur.metadata_extractor.subprocess.run",
                    return_value=_ffprobe_ok("not json {{{")):
            assert extract_ffprobe_metadata(video).title == ""

    def test_stream_tags_fallback(self, tmp_path: Path):
        """When format tags are empty, stream tags should be used."""
        video = tmp_path / "show.ts"
        video.touch()
        probe = json.dumps({
            "format": {"tags": {}},
            "streams": [{"codec_type": "video", "tags": {
                "title": "From Stream", "description": "Stream desc",
            }}],
        })
        with patch("epicur.metadata_extractor.subprocess.run", return_value=_ffprobe_ok(probe)):
            meta = extract_ffprobe_metadata(video)
            assert meta.title == "From Stream"
            assert meta.description == "Stream desc"


class TestExtractSubtitleText:
    def test_srt_parsing(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.touch()
        with patch("epicur.metadata_extractor.subprocess.run",
                    return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=SRT_OUTPUT, stderr="")):
            lines = extract_subtitle_text(video)
            assert "Willkommen bei JAG." in lines
            assert "Ein Militärdrama." in lines
            assert not any("-->" in l for l in lines)
            assert not any(l.strip().isdigit() for l in lines)

    def test_fallback_to_data_stream(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.touch()
        ok = subprocess.CompletedProcess(args=[], returncode=0, stdout=SRT_OUTPUT, stderr="")
        with patch("epicur.metadata_extractor.subprocess.run", side_effect=[_run_fail(), ok]):
            lines = extract_subtitle_text(video)
            assert len(lines) == 2

    def test_ffmpeg_not_found(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.touch()
        with patch("epicur.metadata_extractor.subprocess.run", side_effect=FileNotFoundError):
            assert extract_subtitle_text(video) == []

    def test_no_subtitles(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.touch()
        with patch("epicur.metadata_extractor.subprocess.run", return_value=_run_fail()):
            assert extract_subtitle_text(video) == []


class TestExtractAllMetadata:
    def _run_side(self, *args, **kwargs):
        """Return ffprobe OK for ffprobe calls, fail for ffmpeg (subtitles)."""
        cmd = args[0] if args else kwargs.get("args", [])
        if cmd and cmd[0] == "ffprobe":
            return _ffprobe_ok()
        return _run_fail()

    def test_combines_ffprobe_and_tvh(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.touch()
        tvh = {"subtitle": "Angst", "description": "TVH desc", "channel": "RTL", "start": 1000, "stop": 2000}
        with patch("epicur.metadata_extractor.subprocess.run", side_effect=self._run_side):
            meta = extract_all_metadata(video, tvh_entry=tvh)
            assert meta.title == "J.A.G. - Im Auftrag der Ehre"
            assert meta.tvh_subtitle == "Angst"
            assert meta.tvh_description == "TVH desc"
            # ffprobe provides channel (ZDF), so TVH doesn't override
            assert meta.channel == "ZDF"

    def test_tvh_supplements_empty_fields(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.touch()
        empty = json.dumps({"format": {"tags": {}}, "streams": []})
        tvh = {"subtitle": "Ep1", "description": "TVH desc", "channel": "ARD", "start": 1000, "stop": 2000}

        def side(*a, **kw):
            cmd = a[0] if a else kw.get("args", [])
            if cmd and cmd[0] == "ffprobe":
                return _ffprobe_ok(empty)
            return _run_fail()

        with patch("epicur.metadata_extractor.subprocess.run", side_effect=side):
            meta = extract_all_metadata(video, tvh_entry=tvh)
            assert meta.channel == "ARD"
            assert meta.description == "TVH desc"

    def test_without_tvh(self, tmp_path: Path):
        video = tmp_path / "show.ts"
        video.touch()
        with patch("epicur.metadata_extractor.subprocess.run", side_effect=self._run_side):
            meta = extract_all_metadata(video, tvh_entry=None)
            assert meta.title == "J.A.G. - Im Auftrag der Ehre"
            assert meta.tvh_subtitle == ""


# ═══════════════════════════════════════════════════════════════════════
# 3. END-TO-END PIPELINE — process_directory
# ═══════════════════════════════════════════════════════════════════════


def _subprocess_side(*args, **kwargs):
    """Default subprocess side-effect: ffprobe returns empty, ffmpeg fails."""
    cmd = args[0] if args else kwargs.get("args", [])
    if cmd and cmd[0] == "ffprobe":
        return _ffprobe_ok(json.dumps({"format": {"tags": {}}, "streams": []}))
    return _run_fail()


class TestProcessDirectory:
    def test_dry_run(self, tmp_path: Path):
        from epicur.main import process_directory

        root = tmp_path / "recordings"
        show = root / "TestShow"
        show.mkdir(parents=True)
        (show / "TestShow S03E12.ts").write_text("x")

        with patch("epicur.metadata_extractor.subprocess.run", side_effect=_subprocess_side):
            results = process_directory(root, dry_run=True, use_tvmaze=False, min_age=0)
            moved = [r for r in results if r.action == "moved"]
            assert len(moved) == 1
            assert "S03E12" in moved[0].target_path.name
            # File should still exist (dry-run)
            assert (show / "TestShow S03E12.ts").exists()

    def test_confidence_threshold_skips(self, tmp_path: Path):
        from epicur.main import process_directory

        root = tmp_path / "recordings"
        show = root / "Show"
        show.mkdir(parents=True)
        (show / "Show S01E01.ts").write_text("x")

        with patch("epicur.metadata_extractor.subprocess.run", side_effect=_subprocess_side):
            results = process_directory(root, min_confidence=0.9, use_tvmaze=False, dry_run=True, min_age=0)
            assert all(r.action == "skipped" for r in results)
            # .meta should still be written
            assert (show / "Show S01E01.meta").exists()

    def test_empty_directory(self, tmp_path: Path):
        from epicur.main import process_directory

        root = tmp_path / "empty"
        root.mkdir()
        assert process_directory(root) == []

    def test_nonexistent_directory(self, tmp_path: Path):
        from epicur.main import process_directory

        assert process_directory(tmp_path / "nope") == []

    def test_with_tvh_entries(self, tmp_path: Path):
        from epicur.main import process_directory

        root = tmp_path / "recordings"
        show = root / "JAG"
        show.mkdir(parents=True)
        video = show / "JAG.ts"
        video.write_text("x")

        tvh = [{
            "filename": str(video), "basename": "JAG.ts",
            "title": "JAG", "subtitle": "Angst", "description": "Drama",
            "directory": "", "channel": "ZDF",
            "start": int(video.stat().st_mtime), "stop": int(video.stat().st_mtime) + 3600,
            "duration": 3600.0, "file_size": 100, "_raw": {},
        }]

        with patch("epicur.metadata_extractor.subprocess.run", side_effect=_subprocess_side):
            process_directory(root, dry_run=True, use_tvmaze=False, tvh_entries=tvh, min_age=0)
            meta = show / "JAG.meta"
            assert meta.exists()
            assert "Angst" in meta.read_text(encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════
# 4. INTERACTIVE REVIEW TESTS
# ═══════════════════════════════════════════════════════════════════════


def _create_review_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a series dir with a video + meta file for review testing."""
    root = tmp_path / "recordings"
    series = root / "JAG"
    series.mkdir(parents=True)
    video = series / "JAG.ts"
    video.write_text("content")
    meta = series / "JAG.meta"
    meta.write_text(
        "# Metadata\n\n"
        "[metadata]\n"
        "title           = J.A.G.\n"
        "description     = Military drama\n"
        "channel         = ZDF\n"
        "episode_title   =\n"
        "duration        = 2700.0s\n\n"
        "[episode_match]\n"
        "series          = JAG\n"
        "season          = 3\n"
        "episode         = 12\n"
        "episode_title   = Angst\n"
        "episode_summary = Fear\n"
        "confidence      = 0.750\n"
        "source          = tmdb\n\n"
        "[tvheadend]\n"
        "subtitle        = Angst\n"
        "description     =\n"
        "channel         = ZDF\n",
        encoding="utf-8",
    )
    return root, series, video


class TestReviewUnmatched:
    def test_accept_dry_run(self, tmp_path: Path):
        from epicur.review import review_unmatched

        root, series, video = _create_review_fixture(tmp_path)
        with patch("builtins.input", return_value="a"), patch("sys.stdout", new_callable=StringIO):
            review_unmatched(root, {".ts"}, dry_run=True)
        assert video.exists()

    def test_skip(self, tmp_path: Path, capsys):
        from epicur.review import review_unmatched

        root, series, video = _create_review_fixture(tmp_path)
        with patch("builtins.input", return_value="s"):
            review_unmatched(root, {".ts"}, dry_run=True)
        assert "Übersprungen" in capsys.readouterr().out

    def test_quit(self, tmp_path: Path, capsys):
        from epicur.review import review_unmatched

        root, series, video = _create_review_fixture(tmp_path)
        with patch("builtins.input", return_value="q"):
            review_unmatched(root, {".ts"})
        out = capsys.readouterr().out
        assert "Verbleibend" in out or "Akzeptiert" in out

    def test_override(self, tmp_path: Path):
        from epicur.review import review_unmatched

        root, series, video = _create_review_fixture(tmp_path)
        with patch("builtins.input", side_effect=["ü", "2 5"]), patch("sys.stdout", new_callable=StringIO):
            review_unmatched(root, {".ts"}, dry_run=True)
        assert video.exists()

    def test_eof_graceful(self, tmp_path: Path, capsys):
        from epicur.review import review_unmatched

        root, series, video = _create_review_fixture(tmp_path)
        with patch("builtins.input", side_effect=EOFError):
            review_unmatched(root, {".ts"})
        assert "Abgebrochen" in capsys.readouterr().out

    def test_no_unmatched(self, tmp_path: Path, capsys):
        from epicur.review import review_unmatched

        root = tmp_path / "empty"
        root.mkdir()
        review_unmatched(root, {".ts"})
        assert "Keine unbearbeiteten" in capsys.readouterr().out

    def test_actual_move_on_accept(self, tmp_path: Path):
        from epicur.review import review_unmatched

        root, series, video = _create_review_fixture(tmp_path)
        with patch("builtins.input", return_value="a"), patch("sys.stdout", new_callable=StringIO):
            review_unmatched(root, {".ts"}, dry_run=False)
        assert not video.exists()
        season_dir = series / "Season 3"
        assert season_dir.exists()
        moved = list(season_dir.iterdir())
        assert any("S03E12" in f.name for f in moved)


class TestPromptOverride:
    def test_space_format(self):
        from epicur.review import _prompt_override

        with patch("builtins.input", return_value="3 12"):
            result = _prompt_override("JAG", None)
            assert result is not None
            assert result.season_number == 3
            assert result.episode_number == 12
            assert result.confidence == 1.0
            assert result.source == "manual"

    def test_sxxexx_format(self):
        from epicur.review import _prompt_override

        with patch("builtins.input", return_value="S03E12"):
            r = _prompt_override("JAG", None)
            assert r is not None and r.season_number == 3 and r.episode_number == 12

    def test_slash_format(self):
        from epicur.review import _prompt_override

        with patch("builtins.input", return_value="3/12"):
            r = _prompt_override("JAG", None)
            assert r is not None and r.season_number == 3 and r.episode_number == 12

    def test_empty_cancels(self):
        from epicur.review import _prompt_override

        with patch("builtins.input", return_value=""):
            assert _prompt_override("JAG", None) is None

    def test_invalid_format(self):
        from epicur.review import _prompt_override

        with patch("builtins.input", return_value="abc"):
            assert _prompt_override("JAG", None) is None

    def test_eof_cancels(self):
        from epicur.review import _prompt_override

        with patch("builtins.input", side_effect=EOFError):
            assert _prompt_override("JAG", None) is None

    def test_preserves_existing_title(self):
        from epicur.review import _prompt_override

        existing = EpisodeMatch(
            series_title="JAG", season_number=3, episode_number=12,
            episode_title="Angst", confidence=0.75, source="tmdb",
        )
        with patch("builtins.input", return_value="3 12"):
            result = _prompt_override("JAG", existing)
            assert result is not None
            assert result.episode_title == "Angst"


# ═══════════════════════════════════════════════════════════════════════
# 5. MAIN CLI (main() orchestration)
# ═══════════════════════════════════════════════════════════════════════


class TestMainCli:
    def test_normal_run(self, tmp_path: Path):
        from epicur.main import main

        root = tmp_path / "recordings"
        show = root / "Show"
        show.mkdir(parents=True)
        (show / "Show S01E01.ts").write_text("x")

        with patch("epicur.metadata_extractor.subprocess.run", side_effect=_subprocess_side):
            rc = main(["recognize", "--dry-run", "--no-tvmaze", str(root)])
            assert rc == 0

    def test_review_mode(self, tmp_path: Path):
        from epicur.main import main

        root = tmp_path / "recordings"
        root.mkdir()
        with patch("epicur.main.review_unmatched") as mock_review:
            rc = main(["review", "--dry-run", str(root)])
            assert rc == 0
            mock_review.assert_called_once()
            _, kwargs = mock_review.call_args
            assert kwargs["dry_run"] is True

    def test_tvh_api_fallback(self, tmp_path: Path):
        from epicur.main import main

        root = tmp_path / "recordings"
        root.mkdir()
        # Point --tvh-dvr-log to a non-existent dir so API fallback triggers
        fake_log = tmp_path / "no_dvr_log"
        with patch("epicur.main.fetch_dvr_entries_api", return_value=[]) as mock_api, \
             patch("epicur.main.process_directory", return_value=[]):
            rc = main([
                "recognize",
                "--no-tvmaze", "--dry-run",
                "--tvh-dvr-log", str(fake_log),
                "--tvh-url", "http://localhost:9981",
                "--tvh-user", "admin", "--tvh-pass", "pass",
                str(root),
            ])
            assert rc == 0
            mock_api.assert_called_once_with("http://localhost:9981", "admin", "pass")

    def test_error_results_return_1(self, tmp_path: Path):
        from epicur.main import main
        from epicur.models import OrganizationResult

        root = tmp_path / "recordings"
        root.mkdir()
        err = OrganizationResult(
            source_path=tmp_path / "x.ts", target_path=None,
            action="error", error_message="fail",
        )
        with patch("epicur.main.process_directory", return_value=[err]):
            rc = main(["recognize", "--no-tvmaze", "--dry-run", str(root)])
            assert rc == 1

    def test_tvmaze_disabled_for_german(self, tmp_path: Path):
        from epicur.main import main

        root = tmp_path / "recordings"
        root.mkdir()
        with patch("epicur.main.process_directory", return_value=[]) as mock_proc:
            main(["recognize", "--language", "de-DE", str(root)])
            _, kwargs = mock_proc.call_args
            assert kwargs["use_tvmaze"] is False

    def test_postprocess_mode_args(self, tmp_path: Path):
        from epicur.main import main

        root = tmp_path / "recordings"
        root.mkdir()
        lib = tmp_path / "library"
        lib.mkdir()

        with patch("epicur.postprocess.postprocess_all", return_value=[]) as mock_pp:
            rc = main([
                "postprocess", str(root),
                "--library-dir", str(lib),
                "--crf", "18", "--preset", "medium",
                "--no-tvmaze",
            ])
            assert rc == 0
            mock_pp.assert_called_once()
            _, kwargs = mock_pp.call_args
            assert kwargs["crf"] == 18
            assert kwargs["preset"] == "medium"
            assert kwargs["library_dir"] == lib

    def test_postprocess_library_same_as_root_warns(self, tmp_path: Path, caplog):
        from epicur.main import main
        import logging

        root = tmp_path / "recordings"
        root.mkdir()

        with patch("epicur.postprocess.postprocess_all", return_value=[]), \
             caplog.at_level(logging.WARNING):
            main(["postprocess", str(root), "--library-dir", str(root)])
            assert "same as the recordings directory" in caplog.text


# ═══════════════════════════════════════════════════════════════════════
# 6. EDGE CASES — remaining uncovered paths
# ═══════════════════════════════════════════════════════════════════════


class TestFfprobeDurationEdge:
    def test_invalid_duration_parsed_gracefully(self, tmp_path: Path):
        """Duration value that can't be parsed as float should not crash."""
        video = tmp_path / "show.ts"
        video.touch()
        probe = json.dumps({
            "format": {"duration": "invalid", "tags": {}},
            "streams": [],
        })
        with patch("epicur.metadata_extractor.subprocess.run", return_value=_ffprobe_ok(probe)):
            meta = extract_ffprobe_metadata(video)
            assert meta.duration_seconds == 0.0


class TestSubtitleExceptionCatch:
    def test_subtitle_exception_caught(self, tmp_path: Path):
        """If extract_subtitle_text raises, extract_all_metadata survives."""
        video = tmp_path / "show.ts"
        video.touch()

        def side(*a, **kw):
            cmd = a[0] if a else kw.get("args", [])
            if cmd and cmd[0] == "ffprobe":
                return _ffprobe_ok()
            raise RuntimeError("ffmpeg segfault")

        with patch("epicur.metadata_extractor.subprocess.run", side_effect=side):
            meta = extract_all_metadata(video)
            assert meta.title == "J.A.G. - Im Auftrag der Ehre"
            assert meta.subtitle_texts == []


class TestTmdbEdgeCases:
    def test_season_count_none_response(self):
        """get_tmdb_season_count returns 0 when API returns None."""
        from epicur.episode_identifier import get_tmdb_season_count

        with patch("epicur.episode_identifier._http_get_json", return_value=None):
            assert get_tmdb_season_count(999, "key") == 0

    def test_match_tmdb_empty_episodes(self):
        """match_episode_tmdb returns None when no episodes found."""
        from epicur.episode_identifier import match_episode_tmdb

        with patch("epicur.episode_identifier._http_get_json") as m:
            m.side_effect = [
                TMDB_SEARCH,   # search → found
                {"number_of_seasons": 1},  # detail
                {"episodes": []},  # empty season
            ]
            result = match_episode_tmdb("J.A.G.", ExtractedMetadata(), "key")
            assert result is None


class TestIdentifyEpisodeEdges:
    def test_tvh_direct_match_path(self):
        """TVH subtitle triggers direct-match step in identify_episode."""
        from epicur.episode_identifier import identify_episode

        meta = ExtractedMetadata(tvh_subtitle="Angst")
        with patch("epicur.episode_identifier._http_get_json") as m:
            m.side_effect = [TVMAZE_SHOW, TVMAZE_EPISODES]
            result = identify_episode(
                "J.A.G.", meta, Path("/rec/JAG/JAG.ts"),
                use_tvmaze=True,
            )
            assert result is not None
            assert result.source == "tvh+tvmaze"

    def test_tmdb_exception_caught(self):
        """If TMDB raises, pipeline continues to filename fallback."""
        from epicur.episode_identifier import identify_episode

        meta = ExtractedMetadata()
        with patch("epicur.episode_identifier.match_episode_tmdb", side_effect=RuntimeError("boom")):
            result = identify_episode(
                "Show", meta, Path("/rec/Show/Show S01E01.ts"),
                use_tvmaze=False, tmdb_api_key="key",
            )
            assert result is not None
            assert result.source == "filename"


class TestProcessDirectoryEdges:
    def test_duplicates_dir_skipped(self, tmp_path: Path):
        from epicur.main import process_directory

        root = tmp_path / "recordings"
        (root / "duplicates").mkdir(parents=True)
        (root / "Show").mkdir()
        (root / "Show" / "Show S01E01.ts").write_text("x")

        with patch("epicur.metadata_extractor.subprocess.run", side_effect=_subprocess_side):
            results = process_directory(root, dry_run=True, use_tvmaze=False, min_age=0)
            # Only Show should be processed, not duplicates
            assert len(results) == 1

    def test_no_video_files_in_series(self, tmp_path: Path):
        from epicur.main import process_directory

        root = tmp_path / "recordings"
        (root / "EmptyShow").mkdir(parents=True)
        (root / "EmptyShow" / "readme.txt").write_text("not a video")

        results = process_directory(root, use_tvmaze=False)
        assert results == []

    def test_match_below_threshold_logged(self, tmp_path: Path):
        """A match below threshold should still write .meta but skip organizing."""
        from epicur.main import process_directory

        root = tmp_path / "recordings"
        show = root / "Show"
        show.mkdir(parents=True)
        (show / "Show S01E01.ts").write_text("x")

        with patch("epicur.metadata_extractor.subprocess.run", side_effect=_subprocess_side):
            results = process_directory(root, min_confidence=0.99, use_tvmaze=False, dry_run=True, min_age=0)
            assert all(r.action == "skipped" for r in results)


class TestReviewEdges:
    def test_invalid_then_skip(self, tmp_path: Path, capsys):
        """Invalid input prints error, then 's' succeeds."""
        from epicur.review import review_unmatched

        root, _, _ = _create_review_fixture(tmp_path)
        with patch("builtins.input", side_effect=["xyz", "s"]):
            review_unmatched(root, {".ts"})
        out = capsys.readouterr().out
        assert "Ungültige Eingabe" in out
        assert "Übersprungen" in out

    def test_override_cancel_then_skip(self, tmp_path: Path, capsys):
        """Override with empty input cancels, then skip."""
        from epicur.review import review_unmatched

        root, _, _ = _create_review_fixture(tmp_path)
        # "ü" triggers override, "" cancels it, "s" skips
        with patch("builtins.input", side_effect=["ü", "", "s"]):
            review_unmatched(root, {".ts"})
        out = capsys.readouterr().out
        assert "Übersprungen" in out


# ---------------------------------------------------------------------------
# Recording skip – CLI integration
# ---------------------------------------------------------------------------
class TestRecordingSkip:
    """Integration tests for --min-age CLI flag."""

    def test_cli_min_age_skips_recent(self, tmp_path: Path, capsys):
        """--min-age 9999 should mark all freshly-created files as recording."""
        from epicur.main import main

        root = tmp_path / "recordings"
        series = root / "Show"
        series.mkdir(parents=True)
        (series / "ep.ts").write_text("x")

        fake_log = tmp_path / "no_dvr"
        rc = main([
            "recognize",
            "--no-tvmaze", "--dry-run",
            "--min-age", "9999",
            "--tvh-dvr-log", str(fake_log),
            str(root),
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Recording (skipped)" in out
        assert "[RECORDING]" in out

    def test_cli_min_age_zero_processes(self, tmp_path: Path, capsys):
        """--min-age 0 should disable the recording check."""
        from epicur.main import main

        root = tmp_path / "recordings"
        series = root / "Show"
        series.mkdir(parents=True)
        (series / "ep.ts").write_text("x")

        fake_log = tmp_path / "no_dvr"
        with patch("epicur.main.extract_all_metadata") as mock_meta, \
             patch("epicur.main.identify_episode", return_value=None), \
             patch("epicur.main.write_meta_file"):
            mock_meta.return_value = MagicMock()
            rc = main([
                "recognize",
                "--no-tvmaze", "--dry-run",
                "--min-age", "0",
                "--tvh-dvr-log", str(fake_log),
                str(root),
            ])
        assert rc == 0
        out = capsys.readouterr().out
        # File should NOT be marked as recording
        assert "Recording (skipped)   : 0" in out or "Recording" not in out or "Recording (skipped)   : 0" in out
