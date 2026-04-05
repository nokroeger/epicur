"""Tests for epicur postprocess module."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from epicur.models import PostprocessResult, SeasonInfo
from epicur.postprocess import (
    _scan_season_dir,
    cleanup_files,
    convert_to_mp4,
    detect_commercials,
    find_complete_seasons,
    generate_ffmetadata,
    postprocess_all,
    postprocess_episode,
    print_postprocess_report,
)


# ═══════════════════════════════════════════════════════════════════════
# Season completeness
# ═══════════════════════════════════════════════════════════════════════


class TestScanSeasonDir:
    def test_finds_episodes(self, tmp_path: Path):
        season = tmp_path / "Season 1"
        season.mkdir()
        (season / "Show S01E01.ts").write_text("x")
        (season / "Show S01E03.ts").write_text("x")
        (season / "random.txt").write_text("x")  # non-video
        result = _scan_season_dir(season, {".ts"})
        assert result == {1: season / "Show S01E01.ts", 3: season / "Show S01E03.ts"}

    def test_ignores_wrong_extensions(self, tmp_path: Path):
        season = tmp_path / "Season 1"
        season.mkdir()
        (season / "Show S01E01.mp4").write_text("x")
        result = _scan_season_dir(season, {".ts"})
        assert result == {}

    def test_ignores_files_without_pattern(self, tmp_path: Path):
        season = tmp_path / "Season 1"
        season.mkdir()
        (season / "random.ts").write_text("x")
        result = _scan_season_dir(season, {".ts"})
        assert result == {}


class TestFindCompleteSeasons:
    @patch("epicur.postprocess._get_episode_count_per_season")
    def test_complete_season(self, mock_counts, tmp_path: Path):
        root = tmp_path / "recordings"
        series = root / "Show"
        season = series / "Season 1"
        season.mkdir(parents=True)
        (season / "Show S01E01.ts").write_text("x")
        (season / "Show S01E02.ts").write_text("x")

        mock_counts.return_value = {1: 2}
        result = find_complete_seasons(root, {".ts"})
        assert len(result) == 1
        assert result[0].is_complete
        assert result[0].series_title == "Show"
        assert result[0].season_number == 1
        assert result[0].missing_episodes == []

    @patch("epicur.postprocess._get_episode_count_per_season")
    def test_incomplete_season(self, mock_counts, tmp_path: Path):
        root = tmp_path / "recordings"
        series = root / "Show"
        season = series / "Season 1"
        season.mkdir(parents=True)
        (season / "Show S01E01.ts").write_text("x")
        # Missing E02

        mock_counts.return_value = {1: 2}
        result = find_complete_seasons(root, {".ts"})
        assert len(result) == 0

    @patch("epicur.postprocess._get_episode_count_per_season")
    def test_empty_season_dir(self, mock_counts, tmp_path: Path):
        root = tmp_path / "recordings"
        series = root / "Show"
        season = series / "Season 1"
        season.mkdir(parents=True)

        mock_counts.return_value = {1: 2}
        result = find_complete_seasons(root, {".ts"})
        assert len(result) == 0

    @patch("epicur.postprocess._get_episode_count_per_season")
    def test_multiple_series_mixed(self, mock_counts, tmp_path: Path):
        root = tmp_path / "recordings"

        # Complete series
        s1 = root / "Complete" / "Season 1"
        s1.mkdir(parents=True)
        (s1 / "Complete S01E01.ts").write_text("x")

        # Incomplete series
        s2 = root / "Incomplete" / "Season 1"
        s2.mkdir(parents=True)
        (s2 / "Incomplete S01E01.ts").write_text("x")

        def side_effect(title, **kw):
            if title == "Complete":
                return {1: 1}
            return {1: 3}

        mock_counts.side_effect = side_effect
        result = find_complete_seasons(root, {".ts"})
        assert len(result) == 1
        assert result[0].series_title == "Complete"

    @patch("epicur.postprocess._get_episode_count_per_season")
    def test_no_episode_counts(self, mock_counts, tmp_path: Path):
        root = tmp_path / "recordings"
        series = root / "Unknown" / "Season 1"
        series.mkdir(parents=True)
        (series / "Unknown S01E01.ts").write_text("x")

        mock_counts.return_value = {}
        result = find_complete_seasons(root, {".ts"})
        assert len(result) == 0

    def test_nonexistent_root(self, tmp_path: Path):
        result = find_complete_seasons(tmp_path / "nope", {".ts"})
        assert result == []

    @patch("epicur.postprocess._get_episode_count_per_season")
    def test_skips_duplicates_dir(self, mock_counts, tmp_path: Path):
        root = tmp_path / "recordings"
        (root / "duplicates" / "Season 1").mkdir(parents=True)
        mock_counts.return_value = {1: 1}
        result = find_complete_seasons(root, {".ts"})
        assert result == []


# ═══════════════════════════════════════════════════════════════════════
# Commercial detection
# ═══════════════════════════════════════════════════════════════════════


class TestDetectCommercials:
    def test_successful_detection(self, tmp_path: Path):
        ts = tmp_path / "show.ts"
        ts.write_text("video data")
        edl = tmp_path / "show.edl"
        edl.write_text("10.5\t60.0\t0\n120.0\t180.5\t0\n")
        ini = tmp_path / "comskip.ini"
        ini.write_text("")

        with patch("epicur.postprocess.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = detect_commercials(ts, ini)

        assert result == [(10.5, 60.0), (120.0, 180.5)]

    def test_no_commercials_found(self, tmp_path: Path):
        ts = tmp_path / "show.ts"
        ts.write_text("video data")
        ini = tmp_path / "comskip.ini"
        ini.write_text("")

        with patch("epicur.postprocess.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = detect_commercials(ts, ini)

        assert result == []

    def test_comskip_not_installed(self, tmp_path: Path):
        ts = tmp_path / "show.ts"
        ts.write_text("video data")
        ini = tmp_path / "comskip.ini"
        ini.write_text("")

        with patch("epicur.postprocess.subprocess.run", side_effect=FileNotFoundError):
            result = detect_commercials(ts, ini)

        assert result == []

    def test_comskip_timeout(self, tmp_path: Path):
        ts = tmp_path / "show.ts"
        ts.write_text("video data")
        ini = tmp_path / "comskip.ini"
        ini.write_text("")

        with patch("epicur.postprocess.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 7200)):
            result = detect_commercials(ts, ini)

        assert result == []


# ═══════════════════════════════════════════════════════════════════════
# Chapter metadata
# ═══════════════════════════════════════════════════════════════════════


class TestGenerateFFMetadata:
    def test_with_commercials(self, tmp_path: Path):
        ts = tmp_path / "show.ts"
        ts.write_text("x")

        with patch("epicur.postprocess.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            # simulate ffmpeg writing a minimal ffmeta file
            def write_ffmeta(*args, **kwargs):
                ts.with_suffix(".ffmeta").write_text(";FFMETADATA1\n")
            mock_run.side_effect = write_ffmeta

            result = generate_ffmetadata(
                ts, [(10.0, 60.0), (120.0, 180.0)], "My Show", 3600.0,
            )

        content = result.read_text()
        assert "TIMEBASE=1/1000" in content
        assert "title=Chapter 1" in content
        assert "title=Commercial 1" in content
        assert "title=Chapter 2" in content
        assert "title=Commercial 2" in content
        assert "title=Chapter 3" in content
        assert "START=0" in content
        assert "START=10000" in content
        assert "END=10000" in content
        assert "END=60000" in content

    def test_no_commercials(self, tmp_path: Path):
        ts = tmp_path / "show.ts"
        ts.write_text("x")

        with patch("epicur.postprocess.subprocess.run") as mock_run:
            def write_ffmeta(*args, **kwargs):
                ts.with_suffix(".ffmeta").write_text(";FFMETADATA1\n")
            mock_run.side_effect = write_ffmeta

            result = generate_ffmetadata(ts, [], "My Show", 3600.0)

        content = result.read_text()
        assert "title=My Show" in content
        assert "[CHAPTER]" not in content

    def test_single_commercial(self, tmp_path: Path):
        ts = tmp_path / "show.ts"
        ts.write_text("x")

        with patch("epicur.postprocess.subprocess.run") as mock_run:
            def write_ffmeta(*args, **kwargs):
                ts.with_suffix(".ffmeta").write_text(";FFMETADATA1\n")
            mock_run.side_effect = write_ffmeta

            result = generate_ffmetadata(
                ts, [(30.0, 90.0)], "Show", 600.0,
            )

        content = result.read_text()
        # Should have: Chapter 1, Commercial 1, Chapter 2
        assert content.count("[CHAPTER]") == 3
        assert "title=Chapter 1" in content
        assert "title=Commercial 1" in content
        assert "title=Chapter 2" in content

    def test_ffmpeg_fails_creates_minimal(self, tmp_path: Path):
        ts = tmp_path / "show.ts"
        ts.write_text("x")

        with patch("epicur.postprocess.subprocess.run", side_effect=FileNotFoundError):
            result = generate_ffmetadata(ts, [(10.0, 20.0)], "Show", 100.0)

        content = result.read_text()
        assert ";FFMETADATA1" in content
        assert "[CHAPTER]" in content  # chapters still added


# ═══════════════════════════════════════════════════════════════════════
# Conversion
# ═══════════════════════════════════════════════════════════════════════


class TestConvertToMp4:
    def test_successful_with_chapters(self, tmp_path: Path):
        ts = tmp_path / "show.ts"
        ts.write_text("x")
        ffmeta = tmp_path / "show.ffmeta"
        ffmeta.write_text(";FFMETADATA1\n")
        output = tmp_path / "out" / "show.mp4"

        with patch("epicur.postprocess.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = convert_to_mp4(ts, ffmeta, output, crf=20, preset="slow")

        assert result is True
        # Verify ffmpeg was called with chapter args
        ffmpeg_call = mock_run.call_args_list[0]
        cmd = ffmpeg_call[0][0]
        assert "-i" in cmd
        assert str(ffmeta) in cmd
        assert "-map_metadata" in cmd
        assert "1" in cmd
        assert "-c:v" in cmd
        assert "libx264" in cmd
        assert "-c:a" in cmd
        assert "aac" in cmd

    def test_successful_without_chapters(self, tmp_path: Path):
        ts = tmp_path / "show.ts"
        ts.write_text("x")
        output = tmp_path / "out" / "show.mp4"

        with patch("epicur.postprocess.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = convert_to_mp4(ts, None, output)

        assert result is True
        ffmpeg_call = mock_run.call_args_list[0]
        cmd = ffmpeg_call[0][0]
        assert "-map_metadata" not in cmd

    def test_ffmpeg_failure(self, tmp_path: Path):
        ts = tmp_path / "show.ts"
        ts.write_text("x")
        output = tmp_path / "out" / "show.mp4"

        with patch("epicur.postprocess.subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
            result = convert_to_mp4(ts, None, output)

        assert result is False

    def test_ffmpeg_not_installed(self, tmp_path: Path):
        ts = tmp_path / "show.ts"
        ts.write_text("x")
        output = tmp_path / "out" / "show.mp4"

        with patch("epicur.postprocess.subprocess.run", side_effect=FileNotFoundError):
            result = convert_to_mp4(ts, None, output)

        assert result is False

    def test_ffprobe_validation_fails(self, tmp_path: Path):
        ts = tmp_path / "show.ts"
        ts.write_text("x")
        output = tmp_path / "out" / "show.mp4"
        output.parent.mkdir(parents=True)

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # ffmpeg succeeds — create the fake output
                output.write_text("fake")
                return MagicMock(returncode=0)
            # ffprobe fails
            raise subprocess.CalledProcessError(1, "ffprobe")

        with patch("epicur.postprocess.subprocess.run", side_effect=side_effect):
            result = convert_to_mp4(ts, None, output)

        assert result is False
        assert not output.exists()


# ═══════════════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════════════


class TestCleanupFiles:
    def test_cleanup_all(self, tmp_path: Path):
        ts = tmp_path / "show.ts"
        ts.write_text("x")
        edl = tmp_path / "show.edl"
        edl.write_text("x")
        ffmeta = tmp_path / "show.ffmeta"
        ffmeta.write_text("x")
        meta = tmp_path / "show.meta"
        meta.write_text("x")
        log = tmp_path / "show.log"
        log.write_text("x")

        cleanup_files(ts, ffmeta)

        assert not ts.exists()
        assert not edl.exists()
        assert not ffmeta.exists()
        assert not meta.exists()
        assert not log.exists()

    def test_cleanup_missing_files(self, tmp_path: Path):
        ts = tmp_path / "show.ts"
        ts.write_text("x")
        # No other files exist — should not crash
        cleanup_files(ts, None)
        assert not ts.exists()

    def test_cleanup_with_none_ffmeta(self, tmp_path: Path):
        ts = tmp_path / "show.ts"
        ts.write_text("x")
        cleanup_files(ts, None)
        assert not ts.exists()


# ═══════════════════════════════════════════════════════════════════════
# Per-episode pipeline
# ═══════════════════════════════════════════════════════════════════════


class TestPostprocessEpisode:
    @patch("epicur.postprocess.cleanup_files")
    @patch("epicur.postprocess.convert_to_mp4", return_value=True)
    @patch("epicur.postprocess.generate_ffmetadata")
    @patch("epicur.postprocess.extract_ffprobe_metadata")
    @patch("epicur.postprocess.detect_commercials", return_value=[(10.0, 60.0)])
    def test_success_with_commercials(self, mock_detect, mock_ffprobe, mock_ffmeta, mock_convert, mock_cleanup, tmp_path):
        ts = tmp_path / "show.ts"
        ts.write_text("x")
        output = tmp_path / "out" / "show.mp4"
        ini = tmp_path / "comskip.ini"
        ini.write_text("")
        mock_ffprobe.return_value = MagicMock(duration_seconds=3600.0)
        mock_ffmeta.return_value = ts.with_suffix(".ffmeta")

        result = postprocess_episode(ts, output, ini)

        assert result.action == "converted"
        assert result.output_path == output
        mock_detect.assert_called_once()
        mock_ffmeta.assert_called_once()
        mock_convert.assert_called_once()
        mock_cleanup.assert_called_once()

    @patch("epicur.postprocess.cleanup_files")
    @patch("epicur.postprocess.convert_to_mp4", return_value=True)
    @patch("epicur.postprocess.detect_commercials", return_value=[])
    def test_success_no_commercials(self, mock_detect, mock_convert, mock_cleanup, tmp_path):
        ts = tmp_path / "show.ts"
        ts.write_text("x")
        output = tmp_path / "out" / "show.mp4"
        ini = tmp_path / "comskip.ini"
        ini.write_text("")

        result = postprocess_episode(ts, output, ini)

        assert result.action == "converted"
        mock_convert.assert_called_once_with(ts, None, output, crf=20, preset="slow")

    @patch("epicur.postprocess.convert_to_mp4", return_value=False)
    @patch("epicur.postprocess.detect_commercials", return_value=[])
    def test_conversion_failure(self, mock_detect, mock_convert, tmp_path):
        ts = tmp_path / "show.ts"
        ts.write_text("x")
        output = tmp_path / "out" / "show.mp4"
        ini = tmp_path / "comskip.ini"
        ini.write_text("")

        result = postprocess_episode(ts, output, ini)

        assert result.action == "error"
        assert "FFmpeg conversion failed" in result.error_message
        # Source should NOT be deleted (cleanup_files NOT called for ts)
        assert ts.exists()


# ═══════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════


class TestPostprocessAll:
    @patch("epicur.postprocess.find_complete_seasons")
    def test_no_complete_seasons(self, mock_find, tmp_path: Path):
        mock_find.return_value = []
        results = postprocess_all(tmp_path, tmp_path / "lib", None)
        assert results == []

    @patch("epicur.postprocess.postprocess_episode")
    @patch("epicur.postprocess.find_complete_seasons")
    def test_dry_run_no_subprocess(self, mock_find, mock_pp, tmp_path: Path):
        root = tmp_path / "rec"
        series = root / "Show"
        season_dir = series / "Season 1"
        season_dir.mkdir(parents=True)
        ep1 = season_dir / "Show S01E01.ts"
        ep1.write_text("x")

        mock_find.return_value = [SeasonInfo(
            series_title="Show", series_dir=series,
            season_number=1, total_episodes=1,
            present_episodes={1: ep1}, missing_episodes=[],
        )]

        results = postprocess_all(root, tmp_path / "lib", None, dry_run=True)
        assert len(results) == 1
        assert results[0].action == "converted"
        mock_pp.assert_not_called()  # no actual postprocessing in dry-run

    @patch("epicur.postprocess.postprocess_episode")
    @patch("epicur.postprocess.find_complete_seasons")
    def test_mixed_results(self, mock_find, mock_pp, tmp_path: Path):
        root = tmp_path / "rec"
        series = root / "Show"
        season_dir = series / "Season 1"
        season_dir.mkdir(parents=True)
        ep1 = season_dir / "Show S01E01.ts"
        ep1.write_text("x")
        ep2 = season_dir / "Show S01E02.ts"
        ep2.write_text("x")

        mock_find.return_value = [SeasonInfo(
            series_title="Show", series_dir=series,
            season_number=1, total_episodes=2,
            present_episodes={1: ep1, 2: ep2}, missing_episodes=[],
        )]

        mock_pp.side_effect = [
            PostprocessResult(source_path=ep1, output_path=tmp_path / "lib" / "show1.mp4", action="converted"),
            PostprocessResult(source_path=ep2, output_path=None, action="error", error_message="fail"),
        ]

        results = postprocess_all(root, tmp_path / "lib", None)
        assert len(results) == 2
        assert results[0].action == "converted"
        assert results[1].action == "error"

    @patch("epicur.postprocess.postprocess_episode")
    @patch("epicur.postprocess.find_complete_seasons")
    def test_skips_already_converted(self, mock_find, mock_pp, tmp_path: Path):
        root = tmp_path / "rec"
        lib = tmp_path / "lib"
        series = root / "Show"
        season_dir = series / "Season 1"
        season_dir.mkdir(parents=True)
        ep = season_dir / "Show S01E01.ts"
        ep.write_text("x")

        # Create the output file in library
        lib_out = lib / "Show" / "Season 1" / "Show S01E01.mp4"
        lib_out.parent.mkdir(parents=True)
        lib_out.write_text("already there")

        mock_find.return_value = [SeasonInfo(
            series_title="Show", series_dir=series,
            season_number=1, total_episodes=1,
            present_episodes={1: ep}, missing_episodes=[],
        )]

        results = postprocess_all(root, lib, None)
        assert len(results) == 1
        assert results[0].action == "skipped"
        mock_pp.assert_not_called()

    @patch("epicur.postprocess.find_complete_seasons")
    def test_skips_non_ts_files(self, mock_find, tmp_path: Path):
        root = tmp_path / "rec"
        series = root / "Show"
        season_dir = series / "Season 1"
        season_dir.mkdir(parents=True)
        ep = season_dir / "Show S01E01.mp4"
        ep.write_text("x")

        mock_find.return_value = [SeasonInfo(
            series_title="Show", series_dir=series,
            season_number=1, total_episodes=1,
            present_episodes={1: ep}, missing_episodes=[],
        )]

        results = postprocess_all(root, tmp_path / "lib", None)
        assert len(results) == 1
        assert results[0].action == "skipped"


# ═══════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════


class TestPrintPostprocessReport:
    def test_empty_results(self, capsys):
        print_postprocess_report([])
        out = capsys.readouterr().out
        assert "No episodes to postprocess" in out

    def test_mixed_results(self, capsys, tmp_path: Path):
        results = [
            PostprocessResult(source_path=tmp_path / "a.ts", output_path=tmp_path / "a.mp4", action="converted"),
            PostprocessResult(source_path=tmp_path / "b.ts", output_path=None, action="skipped"),
            PostprocessResult(source_path=tmp_path / "c.ts", output_path=None, action="error", error_message="fail"),
        ]
        print_postprocess_report(results)
        out = capsys.readouterr().out
        assert "POSTPROCESS REPORT" in out
        assert "Converted           : 1" in out
        assert "Skipped             : 1" in out
        assert "Errors              : 1" in out
        assert "[CONVERTED]" in out
        assert "[SKIPPED  ]" in out
        assert "[ERROR    ]" in out
