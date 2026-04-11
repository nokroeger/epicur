import pytest
import os, time
import tempfile
from pathlib import Path
from unittest import mock

import epicur.postprocess as postprocess
from epicur.models import PostprocessResult

@pytest.fixture
def temp_dirs(tmp_path):
    root_dir = tmp_path / "movies"
    root_dir.mkdir()
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    return root_dir, library_dir

@mock.patch("epicur.postprocess.postprocess_episode")
def test_single_ts_file_processed(mock_postprocess_episode, temp_dirs):
    root_dir, library_dir = temp_dirs
    ts_file = root_dir / "movie1.ts"
    ts_file.write_bytes(b"dummy")
    two_minutes_ago = (time.time() - 120)
    os.utime(ts_file, (two_minutes_ago, two_minutes_ago))
    mock_postprocess_episode.return_value = PostprocessResult(
        source_path=ts_file, output_path=library_dir / "movie1.mp4", action="converted"
    )
    results = postprocess.postprocess_movies(root_dir, library_dir, comskip_ini=None)
    assert len(results) == 1
    assert results[0].action == "converted"
    assert results[0].output_path.name == "movie1.mp4"
    mock_postprocess_episode.assert_called_once()

@mock.patch("epicur.postprocess.postprocess_episode")
def test_multiple_ts_files_processed(mock_postprocess_episode, temp_dirs):
    root_dir, library_dir = temp_dirs
    ts_files = [root_dir / f"movie{i}.ts" for i in range(3)]
    for f in ts_files:
        f.write_bytes(b"dummy")
        two_minutes_ago = (time.time() - 120)
        os.utime(f, (two_minutes_ago, two_minutes_ago))        
    mock_postprocess_episode.side_effect = [
        PostprocessResult(source_path=f, output_path=library_dir / f"{f.stem}.mp4", action="converted")
        for f in ts_files
    ]
    results = postprocess.postprocess_movies(root_dir, library_dir, comskip_ini=None)
    assert len(results) == 3
    assert all(r.action == "converted" for r in results)
    assert {r.source_path for r in results} == set(ts_files)

@mock.patch("epicur.postprocess.postprocess_episode")
def test_mp4_files_skipped(mock_postprocess_episode, temp_dirs):
    root_dir, library_dir = temp_dirs
    ts_file = root_dir / "movie1.ts"
    ts_file.write_bytes(b"dummy")
    two_minutes_ago = (time.time() - 120)
    os.utime(ts_file, (two_minutes_ago, two_minutes_ago))    
    mp4_file = root_dir / "movie2.mp4"
    mp4_file.write_bytes(b"dummy")
    os.utime(mp4_file, (two_minutes_ago, two_minutes_ago))
    mock_postprocess_episode.return_value = PostprocessResult(
        source_path=ts_file, output_path=library_dir / "movie1.mp4", action="converted"
    )
    results = postprocess.postprocess_movies(root_dir, library_dir, comskip_ini=None)
    assert any(r.source_path == mp4_file and r.action == "skipped" for r in results)
    assert any(r.source_path == ts_file and r.action == "converted" for r in results)

@mock.patch("epicur.postprocess.postprocess_episode")
def test_non_supported_files_ignored(mock_postprocess_episode, temp_dirs):
    root_dir, library_dir = temp_dirs
    avi_file = root_dir / "movie1.avi"
    avi_file.write_bytes(b"dummy")
    two_minutes_ago = (time.time() - 120)
    os.utime(avi_file, (two_minutes_ago, two_minutes_ago))
    results = postprocess.postprocess_movies(root_dir, library_dir, comskip_ini=None)
    assert results == []
    mock_postprocess_episode.assert_not_called()

def test_empty_movie_dir(temp_dirs):
    root_dir, library_dir = temp_dirs
    results = postprocess.postprocess_movies(root_dir, library_dir, comskip_ini=None)
    assert results == []

@mock.patch("epicur.postprocess.postprocess_episode")
def test_dry_run_option(mock_postprocess_episode, temp_dirs):
    root_dir, library_dir = temp_dirs
    ts_file = root_dir / "movie1.ts"
    ts_file.write_bytes(b"dummy")
    two_minutes_ago = (time.time() - 120)
    os.utime(ts_file, (two_minutes_ago, two_minutes_ago))

    results = postprocess.postprocess_movies(root_dir, library_dir, comskip_ini=None, dry_run=True)
    assert len(results) == 1
    assert results[0].action == "converted"
    mock_postprocess_episode.assert_not_called()

@mock.patch("epicur.postprocess.postprocess_episode")
def test_custom_extensions(mock_postprocess_episode, temp_dirs):
    root_dir, library_dir = temp_dirs
    mkv_file = root_dir / "movie1.mkv"
    mkv_file.write_bytes(b"dummy")
    two_minutes_ago = (time.time() - 120)
    os.utime(mkv_file, (two_minutes_ago, two_minutes_ago))
    mock_postprocess_episode.return_value = PostprocessResult(
        source_path=mkv_file, output_path=library_dir / "movie1.mp4", action="converted"
    )
    results = postprocess.postprocess_movies(root_dir, library_dir, comskip_ini=None, extensions={".mkv"})
    assert len(results) == 1
    assert results[0].source_path == mkv_file
    assert results[0].action == "converted"
    mock_postprocess_episode.assert_called_once()
    
@mock.patch("epicur.postprocess.postprocess_episode")
def test_recent_ts_file_skipped(mock_postprocess_episode, temp_dirs):
    root_dir, library_dir = temp_dirs
    ts_file = root_dir / "movie1.ts"
    ts_file.write_bytes(b"dummy")
    mock_postprocess_episode.return_value = PostprocessResult(
        source_path=ts_file, output_path=library_dir / "movie1.mp4", action="converted"
    )
    results = postprocess.postprocess_movies(root_dir, library_dir, comskip_ini=None)
    assert len(results) == 1
    assert results[0].action == "skipped"
    mock_postprocess_episode.assert_not_called()
