"""Tests for epicur.kodi_client module."""
from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from epicur.kodi_client import scan_video_library


class TestScanVideoLibrary:
    """Tests for scan_video_library()."""

    @patch("epicur.kodi_client.urllib.request.urlopen")
    def test_successful_scan(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "result": "OK"}
        ).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = scan_video_library("http://localhost:8080")
        assert result is True

        # Verify the request
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.full_url == "http://localhost:8080/jsonrpc"
        assert req.get_header("Content-type") == "application/json"
        body = json.loads(req.data)
        assert body["method"] == "VideoLibrary.Scan"
        assert body["params"] == {}

    @patch("epicur.kodi_client.urllib.request.urlopen")
    def test_scan_with_directory(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "result": "OK"}
        ).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = scan_video_library(
            "http://localhost:8080", directory="/media/tv/ShowName"
        )
        assert result is True

        body = json.loads(mock_urlopen.call_args[0][0].data)
        assert body["params"]["directory"] == "/media/tv/ShowName"

    @patch("epicur.kodi_client.urllib.request.urlopen")
    def test_basic_auth_header(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "result": "OK"}
        ).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        scan_video_library(
            "http://kodi:8080", username="admin", password="secret"
        )

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization").startswith("Basic ")

    @patch("epicur.kodi_client.urllib.request.urlopen")
    def test_no_auth_without_username(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "result": "OK"}
        ).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        scan_video_library("http://kodi:8080")

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") is None

    @patch("epicur.kodi_client.urllib.request.urlopen")
    def test_http_error_returns_false(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "http://kodi:8080/jsonrpc", 401, "Unauthorized", {}, BytesIO(b"")
        )
        result = scan_video_library("http://kodi:8080")
        assert result is False

    @patch("epicur.kodi_client.urllib.request.urlopen")
    def test_url_error_returns_false(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        result = scan_video_library("http://kodi:8080")
        assert result is False

    @patch("epicur.kodi_client.urllib.request.urlopen")
    def test_trailing_slash_stripped(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "result": "OK"}
        ).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        scan_video_library("http://kodi:8080/")
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://kodi:8080/jsonrpc"
