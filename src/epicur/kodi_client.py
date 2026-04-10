"""Kodi JSON-RPC client for triggering video library updates.

Uses the Kodi HTTP JSON-RPC interface to trigger a ``VideoLibrary.Scan``
after postprocessing has placed new files into the media library.
"""
from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


def scan_video_library(
    base_url: str,
    directory: str = "",
    username: str = "",
    password: str = "",
    timeout: int = 10,
) -> bool:
    """Trigger a Kodi video library scan via JSON-RPC HTTP.

    Parameters
    ----------
    base_url:
        Kodi HTTP base URL, e.g. ``http://192.168.1.10:8080``.
    directory:
        Optional path to limit the scan to a specific directory.
        When empty, a full library scan is performed.
    username:
        HTTP Basic-Auth username (leave empty if auth is disabled).
    password:
        HTTP Basic-Auth password.
    timeout:
        HTTP request timeout in seconds.  Kodi acknowledges the scan
        request immediately; the actual scan runs asynchronously.

    Returns
    -------
    bool
        ``True`` if the scan was triggered successfully, ``False`` on error.
    """
    url = f"{base_url.rstrip('/')}/jsonrpc"

    params: dict = {}
    if directory:
        params["directory"] = directory

    payload = json.dumps({
        "jsonrpc": "2.0",
        "method": "VideoLibrary.Scan",
        "params": params,
        "id": 1,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    if username:
        credentials = base64.b64encode(
            f"{username}:{password}".encode()
        ).decode("ascii")
        req.add_header("Authorization", f"Basic {credentials}")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            result = body.get("result", "")
            if result == "OK":
                logger.info("Kodi library scan triggered: %s", directory or "(full)")
            else:
                logger.warning("Kodi returned unexpected result: %s", body)
            return True
    except urllib.error.HTTPError as exc:
        logger.warning("Kodi HTTP error %d: %s", exc.code, exc.reason)
        return False
    except urllib.error.URLError as exc:
        logger.warning("Cannot reach Kodi at %s: %s", base_url, exc.reason)
        return False
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Kodi request failed: %s", exc)
        return False
