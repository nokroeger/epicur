from __future__ import annotations

import logging
import math
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from .models import EpisodeMatch, ExtractedMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter – TVMaze allows 20 requests per 10 seconds
# ---------------------------------------------------------------------------

_last_request_times: list[float] = []
RATE_LIMIT_WINDOW = 10.0
RATE_LIMIT_MAX = 20


def _rate_limit() -> None:
    """Block until a request slot is available within the rate-limit window."""
    now = time.monotonic()
    # Remove timestamps outside the window
    while _last_request_times and _last_request_times[0] < now - RATE_LIMIT_WINDOW:
        _last_request_times.pop(0)
    if len(_last_request_times) >= RATE_LIMIT_MAX:
        wait = RATE_LIMIT_WINDOW - (now - _last_request_times[0])
        if wait > 0:
            logger.debug("Rate limit reached, sleeping %.1fs", wait)
            time.sleep(wait)
    _last_request_times.append(time.monotonic())


def _http_get_json(url: str, params: dict[str, str] | None = None, timeout: int = 15) -> Any | None:
    """Perform an HTTP GET and return parsed JSON, or None on failure."""
    import urllib.request
    import urllib.error
    import json

    if params:
        url = f"{url}?{urlencode(params)}" if "?" not in url else f"{url}&{urlencode(params)}"

    _rate_limit()
    # Mask sensitive query parameters before logging
    log_url = re.sub(r'(api_key=)[^&]+', r'\1***', url)
    logger.debug("HTTP GET %s", log_url)

    req = urllib.request.Request(url, headers={"User-Agent": "epicur/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        logger.warning("HTTP %d for %s", exc.code, log_url)
        return None
    except Exception as exc:
        logger.warning("HTTP request failed for %s: %s", log_url, exc)
        return None


# ---------------------------------------------------------------------------
# Fuzzy matching helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for fuzzy comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _similarity(a: str, b: str) -> float:
    """Return similarity ratio (0.0–1.0) between two strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


# German stopwords – common function words that dilute keyword matching
_STOPWORDS_DE = frozenset(
    "aber alle allem allen aller allerdings alles also am an andere anderem"
    " anderen anderer anderes anderm andern anderr anders auch auf aus bei"
    " beim bereits besonders bin bis bist bitte da dabei dadurch dafür dagegen"
    " daher dahin damals damit danach daneben dann daran darauf daraus darf"
    " darfst darin darum darunter das dass davon davor dazu dein deine deinem"
    " deinen deiner dem den denn dennoch der deren des deshalb dessen die"
    " dies diese dieselbe dieselben diesem diesen dieser dieses doch dort"
    " drei drin dritte dritten dritter drittes du dumm durch dürfen ein"
    " einander eine einem einen einer einige einigem einigen einiger einiges"
    " einmal er erst erste erstem ersten erster erstes es etwas euch euer"
    " eure eurem euren eurer eures für ganz gar gegen gewesen hab habe haben"
    " hat hatte hätte hier hin hinter ich ihm ihn ihnen ihr ihre ihrem ihren"
    " ihrer im immer in indem ins irgend ist ja jede jedem jeden jeder jedes"
    " jedoch jemals jene jenem jenen jener jenes jetzt kann kannst kein keine"
    " keinem keinen keiner konnte können könnte machen mag manche manchem"
    " manchen mancher manches man mehr mein meine meinem meinen meiner mich"
    " mir mit mochte möchte morgen morgens muss musste müssen nach nachdem"
    " nachher nächster neben nein nicht nichts noch nun nur ob oder ohne sehr"
    " seid sein seine seinem seinen seiner seit seitdem selbst sich sie sind"
    " so sofort sogar solch solche solchem solchen solcher soll sollen sollte"
    " sollten sondern sonst sowie über um und uns unser unsere unserem unseren"
    " unserer unter viel vielleicht vom von vor vorbei vorher warum was weder"
    " weil weit welch welche welchem welchen welcher welches wem wen wenig"
    " wenige wenigem wenigen weniger wenigstens wenn wer werde werden werdet"
    " wessen wie wieder will wir wird wirklich wo wollen wollt wollte"
    " wollten worin wurde würde während würden zu zum zur zusammen zwar"
    " zwischen".split()
)


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Return Jaccard similarity of content-word sets from two texts.

    German stopwords are removed so that function words (der, die, und, …)
    don't dilute the comparison of meaningful keywords.
    """
    words_a = set(_normalize(text_a).split()) - _STOPWORDS_DE
    words_b = set(_normalize(text_b).split()) - _STOPWORDS_DE
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _content_words(text: str) -> set[str]:
    """Extract content words (stopword-filtered) from text."""
    return set(_normalize(text).split()) - _STOPWORDS_DE


def _prefix_match(words_a: set[str], words_b: set[str], min_prefix: int = 7) -> set[str]:
    """Find words in *words_a* that fuzzy-match a word in *words_b* via shared prefix.

    Returns the subset of *words_a* that matched (excluding exact matches
    which should be handled separately).
    """
    only_a = words_a - words_b
    only_b = words_b - words_a
    matched: set[str] = set()
    for wa in only_a:
        if len(wa) < min_prefix:
            continue
        for wb in only_b:
            if len(wb) < min_prefix:
                continue
            # Count common prefix length
            prefix_len = 0
            for ca, cb in zip(wa, wb):
                if ca != cb:
                    break
                prefix_len += 1
            if prefix_len >= min_prefix:
                matched.add(wa)
                break
    return matched


def _compute_idf(documents: list[str]) -> dict[str, float]:
    """Compute IDF weights from a collection of document strings."""
    n = len(documents)
    if n == 0:
        return {}
    doc_freq: dict[str, int] = {}
    for doc in documents:
        for w in _content_words(doc):
            doc_freq[w] = doc_freq.get(w, 0) + 1
    # +1 smoothing in denominator to avoid division by zero for query-only words
    return {w: math.log(n / df) for w, df in doc_freq.items()}


def _idf_keyword_score(
    text_a: str,
    text_b: str,
    idf: dict[str, float],
    default_idf: float = 3.0,
) -> float:
    """IDF-weighted keyword F1 score between two texts.

    Words not present in the IDF table (e.g. from the query but absent from
    all episode summaries) receive *default_idf* which acts as a high weight
    for truly unique terms.  Prefix matching (>=7 chars) catches German
    inflections (e.g. spirituell / spirituellen).
    """
    words_a = _content_words(text_a)
    words_b = _content_words(text_b)
    if not words_a or not words_b:
        return 0.0

    # Exact matches
    exact = words_a & words_b
    # Prefix matches (additional, non-exact)
    prefix_a = _prefix_match(words_a, words_b)
    prefix_b = _prefix_match(words_b, words_a)

    def _w(word: str) -> float:
        return idf.get(word, default_idf)

    match_weight_a = sum(_w(w) for w in exact) + sum(_w(w) * 0.8 for w in prefix_a)
    total_weight_a = sum(_w(w) for w in words_a)
    recall = match_weight_a / total_weight_a if total_weight_a > 0 else 0.0

    match_weight_b = sum(_w(w) for w in exact) + sum(_w(w) * 0.8 for w in prefix_b)
    total_weight_b = sum(_w(w) for w in words_b)
    precision = match_weight_b / total_weight_b if total_weight_b > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# TVMaze client
# ---------------------------------------------------------------------------

_tvmaze_cache: dict[str, list[dict]] = {}
_tvmaze_show_id_cache: dict[str, int | None] = {}


def search_tvmaze(series_title: str) -> int | None:
    """Search TVMaze for a show by title. Returns the show ID or None.

    Tries the original title first, then falls back to fuzzy search which
    returns multiple results (useful for German titles).  Results are cached.
    """
    if series_title in _tvmaze_show_id_cache:
        return _tvmaze_show_id_cache[series_title]

    # Try single-search (exact)
    data = _http_get_json(
        "https://api.tvmaze.com/singlesearch/shows",
        params={"q": series_title},
    )
    if data and "id" in data:
        logger.info("TVMaze: found show '%s' (id=%d) for query '%s'", data.get("name"), data["id"], series_title)
        _tvmaze_show_id_cache[series_title] = data["id"]
        return data["id"]

    # Fallback: fuzzy search returning multiple results
    results = _http_get_json(
        "https://api.tvmaze.com/search/shows",
        params={"q": series_title},
    )
    if results and isinstance(results, list) and results:
        show = results[0].get("show", {})
        if show.get("id"):
            logger.info(
                "TVMaze: fuzzy match '%s' (id=%d, score=%.1f) for query '%s'",
                show.get("name"), show["id"], results[0].get("score", 0), series_title,
            )
            _tvmaze_show_id_cache[series_title] = show["id"]
            return show["id"]

    _tvmaze_show_id_cache[series_title] = None
    return None


def get_tvmaze_episodes(show_id: int) -> list[dict]:
    """Fetch all episodes for a TVMaze show. Results are cached per show."""
    cache_key = str(show_id)
    if cache_key in _tvmaze_cache:
        return _tvmaze_cache[cache_key]

    data = _http_get_json(f"https://api.tvmaze.com/shows/{show_id}/episodes")
    episodes = data if isinstance(data, list) else []
    _tvmaze_cache[cache_key] = episodes
    logger.info("TVMaze: fetched %d episodes for show %d", len(episodes), show_id)
    return episodes


def match_episode_tvmaze(series_title: str, metadata: ExtractedMetadata) -> EpisodeMatch | None:
    """Try to identify an episode via TVMaze.

    Searches for the show, fetches all episodes, then uses fuzzy matching
    against the extracted metadata.
    """
    show_id = search_tvmaze(series_title)
    if show_id is None:
        return None

    episodes = get_tvmaze_episodes(show_id)
    if not episodes:
        return None

    return _fuzzy_match_episode(series_title, metadata, episodes, source="tvmaze")


# ---------------------------------------------------------------------------
# TMDB client
# ---------------------------------------------------------------------------

_tmdb_show_cache: dict[str, int | None] = {}
_tmdb_episode_cache: dict[str, list[dict]] = {}


def search_tmdb(series_title: str, api_key: str, language: str = "de-DE") -> int | None:
    """Search TMDB for a TV show. Returns the show ID or None."""
    if series_title in _tmdb_show_cache:
        return _tmdb_show_cache[series_title]

    data = _http_get_json(
        "https://api.themoviedb.org/3/search/tv",
        params={"query": series_title, "api_key": api_key, "language": language},
    )
    if data and data.get("results"):
        show = data["results"][0]
        show_id = show["id"]
        logger.info("TMDB: found show '%s' (id=%d) for query '%s'", show.get("name"), show_id, series_title)
        _tmdb_show_cache[series_title] = show_id
        return show_id
    _tmdb_show_cache[series_title] = None
    return None


def get_tmdb_season_count(show_id: int, api_key: str, language: str = "de-DE") -> int:
    """Get the number of seasons for a show from TMDB."""
    data = _http_get_json(
        f"https://api.themoviedb.org/3/tv/{show_id}",
        params={"api_key": api_key, "language": language},
    )
    if data:
        return data.get("number_of_seasons", 0)
    return 0


def get_tmdb_episodes(show_id: int, season: int, api_key: str, language: str = "de-DE") -> list[dict]:
    """Fetch episodes for a TMDB show season. Results are cached."""
    cache_key = f"{show_id}_{season}"
    if cache_key in _tmdb_episode_cache:
        return _tmdb_episode_cache[cache_key]

    data = _http_get_json(
        f"https://api.themoviedb.org/3/tv/{show_id}/season/{season}",
        params={"api_key": api_key, "language": language},
    )
    episodes = data.get("episodes", []) if data else []
    _tmdb_episode_cache[cache_key] = episodes
    return episodes


def get_all_tmdb_episodes(show_id: int, api_key: str, language: str = "de-DE") -> list[dict]:
    """Fetch all episodes across all seasons from TMDB."""
    season_count = get_tmdb_season_count(show_id, api_key, language)
    all_episodes: list[dict] = []
    for season_num in range(1, season_count + 1):
        episodes = get_tmdb_episodes(show_id, season_num, api_key, language)
        for ep in episodes:
            ep["_season"] = season_num  # normalize field name
        all_episodes.extend(episodes)
    return all_episodes


def match_episode_tmdb(series_title: str, metadata: ExtractedMetadata, api_key: str, language: str = "de-DE") -> EpisodeMatch | None:
    """Try to identify an episode via TMDB.

    Searches for the show, fetches all episodes across seasons, then uses
    fuzzy matching against the extracted metadata.
    """
    show_id = search_tmdb(series_title, api_key, language)
    if show_id is None:
        return None

    episodes = get_all_tmdb_episodes(show_id, api_key, language)
    if not episodes:
        return None

    # Normalize TMDB episode format to match TVMaze-like structure
    normalized: list[dict] = []
    for ep in episodes:
        normalized.append({
            "season": ep.get("_season", ep.get("season_number", 0)),
            "number": ep.get("episode_number", 0),
            "name": ep.get("name", ""),
            "summary": ep.get("overview", ""),
            "runtime": ep.get("runtime") or 0,
        })

    return _fuzzy_match_episode(series_title, metadata, normalized, source="tmdb")


# ---------------------------------------------------------------------------
# Fuzzy matching engine
# ---------------------------------------------------------------------------

def _fuzzy_match_episode(
    series_title: str,
    metadata: ExtractedMetadata,
    episodes: list[dict],
    source: str,
) -> EpisodeMatch | None:
    """Score each episode against the extracted metadata and return the best match.

    Matching strategies (combined):
      1. Title match -- embedded title / TVH subtitle vs episode name (weight 0.55)
      2. Description match -- metadata + TVH description vs episode summary (weight 0.45)

    When no usable title is available, description gets 100% weight.
    Duration is intentionally excluded: TVHeadend recordings have variable
    pre-/post-roll padding that makes runtime comparison unreliable.
    """
    best_match: EpisodeMatch | None = None
    best_score = 0.0

    # Determine the best title candidate for matching
    # TVH subtitle is the episode name from EPG; validate it is not just
    # a repetition of the series title before using it.
    tvh_sub = metadata.tvh_subtitle
    if tvh_sub and _similarity(tvh_sub, series_title) > 0.85:
        tvh_sub = ""  # discard -- it's just the series title repeated

    # Combine all available descriptions
    combined_description = " ".join(filter(None, [
        metadata.description,
        metadata.tvh_description if metadata.tvh_description != metadata.description else "",
    ]))

    # Precompute IDF weights from all episode summaries
    all_summaries = [_strip_html(ep.get("summary") or "") for ep in episodes]
    idf_weights = _compute_idf(all_summaries)

    for ep in episodes:
        ep_name = ep.get("name", "")
        ep_summary = _strip_html(ep.get("summary") or "")
        ep_season = ep.get("season", 0)
        ep_number = ep.get("number", 0)
        ep_runtime_minutes = ep.get("runtime") or 0

        score = 0.0
        weights_sum = 0.0

        # Strategy 1: Title match (weight 0.55)
        title_candidates = [
            metadata.embedded_episode_title,
            metadata.title,
            tvh_sub,
        ]
        title_sims = [
            _similarity(t, ep_name)
            for t in title_candidates if t
        ]
        if title_sims:
            title_sim = max(title_sims)
            score += title_sim * 0.55
            weights_sum += 0.55

        # Strategy 2: Description match (weight 0.45)
        if combined_description and ep_summary:
            seq_sim = _similarity(combined_description, ep_summary)
            # SequenceMatcher below 0.5 is noise on German paraphrased text
            seq_sim = seq_sim if seq_sim >= 0.5 else 0.0
            desc_sim = max(
                seq_sim,
                _keyword_overlap(combined_description, ep_summary),
                _idf_keyword_score(combined_description, ep_summary, idf_weights),
            )
            score += desc_sim * 0.45
            weights_sum += 0.45

        # Normalize score by the weights that were actually applied
        if weights_sum > 0:
            score = score / weights_sum

        if score > best_score:
            best_score = score
            best_match = EpisodeMatch(
                series_title=series_title,
                season_number=ep_season,
                episode_number=ep_number,
                episode_title=ep_name,
                episode_summary=ep_summary,
                confidence=round(score, 3),
                source=source,
            )

    if best_match:
        logger.info(
            "Best %s match: S%02dE%02d '%s' (confidence=%.3f)",
            source, best_match.season_number, best_match.episode_number,
            best_match.episode_title, best_match.confidence,
        )
    return best_match


def _strip_html(text: str) -> str:
    """Remove HTML tags from a string (TVMaze summaries contain <p> tags)."""
    return re.sub(r"<[^>]+>", "", text)


# ---------------------------------------------------------------------------
# Filename pattern fallback
# ---------------------------------------------------------------------------

_SEASON_EPISODE_RE = re.compile(
    r"[Ss](\d{1,2})\s*[Ee](\d{1,2})",
)


def match_from_filename(file_path: Path, series_title: str) -> EpisodeMatch | None:
    """Try to extract S##E## pattern from the filename as a last resort."""
    m = _SEASON_EPISODE_RE.search(file_path.name)
    if m:
        season = int(m.group(1))
        episode = int(m.group(2))
        logger.info("Filename pattern match: S%02dE%02d from '%s'", season, episode, file_path.name)
        return EpisodeMatch(
            series_title=series_title,
            season_number=season,
            episode_number=episode,
            confidence=0.7,
            source="filename",
        )
    return None


# ---------------------------------------------------------------------------
# Combined identification pipeline
# ---------------------------------------------------------------------------

def identify_episode(
    series_title: str,
    metadata: ExtractedMetadata,
    file_path: Path,
    *,
    min_confidence: float = 0.6,
    tmdb_api_key: str = "",
    use_tvmaze: bool = True,
    language: str = "de-DE",
) -> EpisodeMatch | None:
    """Identify an episode using all available methods.

    Order of attempts:
      0. TVH subtitle direct-match against episode lists (high confidence)
      1. TVMaze fuzzy match (free, no key)
      2. TMDB fuzzy match (if api_key provided)
      3. S##E## filename pattern (fallback)

    Returns the best match at or above *min_confidence*, or None.
    """
    candidates: list[EpisodeMatch] = []

    # 0. TVH subtitle direct-match: if TVH provides a subtitle that is
    #    a real episode name (not just the series title), try to find an
    #    exact match in the episode list for a high-confidence result.
    tvh_sub = metadata.tvh_subtitle
    if tvh_sub and _similarity(tvh_sub, series_title) <= 0.85:
        match = _tvh_direct_match(series_title, tvh_sub, tmdb_api_key, use_tvmaze=use_tvmaze, language=language)
        if match:
            candidates.append(match)

    # 1. TVMaze fuzzy match
    if use_tvmaze:
        try:
            match = match_episode_tvmaze(series_title, metadata)
            if match:
                candidates.append(match)
        except Exception as exc:
            logger.warning("TVMaze lookup failed: %s", exc)

    # 2. TMDB fuzzy match
    if tmdb_api_key:
        try:
            match = match_episode_tmdb(series_title, metadata, tmdb_api_key, language)
            if match:
                candidates.append(match)
        except Exception as exc:
            logger.warning("TMDB lookup failed: %s", exc)

    # 3. Filename pattern
    match = match_from_filename(file_path, series_title)
    if match:
        candidates.append(match)

    # Pick best candidate
    candidates.sort(key=lambda m: m.confidence, reverse=True)
    if candidates:
        best = candidates[0]
        logger.info(
            "Best episode match: %s S%02dE%02d '%s' (confidence=%.3f, source=%s)",
            best.series_title, best.season_number, best.episode_number,
            best.episode_title, best.confidence, best.source,
        )
        return best

    logger.info("No episode match found for %s", file_path.name)
    return None


def _tvh_direct_match(
    series_title: str,
    tvh_subtitle: str,
    tmdb_api_key: str = "",
    *,
    use_tvmaze: bool = True,
    language: str = "de-DE",
) -> EpisodeMatch | None:
    """Try to match a TVH subtitle (episode name) directly against episode lists.

    Searches TVMaze (and optionally TMDB) episode lists for an episode whose
    name closely matches *tvh_subtitle*.  Returns a match with boosted
    confidence if a good match is found.
    """
    threshold = 0.80

    # Try TVMaze first
    if use_tvmaze:
        show_id = search_tvmaze(series_title)
        if show_id is not None:
            episodes = get_tvmaze_episodes(show_id)
            for ep in episodes:
                ep_name = ep.get("name", "")
                sim = _similarity(tvh_subtitle, ep_name)
                if sim >= threshold:
                    confidence = min(1.0, 0.80 + sim * 0.20)
                    logger.info(
                        "TVH direct match (TVMaze): '%s' ~ '%s' (sim=%.3f, conf=%.3f)",
                        tvh_subtitle, ep_name, sim, confidence,
                    )
                    return EpisodeMatch(
                        series_title=series_title,
                        season_number=ep.get("season", 0),
                        episode_number=ep.get("number", 0),
                        episode_title=ep_name,
                        episode_summary=_strip_html(ep.get("summary") or ""),
                        confidence=round(confidence, 3),
                        source="tvh+tvmaze",
                    )

    # Try TMDB if available
    if tmdb_api_key:
        tmdb_id = search_tmdb(series_title, tmdb_api_key, language)
        if tmdb_id is not None:
            episodes = get_all_tmdb_episodes(tmdb_id, tmdb_api_key, language)
            for ep in episodes:
                ep_name = ep.get("name", "")
                sim = _similarity(tvh_subtitle, ep_name)
                if sim >= threshold:
                    confidence = min(1.0, 0.80 + sim * 0.20)
                    logger.info(
                        "TVH direct match (TMDB): '%s' ~ '%s' (sim=%.3f, conf=%.3f)",
                        tvh_subtitle, ep_name, sim, confidence,
                    )
                    return EpisodeMatch(
                        series_title=series_title,
                        season_number=ep.get("_season", ep.get("season_number", 0)),
                        episode_number=ep.get("episode_number", 0),
                        episode_title=ep_name,
                        episode_summary=ep.get("overview", ""),
                        confidence=round(confidence, 3),
                        source="tvh+tmdb",
                    )

    return None
