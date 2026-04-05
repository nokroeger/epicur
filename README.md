# epicur – Episode Curator

Automatically identify, organize, and postprocess TV recordings into a
Kodi-ready media library.

epicur scans a directory of TV series recordings, extracts metadata via
ffprobe and TVHeadend EPG data, identifies episodes through TVMaze and TMDB,
renames/organizes files into a clean `Season N/` structure, and converts
completed seasons to `.mp4` with commercial-skipping chapter markers.

## Features

- **TVHeadend integration** – reads DVR log entries for EPG subtitle,
  description, and channel info
- **Multi-source matching** – TVMaze (english only) and TMDB (for any other language), requires a an API key which can be obtained free of charge for personal use
- **IDF-weighted keyword matching** – handles paraphrased descriptions
- **Interactive review** – manually assign unmatched recordings via CLI
- **Postprocessing** – detect commercials via comskip, convert `.ts` to `.mp4`
  with chapter markers, and move completed seasons to a Kodi media library
- **Dry-run mode** – preview all changes before committing
- **Duplicate detection** – safely handles re-recordings

## Requirements

- Python ≥ 3.10
- No python packages beyond the standard library
- `ffprobe` / `ffmpeg` installed on the system
- Optional: [comskip](https://github.com/erikkaashoek/Comskip) for commercial
  detection during postprocessing
- Optional: TMDB API key for German episode metadata

## Installation

epicur can be installed via pip, but this is optional. See usage below which provides instructions for both installed and uninstalled mode. 

```bash
pip install .
```

Or in development mode:

```bash
pip install -e .
```

## Usage

epicur uses three subcommands: `recognize`, `review`, and `postprocess`.

### After installation via pip

```bash
epicur <subcommand> ~/recordings [options]
```

### Without installation

You can run epicur directly from the source tree without pip:

```bash
python3 epicur.py <subcommand> ~/recordings [options]
```

All examples below use the installed form. Replace `epicur` with
`python3 epicur.py` if running from source. 

The following sections show typical examples for the subcommands. For a full list of options, run

`epicur <subcommand> --help`

### recognize

Scan for new recordings, identify episodes via TVMaze/TMDB, and organize
them into `Season N/` folders.

```bash
# Scan and organize recordings
epicur recognize ~/recordings --tmdb-api-key YOUR_KEY

# Dry run (preview only)
epicur recognize ~/recordings --tmdb-api-key YOUR_KEY --dry-run

# Include the kodi library in duplicate detection (recommended)
epicur recognize ~/recordings --tmdb-api-key YOUR_KEY --library-dir /media/tv

# Disable TVMaze (use TMDB only)
epicur recognize ~/recordings --tmdb-api-key YOUR_KEY --no-tvmaze

# Use English TMDB metadata (automatically disables TVMaze for non-English)
epicur recognize ~/recordings --tmdb-api-key YOUR_KEY --language en-US

# Adjust confidence threshold
epicur recognize ~/recordings --tmdb-api-key YOUR_KEY --confidence 0.4
```

### review

Interactively review and manually assign recordings that could not be
matched automatically.

```bash
epicur review ~/recordings
```

### postprocess

Find seasons where all episodes are present, detect commercials with
comskip, convert `.ts` files to `.mp4` (H.264/AAC) with chapter markers
that let players skip commercials, and move the result to the library.

```bash
# Convert and move completed seasons to library
epicur postprocess ~/recordings --library-dir /media/tv --tmdb-api-key YOUR_KEY

# Preview what would be processed
epicur postprocess ~/recordings --library-dir /media/tv --tmdb-api-key YOUR_KEY --dry-run

# Custom encoding settings
epicur postprocess ~/recordings --library-dir /media/tv --tmdb-api-key YOUR_KEY \
  --crf 18 --preset medium

# Use a custom comskip configuration
epicur postprocess ~/recordings --library-dir /media/tv --tmdb-api-key YOUR_KEY \
  --comskip-ini /path/to/comskip.ini
```

If comskip is not installed or fails, episodes are converted without chapter
markers.

## Directory structure

epicur expects recordings organized as:

```
~/recordings/
  Star Trek Lower Decks/
    Star Trek_ Lower Decks.ts
    Star Trek_ Lower Decks-2.ts
  JAG/
    J.A.G. - Im Auftrag der Ehre.ts
    J.A.G. - Im Auftrag der Ehre-1.ts
```

After `recognize`:

```
~/recordings/
  Star Trek Lower Decks/
    Season 1/
      Star Trek_ Lower Decks S01E04.ts
    Season 4/
      Star Trek_ Lower Decks S04E07.ts
```

After `postprocess` (with `--library-dir /media/tv`):

```
/media/tv/
  Star Trek Lower Decks/
    Season 1/
      Star Trek_ Lower Decks S01E04.mp4
    Season 4/
      Star Trek_ Lower Decks S04E07.mp4
```

## License

MIT
