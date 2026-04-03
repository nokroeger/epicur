# epicur – Episode Curator

Automatically identify and organize TV recordings into season folders.

epicur scans a directory of TV series recordings, extracts metadata via
ffprobe and TVHeadend EPG data, identifies episodes through TVMaze and TMDB,
and renames/organizes files into a clean `Season N/` structure.

## Features

- **TVHeadend integration** – reads DVR log entries for EPG subtitle,
  description, and channel info
- **Multi-source matching** – TVMaze (free) and TMDB (German episode names)
- **IDF-weighted keyword matching** – handles German paraphrased descriptions
- **Interactive review** – manually assign unmatched recordings via CLI
- **Dry-run mode** – preview all changes before committing
- **Duplicate detection** – safely handles re-recordings

## Requirements

- Python ≥ 3.10
- No python packages beyond the standard library
- `ffprobe` (part of ffmpeg) installed on the system
- Optional: TMDB API key for German episode metadata

## Installation

epicur can be installed via pip, but this is optional.

```bash
pip install .
```

Or in development mode:

```bash
pip install -e .
```

## Usage

### Without installation

You can run epicur directly from the source tree without pip:

```bash
python3 epicur.py ~/recordings --tmdb-api-key YOUR_KEY
```

### After installation via pip

```bash
# Scan and organize recordings
epicur ~/recordings --tmdb-api-key YOUR_KEY

# Dry run (preview only)
epicur ~/recordings --tmdb-api-key YOUR_KEY --dry-run

# Interactive review of unmatched files
epicur ~/recordings --review

# Disable TVMaze (use TMDB only)
epicur ~/recordings --tmdb-api-key YOUR_KEY --no-tvmaze

# Use English TMDB metadata (automatically disables TVMaze for non-English)
epicur ~/recordings --tmdb-api-key YOUR_KEY --language en-US

# Adjust confidence threshold
epicur ~/recordings --tmdb-api-key YOUR_KEY --confidence 0.4
```

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

After processing:

```
~/recordings/
  Star Trek Lower Decks/
    Season 1/
      Star Trek_ Lower Decks S01E04.ts
    Season 4/
      Star Trek_ Lower Decks S04E07.ts
```

## License

MIT
