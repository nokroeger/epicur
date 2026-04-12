# Copilot Workspace Instructions for epicur

## Übersicht

Dieses Projekt ist ein CLI-Tool zur automatischen Identifikation, Organisation und Nachbearbeitung von TV-Serienaufnahmen für Kodi-Medienbibliotheken. Es nutzt Metadaten aus TVHeadend, ffprobe, TVMaze und TMDB, erkennt Episoden, benennt Dateien um, verschiebt sie in Season-Ordner und kann Werbeblöcke mit comskip erkennen und Kapitelmarken setzen.

## Build- und Test-Befehle

- **Installation (optional):**
  ```bash
  pip install .
  # oder für Entwicklung:
  pip install -e .
  ```
- **Direkt aus dem Quellcode ausführen:**
  ```bash
  python3 epicur.py <subcommand> <recordings-dir> [optionen]
  ```
- **Tests ausführen:**
  ```bash
  pytest
  ```

## Wichtige Konventionen

- **Keine externen Python-Abhängigkeiten:** Nur Standardbibliothek wird verwendet.
- **Unterstützte Videoformate:** .ts, .mp4, .mkv (konfigurierbar)
- **Subcommands:**
  - `recognize`: Scannt, erkennt und organisiert Episoden
  - `review`: Interaktive Nachbearbeitung nicht erkannter Aufnahmen
  - `postprocess`: Konvertiert Staffeln, erkennt Werbung, verschiebt in Bibliothek
  - `movie-postprocess`: Für Filme, flache Bibliotheksstruktur
- **Konfigurierbare Optionen:** Siehe `epicur <subcommand> --help` für alle Parameter (API-Keys, Verzeichnisse, Sprache, etc.)
- **Dry-Run:** Mit `--dry-run` werden keine Änderungen vorgenommen, sondern nur simuliert.

## Typische Stolperfallen

- **Fehlende Systemtools:** ffprobe/ffmpeg und optional comskip müssen installiert und im PATH sein.
- **API-Keys:** Für TMDB (deutsche Metadaten) ist ein API-Key nötig. TVMaze funktioniert nur für englische Metadaten.
- **Dateialter:** Standardmäßig werden nur Dateien verarbeitet, die älter als 5 Minuten sind (`--min-age`).
- **Verzeichnisstruktur:**
  - Rohaufnahmen: `<root>/<Serie>/<Dateien>`
  - Nach `recognize`: `<root>/<Serie>/Season N/<Dateien>`
  - Nach `postprocess`: `<library>/<Serie>/Season N/<Dateien>.mp4`

## Wichtige Dateien & Verzeichnisse

- `src/epicur/`: Hauptlogik (main.py, episode_identifier.py, ...)
- `tests/`: Pytest-Tests
- `pyproject.toml`: Build- und Metadaten
- `README.md`: Ausführliche Anleitung und Beispiele

## Weiterführende Doku

- Siehe [README.md](../README.md) für vollständige Feature- und Optionsbeschreibung.
- [Comskip](https://github.com/erikkaashoek/Comskip) für Werbeerkennung.

## Beispiel-Prompts

- "Führe alle Tests aus."
- "Starte recognize im Dry-Run für ~/recordings."
- "Wie kann ich comskip für die Nachbearbeitung konfigurieren?"
- "Welche Optionen gibt es für epicur postprocess?"

## Vorschlag für weitere Agent-Customizations

- **applyTo:**
  - `src/epicur/` → Python-Logik, keine externen Abhängigkeiten
  - `tests/` → Test-spezifische Regeln (z.B. keine Netzwerkaufrufe, nur Mocking)
- **Eigener Agent für Comskip-Integration:**
  - Automatisches Prüfen, ob comskip installiert ist und wie es aufgerufen wird
  - Vorschläge für comskip.ini-Konfiguration

---

*Diese Datei wurde automatisch generiert. Bitte bei Änderungen an Build-/Test-Workflows oder Konventionen aktualisieren.*
