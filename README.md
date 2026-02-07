# Obsidian Linker

Automatically analyze any [Obsidian](https://obsidian.md) vault, discover topics and people, then apply YAML frontmatter tags and `[[wikilink]]` interlinks to all markdown files.

Obsidian Linker is a single portable Python script that works on any vault. It requires no configuration files, no hardcoded entity lists, and no manual tagging. Point it at a vault and it figures out the rest.

## Features

- **Fully automatic entity discovery** -- discovers topics, acronyms, and people from your vault content itself. No hardcoded lists.
- **Smart person detection** -- recognizes `Firstname Lastname` patterns, handles ambiguous single names using context-based disambiguation, and merges typo variants while keeping genuinely different people separate.
- **OCR for image-heavy notes** -- extracts text from embedded Obsidian screenshots (`![[Pasted image ...]]`) using Tesseract OCR to improve tagging of image-only files.
- **Hierarchical tag taxonomy** -- generates `source/`, `project/`, `area/`, `type/`, `topic/`, and `person/` tags from your folder structure and content.
- **Intelligent interlinking** -- computes pairwise similarity between notes using topic overlap, people overlap, and folder proximity, then adds `[[wikilink]]` sections.
- **Note type classification** -- automatically classifies notes as meeting, one-on-one, strategic, script, analysis, reference, or stub.
- **Auto-titling** -- generates descriptive titles for untitled documents and renames them.
- **Safe by default** -- dry-run mode is the default. Must pass `--apply` to write changes. Full backup is created before any modifications.
- **Content integrity verification** -- after each file write, verifies that all image embeds and callout blocks are preserved.
- **Idempotent** -- re-running the script won't duplicate Related Notes sections.

## How It Works

The script operates in two phases:

**Phase 1 -- Analysis (always runs)**
1. Scans the vault for all `.md` files (skips `.excalidraw.md`)
2. Runs OCR on embedded images to extract text from screenshots
3. Discovers entities: extracts topics, acronyms, and people names from all files
4. Classifies each note by type (meeting, strategic, script, etc.)
5. Computes pairwise similarity between notes
6. Assigns tags and determines interlinks
7. Prints a detailed analysis report

**Phase 2 -- Apply (only with `--apply` flag)**
1. Creates a timestamped backup of the entire vault
2. Injects YAML frontmatter tags into each file
3. Appends a `## Related Notes` section with `[[wikilinks]]`
4. Renames untitled documents with auto-generated titles
5. Verifies content integrity after each write
6. Writes a JSON changelog

## Installation

### Prerequisites

- Python 3.9 or higher
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (optional, for image text extraction)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install Tesseract (optional, for OCR)

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr
```

**Windows:**

Download the installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add it to your PATH.

If Tesseract is not installed, the script will still work -- it will just skip OCR and rely on text content and filenames for tagging image-heavy files. You can also explicitly skip OCR with the `--skip-ocr` flag.

## Usage

### Dry run (recommended first step)

Analyze the vault and print a report without modifying any files:

```bash
python obsidian_linker.py --vault-path "/path/to/your/vault"
```

### Dry run with verbose per-file detail

```bash
python obsidian_linker.py --vault-path "/path/to/your/vault" --verbose
```

### Apply changes

Create a backup and apply tags and interlinks:

```bash
python obsidian_linker.py --vault-path "/path/to/your/vault" --apply
```

### Skip OCR (faster, no Tesseract needed)

```bash
python obsidian_linker.py --vault-path "/path/to/your/vault" --apply --skip-ocr
```

### Custom options

```bash
python obsidian_linker.py \
  --vault-path "/path/to/your/vault" \
  --apply \
  --backup-dir ./my-backups \
  --min-similarity 0.4 \
  --verbose
```

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--vault-path` | *(required)* | Path to the Obsidian vault root directory |
| `--apply` | `false` | Apply changes to files (default is dry-run) |
| `--backup-dir` | sibling to vault | Directory for the timestamped backup |
| `--skip-backup` | `false` | Skip creating a backup before applying |
| `--skip-ocr` | `false` | Skip OCR processing of images |
| `--min-similarity` | `0.35` | Minimum similarity score for interlinking (0.0-1.0) |
| `--verbose` | `false` | Print per-file detail in the analysis report |

## Tag Taxonomy

The script generates a hierarchical tag system:

| Prefix | Source | Example |
|--------|--------|---------|
| `source/` | Top-level folder | `source/projects`, `source/areas` |
| `project/` | Subfolder under Projects | `project/habp`, `project/ai-agents-learning` |
| `area/` | Subfolder under Areas | `area/legal`, `area/l-and-d` |
| `type/` | Content classification | `type/meeting`, `type/strategic`, `type/stub` |
| `topic/` | Auto-discovered topics | `topic/dlt`, `topic/landd`, `topic/api` |
| `person/` | Auto-discovered people | `person/john-oghia`, `person/karla-martinez` |

## Similarity Algorithm

Interlinking uses a weighted Jaccard similarity with three signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| Topic overlap | 40% | Shared topics between two notes |
| People overlap | 35% | Shared people mentioned in both notes |
| Folder proximity | 25% | Same subfolder (1.0), same top-level (0.5), different (0.0) |

Notes are linked when their combined similarity score meets or exceeds the threshold (default 0.35). Each note gets up to 5 related notes, sorted by score.

## Backup and Recovery

By default, the script creates a full timestamped backup before applying any changes:

```
vault_backup_YYYYMMDD_HHMMSS/
```

The backup is placed as a sibling directory to your vault (or in the location specified by `--backup-dir`). To restore, simply replace your vault folder with the backup.

A JSON changelog is also written after each run:

```
linker_changelog_YYYYMMDD_HHMMSS.json
```

## Limitations

- Entity discovery uses regex-based heuristics, not NLP models. It works well for structured meeting notes but may miss entities in free-form prose.
- Person detection requires a known first name from the built-in common names list (~200 names). Uncommon first names may not be detected.
- OCR quality depends on screenshot resolution and Tesseract's ability to read the text.
- The script does not parse Obsidian plugins, dataview queries, or templater syntax.
- Excalidraw files (`.excalidraw.md`) are intentionally skipped as they have specialized frontmatter.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
