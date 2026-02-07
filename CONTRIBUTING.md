# Contributing to Obsidian Linker

Thank you for your interest in contributing. This guide will help you get started.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a branch for your changes

```bash
git clone https://github.com/YOUR_USERNAME/obsidian-linker.git
cd obsidian-linker
git checkout -b my-feature
```

## Development Setup

Install the dependencies:

```bash
pip install pytesseract Pillow
brew install tesseract  # macOS
```

Test against a sample vault:

```bash
python obsidian_linker.py --vault-path "/path/to/test/vault" --verbose
```

## Architecture

The entire tool is a single Python file (`obsidian_linker.py`) by design. This keeps it portable and easy to use. When contributing, please keep all logic in this single file unless there is a strong reason to split it.

### Key sections of the script

| Section | Purpose |
|---------|---------|
| Data Structures | `NoteRecord` and `VaultGraph` dataclasses |
| Phase 1A: Scan | Walk vault, parse `.md` files, detect frontmatter |
| Phase 1B: OCR | Extract text from embedded images via Tesseract |
| Phase 1C: Entity Discovery | Auto-discover topics, acronyms, and people |
| Phase 1D: Classification | Classify notes by type (meeting, strategic, etc.) |
| Phase 1E: Similarity | Compute pairwise similarity and determine links |
| Phase 1F: Tags | Assign hierarchical tags to each note |
| Phase 1G: Report | Print analysis summary |
| Phase 2: Apply | Backup, inject frontmatter, add Related Notes, verify integrity |

## Areas for Contribution

Here are some areas where help is welcome:

### Entity Discovery Improvements
- Better handling of non-English names
- Support for additional entity types (locations, dates, projects)
- NLP-based entity extraction as an optional enhancement

### Interlinking Algorithm
- Additional similarity signals (shared headings, shared links)
- Configurable weights via CLI flags
- Clustering algorithms for grouping related notes

### OCR Enhancements
- Support for additional image formats beyond PNG
- Caching OCR results to speed up re-runs
- Support for PDF attachments

### Output Formats
- Obsidian Dataview compatible metadata
- Export analysis report to markdown or HTML
- Graph visualization of note relationships

### Testing
- Unit tests for entity discovery
- Unit tests for similarity computation
- Integration tests with sample vaults

### Documentation
- More usage examples
- Tutorial for common vault structures

## How to Submit Changes

1. Make your changes on a feature branch
2. Test against a real Obsidian vault (use `--verbose` dry-run mode)
3. Verify that existing functionality still works (no content corruption, correct tag format)
4. Submit a pull request with a clear description of what changed and why

## Pull Request Guidelines

- Keep PRs focused on a single change
- Include a description of what the change does and why
- If adding a new feature, include usage examples in the PR description
- If fixing a bug, describe how to reproduce it
- Test with both `--apply` and dry-run modes

## Code Style

- Follow existing code patterns in the script
- Use type hints for function signatures
- Add comments for non-obvious logic
- Keep the single-file architecture unless there is a compelling reason to change it

## Reporting Issues

When reporting a bug, please include:

- Python version (`python --version`)
- Operating system
- The command you ran
- The error message or unexpected behavior
- A sanitized example of a vault file that triggers the issue (remove any sensitive content)

## Code of Conduct

Be respectful and constructive. We are all here to build something useful.
