#!/usr/bin/env python3
from __future__ import annotations

"""
Obsidian Vault Auto-Tagger & Interlinker

Automatically analyzes any Obsidian vault, discovers topics/people/patterns,
then applies YAML frontmatter tags and [[wikilink]] interlinks to all markdown files.

Usage:
    # Dry run (default)
    python obsidian_linker.py --vault-path "/path/to/vault"

    # Apply changes
    python obsidian_linker.py --vault-path "/path/to/vault" --apply

    # Skip OCR (faster, no Tesseract needed)
    python obsidian_linker.py --vault-path "/path/to/vault" --apply --skip-ocr
"""

import argparse
import json
import logging
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Set

# Optional OCR dependencies
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class NoteRecord:
    filepath: Path
    filename: str                                   # stem without .md
    relative_path: str
    folder_chain: list[str] = field(default_factory=list)
    raw_content: str = ""
    text_content: str = ""                          # stripped of image embeds
    ocr_text: str = ""                              # text from OCR on images
    combined_text: str = ""                         # text_content + ocr_text
    has_frontmatter: bool = False
    is_excalidraw: bool = False
    is_image_only: bool = False
    topics: set[str] = field(default_factory=set)
    people: set[str] = field(default_factory=set)
    note_type: str = ""
    tags: list[str] = field(default_factory=list)
    related_notes: list[str] = field(default_factory=list)
    image_embeds: list[str] = field(default_factory=list)


@dataclass
class VaultGraph:
    vault_path: Path = field(default_factory=Path)
    notes: dict[str, NoteRecord] = field(default_factory=dict)
    topic_index: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    people_index: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    folder_index: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    type_index: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # Auto-discovered vocabularies
    discovered_topics: dict[str, list[str]] = field(default_factory=dict)
    discovered_people: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 1A: Scan & Parse
# ---------------------------------------------------------------------------

IMAGE_EMBED_PATTERN = re.compile(r'!\[\[([^\]]+\.(png|jpg|jpeg|gif|bmp|svg|webp))\]\]', re.IGNORECASE)
FRONTMATTER_PATTERN = re.compile(r'^---\s*\n', re.MULTILINE)
CALLOUT_CONTENT_PATTERN = re.compile(r'>\s*\[!([^\]]*)\][-+]?\s*(.*)', re.MULTILINE)


def scan_vault(vault_path: Path) -> list[NoteRecord]:
    """Walk the vault directory and create a NoteRecord for each .md file."""
    notes = []
    for md_file in sorted(vault_path.rglob("*.md")):
        rel = md_file.relative_to(vault_path)
        filename = md_file.stem

        # Skip excalidraw files
        if ".excalidraw" in md_file.name:
            logging.debug(f"Skipping excalidraw: {rel}")
            continue

        # Skip the script itself
        if md_file.name == "obsidian_linker.py":
            continue

        note = NoteRecord(
            filepath=md_file,
            filename=filename,
            relative_path=str(rel),
        )

        # Extract folder chain
        parts = list(rel.parts[:-1])  # exclude the filename
        note.folder_chain = parts

        # Read content
        try:
            note.raw_content = md_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as e:
            logging.warning(f"Could not read {rel}: {e}")
            continue

        # Detect existing frontmatter
        if note.raw_content.startswith("---"):
            note.has_frontmatter = True

        # Extract image embeds
        note.image_embeds = IMAGE_EMBED_PATTERN.findall(note.raw_content)

        # Build text_content by stripping image embed lines and callout markers
        lines = note.raw_content.split("\n")
        text_lines = []
        in_frontmatter = False
        frontmatter_count = 0
        for line in lines:
            stripped = line.strip()
            if stripped == "---":
                frontmatter_count += 1
                if frontmatter_count <= 2 and note.has_frontmatter:
                    in_frontmatter = frontmatter_count == 1
                    continue
            if in_frontmatter:
                continue
            # Skip pure image embed lines
            if IMAGE_EMBED_PATTERN.match(stripped):
                continue
            # Strip callout prefix but keep content
            callout_match = re.match(r'^>\s*\[!([^\]]*)\][-+]?\s*(.*)', stripped)
            if callout_match:
                text_lines.append(callout_match.group(1) + " " + callout_match.group(2))
                continue
            # Strip blockquote prefix
            if stripped.startswith("> "):
                text_lines.append(stripped[2:])
                continue
            text_lines.append(stripped)

        note.text_content = "\n".join(text_lines).strip()

        # Detect image-only
        clean_text = re.sub(r'\s+', '', note.text_content)
        if len(clean_text) < 20 and len(note.image_embeds) > 0:
            note.is_image_only = True

        notes.append(note)

    return notes


# ---------------------------------------------------------------------------
# Phase 1B: OCR for Image-Heavy Files
# ---------------------------------------------------------------------------

def find_image_file(vault_path: Path, note_dir: Path, image_name: str) -> Optional[Path]:
    """Locate an embedded image file by searching common Obsidian attachment locations."""
    # Search order: same dir, attachments subdir, vault-level attachments, recursive search
    candidates = [
        note_dir / image_name,
        note_dir / "attachments" / image_name,
        vault_path / "attachments" / image_name,
    ]
    # Also check parent directory attachments
    if note_dir != vault_path:
        candidates.append(note_dir.parent / "attachments" / image_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback: search the entire vault
    for found in vault_path.rglob(image_name):
        return found

    return None


def perform_ocr(vault_path: Path, notes: list[NoteRecord], skip_ocr: bool) -> None:
    """Extract text from embedded images using OCR."""
    if skip_ocr:
        logging.info("OCR skipped (--skip-ocr flag)")
        return

    if not OCR_AVAILABLE:
        logging.warning(
            "OCR dependencies not available. Install with: pip install pytesseract Pillow\n"
            "Also install Tesseract: brew install tesseract (macOS)"
        )
        return

    # Test if tesseract binary is available
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        logging.warning(
            "Tesseract binary not found. Install with: brew install tesseract (macOS)"
        )
        return

    ocr_count = 0
    for note in notes:
        if not note.image_embeds:
            continue

        ocr_texts = []
        note_dir = note.filepath.parent

        for embed_tuple in note.image_embeds:
            image_name = embed_tuple[0] if isinstance(embed_tuple, tuple) else embed_tuple
            image_path = find_image_file(vault_path, note_dir, image_name)

            if image_path is None:
                logging.debug(f"Image not found: {image_name} (referenced in {note.relative_path})")
                continue

            try:
                img = Image.open(image_path)
                text = pytesseract.image_to_string(img)
                if text.strip():
                    ocr_texts.append(text.strip())
                    ocr_count += 1
            except Exception as e:
                logging.debug(f"OCR failed for {image_path}: {e}")

        if ocr_texts:
            note.ocr_text = "\n".join(ocr_texts)

    logging.info(f"OCR extracted text from {ocr_count} images")

    # Update combined_text for all notes
    for note in notes:
        note.combined_text = (note.text_content + "\n" + note.ocr_text).strip()


# ---------------------------------------------------------------------------
# Phase 1C: Auto-Discovery Engine
# ---------------------------------------------------------------------------

# Patterns for entity extraction
NAME_PATTERN = re.compile(r'\b([A-Z][a-z]{1,15})\s+([A-Z][a-z]{1,20})\b')
SINGLE_CAPITALIZED = re.compile(r'\b([A-Z][a-z]{2,15})\b')
ACRONYM_PATTERN = re.compile(r'\b([A-Z][A-Za-z&]{1,8})\b')

# Common English words to filter out of entity detection
STOP_WORDS = {
    # Function words and basic verbs
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "his", "how", "its", "may",
    "new", "now", "old", "see", "way", "who", "did", "get", "got", "let",
    "say", "she", "too", "use", "with", "this", "that", "from", "they",
    "been", "have", "many", "some", "them", "than", "each", "make",
    "like", "long", "look", "come", "could", "more", "into", "time",
    "very", "when", "what", "your", "about", "would", "there", "their",
    "which", "could", "other", "after", "first", "also", "back",
    "these", "those", "then", "just", "only", "where", "most",
    "know", "take", "people", "year", "every", "good", "give", "well",
    "want", "because", "still", "over", "think", "here", "work",
    "will", "much", "being", "need", "before", "should", "through",
    "between", "both", "same", "going", "right", "while", "last",
    "at", "in", "on", "to", "of", "do", "if", "it", "is", "or", "so",
    "up", "no", "an", "be", "by", "as", "he", "we", "my", "us", "am",
    # Common nouns/adjectives that appear capitalized at sentence starts
    "learn", "learning", "skill", "skills", "digital", "business",
    "team", "product", "partner", "platform", "strategy", "change",
    "impact", "value", "power", "direct", "cross", "low", "high",
    "future", "role", "share", "help", "connect", "table", "experience",
    "leaders", "colleagues", "colleague", "agent", "agents", "client",
    "cost", "content", "chief", "health", "meet", "better", "app",
    "data", "model", "system", "process", "service", "case", "code",
    "design", "build", "plan", "show", "talk", "line", "vision",
    "solutions", "services", "management", "development", "building",
    "coaching", "assessment", "mapping", "strategy", "inference",
    "orientation", "coordinator", "path", "style", "journey",
    "integration", "ranking", "relevance", "readiness", "event",
    "transactions", "launch", "comms", "spotlights", "registry",
    "features", "complexity", "discoverability", "foundations",
    "profeciency", "enablement", "functional", "scale", "create",
    "rating", "apps", "user", "working", "global", "based", "level",
    "open", "full", "current", "real", "best", "next", "part", "end",
    "own", "set", "big", "small", "large", "early", "late", "hard",
    "soft", "clear", "close", "fast", "slow", "free", "sure", "safe",
    # Common markdown/formatting words
    "todo", "note", "notes", "update", "updates", "meeting", "action",
    "item", "items", "discussion", "summary", "agenda", "review",
    "steps", "follow", "key", "point", "points", "question",
    "answer", "topic", "topics", "section", "page", "link", "file",
    # Common date/time words
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday",
    "today", "tomorrow", "yesterday", "week", "month",
    # Obsidian / formatting
    "pasted", "image", "untitled", "drawing", "canvas",
    # Common capitalized words that aren't proper nouns
    "great", "important", "interesting", "different", "possible",
    "really", "actually", "probably", "especially", "however",
    "although", "perhaps", "indeed", "certainly", "areas",
    "projects", "resources", "tasks", "type", "source",
    "project", "area", "resource", "task",
}

STOP_WORDS_LOWER = {w.lower() for w in STOP_WORDS}


# Common first names to help identify actual person names vs noun phrases
COMMON_FIRST_NAMES = {
    "james", "john", "robert", "michael", "william", "david", "richard",
    "joseph", "thomas", "charles", "christopher", "daniel", "matthew",
    "anthony", "mark", "donald", "steven", "paul", "andrew", "joshua",
    "kenneth", "kevin", "brian", "george", "timothy", "ronald", "edward",
    "jason", "jeffrey", "ryan", "jacob", "gary", "nicholas", "eric",
    "jonathan", "stephen", "larry", "justin", "scott", "brandon",
    "benjamin", "samuel", "raymond", "gregory", "frank", "alexander",
    "patrick", "jack", "dennis", "jerry", "tyler", "aaron", "jose",
    "adam", "nathan", "henry", "peter", "zachary", "douglas", "harold",
    "carl", "arthur", "gerald", "roger", "keith", "jeremy", "terry",
    "lawrence", "sean", "albert", "joe", "christian", "austin", "jesse",
    "ethan", "dylan", "bryan", "louis", "russell", "vincent", "philip",
    "bobby", "johnny", "bradley", "roy", "eugene", "randy", "wayne",
    "alan", "ralph", "gabriel", "bruce", "willie", "fred", "billy",
    "mary", "patricia", "jennifer", "linda", "barbara", "elizabeth",
    "susan", "jessica", "sarah", "karen", "lisa", "nancy", "betty",
    "margaret", "sandra", "ashley", "dorothy", "kimberly", "emily",
    "donna", "michelle", "carol", "amanda", "melissa", "deborah",
    "stephanie", "rebecca", "sharon", "laura", "cynthia", "kathleen",
    "amy", "angela", "shirley", "anna", "brenda", "pamela", "emma",
    "nicole", "helen", "samantha", "katherine", "christine", "debra",
    "rachel", "carolyn", "janet", "catherine", "maria", "heather",
    "diane", "ruth", "julie", "olivia", "joyce", "virginia", "victoria",
    "kelly", "lauren", "christina", "joan", "evelyn", "judith",
    "megan", "andrea", "cheryl", "hannah", "jacqueline", "martha",
    "gloria", "teresa", "ann", "sara", "madison", "frances", "kathryn",
    "janice", "jean", "abigail", "alice", "judy", "sophia", "grace",
    "denise", "amber", "doris", "marilyn", "danielle", "beverly",
    "isabella", "theresa", "diana", "natalie", "brittany", "charlotte",
    "marie", "kayla", "alexis", "lori", "jane", "elaine", "alison",
}


# Words that commonly follow names but aren't last names
NOT_LAST_NAMES = {
    "comments", "comment", "investigated", "investigation", "pickup",
    "update", "updates", "meeting", "notes", "note", "report",
    "discussion", "session", "review", "summary", "feedback",
    "agenda", "action", "actions", "items", "points", "overview",
    "debrief", "roundtable", "offsite", "workshop", "presentation",
    "proposal", "document", "draft", "final", "version", "copy",
    "response", "request", "reply", "email", "message", "call",
    "group", "team", "leads", "lead", "manager", "director",
    "mentioned", "suggested", "said", "asked", "told", "shared",
    "believes", "thinks", "wants", "needs", "feels", "supports",
    "don", "emma", "spring", "fall", "winter", "summer",
    "related", "based", "driven", "focused", "oriented",
}


def extract_candidate_names(text: str) -> list[str]:
    """Extract potential person names (Firstname Lastname patterns).

    Only matches where the first word is a known first name, to avoid
    matching noun phrases like 'Learning Strategy' or 'Digital Foundations'.
    """
    matches = NAME_PATTERN.findall(text)
    names = []
    for first, last in matches:
        first_lower = first.lower()
        last_lower = last.lower()
        # Both words must not be stop words
        if first_lower in STOP_WORDS_LOWER or last_lower in STOP_WORDS_LOWER:
            continue
        # First word must be a known first name
        if first_lower not in COMMON_FIRST_NAMES:
            continue
        # Last word must not be a common non-name word
        if last_lower in NOT_LAST_NAMES:
            continue
        name = f"{first} {last}"
        names.append(name)
    return names


def extract_candidate_topics(text: str, folder_chain: list[str], filename: str) -> list[str]:
    """Extract candidate topic phrases from content, folders, and filename."""
    candidates = []

    # From folder names (strip numeric prefixes)
    for folder in folder_chain:
        clean = re.sub(r'^\d+\.\s*', '', folder).strip()
        if clean and clean.lower() not in STOP_WORDS_LOWER:
            candidates.append(clean)

    # From filename (split on common separators)
    fn_clean = re.sub(r'^\d{6,8}\s*', '', filename)  # strip date prefixes
    fn_clean = re.sub(r'\s*-\s*', ' ', fn_clean)
    fn_words = fn_clean.split()
    # Add multi-word phrases from filename
    if len(fn_words) >= 2:
        candidates.append(fn_clean.strip())

    # From content: extract acronyms (must be ALL CAPS, 2-6 chars, not stop words)
    acronyms = re.findall(r'\b([A-Z][A-Z&]{1,5})\b', text)
    for acr in acronyms:
        if acr.lower() in STOP_WORDS_LOWER:
            continue
        # Must be genuinely all uppercase (not just a capitalized word)
        if not acr.replace('&', '').isupper():
            continue
        if len(acr) >= 2:
            candidates.append(acr)

    # From content: extract capitalized multi-word phrases (up to 3 words)
    multi_cap = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b', text)
    for phrase in multi_cap:
        words = phrase.split()
        if all(w.lower() not in STOP_WORDS_LOWER for w in words):
            candidates.append(phrase)

    return candidates


def normalize_topic(topic: str) -> str:
    """Normalize a topic string to a consistent tag-friendly format."""
    # Lowercase, replace spaces/special chars with hyphens
    normalized = topic.lower().strip()
    normalized = re.sub(r'[&]+', 'and', normalized)
    normalized = re.sub(r'[^a-z0-9]+', '-', normalized)
    normalized = normalized.strip('-')
    # Collapse multiple hyphens
    normalized = re.sub(r'-+', '-', normalized)
    return normalized


def normalize_person(name: str) -> str:
    """Normalize a person name to a tag-friendly format."""
    return name.lower().replace(' ', '-').strip('-')


def discover_entities(notes: list[NoteRecord]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Auto-discover topics and people from the vault content.

    Returns:
        discovered_topics: dict of tag_name -> [keyword variants]
        discovered_people: dict of tag_name -> [name variants]
    """
    # --- Collect candidate entities across all files ---
    all_names: Counter = Counter()
    all_topics: Counter = Counter()
    name_to_files: dict[str, set[str]] = defaultdict(set)
    topic_to_files: dict[str, set[str]] = defaultdict(set)

    # Track which raw names map to which normalized form
    name_variants: dict[str, set[str]] = defaultdict(set)
    topic_variants: dict[str, set[str]] = defaultdict(set)

    for note in notes:
        text = note.combined_text or note.text_content
        if not text and not note.folder_chain:
            continue

        # Extract names
        names = extract_candidate_names(text)
        # Also check filename for names
        fn_names = extract_candidate_names(note.filename.replace("-", " ").replace("_", " "))
        names.extend(fn_names)

        for name in names:
            norm = normalize_person(name)
            all_names[norm] += 1
            name_to_files[norm].add(note.filename)
            name_variants[norm].add(name)
            # Also add just the first name as a variant
            first = name.split()[0]
            name_variants[norm].add(first)

        # Also detect single first names that appear frequently (like "Lidia", "Albert")
        # But ONLY if they are known first names
        single_caps = SINGLE_CAPITALIZED.findall(text)
        for word in single_caps:
            word_lower = word.lower()
            if word_lower in STOP_WORDS_LOWER:
                continue
            if len(word) < 3:
                continue
            # Must be a known first name to count as a person
            if word_lower not in COMMON_FIRST_NAMES:
                continue
            norm = normalize_person(word)
            all_names[norm] += 1
            name_to_files[norm].add(note.filename)
            name_variants[norm].add(word)

        # Extract topics
        topics = extract_candidate_topics(text, note.folder_chain, note.filename)
        for topic in topics:
            norm = normalize_topic(topic)
            if not norm or len(norm) < 3:
                continue
            all_topics[norm] += 1
            topic_to_files[norm].add(note.filename)
            topic_variants[norm].add(topic)

    # --- Filter: people must appear in 2+ files ---
    discovered_people: dict[str, list[str]] = {}

    full_name_keys = {k for k in name_variants if '-' in k}
    single_name_keys = {k for k in name_variants if '-' not in k}

    # Group full names by first name
    first_name_groups: dict[str, list[str]] = defaultdict(list)
    for full_key in full_name_keys:
        first_part = full_key.split('-')[0]
        first_name_groups[first_part].append(full_key)

    # Merge logic: only merge full-name variants that are likely typos
    # (e.g. "berta-rodriguez" and "berta-rodriguex" — same first name, similar last name)
    # Do NOT merge genuinely different people (e.g. "sarah-blenderman" and "sarah-nelson")
    merged_names: dict[str, set[str]] = defaultdict(set)
    merged_files: dict[str, set[str]] = defaultdict(set)

    def _last_name_similar(a: str, b: str) -> bool:
        """Check if two last names are likely typo variants of each other."""
        a_parts = a.split('-')[1:]
        b_parts = b.split('-')[1:]
        a_last = '-'.join(a_parts)
        b_last = '-'.join(b_parts)
        if a_last == b_last:
            return True
        # Simple edit distance check: if names differ by at most 2 characters
        # and are at least 4 chars long, consider them typo variants
        if len(a_last) >= 4 and len(b_last) >= 4:
            if abs(len(a_last) - len(b_last)) <= 1:
                diffs = sum(1 for x, y in zip(a_last, b_last) if x != y)
                diffs += abs(len(a_last) - len(b_last))
                if diffs <= 2:
                    return True
        return False

    for first_name, full_keys in first_name_groups.items():
        # Cluster full names that are typo variants of each other
        clusters: list[list[str]] = []
        used = set()
        for fk in full_keys:
            if fk in used:
                continue
            cluster = [fk]
            used.add(fk)
            for other in full_keys:
                if other in used:
                    continue
                if _last_name_similar(fk, other):
                    cluster.append(other)
                    used.add(other)
            clusters.append(cluster)

        for cluster in clusters:
            # Pick the most common variant as canonical
            canonical = max(cluster, key=lambda k: len(name_to_files.get(k, set())))

            for fk in cluster:
                merged_names[canonical].update(name_variants[fk])
                merged_files[canonical].update(name_to_files.get(fk, set()))

            # Only merge single first-name mentions if there's exactly ONE cluster
            # for this first name (i.e., unambiguous which person "Sarah" refers to)
            if len(clusters) == 1 and first_name in single_name_keys:
                merged_names[canonical].update(name_variants[first_name])
                merged_files[canonical].update(name_to_files.get(first_name, set()))

    # Keep single names that don't have any full-name match
    for single_key in single_name_keys:
        has_full = single_key in first_name_groups
        if not has_full:
            merged_names[single_key] = name_variants[single_key]
            merged_files[single_key] = name_to_files.get(single_key, set())

    for norm_name, variants in merged_names.items():
        file_count = len(merged_files.get(norm_name, set()))
        if file_count >= 2:
            # Update name_to_files with merged data for later use
            name_to_files[norm_name] = merged_files[norm_name]
            discovered_people[norm_name] = sorted(variants, key=len, reverse=True)

    # --- Filter: topics must appear in 3+ files or be folder names ---
    discovered_topics: dict[str, list[str]] = {}

    # Folder-derived topics: only from depth >= 2 (subfolders, not top-level structural folders)
    folder_topics = set()
    structural_folders = {"projects", "areas", "resources", "tasks", "work"}
    for note in notes:
        for i, folder in enumerate(note.folder_chain):
            clean = re.sub(r'^\d+\.\s*', '', folder).strip()
            norm = normalize_topic(clean)
            if not norm or len(norm) < 3:
                continue
            # Skip top-level structural folders (PROJECTS, AREAS, etc.)
            if norm in structural_folders:
                continue
            folder_topics.add(norm)
            topic_variants.setdefault(norm, set()).add(clean)

    for norm_topic, variants in topic_variants.items():
        file_count = len(topic_to_files.get(norm_topic, set()))
        is_folder = norm_topic in folder_topics
        # Include if appears in 3+ files OR is a folder-derived topic
        if file_count >= 3 or is_folder:
            # Filter out topics that are actually people names
            if norm_topic in discovered_people:
                continue
            discovered_topics[norm_topic] = sorted(variants, key=len, reverse=True)

    # --- Deduplicate: merge topics that are substrings of each other ---
    # e.g., "ld" and "l-and-d" should merge
    topic_keys = sorted(discovered_topics.keys(), key=len)
    merged_out = set()
    for i, short in enumerate(topic_keys):
        for long_topic in topic_keys[i+1:]:
            if short in long_topic and len(short) < len(long_topic):
                # Merge short into long if they share significant file overlap
                short_files = topic_to_files.get(short, set())
                long_files = topic_to_files.get(long_topic, set())
                if short_files and long_files:
                    overlap = len(short_files & long_files) / max(len(short_files), 1)
                    if overlap > 0.5:
                        # Merge variants
                        discovered_topics[long_topic] = list(
                            set(discovered_topics.get(long_topic, []))
                            | set(discovered_topics.get(short, []))
                        )
                        topic_to_files[long_topic].update(short_files)
                        merged_out.add(short)

    for m in merged_out:
        discovered_topics.pop(m, None)

    logging.info(f"Discovered {len(discovered_topics)} topics, {len(discovered_people)} people")
    return discovered_topics, discovered_people


# ---------------------------------------------------------------------------
# Phase 1D: Note Type Classification
# ---------------------------------------------------------------------------

MONTH_PATTERN = re.compile(
    r'\b(january|february|march|april|may|june|july|august|september|'
    r'october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b',
    re.IGNORECASE
)
DATE_IN_FILENAME = re.compile(r'\d{6,8}')


def classify_note_type(note: NoteRecord) -> str:
    """Classify a note into a type category."""
    fn_lower = note.filename.lower()
    text = note.combined_text or note.text_content
    text_lower = text.lower()
    text_len = len(text.strip())

    # Stubs: minimal text content or image-only
    if note.is_image_only and not note.ocr_text:
        return "stub"
    if text_len < 50 and not note.ocr_text:
        return "stub"

    # One-on-ones
    if "1x1" in fn_lower or "1 x 1" in fn_lower or "one on one" in fn_lower:
        return "one-on-one"

    # Scripts (video/podcast/presentation)
    script_signals = [
        "narrator:" in text_lower,
        "## opening sequence" in text_lower,
        "production notes" in text_lower,
        "[visuals:" in text_lower,
        "video script" in fn_lower,
        "podcast" in fn_lower,
    ]
    if sum(script_signals) >= 2:
        return "script"

    # Analysis (AI prompt/response patterns)
    analysis_signals = [
        "**prompt:**" in text_lower or "**prompt**" in text_lower,
        "**response:**" in text_lower or "**response**" in text_lower,
        "## prompt" in text_lower,
        "ai summary" in fn_lower,
        "response on" in fn_lower,
    ]
    if sum(analysis_signals) >= 1:
        return "analysis"

    # Reference (templates, org charts, plugin lists, tools)
    reference_signals = [
        "template" in fn_lower,
        "plug-in" in fn_lower or "plugin" in fn_lower,
        "___" in text,  # horizontal rule separators in templates
        text.count("###") >= 5 and text_len < 500,  # org chart style
    ]
    if sum(reference_signals) >= 1:
        return "reference"

    # Meeting notes detection
    meeting_signals = [
        "meeting" in fn_lower,
        "roundtable" in fn_lower,
        "debrief" in fn_lower,
        "discussion" in fn_lower,
        "coaching" in fn_lower,
        "offsite" in fn_lower,
        "update" in fn_lower,
        "dlt" in fn_lower,
        bool(MONTH_PATTERN.search(fn_lower)),
        bool(DATE_IN_FILENAME.search(note.filename)),
        "> [!" in note.raw_content,
        text.count("- ") > 5,
    ]
    if sum(meeting_signals) >= 2:
        return "meeting"

    # Strategic documents (multiple headers, longer content)
    header_count = text.count("## ")
    if header_count >= 3 and text_len > 500:
        return "strategic"

    # Default: if has bullet points, treat as meeting; otherwise reference
    if text.count("- ") > 3:
        return "meeting"

    return "reference"


# ---------------------------------------------------------------------------
# Phase 1E: Similarity & Linking
# ---------------------------------------------------------------------------

def compute_pairwise_similarity(
    notes: dict[str, NoteRecord],
    topic_weight: float = 0.40,
    people_weight: float = 0.35,
    folder_weight: float = 0.25,
) -> dict[tuple[str, str], float]:
    """Compute similarity scores between all pairs of notes."""
    similarity = {}
    keys = list(notes.keys())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = notes[keys[i]]
            b = notes[keys[j]]

            # Skip stubs and excalidraw
            if a.note_type == "stub" or b.note_type == "stub":
                continue
            if a.is_excalidraw or b.is_excalidraw:
                continue

            # Topic overlap (Jaccard)
            if a.topics and b.topics:
                t_inter = len(a.topics & b.topics)
                t_union = len(a.topics | b.topics)
                topic_sim = t_inter / t_union if t_union > 0 else 0.0
            else:
                topic_sim = 0.0

            # People overlap (Jaccard)
            if a.people and b.people:
                p_inter = len(a.people & b.people)
                p_union = len(a.people | b.people)
                people_sim = p_inter / p_union if p_union > 0 else 0.0
            else:
                people_sim = 0.0

            # Folder proximity
            if len(a.folder_chain) >= 2 and len(b.folder_chain) >= 2:
                if a.folder_chain[:2] == b.folder_chain[:2]:
                    folder_sim = 1.0
                elif a.folder_chain[0] == b.folder_chain[0]:
                    folder_sim = 0.5
                else:
                    folder_sim = 0.0
            elif a.folder_chain and b.folder_chain:
                folder_sim = 0.5 if a.folder_chain[0] == b.folder_chain[0] else 0.0
            else:
                folder_sim = 0.0

            score = (topic_weight * topic_sim) + (people_weight * people_sim) + (folder_weight * folder_sim)

            if score > 0:
                pair = (keys[i], keys[j])
                similarity[pair] = score

    return similarity


def determine_links(
    notes: dict[str, NoteRecord],
    similarity: dict[tuple[str, str], float],
    min_similarity: float = 0.35,
    max_links: int = 5,
) -> None:
    """Assign related_notes to each note based on similarity scores."""
    # Build per-note link candidates
    candidates: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for (a, b), score in similarity.items():
        if score >= min_similarity:
            candidates[a].append((b, score))
            candidates[b].append((a, score))

    # Assign top-N links per note
    for key, cands in candidates.items():
        if key not in notes:
            continue
        note = notes[key]
        if note.note_type == "stub":
            continue
        # Sort by score descending, take top max_links
        sorted_cands = sorted(cands, key=lambda x: -x[1])[:max_links]
        note.related_notes = [c[0] for c in sorted_cands]


# ---------------------------------------------------------------------------
# Phase 1F: Tag Assignment
# ---------------------------------------------------------------------------

def slugify_folder(folder_name: str) -> str:
    """Convert a folder name to a tag-friendly slug."""
    # Strip numeric prefix
    clean = re.sub(r'^\d+\.\s*', '', folder_name).strip()
    slug = clean.lower()
    slug = re.sub(r'[&]+', '-and-', slug)
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')
    slug = re.sub(r'-+', '-', slug)
    return slug


def assign_tags(note: NoteRecord, graph: VaultGraph) -> list[str]:
    """Determine the full tag list for a note."""
    tags = []

    # 1. Source tag (from top-level folder)
    if note.folder_chain:
        top_folder = slugify_folder(note.folder_chain[0])
        if "project" in top_folder:
            tags.append("source/projects")
        elif "area" in top_folder:
            tags.append("source/areas")
        elif "resource" in top_folder:
            tags.append("source/resources")
        elif "task" in top_folder:
            tags.append("source/tasks")
        else:
            tags.append(f"source/{top_folder}")

    # 2. Project or Area tag (from subfolder)
    if len(note.folder_chain) >= 2:
        sub_slug = slugify_folder(note.folder_chain[1])
        if note.folder_chain[0] and "project" in slugify_folder(note.folder_chain[0]).lower():
            tags.append(f"project/{sub_slug}")
        elif note.folder_chain[0] and "area" in slugify_folder(note.folder_chain[0]).lower():
            tags.append(f"area/{sub_slug}")

    # 3. Type tag
    if note.note_type:
        tags.append(f"type/{note.note_type}")

    # 4. Topic tags (up to 3)
    topic_tags = sorted(note.topics)[:3]
    for t in topic_tags:
        tags.append(f"topic/{t}")

    # 5. Person tags (all detected)
    for p in sorted(note.people):
        tags.append(f"person/{p}")

    return sorted(set(tags))


# ---------------------------------------------------------------------------
# Phase 1G: Analysis Report
# ---------------------------------------------------------------------------

def generate_report(graph: VaultGraph, verbose: bool = False) -> None:
    """Print a summary of the analysis."""
    total = len(graph.notes)
    skipped_stub = sum(1 for n in graph.notes.values() if n.note_type == "stub")
    to_modify = total - skipped_stub

    print("\n" + "=" * 70)
    print("  OBSIDIAN VAULT ANALYSIS REPORT")
    print("=" * 70)

    print(f"\n  Vault: {graph.vault_path}")
    print(f"  Total files scanned: {total}")
    print(f"  Files to tag & link: {to_modify}")
    print(f"  Stubs/image-only (tags only): {skipped_stub}")

    # Note type distribution
    print(f"\n{'─' * 40}")
    print("  NOTE TYPES")
    print(f"{'─' * 40}")
    type_counts = Counter(n.note_type for n in graph.notes.values())
    for ntype, count in type_counts.most_common():
        print(f"    {ntype:20s} : {count}")

    # Discovered topics
    print(f"\n{'─' * 40}")
    print("  DISCOVERED TOPICS")
    print(f"{'─' * 40}")
    topic_file_counts = {
        topic: len(files)
        for topic, files in graph.topic_index.items()
    }
    for topic, count in sorted(topic_file_counts.items(), key=lambda x: -x[1]):
        variants = graph.discovered_topics.get(topic, [topic])
        print(f"    topic/{topic:25s} ({count} files)  variants: {', '.join(variants[:3])}")

    # Discovered people
    print(f"\n{'─' * 40}")
    print("  DISCOVERED PEOPLE")
    print(f"{'─' * 40}")
    people_file_counts = {
        person: len(files)
        for person, files in graph.people_index.items()
    }
    for person, count in sorted(people_file_counts.items(), key=lambda x: -x[1]):
        variants = graph.discovered_people.get(person, [person])
        print(f"    person/{person:25s} ({count} files)  variants: {', '.join(variants[:3])}")

    # Linking summary
    linked_notes = [n for n in graph.notes.values() if n.related_notes]
    total_links = sum(len(n.related_notes) for n in linked_notes)
    print(f"\n{'─' * 40}")
    print("  INTERLINKING SUMMARY")
    print(f"{'─' * 40}")
    print(f"    Files with related notes: {len(linked_notes)}")
    print(f"    Total links created: {total_links}")

    # Auto-title suggestions
    title_suggestions = suggest_titles(graph)
    if title_suggestions:
        print(f"\n{'─' * 40}")
        print("  AUTO-TITLE SUGGESTIONS (for Untitled docs)")
        print(f"{'─' * 40}")
        for rel_path, title in title_suggestions.items():
            print(f"    {rel_path}")
            print(f"      -> Suggested: \"{title}\"")
            print(f"      (will add H1 header + rename file on --apply)")

    # Per-file detail (verbose)
    if verbose:
        print(f"\n{'─' * 40}")
        print("  PER-FILE DETAIL")
        print(f"{'─' * 40}")
        for key in sorted(graph.notes.keys()):
            note = graph.notes[key]
            print(f"\n  FILE: {note.relative_path}")
            print(f"    Type: {note.note_type}")
            if note.tags:
                print(f"    Tags: {', '.join(note.tags)}")
            if note.topics:
                print(f"    Topics: {', '.join(sorted(note.topics))}")
            if note.people:
                print(f"    People: {', '.join(sorted(note.people))}")
            if note.related_notes:
                # Show filenames for readability
                related_names = []
                for rp in note.related_notes:
                    if rp in graph.notes:
                        related_names.append(graph.notes[rp].filename)
                    else:
                        related_names.append(rp)
                print(f"    Related: {', '.join(related_names)}")

    print(f"\n{'=' * 70}")
    print()


# ---------------------------------------------------------------------------
# Auto-Title Generation for Untitled Documents
# ---------------------------------------------------------------------------

def generate_title(note: NoteRecord, graph: VaultGraph) -> Optional[str]:
    """Generate a descriptive title for untitled documents.

    Uses content, topics, people, folder location, and note type to create
    a meaningful title. Returns None if a good title can't be determined.
    """
    if note.filename.lower() != "untitled":
        return None

    parts = []

    # Use the most prominent topic
    if note.topics:
        top_topic = sorted(note.topics)[0]
        # Map topic keys back to readable names
        variants = graph.discovered_topics.get(top_topic, [top_topic])
        readable = variants[0] if variants else top_topic.replace('-', ' ').title()
        parts.append(readable)

    # Use the most prominent person
    if note.people:
        top_person = sorted(note.people)[0]
        variants = graph.discovered_people.get(top_person, [top_person])
        readable = variants[0] if variants else top_person.replace('-', ' ').title()
        parts.append(readable)

    # Use the subfolder as context
    if len(note.folder_chain) >= 2:
        subfolder = re.sub(r'^\d+\.\s*', '', note.folder_chain[1]).strip()
        if subfolder and subfolder not in ' '.join(parts):
            parts.append(subfolder)

    # Add note type suffix
    type_suffix = {
        "meeting": "Notes",
        "one-on-one": "1x1 Notes",
        "strategic": "Strategy",
        "reference": "Reference",
        "analysis": "Analysis",
        "stub": "Notes",
    }
    suffix = type_suffix.get(note.note_type, "Notes")

    if parts:
        title = f"{' - '.join(parts)} {suffix}"
    else:
        # Fallback: use folder name + Notes
        if note.folder_chain:
            folder = re.sub(r'^\d+\.\s*', '', note.folder_chain[-1]).strip()
            title = f"{folder} {suffix}"
        else:
            return None

    # If the content has a clear first line that could serve as a title, consider it
    text = note.text_content.strip()
    first_lines = [l.strip() for l in text.split('\n') if l.strip() and not l.strip().startswith('!')]
    if first_lines:
        first_line = first_lines[0].lstrip('#').lstrip('- ').lstrip('•').strip()
        # Only use first line if it's a genuine title-like line:
        # - Between 10 and 60 chars
        # - Doesn't end with a dash or comma (fragment indicator)
        # - Doesn't start with blockquote
        # - Has at least 2 words
        is_title_like = (
            10 <= len(first_line) <= 60
            and not first_line.startswith('>')
            and not first_line.endswith('-')
            and not first_line.endswith(',')
            and len(first_line.split()) >= 2
        )
        if is_title_like:
            title = first_line

    return title


def suggest_titles(graph: VaultGraph) -> dict[str, str]:
    """Generate title suggestions for all untitled documents.

    Returns dict of relative_path -> suggested title.
    """
    suggestions = {}
    for key, note in graph.notes.items():
        if note.filename.lower() == "untitled":
            title = generate_title(note, graph)
            if title:
                suggestions[key] = title
    return suggestions


# ---------------------------------------------------------------------------
# Phase 2A: Backup
# ---------------------------------------------------------------------------

def backup_vault(vault_path: Path, backup_dir: Optional[Path] = None) -> Path:
    """Create a timestamped full backup of the vault."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if backup_dir is None:
        backup_dir = vault_path.parent

    backup_path = backup_dir / f"vault_backup_{timestamp}"
    logging.info(f"Creating backup at: {backup_path}")
    shutil.copytree(vault_path, backup_path)
    logging.info("Backup complete.")
    return backup_path


# ---------------------------------------------------------------------------
# Phase 2B: Apply Changes
# ---------------------------------------------------------------------------

def build_frontmatter(tags: list[str]) -> str:
    """Build a YAML frontmatter block with tags."""
    lines = ["---", "tags:"]
    for tag in tags:
        lines.append(f"  - {tag}")
    lines.append("---")
    return "\n".join(lines)


def build_related_notes_section(related: list[str], notes: dict[str, 'NoteRecord'],
                                duplicated_filenames: set[str]) -> str:
    """Build a Related Notes section with wikilinks.

    Uses shortest unambiguous path for Obsidian wikilinks.
    For unique filenames: [[filename]]
    For duplicated filenames: [[folder/filename]] for disambiguation.
    """
    lines = ["", "", "---", "", "## Related Notes", ""]
    for rel_path in related:
        if rel_path in notes:
            note = notes[rel_path]
            if note.filename in duplicated_filenames:
                # Use parent folder + filename for disambiguation
                if note.folder_chain:
                    link_name = f"{note.folder_chain[-1]}/{note.filename}"
                else:
                    link_name = note.filename
            else:
                link_name = note.filename
        else:
            # Fallback: use the relative path without .md
            link_name = rel_path.replace(".md", "")
        lines.append(f"- [[{link_name}]]")
    return "\n".join(lines) + "\n"


def inject_frontmatter(raw_content: str, tags: list[str], has_existing_frontmatter: bool) -> str:
    """Add YAML frontmatter tags to file content."""
    fm = build_frontmatter(tags)

    if has_existing_frontmatter:
        # Find the closing --- and insert tags into existing frontmatter
        parts = raw_content.split("---", 2)
        if len(parts) >= 3:
            existing_yaml = parts[1]
            rest = parts[2]
            if "tags:" not in existing_yaml:
                tag_yaml = "\ntags:\n"
                for tag in tags:
                    tag_yaml += f"  - {tag}\n"
                existing_yaml = existing_yaml.rstrip("\n") + tag_yaml
            return "---" + existing_yaml + "---" + rest
        # Fallback: prepend
        return fm + "\n" + raw_content
    else:
        # No existing frontmatter: prepend
        content = raw_content.lstrip("\n")
        return fm + "\n" + content


def inject_related_notes(raw_content: str, related_notes: list[str],
                         notes: dict[str, 'NoteRecord'],
                         duplicated_filenames: set[str]) -> str:
    """Append Related Notes section at the bottom of the file."""
    if not related_notes:
        return raw_content

    # Check if section already exists (idempotent)
    if "## Related Notes" in raw_content:
        return raw_content

    section = build_related_notes_section(related_notes, notes, duplicated_filenames)
    return raw_content.rstrip("\n") + section


def verify_content_integrity(original: str, modified: str, note: NoteRecord) -> bool:
    """Verify that critical content elements are preserved after modification."""
    # Check all image embeds preserved
    for embed in note.image_embeds:
        img_name = embed[0] if isinstance(embed, tuple) else embed
        if img_name not in modified:
            logging.error(f"Image embed lost in {note.relative_path}: {img_name}")
            return False

    # Check all callout blocks preserved
    original_callouts = CALLOUT_CONTENT_PATTERN.findall(original)
    for callout_type, callout_text in original_callouts:
        if callout_type not in modified:
            logging.error(f"Callout lost in {note.relative_path}: [!{callout_type}]")
            return False

    return True


def apply_changes(
    graph: VaultGraph,
    apply: bool = False,
    backup_dir: Optional[Path] = None,
    skip_backup: bool = False,
) -> None:
    """Apply tags and links to all files."""
    if not apply:
        print("\n  DRY RUN — no files modified. Use --apply to write changes.\n")
        return

    # Backup first
    if not skip_backup:
        bp = backup_vault(graph.vault_path, backup_dir)
        print(f"  Backup created: {bp}")

    # Generate auto-title suggestions for untitled docs
    title_suggestions = suggest_titles(graph)

    changelog = []
    modified_count = 0
    renamed_count = 0
    error_count = 0

    for key, note in sorted(graph.notes.items()):
        if not note.tags and not note.related_notes and key not in title_suggestions:
            continue

        original = note.raw_content
        content = original

        # Inject frontmatter
        if note.tags:
            content = inject_frontmatter(content, note.tags, note.has_frontmatter)

        # Inject related notes (only for non-stubs)
        if note.related_notes and note.note_type != "stub":
            duplicated = getattr(graph, '_duplicated_filenames', set())
            content = inject_related_notes(content, note.related_notes, graph.notes, duplicated)

        # Add H1 title for untitled documents
        suggested_title = title_suggestions.get(key)
        if suggested_title:
            # Add H1 header after frontmatter
            if content.startswith("---"):
                # Find end of frontmatter
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    content = "---" + parts[1] + "---\n# " + suggested_title + "\n" + parts[2]
            else:
                content = "# " + suggested_title + "\n\n" + content

        # Skip if nothing changed
        if content == original:
            continue

        # Verify integrity
        if not verify_content_integrity(original, content, note):
            logging.error(f"Integrity check failed for {note.relative_path} — skipping")
            error_count += 1
            continue

        # Write
        try:
            note.filepath.write_text(content, encoding="utf-8")
            modified_count += 1

            change_record = {
                "file": note.relative_path,
                "tags_added": note.tags,
                "related_notes_added": note.related_notes,
                "timestamp": datetime.now().isoformat(),
            }

            # Rename untitled files
            if suggested_title and note.filename.lower() == "untitled":
                # Sanitize title for filename
                safe_name = re.sub(r'[<>:"/\\|?*]', '', suggested_title)
                safe_name = safe_name.strip('. ')
                if safe_name:
                    new_path = note.filepath.parent / f"{safe_name}.md"
                    if not new_path.exists():
                        note.filepath.rename(new_path)
                        change_record["renamed_to"] = str(new_path.relative_to(graph.vault_path))
                        renamed_count += 1
                        logging.info(f"Renamed: {note.relative_path} -> {safe_name}.md")
                    else:
                        logging.warning(f"Cannot rename {note.relative_path}: {safe_name}.md already exists")

            changelog.append(change_record)
            logging.debug(f"Modified: {note.relative_path}")
        except OSError as e:
            logging.error(f"Failed to write {note.relative_path}: {e}")
            # Attempt restore
            try:
                note.filepath.write_text(original, encoding="utf-8")
            except OSError:
                pass
            error_count += 1

    # Write changelog
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    changelog_path = graph.vault_path.parent / f"linker_changelog_{timestamp}.json"
    with open(changelog_path, "w", encoding="utf-8") as f:
        json.dump(changelog, f, indent=2)

    print(f"\n  Changes applied:")
    print(f"    Files modified: {modified_count}")
    print(f"    Files renamed: {renamed_count}")
    print(f"    Errors: {error_count}")
    print(f"    Changelog: {changelog_path}")
    print()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def build_graph(vault_path: Path, skip_ocr: bool = False, min_similarity: float = 0.35) -> VaultGraph:
    """Run the full Phase 1 analysis pipeline."""
    graph = VaultGraph(vault_path=vault_path)

    # 1A: Scan & Parse
    logging.info("Phase 1A: Scanning vault...")
    notes_list = scan_vault(vault_path)
    logging.info(f"Found {len(notes_list)} markdown files")

    # 1B: OCR
    logging.info("Phase 1B: Running OCR on images...")
    perform_ocr(vault_path, notes_list, skip_ocr)

    # Populate combined_text for all notes
    for note in notes_list:
        if not note.combined_text:
            note.combined_text = (note.text_content + "\n" + note.ocr_text).strip()

    # 1C: Entity Discovery
    logging.info("Phase 1C: Discovering entities...")
    discovered_topics, discovered_people = discover_entities(notes_list)
    graph.discovered_topics = discovered_topics
    graph.discovered_people = discovered_people

    # 1D: Classify note types
    logging.info("Phase 1D: Classifying note types...")
    for note in notes_list:
        note.note_type = classify_note_type(note)

    # Assign topics and people to each note
    logging.info("Assigning topics and people to notes...")
    for note in notes_list:
        text = (note.combined_text + " " + note.filename).lower()

        # Match topics
        for topic_key, variants in discovered_topics.items():
            for variant in variants:
                if variant.lower() in text:
                    note.topics.add(topic_key)
                    break

        # Limit to top 3 topics by specificity (fewer files = more specific)
        if len(note.topics) > 3:
            # Prefer topics that appear in fewer files (more specific)
            topic_specificity = {}
            for t in note.topics:
                # Count how many notes have this topic
                count = sum(1 for n in notes_list if t in n.topics) if hasattr(note, '_counted') else 0
                topic_specificity[t] = count
            # If we haven't counted yet, just keep alphabetical top 3
            note.topics = set(sorted(note.topics)[:3])

        # Match people — two passes:
        # Pass 1: Match full names (unambiguous)
        # Pass 2: Match single first names with context-based disambiguation
        matched_full_names = set()
        for person_key, variants in discovered_people.items():
            for variant in variants:
                variant_lower = variant.lower()
                # Only match multi-word (full name) variants in this pass
                if ' ' not in variant_lower:
                    continue
                if variant_lower in text:
                    note.people.add(person_key)
                    matched_full_names.add(person_key)
                    break

        # Pass 2: Single first names — use context-based disambiguation
        # Group people by first name to find ambiguous cases
        first_name_to_people: dict[str, list[str]] = defaultdict(list)
        for person_key in discovered_people:
            first = person_key.split('-')[0]
            first_name_to_people[first].append(person_key)

        for first_name, candidate_keys in first_name_to_people.items():
            # Skip if we already matched a full name for any candidate
            already_matched = [ck for ck in candidate_keys if ck in matched_full_names]
            if already_matched:
                continue

            # Check if the single first name appears in the text
            if len(first_name) <= 4:
                if not re.search(r'\b' + re.escape(first_name) + r'\b', text):
                    continue
            else:
                if first_name not in text:
                    continue

            if len(candidate_keys) == 1:
                # Unambiguous: only one person with this first name
                note.people.add(candidate_keys[0])
            else:
                # Ambiguous: multiple people share this first name
                # Use context clues: which candidate co-occurs most with
                # the other people, topics, and folder of this note
                best_key = None
                best_score = -1
                for ck in candidate_keys:
                    score = 0
                    ck_files = name_to_files.get(ck, set())
                    # Score by how many of note's other matched people also
                    # appear in files where this candidate appears
                    for other_person in note.people:
                        other_files = name_to_files.get(other_person, set())
                        if ck_files & other_files:
                            score += 2
                    # Score by folder proximity
                    for ck_file in ck_files:
                        # Find the note for this file
                        for n in notes_list:
                            if n.filename in ck_file or n.relative_path == ck_file:
                                if n.folder_chain and note.folder_chain:
                                    if len(n.folder_chain) >= 2 and len(note.folder_chain) >= 2:
                                        if n.folder_chain[:2] == note.folder_chain[:2]:
                                            score += 1
                                break
                    if score > best_score:
                        best_score = score
                        best_key = ck
                if best_key and best_score > 0:
                    note.people.add(best_key)
                    logging.debug(
                        f"Disambiguated '{first_name}' -> '{best_key}' "
                        f"in {note.filename} (score={best_score})"
                    )

    # Detect duplicate filenames for Obsidian path disambiguation
    filename_counts = Counter(n.filename for n in notes_list)
    duplicated_filenames = {k for k, v in filename_counts.items() if v > 1}

    # Index notes into graph using relative_path as the unique key
    for note in notes_list:
        # Use relative_path as unique key to avoid collisions
        key = note.relative_path
        graph.notes[key] = note
        for t in note.topics:
            graph.topic_index[t].add(key)
        for p in note.people:
            graph.people_index[p].add(key)
        if note.folder_chain:
            folder_key = "/".join(note.folder_chain)
            graph.folder_index[folder_key].add(key)
        graph.type_index[note.note_type].add(key)

    # Store which filenames are duplicated for wikilink disambiguation
    graph._duplicated_filenames = duplicated_filenames

    # 1E: Similarity & Linking
    logging.info("Phase 1E: Computing similarity and links...")
    similarity = compute_pairwise_similarity(graph.notes)
    determine_links(graph.notes, similarity, min_similarity=min_similarity)

    # 1F: Tag Assignment
    logging.info("Phase 1F: Assigning tags...")
    for key, note in graph.notes.items():
        note.tags = assign_tags(note, graph)

    return graph


def main():
    parser = argparse.ArgumentParser(
        description="Obsidian Vault Auto-Tagger & Interlinker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (analyze only)
  python obsidian_linker.py --vault-path "/path/to/vault"

  # Verbose dry run
  python obsidian_linker.py --vault-path "/path/to/vault" --verbose

  # Apply changes
  python obsidian_linker.py --vault-path "/path/to/vault" --apply

  # Skip OCR (faster)
  python obsidian_linker.py --vault-path "/path/to/vault" --apply --skip-ocr
        """,
    )
    parser.add_argument("--vault-path", required=True, type=Path, help="Path to the Obsidian vault root")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")
    parser.add_argument("--backup-dir", type=Path, default=None, help="Directory for backups (default: vault parent)")
    parser.add_argument("--skip-backup", action="store_true", help="Skip backup creation")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR on images")
    parser.add_argument("--min-similarity", type=float, default=0.35, help="Minimum similarity for interlinking (0-1)")
    parser.add_argument("--verbose", action="store_true", help="Show per-file detail in report")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    # Validate vault path
    if not args.vault_path.is_dir():
        print(f"Error: Vault path does not exist: {args.vault_path}")
        sys.exit(1)

    # Run analysis
    graph = build_graph(
        vault_path=args.vault_path,
        skip_ocr=args.skip_ocr,
        min_similarity=args.min_similarity,
    )

    # Generate report
    generate_report(graph, verbose=args.verbose)

    # Apply changes
    apply_changes(
        graph,
        apply=args.apply,
        backup_dir=args.backup_dir,
        skip_backup=args.skip_backup,
    )


if __name__ == "__main__":
    main()
