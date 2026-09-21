#!/usr/bin/env python3
"""
Clean up rolling-caption duplication already baked into the transcript
.md files under childress_transcripts/.

Background: YouTube auto-generated captions render as a rolling window —
each caption cue repeats the tail of the previous cue's words before
adding new ones. The original scraper (childress_transcripts.py) only
skipped a cue that was an *exact* duplicate of the one before it, which
almost never happens; the much more common partial overlap slipped
through, so a large fraction of words in the already-saved transcripts
are duplicated (e.g. "...coffee with Dr. Sunday morning coffee with Dr.
Childress. So today I'm going to be Childress. So today I'm going to be
talking about..."). childress_transcripts.py has been fixed so future
scrapes don't have this problem, but it doesn't touch files already on
disk — this script does that part.

It re-uses the same word-overlap-collapse logic, applied to the already
-flattened text: since the duplication is always a locally adjacent
repeat (segment N's tail == segment N+1's head), scanning token-by-token
and dropping a run that repeats the last ~20 output words finds and
removes it even though the original per-cue boundaries are gone.

Usage:
    python dedupe_transcripts.py --dry-run          # report only, no writes
    python dedupe_transcripts.py --dry-run --file "path/to/one.md"   # inspect one file
    python dedupe_transcripts.py                    # clean every file in place
"""

import argparse
import glob
import os
import re

TRANSCRIPTS_DIR = "childress_transcripts"
MAX_OVERLAP_WORDS = 20   # how far back to look for a repeated run
MIN_COLLAPSE_WORDS = 4   # don't collapse a match shorter than this (avoid false positives on short common phrases)

# Timestamp markers like "[05:23](https://...)" must stay intact as a
# single atomic token — splitting on whitespace would otherwise break
# them apart, and the URL text itself could spuriously "match" body text.
_TOKEN_RE = re.compile(r"\[\d{2}:\d{2}\]\([^)]*\)|\S+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def dedupe_tokens(tokens: list[str]) -> tuple[list[str], int]:
    """Collapse locally-repeated runs of tokens. Returns (clean_tokens, num_tokens_removed)."""
    out: list[str] = []
    i = 0
    removed = 0
    n = len(tokens)
    while i < n:
        tail = out[-MAX_OVERLAP_WORDS:]
        matched = False
        # Try the longest possible overlap first so we collapse the whole
        # duplicated run in one go rather than one word at a time.
        max_check = min(len(tail), n - i, MAX_OVERLAP_WORDS)
        for overlap_len in range(max_check, MIN_COLLAPSE_WORDS - 1, -1):
            if tail[-overlap_len:] == tokens[i:i + overlap_len]:
                i += overlap_len
                removed += overlap_len
                matched = True
                break
        if not matched:
            out.append(tokens[i])
            i += 1
    return out, removed


def clean_body(body: str) -> tuple[str, int, int]:
    """Dedupe a transcript body, preserving paragraph breaks around timestamp markers."""
    # Timestamp markers were inserted as "\n\n[MM:SS](url)" — split on those
    # so we dedupe within each timestamped segment, and never let a repeat
    # spuriously span across them (also fine either way, but this is safer
    # and matches how the text was originally assembled segment-by-segment).
    parts = re.split(r"(\n\n\[\d{2}:\d{2}\]\([^)]*\))", body)

    total_before = 0
    total_removed = 0
    cleaned_parts = []
    for part in parts:
        if part.startswith("\n\n["):
            cleaned_parts.append(part)
            continue
        tokens = tokenize(part)
        total_before += len(tokens)
        clean_tokens, removed = dedupe_tokens(tokens)
        total_removed += removed
        cleaned_parts.append(" ".join(clean_tokens))

    cleaned = "".join(cleaned_parts)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" \n", "\n", cleaned)
    return cleaned.strip(), total_before, total_removed


def process_file(path: str, dry_run: bool) -> tuple[int, int]:
    text = open(path, encoding="utf-8", errors="replace").read()
    if "---\n" not in text:
        return 0, 0
    header, body = text.split("---\n", 1)
    body = body.strip()

    cleaned_body, before, removed = clean_body(body)
    if removed == 0:
        return before, 0

    pct = removed / before * 100 if before else 0
    print(f"  {os.path.relpath(path)}: {removed}/{before} words removed ({pct:.0f}%)")

    if not dry_run:
        new_text = header + "---\n\n" + cleaned_body + "\n"
        open(path, "w", encoding="utf-8").write(new_text)

    return before, removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report what would change without writing files")
    ap.add_argument("--file", help="process a single file instead of the whole corpus")
    args = ap.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = [
            f for f in glob.glob(f"{TRANSCRIPTS_DIR}/**/*.md", recursive=True)
            if not os.path.basename(f).startswith("_")
        ]

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing {len(files)} file(s)...\n")

    total_before = 0
    total_removed = 0
    files_changed = 0
    for path in sorted(files):
        before, removed = process_file(path, args.dry_run)
        total_before += before
        total_removed += removed
        if removed:
            files_changed += 1

    pct = total_removed / total_before * 100 if total_before else 0
    print(f"\n{'Would change' if args.dry_run else 'Changed'} {files_changed}/{len(files)} files.")
    print(f"Total: {total_removed:,}/{total_before:,} words removed ({pct:.1f}%).")
    if args.dry_run:
        print("\nRun without --dry-run to write the cleaned files. Don't forget to")
        print("re-run build_embeddings.py and commit data/index/ afterwards, since")
        print("the underlying text changed.")


if __name__ == "__main__":
    main()
