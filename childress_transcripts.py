#!/usr/bin/env python3
"""
Download auto-generated YouTube transcripts for all Dr. Childress videos,
organized by playlist, with a master file combining everything.

Requirements:
    pip install yt-dlp
"""

import os
import re
import json
import subprocess
import sys
from pathlib import Path

CHANNEL_URL = "https://www.youtube.com/@dr.c.a.childress673"
OUTPUT_DIR = Path("childress_transcripts")
MASTER_FILE = OUTPUT_DIR / "_ALL_TRANSCRIPTS.md"

# Use 'python -m yt_dlp' to avoid Windows PATH issues
YT_DLP = [sys.executable, "-m", "yt_dlp"]


def sanitize_filename(name: str) -> str:
    """Remove/replace characters that are invalid in folder/file names."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.replace("#", "-")
    name = re.sub(r"-{2,}", "-", name)
    name = name.strip().strip(".-")
    return name[:100]


def find_existing_transcript(folder: Path, video_title: str) -> Path | None:
    """
    Look for a transcript already saved for this video, so re-running the
    script doesn't re-download (via yt-dlp) videos it already has — that's
    two subprocess calls out to YouTube per video (get_video_upload_date and
    download_transcript), and they dominate the script's runtime once most
    of the channel has already been scraped.

    Filenames are "{date}_{safe_title}.md", but the date isn't known yet at
    this point without one of those same expensive calls, so this matches
    on the title suffix alone (glob "*_{safe_title}.md"), which is already
    unique enough in practice given titles are truncated the same way.
    """
    safe_title = sanitize_filename(video_title)[:60]
    matches = list(folder.glob(f"*_{safe_title}.md"))
    return matches[0] if matches else None


def master_entry_from_saved(filepath: Path, playlist_title: str) -> tuple[str, bool]:
    """Rebuild a master-file entry from an already-saved transcript file."""
    text = filepath.read_text(encoding="utf-8", errors="replace")

    title, url, date = "Unknown", "", ""
    for line in text.splitlines()[:10]:
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("**Link:**"):
            url = line.replace("**Link:**", "").strip()
        elif line.startswith("**Date:**"):
            date = line.replace("**Date:**", "").strip()

    body = text.split("---\n", 1)[-1].strip()
    has_transcript = body != "*No automatic transcript available for this video.*"

    status = "" if has_transcript else " *(no transcript)*"
    entry = (
        f"## {title}{status}\n\n"
        f"**Link:** {url}  \n"
        f"**Playlist:** {playlist_title}  \n"
        f"**Date:** {date}\n\n"
    )
    if has_transcript:
        entry += f"{body}\n\n"
    entry += "---\n\n"
    return entry, has_transcript


def get_channel_playlists(channel_url: str) -> list[dict]:
    """Fetch all playlists from the channel."""
    print("Fetching playlists from channel...")
    result = subprocess.run(
        YT_DLP + [
            "--flat-playlist",
            "--dump-json",
            f"{channel_url}/playlists",
        ],
        capture_output=True,
        text=True,
    )
    playlists = []
    for line in result.stdout.strip().splitlines():
        try:
            data = json.loads(line)
            if data.get("url"):
                playlists.append(
                    {
                        "title": data.get("title", "Unknown playlist"),
                        "url": f"https://www.youtube.com/playlist?list={data['id']}"
                        if not data["url"].startswith("http")
                        else data["url"],
                        "id": data.get("id"),
                    }
                )
        except json.JSONDecodeError:
            pass
    print(f"  Found {len(playlists)} playlists.")
    return playlists


def get_playlist_videos(playlist_url: str) -> list[dict]:
    """Fetch all videos in a playlist."""
    result = subprocess.run(
        YT_DLP + ["--flat-playlist", "--dump-json", playlist_url],
        capture_output=True,
        text=True,
    )
    videos = []
    for line in result.stdout.strip().splitlines():
        try:
            data = json.loads(line)
            vid_id = data.get("id") or data.get("url", "").split("v=")[-1]
            videos.append(
                {
                    "title": data.get("title", "Unknown title"),
                    "url": f"https://www.youtube.com/watch?v={vid_id}",
                    "id": vid_id,
                    "upload_date": data.get("upload_date", ""),
                }
            )
        except json.JSONDecodeError:
            pass
    return videos


def get_video_upload_date(video_url: str) -> str:
    """Fetch the actual upload date from YouTube video metadata."""
    result = subprocess.run(
        YT_DLP + ["--skip-download", "--print", "%(upload_date)s", video_url],
        capture_output=True,
        text=True,
    )
    date = result.stdout.strip()
    # Returns YYYYMMDD or empty
    return date if re.match(r"^\d{8}$", date) else ""


def format_date(yyyymmdd: str) -> str:
    """Format YYYYMMDD as YYYY-MM-DD, or return 'Ukendt'."""
    if re.match(r"^\d{8}$", yyyymmdd):
        return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
    return "Unknown"


def download_transcript(video_url: str, output_path: Path) -> str | None:
    """
    Download the auto-generated English transcript for a video.
    Returns the raw transcript text with timestamps, or None if unavailable.
    """
    tmp_dir = output_path.parent / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_base = tmp_dir / "transcript"

    subprocess.run(
        YT_DLP + [
            "--skip-download",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--sub-format", "vtt",
            "-o", str(tmp_base),
            video_url,
        ],
        capture_output=True,
        text=True,
    )

    vtt_files = list(tmp_dir.glob("*.vtt"))
    if not vtt_files:
        return None

    vtt_text = vtt_files[0].read_text(encoding="utf-8", errors="replace")

    for f in vtt_files:
        f.unlink()

    return vtt_to_clean_text(vtt_text, video_url=video_url)


def vtt_time_to_seconds(t: str) -> int:
    """Convert VTT timestamp like 00:05:23.000 to seconds."""
    t = t.strip().split(".")[0]  # remove milliseconds
    parts = t.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        pass
    return 0


def vtt_to_clean_text(vtt: str, video_url: str = "") -> str:
    """
    Convert VTT subtitle format to readable text with timestamp markers.
    Adds a [MM:SS](url?t=N) link every ~60 seconds so answers can cite
    exact positions in the video.
    """
    STAMP_INTERVAL = 60  # insert a timestamp marker every N seconds

    lines = vtt.splitlines()
    segments = []   # list of (seconds, text)
    current_seconds = 0

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Timestamp line e.g. "00:00:01.000 --> 00:00:04.000"
        ts_match = re.match(r"^(\d{1,2}:\d{2}[:\d.]*)\s*-->", line)
        if ts_match:
            current_seconds = vtt_time_to_seconds(ts_match.group(1))
            i += 1
            # Collect text lines following this timestamp
            text_parts = []
            while i < len(lines):
                tline = lines[i].strip()
                if not tline or re.match(r"^\d{1,2}:\d{2}", tline) or re.match(r"^\d+$", tline):
                    break
                if not any(tline.startswith(x) for x in ("WEBVTT", "Kind:", "Language:", "align:", "position:")):
                    clean = re.sub(r"<[^>]+>", "", tline).strip()
                    if clean:
                        text_parts.append(clean)
                i += 1
            if text_parts:
                segments.append((current_seconds, " ".join(text_parts)))
            continue
        i += 1

    if not segments:
        return ""

    # Build output: merge consecutive lines, insert timestamp marker every 60s
    result = []
    last_stamp_seconds = -STAMP_INTERVAL
    last_text = ""

    for seconds, text in segments:
        # Skip duplicate lines (YouTube repeats lines in VTT)
        if text == last_text:
            continue
        last_text = text

        # Insert timestamp marker if enough time has passed
        if seconds - last_stamp_seconds >= STAMP_INTERVAL:
            mins = seconds // 60
            secs = seconds % 60
            if video_url:
                # Clickable link: [05:23](https://youtu.be/...?t=323)
                vid_id = re.search(r"[?&]v=([^&]+)", video_url)
                if vid_id:
                    stamp = f"\n\n[{mins:02d}:{secs:02d}](https://www.youtube.com/watch?v={vid_id.group(1)}&t={seconds})"
                else:
                    stamp = f"\n\n[{mins:02d}:{secs:02d}]({video_url}&t={seconds})"
            else:
                stamp = f"\n\n[{mins:02d}:{secs:02d}]"
            result.append(stamp)
            last_stamp_seconds = seconds

        result.append(text)

    text = " ".join(result)
    # Collapse repeated phrases
    text = re.sub(r"(\b.{10,80}\b) \1", r"\1", text)
    return text.strip()


def save_transcript(
    video: dict,
    transcript: str | None,
    folder: Path,
    playlist_title: str,
) -> str:
    """Save transcript to a .md file and return formatted entry for master file."""
    safe_title = sanitize_filename(video["title"])[:60]
    date_raw = video["upload_date"]           # YYYYMMDD or ""
    date_prefix = f"{date_raw[:4]}.{date_raw[4:6]}.{date_raw[6:8]}" if date_raw else "0000.00.00"
    date_display = format_date(date_raw)
    filename = f"{date_prefix}_{safe_title}.md"
    filepath = folder / filename

    content_lines = [
        f"# {video['title']}",
        f"",
        f"**Link:** {video['url']}",
        f"**Playlist:** {playlist_title}",
        f"**Date:** {date_display}",
        f"",
        f"---",
        f"",
    ]

    if transcript:
        content_lines.append(transcript)
    else:
        content_lines.append("*No automatic transcript available for this video.*")

    filepath.write_text("\n".join(content_lines), encoding="utf-8")

    # Return entry for master file
    status = "" if transcript else " *(no transcript)*"
    master_entry = (
        f"## {video['title']}{status}\n\n"
        f"**Link:** {video['url']}  \n"
        f"**Playlist:** {playlist_title}  \n"
        f"**Date:** {date_display}\n\n"
    )
    if transcript:
        master_entry += f"{transcript}\n\n"
    master_entry += "---\n\n"
    return master_entry


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master_entries = []
    total_videos = 0
    total_with_transcript = 0

    # ── Get playlists ──────────────────────────────────────────────
    playlists = get_channel_playlists(CHANNEL_URL)

    if not playlists:
        print("No playlists found. Check that the channel URL is correct.")
        sys.exit(1)

    # ── Process each playlist ──────────────────────────────────────
    for playlist in playlists:
        safe_playlist = sanitize_filename(playlist["title"])[:50]
        playlist_folder = OUTPUT_DIR / safe_playlist
        playlist_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n📂 Playlist: {playlist['title']}")
        videos = get_playlist_videos(playlist["url"])
        print(f"   {len(videos)} videos")

        master_entries.append(
            f"# 📂 {playlist['title']}\n\n"
            f"**Playlist URL:** {playlist['url']}\n\n"
            f"---\n\n"
        )

        for i, video in enumerate(videos, 1):
            print(f"  [{i}/{len(videos)}] {video['title'][:60]}...", end=" ", flush=True)

            existing = find_existing_transcript(playlist_folder, video["title"])
            if existing is not None:
                print("⏭ already scraped, skipping")
                entry, had_transcript = master_entry_from_saved(existing, playlist["title"])
                master_entries.append(entry)
                total_videos += 1
                if had_transcript:
                    total_with_transcript += 1
                continue

            # Fetch real upload date from YouTube metadata if not in flat playlist
            if not video["upload_date"]:
                video["upload_date"] = get_video_upload_date(video["url"])

            transcript = download_transcript(video["url"], playlist_folder)

            if transcript:
                print("✓")
                total_with_transcript += 1
            else:
                print("– no transcript")

            entry = save_transcript(video, transcript, playlist_folder, playlist["title"])
            master_entries.append(entry)
            total_videos += 1

    # ── Write master file ──────────────────────────────────────────
    master_header = (
        f"# Dr. Childress – All Transcripts\n\n"
        f"**Channel:** {CHANNEL_URL}  \n"
        f"**Total videos:** {total_videos}  \n"
        f"**With transcript:** {total_with_transcript}  \n\n"
        f"---\n\n"
    )

    MASTER_FILE.write_text(
        master_header + "".join(master_entries),
        encoding="utf-8",
    )

    print(f"\n✅ Done!")
    print(f"   {total_videos} videos processed, {total_with_transcript} transcripts downloaded.")
    print(f"   Files saved to: {OUTPUT_DIR.resolve()}")
    print(f"   Master file: {MASTER_FILE.resolve()}")


if __name__ == "__main__":
    main()
