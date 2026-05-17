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
    """Remove characters that are invalid in folder/file names."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.strip().strip(".")
    return name[:100]  # max 100 chars


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
    Returns the raw transcript text, or None if unavailable.
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

    # Find the downloaded .vtt file
    vtt_files = list(tmp_dir.glob("*.vtt"))
    if not vtt_files:
        return None

    vtt_text = vtt_files[0].read_text(encoding="utf-8", errors="replace")

    # Clean up tmp files
    for f in vtt_files:
        f.unlink()

    return vtt_to_clean_text(vtt_text)


def vtt_to_clean_text(vtt: str) -> str:
    """Convert VTT subtitle format to clean readable text, removing duplicates."""
    lines = vtt.splitlines()
    seen = []
    for line in lines:
        line = line.strip()
        # Skip headers, timestamps, and empty lines
        if (
            not line
            or line.startswith("WEBVTT")
            or line.startswith("Kind:")
            or line.startswith("Language:")
            or re.match(r"^\d{2}:\d{2}", line)  # timestamps
            or re.match(r"^\d+$", line)           # sequence numbers
            or line.startswith("align:")
            or line.startswith("position:")
        ):
            continue
        # Remove inline tags like <00:00:01.000><c>text</c>
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line and (not seen or seen[-1] != line):
            seen.append(line)

    # Join and collapse repeated sentences (YouTube often duplicates lines)
    text = " ".join(seen)
    # Remove runs of duplicate phrases
    text = re.sub(r"(\b.{10,80}\b) \1", r"\1", text)
    return text.strip()


def save_transcript(
    video: dict,
    transcript: str | None,
    folder: Path,
    playlist_title: str,
) -> str:
    """Save transcript to a .md file and return formatted entry for master file."""
    safe_title = sanitize_filename(video["title"])
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
        safe_playlist = sanitize_filename(playlist["title"])
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
