#!/usr/bin/env python3
"""
Helper script to download podcast transcripts.
Supports various sources and formats.
"""

import argparse
import sys
from pathlib import Path


def download_from_youtube(video_id: str, output_path: str) -> bool:
    """
    Download transcript from YouTube video.
    Requires youtube-transcript-api package.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        print(f"Downloading transcript for YouTube video: {video_id}")

        # Get transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

        # Combine into text
        full_text = "\n".join([entry['text'] for entry in transcript_list])

        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)

        print(f"Saved transcript to: {output_path}")
        print(f"Total length: {len(full_text):,} characters")

        return True

    except ImportError:
        print("Error: youtube-transcript-api not installed")
        print("Install with: pip install youtube-transcript-api")
        return False
    except Exception as e:
        print(f"Error downloading transcript: {e}")
        return False


def download_from_url(url: str, output_path: str) -> bool:
    """
    Download transcript from URL.
    """
    try:
        import requests

        print(f"Downloading from URL: {url}")

        response = requests.get(url)
        response.raise_for_status()

        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f"Saved transcript to: {output_path}")
        print(f"Total length: {len(response.text):,} characters")

        return True

    except ImportError:
        print("Error: requests not installed")
        print("Install with: pip install requests")
        return False
    except Exception as e:
        print(f"Error downloading from URL: {e}")
        return False


def convert_srt_to_text(srt_path: str, output_path: str) -> bool:
    """
    Convert SRT subtitle file to plain text.
    """
    try:
        import re

        print(f"Converting SRT file: {srt_path}")

        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove SRT formatting
        # Pattern: number, timestamp, text
        lines = content.split('\n')
        text_lines = []

        skip_next = False
        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                skip_next = False
                continue

            # Skip sequence numbers
            if line.isdigit():
                continue

            # Skip timestamps
            if '-->' in line:
                continue

            # Add text
            text_lines.append(line)

        full_text = ' '.join(text_lines)

        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)

        print(f"Saved converted text to: {output_path}")
        print(f"Total length: {len(full_text):,} characters")

        return True

    except Exception as e:
        print(f"Error converting SRT: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert podcast transcripts"
    )

    parser.add_argument(
        "--source",
        choices=["youtube", "url", "srt"],
        required=True,
        help="Transcript source type"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="YouTube video ID, URL, or SRT file path"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path"
    )

    args = parser.parse_args()

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download based on source
    if args.source == "youtube":
        success = download_from_youtube(args.input, args.output)
    elif args.source == "url":
        success = download_from_url(args.input, args.output)
    elif args.source == "srt":
        success = convert_srt_to_text(args.input, args.output)
    else:
        print(f"Unsupported source: {args.source}")
        sys.exit(1)

    if success:
        print("\nSuccess!")
        sys.exit(0)
    else:
        print("\nFailed to download/convert transcript")
        sys.exit(1)


if __name__ == "__main__":
    main()
