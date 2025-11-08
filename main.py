"""Entry point for launching the Centroid visual experience."""
from __future__ import annotations

import argparse
from pathlib import Path

from centroid.visualizer import AudioReactiveVisualizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Centroid visualizer")
    parser.add_argument(
        "--file",
        type=str,
        help="Optional path to an audio file to visualise instead of the live microphone",
    )
    args = parser.parse_args()

    audio_path = Path(args.file).expanduser() if args.file else None
    if audio_path and not audio_path.exists():
        parser.error(f"Audio file not found: {audio_path}")

    visualizer = AudioReactiveVisualizer(audio_path=str(audio_path) if audio_path else None)
    visualizer.run()


if __name__ == "__main__":
    main()
