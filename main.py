"""Entry point for launching the Centroid visual experience."""
from __future__ import annotations

from centroid.visualizer import AudioReactiveVisualizer


def main() -> None:
    visualizer = AudioReactiveVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()
