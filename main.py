"""Local Video Maker: turn long videos from a folder into YouTube Shorts.

    python main.py            # interactive
    python main.py --help
"""
import sys

from video_maker.cli import main

if __name__ == "__main__":
    sys.exit(main())
