"""Local Video Maker: turn long videos from a folder into YouTube Shorts.

    python main.py            # interactive
    python main.py --help
"""
import sys
from pathlib import Path

from video_maker.cli import main

# ============================================================================
#  YOUR FOLDERS - type them here (or leave "" to use config.ini)
#
#  Examples:   INPUT_DIR  = r"D:\Videos\Raw"
#              MUSIC_DIR  = r"D:\Music\Background"
#              OUTPUT_DIR = r"D:\Videos\Shorts"
#
#  Relative paths are counted from the folder where this main.py lies.
#  Command-line flags (--input-dir ...) still win over these values.
# ============================================================================
INPUT_DIR = r""     # where the long source videos are taken from
MUSIC_DIR = r""     # where the background music (.mp3 / .wav) is taken from
OUTPUT_DIR = r""    # where the finished shorts are saved
WORK_DIR = r""      # temporary files (optional)

# True = the program asks for the three folders every time it starts
# (Enter keeps the value shown in brackets).
ASK_FOLDERS = False
# ============================================================================

if __name__ == "__main__":
    sys.exit(main(
        folders={"input_dir": INPUT_DIR, "music_dir": MUSIC_DIR,
                 "output_dir": OUTPUT_DIR, "work_dir": WORK_DIR},
        folders_base=Path(__file__).resolve().parent,
        ask_for_folders=ASK_FOLDERS,
    ))
