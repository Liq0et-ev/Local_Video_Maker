import os
from PIL import Image, ImageDraw, ImageFont

from ..config import (

    SAFE_ZONE_LEFT, SAFE_ZONE_RIGHT,
)

WORDS_PER_GROUP = 3


def _hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _hex_to_rgba(hex_color):
    h = hex_color.lstrip("#")
    if len(h) == 8:
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4, 6))
    return (*_hex_to_rgb(hex_color), 255)


def _apply_opacity(rgb, opacity_pct):
    """Apply opacity (0-100%) to an RGB tuple, returning RGBA."""
    alpha = int((opacity_pct / 100.0) * 255)
    return (*rgb, alpha)


class SubtitleRenderer:

    def __init__(self, width, height, style):
        self.width = width
        self.height = height
        self.style = style
        self.font = self._load_font()

        # Usable horizontal area
        self.safe_width = self.width - SAFE_ZONE_LEFT - SAFE_ZONE_RIGHT

        # Opacity
        self.base_opacity = self.style.get("opacity", 100)

    def _load_font(self):
        size = self.style["font_size"]
        bold = self.style.get("bold", True)
        italic = self.style.get("italic", False)

        if bold and italic:
            candidates = [
                "arialbi.ttf", "Arial Bold Italic.ttf",
                "DejaVuSans-BoldOblique.ttf",
                "LiberationSans-BoldItalic.ttf",
            ]
        elif bold:
            candidates = [
                "arialbd.ttf", "Arial Bold.ttf", "Arial-Bold",
                "DejaVuSans-Bold.ttf",
                "LiberationSans-Bold.ttf",
            ]
        elif italic:
            candidates = [
                "ariali.ttf", "Arial Italic.ttf",
                "DejaVuSans-Oblique.ttf",
                "LiberationSans-Italic.ttf",
            ]
        else:
            candidates = [
                "arial.ttf", "Arial.ttf",
                "DejaVuSans.ttf",
                "LiberationSans-Regular.ttf",
            ]

        font_dirs = []
        if os.name == "nt":
            font_dirs.append(r"C:\Windows\Fonts")
        else:
            font_dirs.extend([
                "/usr/share/fonts/truetype",
                "/usr/share/fonts",
                os.path.expanduser("~/.local/share/fonts"),
            ])

        for name in candidates:
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                pass
            for d in font_dirs:
                if not os.path.isdir(d):
                    continue
                for root, _, files in os.walk(d):
                    for f in files:
                        if name.lower() in f.lower():
                            try:
                                return ImageFont.truetype(os.path.join(root, f), size)
                            except OSError:
                                continue

        print("  Warning: font not found, using default.")
        return ImageFont.load_default()

    # --------------------------------------------------------------
    # Vertical positioning -- strictly enforced
    # --------------------------------------------------------------
    def _get_position(self, text_w, text_h):
        """
        Top:    Y = 10% from top edge
        Middle: Y = dead center
        Bottom: Y = 10% from bottom edge
        Horizontal: always centered
        """
        pos = self.style["position"]

        # Horizontal center
        x = (self.width - text_w) // 2
        x = max(SAFE_ZONE_LEFT, x)

        # Vertical
        if pos == "top":
            y = int(self.height * 0.10)

        elif pos == "bottom":
            y = int(self.height * 0.90) - text_h

        else:  # "center" / "middle"
            y = (self.height - text_h) // 2

        return x, y

    def _draw_outlined(self, draw, x, y, text, fill):
        """Draw text with outline. fill can be RGB or RGBA."""
        ow = self.style["outline_width"]
        outline_rgb = _hex_to_rgb(self.style["outline_color"])

        if len(fill) == 4:
            outline_color = (*outline_rgb, fill[3])
        else:
            outline_color = outline_rgb

        for dx in range(-ow, ow + 1):
            for dy in range(-ow, ow + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text, font=self.font, fill=outline_color)
        draw.text((x, y), text, font=self.font, fill=fill)

    # --------------------------------------------------------------
    # THREE-WORD RULE: find active 3-word group for current time
    # --------------------------------------------------------------
    def _get_active_group(self, words, current_time):
        """
        Split all words into groups of 3. Return the group that
        should be visible at `current_time`, or None if no group
        is active.

        A group is visible from its first word's start time
        until its last word's end time. This ensures:
          - Exactly 3 words (or fewer for the final group) on screen
          - Screen clears between groups
          - Speed adapts to speech pace
        """
        groups = []
        for i in range(0, len(words), WORDS_PER_GROUP):
            groups.append(words[i:i + WORDS_PER_GROUP])

        for group in groups:
            group_start = group[0]["start"]
            group_end = group[-1]["end"]
            if group_start <= current_time <= group_end:
                return group

        return None

    # --------------------------------------------------------------
    # Main render entry point
    # --------------------------------------------------------------
    def render_frame(self, segment, current_time, frame):
        """Burn subtitles onto a single PIL Image frame."""
        words = segment["words"]
        if not words:
            return frame

        # Find which 3-word group is active right now
        group = self._get_active_group(words, current_time)
        if group is None:
            return frame  # Between groups -- screen is clear

        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Build the display text and measure it
        group_text = " ".join(w["word"] for w in group)
        bbox = draw.textbbox((0, 0), group_text, font=self.font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # If 3 words still overflow, wrap to 2 lines
        if text_w > self.safe_width:
            lines = self._wrap_group(group, draw)
        else:
            lines = [group]

        line_h = self.style["font_size"] + 10
        total_h = len(lines) * line_h

        # Measure widest line
        max_w = 0
        for line in lines:
            txt = " ".join(w["word"] for w in line)
            b = draw.textbbox((0, 0), txt, font=self.font)
            max_w = max(max_w, b[2] - b[0])

        base_x, base_y = self._get_position(max_w, total_h)

        # Optional background
        bg = self.style.get("background_color")
        if bg:
            pad = self.style["background_padding"]
            draw.rounded_rectangle(
                [base_x - pad, base_y - pad,
                 base_x + max_w + pad, base_y + total_h + pad],
                radius=8,
                fill=_hex_to_rgba(bg),
            )

        # Colors with opacity
        primary_rgb = _hex_to_rgb(self.style["primary_color"])
        highlight_rgb = _hex_to_rgb(self.style["highlight_color"])
        primary_color = _apply_opacity(primary_rgb, self.base_opacity)
        highlight_color = (*highlight_rgb, 255)  # Always fully opaque

        # Draw each line with word-by-word highlighting
        for li, line in enumerate(lines):
            y = base_y + li * line_h
            line_txt = " ".join(w["word"] for w in line)
            b = draw.textbbox((0, 0), line_txt, font=self.font)
            line_w = b[2] - b[0]
            cx = base_x + (max_w - line_w) // 2

            for w in line:
                active = w["start"] <= current_time <= w["end"]
                color = highlight_color if active else primary_color
                self._draw_outlined(draw, cx, y, w["word"], color)
                wb = draw.textbbox((0, 0), w["word"] + " ", font=self.font)
                cx += wb[2] - wb[0]

        # Composite
        if frame.mode != "RGBA":
            frame = frame.convert("RGBA")
        result = Image.alpha_composite(frame, overlay)
        return result.convert("RGB")

    def _wrap_group(self, group, draw):
        """
        If even 3 words overflow the safe width, split into
        multiple lines. Each line must fit within safe_width.
        """
        lines = []
        current_line = []

        for w in group:
            test = current_line + [w]
            txt = " ".join(wd["word"] for wd in test)
            b = draw.textbbox((0, 0), txt, font=self.font)
            if current_line and (b[2] - b[0]) > self.safe_width:
                lines.append(current_line)
                current_line = [w]
            else:
                current_line = test

        if current_line:
            lines.append(current_line)

        return lines
