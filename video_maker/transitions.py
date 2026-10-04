"""Transition advisor: looks at every join between two highlight segments, decides which transition
fits best, and explains why, so the user can accept or change the proposal.

Selection rules follow the Clipchamp guide on transitions:
  * match the mood and tempo of the content (calm -> soft dissolves, energetic -> push / zoom);
  * wipes and pushes signal a change of place or topic, "fade through black" a pause or a mood drop;
  * do not overuse flashy effects, keep spacing and variety (no same effect twice in a row).

Everything is derived from cheap measurements around each cut: how far apart the two moments are in the
source, how similar the frames look, how bright they are, how much motion and sound there is.
"""
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import PipelineOptions
from .snap import Envelope
from .transition_catalog import CATALOG, KEYS

CUT_SECONDS = 0.04      # a "hard cut" inside a cross-fade chain: one or two frames
MIN_SEC, MAX_SEC = 0.15, 0.9
STYLE_BASE_SEC = {"calm": 0.6, "neutral": 0.4, "dynamic": 0.25}
SOFT_KEYS = ("fade", "dissolve", "blur", "fadeblack")


@dataclass
class TransitionChoice:
    index: int                     # join between segment `index` and `index + 1` (0-based)
    at_a: float                    # source time where the first segment ends
    at_b: float                    # source time where the next one starts
    key: str
    seconds: float
    reason: str
    alternatives: list[str] = field(default_factory=list)
    features: dict = field(default_factory=dict)

    @property
    def xfade(self) -> str:
        return CATALOG[self.key].xfade or "fade"


@dataclass
class TransitionPlan:
    style: str
    choices: list[TransitionChoice]

    def joins(self) -> list[tuple[str, float]]:
        """(ffmpeg xfade effect, seconds) per join, in order; hard cuts become a ~1 frame fade."""
        return [(c.xfade, CUT_SECONDS if c.key == "cut" else c.seconds) for c in self.choices]


# ---------------------------------------------------------------------------------------------
#  availability of effects in the installed ffmpeg
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def available_xfade() -> frozenset[str]:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-h", "filter=xfade"], capture_output=True, text=True,
                             timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1].lstrip("-").isdigit():
            names.add(parts[0])
    return frozenset(names)


def usable_keys() -> list[str]:
    """Catalogue keys the local ffmpeg can render (an old ffmpeg may lack some effects)."""
    have = available_xfade()
    return [k for k in KEYS if CATALOG[k].xfade is None or not have or CATALOG[k].xfade in have]


# ---------------------------------------------------------------------------------------------
#  measuring a cut
# ---------------------------------------------------------------------------------------------
def _grab(cap, t: float):
    import cv2

    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    return cv2.resize(frame, (160, max(1, int(160 * h / w))))


def _luma(frame) -> float:
    import cv2

    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean() / 255.0)


def _similarity(fa, fb) -> float:
    """Colour-histogram correlation, 1 = same look, <= 0.3 = clearly different scenes."""
    import cv2

    hists = []
    for f in (fa, fb):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(h, h)
        hists.append(h)
    return float(cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL))


def _motion(f1, f2) -> float:
    import cv2

    if f1 is None or f2 is None:
        return 0.5
    g1, g2 = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (f1, f2))
    return float(np.clip(cv2.absdiff(g1, g2).mean() / 12.0, 0.0, 1.0))


def _loudness(env: Envelope | None, t: float, reference: float) -> float:
    if env is None or reference <= 0:
        return 0.5
    times, rms = env
    near = (times >= t - 0.3) & (times <= t + 0.3)
    if not near.any():
        return 0.5
    return float(np.clip(rms[near].mean() / reference, 0.0, 1.0))


def measure_cut(cap, a_end: float, b_start: float, env: Envelope | None, loud_ref: float) -> dict:
    """Features of one join. Missing frames degrade to neutral values instead of failing."""
    fa, fb = _grab(cap, a_end - 0.12), _grab(cap, b_start + 0.12)
    motion_a = _motion(_grab(cap, a_end - 0.45), fa)
    motion_b = _motion(fb, _grab(cap, b_start + 0.45))
    loud_a, loud_b = _loudness(env, a_end - 0.2, loud_ref), _loudness(env, b_start + 0.2, loud_ref)
    energy = lambda m, l: float(0.6 * m + 0.4 * l) if env is not None else float(m)  # noqa: E731
    have = fa is not None and fb is not None
    return {
        "gap": max(0.0, b_start - a_end),
        "sim": _similarity(fa, fb) if have else 0.5,
        "luma_a": _luma(fa) if fa is not None else 0.5,
        "luma_b": _luma(fb) if fb is not None else 0.5,
        "energy_a": energy(motion_a, loud_a),
        "energy_b": energy(motion_b, loud_b),
        "loud_a": loud_a,
        "loud_b": loud_b,
    }


# ---------------------------------------------------------------------------------------------
#  deciding
# ---------------------------------------------------------------------------------------------
def detect_style(features: list[dict]) -> str:
    """Overall tempo of the material: calm / neutral / dynamic."""
    if not features:
        return "neutral"
    mean = float(np.mean([(f["energy_a"] + f["energy_b"]) / 2 for f in features]))
    return "dynamic" if mean >= 0.55 else "calm" if mean <= 0.30 else "neutral"


def rank_transitions(f: dict, style: str, allowed: list[str] | None = None) -> list[tuple[str, float, list[str]]]:
    """Score every transition for one join. Returns [(key, score, reasons)] best first."""
    score: dict[str, float] = {k: 0.0 for k in (allowed or list(KEYS))}
    why: dict[str, list[str]] = {k: [] for k in score}

    def add(key: str, pts: float, text: str) -> None:
        if key in score:
            score[key] += pts
            why[key].append(text)

    ea, eb = f["energy_a"], f["energy_b"]
    calm_both, busy_both = max(ea, eb) < 0.35, min(ea, eb) > 0.6
    continuous = f["gap"] < 1.5
    scene_change = f["gap"] >= 6.0 or f["sim"] < 0.35
    similar = f["sim"] >= 0.7
    dark_both = f["luma_a"] < 0.30 and f["luma_b"] < 0.30
    bright_jump = abs(f["luma_a"] - f["luma_b"]) > 0.35
    speech_both = f["loud_a"] > 0.6 and f["loud_b"] > 0.6

    add("fade", 1.0, "универсальное мягкое растворение")
    if continuous:
        add("fade", 3.0, "моменты идут почти подряд, это одна сцена: нужен мягкий стык")
        add("dissolve", 1.0, "одна сцена")
        add("cut", 1.5, "моменты идут почти подряд")
    if similar:
        add("fade", 2.0, "кадры похожи по цвету и свету")
        add("dissolve", 1.5, "кадры похожи")
        add("blur", 1.0, "кадры похожи")
    if calm_both:
        add("fade", 2.0, "спокойный ритм")
        add("blur", 2.5, "спокойный ритм: размытие выглядит профессионально")
        add("dissolve", 1.5, "спокойный ритм")
        add("fadeblack", 1.0, "спокойный ритм")
    if ea - eb > 0.35:
        add("fadeblack", 3.0, "после яркого момента нужна пауза")
        add("blur", 1.5, "спад энергии")
    if eb - ea > 0.35:
        add("zoom", 3.0, "энергия нарастает: акцент на следующем моменте")
        add("pushup", 2.0, "энергия нарастает")
        add("fadewhite", 1.5 if f["luma_a"] > 0.4 else -2.0, "вспышка перед активной сценой")
    if busy_both:
        add("pushup", 2.5, "высокая энергия с обеих сторон: быстрая смена")
        add("zoom", 2.5, "высокая энергия")
        add("wipeleft", 2.0, "высокая энергия")
        add("pixelize", 1.0, "высокая энергия")
        add("radial", 1.0, "высокая энергия")
    if scene_change:
        add("wipeleft", 2.5, "смена места или темы: затирание")
        add("pushup", 2.0, "смена места или темы: вытеснение")
        add("pushleft", 1.5, "смена места или темы")
        add("diagonal", 1.5, "смена темы")
        add("blinds", 1.0, "смена блока")
        add("reveal", 1.0, "новая сцена")
        add("fade", -1.0, "обычное растворение между разными сценами выглядит случайным")
    if similar:
        for k in ("wipeleft", "pushup", "pushleft", "blinds", "reveal", "pixelize"):
            add(k, -1.5, "кадры похожи: эффект выглядел бы необоснованным")
    if dark_both:
        add("fadeblack", 2.5, "тёмные кадры: уход в чёрное незаметен")
        add("fadewhite", -3.0, "вспышка белого на тёмном кадре режет глаз")
    if bright_jump:
        add("fadeblack", 2.0, "резкая смена яркости: плавный уход сгладит скачок")
        add("dissolve", -1.0, "скачок яркости")
    if speech_both:
        add("cut", 1.0, "речь с обеих сторон: длинное наложение звука смазало бы слова")
    if f["luma_a"] < 0.35 and f["luma_b"] >= 0.35 and not dark_both:
        add("fadewhite", -2.0, "тёмный кадр перед светлым")

    # tempo of the whole video: the guide asks for transitions that match style and mood
    for key in score:
        t = CATALOG[key]
        if score[key] > 0 and t.flashy:
            score[key] *= {"dynamic": 1.3, "neutral": 0.8, "calm": 0.25}[style]
        elif score[key] > 0 and key in SOFT_KEYS and style == "calm":
            score[key] *= 1.3

    order = {k: i for i, k in enumerate(KEYS)}
    ranked = sorted(score, key=lambda k: (-score[k], order[k]))
    return [(k, score[k], why[k]) for k in ranked]


def pick_duration(key: str, f: dict, style: str, fixed: float | None) -> float:
    if key == "cut":
        return CUT_SECONDS
    if fixed is not None:
        return float(np.clip(fixed, MIN_SEC, 2.0))
    sec = STYLE_BASE_SEC[style] * (1.25 - 0.5 * (f["energy_a"] + f["energy_b"]) / 2)
    if CATALOG[key].family == "dip":
        sec *= 1.2
    if f["loud_a"] > 0.6 and f["loud_b"] > 0.6:
        sec = min(sec, 0.25)  # keep the audio overlap short when someone is talking
    return float(np.clip(sec, MIN_SEC, MAX_SEC))


def _reason(key: str, reasons: list[str]) -> str:
    if len(reasons) > 1 and reasons[0].startswith("универсальное"):
        reasons = reasons[1:]  # the generic baseline is only worth mentioning when nothing else applies
    text = "; ".join(reasons[:2]) if reasons else CATALOG[key].when
    return text[0].upper() + text[1:]


def plan_transitions(source: str | Path, intervals: list[tuple[float, float]], options: PipelineOptions,
                     env: Envelope | None = None) -> TransitionPlan:
    """Build the proposal for all joins of a highlight (len(intervals) - 1 of them)."""
    joins = list(zip(intervals[:-1], intervals[1:]))
    allowed = usable_keys()
    fixed_key = options.transition if options.transition in KEYS else None

    if fixed_key is not None:  # the user chose one effect for everything: no analysis needed
        key = fixed_key if fixed_key in allowed else "fade"
        sec = options.transition_sec if options.transition_sec is not None else 0.35
        return TransitionPlan("manual", [
            TransitionChoice(i, a[1], b[0], key, CUT_SECONDS if key == "cut" else float(sec),
                             "задано в настройках", [], {})
            for i, (a, b) in enumerate(joins)])

    import cv2

    cap = cv2.VideoCapture(str(source))
    try:
        loud_ref = float(np.percentile(env[1], 90)) if env is not None else 0.0
        feats = [measure_cut(cap, a[1], b[0], env, loud_ref) for a, b in joins]
    finally:
        cap.release()

    return plan_from_features(joins, feats, options)


def plan_from_features(joins: list[tuple[tuple[float, float], tuple[float, float]]], feats: list[dict],
                       options: PipelineOptions) -> TransitionPlan:
    """Choose a transition for every join from already measured features (pure logic, no video access)."""
    allowed = usable_keys()
    style = options.transition_style if options.transition_style != "auto" else detect_style(feats)
    ranked_all = [rank_transitions(f, style, allowed) for f in feats]

    # pacing and variety: never the same effect twice, flashy ones are rationed
    max_flashy = {"calm": 0, "neutral": -(-len(joins) // 3), "dynamic": -(-len(joins) // 2)}[style]
    choices: list[TransitionChoice] = []
    flashy_used, prev_key, prev_flashy = 0, None, False
    for i, ((a, b), f, ranked) in enumerate(zip(joins, feats, ranked_all)):
        adjusted = []
        for key, pts, reasons in ranked:
            t = CATALOG[key]
            if key == prev_key and key != "cut":
                pts -= 1.0 if (style == "calm" and key in SOFT_KEYS) else 2.5
            elif prev_key and CATALOG[prev_key].family == t.family and t.family not in ("soft", "cut"):
                pts -= 1.0
            if t.flashy and (prev_flashy or flashy_used >= max_flashy):
                pts -= 100.0
            adjusted.append((key, pts, reasons))
        adjusted.sort(key=lambda x: -x[1])
        key, _, reasons = adjusted[0]
        alternatives = [k for k, p, _ in adjusted[1:] if p > -50][:3]
        if CATALOG[key].flashy:
            flashy_used += 1
        prev_flashy, prev_key = CATALOG[key].flashy, key
        choices.append(TransitionChoice(
            i, a[1], b[0], key, pick_duration(key, f, style, options.transition_sec),
            _reason(key, reasons), alternatives, f))
    return TransitionPlan(style, choices)


# ---------------------------------------------------------------------------------------------
#  talking to the user
# ---------------------------------------------------------------------------------------------
def _mmss(t: float) -> str:
    s = int(t)
    return f"{s // 60}:{s % 60:02d}"


def format_plan(plan: TransitionPlan) -> str:
    lines = [f"  Предлагаемые переходы (темп материала: {plan.style}):", ""]
    lines.append(f"  {'№':>2}  {'стык в исходнике':<16} {'переход':<26} {'сек':>4}  причина")
    lines.append(f"  {'-' * 2}  {'-' * 16} {'-' * 26} {'-' * 4}  {'-' * 40}")
    for c in plan.choices:
        t = CATALOG[c.key]
        where = f"{_mmss(c.at_a)} -> {_mmss(c.at_b)}"
        lines.append(f"  {c.index + 1:>2}  {where:<16} {t.ru[:26]:<26} {c.seconds:>4.2f}  {c.reason}")
    return "\n".join(lines)


def format_catalog() -> str:
    lines = ["  Доступные переходы:"]
    have = set(usable_keys())
    for k in KEYS:
        if k in have:
            lines.append(f"    {k:<10} {CATALOG[k].ru:<28} {CATALOG[k].when}")
    return "\n".join(lines)


def apply_user_edits(plan: TransitionPlan, text: str) -> tuple[TransitionPlan, list[str]]:
    """Apply edits typed by the user. Grammar (comma separated):

        2=zoom            set join 2 to a transition from the catalogue
        2=zoom:0.5        ... with a 0.5 s duration
        2=alt             take the next alternative the program suggested
        all=fade          one transition everywhere

    Returns the changed plan and a list of problems found in the input (empty = all applied).
    """
    errors: list[str] = []
    allowed = set(usable_keys())
    n = len(plan.choices)
    for part in (p.strip() for p in text.replace(";", ",").split(",")):
        if not part:
            continue
        target, sep, value = part.partition("=")
        target, value = target.strip().lower(), value.strip().lower()
        if not sep or not value:
            errors.append(f"'{part}': ожидается вид 2=zoom или all=fade")
            continue
        if target == "all":
            idxs = list(range(n))
        elif target.isdigit() and 1 <= int(target) <= n:
            idxs = [int(target) - 1]
        else:
            errors.append(f"'{part}': нет стыка '{target}' (есть 1-{n})")
            continue
        key, _, sec_text = value.partition(":")
        seconds = None
        if sec_text:
            try:
                seconds = float(sec_text.replace(",", "."))
            except ValueError:
                errors.append(f"'{part}': неверная длительность '{sec_text}'")
                continue
            if not MIN_SEC <= seconds <= 2.0:
                errors.append(f"'{part}': длительность должна быть {MIN_SEC}-2 с")
                continue
        for i in idxs:
            c = plan.choices[i]
            if key == "alt":
                options = [k for k in c.alternatives if k in allowed]
                if not options:
                    errors.append(f"'{part}': для стыка {i + 1} нет альтернатив")
                    continue
                c.alternatives = [*options[1:], c.key]
                c.key = options[0]
            elif key in allowed:
                if key != c.key:
                    c.alternatives = [k for k in [c.key, *c.alternatives] if k != key][:3]
                c.key = key
            else:
                errors.append(f"'{part}': неизвестный переход '{key}' (список: list)")
                break
            if c.key == "cut":
                c.seconds = CUT_SECONDS
            elif seconds is not None:
                c.seconds = seconds
            elif c.seconds <= CUT_SECONDS:
                c.seconds = STYLE_BASE_SEC.get(plan.style, 0.4)
            c.reason = "выбрано пользователем"
    return plan, errors
