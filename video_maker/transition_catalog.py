"""Catalogue of transitions the program can render, mapped to ffmpeg `xfade` effects.

The names and the "when to use it" ideas follow the Clipchamp guide
https://clipchamp.com/ru/blog/video-transitions-transform-video-editing/ . Effects that need artwork
(Burn, Fire, Heart, Page Turn, Liquid, Glitch, Collage, Swap) have no ffmpeg equivalent and are not offered.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Transition:
    key: str            # id used in config.ini / command line
    xfade: str | None   # ffmpeg xfade effect (None = hard cut)
    ru: str             # name shown to the user
    analog: str         # closest Clipchamp transition
    family: str         # soft | dip | wipe | push | impact | cut
    flashy: bool        # attention-grabbing: limited so it does not distract (guide: "don't overuse")
    when: str           # short usage hint


_ITEMS = [
    Transition("cut", None, "Прямая склейка", "—", "cut", False,
               "моменты идут подряд или речь звучит с обеих сторон"),
    Transition("fade", "fade", "Плавное растворение", "Затухание при переходе (Cross Fade)", "soft", False,
               "универсальный мягкий переход, влог, спокойный ритм"),
    Transition("dissolve", "dissolve", "Зернистое растворение", "Затухание при переходе (Cross Fade)", "soft", False,
               "похожие кадры, мягкая смена"),
    Transition("blur", "hblur", "Размытие при переходе", "Размытие при переходе (Cross Blur)", "soft", False,
               "приключения, объяснения, спокойное повествование"),
    Transition("fadeblack", "fadeblack", "Исчезание через чёрное", "Исчезание через чёрное (Fade to Black)", "dip",
               False, "пауза между сценами, спад энергии, тёмные кадры"),
    Transition("fadewhite", "fadewhite", "Исчезание через белое", "Исчезание через белое (Fade to White)", "dip",
               True, "вспышка перед яркой сценой"),
    Transition("wipeleft", "wipeleft", "Затирание влево", "Затирание влево (Wipe Left)", "wipe", False,
               "смена места или темы, Shorts и Stories"),
    Transition("wiperight", "wiperight", "Затирание вправо", "Затирание вправо (Wipe Right)", "wipe", False,
               "смена места или темы"),
    Transition("wipeup", "wipeup", "Затирание вверх", "Затирание (Wipe)", "wipe", False,
               "смена темы в вертикальном видео"),
    Transition("pushup", "slideup", "Вытеснение вверх", "Вытеснение вверх (Displace Up)", "push", False,
               "быстрая смена, привычный для Shorts жест «свайп вверх»"),
    Transition("pushdown", "slidedown", "Вытеснение вниз", "Вытеснение вниз (Displace Down)", "push", False,
               "быстрая смена"),
    Transition("pushleft", "slideleft", "Вытеснение влево", "Вытеснение влево (Displace Left)", "push", False,
               "переключение места или темы, динамичный влог"),
    Transition("pushright", "slideright", "Вытеснение вправо", "Вытеснение вправо (Displace Right)", "push", False,
               "переключение места или темы"),
    Transition("zoom", "zoomin", "Увеличение", "Увеличение (Zoom In)", "impact", True,
               "нарастание энергии, игры, реакции, привлечь внимание"),
    Transition("pixelize", "pixelize", "Пикселизация", "Пикселизация (Pixelization)", "impact", True,
               "игровые и технологичные ролики"),
    Transition("diagonal", "diagtl", "Мягкая шторка по диагонали", "Мягкая шторка по диагонали", "wipe", False,
               "слайд-шоу, плавная смена темпа"),
    Transition("blinds", "hlslice", "Жалюзи", "Жалюзи (Blinds)", "impact", True,
               "обзоры, таймлапсы, смена блока"),
    Transition("reveal", "horzopen", "Раскрытие", "Раскрытие (Reveal)", "impact", True,
               "напряжённые моменты, развязка"),
    Transition("radial", "radial", "Вращательное затирание", "Вращение (Spin), приближённо", "impact", True,
               "спорт, музыка, энергичные клипы"),
    Transition("circle", "circleopen", "Расходящийся круг", "Цветение (Bloom), приближённо", "impact", True,
               "яркое начало нового эпизода"),
]

CATALOG: dict[str, Transition] = {t.key: t for t in _ITEMS}
KEYS: tuple[str, ...] = tuple(CATALOG)
# values accepted in config.ini besides the concrete keys
MODES: tuple[str, ...] = ("auto", "ask")
STYLES: tuple[str, ...] = ("auto", "calm", "neutral", "dynamic")
