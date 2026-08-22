"""ANSI color conversion for the launcher console.

Search keywords: ANSI_HTML, CONSOLE_COLORS, XTERM_256
"""

from __future__ import annotations

import html
import re


ANSI_PATTERN = re.compile(r"\x1b\[([0-9;]*)m")
FG_COLORS = {
    30: "#0c0c0c", 31: "#c50f1f", 32: "#13a10e", 33: "#c19c00",
    34: "#0037da", 35: "#881798", 36: "#3a96dd", 37: "#cccccc",
    90: "#767676", 91: "#e74856", 92: "#16c60c", 93: "#f9f1a5",
    94: "#3b78ff", 95: "#b4009e", 96: "#61d6d6", 97: "#f2f2f2",
}


def _xterm_256_to_hex(code: int) -> str:
    legacy = [
        "#000000", "#800000", "#008000", "#808000", "#000080", "#800080",
        "#008080", "#c0c0c0", "#808080", "#ff0000", "#00ff00", "#ffff00",
        "#0000ff", "#ff00ff", "#00ffff", "#ffffff",
    ]
    if 0 <= code <= 15:
        return legacy[code]
    if 16 <= code <= 231:
        value = code - 16
        levels = [0, 95, 135, 175, 215, 255]
        red = levels[(value // 36) % 6]
        green = levels[(value // 6) % 6]
        blue = levels[value % 6]
        return f"#{red:02x}{green:02x}{blue:02x}"
    gray = max(0, min(8 + ((code - 232) * 10), 255))
    return f"#{gray:02x}{gray:02x}{gray:02x}"


def _style(state: dict[str, str | bool]) -> str:
    rules: list[str] = []
    if state.get("color"):
        rules.append(f"color:{state['color']}")
    if state.get("bold"):
        rules.append("font-weight:600")
    return ";".join(rules)


def ansi_text_to_html(text: str, state: dict[str, str | bool]) -> str:
    """Convert ANSI SGR text to HTML while retaining style state."""
    result: list[str] = []
    last_index = 0

    def append(raw: str) -> None:
        if not raw:
            return
        escaped = html.escape(raw).replace(" ", "&nbsp;")
        css = _style(state)
        result.append(f'<span style="{css}">{escaped}</span>' if css else escaped)

    for match in ANSI_PATTERN.finditer(text):
        append(text[last_index:match.start()])
        codes = [int(part) for part in match.group(1).split(";") if part.isdigit()] or [0]
        index = 0
        while index < len(codes):
            code = codes[index]
            if code == 0:
                state.update(color="", bold=False)
            elif code == 1:
                state["bold"] = True
            elif code == 22:
                state["bold"] = False
            elif code == 39:
                state["color"] = ""
            elif code in FG_COLORS:
                state["color"] = FG_COLORS[code]
            elif code == 38 and index + 1 < len(codes):
                mode = codes[index + 1]
                if mode == 5 and index + 2 < len(codes):
                    state["color"] = _xterm_256_to_hex(codes[index + 2])
                    index += 2
                elif mode == 2 and index + 4 < len(codes):
                    red, green, blue = (max(0, min(value, 255)) for value in codes[index + 2:index + 5])
                    state["color"] = f"#{red:02x}{green:02x}{blue:02x}"
                    index += 4
            index += 1
        last_index = match.end()
    append(text[last_index:])
    return "".join(result)
