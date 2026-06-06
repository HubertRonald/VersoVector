from __future__ import annotations

from html import escape


TAG_CATALOG: list[dict[str, str]] = [
    {"id": "memory", "label": "memory", "icon": "spiral"},
    {"id": "light", "label": "light", "icon": "spark"},
    {"id": "night", "label": "night", "icon": "moon"},
    {"id": "rain", "label": "rain", "icon": "rain"},
    {"id": "nature", "label": "nature", "icon": "leaf"},
    {"id": "hope", "label": "hope", "icon": "seedling"},
    {"id": "loneliness", "label": "loneliness", "icon": "circle"},
    {"id": "journey", "label": "journey", "icon": "path"},
    {"id": "reflection", "label": "reflection", "icon": "water"},
    {"id": "change", "label": "change", "icon": "refresh"},
    {"id": "time", "label": "time", "icon": "hourglass"},
    {"id": "love", "label": "love", "icon": "heart"},
    {"id": "loss", "label": "loss", "icon": "circle"},
    {"id": "home", "label": "home", "icon": "home"},
    {"id": "silence", "label": "silence", "icon": "wave"},
]

TAG_ICON_BY_LABEL: dict[str, str] = {
    item["label"]: item["icon"]
    for item in TAG_CATALOG
}

DEFAULT_TAGS: list[str] = [
    "memory",
    "light",
]

SVG_ICONS: dict[str, str] = {
    "leaf": """
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5 19C6.5 11 11 6.5 19 5C17.5 13 13 17.5 5 19Z"/>
      <path d="M5 19C9 14 13 10 19 5"/>
    </svg>
    """,
    "spark": """
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3L13.8 9.2L20 11L13.8 12.8L12 19L10.2 12.8L4 11L10.2 9.2L12 3Z"/>
    </svg>
    """,
    "moon": """
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M18 15.5A7 7 0 0 1 8.5 6A8 8 0 1 0 18 15.5Z"/>
    </svg>
    """,
    "rain": """
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M7 14H17A4 4 0 0 0 17 6A6 6 0 0 0 5.5 8.5A3.5 3.5 0 0 0 7 14Z"/>
      <path d="M8 18L7 21"/>
      <path d="M12 18L11 21"/>
      <path d="M16 18L15 21"/>
    </svg>
    """,
    "hourglass": """
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M7 4H17"/>
      <path d="M7 20H17"/>
      <path d="M8 4C8 9 16 9 16 14C16 16 14 18 12 20C10 18 8 16 8 14C8 9 16 9 16 4"/>
    </svg>
    """,
    "heart": """
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 20S5 15.5 5 9.5A4 4 0 0 1 12 7A4 4 0 0 1 19 9.5C19 15.5 12 20 12 20Z"/>
    </svg>
    """,
    "book": """
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 5.5C6.8 4.5 9.2 4.8 12 6V20C9.2 18.8 6.8 18.5 4 19.5V5.5Z"/>
      <path d="M20 5.5C17.2 4.5 14.8 4.8 12 6V20C14.8 18.8 17.2 18.5 20 19.5V5.5Z"/>
    </svg>
    """,
    "circle": """
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="7"/>
    </svg>
    """,
    "default": """
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 4L19 12L12 20L5 12L12 4Z"/>
    </svg>
    """,
}


def tag_choices() -> list[str]:
    """Return visible tag choices for the frontend."""
    return [item["label"] for item in TAG_CATALOG]


def icon_name_for_tag(tag: str) -> str:
    """Return icon name for a tag label."""
    return TAG_ICON_BY_LABEL.get(str(tag).lower().strip(), "default")


def svg_icon(name: str, css_class: str = "vv-svg-icon") -> str:
    """Return a small inline SVG icon."""
    svg = SVG_ICONS.get(name, SVG_ICONS["default"])
    return f'<span class="{escape(css_class)}">{svg}</span>'
