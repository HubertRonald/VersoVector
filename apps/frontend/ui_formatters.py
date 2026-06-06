from __future__ import annotations

import json
import re
from html import escape
from typing import Any

from tag_catalog import icon_name_for_tag, svg_icon


def clean_display_text(value: Any, fallback: str = "Unknown") -> str:
    """Clean user-facing display text from raw metadata."""
    if value is None:
        return fallback

    text = str(value)
    text = text.replace("\\r", " ").replace("\\n", " ")
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("\"'")

    return text if text else fallback


def pick_display_value(
    primary: Any,
    fallback_value: Any,
    fallback: str = "Unknown",
) -> str:
    """Prefer raw display metadata, then fallback to normalized value."""
    cleaned_primary = clean_display_text(primary, fallback="")
    if cleaned_primary:
        return cleaned_primary

    return clean_display_text(fallback_value, fallback=fallback)


def score_to_percent(score: Any) -> float:
    """Convert a model score to a percentage value."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0

    if value <= 1.0:
        value *= 100.0

    return max(0.0, min(value, 100.0))


def format_percent(score: Any) -> str:
    """Format a score as ##.#%."""
    return f"{score_to_percent(score):.1f}%"


def split_terms(value: Any) -> list[str]:
    """Split topic terms into a clean list."""
    if value is None:
        return []

    if isinstance(value, list):
        return [clean_display_text(item, fallback="") for item in value if str(item).strip()]

    return [
        clean_display_text(item, fallback="")
        for item in str(value).split(",")
        if str(item).strip()
    ]


def render_api_status(status: str = "unknown", detail: str | None = None) -> str:
    """Render a compact API status pill."""
    normalized = status.lower().strip()

    if normalized in {"ok", "ready", "healthy"}:
        css_class = "vv-status-ok"
        label = "API healthy"
        icon = "●"
    elif normalized in {"checking"}:
        css_class = "vv-status-warn"
        label = "Checking API"
        icon = "●"
    else:
        css_class = "vv-status-error"
        label = "API unavailable"
        icon = "●"

    title = escape(detail or label)

    return f"""
    <div class="vv-status-pill {css_class}" title="{title}">
        <span>{icon}</span>
        <span>{escape(label)}</span>
    </div>
    """


def render_empty_results() -> str:
    """Render empty state for predicted tags."""
    return """
    <div class="vv-card vv-empty-card">
        <div class="vv-card-title">
            <span class="vv-icon">🌿</span>
            <div>
                <h3>Predicted tags</h3>
                <p>Your predicted emotional tags will appear here after analysis.</p>
            </div>
        </div>
    </div>
    """


def render_empty_similar_poems() -> str:
    """Render empty state for similar poems."""
    return """
    <div class="vv-card vv-empty-card">
        <div class="vv-card-title">
            <span class="vv-icon">📖</span>
            <div>
                <h3>Similar poems</h3>
                <p>Semantic neighbors will appear here once your text has been analyzed.</p>
            </div>
        </div>
    </div>
    """


def render_empty_literary_insights() -> str:
    """Render empty state for literary insights."""
    return """
    <div class="vv-card vv-empty-card">
        <div class="vv-card-title">
            <span class="vv-icon">✨</span>
            <div>
                <h3>Literary insights</h3>
                <p>Topic and cluster signals will appear after analysis.</p>
            </div>
        </div>
    </div>
    """


def render_predicted_tags(result: dict[str, Any]) -> str:
    """Render predicted tags as polished progress bars."""
    tags = result.get("predicted_tags") or []

    if not tags:
        return render_empty_results()

    rows = []

    for item in tags:
        tag = clean_display_text(item.get("tag"), fallback="unknown")
        score = item.get("score")
        percent_value = score_to_percent(score)
        percent_label = format_percent(score)
        icon = svg_icon(icon_name_for_tag(tag))

        rows.append(
            f"""
            <div class="vv-tag-row">
                <div class="vv-tag-name">
                    <span class="vv-row-icon">{icon}</span>
                    <span>{escape(tag)}</span>
                </div>
                <div class="vv-progress-track" aria-label="{escape(tag)} score">
                    <div class="vv-progress-fill" style="width: {percent_value:.2f}%"></div>
                </div>
                <div class="vv-percent">{escape(percent_label)}</div>
            </div>
            """
        )

    return f"""
    <div class="vv-card">
        <div class="vv-card-header">
            <div class="vv-card-title">
                <span class="vv-icon">🌿</span>
                <div>
                    <h3>Predicted tags</h3>
                    <p>Emotional and semantic keywords detected in your text.</p>
                </div>
            </div>
            <span class="vv-subtle-badge">Top tags</span>
        </div>
        <div class="vv-tag-list">
            {''.join(rows)}
        </div>
    </div>
    """


def render_similar_poems(result: dict[str, Any]) -> str:
    """Render similar poems using clean title_raw and poet_raw values."""
    similar_poems = result.get("similar_poems") or []

    if not similar_poems:
        return render_empty_similar_poems()

    rows = []

    for item in similar_poems:
        title = pick_display_value(
            item.get("title_raw"),
            item.get("title"),
            fallback="Untitled",
        )
        poet = pick_display_value(
            item.get("poet_raw"),
            item.get("poet"),
            fallback="Unknown",
        )
        score = item.get("score")
        percent_value = score_to_percent(score)
        percent_label = format_percent(score)

        rows.append(
            f"""
            <tr>
                <td>
                    <span class="vv-book-icon">{svg_icon("book")}</span>
                    {escape(title)}
                </td>
                <td>{escape(poet)}</td>
                <td>
                    <div class="vv-similarity-cell">
                        <span>{escape(percent_label)}</span>
                        <span class="vv-mini-track">
                            <span class="vv-mini-fill" style="width: {percent_value:.2f}%"></span>
                        </span>
                    </div>
                </td>
            </tr>
            """
        )

    return f"""
    <div class="vv-card">
        <div class="vv-card-header">
            <div class="vv-card-title">
                <span class="vv-icon">📖</span>
                <div>
                    <h3>Similar poems</h3>
                    <p>Top semantic neighbors most similar to your text.</p>
                </div>
            </div>
            <span class="vv-subtle-badge">Semantic neighbors</span>
        </div>
        <div class="vv-table-wrap">
            <table class="vv-table">
                <thead>
                    <tr>
                        <th>Title</th>
                        <th>Poet</th>
                        <th>Similarity</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
    </div>
    """


def render_literary_insights(result: dict[str, Any]) -> str:
    """Render topic and cluster model signals."""
    topic = result.get("topic") or {}
    cluster = result.get("cluster") or {}
    model_info = result.get("model_info") or {}

    topic_id = topic.get("topic_id")
    probability = topic.get("probability")
    probability_label = format_percent(probability)
    probability_width = score_to_percent(probability)

    terms = split_terms(topic.get("terms"))
    term_chips = "".join(
        f'<span class="vv-term-chip">{escape(term)}</span>'
        for term in terms[:8]
    )

    if len(terms) > 8:
        term_chips += f'<span class="vv-term-chip">+{len(terms) - 8}</span>'

    kmeans = cluster.get("kmeans")
    gmm = cluster.get("gmm")

    project = clean_display_text(model_info.get("project"), fallback="versovector")
    task = clean_display_text(
        model_info.get("task"),
        fallback="emotional_semantic_recommendation",
    )

    return f"""
    <div class="vv-card">
        <div class="vv-card-header">
            <div class="vv-card-title">
                <span class="vv-icon">✨</span>
                <div>
                    <h3>Literary insights</h3>
                    <p>Topic modeling and clustering signals. Useful for exploration, not absolute interpretation.</p>
                </div>
            </div>
            <span class="vv-premium-badge">Advanced</span>
        </div>

        <div class="vv-insights-grid">
            <div class="vv-mini-card">
                <div class="vv-mini-title">
                    Topic summary
                    <span class="vv-tooltip" title="Topic modeling signal inferred from the text.">ⓘ</span>
                </div>
                <div class="vv-kpi-row">
                    <div>
                        <span class="vv-kpi-label">Topic ID</span>
                        <strong>{escape(str(topic_id))}</strong>
                    </div>
                    <div>
                        <span class="vv-kpi-label">Probability</span>
                        <strong>{escape(probability_label)}</strong>
                    </div>
                </div>
                <div class="vv-progress-track vv-topic-track">
                    <div class="vv-progress-fill" style="width: {probability_width:.2f}%"></div>
                </div>
                <div class="vv-term-list">
                    {term_chips or '<span class="vv-muted">No topic terms available</span>'}
                </div>
            </div>

            <div class="vv-mini-card">
                <div class="vv-mini-title">
                    Cluster summary
                    <span class="vv-tooltip" title="Model grouping based on learned semantic patterns.">ⓘ</span>
                </div>
                <div class="vv-kpi-row">
                    <div>
                        <span class="vv-kpi-label">KMeans</span>
                        <strong>{escape(str(kmeans))}</strong>
                    </div>
                    <div>
                        <span class="vv-kpi-label">GMM</span>
                        <strong>{escape(str(gmm))}</strong>
                    </div>
                </div>
                <div class="vv-model-meta">
                    <span><strong>Model:</strong> {escape(project)}</span>
                    <span><strong>Task:</strong> {escape(task)}</span>
                </div>
            </div>
        </div>
    </div>
    """


def render_error(message: str) -> str:
    """Render a user-facing error card."""
    return f"""
    <div class="vv-card vv-error-card">
        <h3>Could not complete the analysis</h3>
        <p>{escape(message)}</p>
    </div>
    """


def render_raw_json(data: dict[str, Any] | None) -> str:
    """Format raw JSON for display."""
    return json.dumps(data or {}, ensure_ascii=False, indent=2, default=str)
