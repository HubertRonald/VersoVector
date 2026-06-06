from __future__ import annotations

import os
from typing import Any

import gradio as gr
import requests

from client import analyze_poem, get_api_base_url, health_check, readiness_check
from ui_formatters import (
    render_api_status,
    render_empty_literary_insights,
    render_empty_results,
    render_empty_similar_poems,
    render_error,
    render_literary_insights,
    render_predicted_tags,
    render_raw_json,
    render_similar_poems,
)
from tag_catalog import DEFAULT_TAGS, tag_choices


APP_TITLE = "VersoVector"
APP_SUBTITLE = "Emotional-semantic analysis for poetic and lyrical language."
DEFAULT_SAMPLE_TEXT = "I walk through the rain carrying a memory of light."

selected_tags_input = gr.CheckboxGroup(
    label="Suggested tags",
    choices=tag_choices(),
    value=DEFAULT_TAGS,
    info="Select words that describe the mood or themes you expect.",
    elem_id="suggested-tags",
)


CUSTOM_CSS = """
:root {
    --vv-bg: #F8F4EC;
    --vv-card: #FFFDF8;
    --vv-card-soft: #FBF7EF;
    --vv-text: #252A27;
    --vv-muted: #5F625D;
    --vv-border: #E5DED2;
    --vv-green: #315F4A;
    --vv-green-dark: #163D2F;
    --vv-green-soft: #EEF5EF;
    --vv-green-muted: #6F8F7A;
    --vv-teal: #527C70;
    --vv-button: #315F4A;
    --vv-button-hover: #284E3D;
    --vv-premium-bg: #FBF2D8;
    --vv-premium-border: #E2C878;
    --vv-premium-text: #6A531F;
    --vv-warning: #FFF5D8;
    --vv-shadow: 0 14px 40px rgba(52, 45, 35, 0.08);
}

#vv-app-shell,
.gradio-container {
    --button-primary-background-fill: #315F4A !important;
    --button-primary-background-fill-hover: #284E3D !important;
    --button-primary-text-color: #FFFDF8 !important;
    --checkbox-label-background-fill-selected: #EEF5EF !important;
    --checkbox-label-border-color-selected: #315F4A !important;
    --checkbox-label-text-color-selected: #163D2F !important;
}

.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(234, 241, 232, 0.9) 0%, transparent 28%),
        linear-gradient(135deg, #F8F4EC 0%, #FBF7EF 48%, #F2E8DA 100%) !important;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    color: var(--vv-text);
}

#vv-app-shell {
    max-width: 1360px;
    margin: 0 auto;
}

.vv-hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    padding: 24px 28px;
    margin-bottom: 18px;
    border: 1px solid var(--vv-border);
    border-radius: 22px;
    background:
        linear-gradient(90deg, rgba(255, 253, 248, 0.98), rgba(255, 253, 248, 0.88)),
        radial-gradient(circle at 90% 30%, rgba(95, 141, 131, 0.15), transparent 32%);
    box-shadow: var(--vv-shadow);
}

.vv-brand {
    display: flex;
    align-items: center;
    gap: 18px;
}

.vv-logo {
    width: 64px;
    height: 64px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(238, 245, 239, 0.95), rgba(255, 253, 248, 0.96));
    display: grid;
    place-items: center;
    color: var(--vv-green);
    border: 1px solid rgba(49, 95, 74, 0.18);
    box-shadow: inset 0 0 0 1px rgba(255, 253, 248, 0.7);
}

.vv-logo-svg {
    width: 46px;
    height: 46px;
}

.vv-logo-svg path {
    fill: none;
    stroke: var(--vv-green);
    stroke-width: 3.1;
    stroke-linecap: round;
    stroke-linejoin: round;
}

.vv-logo-leaf {
    fill: rgba(238, 245, 239, 0.65);
}

.vv-brand h1 {
    margin: 0;
    color: var(--vv-green-dark);
    font-family: Georgia, "Times New Roman", serif;
    font-size: 44px;
    line-height: 1;
    letter-spacing: -0.04em;
}

.vv-brand p {
    margin: 8px 0 0 0;
    color: var(--vv-muted);
    font-size: 15px;
}

.vv-hero-actions {
    display: flex;
    align-items: center;
    gap: 10px;
}

.vv-status-pill,
.vv-soft-pill,
.vv-premium-badge,
.vv-subtle-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 999px;
    padding: 9px 13px;
    font-size: 13px;
    font-weight: 650;
    border: 1px solid var(--vv-border);
    background: rgba(255, 253, 248, 0.85);
    white-space: nowrap;
}

.vv-status-ok {
    color: var(--vv-green-dark);
    background: #EEF7EE;
    border-color: rgba(63, 111, 88, 0.18);
}

.vv-status-warn {
    color: #7A5A18;
    background: var(--vv-warning);
}

.vv-status-error {
    color: #8A2D2D;
    background: #FFF0EF;
}

.vv-premium-badge {
    background: var(--vv-premium-bg);
    color: var(--vv-premium-text);
    border-color: var(--vv-premium-border);
}

.vv-card {
    border: 1px solid var(--vv-border);
    border-radius: 20px;
    background: rgba(255, 253, 248, 0.96);
    box-shadow: var(--vv-shadow);
    padding: 22px;
    margin-bottom: 14px;
}

.vv-empty-card {
    min-height: 170px;
    display: flex;
    align-items: center;
}

.vv-error-card {
    border-color: #F0C4BE;
    background: #FFF5F2;
}

.vv-error-card h3 {
    margin-top: 0;
    color: #8A2D2D;
}

.vv-card-header,
.vv-card-title {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
}

.vv-card-title {
    justify-content: flex-start;
}

.vv-card h3 {
    margin: 0;
    font-size: 18px;
    color: var(--vv-text);
}

.vv-card p {
    margin: 5px 0 0 0;
    color: var(--vv-muted);
    font-size: 13px;
}

.vv-icon {
    width: 34px;
    height: 34px;
    display: inline-grid;
    place-items: center;
    border-radius: 12px;
    background: var(--vv-green-soft);
    color: var(--vv-green);
    flex: 0 0 auto;
}

.vv-tag-list {
    margin-top: 20px;
}

.vv-tag-row {
    display: grid;
    grid-template-columns: 165px 1fr 62px;
    gap: 14px;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid rgba(229, 222, 210, 0.72);
}

.vv-tag-row:last-child {
    border-bottom: none;
}

.vv-tag-name {
    display: flex;
    align-items: center;
    gap: 10px;
    color: var(--vv-text);
    font-weight: 560;
}

.vv-row-icon {
    width: 26px;
    text-align: center;
}

.vv-progress-track {
    width: 100%;
    height: 9px;
    border-radius: 999px;
    background: #EFEAE3;
    overflow: hidden;
}

.vv-progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--vv-green), var(--vv-teal));
}

.vv-percent {
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: var(--vv-text);
    font-weight: 650;
}

.vv-table-wrap {
    margin-top: 18px;
    overflow-x: auto;
    border: 1px solid var(--vv-border);
    border-radius: 14px;
}

.vv-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}

.vv-table th {
    text-align: left;
    background: #F8F4EC;
    color: var(--vv-muted);
    font-size: 12px;
    padding: 11px 12px;
    border-bottom: 1px solid var(--vv-border);
}

.vv-table td {
    padding: 13px 12px;
    border-bottom: 1px solid rgba(229, 222, 210, 0.72);
    vertical-align: middle;
}

.vv-table tr:last-child td {
    border-bottom: none;
}

.vv-book-icon {
    color: var(--vv-lavender-text);
    margin-right: 8px;
}

.vv-similarity-cell {
    display: flex;
    align-items: center;
    gap: 10px;
    justify-content: flex-end;
    font-variant-numeric: tabular-nums;
}

.vv-mini-track {
    width: 52px;
    height: 6px;
    border-radius: 999px;
    background: #EFEAE3;
    overflow: hidden;
}

.vv-mini-fill {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: var(--vv-green);
}

.vv-insights-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-top: 18px;
}

.vv-mini-card {
    border: 1px solid var(--vv-border);
    border-radius: 16px;
    background: #FFFDF8;
    padding: 16px;
}

.vv-mini-title {
    font-weight: 700;
    margin-bottom: 13px;
    color: var(--vv-text);
}

.vv-tooltip {
    color: var(--vv-muted);
    cursor: help;
    font-size: 13px;
}

.vv-kpi-row {
    display: flex;
    gap: 38px;
    margin-bottom: 12px;
}

.vv-kpi-label {
    display: block;
    color: var(--vv-muted);
    font-size: 12px;
    margin-bottom: 4px;
}

.vv-term-list {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 14px;
}

.vv-term-chip {
    padding: 7px 10px;
    border-radius: 999px;
    background: #F7F5EF;
    border: 1px solid var(--vv-border);
    color: var(--vv-green-dark);
    font-size: 12px;
    font-weight: 620;
}

.vv-model-meta {
    display: grid;
    gap: 8px;
    color: var(--vv-muted);
    font-size: 13px;
}

.vv-muted {
    color: var(--vv-muted);
}

.vv-footer {
    text-align: center;
    color: var(--vv-muted);
    font-size: 12px;
    padding: 18px 0 8px 0;
}

#suggested-tags label {
    border-radius: 999px !important;
    border: 1px solid var(--vv-border) !important;
    background: #F7F5EF !important;
    color: var(--vv-green-dark) !important;
    padding: 8px 12px !important;
}

#suggested-tags label:has(input:checked) {
    background: var(--vv-green-soft) !important;
    border-color: var(--vv-green) !important;
}

.vv-primary-button button,
.vv-primary-button button.primary,
.vv-primary-button .primary,
button.primary {
    background: linear-gradient(90deg, var(--vv-button), var(--vv-button-hover)) !important;
    border: 1px solid rgba(22, 61, 47, 0.18) !important;
    color: #FFFDF8 !important;
    border-radius: 14px !important;
    font-weight: 750 !important;
    box-shadow: 0 12px 26px rgba(49, 95, 74, 0.22) !important;
}

.vv-primary-button button:hover,
.vv-primary-button button.primary:hover,
button.primary:hover {
    background: linear-gradient(90deg, var(--vv-button-hover), #203F31) !important;
    color: #FFFDF8 !important;
}

.vv-accordion {
    border-radius: 18px !important;
    border-color: var(--vv-border) !important;
}

textarea,
input {
    border-radius: 14px !important;
}

@media (max-width: 900px) {
    .vv-hero {
        flex-direction: column;
        align-items: flex-start;
    }

    .vv-brand h1 {
        font-size: 34px;
    }

    .vv-hero-actions {
        flex-wrap: wrap;
    }

    .vv-tag-row {
        grid-template-columns: 1fr;
        gap: 8px;
    }

    .vv-percent {
        text-align: left;
    }

    .vv-insights-grid {
        grid-template-columns: 1fr;
    }

    .vv-similarity-cell {
        justify-content: flex-start;
    }


/* Suggested tag chips */
#suggested-tags {
    --checkbox-label-background-fill: #F7F5EF !important;
    --checkbox-label-background-fill-selected: #EAF1E8 !important;
    --checkbox-label-border-color: #E5DED2 !important;
    --checkbox-label-border-color-selected: #3F6F58 !important;
    --checkbox-label-text-color: #294D3D !important;
    --checkbox-label-text-color-selected: #294D3D !important;
}

#suggested-tags .wrap,
#suggested-tags .wrap-inner {
    gap: 10px !important;
}

#suggested-tags label {
    border-radius: 999px !important;
    border: 1px solid var(--vv-border) !important;
    background: #F7F5EF !important;
    color: var(--vv-green-dark) !important;
    padding: 8px 14px !important;
    box-shadow: 0 6px 14px rgba(52, 45, 35, 0.06) !important;
}

#suggested-tags label span {
    color: var(--vv-green-dark) !important;
}

#suggested-tags label:has(input:checked) {
    background: var(--vv-green-soft) !important;
    border-color: var(--vv-green) !important;
    color: var(--vv-green-dark) !important;
    box-shadow: inset 0 0 0 1px rgba(63, 111, 88, 0.25) !important;
}

#suggested-tags label:has(input:checked) span {
    color: var(--vv-green-dark) !important;
}

#suggested-tags input[type="checkbox"] {
    accent-color: var(--vv-green) !important;
}

#suggested-tags label:has(input:checked)::before {
    content: "✓";
    display: inline-grid;
    place-items: center;
    width: 16px;
    height: 16px;
    margin-right: 7px;
    border-radius: 999px;
    background: var(--vv-green);
    color: #FFFDF8;
    font-size: 10px;
    font-weight: 800;
}

}
"""


def parse_custom_tags(value: str | None) -> list[str]:
    """Parse optional comma-separated custom tags."""
    if not value:
        return []

    return [item.strip() for item in value.split(",") if item.strip()]


def merge_tags(selected_tags: list[str] | None, custom_tags: str | None) -> list[str]:
    """Merge selected chip tags and manually typed custom tags."""
    merged: list[str] = []

    for tag in selected_tags or []:
        if tag not in merged:
            merged.append(tag)

    for tag in parse_custom_tags(custom_tags):
        if tag not in merged:
            merged.append(tag)

    return merged



LOGO_SVG = """
<svg class="vv-logo-svg" viewBox="0 0 72 72" aria-hidden="true">
    <path class="vv-logo-leaf" d="M13 52C20 31 36 16 59 10C55 32 37 49 13 52Z"/>
    <path class="vv-logo-vein" d="M15 51C27 40 40 26 58 11"/>
    <path class="vv-logo-vein" d="M28 39C25 35 24 30 27 25"/>
    <path class="vv-logo-vein" d="M39 28C36 25 36 21 39 17"/>
</svg>
"""


def build_header() -> str:
    """Render the product header."""
    return f"""
    <div class="vv-hero">
        <div class="vv-brand">
            <div class="vv-logo">
                {LOGO_SVG}
            </div>
            <div>
                <h1>{APP_TITLE}</h1>
                <p>{APP_SUBTITLE}</p>
            </div>
        </div>
        <div class="vv-hero-actions">
            {render_api_status("unknown", "Click Check API in the developer section.")}
            <span class="vv-soft-pill" title="Developer guide and advanced controls are below.">API & Developer</span>
            <span class="vv-premium-badge" title="Future product tier placeholder.">Advanced concept</span>
        </div>
    </div>
    """


def run_analysis(
    poem: str,
    title: str,
    poet: str,
    selected_tags: list[str],
    custom_tags: str,
    top_k_tags: int,
    top_n_similar: int,
    tag_threshold_enabled: bool,
    tag_threshold: float,
    user_tag_threshold: float,
) -> tuple[str, str, str, str, str]:
    """Run poem analysis through the API and render UI sections."""
    del user_tag_threshold  # Reserved for future backend support.

    if not poem or not poem.strip():
        error = render_error("Please enter a poem, lyric-like fragment, or reflective text.")
        return error, render_empty_similar_poems(), render_empty_literary_insights(), "{}", render_api_status("unknown")

    threshold = float(tag_threshold) if tag_threshold_enabled else None

    try:
        result = analyze_poem(
            poem=poem,
            title=title,
            poet=poet,
            user_tags=merge_tags(selected_tags, custom_tags),
            top_k_tags=int(top_k_tags),
            top_n_similar=int(top_n_similar),
            tag_threshold=threshold,
        )

        return (
            render_predicted_tags(result),
            render_similar_poems(result),
            render_literary_insights(result),
            render_raw_json(result),
            render_api_status("ready"),
        )

    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        error = render_error(f"API request failed. Detail: {detail}")
        return error, render_empty_similar_poems(), render_empty_literary_insights(), "{}", render_api_status("error", detail)

    except requests.RequestException as exc:
        message = (
            "Could not connect to the VersoVector API. "
            f"Configured API base URL: {get_api_base_url()}. "
            f"Error: {exc}"
        )
        error = render_error(message)
        return error, render_empty_similar_poems(), render_empty_literary_insights(), "{}", render_api_status("error", str(exc))


def run_health_check() -> tuple[str, str]:
    """Run API health and readiness checks."""
    payload: dict[str, Any] = {
        "api_base_url": get_api_base_url(),
    }

    status = "unknown"
    detail = None

    try:
        payload["health"] = health_check()
        status = payload["health"].get("status", "ok")
    except requests.RequestException as exc:
        payload["health"] = {
            "status": "error",
            "detail": str(exc),
        }
        status = "error"
        detail = str(exc)

    try:
        payload["readiness"] = readiness_check()
        status = "ready"
    except requests.RequestException as exc:
        payload["readiness"] = {
            "status": "error",
            "detail": str(exc),
        }
        if status != "ok":
            status = "error"
            detail = str(exc)

    return render_api_status(status, detail), render_raw_json(payload)


def build_app() -> gr.Blocks:
    """Build the Gradio application."""
    with gr.Blocks(
        title=APP_TITLE,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="green",
            secondary_hue="stone",
            neutral_hue="stone",
        ),
        elem_id="vv-app-shell",
    ) as demo:
        gr.HTML(build_header())

        with gr.Row():
            with gr.Column(scale=11, min_width=420):
                with gr.Group(elem_classes=["vv-card"]):
                    gr.HTML(
                        """
                        <div class="vv-card-title">
                            <span class="vv-icon">✨</span>
                            <div>
                                <h3>Analyze the emotion and meaning of your poetry</h3>
                                <p>Paste a poem, lyric-like fragment, or reflective text to uncover its emotional landscape.</p>
                            </div>
                        </div>
                        """
                    )

                    poem_input = gr.Textbox(
                        label="Poem or text fragment",
                        value=DEFAULT_SAMPLE_TEXT,
                        placeholder="Paste a poem, lyric-like fragment, or reflective text here.",
                        lines=10,
                        max_lines=18,
                        info="Avoid submitting copyrighted lyrics unless you have permission.",
                    )

                    with gr.Row():
                        title_input = gr.Textbox(
                            label="Title",
                            placeholder="e.g., Walk in the Rain",
                            info="Optional title used only for display and context.",
                        )
                        poet_input = gr.Textbox(
                            label="Poet",
                            placeholder="e.g., Unknown",
                            info="Optional author name.",
                        )

                    selected_tags_input = gr.CheckboxGroup(
                        label="Suggested tags",
                        choices=tag_choices(),
                        value=DEFAULT_TAGS,
                        info="Select words that describe the mood or themes you expect.",
                        elem_id="suggested-tags",
                    )

                    analyze_button = gr.Button(
                        "❦ Analyze poem",
                        variant="primary",
                        elem_classes=["vv-primary-button"],
                    )

                    with gr.Accordion(
                        "Advanced options",
                        open=False,
                        elem_classes=["vv-accordion"],
                    ):
                        gr.Markdown(
                            "Adjust analysis behavior and thresholds. These controls are optional."
                        )

                        custom_tags_input = gr.Textbox(
                            label="Additional custom tags",
                            placeholder="Optional comma-separated tags, e.g. longing, exile, tenderness",
                        )

                        with gr.Row():
                            top_k_tags_input = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=5,
                                step=1,
                                label="Top K tags",
                                info="Number of emotional-semantic tags shown in the result.",
                            )
                            top_n_similar_input = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=5,
                                step=1,
                                label="Top N similar poems",
                                info="Number of related poems returned from the semantic index.",
                            )

                        with gr.Row():
                            tag_threshold_enabled_input = gr.Checkbox(
                                label="Use tag threshold",
                                value=False,
                                info="Enable a minimum score for model-predicted tags.",
                            )
                            tag_threshold_input = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.25,
                                step=0.05,
                                label="Tag threshold",
                                info="Minimum score required for model-predicted tags.",
                            )

                        user_tag_threshold_input = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.30,
                            step=0.05,
                            label="User tag threshold",
                            info="Reserved for future backend support.",
                        )

            with gr.Column(scale=10, min_width=420):
                predicted_tags_output = gr.HTML(value=render_empty_results())
                similar_poems_output = gr.HTML(value=render_empty_similar_poems())

                with gr.Accordion(
                    "Literary insights",
                    open=False,
                    elem_classes=["vv-accordion"],
                ):
                    literary_insights_output = gr.HTML(value=render_empty_literary_insights())

        with gr.Accordion(
            "API & Developer Guide",
            open=False,
            elem_classes=["vv-accordion"],
        ):
            with gr.Row():
                with gr.Column(scale=1):
                    api_status_output = gr.HTML(
                        value=render_api_status(
                            "unknown",
                            "Click Check API health to validate backend readiness.",
                        )
                    )
                    health_button = gr.Button("Check API health")
                    health_output = gr.Code(
                        label="Health and readiness response",
                        language="json",
                        value=render_raw_json({"api_base_url": get_api_base_url()}),
                    )

                with gr.Column(scale=2):
                    gr.Markdown(
                        f"""
                        ### Developer usage

                        Base URL:

                        ```text
                        {get_api_base_url()}
                        ```

                        Main endpoints:

                        ```text
                        GET  /health
                        GET  /ready
                        GET  /v1/model-info
                        POST /v1/analyze
                        POST /v1/predict-tags
                        POST /v1/similar
                        ```

                        Responsible use:

                        Avoid submitting or redistributing full copyrighted lyrics unless properly licensed.
                        This public demo is intended for poetry, public-domain samples, short fragments, and user-provided text.
                        """
                    )

            raw_output = gr.Code(
                label="Raw model output",
                language="json",
                value="{}",
            )

        gr.HTML(
            """
            <div class="vv-footer">
                VersoVector · Public portfolio implementation · Production datasets,
                private deployment settings, and commercial features are intentionally excluded.
            </div>
            """
        )

        analyze_button.click(
            fn=run_analysis,
            inputs=[
                poem_input,
                title_input,
                poet_input,
                selected_tags_input,
                custom_tags_input,
                top_k_tags_input,
                top_n_similar_input,
                tag_threshold_enabled_input,
                tag_threshold_input,
                user_tag_threshold_input,
            ],
            outputs=[
                predicted_tags_output,
                similar_poems_output,
                literary_insights_output,
                raw_output,
                api_status_output,
            ],
            show_progress=True,
        )

        health_button.click(
            fn=run_health_check,
            inputs=[],
            outputs=[
                api_status_output,
                health_output,
            ],
            show_progress=True,
        )

    return demo


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))

    app = build_app()
    app.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=port,
    )