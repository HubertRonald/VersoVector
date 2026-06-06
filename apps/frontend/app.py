from __future__ import annotations

import os
from typing import Any

import gradio as gr
import requests

from client import analyze_poem, get_api_base_url, health_check, readiness_check
from tag_catalog import DEFAULT_TAGS, svg_icon, tag_choices
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


APP_TITLE = "VersoVector"
APP_SUBTITLE = "Emotional-semantic analysis for poetic and lyrical language."
DEFAULT_SAMPLE_TEXT = "I walk through the rain carrying a memory of light."


LOGO_SVG = """
<svg class="vv-logo-svg" viewBox="0 0 72 72" aria-hidden="true">
    <path d="M13 51C22 31 38 17 59 10C54 31 38 47 13 51Z"/>
    <path d="M15 50C29 38 42 25 58 11"/>
    <path d="M25 40C24 34 26 29 31 24"/>
    <path d="M37 29C36 24 39 20 44 16"/>
</svg>
"""


GEAR_SVG = """
<svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M12 8.5A3.5 3.5 0 1 1 12 15.5A3.5 3.5 0 0 1 12 8.5Z"/>
    <path d="M19 12A7.2 7.2 0 0 0 18.9 10.8L21 9.2L19 5.8L16.5 6.8A7 7 0 0 0 14.5 5.6L14.2 3H9.8L9.5 5.6A7 7 0 0 0 7.5 6.8L5 5.8L3 9.2L5.1 10.8A7.2 7.2 0 0 0 5 12A7.2 7.2 0 0 0 5.1 13.2L3 14.8L5 18.2L7.5 17.2A7 7 0 0 0 9.5 18.4L9.8 21H14.2L14.5 18.4A7 7 0 0 0 16.5 17.2L19 18.2L21 14.8L18.9 13.2A7.2 7.2 0 0 0 19 12Z"/>
</svg>
"""


CUSTOM_CSS = """
:root {
    --vv-bg: #F4F0E8;
    --vv-card: #FBF8F2;
    --vv-card-soft: #F7F3EC;
    --vv-surface: #FBF8F2;
    --vv-surface-2: #F7F3EC;

    --vv-text: #252A27;
    --vv-muted: #5F625D;
    --vv-border: #DDD5C7;

    --vv-green: #315F4A;
    --vv-green-dark: #163D2F;
    --vv-green-soft: #EEF5EF;
    --vv-green-soft-2: #F4F8F3;
    --vv-green-muted: #6F8F7A;
    --vv-teal: #527C70;

    --vv-button: #315F4A;
    --vv-button-hover: #284E3D;

    --vv-chip-bg: #F4F1EA;
    --vv-chip-border: #DDD5C7;
    --vv-chip-text: #435247;

    --vv-accordion-bg: #F7F3EC;
    --vv-accordion-bg-hover: #F2EEE7;
    --vv-accordion-border: #DDD5C7;

    --vv-label-bg: #EEF5EF;
    --vv-label-text: #315F4A;
    --vv-label-border: #CFE0D4;

    --vv-premium-bg: #FBF2D8;
    --vv-premium-border: #E2C878;
    --vv-premium-text: #6A531F;

    --vv-warning: #FFF5D8;
    --vv-shadow: 0 14px 40px rgba(52, 45, 35, 0.08);
    --vv-soft-shadow: 0 10px 28px rgba(61, 51, 37, 0.06);
}

#vv-app-shell,
.gradio-container {
    --button-primary-background-fill: #315F4A !important;
    --button-primary-background-fill-hover: #284E3D !important;
    --button-primary-text-color: #FFFDF8 !important;

    --checkbox-label-background-fill: #F7F5EF !important;
    --checkbox-label-background-fill-selected: #EEF5EF !important;
    --checkbox-label-border-color: #E5DED2 !important;
    --checkbox-label-border-color-selected: #315F4A !important;
    --checkbox-label-text-color: #294D3D !important;
    --checkbox-label-text-color-selected: #163D2F !important;

    --block-label-background-fill: #EEF5EF !important;
    --block-label-text-color: #315F4A !important;
    --block-label-border-color: #CFE0D4 !important;

    --slider-color: #315F4A !important;
}

.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(234, 241, 232, 0.90) 0%, transparent 28%),
        linear-gradient(135deg, #F4F0E8 0%, #F7F3EC 48%, #F2E8DA 100%) !important;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    color: var(--vv-text);
}

#vv-app-shell {
    max-width: 1360px;
    margin: 0 auto;
}

/* ---------------------------------------------------------
   Header
   --------------------------------------------------------- */

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
        linear-gradient(90deg, rgba(251, 248, 242, 0.98), rgba(251, 248, 242, 0.88)),
        radial-gradient(circle at 90% 30%, rgba(95, 141, 131, 0.14), transparent 32%);
    box-shadow: var(--vv-shadow);
}

.vv-brand {
    display: flex;
    align-items: center;
    gap: 18px;
}

.vv-logo {
    width: 54px !important;
    height: 54px !important;
    border-radius: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    display: grid !important;
    place-items: center !important;
}

.vv-logo-svg {
    width: 48px !important;
    height: 48px !important;
}

.vv-logo-svg path {
    fill: none !important;
    stroke: var(--vv-green) !important;
    stroke-width: 3.1 !important;
    stroke-linecap: round !important;
    stroke-linejoin: round !important;
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
    background: rgba(251, 248, 242, 0.88);
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
    background: var(--vv-premium-bg) !important;
    color: var(--vv-premium-text) !important;
    border-color: var(--vv-premium-border) !important;
}

/* ---------------------------------------------------------
   Layout
   --------------------------------------------------------- */

.vv-main-grid {
    gap: 22px !important;
    align-items: flex-start !important;
}

.vv-left-column,
.vv-right-column {
    background: transparent !important;
}

.vv-right-column {
    gap: 18px !important;
}

.vv-bottom-row {
    margin-top: 18px !important;
}

.vv-bottom-column {
    background: transparent !important;
}

/* ---------------------------------------------------------
   Cards
   --------------------------------------------------------- */

.vv-card {
    border: 1px solid var(--vv-border) !important;
    border-radius: 24px !important;
    background: var(--vv-card) !important;
    box-shadow: var(--vv-soft-shadow) !important;
    padding: 22px !important;
    margin-bottom: 14px !important;
}

.vv-form-card {
    background: var(--vv-card) !important;
}

.vv-empty-card {
    min-height: 170px;
    display: flex;
    align-items: center;
}

.vv-error-card {
    border-color: #F0C4BE !important;
    background: #FFF5F2 !important;
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

/* Remove harsh gray/white strips inside Gradio blocks */
.vv-form-card .gr-block,
.vv-form-card .gr-box,
.vv-form-card .gr-group,
.vv-form-card > div {
    background-color: transparent !important;
}

/* ---------------------------------------------------------
   Inline SVG icons
   --------------------------------------------------------- */

.vv-svg-icon,
.vv-row-icon,
.vv-book-icon,
.vv-gear-icon {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: var(--vv-green) !important;
    flex: 0 0 auto !important;
}

.vv-svg-icon {
    width: 18px !important;
    height: 18px !important;
}

.vv-row-icon {
    width: 24px !important;
    height: 24px !important;
}

.vv-book-icon {
    width: 20px !important;
    height: 20px !important;
    margin-right: 8px !important;
    color: var(--vv-teal) !important;
    vertical-align: middle !important;
}

.vv-gear-icon {
    width: 32px !important;
    height: 32px !important;
    display: inline-grid !important;
    place-items: center !important;
    border-radius: 10px !important;
    background: #EAF1EB !important;
    color: var(--vv-green) !important;
    flex: 0 0 auto !important;
}

.vv-svg-icon svg,
.vv-row-icon svg,
.vv-book-icon svg,
.vv-gear-icon svg {
    width: 18px !important;
    height: 18px !important;
    display: block !important;
}

.vv-gear-icon svg {
    width: 18px !important;
    height: 18px !important;
}

.vv-svg-icon svg path,
.vv-svg-icon svg circle,
.vv-row-icon svg path,
.vv-row-icon svg circle,
.vv-book-icon svg path,
.vv-book-icon svg circle,
.vv-gear-icon svg path,
.vv-gear-icon svg circle {
    fill: none !important;
    stroke: currentColor !important;
    stroke-width: 2.1 !important;
    stroke-linecap: round !important;
    stroke-linejoin: round !important;
}

/* ---------------------------------------------------------
   Tags / chips
   --------------------------------------------------------- */

#suggested-tags {
    --checkbox-label-background-fill: #F7F5EF !important;
    --checkbox-label-background-fill-selected: #EEF5EF !important;
    --checkbox-label-border-color: #E5DED2 !important;
    --checkbox-label-border-color-selected: #315F4A !important;
    --checkbox-label-text-color: #294D3D !important;
    --checkbox-label-text-color-selected: #163D2F !important;
}

#suggested-tags .wrap,
#suggested-tags .wrap-inner {
    gap: 10px !important;
}

#suggested-tags label {
    position: relative !important;
    border-radius: 999px !important;
    border: 1px solid var(--vv-chip-border) !important;
    background: var(--vv-chip-bg) !important;
    color: var(--vv-green-dark) !important;
    padding: 8px 14px !important;
    box-shadow: 0 6px 14px rgba(52, 45, 35, 0.06) !important;
}

#suggested-tags label span {
    color: var(--vv-green-dark) !important;
}

#suggested-tags input[type="checkbox"] {
    opacity: 0 !important;
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
    pointer-events: none !important;
}

#suggested-tags label::before {
    content: "";
    display: inline-grid;
    place-items: center;
    width: 15px;
    height: 15px;
    margin-right: 8px;
    border-radius: 999px;
    background: #FFFDF8;
    border: 1px solid var(--vv-border);
    color: #FFFDF8;
    font-size: 10px;
    font-weight: 800;
}

#suggested-tags label:has(input:checked) {
    background: var(--vv-green-soft) !important;
    border-color: var(--vv-green) !important;
    color: var(--vv-green-dark) !important;
    box-shadow:
        inset 0 0 0 1px rgba(49, 95, 74, 0.24),
        0 6px 14px rgba(52, 45, 35, 0.06) !important;
}

#suggested-tags label:has(input:checked)::before {
    content: "✓";
    background: var(--vv-green);
    border-color: var(--vv-green);
    color: #FFFDF8;
}

/* Softer Gradio block labels */
.gradio-container span[data-testid="block-label"],
.gradio-container .block-label {
    background: var(--vv-label-bg) !important;
    color: var(--vv-label-text) !important;
    border-color: var(--vv-label-border) !important;
}

/* Labels inside advanced controls should be calmer */
.vv-advanced-accordion .block-label,
.vv-advanced-accordion span[data-testid="block-label"] {
    background: #EFE9E1 !important;
    border: 1px solid #DDD5C7 !important;
    color: #6A6A63 !important;
}

/* ---------------------------------------------------------
   Predicted tags
   --------------------------------------------------------- */

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
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    color: var(--vv-text);
    font-weight: 560;
}

.vv-tag-name .vv-row-icon {
    color: var(--vv-green) !important;
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

/* ---------------------------------------------------------
   Similar poems table
   --------------------------------------------------------- */

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
    border-color: var(--vv-border) !important;
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

.vv-table th,
.vv-table td {
    border-color: rgba(229, 222, 210, 0.85) !important;
}

.vv-table tr:last-child td {
    border-bottom: none;
}

.vv-table td:first-child {
    display: table-cell !important;
    vertical-align: middle !important;
}

.vv-table td:first-child .vv-book-icon {
    position: relative !important;
    top: 3px !important;
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

/* ---------------------------------------------------------
   Literary insights
   --------------------------------------------------------- */

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

/* ---------------------------------------------------------
   Unified accordions
   --------------------------------------------------------- */

.vv-panel-accordion {
    margin-top: 10px !important;
    border: 1px solid var(--vv-accordion-border) !important;
    border-radius: 18px !important;
    background: var(--vv-card) !important;
    overflow: hidden !important;
    box-shadow: 0 8px 22px rgba(61, 51, 37, 0.05) !important;
}

.vv-panel-accordion > div:first-child,
.vv-panel-accordion summary,
.vv-panel-accordion button {
    background: var(--vv-accordion-bg) !important;
    color: var(--vv-green-dark) !important;
    border: none !important;
    box-shadow: none !important;
}

.vv-panel-accordion summary,
.vv-panel-accordion button {
    min-height: 52px !important;
    padding: 0 18px !important;
    font-weight: 650 !important;
    font-size: 15px !important;
    border-radius: 16px !important;
}

.vv-panel-accordion summary:hover,
.vv-panel-accordion button:hover {
    background: var(--vv-accordion-bg-hover) !important;
}

.vv-panel-accordion > div:last-child {
    background: var(--vv-card) !important;
    border-top: 1px solid rgba(221, 213, 199, 0.9) !important;
    padding: 14px 16px 18px 16px !important;
}

.vv-panel-accordion summary *,
.vv-panel-accordion button * {
    color: var(--vv-green-dark) !important;
}

.vv-panel-accordion svg {
    color: var(--vv-green-dark) !important;
}

.vv-panel-accordion > div:first-child {
    border-bottom: 1px solid rgba(221, 213, 199, 0.65) !important;
}

.vv-advanced-accordion {
    margin-top: 18px !important;
}

.vv-insights-accordion,
.vv-api-accordion {
    width: 100% !important;
    max-width: 100% !important;
}

.vv-insights-accordion {
    margin-top: 6px !important;
}

.vv-api-accordion {
    margin-top: 0 !important;
}

.vv-insights-accordion > div:last-child,
.vv-api-accordion > div:last-child,
.vv-advanced-accordion > div:last-child {
    background: var(--vv-card) !important;
}

/* Advanced intro: no duplicated title */
.vv-advanced-intro {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 2px 6px 16px 6px;
    padding: 14px 16px;
    border: 1px solid var(--vv-border);
    border-radius: 18px;
    background: var(--vv-green-soft-2);
}

.vv-advanced-intro p {
    margin: 0 !important;
    font-size: 14px !important;
    color: var(--vv-muted) !important;
}

/* ---------------------------------------------------------
   Primary button
   --------------------------------------------------------- */

.vv-primary-button {
    margin: 20px 0 18px 0 !important;
}

.vv-primary-button button,
.vv-primary-button button.primary,
.vv-primary-button .primary,
button.primary {
    min-height: 54px !important;
    background: linear-gradient(90deg, var(--vv-button), var(--vv-button-hover)) !important;
    border: 1px solid rgba(22, 61, 47, 0.18) !important;
    border-radius: 16px !important;
    color: #FFFDF8 !important;
    font-weight: 750 !important;
    box-shadow: 0 12px 24px rgba(49, 95, 74, 0.18) !important;
}

.vv-primary-button button:hover,
.vv-primary-button button.primary:hover,
button.primary:hover {
    background: linear-gradient(90deg, var(--vv-button-hover), #203F31) !important;
}

/* Sliders */
input[type="range"] {
    accent-color: var(--vv-green) !important;
}

textarea,
input {
    border-radius: 14px !important;
}

/* ---------------------------------------------------------
   Footer
   --------------------------------------------------------- */

.vv-muted {
    color: var(--vv-muted);
}

.vv-footer {
    text-align: center;
    color: var(--vv-muted);
    font-size: 12px;
    padding: 18px 0 8px 0;
}

/* ---------------------------------------------------------
   Responsive
   --------------------------------------------------------- */

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

    .vv-main-grid {
        gap: 16px !important;
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
            <span class="vv-premium-badge" title="Future advanced product layer.">Advanced concept</span>
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
        return (
            error,
            render_empty_similar_poems(),
            render_empty_literary_insights(),
            "{}",
            render_api_status("unknown"),
        )

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
        return (
            error,
            render_empty_similar_poems(),
            render_empty_literary_insights(),
            "{}",
            render_api_status("error", detail),
        )

    except requests.RequestException as exc:
        message = (
            "Could not connect to the VersoVector API. "
            f"Configured API base URL: {get_api_base_url()}. "
            f"Error: {exc}"
        )
        error = render_error(message)
        return (
            error,
            render_empty_similar_poems(),
            render_empty_literary_insights(),
            "{}",
            render_api_status("error", str(exc)),
        )


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
            primary_hue="stone",
            secondary_hue="stone",
            neutral_hue="stone",
        ),
        elem_id="vv-app-shell",
    ) as demo:
        gr.HTML(build_header())

        with gr.Row(elem_classes=["vv-main-grid"]):
            with gr.Column(
                scale=11,
                min_width=420,
                elem_classes=["vv-left-column"],
            ):
                with gr.Group(elem_classes=["vv-card", "vv-form-card"]):
                    gr.HTML(
                        f"""
                        <div class="vv-card-title">
                            <span class="vv-icon">{svg_icon("spark")}</span>
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
                        "Analyze poem",
                        variant="primary",
                        elem_classes=["vv-primary-button"],
                    )

                    with gr.Accordion(
                        "Advanced options",
                        open=False,
                        elem_classes=[
                            "vv-panel-accordion",
                            "vv-advanced-accordion",
                        ],
                    ):
                        gr.HTML(
                            f"""
                            <div class="vv-advanced-intro">
                                <span class="vv-gear-icon">{GEAR_SVG}</span>
                                <p>Adjust analysis behavior and thresholds. These controls are optional.</p>
                            </div>
                            """
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

            with gr.Column(
                scale=10,
                min_width=420,
                elem_classes=["vv-right-column"],
            ):
                predicted_tags_output = gr.HTML(value=render_empty_results())
                similar_poems_output = gr.HTML(value=render_empty_similar_poems())

                with gr.Accordion(
                    "Literary insights",
                    open=False,
                    elem_classes=[
                        "vv-panel-accordion",
                        "vv-insights-accordion",
                    ],
                ):
                    literary_insights_output = gr.HTML(
                        value=render_empty_literary_insights()
                    )

        with gr.Row(elem_classes=["vv-bottom-row"]):
            with gr.Column(elem_classes=["vv-bottom-column"]):
                with gr.Accordion(
                    "API & Developer Guide",
                    open=False,
                    elem_classes=[
                        "vv-panel-accordion",
                        "vv-api-accordion",
                    ],
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
                                value=render_raw_json(
                                    {"api_base_url": get_api_base_url()}
                                ),
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
