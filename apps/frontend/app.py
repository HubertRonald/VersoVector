from __future__ import annotations

import json
import os
from typing import Any

import gradio as gr
import pandas as pd
import requests

from client import analyze_poem, get_api_base_url, health_check


APP_TITLE = "VersoVector"
APP_SUBTITLE = "Emotional-semantic analysis for poetic and lyrical language."


CUSTOM_CSS = """
:root {
    --vv-bg: #f7f1e8;
    --vv-card: #fffaf2;
    --vv-text: #2c2621;
    --vv-muted: #6f6258;
    --vv-accent: #9a6b3f;
}

.gradio-container {
    background: radial-gradient(circle at top left, #fff8ef 0%, #f7f1e8 45%, #efe3d3 100%);
}

#vv-header {
    padding: 1.2rem 1.4rem;
    border-radius: 18px;
    background: rgba(255, 250, 242, 0.86);
    border: 1px solid rgba(154, 107, 63, 0.18);
    box-shadow: 0 10px 30px rgba(60, 45, 30, 0.08);
}

#vv-header h1 {
    color: var(--vv-text);
    margin-bottom: 0.25rem;
}

#vv-header p {
    color: var(--vv-muted);
    font-size: 1rem;
}

#vv-note {
    color: var(--vv-muted);
    font-size: 0.92rem;
}
"""


def parse_user_tags(value: str) -> list[str]:
    """Parse comma-separated user tags."""
    if not value.strip():
        return []

    return [item.strip() for item in value.split(",") if item.strip()]


def format_tags(result: dict[str, Any]) -> pd.DataFrame:
    """Format predicted tags into a table."""
    tags = result.get("predicted_tags", [])

    rows = [
        {
            "tag": item.get("tag"),
            "score": item.get("score"),
        }
        for item in tags
    ]

    return pd.DataFrame(rows, columns=["tag", "score"])


def format_similar_poems(result: dict[str, Any]) -> pd.DataFrame:
    """Format similar poems into a table."""
    similar_poems = result.get("similar_poems", [])

    rows = [
        {
            "title": item.get("title"),
            "poet": item.get("poet") or item.get("poet_raw"),
            "source": item.get("source"),
            "score": item.get("score"),
        }
        for item in similar_poems
    ]

    return pd.DataFrame(rows, columns=["title", "poet", "source", "score"])


def format_summary(result: dict[str, Any]) -> str:
    """Build a compact markdown summary."""
    topic = result.get("topic") or {}
    cluster = result.get("cluster") or {}
    model_info = result.get("model_info") or {}

    topic_terms = topic.get("terms") or "not available"
    topic_id = topic.get("topic_id")
    topic_probability = topic.get("probability")

    kmeans_cluster = cluster.get("kmeans")
    gmm_cluster = cluster.get("gmm")

    return f"""
### Analysis Summary

**Processed text sample**

`{result.get("poem_processed", "")[:240]}`

**Topic**

- Topic ID: `{topic_id}`
- Probability: `{topic_probability}`
- Terms: `{topic_terms}`

**Clusters**

- KMeans: `{kmeans_cluster}`
- GMM: `{gmm_cluster}`

**Model**

- Project: `{model_info.get("project", "versovector")}`
- Task: `{model_info.get("task", "emotional_semantic_recommendation")}`
"""


def run_analysis(
    poem: str,
    title: str,
    poet: str,
    user_tags: str,
    top_k_tags: int,
    top_n_similar: int,
    tag_threshold_enabled: bool,
    tag_threshold: float,
) -> tuple[str, pd.DataFrame, pd.DataFrame, str]:
    """Run poem analysis through the API."""
    if not poem.strip():
        empty = pd.DataFrame()
        return "Please enter a poem or text fragment.", empty, empty, "{}"

    threshold = tag_threshold if tag_threshold_enabled else None

    try:
        result = analyze_poem(
            poem=poem,
            title=title,
            poet=poet,
            user_tags=parse_user_tags(user_tags),
            top_k_tags=int(top_k_tags),
            top_n_similar=int(top_n_similar),
            tag_threshold=threshold,
        )

        return (
            format_summary(result),
            format_tags(result),
            format_similar_poems(result),
            json.dumps(result, ensure_ascii=False, indent=2),
        )

    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        empty = pd.DataFrame()
        return f"API request failed:\n\n```text\n{detail}\n```", empty, empty, "{}"

    except requests.RequestException as exc:
        empty = pd.DataFrame()
        return (
            "Could not connect to the VersoVector API.\n\n"
            f"Configured API base URL: `{get_api_base_url()}`\n\n"
            f"Error: `{exc}`",
            empty,
            empty,
            "{}",
        )


def run_health_check() -> str:
    """Run API health check."""
    try:
        result = health_check()
        return json.dumps(result, ensure_ascii=False, indent=2)
    except requests.RequestException as exc:
        return json.dumps(
            {
                "status": "error",
                "api_base_url": get_api_base_url(),
                "detail": str(exc),
            },
            indent=2,
        )


def build_app() -> gr.Blocks:
    """Build the Gradio application."""
    with gr.Blocks(
        title=APP_TITLE,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
    ) as demo:
        gr.HTML(
            f"""
            <div id="vv-header">
                <h1>{APP_TITLE}</h1>
                <p>{APP_SUBTITLE}</p>
                <p id="vv-note">
                    Paste a poem, lyric-like fragment, or reflective text.
                    VersoVector will call the FastAPI backend and return emotional tags,
                    similar poems, topic information, and cluster assignments.
                </p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                poem_input = gr.Textbox(
                    label="Poem or text fragment",
                    placeholder="I walk through the rain carrying a memory of light...",
                    lines=12,
                )
                with gr.Row():
                    title_input = gr.Textbox(label="Title", placeholder="Optional")
                    poet_input = gr.Textbox(label="Poet", placeholder="Optional")

                user_tags_input = gr.Textbox(
                    label="User tags",
                    placeholder="Optional comma-separated tags, e.g. memory, longing, hope",
                )

                with gr.Row():
                    top_k_tags_input = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Top K tags",
                    )
                    top_n_similar_input = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Top N similar poems",
                    )

                with gr.Row():
                    threshold_enabled_input = gr.Checkbox(
                        label="Use tag threshold",
                        value=False,
                    )
                    threshold_input = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="Tag threshold",
                    )

                analyze_button = gr.Button(
                    "Analyze text",
                    variant="primary",
                )

            with gr.Column(scale=2):
                health_button = gr.Button("Check API health")
                health_output = gr.Code(
                    label="API health",
                    language="json",
                    value=f'{{"api_base_url": "{get_api_base_url()}"}}',
                )

        summary_output = gr.Markdown(label="Summary")

        with gr.Row():
            tags_output = gr.Dataframe(
                label="Predicted tags",
                headers=["tag", "score"],
                interactive=False,
            )
            similar_output = gr.Dataframe(
                label="Similar poems",
                headers=["title", "poet", "source", "score"],
                interactive=False,
            )

        raw_output = gr.Code(
            label="Raw API response",
            language="json",
        )

        analyze_button.click(
            fn=run_analysis,
            inputs=[
                poem_input,
                title_input,
                poet_input,
                user_tags_input,
                top_k_tags_input,
                top_n_similar_input,
                threshold_enabled_input,
                threshold_input,
            ],
            outputs=[
                summary_output,
                tags_output,
                similar_output,
                raw_output,
            ],
        )

        health_button.click(
            fn=run_health_check,
            inputs=[],
            outputs=[health_output],
        )

    return demo


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
