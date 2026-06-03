# equivalent to notebook 01
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from modules.io import get_nested, load_csv, load_toml_config, project_path, save_csv
from modules.preprocessing import normalize_poetry_columns, preprocess
from utils import Constants, make_short_hash


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build the processed VersoVector dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.toml",
        help="Path to the TOML configuration file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuilding the processed corpus.",
    )
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a project-relative path."""
    path = Path(value)
    return path if path.is_absolute() else project_path(str(path))


def load_raw_poetry_foundation(path: Path) -> pd.DataFrame:
    """Load the raw Poetry Foundation dataset."""
    if not path.is_file():
        raise FileNotFoundError(f"Raw Poetry Foundation dataset not found: {path}")

    return (
        pd.read_csv(
            path,
            sep=Constants.COMMA_STR,
            encoding=Constants.ENCODING,
            usecols=lambda col: "unnamed" not in col.lower(),
        )
        .dropna(subset=["Title", "Poem"])
        .reset_index(drop=True)
    )


def ensure_source_columns(
        df: pd.DataFrame,
        source: str,
        corpus_role: str,
    ) -> pd.DataFrame:
    """Ensure source and corpus role columns exist."""
    df = df.copy()
    df["source"] = source
    df["corpus_role"] = corpus_role
    return df


def ensure_poem_processed(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure poem_processed exists without reprocessing existing values."""
    df = df.copy()

    if "poem_raw" not in df.columns:
        df["poem_raw"] = df["poem"].astype(str)

    if "poem_processed" not in df.columns:
        df["poem_processed"] = ""

    missing_mask = (
        df["poem_processed"].isna()
        | df["poem_processed"].astype(str).str.strip().eq("")
    )

    if missing_mask.any():
        df.loc[missing_mask, "poem_processed"] = (
            df.loc[missing_mask, "poem_raw"]
            .astype(str)
            .apply(preprocess)
        )

    return df


def add_poem_identity(df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Add a stable technical poem identifier."""
    df = df.copy().reset_index(drop=True)

    has_valid_id = (
        "poem_id" in df.columns
        and not df["poem_id"].isna().any()
        and not df["poem_id"].duplicated().any()
    )

    if has_valid_id and not force:
        return df

    required_cols = ["source", "poet_raw", "title_raw", "poem_raw"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Cannot build poem_id. Missing columns: {missing_cols}")

    df["source_row_id"] = df.groupby("source", dropna=False).cumcount()

    identity_text = (
        df["source"].astype(str)
        + "||"
        + df["source_row_id"].astype(str)
        + "||"
        + df["poet_raw"].astype(str)
        + "||"
        + df["title_raw"].astype(str)
        + "||"
        + df["poem_raw"].astype(str)
    )

    df["poem_hash"] = identity_text.apply(make_short_hash)

    df["poem_id"] = (
        df["source"].astype(str)
        + "::"
        + df["source_row_id"].astype(str).str.zfill(6)
        + "::"
        + df["poem_hash"].astype(str)
    )

    if df["poem_id"].duplicated().any():
        duplicates = df.loc[
            df["poem_id"].duplicated(keep=False),
            ["poem_id", "title", "poet", "source", "source_row_id"],
        ]

        raise ValueError(
            "poem_id is not unique. Sample duplicated rows:\n"
            f"{duplicates.head(20)}"
        )

    return df


def validate_processed_dataset(df: pd.DataFrame) -> None:
    """Validate the final processed dataset schema."""
    required_cols = {
        "poem_id",
        "source_row_id",
        "poem_hash",
        "title",
        "title_raw",
        "poet",
        "poet_raw",
        "poem",
        "tags",
        "source",
        "corpus_role",
        "poem_raw",
        "poem_processed",
    }

    missing_cols = required_cols.difference(df.columns)

    if missing_cols:
        raise ValueError(f"Processed dataset is missing columns: {missing_cols}")

    if df["poem_id"].duplicated().any():
        raise ValueError("Processed dataset contains duplicated poem_id values.")

    if df["poem_processed"].isna().any():
        raise ValueError("Processed dataset contains null poem_processed values.")


def get_external_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Read external sources from config.

    If the config does not define external_sources, use César Vallejo as the
    default external corpus if the file exists.
    """
    external_sources = config.get("external_sources", [])

    if external_sources:
        return external_sources

    default_vallejo = project_path("data/vallejo_poems_en.csv")

    if default_vallejo.is_file():
        return [
            {
                "path": "data/vallejo_poems_en.csv",
                "source": "cesar_vallejo",
                "poet": "César Vallejo",
                "corpus_role": "external",
            }
        ]

    return []


def load_external_source(item: dict[str, Any]) -> pd.DataFrame:
    """Load and normalize one external poetry source."""
    path = resolve_project_path(item["path"])

    if not path.is_file():
        raise FileNotFoundError(f"External poetry source not found: {path}")

    df = (
        load_csv(path)
        .dropna(subset=["title", "poem"])
        .reset_index(drop=True)
    )

    df["poet"] = item["poet"]
    df["tags"] = [[] for _ in range(len(df))]

    df = normalize_poetry_columns(df)
    df = ensure_source_columns(
        df=df,
        source=item["source"],
        corpus_role=item.get("corpus_role", "external"),
    )
    df = ensure_poem_processed(df)

    return df


def main() -> None:
    """Build data/poems_processed.csv from raw and external sources."""
    args = parse_args()
    config = load_toml_config(args.config)

    processed_path = resolve_project_path(
        get_nested(config, "data", "processed_corpus", default="data/poems_processed.csv")
    )

    if processed_path.is_file() and not args.force:
        print(f"Processed corpus already exists: {processed_path}")
        print("Use --force to rebuild it.")
        return

    raw_poetry_path = resolve_project_path(
        get_nested(
            config,
            "data",
            "raw_poetry_foundation",
            default="data/PoetryFoundationData.csv",
        )
    )

    cleaned_poetry_path = resolve_project_path(
        get_nested(
            config,
            "data",
            "cleaned_poetry_foundation",
            default="data/CleanedPoetryFoundationData.csv",
        )
    )

    poetry_df = load_raw_poetry_foundation(raw_poetry_path)
    poetry_df = normalize_poetry_columns(poetry_df)
    poetry_df = ensure_source_columns(
        df=poetry_df,
        source="poetry_foundation",
        corpus_role="reference",
    )
    poetry_df = ensure_poem_processed(poetry_df)

    save_csv(poetry_df, cleaned_poetry_path)

    external_frames = [
        load_external_source(item)
        for item in get_external_sources(config)
    ]

    base_columns = [
        "title",
        "title_raw",
        "poet",
        "poet_raw",
        "poem",
        "tags",
        "source",
        "corpus_role",
        "poem_raw",
        "poem_processed",
    ]

    frames = [poetry_df[base_columns]]

    if external_frames:
        frames.extend(frame[base_columns] for frame in external_frames)

    processed_df = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["title", "poem_processed"])
        .reset_index(drop=True)
    )

    processed_df = add_poem_identity(processed_df, force=True)

    final_columns = [
        "poem_id",
        "source_row_id",
        "poem_hash",
        "title",
        "title_raw",
        "poet",
        "poet_raw",
        "poem",
        "tags",
        "source",
        "corpus_role",
        "poem_raw",
        "poem_processed",
    ]

    processed_df = processed_df[final_columns]

    validate_processed_dataset(processed_df)
    save_csv(processed_df, processed_path)

    print("Processed dataset generated.")
    print(f"Rows: {processed_df.shape[0]}")
    print(f"Output: {processed_path}")


if __name__ == "__main__":
    main()
