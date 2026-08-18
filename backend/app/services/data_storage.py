from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.services.supabase_client import get_supabase_client


DATASET_BUCKET = os.getenv(
    "SUPABASE_DATASET_BUCKET",
    "analysis-datasets",
)

DATASET_FILENAME = "cleaned_dataset.parquet"


class DatasetStorageError(Exception):
    """Raised when a cleaned dataset cannot be stored or retrieved."""


class DatasetNotAvailableError(Exception):
    """Raised when an analysis has no stored predictive dataset."""


def build_dataset_storage_path(
    user_id: str,
    analysis_id: str,
) -> str:
    return (
        f"{user_id}/"
        f"{analysis_id}/"
        f"{DATASET_FILENAME}"
    )


def dataframe_to_parquet_bytes(
    dataframe: pd.DataFrame,
) -> bytes:
    if dataframe is None or dataframe.empty:
        raise DatasetStorageError(
            "An empty dataset cannot be stored."
        )

    buffer = io.BytesIO()

    try:
        dataframe.to_parquet(
            buffer,
            index=False,
            engine="pyarrow",
        )

    except Exception as error:
        raise DatasetStorageError(
            f"Unable to convert the dataset to Parquet: {error}"
        ) from error

    return buffer.getvalue()


def parquet_bytes_to_dataframe(
    content: bytes,
) -> pd.DataFrame:
    if not content:
        raise DatasetStorageError(
            "The stored dataset file is empty."
        )

    buffer = io.BytesIO(content)

    try:
        dataframe = pd.read_parquet(
            buffer,
            engine="pyarrow",
        )

    except Exception as error:
        raise DatasetStorageError(
            f"Unable to read the stored dataset: {error}"
        ) from error

    if dataframe.empty:
        raise DatasetStorageError(
            "The stored dataset contains no rows."
        )

    return dataframe


def store_cleaned_dataset(
    dataframe: pd.DataFrame,
    user_id: str,
    analysis_id: str,
) -> dict[str, Any]:
    """
    Save a cleaned dataframe to the private Supabase Storage bucket.
    """

    supabase = get_supabase_client()

    storage_path = build_dataset_storage_path(
        user_id=user_id,
        analysis_id=analysis_id,
    )

    parquet_content = dataframe_to_parquet_bytes(
        dataframe
    )

    try:
        (
            supabase.storage
            .from_(DATASET_BUCKET)
            .upload(
                path=storage_path,
                file=parquet_content,
                file_options={
                    "content-type": "application/octet-stream",
                    "upsert": "true",
                },
            )
        )

    except Exception as error:
        raise DatasetStorageError(
            f"Unable to store the cleaned dataset: {error}"
        ) from error

    return {
        "dataset_storage_path": storage_path,
        "dataset_format": "parquet",
        "dataset_size_bytes": len(parquet_content),
        "dataset_row_count": int(len(dataframe)),
        "dataset_column_count": int(len(dataframe.columns)),
        "dataset_stored_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset_expires_at": None,
        "prediction_ready": True,
    }


def download_stored_dataset(
    storage_path: str,
) -> pd.DataFrame:
    """
    Download and reconstruct a cleaned dataframe from private storage.
    """

    if not storage_path:
        raise DatasetNotAvailableError(
            "The original dataset is not available for predictive modelling."
        )

    supabase = get_supabase_client()

    try:
        content = (
            supabase.storage
            .from_(DATASET_BUCKET)
            .download(storage_path)
        )

    except Exception as error:
        raise DatasetStorageError(
            f"Unable to download the stored dataset: {error}"
        ) from error

    return parquet_bytes_to_dataframe(content)


def get_analysis_dataset(
    analysis_id: str,
    user_id: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Verify ownership of an analysis and retrieve its stored dataframe.
    """

    supabase = get_supabase_client()

    response = (
        supabase.table("analyses")
        .select(
            "id, user_id, filename, dataset_storage_path, "
            "dataset_format, dataset_size_bytes, "
            "dataset_row_count, dataset_column_count, "
            "dataset_stored_at, dataset_expires_at, "
            "prediction_ready"
        )
        .eq("id", analysis_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )

    if not response.data:
        raise ValueError(
            "Analysis not found or access denied."
        )

    analysis_record = response.data[0]

    if not analysis_record.get("prediction_ready"):
        raise DatasetNotAvailableError(
            "This analysis does not have a stored dataset available "
            "for predictive modelling. Re-upload the dataset to create "
            "a prediction-ready analysis."
        )

    storage_path = analysis_record.get(
        "dataset_storage_path"
    )

    if not storage_path:
        raise DatasetNotAvailableError(
            "This analysis does not contain a stored dataset reference."
        )

    dataframe = download_stored_dataset(
        storage_path=storage_path
    )

    return analysis_record, dataframe


def delete_stored_dataset(
    storage_path: str,
) -> None:
    """
    Delete a stored dataset from the private bucket.
    """

    if not storage_path:
        return

    supabase = get_supabase_client()

    try:
        (
            supabase.storage
            .from_(DATASET_BUCKET)
            .remove([storage_path])
        )

    except Exception as error:
        raise DatasetStorageError(
            f"Unable to delete the stored dataset: {error}"
        ) from error