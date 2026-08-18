"""
correlation_analysis.py

Utilities for studying relationships between
dataset characteristics and pneumonia labels.
"""

import pandas as pd

from src.analysis.metadata_analysis import convert_age_to_years


def encode_metadata(
    metadata_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert metadata into numerical representations
    suitable for correlation analysis.

    M -> 1
    F -> 0
    Unknown -> NaN
    """

    df = metadata_df.copy()

    # Convert DICOM age such as "030Y" -> 30
    df["age_years"] = (
        df["age"]
        .apply(convert_age_to_years)
    )

    # Encode sex
    df["sex_numeric"] = (
        df["sex"]
        .map({
            "M": 1,
            "F": 0
        })
    )

    return df


def aggregate_bounding_boxes(
    bbox_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate bounding-box information
    at patient/image level.
    """

    if bbox_df.empty:

        return pd.DataFrame(
            columns=[
                "patient_id",
                "bbox_count",
                "total_bbox_area",
                "max_bbox_area"
            ]
        )

    grouped = (
        bbox_df
        .groupby("patient_id")
        .agg(
            bbox_count=("area", "count"),
            total_bbox_area=("area", "sum"),
            max_bbox_area=("area", "max")
        )
        .reset_index()
    )

    return grouped


def build_correlation_dataset(
    labels: dict,
    metadata_df: pd.DataFrame,
    pixel_df: pd.DataFrame,
    bbox_df: pd.DataFrame,
    image_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine dataset-level characteristics into a single
    dataframe suitable for correlation analysis.
    """

    # --------------------------------------------------
    # LABELS
    # --------------------------------------------------

    label_records = []

    for patient_id, info in labels.items():

        label_records.append({
            "patient_id": patient_id,
            "target": info["target"]
        })

    labels_df = pd.DataFrame(
        label_records
    )

    # --------------------------------------------------
    # METADATA
    # --------------------------------------------------

    metadata = encode_metadata(
        metadata_df
    )

    metadata = metadata[
        [
            "patient_id",
            "age_years",
            "sex_numeric"
        ]
    ]

    # --------------------------------------------------
    # PIXEL STATISTICS
    # --------------------------------------------------

    pixel = pixel_df.copy()

    pixel = pixel[
        [
            "patient_id",
            "mean_intensity"
        ]
    ]

    # --------------------------------------------------
    # BOUNDING BOXES
    # --------------------------------------------------

    bbox = aggregate_bounding_boxes(
        bbox_df
    )

    # --------------------------------------------------
    # IMAGE STATISTICS
    # --------------------------------------------------

    image = image_df.copy()

    image = image[
        [
            "patient_id",
            "width",
            "height"
        ]
    ]

    # --------------------------------------------------
    # MERGE EVERYTHING
    # --------------------------------------------------

    correlation_df = labels_df.merge(
        metadata,
        on="patient_id",
        how="left"
    )

    correlation_df = correlation_df.merge(
        pixel,
        on="patient_id",
        how="left"
    )

    correlation_df = correlation_df.merge(
        bbox,
        on="patient_id",
        how="left"
    )

    correlation_df = correlation_df.merge(
        image,
        on="patient_id",
        how="left"
    )

    # --------------------------------------------------
    # HANDLE IMAGES WITHOUT BOUNDING BOXES
    # --------------------------------------------------

    correlation_df[
        [
            "bbox_count",
            "total_bbox_area",
            "max_bbox_area"
        ]
    ] = correlation_df[
        [
            "bbox_count",
            "total_bbox_area",
            "max_bbox_area"
        ]
    ].fillna(0)

    return correlation_df


def calculate_correlation_matrix(
    correlation_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate Spearman correlations between
    numerical variables.
    """

    columns = [
        "target",
        "age_years",
        "sex_numeric",
        "mean_intensity",
        "bbox_count",
        "total_bbox_area",
        "max_bbox_area",
        "width",
        "height"
    ]

    available_columns = [
        column
        for column in columns
        if column in correlation_df.columns
    ]

    return correlation_df[
        available_columns
    ].corr(
        method="spearman"
    )