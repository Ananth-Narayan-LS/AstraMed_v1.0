"""
Clinical (non-image) data for the Sprint 8 multimodal research model.

Only fields that genuinely exist are used. For RSNA stage-2 these are the DICOM
header fields PatientAge, PatientSex and ViewPosition. The dataset has NO symptom
or clinical-history text; those fields are always empty strings here and their
modality masks are always 0. Nothing is ever imputed from the target label.

Everything that is "learned" from data (age mean/std, category lists) is fitted on
the TRAINING patients only (ClinicalFeatureEncoder.fit) and then reused unchanged.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

SEX_CATEGORIES = ("M", "F")
VIEW_CATEGORIES = ("AP", "PA")
MAX_PLAUSIBLE_AGE = 120

TABLE_COLUMNS = ["patient_id", "age", "sex", "view_position"]


def parse_age(raw) -> float:
    """DICOM age strings look like '058Y'. Anything unparseable/implausible -> NaN."""
    if raw is None:
        return math.nan
    text = str(raw).strip().upper()
    if not text:
        return math.nan
    unit = text[-1] if text[-1] in "DWMY" else "Y"
    digits = text[:-1] if text[-1] in "DWMY" else text
    if not digits.isdigit():
        return math.nan
    value = float(int(digits))
    years = {"Y": value, "M": value / 12, "W": value / 52, "D": value / 365}[unit]
    return years if 0 <= years <= MAX_PLAUSIBLE_AGE else math.nan


def parse_category(raw, allowed) -> str | None:
    text = "" if raw is None else str(raw).strip().upper()
    return text if text in allowed else None


def _read_one(args):
    patient_id, path = args
    header = pydicom.dcmread(path, stop_before_pixels=True)
    dicom_patient_id = str(getattr(header, "PatientID", ""))
    if dicom_patient_id != patient_id:
        raise ValueError(
            f"DICOM PatientID {dicom_patient_id!r} does not match file id {patient_id!r}."
        )
    return {
        "patient_id": patient_id,
        "age": parse_age(getattr(header, "PatientAge", None)),
        "sex": parse_category(getattr(header, "PatientSex", None), SEX_CATEGORIES),
        "view_position": parse_category(getattr(header, "ViewPosition", None), VIEW_CATEGORIES),
    }


def extract_header_table(patient_ids, image_dir, workers: int = 8) -> pd.DataFrame:
    """Read age/sex/view from DICOM headers (no pixel data) for the given patients."""
    image_dir = Path(image_dir)
    jobs = [(pid, image_dir / f"{pid}.dcm") for pid in patient_ids]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_read_one, jobs))
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def validate_join(table: pd.DataFrame, label_patient_ids) -> None:
    """The clinical table must map one-to-one onto the labelled patients."""
    missing_columns = [c for c in TABLE_COLUMNS if c not in table.columns]
    if missing_columns:
        raise ValueError(f"Clinical table is missing columns: {missing_columns}")
    if table["patient_id"].isna().any():
        raise ValueError("Clinical table contains empty patient IDs.")
    duplicated = table["patient_id"][table["patient_id"].duplicated()]
    if len(duplicated):
        raise ValueError(f"Duplicate patient IDs in clinical table, e.g. {duplicated.iloc[0]!r}.")
    table_ids, label_ids = set(table["patient_id"]), set(label_patient_ids)
    if table_ids - label_ids:
        raise ValueError(f"{len(table_ids - label_ids)} clinical rows have no label.")
    if label_ids - table_ids:
        raise ValueError(f"{len(label_ids - table_ids)} labelled patients have no clinical row.")


def missingness_report(table: pd.DataFrame) -> dict:
    n = len(table)
    return {
        "n": int(n),
        "age_missing": int(table["age"].isna().sum()),
        "sex_missing": int(table["sex"].isna().sum()),
        "view_position_missing": int(table["view_position"].isna().sum()),
    }


class ClinicalFeatureEncoder:
    """
    Turn (age, sex[, view_position]) into a fixed numeric vector.

    Vector layout: age_z, age_missing, sex one-hot (M, F, unknown),
    and, only if use_view_position, view one-hot (AP, PA, unknown).
    Fit on training rows only.
    """

    def __init__(self, use_view_position: bool = False):
        self.use_view_position = use_view_position
        self.age_mean: float | None = None
        self.age_std: float | None = None

    @property
    def feature_names(self) -> list[str]:
        names = ["age_z", "age_missing", "sex_M", "sex_F", "sex_unknown"]
        if self.use_view_position:
            names += ["view_AP", "view_PA", "view_unknown"]
        return names

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    def fit(self, train_table: pd.DataFrame) -> "ClinicalFeatureEncoder":
        ages = train_table["age"].dropna().to_numpy(dtype=np.float64)
        if len(ages) == 0:
            raise ValueError("Cannot fit age normalization: no training ages available.")
        self.age_mean = float(ages.mean())
        self.age_std = float(ages.std()) or 1.0
        return self

    def _check_fitted(self):
        if self.age_mean is None:
            raise RuntimeError("ClinicalFeatureEncoder must be fitted on training data first.")

    def transform(self, table: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        age = table["age"].to_numpy(dtype=np.float64)
        age_missing = np.isnan(age)
        age_z = np.where(age_missing, 0.0, (age - self.age_mean) / self.age_std)
        columns = [age_z, age_missing.astype(np.float64)]
        columns += self._one_hot(table["sex"], SEX_CATEGORIES)
        if self.use_view_position:
            columns += self._one_hot(table["view_position"], VIEW_CATEGORIES)
        return np.stack(columns, axis=1).astype(np.float32)

    @staticmethod
    def _one_hot(series: pd.Series, categories) -> list[np.ndarray]:
        values = series.tolist()
        known = [np.array([v == c for v in values], dtype=np.float64) for c in categories]
        unknown = np.array([v not in categories for v in values], dtype=np.float64)
        return known + [unknown]

    def structured_mask(self, table: pd.DataFrame) -> np.ndarray:
        """1.0 where at least one structured value is genuinely present, else 0.0."""
        present = ~table["age"].isna() | ~table["sex"].isna()
        if self.use_view_position:
            present = present | ~table["view_position"].isna()
        return present.to_numpy(dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "use_view_position": self.use_view_position,
            "age_mean": self.age_mean,
            "age_std": self.age_std,
            "feature_names": self.feature_names,
            "fitted_on": "training patients only",
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClinicalFeatureEncoder":
        encoder = cls(use_view_position=bool(data["use_view_position"]))
        encoder.age_mean, encoder.age_std = data["age_mean"], data["age_std"]
        return encoder
