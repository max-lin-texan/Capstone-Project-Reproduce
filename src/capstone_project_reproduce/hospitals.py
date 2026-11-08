"""Named hospital cohorts shared by clustering, CPD, and non-negative CPD."""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
MAP_PATH = PROJECT_DIR / "results" / "hospital_map.csv"
PATIENT_PATH = Path.home() / "eICU" / "patient.csv"


def cohort_name(hospital: str | None) -> str:
    """Return ``overall`` or a name listed in the hospital map."""
    if hospital is None:
        return "overall"
    names = set(pd.read_csv(MAP_PATH)["name"].astype(str))
    if hospital not in names:
        known = ", ".join(sorted(names, key=lambda name: int(name.split("_")[1])))
        raise ValueError(f"Unknown hospital {hospital!r}. Names in {MAP_PATH.name}: {known}")
    return hospital


def stay_ids(hospital: str | None) -> set[int] | None:
    """ICU stays in one mapped hospital. ``None`` means the whole table."""
    if hospital is None:
        return None
    table = pd.read_csv(MAP_PATH)
    matched = table.loc[table["name"].astype(str) == hospital, "hospitalid"]
    if matched.empty:
        cohort_name(hospital)
    hospital_id = int(matched.iloc[0])
    patients = pd.read_csv(PATIENT_PATH, usecols=["patientunitstayid", "hospitalid"])
    chosen = patients.loc[patients["hospitalid"] == hospital_id, "patientunitstayid"]
    if chosen.empty:
        raise ValueError(f"No stays for hospital {hospital_id}")
    return set(chosen.astype(int))


def subset_stays(frame: pd.DataFrame, hospital: str | None) -> pd.DataFrame:
    """Keep rows whose stay belongs to this hospital."""
    chosen = stay_ids(hospital)
    if chosen is None:
        return frame
    selected = frame.loc[frame["patientunitstayid"].isin(chosen)].copy()
    if selected.empty:
        raise ValueError(f"{hospital} has no rows in this table")
    return selected
