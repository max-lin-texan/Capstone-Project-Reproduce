"""Named hospital cohorts shared by clustering, CPD, and non-negative CPD."""

import argparse
import shutil
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
MAP_PATH = PROJECT_DIR / "results" / "hospital_map.csv"
PATIENT_PATH = Path.home() / "eICU" / "patient.csv"
MIN_HOSPITAL_PATIENTS = 100
MIN_SINGLE_ORGAN = 5


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


def clear_previous_results() -> None:
    """Remove the 14-hospital outputs once, before the screened map is written."""
    results = MAP_PATH.parent
    old_directory = (results / "feature_7" / "hospital_14").exists()
    old_map = False
    if MAP_PATH.exists():
        names = set(pd.read_csv(MAP_PATH)["name"].astype(str))
        old_map = "hospital_14" in names
    if not old_directory and not old_map:
        return
    for name in ("feature_7", "feature_11", "cosine_similarity_heatmaps"):
        folder = results / name
        if folder.exists():
            shutil.rmtree(folder)
    if MAP_PATH.exists():
        MAP_PATH.unlink()
    print(f"removed previous results under {results}")


def write_hospital_map(path: Path = MAP_PATH) -> Path:
    """Rank screened hospitals that have enough patients in each organ group.

    The count is the eleven-feature screened cohort. A hospital needs at least
    100 of those patients, and at least five patients whose only organ label is
    liver, five whose only label is kidney, and five whose only label is lung.
    ``hospital_1`` is the largest.
    """
    from clustering import organ_labels
    from data_preprocessing import ensure_icd_lookup, screened_stay_ids

    diagnoses = ensure_icd_lookup()
    stays = screened_stay_ids(new=True)
    selected = diagnoses.loc[diagnoses["patientunitstayid"].isin(stays)].copy()
    labels = organ_labels(selected["icd_code"].fillna("").to_numpy())
    single = labels.sum(axis=1) == 1
    selected["liver_only"] = single & labels[:, 0]
    selected["kidney_only"] = single & labels[:, 1]
    selected["lung_only"] = single & labels[:, 2]
    patients = pd.read_csv(PATIENT_PATH, usecols=["patientunitstayid", "hospitalid"])
    merged = selected.merge(patients, on="patientunitstayid", how="left")
    grouped = merged.groupby("hospitalid", sort=False).agg(
        n_patients=("patientunitstayid", "size"),
        liver=("liver_only", "sum"),
        kidney=("kidney_only", "sum"),
        lung=("lung_only", "sum"),
    )
    chosen = grouped[
        (grouped["n_patients"] >= MIN_HOSPITAL_PATIENTS)
        & (grouped["liver"] >= MIN_SINGLE_ORGAN)
        & (grouped["kidney"] >= MIN_SINGLE_ORGAN)
        & (grouped["lung"] >= MIN_SINGLE_ORGAN)
    ]
    chosen = chosen.sort_values(["n_patients", "hospitalid"], ascending=[False, True])
    if chosen.empty:
        raise ValueError("No hospital meets the screened cohort rules")
    table = pd.DataFrame(
        {
            "name": [f"hospital_{index}" for index in range(1, len(chosen) + 1)],
            "hospitalid": chosen.index.astype(int),
            "n_patients": chosen["n_patients"].astype(int).to_numpy(),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    print(table.to_string(index=False))
    print(f"wrote {len(table)} hospitals to {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if not args.prepare:
        parser.error("pass --prepare to replace the old hospital map")
    clear_previous_results()
    write_hospital_map()


if __name__ == "__main__":
    main()
