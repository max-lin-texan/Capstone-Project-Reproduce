import argparse
from pathlib import Path

import pandas as pd

EICU_DIR = Path.home() / "eICU"
SEPSIS_CODES_PATH = Path(__file__).resolve().parents[2] / "docs" / "sepsis_code.txt"


def load_sepsis_codes(path: Path = SEPSIS_CODES_PATH) -> set[str]:
    """Codes listed in docs/sepsis_code.txt, one per line."""
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def load_sepsis_stay_ids(
    eicu_dir: Path = EICU_DIR,
    codes_path: Path = SEPSIS_CODES_PATH,
) -> pd.Series:
    """ICU stays with at least one diagnosis code from sepsis_code.txt.

    diagnosis.csv stores codes in icd9code, often several per cell separated
    by commas. The three ICD-10-PCS procedure codes in the text file are not
    in this table, so they do not match any stay.
    """
    codes = load_sepsis_codes(codes_path)
    dx = pd.read_csv(
        eicu_dir / "diagnosis.csv",
        usecols=["patientunitstayid", "icd9code"],
        dtype=str,
    )
    dx = dx.assign(code=dx["icd9code"].fillna("").str.split(r"\s*,\s*")).explode("code")
    dx["code"] = dx["code"].str.strip()
    stay_ids = (
        dx.loc[dx["code"].isin(codes), "patientunitstayid"]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return stay_ids


# First 120 hours after ICU admission. eICU offsets are minutes.
MAX_OFFSET_MINUTES = 120 * 60
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "sepsis_measurement.csv"
NEW_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "sepsis_measurement_new.csv"

VITAL_COLUMNS = {
    "systemicsystolic": "arterial_bp_systolic",
    "heartrate": "heart_rate",
    "respiration": "respiratory_rate",
}
NEW_VITAL_COLUMNS = {
    **VITAL_COLUMNS,
    "sao2": "oxygen_saturation",
}
# Exact labname values. "urinary creatinine" is a different test and is excluded.
LAB_COLUMNS = {
    "creatinine": "creatinine",
    "base excess": "base_excess",
    "pt - inr": "pt_inr",
    "lactate": "lactate",
}
NEW_LAB_COLUMNS = {
    **LAB_COLUMNS,
    "wbc x 1000": "wbc",
    "platelets x 1000": "platelets",
}
MEASURE_COLUMNS = [
    "patientunitstayid",
    "offset",
    "arterial_bp_systolic",
    "heart_rate",
    "respiratory_rate",
    "base_excess",
    "creatinine",
    "pt_inr",
    "lactate",
]
NEW_MEASURE_COLUMNS = [
    *MEASURE_COLUMNS,
    "wbc",
    "platelets",
    "oxygen_saturation",
    "urine_output",
]


def select_last_icu_stays(
    stay_ids: pd.Series,
    eicu_dir: Path = EICU_DIR,
) -> pd.Series:
    """One stay per patient: the latest of the already selected ICU stays.

    Order is discharge year, then hospital stay, then ICU visit number within
    that hospital stay.
    """
    patients = pd.read_csv(
        eicu_dir / "patient.csv",
        usecols=[
            "patientunitstayid",
            "uniquepid",
            "hospitaldischargeyear",
            "patienthealthsystemstayid",
            "unitvisitnumber",
        ],
    )
    selected = set(stay_ids.astype("int64"))
    patients = patients[patients["patientunitstayid"].isin(selected)]
    patients = patients.sort_values(
        [
            "uniquepid",
            "hospitaldischargeyear",
            "patienthealthsystemstayid",
            "unitvisitnumber",
            "patientunitstayid",
        ],
        kind="mergesort",
    )
    return patients.groupby("uniquepid", sort=False)["patientunitstayid"].tail(1).reset_index(drop=True)


def _in_first_120_hours(offset: pd.Series) -> pd.Series:
    return offset.ge(0) & offset.le(MAX_OFFSET_MINUTES)


def _load_vitals(
    stay_ids: set[int],
    eicu_dir: Path,
    vital_columns: dict[str, str] = VITAL_COLUMNS,
) -> pd.DataFrame:
    parts = []
    usecols = ["patientunitstayid", "observationoffset", *vital_columns]
    for chunk in pd.read_csv(eicu_dir / "vitalPeriodic.csv", usecols=usecols, chunksize=2_000_000):
        chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)]
        chunk = chunk[_in_first_120_hours(chunk["observationoffset"])]
        chunk = chunk.dropna(subset=list(vital_columns), how="all")
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=["patientunitstayid", "offset", *vital_columns.values()])
    vitals = pd.concat(parts, ignore_index=True)
    vitals = vitals.rename(columns={"observationoffset": "offset", **vital_columns})
    return vitals.groupby(["patientunitstayid", "offset"], as_index=False).mean(numeric_only=True)


def _load_labs(
    stay_ids: set[int],
    eicu_dir: Path,
    lab_columns: dict[str, str] = LAB_COLUMNS,
) -> pd.DataFrame:
    parts = []
    usecols = ["patientunitstayid", "labresultoffset", "labname", "labresult"]
    for chunk in pd.read_csv(
        eicu_dir / "lab.csv",
        usecols=usecols,
        dtype={"labname": str},
        chunksize=1_000_000,
    ):
        chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)]
        chunk = chunk[_in_first_120_hours(chunk["labresultoffset"])]
        chunk["variable"] = chunk["labname"].str.lower().str.strip().map(lab_columns)
        chunk = chunk[chunk["variable"].notna() & chunk["labresult"].notna()]
        if not chunk.empty:
            parts.append(chunk[["patientunitstayid", "labresultoffset", "variable", "labresult"]])
    columns = ["patientunitstayid", "offset", *lab_columns.values()]
    if not parts:
        return pd.DataFrame(columns=columns)
    labs = pd.concat(parts, ignore_index=True)
    labs = labs.rename(columns={"labresultoffset": "offset"})
    labs["labresult"] = pd.to_numeric(labs["labresult"], errors="coerce")
    labs = labs.dropna(subset=["labresult"])
    labs = labs.groupby(["patientunitstayid", "offset", "variable"], as_index=False)["labresult"].mean()
    wide = labs.pivot(index=["patientunitstayid", "offset"], columns="variable", values="labresult")
    wide = wide.reset_index()
    for column in lab_columns.values():
        if column not in wide.columns:
            wide[column] = pd.NA
    return wide[columns]


def _load_urine(stay_ids: set[int], eicu_dir: Path) -> pd.DataFrame:
    """Urine volume in mL. Several entries at one minute are added together.

    A minute with no urine row stays absent. It is not stored as zero.
    """
    parts = []
    usecols = ["patientunitstayid", "intakeoutputoffset", "celllabel", "cellvaluenumeric"]
    for chunk in pd.read_csv(
        eicu_dir / "intakeOutput.csv",
        usecols=usecols,
        dtype={"celllabel": str},
        chunksize=1_000_000,
    ):
        chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)]
        chunk = chunk[_in_first_120_hours(chunk["intakeoutputoffset"])]
        label = chunk["celllabel"].str.strip().str.casefold()
        chunk = chunk.loc[label.eq("urine")].copy()
        chunk["urine_output"] = pd.to_numeric(chunk["cellvaluenumeric"], errors="coerce")
        chunk = chunk.dropna(subset=["urine_output"])
        # Stay 3185959 at minute 2463 is charted as 325325 mL, between two
        # entries of 325 mL. The digits are 325 written twice.
        doubled = (
            chunk["patientunitstayid"].eq(3185959)
            & chunk["intakeoutputoffset"].eq(2463)
            & chunk["urine_output"].eq(325325)
        )
        chunk.loc[doubled, "urine_output"] = 325
        if not chunk.empty:
            parts.append(chunk[["patientunitstayid", "intakeoutputoffset", "urine_output"]])
    columns = ["patientunitstayid", "offset", "urine_output"]
    if not parts:
        return pd.DataFrame(columns=columns)
    urine = pd.concat(parts, ignore_index=True)
    urine = urine.rename(columns={"intakeoutputoffset": "offset"})
    return urine.groupby(["patientunitstayid", "offset"], as_index=False)["urine_output"].sum()


def _print_summary(matched_stays: int, last_stays: int, measures: pd.DataFrame, path: Path) -> None:
    value_columns = [column for column in measures.columns if column not in {"patientunitstayid", "offset"}]
    print(f"ICD-matched stays: {matched_stays:,}")
    print(f"patients, last ICU stay kept: {last_stays:,}")
    print(f"stays with a measurement in the first 120 hours: {measures['patientunitstayid'].nunique():,}")
    print(f"rows: {len(measures):,}")
    if measures.empty:
        print("offset: none")
    else:
        print(f"offset minutes: {measures['offset'].min():.0f} to {measures['offset'].max():.0f}")
    for column in value_columns:
        print(f"  {column}: {measures[column].notna().sum():,}")
    print(path)


def write_sepsis_measurements(
    path: Path = OUTPUT_PATH,
    eicu_dir: Path = EICU_DIR,
    verbose: bool = False,
    new: bool = False,
) -> Path:
    """Wide table of the sepsis measures during the first 120 hours.

    Rows are one selected patient's last ICU stay at one offset. A column is
    empty when that measure was not recorded at that time. Set verbose=True to
    print the cohort size, row count, and how many values each column has.
    ``new=True`` adds white cells, platelets, oxygen saturation, and urine
    output, and writes ``sepsis_measurement_new.csv`` unless ``path`` is set.
    """
    if new and path == OUTPUT_PATH:
        path = NEW_OUTPUT_PATH
    columns = NEW_MEASURE_COLUMNS if new else MEASURE_COLUMNS
    matched = load_sepsis_stay_ids(eicu_dir)
    last_stays = select_last_icu_stays(matched, eicu_dir)
    stay_ids = set(last_stays.astype("int64"))
    vitals = _load_vitals(stay_ids, eicu_dir, NEW_VITAL_COLUMNS if new else VITAL_COLUMNS)
    labs = _load_labs(stay_ids, eicu_dir, NEW_LAB_COLUMNS if new else LAB_COLUMNS)
    measures = vitals.merge(labs, on=["patientunitstayid", "offset"], how="outer")
    if new:
        urine = _load_urine(stay_ids, eicu_dir)
        measures = measures.merge(urine, on=["patientunitstayid", "offset"], how="outer")
    measures = measures.dropna(subset=columns[2:], how="all")
    measures = measures.sort_values(["patientunitstayid", "offset"], kind="mergesort")
    path.parent.mkdir(parents=True, exist_ok=True)
    measures[columns].to_csv(path, index=False)
    if verbose:
        _print_summary(len(matched), len(last_stays), measures, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", action="store_true")
    args = parser.parse_args()
    if args.new:
        write_sepsis_measurements(path=NEW_OUTPUT_PATH, verbose=True, new=True)
        return
    write_sepsis_measurements(verbose=True)


if __name__ == "__main__":
    main()
