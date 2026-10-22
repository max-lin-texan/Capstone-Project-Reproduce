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


if __name__ == "__main__":
    ids = load_sepsis_stay_ids()
    print(f"stays: {len(ids):,}")
