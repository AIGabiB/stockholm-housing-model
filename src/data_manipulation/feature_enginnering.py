import pandas as pd
import numpy as np
from src.data_manipulation.data_utils import log_changes


def engineer_features(df: pd.DataFrame, debug:bool = False) -> pd.DataFrame : 
    # check columns 
    cols = ["Contract Date", "Year Built", "Living Area (sqm)", "Total Floors", "Floor", "Monthly Fee"]
    assert set(cols).issubset(df.columns), f"Error: must have all: {cols}"

    old_shape = df.shape
    df = df.copy()

    df["Contract Date"] = pd.to_datetime(df["Contract Date"])
    # -- Date extraction -- 
    df["Contract Date Month"] = df["Contract Date"].dt.month 
    df["Contract Date Day"] = df["Contract Date"].dt.day
    df["Contract Date Year"] = df["Contract Date"].dt.year

    TRAIN_START_DATE = pd.Timestamp("2020-01-01")
    df["Days Since Start"] = (df["Contract Date"] - TRAIN_START_DATE).dt.days

    df = df.drop(columns=["Contract Date"])

    # -- Apply feature engineering transformations -- 
    df = df.assign(
        **{
            "Age at Sale": lambda x: x["Contract Date Year"] - x["Year Built"],
            "Sqm per Room": lambda x: (x["Living Area (sqm)"] / x["Rooms"]).round(2),
            "Top Floor Factor": lambda x: np.where(
                x["Total Floors"] == 0, 1.0, x["Floor"] / x["Total Floors"]).round(2)
        }
        )

    if debug:
        log_changes(old_shape, df, "engineer_features")

    return df