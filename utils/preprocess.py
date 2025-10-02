import pandas as pd

def clean_input(df, features):
    """ Ensure uploaded CSV has correct columns """
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[features]
