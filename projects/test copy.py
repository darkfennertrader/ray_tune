import pandas as pd

BASE_DIR1 = "data/AUD-CAD"
BASE_DIR2 = "data/AUD-CHF"
FILENAME = "overall"


def _open_csv(filepath: str) -> pd.DataFrame:
    return pd.read_csv(
        filepath,
        parse_dates=["Datetime"],
        index_col="Datetime",
        date_parser=lambda x: pd.to_datetime(x, utc=True),  # type: ignore
    )


input_df = _open_csv(f"{BASE_DIR1}/overall.csv")
print(input_df.head())


data = input_df["Close"].copy()
print(data.head())
