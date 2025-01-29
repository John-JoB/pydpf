import polars as pl

from pathlib import Path
from typing import Literal

import argparse


def _preprocess_financial_data(file_path: Path, save_path: Path, *, price_column: str = "Adj Close"):
    # file_path = Path(
    #     "/home/ben/Code/dpf-baselining/Financial data (real)/data/unprocessed/^990100-USD-STRD_historical_data.parquet"
    # )
    # save_path = Path(
    #     "/home/ben/Code/dpf-baselining/Financial data (real)/data/processed/^990100-USD-STRD_historical_data_processed.parquet"
    # )
    price_column = "Adj Close"

    # we assume that we have data at 1 trading day intervals, so that we do not need to account for weekends
    # this implicitely discards out of hours stock price movement, which is not desirable, but for the purposes
    # of demonstration is acceptable

    match file_path.suffix:
        case ".pqt" | ".parquet":
            data = pl.scan_parquet(file_path)
        case ".csv":
            data = pl.scan_csv(file_path, try_parse_dates=True)
        case _:
            raise NotImplementedError()

    data = (
        data
        .select(["Date", price_column])
        .sort("Date")
        .with_columns(
            [
                pl.col("Date").arg_sort().alias("t"),
                (pl.col(price_column) / pl.col(price_column).shift(1)).alias(
                    "exp_log_return"
                ),
            ]
        )
        .drop_nulls()
        .collect()
    )


    data_out = data.rename({"exp_log_return": "observation_1"}).select(["t", "observation_1"])

    match file_path.suffix:
        case ".pqt" | ".parquet":
            data_out.write_parquet(save_path)
        case ".csv":
            data_out.write_csv(save_path)
        case _:
            raise NotImplementedError()

    return data_out

def preprocess_financial_data_directory(data_dir: Path, save_dir: Path, output_type: Literal["csv", "parquet"] = "parquet", **kwargs):
    data_files = list(data_dir.glob("*"))
    print(data_files)
    for file in data_files:
        print(f"Processing {file}...")
        try:
            _preprocess_financial_data(file, save_dir / (file.stem + "." + output_type), **kwargs)
        except Exception as e:
            print(e)
    return None

# preprocess_financial_data_directory(Path(
#         "/home/ben/Code/dpf-baselining/Financial data (real)/data/unprocessed"
#     ),
#     Path(
#         "/home/ben/Code/dpf-baselining/Financial data (real)/data/processed"
#     ), "parquet"
#     )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=lambda p: Path(p))
    parser.add_argument("save_dir", type=lambda p: Path(p))
    parser.add_argument("--price_col", type=str, default="Adj Close")
    parser.add_argument("--output_type", type=str, default="parquet")
    args = parser.parse_args()
    preprocess_financial_data_directory(**vars(args))
