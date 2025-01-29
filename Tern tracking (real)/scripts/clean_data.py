import polars as pl

from pathlib import Path

import argparse

# call like /home/ben/mambaforge/envs/dpf_project/bin/python "/home/ben/Code/dpf-baselining/Tern tracking (real)/scripts/clean_data.py" "/home/ben/Code/dpf-baselining/Tern tracking (real)/data/unprocessed" "/home/ben/Code/dpf-baselining/Tern tracking (real)/data/processed/ci_tern_data.csv"


def process_tern_data(data_dir: Path, save_file: Path, *, min_series_obs: int):
    time_column = "TSECS"
    id_columns = ("TRACKID", "TRACK_ID")
    obs_columns = ("BNGX", "BNGY")

    data_files = list(data_dir.glob("[!.]*.xlsx"))
    data_frames = []

    for file in data_files:
        print(f"Processing {file.stem}")
        try:
            id_column = id_columns[0]
            df = pl.read_excel(file, columns=[id_column, time_column, *obs_columns])
        except:
            id_column = id_columns[1]
            df = pl.read_excel(file, columns=[id_column, time_column, *obs_columns])

        df = df.with_columns(
            pl.concat_str([pl.lit(file.stem), pl.col(id_column)]).alias("global_id")
        )
        df = df.select(pl.exclude(id_column))

        df = df.rename(
            {
                time_column: "t",
                obs_columns[0]: "observation_1",
                obs_columns[1]: "observation_2",
            }
        )

        data_frames.append(df)

    print("Performing postprocessing")

    unified_df = pl.concat(data_frames)
    del data_frames

    unified_df_w_id = unified_df.with_columns(
        pl.col("global_id").rank("dense").alias("series_id")
    )
    unified_df_id_count = unified_df_w_id.select(
        pl.col("series_id").value_counts(sort=True)
    ).unnest("series_id")

    unified_df_w_id = (
        unified_df_w_id.join(unified_df_id_count, on="series_id", how="inner")
        .filter(pl.col("count") >= min_series_obs)
        .select(pl.exclude("count"))
    )

    out_df = unified_df_w_id.select(
        ["series_id", "t", "observation_1", "observation_2"]
    )
    out_df = out_df.sort(["series_id", "t"])
    out_df = out_df.with_columns(
        (pl.col("t") - pl.col("t").min() + pl.lit(1)).over("series_id")
    )
    print(f"Saving to {save_file}")
    out_df.write_csv(save_file)
    print("Done!")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=lambda p: Path(p))
    parser.add_argument("save_file", type=lambda p: Path(p))
    parser.add_argument("--min_series_obs", type=int, default=50)
    args = parser.parse_args()
    process_tern_data(**vars(args))
