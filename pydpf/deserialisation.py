import polars as pl
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np


def _load_directory_csv(
    directory: Path,
    *,
    processes=-1,
) -> pl.DataFrame:
    """
    Load and concatenate CSV files from a directory into a single DataFrame.

    This function reads all CSV files in the specified directory, adds a
    'series_id' column to each DataFrame based on the file name, and then
    concatenates them into a single DataFrame. The resulting DataFrame is
    sorted by 'series_id', while retaining the order of other columns.

    Parameters:
    directory (Path): The path to the directory containing the CSV files.
    processes (int, optional): The number of processes to use for parallel
        reading. Defaults to -1, which means using all available processors.

    Returns:
    pl.DataFrame: A DataFrame containing data from all CSV files
    in the directory, with an added 'series_id' column.
    """

    def read_helper(file: Path):
        file_data = pl.read_csv(file)
        file_data = file_data.with_columns(pl.lit(file.stem).alias("series_id"))
        return file_data

    read_output = list(
        Parallel(n_jobs=processes)(
            delayed(read_helper)(file)
            for file in directory.iterdir()
            if file.suffix == ".csv"
        )
    )
    return pl.concat(read_output).sort("series_id", maintain_order=True)


def _load_file_csv(
    file: Path,
) -> pl.DataFrame:
    return pl.read_csv(file)


def _extract_tensor(
    data: pl.DataFrame, prefix: str, series_id_column: str
) -> np.ndarray:
    """
    Extracts tensor data from a given DataFrame by selecting columns that match a specified prefix and grouping by a series ID column.

    Parameters:
    data (pl.DataFrame): The input DataFrame containing the data.
    prefix (str): The prefix of the columns we wish to extract.
    series_id_column (str): The column name used to group the data.

    Returns:
    np.ndarray: An AxBxC tensor, with the (a,b,c)th element being the cth element of the bth timestep of the ath series.
        A is the number of series (number of unique elements of the series_id_column),
        B is the number of timesteps in each series, must be the same for all series (no padding is implemented yet),
        C is the number of elements in each timestep (number of columns prefixed by prefix).
    """
    data = data.select([pl.col(f"^{prefix}.*$"), series_id_column])
    data_gb = data.group_by(series_id_column, maintain_order=True)
    tensor_list = []
    # key_list = []
    for gb_key, series_group_data in data_gb:
        tensor = series_group_data.select(pl.exclude(series_id_column)).to_numpy()
        tensor_list.append(tensor)
        # key_list.append(gb_key[0])
    # key_array = np.array(key_list)
    tensor_array = np.stack(tensor_list)
    return tensor_array


def load_data_csv(
    data_path: Path,
    *,
    series_id_column="series_id",
    state_prefix=None,
    observation_prefix="observation_",
    time_column=None,
    control_prefix=None,
):
    if data_path.is_dir():
        data = _load_directory_csv(data_path, series_id_column=series_id_column)
    elif data_path.is_file():
        data = _load_file_csv(data_path)
    else:
        raise ValueError("Invalid data path")

    tensor_dict = {}

    if state_prefix is not None:
        tensor_dict["state"] = _extract_tensor(data, state_prefix, series_id_column)

    tensor_dict["observation"] = _extract_tensor(
        data, observation_prefix, series_id_column
    )

    if control_prefix is not None:
        tensor_dict["control"] = _extract_tensor(data, control_prefix, series_id_column)

    if time_column is not None:
        tensor_dict["time"] = _extract_tensor(data, time_column, series_id_column)

    current_index_end = 0
    output_dict = {}
    output_dict["indices"] = {}
    tensor_list = []
    for key, tensor in tensor_dict.items():
        tensor_list.append(tensor)

        new_index_end = current_index_end + tensor.shape[-1]

        output_dict["indices"][key] = range(current_index_end, new_index_end)

        current_index_end = new_index_end

    output_dict["tensor"] = np.concatenate(tensor_list, axis=-1)

    return output_dict


TEST_PATH = Path(
    "Dynamical systems (synthetic)/Lorenz 96/data/L96_SYSTEM_2024-11-06_12-20-32_25_DIM_STATE_5_DIM_OBS_WITH_80.0PERCENT_SPARSITY/simulated_data.csv"
)

test_loaded = load_data_csv(
    TEST_PATH, state_prefix="state_", observation_prefix="observation_", time_column="t"
)

# first series of observations
test_loaded["tensor"][:, :, test_loaded["indices"]["observation"]][0]

# time data for 12th series
test_loaded["tensor"][:, :, test_loaded["indices"]["observation"]][11]
