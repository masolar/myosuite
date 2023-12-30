'''
The goal of this script is to give statistics for the Google fingerspelling
dataset. This allows for further processing of the data based on these numbers.
'''
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List
import pyrallis
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

# We use a global here in order to memoize the large parquet files being read
curr_parquet_filepath = None
curr_parquet = None


@dataclass
class ScriptParams:
    '''
    Parameters for running this script
    '''
    # The path to the dataset contents
    dataset_path: Optional[Path] = field(default=None)


def calc_stats(data: pd.Series, dataset_path: Tuple[Path]) -> pd.Series:
    """
    Calculates the number of frames from a particular row of a table

    Expects the data to the function to consist of two things. The parquet
    filepath for the data and the index into the table
    """
    global curr_parquet_filepath
    global curr_parquet

    parquet_filepath: str = data.iloc[0]
    sample_index = data.iloc[2]
    # We load in the data only if it's not already loaded in
    if curr_parquet_filepath is None or parquet_filepath != curr_parquet_filepath:
        curr_parquet_filepath = parquet_filepath
        curr_parquet = pd.read_parquet(dataset_path / parquet_filepath)

    parquet_data = curr_parquet

    user_data = parquet_data.loc[[sample_index]]
    # Get the specific frames we want and return the number
    # We use the right hand for calculating nans as a single missing hand point should mean the entire hand is Nan
    return pd.Series([len(user_data), user_data['x_right_hand_0'].isna().sum()], index=['frames', 'nans'])


@ pyrallis.wrap()
def main(cfg: ScriptParams):
    # Get the files that are under the dataset
    if cfg.dataset_path is not None:
        # First, read in the metadata for the training and supplemental data
        metadata = pd.concat([pd.read_csv(cfg.dataset_path / 'train.csv'),
                             pd.read_csv(cfg.dataset_path / 'supplemental_metadata.csv')], ignore_index=True)

        # For debugging before running the whole script
        # metadata = metadata.head()

        metadata['phrase_len'] = metadata['phrase'].apply(len)

        # Then calculate the number of frames for a phrase
        metadata = pd.concat([metadata, metadata.progress_apply(
            calc_stats, axis=1, args=(cfg.dataset_path,))], axis=1)

        # Finally, calculate the average frames per letter
        print(
            f'Average frames per letter: {(metadata["frames"] / metadata["phrase_len"]).mean()}')

        metadata['nan_ratio'] = (metadata["nans"]/metadata["frames"])
        print(metadata.sort_values(by='nan_ratio', ascending=True).head(
            10)[['path', 'sequence_id', 'phrase', 'nan_ratio']])
        # print(f'Lowest Nan ratio occurs at index: {nan_ratio.idxmin()}')
        # print(f'Lowest Nan ratio: {nan_ratio.min()}')
        # print(
        #     f'Phrase with lowest rate of NANs: {metadata.loc[nan_ratio.idxmin()]["phrase"]}')
    else:
        print(f'Please enter a path to the dataset')


if __name__ == "__main__":
    main()
