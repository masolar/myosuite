"""
A script that creates a lazy data loader for the Google fingerspelling
dataset.

The data loader will return the following for a single request
- A prev, current, and next transition, consisting of 4 ASCII characters representing the start and end point
of each transition. The null character represents no previous start or end state
- The right hand angles for the sign sequence. If there are gaps in the frames, simple interpolation is used.
If there are continguous gaps that are too large for the current sequence (specified by an argument), this
example isn't included in the sample space

"""
from itertools import product
from pathlib import Path
import pandas as pd
from typing import Set, Iterable, Dict, Tuple, List


class GoogleFingerspellingLoader:
    def __init__(self, metadata_path: Path, char_subset: Set[str], max_interp_frames=3):
        '''
        Sets up the data loader

        Args:
            metadata_path: The path to the metadata file containing sequence and phrase info
            char_subset: The characters that should be considered when generating data
                         There should be at least two characters in here. If there are less than 4,
                         the prev, current, and next sequence will be as follows
                            - Prev and next may contain the null character, representing the missing
                                4th char
                            - Current will contain two of the characters specified
            max_interp_frames: The maximum number of frames that may be missing in a sequence for it to
                               be dropped from the dataset.
        '''
        self.parent_folder = metadata_path.parent

        self.metadata = pd.read_csv(metadata_path)
        self.char_subset = char_subset
        self.max_interp_frames = max_interp_frames

        self.possible_chars = self.get_char_product(
            self.char_subset, self.get_4_grams(list(self.metadata['phrase'].astype(str))))

        self.valid_filenames, self.index_to_filename, self.index_to_seq = self.get_valid_filenames_per_ngram(
            self.metadata, self.possible_chars)

    @staticmethod
    def get_4_grams(corpus: Iterable[str]):
        """
        Returns the 4 grams from a corpus of data, assumed to be lines of text. Also adds in
        the 4 grams found from assuming nothing at the start and end of the 4 gram, meaning \0ww\0

        Args:
            corpus: A list of lines of text to be converted to 4 grams.
        """
        four_grams = set()

        for line in corpus:
            possible_grams = [line[i:i+4]
                              for i in range(len(line) - 3) if len(line) >= 4]

            for possible_gram in possible_grams:
                four_grams.add(possible_gram)
                # Add the other combinations of four grams possible
                four_grams.add('\0' + possible_gram[1:])
                four_grams.add(possible_gram[:3] + '\0')
                four_grams.add('\0' + possible_gram[1:3] + '\0')

        return four_grams

    @staticmethod
    def get_char_product(char_subset: Set[str], dictionary: Set[str]):
        '''
        Gets the possible character products to search for, based on a few simple rules

        Args:
            char_subset: The character subset to generate a product for
            dictionary: A set of strings that are valid in the dataset, used to prune the calculated 4-grams
        '''
        # Make sure that space and the null character are included in the character set
        char_subset.add('\0')
        char_subset.add(' ')

        # Then generate the product of all the characters possible here
        # TODO: Do something more efficient here, for larger character sets
        possible_chars = set(map(''.join, product(char_subset, repeat=4)))

        # Remove n-grams that are not present in the dataset
        return possible_chars.intersection(dictionary)

    @staticmethod
    def get_valid_filenames_per_ngram(metadata: pd.DataFrame, possible_chars: Set[str]):
        """
        Returns the filenames, sequence numbers, and start and end positions, and number of signs
        in the sequence for a given n-gram. Also returns two lookup tables for the
        filename and sequence number, to save a bit of memory.

        Args:
            metadata: The metadata from the dataset
            possible_chars: The possible 4-grams that are seen in the data and
                            match the constraints given
        """
        index_to_filename_count = 0
        index_to_filename: Dict[int, str] = {}
        filename_to_index: Dict[str, int] = {}
        index_to_sequence_count = 0
        index_to_sequence: Dict[int, str] = {}
        sequence_to_index: Dict[str, int] = {}

        result_dict: Dict[str, List[Tuple[int, int, int, int, int]]] = {}

        # Looping through a dataframe isn't great, but the data should be small enough that it's not an issue
        for _, (df_path, _, seq_id, _, phrase) in metadata.iterrows():

            possible_grams = [phrase[i:i+4]
                              for i in range(len(phrase) - 3) if len(phrase) >= 4]

            for i, possible_gram in enumerate(possible_grams):
                start_pos = i
                end_pos = i + 4

                # This represents all the ways this sequence could be interpreted
                checklist = [possible_gram, '\0' + possible_gram[1:],
                             possible_gram[:3] + '\0', '\0' + possible_gram[1:3] + '\0']
                for check in checklist:
                    if check in possible_chars:
                        if df_path not in filename_to_index:
                            index_to_filename[index_to_filename_count] = df_path
                            filename_to_index[df_path] = index_to_filename_count
                            index_to_filename_count += 1
                        if seq_id not in sequence_to_index:
                            index_to_sequence[index_to_sequence_count] = seq_id
                            sequence_to_index[seq_id] = index_to_sequence_count
                            index_to_sequence_count += 1

                        if check not in result_dict:
                            result_dict[check] = [
                                (filename_to_index[df_path], sequence_to_index[seq_id], start_pos, end_pos, len(phrase))]
                        else:
                            result_dict[check].append(
                                (filename_to_index[df_path], sequence_to_index[seq_id], start_pos, end_pos, len(phrase)))

        return result_dict, index_to_filename, index_to_sequence
