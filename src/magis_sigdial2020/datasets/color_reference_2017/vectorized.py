import ast
import json
import logging
import os

from magis_sigdial2020.datasets.color_reference_2017.munge import populate_lux_globals, get_round_df
from magis_sigdial2020.datasets.color_reference_2017.raw import ColorsInContext
from magis_sigdial2020.datasets.xkcd import XKCD
from magis_sigdial2020.settings import CIC_VECTORIZED_CSV, CIC_DATA_CSV
from magis_sigdial2020.utils.color import hsl2hsv_matrix
from magis_sigdial2020.utils.data import Dataset
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def make_or_load_cic(coordinate_system='fft', cic_vectorized_csv=CIC_VECTORIZED_CSV, verbose=True, force_reload=False):
    xkcd = XKCD.from_settings(coordinate_system=coordinate_system)
    if os.path.exists(cic_vectorized_csv) and not force_reload:
        cic_df = pd.read_csv(cic_vectorized_csv, 
                             converters={
                                'target': ast.literal_eval,
                                'alt1': ast.literal_eval,
                                'alt2': ast.literal_eval
                            })
        cic_df['utterance_events'] = cic_df.full_text.apply(json.loads)
        if verbose:
            print("Loaded from disk")
    else:
        raw_dataset = ColorsInContext.make_from_csv(CIC_DATA_CSV)
        populate_lux_globals(xkcd.color_vocab._token_to_idx)
        cic_df = get_round_df(raw_dataset)
        cic_df.to_csv(cic_vectorized_csv)
        if verbose:
            print("saved to disk")
    return VectorizedColorsInContext(cic_df, xkcd.color_vocab, coordinate_system=coordinate_system)


class VectorizedColorsInContext(Dataset):
    def __init__(self, df, color_vocab, coordinate_system='x-y', fft_resolution=3):
        targets = hsl2hsv_matrix(np.stack(list(map(np.array, df.target.values))))
        alts1 = hsl2hsv_matrix(np.stack(list(map(np.array, df.alt1.values))))
        alts2 = hsl2hsv_matrix(np.stack(list(map(np.array, df.alt2.values))))
        self._df = df
        self.x_color_values = np.concatenate([targets, alts1, alts2])
        self.indices = {}
        self._offset = targets.shape[0]
        self._active_split = 'train'
        self._active_difficulties = [0, 1, 2, 3]
        self._color_vocab = color_vocab

        self.label_indices = np.ones(targets.shape[0]).astype(np.int64) * -1
        for split_name in df.split.unique():
            for difficulty in df.lux_difficulty_rating.unique():
                dfsub = df[(df.split == split_name) & (df.lux_difficulty_rating == difficulty)]
                self.indices[split_name, difficulty] = dfsub.index
                if difficulty < 4:
                    # Currently skipping no match situations
                    self.label_indices[dfsub.index] = list(map(color_vocab.lookup_token,
                                                               dfsub.lux_label.values))

        if coordinate_system == 'x-y':
            self._x_color_values = self.x_color_values
            num_rows, _ = self.x_color_values.shape
            hue_col = self.x_color_values[:, 0]
            self.x_color_values = np.zeros((num_rows, 4))
            self.x_color_values[:, 0] = np.sin(hue_col * 2 * np.pi)
            self.x_color_values[:, 1] = np.cos(hue_col * 2 * np.pi)
            self.x_color_values[:, 2:] = self._x_color_values[:, 1:]
        elif coordinate_system == 'fft':
            self._x_color_values = self.x_color_values
            data = self.x_color_values.copy()
            resolutions = [fft_resolution] * 3
            gx, gy, gz = np.meshgrid(*[np.arange(r) for r in resolutions])
            data[:, 1:] /= 2
            arg = (np.multiply.outer(data[:, 0], gx) +
                   np.multiply.outer(data[:, 1], gy) +
                   np.multiply.outer(data[:, 2], gz))

            repr_complex = (
                np.exp(-2j * np.pi * (arg % 1.0))
                .swapaxes(1, 2)
                .reshape((data.shape[0], -1))
            )
            self.x_color_values = np.hstack([repr_complex.real, repr_complex.imag]).astype(np.float32)
        elif coordinate_system == "hue":
            logger.info("Using the hue coordinate system")
        else:
            raise Exception(f"Unknown coordinate_system: {coordinate_system}")

        self.refresh_indices()

    def set_split(self, split, difficulty_level=None):
        if split == 'val':
            # the original data frame used 'dev' instead of 'val'; 
            # my base trainer class uses val 
            # so just doing a bit of consistency bookkeeping
            split = 'dev'
        self._active_split = split

        if difficulty_level is not None:
            self.set_difficulty_subset(difficulty_level)
        
        self.refresh_indices()

    def refresh_indices(self):
        indices = []
        for difficulty in self._active_difficulties:
            indices.append(self.indices[self._active_split, difficulty])
        self._active_indices = np.concatenate(indices)

    def set_difficulty_subset(self, difficulty_levels):
        if isinstance(difficulty_levels, int):
            difficulty_levels = [difficulty_levels]

        self._active_difficulties = difficulty_levels
        self.refresh_indices()

    def __getitem__(self, index):
        row_index = self._active_indices[index]
        return {
            "x_colors": self.x_color_values[
                [row_index,
                 row_index + self._offset,
                 row_index + 2 * self._offset]
            ],
            "y_utterance": self.label_indices[row_index],
            "subset_index": index,
            "row_index": row_index
        }

    def __len__(self):
        return self._active_indices.shape[0]
