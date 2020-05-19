import logging
import json
import time

from magis_sigdial2020 import settings
from magis_sigdial2020.utils.data import Dataset
from magis_sigdial2020.vocab import Vocabulary
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def get_xkcd_vocab():
    with open(settings.XKCD_VOCAB, "r") as fp:
        return Vocabulary(token_to_idx=json.load(fp))


def simple_stopwatch(display_string="[{i}] Elapsed: {elapsed_tick:0.2f}s ({elapsed_total:0.2f}s)", enable=False):
    status = {
        "start": time.time(),
        "tick": None,
        "last_tick": time.time(),
        "i": 0
    }
    def tick():
        status["tick"] = time.time()
        status["elapsed_tick"] = status["tick"]  - status["last_tick"]
        status["elapsed_total"] = status["tick"] - status["start"]
        print(display_string.format(**status))
        status["last_tick"] = status["tick"]
        status["i"] += 1
    
    def dummy():
        pass
    
    if enable:
        return tick
    else:
        return dummy
        

class XKCD(Dataset):
    def __init__(self, matrix_filename, annotation_filename, coordinate_system='hue', 
                 subset_function=lambda x: x, fft_resolution=3, timeit=False):
        """
        Args:
            matrix_filename (str)
            annotation_filename (str)
            coordinate_system (str):
                options are 'x-y', 'hue', and 'fft'
        """
        stopwatch = simple_stopwatch(enable=timeit)
        
        with open(matrix_filename, "rb") as fp:
            self.data_matrix = np.load(fp)
            
        # 1
        stopwatch()
            
        self.annotations = subset_function(pd.read_csv(annotation_filename))


        # for some reason, can't get rid of this. 
        # tried index=False in a df.to_csv , no effect
        # tried index_col=0, but it gives a warning. 
        if 'Unnamed: 0' in self.annotations.columns:
            del self.annotations['Unnamed: 0']

        if coordinate_system == 'x-y':
            logger.info("Using the x-y coordinate system")
            self._original_data_matrix = self.data_matrix
            num_rows, _ = self._original_data_matrix.shape
            hue_col = self._original_data_matrix[:, 0]
            self.data_matrix = np.zeros((num_rows, 4))
            self.data_matrix[:, 2:] = self._original_data_matrix[:, 1:]
            self.data_matrix[:, 0] = np.sin(hue_col * 2 * np.pi)
            self.data_matrix[:, 1] = np.cos(hue_col * 2 * np.pi)
        elif coordinate_system == 'super':
            self._original_data_matrix = self.data_matrix
            num_rows, _ = self._original_data_matrix.shape
            hue_col = self._original_data_matrix[:, 0]
            self.data_matrix = np.zeros((num_rows, 5))
            theta = hue_col * 2 * np.pi - np.pi
            self.data_matrix[:, 3:] = self._original_data_matrix[:, 1:]
            self.data_matrix[:, 0] = np.sin(theta)
            self.data_matrix[:, 1] = np.cos(theta)
            self.data_matrix[:, 2] = np.tanh(theta)
        elif coordinate_system == 'fft':
            self._original_data_matrix = self.data_matrix
            data = self.data_matrix.copy()
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
            self.data_matrix = np.hstack([repr_complex.real, repr_complex.imag]).astype(np.float32)
        elif coordinate_system == "hue":
            logger.info("Using the hue coordinate system")
        else:
            raise Exception(f"Unknown coordinate_system: {coordinate_system}")
            
        # 2
        stopwatch()
         
        self.annotations['color_name'] = \
            self.annotations.color_name.apply(lambda s: s.replace("-", " "))   
        color_names = sorted(self.annotations.color_name.unique())
        
        # 3
        stopwatch()
        
        self.color_vocab = get_xkcd_vocab()
        
        # 4
        stopwatch()

        col2index = {col:idx for idx, col in enumerate(self.annotations.columns)}

        self.train_df = self.annotations[self.annotations.split=='train']
        self.train_size = len(self.train_df)
        self.train_fast = []
        for row_values in self.train_df.values:
            row_index = row_values[col2index['row_index']]
            color_name = row_values[col2index['color_name']]
            label_index = self.color_vocab._token_to_idx[color_name]
            self.train_fast.append((row_index, label_index))
            
        # 5
        stopwatch()

        self.val_df = self.annotations[self.annotations.split=='val']
        self.validation_size = len(self.val_df)
        self.val_fast = []
        for row_values in self.val_df.values:
            row_index = row_values[col2index['row_index']]
            color_name = row_values[col2index['color_name']]
            label_index = self.color_vocab._token_to_idx[color_name]
            self.val_fast.append((row_index, label_index))
            
        # 6
        stopwatch()


        self.test_df = self.annotations[self.annotations.split=='test']
        self.test_size = len(self.test_df)
        self.test_fast = []
        for row_values in self.test_df.values:
            row_index = row_values[col2index['row_index']]
            color_name = row_values[col2index['color_name']]
            label_index = self.color_vocab._token_to_idx[color_name]
            self.test_fast.append((row_index, label_index))
            
        # 7
        stopwatch()

        self._lookup_dict = {'train': (self.train_df, 
                                       self.train_size, 
                                       self.train_fast), 
                             'val': (self.val_df, 
                                     self.validation_size, 
                                     self.val_fast), 
                             'test': (self.test_df, 
                                      self.test_size, 
                                      self.test_fast)}

        self.data_vector_size = self.data_matrix.shape[1]

        self.set_split('train')
        
        self.data_matrix = self.data_matrix.astype(settings.FLOATX)
        self._use_fast = True

    @classmethod
    def from_settings(cls, coordinate_system="hue", subset_function=lambda x: x, timeit=False):
        return cls(settings.XKCD_DATASET_FILES['color_values'], 
                   settings.XKCD_DATASET_FILES['annotations'], 
                   coordinate_system,
                   subset_function=subset_function,
                   timeit=timeit)

    def set_split(self, split="train", use_fast=True):
        self._target_split = split
        self._target_df, self._target_size, self._target_fast = self._lookup_dict[split]
        self._use_fast = use_fast

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        if self._use_fast:
            row_index, label_index = self._target_fast[index]
            vector = self.data_matrix[row_index]
        else:
            item = self._target_df.iloc[index]
            vector = self.data_matrix[item.row_index]
            label_index = self.color_vocab.lookup_token(item.color_name)
        return {
            'x_color_value': vector, 
            'y_color_name': label_index,
            'data_index': index
        }  

    def get_num_batches(self, batch_size):
        return len(self) // batch_size
