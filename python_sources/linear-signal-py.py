"""
"""

from abc import abstractmethod
from typing import List, Tuple, Union

import random
import numpy as np
import pandas as pd
from tensorflow import keras


class SignalFeatureGenerator(object):
    """Abstract base class for a feature generator
    """

    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the features
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, df: pd.DataFrame, predict=False) -> Union[Tuple[np.array, np.array], np.array]:
        """Generate an example given a slice of the input.
        # Arguments
          df: Input slice
          predict: Whether to return the labels (False) or only the features (True)

        # Returns
          When predict is False, a Tuple with 2 np.array elements: X and Y, where X contains the
          features and Y the labels.
          When predict is True, an np.array with features.
        """
        raise NotImplementedError


class LinearDatasetAccessor(object):
    """Provide access to an underlying pandas DataFrame that has been sliced into test/dev dataset
    partitions.
    """

    def __init__(self, df: pd.DataFrame, n_blocks: int, indices: List[int]):
        self.dataframe = df
        self.n_blocks = n_blocks
        self.indices = sorted(indices)
        self.block_size = df.shape[0] // n_blocks

    def get_contiguos_blocks(self) -> List[Tuple[int, int]]:
        """Return a list of contiguos indices in the form of a tuple [begin, end)
        where begin is inclusive and end is exclusive.
        """
        blocks = []

        start = self.indices[0]
        current = start
        for i in range(1, len(self.indices)):
            prev = self.indices[i - 1]
            current = self.indices[i]
            if current > prev + 1:
                blocks.append((start, prev + 1))
                start = current
        blocks.append((start, current + 1))
        return blocks

    def get_block_size(self, block: Tuple[int, int]) -> int:
        """Return the size of a contiguos block in terms of data points.

        The last block has more datapoints when the dataset size is not a multiple of n_blocks.
        """
        start, end = block
        bsize = (end - start) * self.block_size
        if end == self.n_blocks:
            last_block_start = (self.n_blocks - 1) * self.block_size
            last_block_size = self.dataframe.shape[0] - last_block_start
            assert last_block_size >= self.block_size, \
                'block_size: {0}, last_size: {1}'.format(
                    self.block_size, last_block_size)
            bsize += last_block_size - self.block_size
        return bsize

    def get_dataframe_slice(self, block: Tuple[int, int], offset: int, size: int) -> pd.DataFrame:
        begin, end = block
        begin_loc = begin * self.block_size + offset
        end_loc = begin_loc + size
        assert end_loc <= end * self.block_size
        assert end_loc <= self.dataframe.shape[0]
        return self.dataframe[begin_loc:end_loc]


class LinearSignalGenerator(keras.utils.Sequence):
    """Iterate through a linear signal which may have been sliced into blocks.

    Within a contiguos block generate examples by moving a window of `segment_size`
    by `strides` within a block. For instance if a given block has 100 datapoints,
    segment_size is 10 and strides is 5, the generator generates an example for:
      [0, 10), [5, 15), [10, 20), [15, 25), ..., [85, 95), [90, 100) 

    """

    def __init__(self, dataset: LinearDatasetAccessor, segment_size: int, feature_gen: SignalFeatureGenerator,
                 strides=None, batch_size=32):
        self.dataset = dataset
        self.segment_size = segment_size
        self.feature_gen = feature_gen
        if strides is None:
            strides = segment_size
        self.strides = strides
        self.batch_size = batch_size

        self.blocks = self.dataset.get_contiguos_blocks()
        self.n_batches = self._compute_n_batches()
        self.batch_assign = self._generate_stride_list()
        random.shuffle(self.batch_assign)

    def _compute_n_batches(self) -> int:
        total_samples = 0
        for block in self.blocks:
            bsize = self.dataset.get_block_size(block)
            total_samples += (bsize - self.segment_size) // self.strides + 1
        return (total_samples - 1) // self.batch_size + 1

    def _generate_stride_list(self) -> List[Tuple[int, int]]:
        """Generate a list of blocks ids and strides. This list will be shuffled at each
        iteration.
        """
        assignments = []
        for i, block in enumerate(self.blocks):
            bsize = self.dataset.get_block_size(block)
            bstrides = (bsize - self.segment_size) // self.strides + 1
            assignments.extend([(i, s) for s in range(bstrides)])
        return assignments

    def __len__(self):
        return self.n_batches

    def _get_batch_assign(self, index: int) -> List[Tuple[int, int]]:
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.batch_assign))
        return self.batch_assign[start:end]

    def __getitem__(self, index: int):
        assign = self._get_batch_assign(index)

        X_list = []
        Y_list = []
        for block_ix, stride in assign:
            block = self.blocks[block_ix]
            df = self.dataset.get_dataframe_slice(
                block, stride * self.strides, self.segment_size)
            data = self.feature_gen.generate(df)
            if isinstance(data[0], list):
                if not X_list:
                    X_list = [[] for _ in data[0]]
                for i, v in enumerate(data[0]):
                    X_list[i].append(v)
            else:
                X_list.append(data[0])
            Y_list.append(data[1])

        if X_list and isinstance(X_list[0], list):
            X = [np.stack(items) for items in X_list]
        else:
            X = np.stack(X_list)
        Y = np.stack(Y_list)
        return X, Y

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        random.shuffle(self.batch_assign)
