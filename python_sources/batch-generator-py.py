# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import os
import random
import pickle
import numpy as np


class BatchGenerator(object):
    def __init__(self, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len

        dataset, labels, self.translation = self.load_dataset()
        ndataset, nlabels = [], []
        for i in range(len(dataset)):
            if len(dataset[i]) >= seq_len + 1:
                ndataset += [dataset[i]]
                nlabels += [labels[i]]
        del dataset, labels
        self.dataset, labels = ndataset, nlabels

        self.num_letters = len(self.translation)
        # pad all labels to be the same length
        max_len = max(map(lambda x: len(x), labels))
        # One hot encoding
        self.labels = np.array([np.concatenate([np.eye(self.num_letters, dtype=np.float32)[l],
                                                np.zeros((max_len - len(l) + 1, self.num_letters),
                                                         dtype=np.float32)],
                                               axis=0)
                                for l in labels])
        self.max_len = self.labels.shape[1]
        self.indices = np.random.choice(len(self.dataset), size=(batch_size,), replace=False)
        self.batches = np.zeros((batch_size,), dtype=np.int32)

    def next_batch(self):
        coords = np.zeros((self.batch_size, self.seq_len + 1, 3), dtype=np.float32)
        sequence = np.zeros((self.batch_size, self.max_len, self.num_letters), dtype=np.float32)
        reset_states = np.ones((self.batch_size, 1), dtype=np.float32)
        needed = False
        # Might be multiple times of batch size
        for i in range(self.batch_size):
            if self.batches[i] + self.seq_len + 1 > self.dataset[self.indices[i]].shape[0]:
                ni = random.randint(0, len(self.dataset) - 1)
                self.indices[i] = ni
                self.batches[i] = 0
                reset_states[i] = 0.
                needed = True
            coords[i, :, :] = self.dataset[self.indices[i]][self.batches[i]: self.batches[i] + self.seq_len + 1]
            sequence[i] = self.labels[self.indices[i]]
            self.batches[i] += self.seq_len

        return coords, sequence, reset_states, needed

    @staticmethod
    def load_dataset():
        dataset = np.load(os.path.join('../input/data/data', 'dataset.npy'), allow_pickle=True)
        dataset = [np.array(d) for d in dataset]
        temp = []
        for d in dataset:
            # TODO TRAIN ON DIFFERENTCE OF TWO POINTS
            # dataset stores actual pen points, but we will train on differences between consecutive points
            offs = d[1:, :2] - d[:-1, :2]
            ends = d[1:, 2]
            temp += [np.concatenate([[[0., 0., 1.]], np.concatenate([offs, ends[:, None]], axis=1)], axis=0)]
        # because lines are of different length, we store them in python array (not numpy)
        dataset = temp
        labels = np.load(os.path.join('../input/data/data', 'labels.npy'), allow_pickle=True)
        with open(os.path.join('../input/data/data', 'translation.pkl'), 'rb') as file:
            translation = pickle.load(file)

        return dataset, labels, translation


if __name__ == '__main__':
    bg = BatchGenerator(5, 256)
    bg.next_batch()
    bg.next_batch()
# Any results you write to the current directory are saved as output.