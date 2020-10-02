# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.

# Preprocessing

# Label string
class IndexedSet:
    def __init__(self):
        self.data = dict()
        self.max_index = -1

    def add(self, item):
        if self.data.get(item) is None:
            self.max_index += 1
            self.data[item] = self.max_index

    def label(self, item):
        result = self.data.get(item)
        if result is None:
            return -1
        return result
        

X = np.loadtxt('../input/kaggle_sample.csv')
