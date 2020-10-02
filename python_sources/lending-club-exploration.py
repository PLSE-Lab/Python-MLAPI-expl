# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



# Any results you write to the current directory are saved as output.

import numpy as np
from scipy.stats import kendalltau
import seaborn as sns
import pandas as pd
loans = pd.read_csv("../input/loan.csv", index_col = 0, low_memory=False)