# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
# data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, roc_curve
from collections import defaultdict, Counter
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

pd.options.display.max_columns=400
from matplotlib_venn import venn2
from sklearn import preprocessing
import matplotlib.pyplot as plt
pd.options.display.max_rows=100
from tqdm import tqdm_notebook
import lightgbm as lgb
import seaborn as sns
from glob import glob
from tqdm import tqdm
import numpy as np
%matplotlib inline
import itertools
import os
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.