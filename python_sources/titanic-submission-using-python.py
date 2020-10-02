#standard
import numpy as np
import pandas as pd
from numpy.random import randn

#plotting
import matplotlib as mpl

import seaborn as sns

#stats
from scipy import stats

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

sns.factorplot('Pclass', data = train,kind='count', hue='Sex')

