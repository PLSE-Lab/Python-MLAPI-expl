# Import libraries

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid',{'axes.grid' : False})
sns.set_context(rc = {'patch.linewidth': 0.0})
bar_settings = {'color': sns.xkcd_rgb['grey'], 'ci': None}
color_settings = {'color': sns.xkcd_rgb['grey']}
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression


# get titanic training dataset & test csv files as a DataFrame
train_df = pd.read_csv('../input/train.csv')
test_df  = pd.read_csv('../input/test.csv')

# preview the data
train_df.head()

train_df.info()
print("----------------------------")
test_df.info()