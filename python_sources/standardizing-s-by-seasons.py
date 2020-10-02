#!/usr/bin/env python
# coding: utf-8

# This notebook shows how distribution of S (speed) changes across seasons. This change is perhaps created by the differences in speed measurements the host explained in the item 3 in this post: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/111918#latest-650683
# 
# You may have a small model performance boost if you standadize S by seasons as demonstrated in this notebook.

# In[ ]:


from kaggle.competitions import nflrush
import pandas as pd
import numpy as np

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False, nrows=None)


# In[ ]:


env = nflrush.make_env()
iter_test = env.iter_test()

for idx, (test_df_tmp, sample_prediction_df) in enumerate(iter_test):
    if idx == 0:
        test_df = test_df_tmp.copy()
    else:
        test_df = pd.concat([test_df, test_df_tmp])
        
    env.predict(sample_prediction_df)


# In[ ]:


S_2017 = train_df['S'][train_df['Season'] == 2017]
S_2018 = train_df['S'][train_df['Season'] == 2018]
S_2019 = test_df['S']


# In[ ]:


sns.distplot(S_2017, label="2017")
sns.distplot(S_2018, label="2018")
sns.distplot(S_2019, label="2019")
plt.legend(prop={'size': 12})


# As you can see in the plot above, the distribution of 2017 is little bit different from that of 2018 or 2019.
# 
# If you standardize S by mean and std of each year, 2017 will looks more similar to other 2 years. 

# In[ ]:


sns.distplot((S_2017 - S_2017.mean())/ S_2017.std(), label="2017")
sns.distplot((S_2018 - S_2018.mean())/ S_2018.std(), label="2018")
sns.distplot((S_2019 - S_2019.mean())/ S_2019.std(), label="2019")
plt.legend(prop={'size': 12})


# The mean and std used hear are showed below in case you want to use them.

# In[ ]:


print("2017 S mean: {:.4f}, S std: {:.4f}".format(S_2017.mean(), S_2017.std()))
print("2018 S mean: {:.4f}, S std: {:.4f}".format(S_2018.mean(), S_2018.std()))
print("2019 S mean: {:.4f}, S std: {:.4f}".format(S_2019.mean(), S_2019.std()))


# Now, you may wonder if another speed related field 'A (acceleration)' has the same issue. A actually has less difference across years than S. You may not benefit from standardizing this field.

# In[ ]:


sns.distplot(train_df['A'][train_df['Season'] == 2017].dropna(), label="2017")
sns.distplot(train_df['A'][train_df['Season'] == 2018].dropna(), label="2018")
sns.distplot(test_df['A'].dropna(), label="2019")
plt.legend(prop={'size': 12})
plt.xlim(0,8)


# Hope this notebook helps. Happy Kaggling!

# In[ ]:




