#!/usr/bin/env python
# coding: utf-8

# Following the ["Standardizing S by Seasons"](https://www.kaggle.com/tnmasui/standardizing-s-by-seasons) kernel by **tnmasui**, I decided to look at other variables. I ended up finding something really weird about Orientation. Let's take a look...

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

from tqdm import tqdm_notebook

test_dfs_tmp = []
with tqdm_notebook(total=3438) as pbar:
    for idx, (test_df_tmp, sample_prediction_df) in enumerate(env.iter_test()):
        test_dfs_tmp.append(test_df_tmp)
        env.predict(sample_prediction_df)
        pbar.update(1)
        
# NB: Putting pdf.concat outside the loop makes things >10x faster
print('Starting concat...')
test_df = pd.concat(test_dfs_tmp)  
print('...Done')


# In[ ]:


# NB: tnmasui's kernel did not fix play direction, so let's do that.
# HT to https://www.kaggle.com/cpmpml/initial-wrangling-voronoi-areas-in-python by CPMP for the assist here.
def fix_play_direction(df):
    df.loc[df['PlayDirection'] == 'left', 'X'] = 120 - df.loc[df['PlayDirection'] == 'left', 'X']
    df.loc[df['PlayDirection'] == 'left', 'Y'] = (160 / 3) - df.loc[df['PlayDirection'] == 'left', 'Y']
    df.loc[df['PlayDirection'] == 'left', 'Orientation'] = np.mod(180 + df.loc[df['PlayDirection'] == 'left', 'Orientation'], 360)
    df.loc[df['PlayDirection'] == 'left', 'Dir'] = np.mod(180 + df.loc[df['PlayDirection'] == 'left', 'Dir'], 360)
    df['FieldPosition'].fillna('', inplace=True)
    df.loc[df['PossessionTeam'] != df['FieldPosition'], 'YardLine'] = 100 - df.loc[df['PossessionTeam'] != df['FieldPosition'], 'YardLine']
    return df

train_df2 = fix_play_direction(train_df)
test_df2 = fix_play_direction(test_df)


# In[ ]:


# NB: Make this a function so we can easily reuse it to check variables. Code is adapted from tnmasui's kernel.
def check_var(train_df, test_df, var):
    S_2017 = train_df[var][train_df2['Season'] == 2017].fillna(0)
    S_2018 = train_df[var][train_df2['Season'] == 2018].fillna(0)
    S_2019 = test_df[var].fillna(0)
    
    sns.distplot(S_2017, label="2017")
    sns.distplot(S_2018, label="2018")
    sns.distplot(S_2019, label="2019")
    plt.legend(prop={'size': 12})
    plt.show()
    
    print("2017 {} mean: {:.4f}, S std: {:.4f}".format(var, S_2017.mean(), S_2017.std()))
    print("2018 {} mean: {:.4f}, S std: {:.4f}".format(var, S_2018.mean(), S_2018.std()))
    print("2019 {} mean: {:.4f}, S std: {:.4f}".format(var, S_2019.mean(), S_2019.std()))


# In[ ]:


check_var(train_df2, test_df2, 'Orientation')


# ...As you can see, even though the mean and std of Orientation data is pretty similar year-over-year, when you plot the distributions you can see the 2017 data for Orientation is shifted relative to the 2018 (train) and 2019 (test) data. ...In fact, it looks like the 2017 data is shifted exactly by 90 relative to the 2018 / 2019 data.
# 
# If we realign Orientation by 90, the distribution then looks perfect...

# In[ ]:


train_df3 = train_df2.copy()

# Add 90 to Orientation for 2017 season only
train_df3.loc[train_df3['Season'] == 2017, 'Orientation'] = np.mod(90 + train_df3.loc[train_df2['Season'] == 2017, 'Orientation'], 360)

check_var(train_df3, test_df2, 'Orientation')


# ...Suspicious.
# 
# Notably, Michael Lopez [says that Orientation is not likely to be accurate in the 2017 data](https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/111918#646009).
# 
# See also [this thread from CPMP](https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112812).
# 
# Don't get too excited about this, however, as unfortunately, "fixing" Orientation in this manner made my LB score **worse**. :/ So I'm not sure what to make of this yet. Let me know if you do.
