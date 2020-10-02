#!/usr/bin/env python
# coding: utf-8

# Congraturations on Baylor University and thanks for hosting this exciting competition!  
# 
# I ended up in the bronze medal, but want to share my simple solution, based on [starter kernel](https://www.kaggle.com/addisonhoward/basic-starter-kernel-ncaa-women-s-dataset-2019).  
# 
# My solution is not complicated:  
# 
# **It predicts the probability of the occurrence of the upset, which means that the low seed rank team beats the high seed rank team, aggregating past game results**  

# In[26]:


import os; print(os.listdir("../input/stage2wdatafiles"))
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.metrics import log_loss

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# ## Preprocess

# In[27]:


data_dir = '../input/stage2wdatafiles/'
df_seed = pd.read_csv(data_dir + 'WNCAATourneySeeds.csv')
df_result = pd.read_csv(data_dir + 'WNCAATourneyCompactResults.csv')

df_seed.tail(3)
df_result.tail(3)


# In[28]:


def seed_to_int(seed):
    s_int = int(seed[1:3])
    return s_int

def clean_df(df_seed, df_result):
    df_seed['seed_int'] = df_seed['Seed'].apply(seed_to_int)
    df_seed.drop(['Seed'], axis=1, inplace=True)
    df_result.drop(['DayNum', 'WLoc', 'NumOT'], axis=1, inplace=True)
    return df_seed, df_result

df_seed, df_result = clean_df(df_seed, df_result)
df_seed.head(3)
df_result.head(3)


# In[29]:


## Merge seed for each team
def merge_seed_result(df_seed, df_result):
    df_win_seed = df_seed.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
    df_loss_seed = df_seed.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})
    df_result = df_result.merge(df_win_seed, how='left', on=['Season', 'WTeamID'])
    df_result = df_result.merge(df_loss_seed, how='left', on=['Season', 'LTeamID'])
    df_result['SeedDiff'] = np.abs(df_result['WSeed'] - df_result['LSeed'])
    df_result['ScoreDiff'] = np.abs(df_result['WScore'] - df_result['LScore'])
    return df_result

df_result = merge_seed_result(df_seed, df_result)
df_result.head(3)


# **Remove the games that end within 3 points difference, which are likely to be the other results**

# In[30]:


df_result = df_result[df_result['ScoreDiff']>3]


# ## Upset Modeling

# ### Validation
# 
# **We can make use of the last competition for validating score**
# 
# - Train: 2009~2017  
# - Test: 2018  

# In[31]:


df_result['upset'] = [1 if ws > ls else 0 for ws, ls, in zip(df_result["WSeed"], df_result["LSeed"])]

print("upset probability")
df_result['upset'].value_counts() / len(df_result) * 100


# **Use only last 10 seasons, since some trends are likely to be changed**

# In[32]:


this_season=2019
total_season=10

train = df_result[ (df_result["Season"]>=(this_season - total_season)) & (df_result["Season"]<(this_season-1)) ]
print(train.shape)


# **The probability of the occurrence of the upset is likely to be different between a game 1st seed vs. 6th seed and a game 11th seed vs. 16th seed, so I want to include the information**

# In[33]:


df_result["Seed_combi"]=[str(ws)+'_'+str(ls) if ws<ls else str(ls)+'_'+str(ws) for ws, ls in zip(df_result["WSeed"], df_result["LSeed"])]
df_result.head(3)


# **Aggregation**

# In[34]:


df_result_aggs = pd.DataFrame()
df_result_filter_aggs = pd.DataFrame()
df_result_season = df_result[ (df_result["Season"]>=(this_season - total_season)) & (df_result["Season"]<(this_season-1)) ]
for value in range(16):
    df_result_agg = df_result_season[df_result_season["SeedDiff"]==value].groupby("SeedDiff").agg({"upset": ["mean", "count"]})
    df_result_agg.columns = [col[0]+"_"+col[1]+"_"+"all" for col in df_result_agg.columns]
    df_result_filter_agg = df_result_season[df_result_season["SeedDiff"]==value].groupby("Seed_combi").agg({"upset": ["mean", "count"]})
    df_result_filter_agg.columns = [col[0]+"_"+col[1] for col in df_result_filter_agg.columns]
    if value==0:
        df_result_agg["upset_mean_all"] = 0.5
        df_result_filter_agg["upset_mean"] = 0.5
    df_result_aggs = pd.concat([df_result_aggs, df_result_agg])
    df_result_filter_aggs = pd.concat([df_result_filter_aggs, df_result_filter_agg])

df_result_aggs
df_result_filter_aggs.tail(10)


# **Let's show barplot**

# In[35]:


sns.barplot(df_result_aggs.index, df_result_aggs.upset_mean_all)
plt.title('probability of upset based on past result aggretation')
plt.show()


# **Merge upset probability**

# In[36]:


df_result = df_result.join(df_result_aggs, how='left', on="SeedDiff").join(df_result_filter_aggs, how='left', on='Seed_combi')
df_result["upset_prob"] = [m if c > 20 else a for a, m, c in zip(df_result["upset_mean_all"], df_result["upset_mean"], df_result["upset_count"])]
df_result.tail()


# **Scoring**

# In[37]:


valid = df_result[ (df_result["Season"]==(this_season-1)) ]
log_loss(valid['upset'], valid['upset_prob'])


# **manual smoothing**

# In[38]:


df_result_aggs.loc[0, 'upset_mean_all'] = 0.5
df_result_aggs.loc[6, 'upset_mean_all'] = (0.0 + df_result_aggs.loc[7, 'upset_mean_all']) / 2
df_result_aggs.loc[12, 'upset_mean_all'] = 0.0
df_result_aggs.loc[14, 'upset_mean_all'] = 0.0
df_result_aggs = df_result_aggs.rename(columns={'upset_mean_all': 'upset_prob_manually'})

sns.barplot(df_result_aggs.index, df_result_aggs.upset_prob_manually)
plt.title('probability of upset based on past result aggretation')
plt.show()


# In[39]:


valid.head(3)


# **Scoring**

# In[40]:


valid = df_result[ (df_result["Season"]==(this_season-1)) ]
valid = valid.join(df_result_aggs.drop("upset_count_all", axis=1), how='left', on='SeedDiff')
valid.fillna(0, inplace=True)
log_loss(valid['upset'], valid['upset_prob_manually'])


# **Clipping**
# 
# **for making the model conservative**

# In[41]:


log_loss(valid['upset'], np.clip(valid['upset_prob_manually'], 0.05, 0.95))


# ---

# ### Test
# 
# - Train: 2009~2018  
# - Test: 2019

# In[42]:


df_seed_2019 = df_seed[df_seed["Season"]==2019]


# In[43]:


this_season=2019
total_season=10

train = df_result[ (df_result["Season"]>=(this_season - total_season)) ]
print(train.shape)

df_result["Seed_combi"]=[str(ws)+'_'+str(ls) if ws<ls else str(ls)+'_'+str(ws) for ws, ls in zip(df_result["WSeed"], df_result["LSeed"])]
df_result.head()


# In[44]:


df_result_aggs = pd.DataFrame()
df_result_filter_aggs = pd.DataFrame()
for value in range(16):
    df_result_agg = df_result[df_result["SeedDiff"]==value].groupby("SeedDiff").agg({"upset": ["mean", "count"]})
    df_result_agg.columns = [col[0]+"_"+col[1]+"_"+"all" for col in df_result_agg.columns]
    df_result_filter_agg = df_result[df_result["SeedDiff"]==value].groupby("Seed_combi").agg({"upset": ["mean", "count"]})
    df_result_filter_agg.columns = [col[0]+"_"+col[1] for col in df_result_filter_agg.columns]
    if value==0:
        df_result_agg["upset_mean_all"] = 0.5
        df_result_filter_agg["upset_mean"] = 0.5
    df_result_aggs = pd.concat([df_result_aggs, df_result_agg])
    df_result_filter_aggs = pd.concat([df_result_filter_aggs, df_result_filter_agg])

df_result_aggs
df_result_filter_aggs.tail(10)


# In[45]:


sns.barplot(df_result_aggs.index, df_result_aggs.upset_mean_all)
plt.title('probability of upset based on past result aggretation')
plt.show()


# In[46]:


# manual smoothing
df_result_aggs.loc[0, 'upset_mean_all'] = 0.5
df_result_aggs.loc[10, 'upset_mean_all'] = (0.0 + df_result_aggs.loc[11, 'upset_mean_all']) / 2
df_result_aggs.loc[11, 'upset_mean_all'] = (0.0 + df_result_aggs.loc[15, 'upset_mean_all']) / 2
df_result_aggs.loc[12, 'upset_mean_all'] = (0.0 + df_result_aggs.loc[15, 'upset_mean_all']) / 2
df_result_aggs.loc[13, 'upset_mean_all'] = (0.0 + df_result_aggs.loc[15, 'upset_mean_all']) / 2
df_result_aggs.loc[14, 'upset_mean_all'] = (0.0 + df_result_aggs.loc[15, 'upset_mean_all']) / 2
df_result_aggs = df_result_aggs.fillna(-1)

sns.barplot(df_result_aggs.index, df_result_aggs.upset_mean_all)
plt.title('probability of upset based on past result aggretation')
plt.show()


# In[47]:


test = pd.read_csv("../input/WSampleSubmissionStage2.csv")
test = pd.DataFrame(np.array([ID.split("_") for ID in test["ID"]]), columns=["Season", "TeamA", "TeamB"], dtype=int)
test.head(3)

test = test.merge(df_seed_2019, how='left', left_on=["Season", "TeamA"], right_on=["Season", "TeamID"])
test = test.rename(columns={"seed_int": "TeamA_seed"}).drop("TeamID", axis=1)

test = test.merge(df_seed_2019, how='left', left_on=["Season", "TeamB"], right_on=["Season", "TeamID"])
test = test.rename(columns={"seed_int": "TeamB_seed"}).drop("TeamID", axis=1)

test['SeedDiff'] = np.abs(test.TeamA_seed - test.TeamB_seed)
test.head(3)


# In[48]:


test["Seed_combi"]=[str(a)+'_'+str(b) if a<b else str(b)+'_'+str(a) for a, b in zip(test["TeamA_seed"], test["TeamB_seed"])]
test.head()

test = test.join(df_result_aggs, how='left', on="SeedDiff").join(df_result_filter_aggs, how='left', on='Seed_combi').fillna(-1)
test["upset_prob"] = [m if c > 20 else a for a, m, c in zip(test["upset_mean_all"], test["upset_mean"], test["upset_count"])]

# convert upset_prob to win_prob
test["win_prob"] = [(1-upset_prob) if teamA<teamB else upset_prob if teamA>teamB else 0.5 
                    for teamA, teamB, upset_prob in zip(test['TeamA_seed'], test['TeamB_seed'], test['upset_prob'])]
test.tail()


# In[49]:


submit = pd.read_csv("../input/WSampleSubmissionStage2.csv")
submit["Pred"] = test['win_prob']
submit.to_csv("submission_agg_all_manually_noclip.csv", index=False)
submit.head()


# In[50]:


clipped_sub = np.clip(test['win_prob'], 0.05, 0.95)

submit = pd.read_csv("../input/WSampleSubmissionStage2.csv")
submit["Pred"] = clipped_sub
submit.to_csv("submission_agg_all_manually_cliped.csv", index=False)
submit.head()


# In[ ]:




