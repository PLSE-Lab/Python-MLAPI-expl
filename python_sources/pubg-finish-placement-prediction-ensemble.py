#!/usr/bin/env python
# coding: utf-8

# #  PUBG Finish Placement Prediction (ENSEMBLE)
# In this ensemble model, I am not using outputs from my own models.
# Instead, I'll be using outputs from several models:
# - [MLP](https://www.kaggle.com/harshitsheoran/mlp-and-fe)
# - [LightGBM](https://www.kaggle.com/chocozzz/lightgbm-baseline)
# - [PyTorch](https://www.kaggle.com/ceshine/pytorch-baseline-model)
# 
# 

# ## Import dependencies

# In[ ]:


import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))


# In[ ]:


df_sub1 = pd.read_csv("../input/pytorch-baseline-model/submission_raw.csv")
df_sub2 = pd.read_csv("../input/lightgbm-baseline/submission_adjusted.csv")
df_sub3 = pd.read_csv("../input/mlp-and-fe/submission.csv")
df_test = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")


# In[ ]:


# restore some columns
df_sub1 = df_sub1.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id",how="left")


# In[ ]:


# sort, rank and assign adjusted ratio
df_sub1_group = df_sub1.groupby(["matchId", "groupId"]).first().reset_index()
df_sub1_group["rank"] = df_sub1_group.groupby(["matchId"])["winPlacePerc"].rank()
df_sub1_group = df_sub1_group.merge(
    df_sub1_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(), 
    on="matchId", how="left")
df_sub1_group["adjusted_perc"] = (df_sub1_group["rank"] - 1) / (df_sub1_group["numGroups"] - 1)
df_sub1 = df_sub1.merge(df_sub1_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
df_sub1["winPlacePerc"] = df_sub1["adjusted_perc"]


# In[ ]:


# Deal with edge cases
df_sub1.loc[df_sub1.maxPlace == 0, "winPlacePerc"] = 0
df_sub1.loc[df_sub1.maxPlace == 1, "winPlacePerc"] = 1


# In[ ]:


# Align with maxPlace
subset = df_sub1.loc[df_sub1.maxPlace > 1]
gap = 1.0 / (subset.maxPlace.values - 1)
new_perc = np.around(subset.winPlacePerc.values / gap) * gap
df_sub1.loc[df_sub1.maxPlace > 1, "winPlacePerc"] = new_perc


# In[ ]:


# Edge case
df_sub1.loc[(df_sub1.maxPlace > 1) & (df_sub1.numGroups == 1), "winPlacePerc"] = 0
assert df_sub1["winPlacePerc"].isnull().sum() == 0


# In[ ]:


df_sub1["winPlacePerc"] = (df_sub1["winPlacePerc"] + df_sub2["winPlacePerc"] + df_sub3["winPlacePerc"]) / 3
df_sub1 = df_sub1[["Id", "winPlacePerc"]]


# In[ ]:


df_test = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")


# In[ ]:


# restore some columns
df_sub1 = df_sub1.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id",how="left")


# In[ ]:


# sort, rank and assign adjusted ratio
df_sub1_group = df_sub1.groupby(["matchId", "groupId"]).first().reset_index()
df_sub1_group["rank"] = df_sub1_group.groupby(["matchId"])["winPlacePerc"].rank()
df_sub1_group = df_sub1_group.merge(
    df_sub1_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(), 
    on="matchId", how="left")
df_sub1_group["adjusted_perc"] = (df_sub1_group["rank"] - 1) / (df_sub1_group["numGroups"] - 1)
df_sub1 = df_sub1.merge(df_sub1_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
df_sub1["winPlacePerc"] = df_sub1["adjusted_perc"]


# In[ ]:


# Deal with edge cases
df_sub1.loc[df_sub1.maxPlace == 0, "winPlacePerc"] = 0
df_sub1.loc[df_sub1.maxPlace == 1, "winPlacePerc"] = 1


# In[ ]:


# Align with maxPlace
subset = df_sub1.loc[df_sub1.maxPlace > 1]
gap = 1.0 / (subset.maxPlace.values - 1)
new_perc = np.around(subset.winPlacePerc.values / gap) * gap
df_sub1.loc[df_sub1.maxPlace > 1, "winPlacePerc"] = new_perc


# In[ ]:


# Edge case
df_sub1.loc[(df_sub1.maxPlace > 1) & (df_sub1.numGroups == 1), "winPlacePerc"] = 0
assert df_sub1["winPlacePerc"].isnull().sum() == 0


# In[ ]:


df_sub1["winPlacePerc"] = df_sub1["winPlacePerc"]


# In[ ]:


df_sub1[["Id", "winPlacePerc"]].to_csv("submission.csv", index=False)

