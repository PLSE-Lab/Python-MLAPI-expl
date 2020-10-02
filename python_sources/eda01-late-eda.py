#!/usr/bin/env python
# coding: utf-8

# # ELO data exploration
# 
# [TOC]
# 
# this is my late EDA for this competition. I made it for myself, but perhaps other people will find it useful. I am still playing with it so please refresh from time to time

# ### imports and listing files

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib_venn import venn2, venn3
import squarify
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Train fields description

# In[ ]:


pd.read_excel("../input/Data_Dictionary.xlsx", sheet = 0, header = 2)


# ## peek at train and train shape

# In[ ]:


train  = pd.read_csv("../input/train.csv")
print(train.shape)

train.head()


# Target distribution

# In[ ]:


sns.distplot(train.target)


# peek at test

# In[ ]:


test = pd.read_csv("../input/test.csv")
print(test.shape)
test.head()


# ## Let's check if any card_ids overlap between train and test

# In[ ]:


venn2([set(train.card_id), set(test.card_id)])


# conclusion: they don't

# ## Just in case, let's check for any dependency of the target on data order or card_id

# In[ ]:


sns.jointplot(train.index.values, train.target)


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


sns.jointplot(LabelEncoder().fit_transform(train.card_id), train.target)


# conclusion: no dependency

# In[ ]:


train["is_test"] = np.int8(0)
test["is_test"] = np.int8(1)
test["target"] = np.NaN
train_test = pd.concat([train,test], ignore_index=True, sort=True, axis = 0)
train_test["first_active_month"] = pd.to_datetime(train_test.first_active_month)
train_test["mon"] = train_test.first_active_month.dt.month
train_test["year"] = train_test.first_active_month.dt.year

print(train_test.shape)
train_test.head()


# In[ ]:


del train
del test


# ## Let's have a look at the "feature_N" categoricals. Is the proportion of different categories different for train and test?

# In[ ]:


pd.merge(train_test.groupby(["is_test", "feature_1"])["card_id"].count().reset_index(),
                     train_test.groupby(["is_test"]).count().reset_index(), on = "is_test" )


# In[ ]:


def compare_cat_pct_counts(df, col_name, compare_col_name = "is_test", ind_col ="card_id"):
    cnt_df = pd.merge(df.groupby([compare_col_name, col_name])[ind_col].count().reset_index(),
                      df.groupby([compare_col_name])[ind_col].count().reset_index(), on = "is_test" )
    cnt_df["cnt"] = cnt_df[ind_col + "_x"] / cnt_df[ind_col + "_y"] 
    sns.barplot(col_name, "cnt", hue = compare_col_name, data = cnt_df)
     


# In[ ]:


compare_cat_pct_counts(train_test, "feature_1")


# In[ ]:


compare_cat_pct_counts(train_test, "feature_2")


# In[ ]:


compare_cat_pct_counts(train_test, "feature_3")


# conclusion: same proportion of categories for train and test

# ## Comparing target stats for different categories

# In[ ]:


plt.subplots(figsize=(15,6))
plt.subplot(131)
sns.boxplot(train_test["feature_1"], train_test.target)
plt.subplot(132)
sns.boxplot(train_test["feature_2"], train_test.target)
plt.subplot(133)
sns.boxplot(train_test["feature_3"], train_test.target)


# conclusion: these features look useless. What do they mean?

# ## Let's look at the first_active_month

# In[ ]:


compare_cat_pct_counts(train_test, "mon")


# In[ ]:


compare_cat_pct_counts(train_test, "year")


# In[ ]:


plt.subplots(figsize=(15,6))
plt.subplot(121)
sns.boxplot(train_test.mon, train_test.target)
plt.subplot(122)
sns.boxplot(train_test.year, train_test.target)


# In[ ]:


pd.read_excel("../input/Data_Dictionary.xlsx", 1)


# In[ ]:


hist_trans = pd.read_csv("../input/historical_transactions.csv")
print(hist_trans.shape)
hist_trans.head()


# In[ ]:


plt.figure(figsize=(20,10))
print(hist_trans.card_id.nunique())
venn3([set(hist_trans.card_id.unique()), set(train_test.query("is_test == 0").card_id), 
                                             set(train_test.query("is_test == 1").card_id)])


# So, all card_ids present in transaction history, are either from train or from test

# ## Number of transactions per card

# In[ ]:


hist_trans.card_id.value_counts().head(10)


# Looks like some cards have a lot of transactions

# In[ ]:


tmp_df = pd.merge(hist_trans.card_id.value_counts().reset_index(), train_test, left_on="index", right_on = "card_id")
plt.subplots(figsize = (14, 6))
plt.subplot(131)
sns.boxplot("is_test", "card_id_x", data=tmp_df )
plt.subplot(132)
sns.distplot(tmp_df.query("is_test == 0")["card_id_x"],  color = "blue" )
sns.distplot(tmp_df.query("is_test == 1")["card_id_x"],  color = "green")
plt.subplot(133)
sns.distplot(np.log(tmp_df.query("is_test == 0")["card_id_x"]),  color = "blue" )
sns.distplot(np.log(tmp_df.query("is_test == 1")["card_id_x"]),  color = "green")
del tmp_df


# Count of transactions has lognormal distribution, with some cards have an extreme number of transactions. However test set happens to have some cards that have even bigger number of transactions (5k vs 3k highest in train)
# 
# By now, I am convinced that train and test split is absolutely random and are selected from the same population and there is no need to compare the two sets further.

# In[ ]:


# a helper function
def plot_cat_treemap(df, col_name, title = None):
    cnts = df[col_name].value_counts()
    cmap = matplotlib.cm.Spectral
    mini=min(cnts)
    maxi=max(cnts)
    norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
    colors = [cmap(norm(value)) for value in cnts]

    
    squarify.plot(sizes = cnts, label = cnts.index.values, value = cnts, color = colors)
    plt.axis("off")
    plt.title(title)


# ## authorized_flag

# ### number of authorized / non authorized and target

# In[ ]:


card_auth_agg = pd.merge(train_test, 
                         hist_trans.groupby(["card_id", "authorized_flag"])["city_id"].count().unstack(level=-1),
                         on = "card_id").rename(columns = {"N":"cnt_unauthorized", "Y":"cnt_authorized"}).fillna(0)

card_auth_agg["cnt_trans"] = card_auth_agg["cnt_unauthorized"] + card_auth_agg["cnt_authorized"] 
#plt.subplot(311)
sns.jointplot(card_auth_agg.target, card_auth_agg.cnt_trans)


# In[ ]:


sns.jointplot(card_auth_agg.target, card_auth_agg.cnt_unauthorized)


# In[ ]:


sns.jointplot(card_auth_agg.target, card_auth_agg.cnt_authorized)


# In[ ]:


del card_auth_agg


# conclusion: seems like for a very high number of transactions (independent on whether they were unathorized or not, the loyalty score is 0). For lower number of transactions, the loyalty score is all over

# ## city_id

# ### number of historical transactions per city

# In[ ]:


plt.figure(figsize=(20,14))
plot_cat_treemap(hist_trans, "city_id", "City ID")


# ### Purchase amt by city

# In[ ]:


city_agg = hist_trans.groupby("city_id")["purchase_amount"].agg(["mean", "min", "max", "median"])


# In[ ]:


city_agg.sort_values("mean", ascending=False).head()


# In[ ]:


city_agg.sort_values("max", ascending=False).head()


# Some purchase amounts are huge, especially knowing they are already scaled

# In[ ]:


del city_agg


# ### Number of citites per persona

# In[ ]:


city_card_agg  = pd.merge(hist_trans.groupby("card_id")["city_id"].agg(["nunique"]).reset_index(), train_test, on="card_id").rename(columns={"nunique":"city_count"})


# In[ ]:


plt.subplot(111)
sns.distplot(city_card_agg.loc[~city_card_agg.target.isnull(), "city_count"], color = "blue")
sns.distplot(city_card_agg.loc[city_card_agg.target.isnull(), "city_count"], color = "green")


# Conclusion: most of the cards conduct transactions from a few cities. there are some, that have 70-100 different cities in their transaction history.

# In[ ]:


del city_card_agg


# In[ ]:


gc.collect()


# ### Target by the most frequent city

# ## state_id

# In[ ]:


plt.figure(figsize=(20,14))
plot_cat_treemap(hist_trans, "state_id", "State ID")


# In[ ]:


stat_city_df = hist_trans.groupby(["state_id", "city_id"])["card_id"].count().reset_index().sort_values("card_id", ascending=False)


# In[ ]:


stat_city_df.query("state_id == -1")


# In[ ]:


stat_city_df.query("city_id == -1")


# In[ ]:


stat_city_df.query("city_id == 179")


# In[ ]:


stat_city_df.query("city_id == 75")


# In[ ]:


del stat_city_df


# In[ ]:




