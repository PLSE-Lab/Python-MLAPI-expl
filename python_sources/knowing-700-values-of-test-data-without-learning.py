#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_json("../input/train.json")
test  = pd.read_json("../input/test.json")
sample_sub = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


from sklearn.preprocessing import MultiLabelBinarizer
def count_ingredients():
    mlb = MultiLabelBinarizer()
    df = pd.DataFrame(mlb.fit_transform(train['ingredients']),columns=mlb.classes_, index=train.index)
    df = pd.concat([train,df],axis=1)
    df = df.drop("ingredients",axis =1)
    df = df.drop("id",axis =1)
    df = df.groupby('cuisine').sum()
    return df


# In[ ]:


df = count_ingredients()
df.head()


# **Checking items which are used by  only one cuisine but are minimum used thrice **

# In[ ]:


dd = pd.DataFrame(np.where(df>0, 1, 0),columns = df.columns)
for_only_one = dd.drop([col for col, val in dd.sum().iteritems() if val >=2], axis=1).columns
for_only_one_final = df[for_only_one].drop([col for col, val in df[for_only_one].sum().iteritems() if val <3], axis=1).columns


# In[ ]:


only_one = df[for_only_one_final].transpose()
df[for_only_one_final]


# In[ ]:


df_only_one = pd.DataFrame(columns=train.cuisine.unique())

def one(cuisine):
    for i in cuisine:
        df_only_one[i] = [only_one[only_one[i]>0].loc[:,i].index.tolist()]
    return df_only_one.transpose()


# In[ ]:


df_only_one = one(train.cuisine.unique())
df_only_one.head(3)


# In[ ]:


common_ingredients = df.sum(axis =0).sort_values(ascending=False, axis=0).iloc[:10]
common_ingredients


# **Checking if I can drop values which are used in all cuisines**
# 
# Difference between min and max of any item is greater then min. 91% . So dropping common items is not good idea.

# In[ ]:


for_all = dd.drop([col for col, val in dd.sum().iteritems() if val <20], axis=1).columns
for_all_final = df[for_all].drop([col for col, val in df[for_all].sum().iteritems() if val <1], axis=1).columns


# In[ ]:


((df[for_all_final].max() - df[for_all_final].min())/df[for_all_final].max()*100 < 92).unique()


# **Seeing how many unique items of training data is unique for test data**

# In[ ]:


def jojo(list):
  new_words = [word for word in list if word in df[for_only_one_final].columns]
  return len(new_words)


# In[ ]:


test['jojo'] = test[["ingredients"]].apply(lambda x: jojo(*x), axis=1)


# In[ ]:


def new_data(list):
  new_words = [word for word in list if word in k]
  if len(new_words) > 0:
    return j
  else:
    return


# In[ ]:


for j in df_only_one.index:
  k = df_only_one.loc[j,:]
  k = sum(k,[])
  test[j] = test[test["jojo"]>0][["ingredients"]].apply(lambda x: new_data(*x), axis=1)


# In[ ]:


test["cuisine"] = test[(test["jojo"]>0)].iloc[:,3:-1].apply(lambda x: ' '.join(x.dropna()), axis=1)


# In[ ]:


test = test.drop(test.iloc[:,2:-1],axis =1)


# In[ ]:


test = test.drop("ingredients",axis = 1)
test_to_train = test.dropna()
test.head()


# **Filled unknown values with "italian"**

# In[ ]:


test = test.fillna("italian")


# In[ ]:


test.to_csv("submission.csv",index = False)


# In[ ]:




