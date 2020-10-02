#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


# # read data

# In[ ]:


df = pd.read_json(open("../input/train.json", "r"))


# In[ ]:


df.index = range(len(df.index))
df.head()


# # naive feature engineering

# In[ ]:


l = [el for row in df['features'].as_matrix() for el in row ]
l


# In[ ]:


vectorizer = CountVectorizer(min_df=2)
vectorizer.fit(l)


# In[ ]:


df["num_photos"] = df["photos"].apply(len)
df["num_features"] = df["features"].apply(len)
df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
df['features_bow'] = df["features"].apply(lambda x: vectorizer.transform(x))
df["created"] = pd.to_datetime(df["created"])
df["created_year"] = df["created"].dt.year
df["created_month"] = df["created"].dt.month
df["created_day"] = df["created"].dt.day


# In[ ]:


feats = df['features'][:]
s = feats.to_string()
feats = s.strip("[]")
df = feats.split(',', expand=True)
df = df.applymap(lambda x: x.replace("'", '').strip())
l = df.values.flatten()
print (l.tolist())


# In[ ]:


vectorizer.fit(df['description'])


# In[ ]:


vectorizer.transform(df['description'])


# In[ ]:


df['description_words_bow'] = vectorizer.fit(df['features'])


# In[ ]:


num_feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_description_words",
             "created_year", "created_month", "created_day"]
X = df[num_feats]
y = df["interest_level"]


# In[ ]:


train = [1, 2, 4]
X.loc[train, :]


# 

# In[ ]:


type(X)
type(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)
type(X_train)
type(y_train)


# In[ ]:


clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, y_train)
y_val_pred = clf.predict_proba(X_val)
log_loss(y_val, y_val_pred)


# # make prediction

# In[ ]:


df = pd.read_json(open("../input/test.json", "r"))
df["num_photos"] = df["photos"].apply(len)
df["num_features"] = df["features"].apply(len)
df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
df["created"] = pd.to_datetime(df["created"])
df["created_year"] = df["created"].dt.year
df["created_month"] = df["created"].dt.month
df["created_day"] = df["created"].dt.day
X = df[num_feats]

y = clf.predict_proba(X)


# In[ ]:


labels2idx = {label: i for i, label in enumerate(clf.classes_)}
labels2idx


# In[ ]:


sub = pd.DataFrame()
sub["listing_id"] = df["listing_id"]
for label in ["high", "medium", "low"]:
    sub[label] = y[:, labels2idx[label]]
sub.to_csv("submission_rf.csv", index=False)


# In[ ]:




