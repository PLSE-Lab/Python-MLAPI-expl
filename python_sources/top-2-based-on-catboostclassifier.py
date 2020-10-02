#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import catboost
import gensim


# In[ ]:


X_train = pd.read_csv("../input/sf-crime/train.csv")
X_test = pd.read_csv("../input/sf-crime/test.csv")


# In[ ]:


X_train[:5]


# In[ ]:


print(X_train.duplicated().sum())
X_train.drop_duplicates(inplace=True)
assert X_train.duplicated().sum() == 0


# In[ ]:


X_test[:5]


# In[ ]:


y_train = X_train['Category']
X_train_description = X_train['Descript']
X_train_resolution = X_train['Resolution']
X_train.drop(["Category", "Descript", "Resolution"], axis=1, inplace=True)


# In[ ]:


test_ID = X_test["Id"]
X_test.drop("Id", axis=1, inplace=True)


# In[ ]:


X_train[:5]


# In[ ]:


X_test[:5]


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_train.value_counts()


# In[ ]:


le = LabelEncoder()
y_train = le.fit_transform(y_train)
print(le.classes_)


# In[ ]:


num_train = X_train.shape[0]
all_data = pd.concat((X_train, X_test), ignore_index=True)


# In[ ]:


date = pd.to_datetime(all_data['Dates'])
all_data['year'] = date.dt.year
all_data['month'] = date.dt.month
all_data['day'] = date.dt.day
all_data['hour'] = date.dt.hour
all_data['minute'] = date.dt.minute
all_data['special_time'] = all_data['minute'].isin([0, 30]).astype(int)
# all_data['second'] = date.dt.second  # all zero
all_data["n_days"] = (date - date.min()).apply(lambda x: x.days)
all_data.drop("Dates", axis=1, inplace=True)


# In[ ]:


all_data["DayOfWeek"].value_counts()


# In[ ]:


all_data["PdDistrict"].value_counts()


# In[ ]:


sentences = []
for s in all_data["Address"]:
    sentences.append(s.split(" "))
address_model = gensim.models.Word2Vec(sentences, min_count=1)
encoded_address = np.zeros((all_data.shape[0], 100))
for i in range(len(sentences)):
    for j in range(len(sentences[i])):
        encoded_address[i] += address_model.wv[sentences[i][j]]
    encoded_address[i] /= len(sentences[i])


# In[ ]:


all_data['block'] = all_data["Address"].str.contains("block", case=False)
all_data.drop("Address", axis=1, inplace=True)


# In[ ]:


print(all_data["X"].min(), all_data["X"].max())
print(all_data["Y"].min(), all_data["Y"].max())


# In[ ]:


X_median = all_data[all_data["X"] < -120.5]["X"].median()
Y_median = all_data[all_data["Y"] < 90]["Y"].median()
all_data.loc[all_data["X"] >= -120.5, "X"] = X_median
all_data.loc[all_data["Y"] >= 90, "Y"] = Y_median


# In[ ]:


print(all_data["X"].min(), all_data["X"].max())
print(all_data["Y"].min(), all_data["Y"].max())


# In[ ]:


all_data["X+Y"] = all_data["X"] + all_data["Y"]
all_data["X-Y"] = all_data["X"] - all_data["Y"]
all_data["XY30_1"] = all_data["X"] * np.cos(np.pi / 6) + all_data["Y"] * np.sin(np.pi / 6)
all_data["XY30_2"] = all_data["Y"] * np.cos(np.pi / 6) - all_data["X"] * np.sin(np.pi / 6)
all_data["XY60_1"] = all_data["X"] * np.cos(np.pi / 3) + all_data["Y"] * np.sin(np.pi / 3)
all_data["XY60_2"] = all_data["Y"] * np.cos(np.pi / 3) - all_data["X"] * np.sin(np.pi / 3)
all_data["XY1"] = (all_data["X"] - all_data["X"].min()) ** 2 + (all_data["Y"] - all_data["Y"].min()) ** 2
all_data["XY2"] = (all_data["X"].max() - all_data["X"]) ** 2 + (all_data["Y"] - all_data["Y"].min()) ** 2
all_data["XY3"] = (all_data["X"] - all_data["X"].min()) ** 2 + (all_data["Y"].max() - all_data["Y"]) ** 2
all_data["XY4"] = (all_data["X"].max() - all_data["X"]) ** 2 + (all_data["Y"].max() - all_data["Y"]) ** 2
all_data["XY5"] = (all_data["X"] - X_median) ** 2 + (all_data["Y"] - Y_median) ** 2
pca = PCA(n_components=2).fit(all_data[["X", "Y"]])
XYt = pca.transform(all_data[["X", "Y"]])
all_data["XYpca1"] = XYt[:, 0]
all_data["XYpca2"] = XYt[:, 1]
# n_components selected by aic/bic
clf = GaussianMixture(n_components=150, covariance_type="diag",
                      random_state=0).fit(all_data[["X", "Y"]])
all_data["XYcluster"] = clf.predict(all_data[["X", "Y"]])


# In[ ]:


categorical_features = ["DayOfWeek", "PdDistrict", "block", "special_time", "XYcluster"]
ct = ColumnTransformer(transformers=[("categorical_features", OrdinalEncoder(), categorical_features)],
                       remainder="passthrough")
all_data = ct.fit_transform(all_data)


# In[ ]:


all_data = np.hstack((all_data, encoded_address))


# In[ ]:


X_train = all_data[:num_train]
X_test = all_data[num_train:]


# In[ ]:


clf = catboost.CatBoostClassifier(n_estimators=5000, learning_rate=0.05,
                                  cat_features=np.arange(len(categorical_features)),
                                  random_seed=0, task_type="GPU", verbose=50)


# In[ ]:


clf.fit(X_train, y_train)
prob = clf.predict_proba(X_test)


# In[ ]:


submission = pd.DataFrame(np.c_[test_ID, prob], columns=["Id"] + list(le.classes_))
submission["Id"] = submission["Id"].astype(int)
submission.to_csv("submission.csv", index=False)

