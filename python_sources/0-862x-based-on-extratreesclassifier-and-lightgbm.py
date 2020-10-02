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
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from mlxtend.classifier import StackingCVClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer


# In[ ]:


# obtain test set class distribution through probing the leaderboard 
class_weight = {1: 0.370530,
                2: 0.496810,
                3: 0.059365,
                4: 0.001037,
                5: 0.012958,
                6: 0.026873,
                7: 0.032427}


# In[ ]:


def balanced_accuracy_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred, sample_weight=[class_weight[label] for label in y_true])
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score, greater_is_better=True)
my_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


# In[ ]:


X_train = pd.read_csv("../input/learn-together/train.csv")
X_test = pd.read_csv("../input/learn-together/test.csv")


# In[ ]:


X_train.drop("Id", axis=1, inplace=True)
test_ID = X_test["Id"]
X_test.drop("Id", axis=1, inplace=True)


# In[ ]:


y_train = np.array(X_train['Cover_Type'])
X_train.drop('Cover_Type', axis=1, inplace=True)


# In[ ]:


assert np.all(X_train.loc[:, "Wilderness_Area1": "Wilderness_Area4"].sum(axis=1) == 1)
assert np.all(X_train.loc[:, "Soil_Type1": "Soil_Type40"].sum(axis=1) == 1)


# In[ ]:


num_train = X_train.shape[0]
all_data = pd.concat([X_train, X_test])


# In[ ]:


pca = PCA(n_components=0.95).fit(all_data)
pca_trans = pca.transform(all_data)
pca_trans.shape


# In[ ]:


for i in range(pca_trans.shape[1]):
    all_data["pca" + str(i)] = pca_trans[:, i]


# In[ ]:


all_data["Degree_To_Hydrology"] = (np.arctan((all_data["Vertical_Distance_To_Hydrology"] + np.finfo("float64").eps) /
                                             (all_data["Horizontal_Distance_To_Hydrology"] + np.finfo("float64").eps)))
all_data["Distance_to_Hydrology"] = (np.square(all_data["Vertical_Distance_To_Hydrology"]) +
                                               np.square(all_data["Vertical_Distance_To_Hydrology"]))


# In[ ]:


hillshade_cols = ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
all_data["Hillshade_mean"] = all_data[hillshade_cols].mean(axis=1)
all_data["Hillshade_std"] = all_data[hillshade_cols].std(axis=1)


# In[ ]:


cols = ["Horizontal_Distance_To_Hydrology",  "Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points"]
names = ["H", "R", "F"]
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        all_data["Horizontal_Distance_combination_" + names[i] + names[j] + "_1"] = all_data[cols[i]] + all_data[cols[j]]
        all_data["Horizontal_Distance_combination_" + names[i] + names[j] + "_2"] = (all_data[cols[i]] + all_data[cols[j]]) / 2
        all_data["Horizontal_Distance_combination_" + names[i] + names[j] + "_3"] = all_data[cols[i]] - all_data[cols[j]]
        all_data["Horizontal_Distance_combination_" + names[i] + names[j] + "_4"] = np.abs(all_data[cols[i]] - all_data[cols[j]])
all_data["Horizontal_Distance_mean"] = all_data[cols].mean(axis=1)


# In[ ]:


all_data["Elevation_Hydrology_1"] = all_data["Elevation"] + all_data["Vertical_Distance_To_Hydrology"]
all_data["Elevation_Hydrology_2"] = all_data["Elevation"] - all_data["Vertical_Distance_To_Hydrology"]


# In[ ]:


X_train = all_data[:num_train]
X_test = all_data[num_train:]


# In[ ]:


clf = ExtraTreesClassifier(n_estimators=250, random_state=0, n_jobs=-1)
scores = cross_validate(clf, X_train, y_train, cv=my_cv,
                        fit_params={"sample_weight":[class_weight[label] for label in y_train]},
                        scoring=balanced_accuracy_scorer, return_train_score=True)
print(np.mean(scores["train_score"]), np.std(scores["train_score"]))
print(np.mean(scores["test_score"]), np.std(scores["test_score"]))


# In[ ]:


clf = lgb.LGBMClassifier(n_estimators=600, random_state=0, n_jobs=-1)
scores = cross_validate(clf, X_train, y_train, cv=my_cv,
                        fit_params={"sample_weight":[class_weight[label] for label in y_train]},
                        scoring=balanced_accuracy_scorer, return_train_score=True)
print(np.mean(scores["train_score"]), np.std(scores["train_score"]))
print(np.mean(scores["test_score"]), np.std(scores["test_score"]))


# In[ ]:


clf1 = ExtraTreesClassifier(n_estimators=250, random_state=0, n_jobs=-1)
clf2 = lgb.LGBMClassifier(n_estimators=600, random_state=0, n_jobs=-1)
clf = StackingCVClassifier(classifiers=[clf1, clf2],
                           meta_classifier=xgb.XGBClassifier(n_estimators=50, random_state=0, n_jobs=-1),
                           cv=my_cv, random_state=0, use_probas=True, use_features_in_secondary=True)
clf.fit(X_train, y_train, sample_weight=[class_weight[label] for label in y_train])
pred = clf.predict(X_test)
submission = pd.DataFrame({'Id':test_ID, 'Cover_Type':pred},
                          columns=['Id', 'Cover_Type'])
submission.to_csv("submission.csv", index=False)

