#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# load the data
df = pd.read_csv("../input/cardio_train.csv",sep=';')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


corr_matrix = df.corr()
corr_matrix["cardio"].sort_values(ascending=False)


# We can see that the feature that has the most effect on the prediction is age. ID has no relevance so we can drop this from our training set. Some of the other features don't appear to have much effect, particularly the 'subjective' ones. We may consider dropping these also. Since age is important I will plot a barchart of age in years. I will also add BMI as we have both height and weight

# In[ ]:


df['BMI'] = df['weight'] / (df['height']/100)**2
corr_matrix = df.corr()
corr_matrix["cardio"].sort_values(ascending=False)


# In[ ]:


years = (df['age'] / 365).round().astype('int')
pd.crosstab(years, df.cardio).plot(kind='bar', figsize=(12,8))
plt.title('Cardio by age')
plt.legend(['Does not have CVD', 'Has CVD'])
plt.show()


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit

s = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
for train_index, test_index in s.split(df, df["cardio"]):
    train_set = df.loc[train_index]
    test_set = df.loc[test_index]


# In[ ]:


training = train_set.copy()


# In[ ]:


training_data = training.drop(["cardio", "id", "alco"], axis=1)
cardio_labels = training["cardio"].copy()


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(training_data)


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = [
{'n_estimators': [100, 500, 1000], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [100, 500, 100], 'max_features': [2, 4, 6]},
]

clf = RandomForestClassifier()

grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(data_scaled, cardio_labels)


# In[ ]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)


# In[ ]:


grid_search.best_params_


# In[ ]:


from sklearn.model_selection import cross_val_score

clf = RandomForestClassifier(max_features=4, n_estimators=500)
scores = cross_val_score(clf, data_scaled, cardio_labels, cv=5, scoring="accuracy")
print("Scores: ", scores)
print("Mean: ", scores.mean())


# In[ ]:


from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict

training_pred = cross_val_predict(clf, data_scaled, cardio_labels, cv=5)
print("Precision: ", precision_score(cardio_labels, training_pred))
print("Recall: ", recall_score(cardio_labels, training_pred))
print("F1 Score: ", f1_score(cardio_labels, training_pred))


# In[ ]:


cols = ["cardio", "id", "alco"]
X_test = test_set.drop(cols, axis=1)
Y_test = test_set["cardio"].copy()

X_test["BMI"] = X_test["weight"] / (X_test["height"]/100)**2
X_test_prepared = scaler.fit_transform(X_test)

clf.fit(data_scaled, cardio_labels)
final_predictions = clf.score(X_test_prepared, Y_test)
print(final_predictions)


# In[ ]:




