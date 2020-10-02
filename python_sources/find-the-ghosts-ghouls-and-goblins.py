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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


ghosts_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

ghosts_df.head()
#All data seems normalized. Should probably do the same for colors / types


# In[ ]:


ghosts_df.info()
print("--------------------")
test_df.info()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

sns.pairplot(ghosts_df, hue="type")


# In[ ]:


ghosts_df = ghosts_df.drop("id",axis=1)
# convert string-values to floats for predicting on them
def remap_as_int(dataframe, indice, value, newValue):
    print("value={} reassigned to id={}".format(value, newValue))
    dataframe.loc[dataframe[indice] == value, indice] = newValue
    
def enumerate_list_unique(dataframe, indice):
    return list(enumerate(np.unique(dataframe[indice])))

monsters_list = enumerate_list_unique(ghosts_df, 'type')
monsters_list = [(0, 'Ghost'),(1, 'Ghoul'),(2, 'Goblin')]
print("All known types of monsters = {}".format(np.unique(ghosts_df['type'])))
for index, monster in monsters_list:
      remap_as_int(ghosts_df, 'type', monster, index)

#colors_list = enumerate_list_unique(ghosts_df, 'color')
colors_list = [(0, 'black'),(0.2, 'blood'),(0.4, 'blue'),(0.6, 'clear'),(0.8, 'green'),(1, 'white')]
print("All known colors of monsters = {}".format(np.unique(ghosts_df['color'])))
for index, color in colors_list:
      remap_as_int(ghosts_df, 'color', color, index)


# Candidates for feature engineering / combination:
# hair_length * has_soul
# hair_length * bone_length
# bone_length * has_soul
# 
# Ideas to proceed:
# 1. use only engineered features to train the model
# 2. add engineered features to dataframe and train on everything. Might confuse the model more than it helps?

# In[ ]:


#Create some new variables.
hair_soul = ghosts_df['hair_length'] * ghosts_df['has_soul']
#print(hair_soul)
temp_df = pd.concat([hair_soul, ghosts_df['type']], axis=1, keys=['hair_soul', 'type'])
sns.lmplot("hair_soul", "type", data=temp_df, hue='type', fit_reg=False)


# In[ ]:


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

#X_train, X_cv, y_train, y_cv = train_test_split(ghosts_df.drop("type",axis=1), ghosts_df["type"], test_size=0.3, random_state=0)
X_train = ghosts_df.drop("type",axis=1)
y_train = ghosts_df["type"]

#remap colors to ints for test-set
for index, color in colors_list:
    remap_as_int(test_df, 'color', color, index)
test_df['color'] = test_df['color'].astype(float)

test_df.drop("color",axis=1)
X_test  = test_df.drop("id",axis=1).copy()


# In[ ]:


logreg = LogisticRegression()

logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_cv, y_cv)


# In[ ]:


svc = SVC()

svc.fit(X_train, y_train)

Y_pred = svc.predict(X_test)

svc.score(X_cv, y_cv)


# In[ ]:


random_forest = RandomForestClassifier(random_state=1, n_estimators=20, min_samples_split=2, min_samples_leaf=1)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_cv, y_cv)

#print(Y_pred)


# In[ ]:


#knn = KNeighborsClassifier(n_neighbors = 3)

#knn.fit(X_train, y_train)

#Y_pred = knn.predict(X_test)

#knn.score(X_cv, y_cv)


# In[ ]:


#for alpha in [0.00001, 0.01, 0.1, 0.5, 1]:
neural_net = MLPClassifier(solver='lbfgs', alpha=1, hidden_layer_sizes=(15,6), random_state=1)
neural_net.fit(X_train, y_train)

Y_pred = neural_net.predict(X_test)

#print(neural_net.score(X_cv, y_cv))


# In[ ]:


#remap to monster labels
Y_pred = Y_pred.astype(object)
Y_pred[Y_pred == monsters_list[0][0]] = monsters_list[0][1]
Y_pred[Y_pred == monsters_list[1][0]] = monsters_list[1][1]
Y_pred[Y_pred == monsters_list[2][0]] = monsters_list[2][1]


# In[ ]:



submission = pd.DataFrame({
        "id": test_df["id"],
        "type": Y_pred
    })
submission.to_csv('new.csv', index=False)

