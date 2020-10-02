#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data_frame = pd.read_csv("../input/fraud-rate/main_data.csv")


# In[ ]:


data_frame.head()


# In[ ]:


data_frame.shape


# In[ ]:


data_frame.columns


# In[ ]:


X = data_frame.drop(["PotentialFraud"], axis=1)
Y = data_frame["PotentialFraud"]

X_values = X.values
Y_values = Y.values


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_values, 
                                                    Y_values, 
                                                    test_size = 0.25)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

RANDOM_FOREST = RandomForestClassifier()


# In[ ]:


RANDOM_FOREST.fit(X_train, Y_train)


# In[ ]:


importances = RANDOM_FOREST.feature_importances_

importances = RANDOM_FOREST.feature_importances_
std = np.std([tree.feature_importances_ for tree in RANDOM_FOREST.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print('%s:  %3f' %(data_frame.columns[f], importances[indices[f]]))


# In[ ]:


plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")

plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[ ]:


X = data_frame.drop(["PotentialFraud"], axis=1)
Y = data_frame["PotentialFraud"]

X_values = X.values
Y_values = Y.values

X_train, X_test, Y_train, Y_test = train_test_split(X_values, 
                                                    Y_values, 
                                                    test_size = 0.25)


# In[ ]:


RANDOM_FOREST = RandomForestClassifier()


# In[ ]:


RANDOM_FOREST.fit(X_train, Y_train)


# In[ ]:


Y_predict = RANDOM_FOREST.predict(X_test)


# In[ ]:


frauded_cards = data_frame[data_frame["PotentialFraud"] == 1]
safe_cards = data_frame[data_frame["PotentialFraud"] == 0]

part_of_frauded_cards = 100 * len(frauded_cards)/len(data_frame)
min_accurancy = 100 - part_of_frauded_cards
round(min_accurancy, 3)


# In[ ]:


from sklearn.metrics import roc_auc_score

r_accuracy = roc_auc_score(Y_predict, Y_test)
round(r_accuracy, 3)


# In[ ]:


RANDOM_FOREST = RandomForestClassifier(max_depth = 10, 
                                       n_estimators = 85, 
                                       max_leaf_nodes = 60,
                                       min_samples_split = 3,
                                       min_samples_leaf = 10,
                                       max_features = "log2",
                                       criterion = "entropy")


# In[ ]:


RANDOM_FOREST.fit(X_train, Y_train)


# In[ ]:


Y_predict = RANDOM_FOREST.predict(X_test)


# In[ ]:


roc_auc_score(Y_predict, Y_test)

