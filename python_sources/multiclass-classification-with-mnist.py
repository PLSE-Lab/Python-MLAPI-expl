#!/usr/bin/env python
# coding: utf-8

# # GLOBAL VARIABLES

# In[ ]:


PATH_DATASET = "/kaggle/input/mnist-in-csv/"
TRAINING_DATASET = "mnist_train.csv"
TESTING_DATASET = "mnist_test.csv"


# # IMPORT DATA

# In[ ]:


import pandas as pd
import os

def load_dataset(filename, path_dataset=PATH_DATASET):
    print("LOADED: " + filename)
    return pd.read_csv(os.path.join(path_dataset, filename))
    
train = load_dataset(TRAINING_DATASET)
test = load_dataset(TESTING_DATASET)


# # EXPLORING DATA

# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.head()


# In[ ]:


test.head()


# > As you can see, both the training set and testing are loaded as panda DataFrame with the first columns as the labels (y) of each row. In total there are 785 columns including the labels column. Therefore, in order to create a training set, and testing set, we need to drop the labels for both train.csv and test.csv.

# In[ ]:


x_train, y_train, x_test, y_test = train.drop(['label'], axis=1), train.label, test.drop(['label'], axis=1), test.label


# # MODEL TRAINING

# > One method that can be used to classify multiclass dataset is the K Nearest Neighbors algorithm.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=3)
knn_clf.fit(x_train, y_train)


# > The parameters used to construct the KNeighborsClassifiers might not be the combinations that yeilds the highest accuracy. To that end, we still need to find the best parameters that will yield highest accuracy. Hence, one possible way is to use Grid Search.

# In[ ]:


# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier

# param_grid = [{"weights": ["uniform", "distance"], "n_neighbors": [3, 4, 5]}]

# knn_clf = KNeighborsClassifier()
# grid_search = GridSearchCV(knn_clf, param_grid, cv=2, verbose=3)
# grid_search.fit(x_train, y_train)


# # MAKING PREDICTIONS

# In[ ]:


y_test_pred = knn_clf.predict(x_test)


# In[ ]:


y_test_pred


# In[ ]:


y_test


# # ACCURACY TESTING

# > In classification problems, accuracy testing algorithms includes the follow:
# - simple accuracy
# - precision
# - recall
# - f1-score
# - roc (auc)

# ### SIMPLE ACCURACY
# simple accuracy = (tp + tn) / (tp + tn + fp + fn)

# In[ ]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_test_pred)


# ### PRECISION
# precision = (tp) / (tp + fp)

# In[ ]:


from sklearn.metrics import precision_score

precision_score(y_test, y_test_pred, average="micro")


# ### RECALL
# recall = (tp) / (tp + fn)

# In[ ]:


from sklearn.metrics import recall_score

recall_score(y_test, y_test_pred, average="micro")


# ### F1 SCORE
# f1 = 2 * (precision * recall) / (precision + recall)

# In[ ]:


from sklearn.metrics import f1_score

f1_score(y_test, y_test_pred, average="micro")

