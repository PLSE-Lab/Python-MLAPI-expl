#!/usr/bin/env python
# coding: utf-8

# # Tuning a kNN Classifier for Malignant Tumor Diagnosis

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import binarize
from sklearn import metrics

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/data.csv")


# In[ ]:


df.head()


# In[ ]:


df.describe()


# **Remove Unnecessary Rows**

# In[ ]:


del df['Unnamed: 32']
del df['id']


# **Convert *diagnosis* to binary**

# In[ ]:


d = {'M':1, 'B':0}
y = df['diagnosis'].map(d).values
X = df[df.columns[1:31]]


# **Split data** Open to suggestions as to how to split data here. I'm trying to use as much data as I can while holding out a reasonable amount for testing. In this case "reasonable" to me simply means that accuracy scores have ~ 1 percentage point resolution.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(y_train.shape, y_test.shape)


# ## kNN Classifier
# - A good place to start, since it's a simple model and the dataset is not large.

# In[ ]:


knn = KNeighborsClassifier()
k_values = range(1, 31)
weight_values = ['uniform', 'distance']
param_dict = {'n_neighbors':k_values, 'weights':weight_values}
grid = GridSearchCV(knn, param_dict, cv=10, scoring='accuracy')
grid.fit(X_train, y_train)


# In[ ]:


print(grid.best_score_)
print(grid.best_params_)


# **Note:** Not bad for kNN. Admittedly I thought this model would perform much worse.

# **Split results by weight type** (*uniform* or *distance*) and visualize scores

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

uniform_scores = []
distance_scores = []
all_scores = grid.cv_results_['mean_test_score']
all_params = grid.cv_results_['params']

#split
for i in range(len(all_scores)):
    if all_params[i]['weights'] == 'uniform':
        uniform_scores.append(all_scores[i])
    else:
        distance_scores.append(all_scores[i])
        
# Plot
plt.plot(k_values, uniform_scores)
plt.xlabel('K')
plt.ylabel('Mean Validation Score')
plt.title("Validation Score by K value, Uniform Weights")
plt.grid(True)


# In[ ]:


plt.plot(k_values, distance_scores)
plt.xlabel('K')
plt.ylabel('Mean Validation Score')
plt.title("Validation Score by K value, Distance Weights")
plt.grid(True)


# ### Thoughts
# - After running this kernel a few times, it doesn't seem like there's an obvious difference between the two weight choices. Both have similar top *k* values, and neither out performs the other by much in terms of validation score.
# - If I had to choose, in light of the tie between the two metrics I suppose I would take uniform since it makes a for a computationally less intensive model.

# In[ ]:


knn_dist = KNeighborsClassifier(n_neighbors=12, weights='distance')
knn_unif = KNeighborsClassifier(n_neighbors=14, weights='uniform')
knn_dist.fit(X_train, y_train)
knn_unif.fit(X_train, y_train)


# In[ ]:


def model_accuracy(model, X, y):
    y_pred = model.predict(X)
    return metrics.accuracy_score(y, y_pred)

acc_dist = model_accuracy(knn_dist, X_test, y_test)
acc_unif = model_accuracy(knn_unif, X_test, y_test)

print("Best distance-weighted model test accuracy:", acc_dist)
print("Best uniform-weighted model test accuracy:", acc_unif)


# #### Take a look at ROC
# - First we'll take a look at the distribution on probabilities assigned by the classifier.
# - Then we'll take a look at the ROC curve.
# - False negatives are more dangerous in this case, so look at what the threshold trade-offs look like

# In[ ]:


y_prob_dist = knn_dist.predict_proba(X_test)
y_prob_unif = knn_unif.predict_proba(X_test)

def plot_hist(y, title='', bins=10):
    plt.hist(y, bins=bins)
    plt.title(title)
    plt.grid(True)

plot_hist(y_prob_dist[:,1], title="Distance kNN Probability Scores")


# In[ ]:


plot_hist(y_prob_unif[:,1], title="Uniform kNN Probability Scores")


# **Note:** Both models have a tendency to favor a "Benign" diagnosis, which reflects the imbalance of diagnosis distributions in the data.
# **Note** Just eyeballing it, it appears as though setting the threshold even at a very low probability wouldn't push that many diagnoses into the "Malignent" category

# In[ ]:


# A finer grained histogram
plot_hist(y_prob_dist[:,1], title="Distance Probability Scores", bins=20)


# In[ ]:


y_pred_dist = knn_dist.predict(X_test)
metrics.confusion_matrix(y_test, y_pred_dist)


# In[ ]:


# plot ROC
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob_dist[:,1])
plt.plot(fpr, tpr)
plt.title("ROC curve for Distance-weighted kNN (k=12)")
plt.xlabel("False Positive Rate (1 - specificity)")
plt.ylabel("True Positive Rate")
plt.grid(True)


# In[ ]:


fpr


# In[ ]:


thresholds


# ## Conclusions
# - By setting the threshold, the model can achieve a decent true positive rate of ~ 95%, with an overall accuracy of about 85% - 90%.
# - While this is decent performance, a false negative diagnosis is obviously potentially catastrophic for the patient, and is not tolerable, so we should look either to other methods, or to preprocessing data to get better performance on the dataset.
