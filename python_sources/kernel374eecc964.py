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


# 

# Assignment - Support Vector Machines
# 

# Brief on subject: 
# 

# A classic problem in the field of pattern recognition is that of handwritten digit recognition. Suppose that you have images of handwritten digits ranging from 0-9 written by various people in boxes of a specific size - similar to the application forms in banks and universities.

# Objective:

# You are required to develop a model using Support Vector Machine which should correctly classify the handwritten digits from 0-9 based on the pixel values given as features. Thus, this is a 10-class classification problem. 

# Data Description:

# For this problem, we use the MNIST data which is a large database of handwritten digits. The 'pixel values' of each digit (image) comprise the features, and the actual number between 0-9 is the label. 
# 
#  Since each image is of 28 x 28 pixels, and each pixel forms a feature, there are 784 features. MNIST digit recognition is a well-studied problem in the ML community, and people have trained numerous models (Neural Networks, SVMs, boosted trees etc.) achieving error rates as low as 0.23% (i.e. accuracy = 99.77%, with a convolutional neural network).
# 
#  

# Input files used:

# Train.csv: There are 785 attributes. Columns are label to pixel783.
# Test.csv: There are 784 attributes. Columns are pixel1 to pixel783.

# In[ ]:


# Import important libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale


# In[ ]:


# import train.csv and test.csv
Train = pd.read_csv('../input/train.csv')
Test = pd.read_csv("../input/test.csv")
# data visualization
Train.head()


# In[ ]:


Test.head()


# In[ ]:


Train.tail()


# In[ ]:


# data types
Train.info()


# In[ ]:


# data types
Test.info()


# In[ ]:


# dimensions
print("Dimensions: ", Train.shape, "\n")


# In[ ]:


# dimensions
print("Dimensions: ", Test.shape, "\n")


# In[ ]:


Train.describe


# In[ ]:


Test.describe


# In[ ]:


# a quirky bug: the column names have a space, e.g. 'xbox ', which throws and error when indexed
print(Train.columns)


# In[ ]:


# a quirky bug: the column names have a space, e.g. 'xbox ', which throws and error when indexed
print(Test.columns)


# In[ ]:


Train.isnull().sum()


# In[ ]:


Test.isnull().sum()


# In[ ]:


Train.isnull().values.any()


# In[ ]:


Test.isnull().values.any()


# In[ ]:


# look at fraction
Train['label'].describe()


# In[ ]:


Train.describe()


# In[ ]:


Test.describe()


# In[ ]:


order = list(np.sort(Train['label'].unique()))
print(order)


# In[ ]:


Train_means = Train.groupby('label').mean()
Train_means.head()


# In[ ]:


pd.set_option('display.max_columns', 785)
Train.describe()


# In[ ]:


plt.figure(figsize=(30, 24))
sns.heatmap(Train_means)


# In[ ]:


#See the distribution of the labels
sns.countplot(Train.label)


# In[ ]:


Train.label.value_counts(dropna = False)


# In[ ]:


Number = Train[0:8000]

y = Number.iloc[:,0]

X = Number.iloc[:,1:]

print(y.shape)
print(X.shape)


# In[ ]:


# train test split
X_scaled = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 0)


# In[ ]:


# linear model

model_linear = SVC(kernel='linear')
model_linear.fit(X_train, y_train)

# predict
y_pred = model_linear.predict(X_test)


# In[ ]:


# confusion matrix and accuracy

# accuracy
print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

# cm
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))


# In[ ]:


# Accuracy
classification_metrics = metrics.classification_report(y_true=y_test, y_pred=y_pred)
print(classification_metrics)


# In[ ]:


# non-linear model
# using rbf kernel, C=1, default value of gamma

# model
non_linear_model = SVC(kernel='rbf')

# fit
non_linear_model.fit(X_train, y_train)

# predict
y_pred = non_linear_model.predict(X_test)


# In[ ]:


# confusion matrix and accuracy

# accuracy
print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

# cm
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))


# In[ ]:


# Accuracy
classification_metrics = metrics.classification_report(y_true=y_test, y_pred=y_pred)
print(classification_metrics)


# In[ ]:


# creating a KFold object with 5 splits 
folds = KFold(n_splits = 5, shuffle = True, random_state = 0)

# specify range of hyperparameters
# Set the parameters by cross-validation
hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]


# specify model
model = SVC(kernel="rbf")

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = model, 
                        param_grid = hyper_params, 
                        scoring= 'accuracy', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model_cv.fit(X_train, y_train)                  


# In[ ]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[ ]:


# converting C to numeric type for plotting on x-axis
cv_results['param_C'] = cv_results['param_C'].astype('int')

# # plotting
plt.figure(figsize=(16,6))

# subplot 1/3
plt.subplot(131)
gamma_01 = cv_results[cv_results['param_gamma']==0.01]

plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])
plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01")
plt.ylim([0.70, 1.20])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

# subplot 2/3
plt.subplot(132)
gamma_001 = cv_results[cv_results['param_gamma']==0.001]

plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])
plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.001")
plt.ylim([0.80, 1.20])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')


# subplot 3/3
plt.subplot(133)
gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]

plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])
plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.0001")
plt.ylim([0.80, 1.20])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')


# The plots above show some useful insights:
# Non-linear models (high gamma) perform much better than the linear ones
# At any value of gamma, a high value of C leads to better performance
# None of the models tend to overfit (even the complex ones), since the training and test accuracies closely follow each other
# This suggests that the problem and the data is inherently non-linear in nature, and a complex model will outperform simple, linear models in this case.
# 
# Let's now choose the best hyperparameters.

# In[ ]:


# printing the optimal accuracy score and hyperparameters
best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_

print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))


# Building and Evaluating the Final Model

# Let's now build and evaluate the final model, i.e. the model with highest test accuracy.

# In[ ]:


# model with optimal hyperparameters

# model
model = SVC(C=10, gamma=0.001, kernel="rbf")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# metrics
print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")
print(metrics.confusion_matrix(y_test, y_pred), "\n")


# Conclusion

# The accuracy achieved using a non-linear kernel (~0.94) is mush higher than that of a linear one (~0.91). We can conclude that the problem is highly non-linear in nature.

# In[ ]:




