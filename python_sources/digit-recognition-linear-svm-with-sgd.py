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


# ## I. LOAD and EXPLORE THE DATA

# In[ ]:


# Load the train and test data sets
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print ("Train data shape:", train.shape)
print ("Test data shape:", test.shape)


# In[ ]:


# Preview raw train data
train.head()


# In[ ]:


# Check data values distribution
print(train.columns)
train.describe()


# In[ ]:


# Check missing values
print(train.isnull().sum().values.sum())
train.isnull().sum().head(10)


# In[ ]:


# check test data types
print(test.info())
# Preview raw test data
test.head()


# In[ ]:


# traing set statistics
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x=train["label"],color="grey")
plt.title('Samples per Label')
train.label.astype('category').value_counts()


# In[ ]:


# Prepare data for training
X_train_ = train.drop(labels = ["label"],axis = 1).values
Y_train_ = train["label"].values
X_test_ = test.values
classes = np.unique(Y_train_)
print("classes :",classes)


# In[ ]:


# Preview a Data Sample
import matplotlib.pyplot as plt
plt.figure(figsize=(16,4))
im_side =  int(np.sqrt(X_train_.shape[1]))
for i in range(16):  
    plt.subplot(2, 8, i+1)
    plt.imshow(X_train_[i].reshape((im_side,im_side)),cmap=plt.cm.gray)
    plt.axis('off')
plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.show()


# ## II. Models Building using Sklearn

# In[ ]:


#----transformer : scaling 
from sklearn.preprocessing import StandardScaler
#----transformer: Dimensionality reduction 
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.decomposition import PCA
#----transformer : sequencing transformers
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
#----transformer :classifiers 
from sklearn import svm

#---
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV


# In[ ]:


## Normalization
X_train = X_train_/255
X_test= X_test_/255
Y_train = Y_train_.copy()


# ### SVM

# The implementation of sklearn.svm.SVC is based on libsvm. The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands of samples. For large datasets consider using sklearn.linear_model.LinearSVC or sklearn.linear_model.SGDClassifier instead, possibly after a sklearn.kernel_approximation.Nystroem transformer. https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

# In[ ]:


clf_sgd_svm = make_pipeline(StandardScaler(),linear_model.SGDClassifier())
n = len(X_train)
print(n,clf_sgd_svm.steps)


# ## III. Parameters Tuning

# Stochastic Gradient Descent is sensitive to feature scaling, so it is highly recommended to scale your data. For example, scale each attribute on the input vector X to [0,1] or [-1,+1], or standardize it to have mean 0 and variance 1. Note that the same scaling must be applied to the test vector to obtain meaningful results. https://scikit-learn.org/stable/modules/sgd.html#tips-on-practical-use

# In[ ]:


# testing different parameters
sgd_svm_param ={
        'sgdclassifier__alpha': 10.0**-np.arange(2,7),
        'sgdclassifier__max_iter': np.arange(1,5)*np.ceil(10**6 / n),
        'sgdclassifier__tol': [1e-3]    }
# creating a KFold object with 5 splits (equivalent different validation tests of 20%), using cross validation so no need to split !!!!
folds = KFold(n_splits = 5, shuffle = True, random_state = 10)
grid_search = GridSearchCV(estimator = clf_sgd_svm, 
                        param_grid = sgd_svm_param, 
                        scoring= 'accuracy', 
                        cv = folds, 
                        verbose = 20,## the higher the positive number the more information about progress is printed
                        return_train_score=True, n_jobs=-1)  #n_jobs=-1 for parallel computing
grid_search.fit(X_train, Y_train)


# In[ ]:


best_score = grid_search.best_score_
best_hyperparams = grid_search.best_params_
# printing the optimal accuracy score and hyperparameters
print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))


# In[ ]:


# cv results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results


# ## IV. Prediction Submission

# In[ ]:


y_test_pred = grid_search.predict(X_test)
results = pd.Series(y_test_pred,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("MNIST-linear-SGD-SVM.csv",index=False)


# In[ ]:


The previous csv file would scored 0.91071 on Kaggle's public leaderboard (evaluated on 25% of the test data) when submitted.

