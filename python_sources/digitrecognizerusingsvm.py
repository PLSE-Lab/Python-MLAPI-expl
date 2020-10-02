#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Importing the datasets train and test

# In[ ]:


# dataset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# Analyzing the train and test datasets

# In[ ]:


train.info()


# In[ ]:


test.info()


# Checking for missing values.

# In[ ]:


# summing up the missing values (column-wise) in train data set
sum(round(100*(train.isnull().sum()/len(train.index)), 2)>0)


# In[ ]:


# summing up the missing values (column-wise) in test data set
sum(round(100*(train.isnull().sum()/len(train.index)), 2)>0)


# There are no missing values in columns, but lot of columns having only 0 values. We cannot drop these columns as they might be different for train and test sets, and as a result we might end up having different column sets for train and test data

# In[ ]:


train.nunique()==1


# In[ ]:


test.nunique()==1


# In[ ]:


## Visualizing the number of class and counts in the datasets
sns.countplot(train["label"])


# In[ ]:


# Plotting some samples
four = train.iloc[3, 1:]
four.shape
four = four.values.reshape(28,28)
plt.imshow(four, cmap='gray')
plt.title("Digit 4")


# In[ ]:


seven = train.iloc[6, 1:]
seven.shape
seven = seven.values.reshape(28, 28)
plt.imshow(seven, cmap='gray')
plt.title("Digit 7")


# In[ ]:


# basic plots: How do various pixels vary with the digits

plt.figure(figsize=(10, 5))
sns.barplot(x='label', y='pixel45', 
            data=train)


# Checking for rows which are completely zero in both train and test daasets, and deleting only those columns which have 0 values in both, else we will end upm havinmg different column sets for train and test.

# In[ ]:


train_1 = train.drop('label',axis=1)


# In[ ]:


zero_val_cols_removal = pd.DataFrame(((train_1 != 0).any(axis=0) &  (test != 0).any(axis=0))==False)


# In[ ]:


zero_val_cols_removal.reset_index(inplace=True)


# In[ ]:


zero_val_cols_removal.columns = ['col_name','is_zero']


# In[ ]:


for col_name in zero_val_cols_removal.loc[(zero_val_cols_removal.is_zero==True),'col_name']:
    train.drop(col_name,axis=1,inplace=True)
    test.drop(col_name,axis=1,inplace=True)


# In[ ]:


train.info()


# In[ ]:


test.info()


# Columns have now been reduced from 785 to 691 for train and 784 tom 690 for test, one less in test due to the absence of label columnm

# Since train dataset is having 42000 rows which would take a long time with GridSearchCV with 5 folds to find optimal values of hyperparameters, we will takme Sub Sample of data like 20% of the train data here.m 

# In[ ]:


train_sample = train.sample(frac =.20,random_state=10) 


# In[ ]:


train_sample.head()


# In[ ]:


train_sample.info()


# Seperating the prediction label and the pixel columns for the train set

# In[ ]:


X_train = train_sample.drop('label',axis=1)
y_train = train_sample['label']


# In[ ]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
test = scaler.transform(test)


# First we will fit a linear SVM model with varying values of C using GridSearchCV and KFold and check which C gives best accuracy

# In[ ]:


# creating a KFold object with 5 splits 
folds = KFold(n_splits = 5, shuffle = True, random_state = 4)

# specify range of parameters (C) as a list
params = {"C": [0.1, 1, 10, 100, 1000]}

model_linear = SVC(kernel='linear', cache_size=10000)

# set up grid search scheme
# note that we are still using the 5 fold CV scheme we set up earlier
model_cv = GridSearchCV(estimator = model_linear, 
                        param_grid = params, 
                        scoring= 'accuracy', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True,
                        n_jobs=-1)      


# In[ ]:


# fit the model - it will fit 5 folds across all values of C
model_cv.fit(X_train, y_train)  


# In[ ]:


# results of grid search CV
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[ ]:


# plot of C versus train and test scores

plt.figure(figsize=(8, 6))
plt.plot(cv_results['param_C'], cv_results['mean_test_score'])
plt.plot(cv_results['param_C'], cv_results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')


# In[ ]:


# printing the optimal accuracy score and hyperparameters
best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_

print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))


# The best train data accuracy that we got is 0.9142 for c=0.1. We will use this to predict the test labels and check for the accuracy score and ranking in kaggle.

# In[ ]:


# model with optimal hyperparameters

# model
model = SVC(C=0.1,  kernel="linear")

model.fit(X_train, y_train)
y_pred = model.predict(test)


# In[ ]:


y_pred = pd.DataFrame(y_pred)


# In[ ]:


y_pred.head()


# In[ ]:


y_pred.reset_index(inplace=True)


# In[ ]:


y_pred.head()


# In[ ]:


y_pred.columns = ['ImageId','Label']


# In[ ]:


y_pred.ImageId = y_pred.ImageId + 1


# In[ ]:


y_pred.to_csv('result_linear.csv', index=False)


# This one scored  0.91457 on kaggle. I have a higher score of 0.91914 which ranked 2514, but that was with more than 20% subsampling

# Further we will check for other SVM kernels Polynomial and RBF

# Now lets run Polynomial SVM on train data with varying values of C, gamma, and degree

# In[ ]:


# creating a KFold object with 5 splits 
folds = KFold(n_splits = 5, shuffle = True, random_state = 4)

# specify range of parameters (C) as a list
hyper_params = [ {'gamma': [1e-1, 1e-2],
                      'C': [0.1, 1],
                 'degree': [2,3]
                 }]


model_poly = SVC(kernel='poly', cache_size=10000)

# set up grid search scheme
# note that we are still using the 5 fold CV scheme we set up earlier
model_cv = GridSearchCV( estimator = model_poly, 
                         param_grid = hyper_params, 
                         scoring= 'accuracy', 
                         cv = folds, 
                         verbose = 1,
                         return_train_score=True,
                         n_jobs=-1)      


# In[ ]:


# fit the model - it will fit 5 folds across all values of C, gamma and degree
model_cv.fit(X_train, y_train) 


# In[ ]:


# results of grid search CV
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[ ]:


# # plotting
plt.figure(figsize=(25,20))

# subplot 4/1
plt.subplot(221)
gamma_1_degree_2 = cv_results.loc[(cv_results.param_gamma==0.1) & (cv_results.param_degree==2)]

plt.plot(gamma_1_degree_2["param_C"], gamma_1_degree_2["mean_test_score"])
plt.plot(gamma_1_degree_2["param_C"], gamma_1_degree_2["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.1 Degree=2")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

# subplot 4/2
plt.subplot(222)
gamma_1_degree_3 = cv_results.loc[(cv_results.param_gamma==0.1) & (cv_results.param_degree==3)]

plt.plot(gamma_1_degree_3["param_C"], gamma_1_degree_3["mean_test_score"])
plt.plot(gamma_1_degree_3["param_C"], gamma_1_degree_3["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.1 Degree=3")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

# subplot 4/3
plt.subplot(223)
gamma_01_degree_2 = cv_results.loc[(cv_results.param_gamma==0.01) & (cv_results.param_degree==2)]

plt.plot(gamma_01_degree_2["param_C"], gamma_01_degree_2["mean_test_score"])
plt.plot(gamma_01_degree_2["param_C"], gamma_01_degree_2["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01 Degree=2")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

# subplot 4/4
plt.subplot(224)
gamma_01_degree_3 = cv_results.loc[(cv_results.param_gamma==0.01) & (cv_results.param_degree==3)]

plt.plot(gamma_01_degree_3["param_C"], gamma_01_degree_3["mean_test_score"])
plt.plot(gamma_01_degree_3["param_C"], gamma_01_degree_3["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01 Degree=3")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')


# In[ ]:


# printing the optimal accuracy score and hyperparameters
best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_

print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))


# So the best score here for Polynomial SVM is for gamma 0.1, degree 3, C 0.1. We will run the model now for these values again, predict scores and check for accuracy in kaggle.

# In[ ]:


# model with optimal hyperparameters

# model
model = SVC(C=0.1, degree=3, gamma=0.1, kernel="poly")

model.fit(X_train, y_train)
y_pred = model.predict(test)


# In[ ]:


y_pred = pd.DataFrame(y_pred)


# In[ ]:


y_pred.head()


# In[ ]:


y_pred.reset_index(inplace=True)


# In[ ]:


y_pred.head()


# In[ ]:


y_pred.columns = ['ImageId','Label']


# In[ ]:


y_pred.ImageId = y_pred.ImageId + 1


# In[ ]:


y_pred.to_csv('result_polynomial.csv', index=False)


# This file scored 0.95371 with a rank of 2343 on kaggle.

# Now we will move on to using RBF SVM and check for the accuracy that it gives. Here we will train the models for varying values of C and Gamma.

# In[ ]:


# creating a KFold object with 5 splits 
folds = KFold(n_splits = 5, shuffle = True, random_state = 4)

# specify range of parameters (C) as a list
hyper_params = [ {'gamma': [1e-1, 1e-2, 1e-3],
                      'C': [0.1, 1, 10]
                 }]


model_poly = SVC(kernel='rbf', cache_size=10000)

# set up grid search scheme
# note that we are still using the 5 fold CV scheme we set up earlier
model_cv = GridSearchCV( estimator = model_poly, 
                         param_grid = hyper_params, 
                         scoring= 'accuracy', 
                         cv = folds, 
                         verbose = 1,
                         return_train_score=True,
                         n_jobs=-1)      


# In[ ]:


# fit the model - it will fit 5 folds across all values of C and gamma
model_cv.fit(X_train, y_train) 


# In[ ]:


# results of grid search CV
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[ ]:


# # plotting
plt.figure(figsize=(25,8))

# subplot 3/1
plt.subplot(131)
gamma_1 = cv_results.loc[(cv_results.param_gamma==0.1)]

plt.plot(gamma_1["param_C"], gamma_1["mean_test_score"])
plt.plot(gamma_1["param_C"], gamma_1["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.1")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

# subplot 3/2
plt.subplot(132)
gamma_01 = cv_results.loc[(cv_results.param_gamma==0.01)]

plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])
plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

# subplot 3/3
plt.subplot(133)
gamma_001 = cv_results.loc[(cv_results.param_gamma==0.001)]

plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])
plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.001")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')


# In[ ]:


# printing the optimal accuracy score and hyperparameters
best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_

print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))


# The best score for rbf(0.9423) was for c=10 and gamma=0.001. we will train the model indivisually with these values and use it to predict test labels.

# In[ ]:


# model with optimal hyperparameters

# model
model = SVC(C=10, gamma=0.001, kernel="rbf")

model.fit(X_train, y_train)
y_pred = model.predict(test)


# In[ ]:


y_pred = pd.DataFrame(y_pred)


# In[ ]:


y_pred.head()


# In[ ]:


y_pred.reset_index(inplace=True)


# In[ ]:


y_pred.head()


# In[ ]:


y_pred.columns = ['ImageId','Label']


# In[ ]:


y_pred.ImageId = y_pred.ImageId + 1


# In[ ]:


y_pred.to_csv('result_rbf.csv', index=False)


# This file scored 0.94442  on kaggle.

# So we got accuracies with different methods with 20% sub-Sampled data as below:
#     1. SVM with Linear Kernel: 0.91457
#     2. SVM with Polynomial Kernel: 0.95371 (This was my personal best rank for this problem statement : 2343)
#     3. SVM with RBF Kernel: 0.94442    
# Also note that I have ran this Kernel on Kaggle Directly as running it on my laptop was very slow due to small RAM size etc.
