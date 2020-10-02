#!/usr/bin/env python
# coding: utf-8

# # MNIST Digit recognition (SVM)

# __Handwritten digit recognition using SVM (Linear as well as Non Linear)__

# ### Table Of Content:
# * [Problem Overview](#Problem-Overview)
# * [Solution Overview](#Solution-Overview)
# * [Importing Libraries](#Importing-Libraries)
# * [Importing Dataset](#Importing-Dataset)
# * [Understanding Dataset](#Understanding-Dataset)
# * [Preparing Dataset](#Preparing-Dataset)
# * [Exploratory Data Analysis](#Exploratory-Data-Analysis)
# * [Dataset Test and Train Split](#Dataset-Test-and-Train-Split)
# * [Building Linear SVM Model](#Building-Linear-SVM-Model)
# * [Optimizing Hyperparameter C and Evaluation of Final Linear SVM Model](#Optimizing-Hyperparameter-C-and-Evaluation-of-Final-Linear-SVM-Model)
# * [Building Non Linear SVM Model](#Building-Non-Linear-SVM-Model)
# * [Optimizing Hyperparameter C Gamma and Evaluation of Final Non Linear SVM Model](#Optimizing-Hyperparameter-C-Gamma-and-Evaluation-of-Final-Non-Linear-SVM-Model)
# * [Observing the performance of our final Model](#Observing-the-performance-of-our-final-Model)
# * [Making predictions for Test Dataset using the final Model](#Making-predictions-for-Test-Dataset-using-the-final-Model)

# ----

# ### Problem Overview

# * You are required to develop a model using Support Vector Machine which should correctly classify the handwritten digits.
# * The digits range from 0-9. The classification is based on the pixel values given as features. 
# * Each image is of 28 x 28 pixels, and each pixel forms a feature, there are 784 features. 
# * This is a 10-class classification problem. 

# ----

# ### Solution Overview

# * We will try to solve the problem firstly using Linear SVM Model.
# * We will optimize the values of Hyper-Parameter (C) to obtain best accuracy.
# * We will then try to solve the problem using Non Linear SVM Model.
# * We will optimize the values of Hyper-Parameters (C, Gamma) to obtain best accuracy.
# * We will compare the performance of each of these models and use the best model to predict for Test Dataset.

# ----

# ### Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, validation_curve, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.svm import SVC


# In[ ]:


# Checking version of imported libraries
np.__version__, pd.__version__, sns.__version__


# In[ ]:


# Making miscellaneous setting for better experience
import warnings
warnings.filterwarnings('ignore')


# ----

# ### Importing Dataset

# In[ ]:


# Importing training dataset (train.csv)
training_dataframe = pd.read_csv('../input/train.csv')

# Importing testing dataset (test.csv)
testing_dataframe = pd.read_csv('../input/test.csv')


# ----

# ### Understanding Dataset

# In[ ]:


# Understanding the training dataset | Shape
training_dataframe.shape


# In[ ]:


# Understanding the training dataset | Meta Data
training_dataframe.info()


# In[ ]:


# Understanding the training dataset | Data Content
training_dataframe.describe()


# In[ ]:


# Understanding the training dataset | Sample Data
training_dataframe.head()


# In[ ]:


# Understanding the training dataset | Missing Values
sum(training_dataframe.isnull().sum())


# **Summary of Dataset Understanding:**
# 1. Dataset is clean (no missing values)
# 2. Dataset is large (42000 rows and 785 columns)
# 3. Dataset is purely numeric (all 785 columns are int64)
# 4. Dataset contains insignificant columns (several columns have single value)

# ----

# ### Preparing Dataset

# In[ ]:


# Dropping Duplicate Values
training_dataframe.drop_duplicates(inplace=True)


# In[ ]:


# Taking a random subset of training dataset (containing 100% of rows from the original dataset)
rcount = int(1.0*training_dataframe.shape[0])
subset_training_dataframe = training_dataframe.sample(n=rcount)


# In[ ]:


# Understanding the processed training dataset | Shape
subset_training_dataframe.shape


# **Summary of Dataset Preparation:**
# 1. We have 76 (785-709) insignificant columns, however, we are leaving them as it is for later use of plotting image.
# 2. We can take a random subset (25% of our dataframe) since the dataset is large, however we took all data for accuracy.

# ----

# ### Exploratory Data Analysis

# In[ ]:


# Clecking if all labels are present almost equally in subset training dataset
plt.figure(figsize=(8,4))
sns.countplot(subset_training_dataframe['label'], palette = 'icefire')


# In[ ]:


# Checking for collinearity in dataset
plt.figure(figsize=(16,8))
sns.heatmap(data=subset_training_dataframe.corr(),annot=False)


# **Summary of Exploratory Data Analysis:**
# 1. All labels are present almost equally in subset training dataset
# 2. Since we see clear pattern in the heatmap, the dataset is highly correlated
# 3. Adjacent/Nearby pixel values are correlated, which we expect as well.

# ----

# ### Dataset Test and Train Split

# In[ ]:


# splitting into X and y
X = subset_training_dataframe.drop("label", axis = 1)
y = subset_training_dataframe.label.values.astype(int)


# In[ ]:


# scaling the features
X = scale(X)


# In[ ]:


# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)


# In[ ]:


# confirm that splitting also has similar distribution
print(y_train.mean())
print(y_test.mean())


# ----

# ### Building Linear SVM Model

# In[ ]:


# Model building

# instantiate an object of class SVC() using cost C=1, gamma='auto'
model = SVC(C = 1, gamma='auto')

# fit
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)


# In[ ]:


# Evaluate the model using confusion matrix 
confusion_matrix(y_true=y_test, y_pred=y_pred)


# In[ ]:


# Model Accuracy
print("Accuracy :", accuracy_score(y_test, y_pred))


# In[ ]:


# K-Fold Cross Validation

# Creating a KFold object with 5 splits 
folds = KFold(n_splits = 5, shuffle = True, random_state = 4)

# Instantiating a model with cost=1, gamma='auto'
model = SVC(C = 1, gamma='auto')

# computing the cross-validation scores 
# Argument cv takes the 'folds' object, and we have specified 'accuracy' as the metric
cv_results = cross_val_score(model, X_train, y_train, cv = folds, scoring = 'accuracy', n_jobs=-1)

# print 5 accuracies obtained from the 5 folds
print(cv_results)
print(f'mean accuracy = {cv_results.mean()}')


# ----

# ### Optimizing Hyperparameter C and Evaluation of Final Linear SVM Model

# In[ ]:


# Grid Search to Find Optimal Hyperparameter C

# specify range of parameters (C) as a list
params = {"C": [0.1, 1, 10, 100, 1000]}

model = SVC(gamma='auto')

# set up grid search scheme
# note that we are still using the 5 fold CV scheme we set up earlier
model_cv = GridSearchCV(estimator = model, param_grid = params, 
                        scoring='accuracy', cv=folds, n_jobs=-1,
                        verbose=1, return_train_score=True)

# fit the model - it will fit 5 folds across all values of C
model_cv.fit(X_train, y_train)  

# results of grid search CV
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[ ]:


# plot of C versus train and test scores

plt.figure(figsize=(4, 4))
plt.plot(cv_results['param_C'], cv_results['mean_test_score'])
plt.plot(cv_results['param_C'], cv_results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')


# In[ ]:


best_score = model_cv.best_score_
best_C = model_cv.best_params_['C']

print(" The highest test accuracy is {0} at C = {1}".format(best_score, best_C))


# In[ ]:


# model with the best value of C
model = SVC(C=best_C, gamma='auto')

# fit
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)


# In[ ]:


# Optimal Final Linear SVM Model Accuracy
print("Accuracy :", accuracy_score(y_test, y_pred))


# ----

# ### Building Non Linear SVM Model

# In[ ]:


# Model building

# instantiate an object of class SVC() using cost C=1, Gamma='auto', Kernel='rbf'
model = SVC(C = 1, gamma='auto', kernel='rbf')

# fit
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)


# In[ ]:


# Evaluate the model using confusion matrix 
confusion_matrix(y_true=y_test, y_pred=y_pred)


# In[ ]:


# Model Accuracy
print("Accuracy :", accuracy_score(y_test, y_pred))


# In[ ]:


# K-Fold Cross Validation

# Creating a KFold object with 5 splits 
folds = KFold(n_splits = 5, shuffle = True, random_state = 4)

# Instantiating a model with cost=1, Gamma='auto', Kernel='rbf'
modelkernel=SVC(C = 1, gamma='auto', kernel='rbf')

# computing the cross-validation scores 
# Argument cv takes the 'folds' object, and we have specified 'accuracy' as the metric
cv_results = cross_val_score(model, X_train, y_train, cv=folds, scoring='accuracy', n_jobs=-1)

# print 5 accuracies obtained from the 5 folds
print(cv_results)
print(f'mean accuracy = {cv_results.mean()}')


# ----

# ### Optimizing Hyperparameter C Gamma and Evaluation of Final Non Linear SVM Model

# In[ ]:


# Grid Search to Find Optimal Hyperparameter C, Gamma

# specify range of hyperparameters
# Set the parameters by cross-validation
hyper_params = [ {'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]

# specify model
model = SVC(kernel="rbf")

# set up GridSearchCV()
model_cv = GridSearchCV(estimator=model, param_grid=hyper_params, 
                        scoring='accuracy', cv=folds, n_jobs=-1,
                        verbose=1, return_train_score=True)      

# fit the model
model_cv.fit(X_train, y_train) 

# results of grid search CV
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[ ]:


# plot of C and Gamma versus train and test scores

# converting C to numeric type for plotting on x-axis
cv_results['param_C'] = cv_results['param_C'].astype('int')

# plotting
plt.figure(figsize=(16,4))

# subplot 1/4
plt.subplot(141)
gamma_1 = cv_results[cv_results['param_gamma']==0.1]

plt.plot(gamma_1["param_C"], gamma_1["mean_test_score"])
plt.plot(gamma_1["param_C"], gamma_1["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.1")
plt.ylim([0.0, 1.1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

# subplot 2/4
plt.subplot(142)
gamma_01 = cv_results[cv_results['param_gamma']==0.01]

plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])
plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01")
plt.ylim([0.6, 1.1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

# subplot 3/4
plt.subplot(143)
gamma_001 = cv_results[cv_results['param_gamma']==0.001]

plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])
plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.001")
plt.ylim([0.8, 1.1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

# subplot 4/4
plt.subplot(144)
gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]

plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])
plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.0001")
plt.ylim([0.8, 1.1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')


# In[ ]:


best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_

print(f'The best test score is {best_score} corresponding to hyperparameters {best_hyperparams}')


# In[ ]:


# model with the best value of C and Gamma
model = SVC(C=best_hyperparams['C'], gamma=best_hyperparams['gamma'], kernel="rbf")

# fit
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)


# In[ ]:


# Optimal Final Linear SVM Model Accuracy
print("Accuracy :", accuracy_score(y_test, y_pred))


# **Selection of best model:**
# * Accuracy is our primary concern, and there is no major different with Linear or a Non Linear SVM Model.
# * Considering our final model is the Final Non Linear SVM Model

# ----

# ### Observing the performance of our final Model

# In[ ]:


# Predicting values for our Test Split of Training Dataset
test_predict = model.predict(X_test)


# In[ ]:


# Plotting the distribution of our prediction
d = {'ImageId': np.arange(1,test_predict.shape[0]+1), 'Label': test_predict}
dataframe_to_export = pd.DataFrame(data=d)
sns.countplot(dataframe_to_export['Label'], palette = 'icefire')


# In[ ]:


# Les't visualize our Final Model in Action for few unseen images from Training Dataset

a = np.random.randint(1,test_predict.shape[0]+1,5)

plt.figure(figsize=(16,4))
for k,v in enumerate(a):
    plt.subplot(150+k+1)
    _2d = X_test[v].reshape(28,28)
    plt.title(f'Predicted Label: {test_predict[v]}')
    plt.imshow(_2d)
plt.show()


# ----

# ### Making predictions for Test Dataset using the final Model

# In[ ]:


# Predicting values for unseen Test Dataset

# scaling the features
testing_dataframe = scale(testing_dataframe)

test_predict = model.predict(testing_dataframe)


# In[ ]:


# Plotting the distribution of our prediction
d = {'ImageId': np.arange(1,test_predict.shape[0]+1), 'Label': test_predict}
dataframe_to_export = pd.DataFrame(data=d)
sns.countplot(dataframe_to_export['Label'], palette = 'icefire')


# In[ ]:


# Les't visualize our Final Model in Action for few images from Test Dataset

a = np.random.randint(1,test_predict.shape[0]+1,5)

plt.figure(figsize=(16,4))
for k,v in enumerate(a):
    plt.subplot(150+k+1)
    _2d = testing_dataframe[v].reshape(28,28)
    plt.title(f'Predicted Label: {test_predict[v]}')
    plt.imshow(_2d)
plt.show()


# In[ ]:


# Exporting the Predicted values for evaluation at Kaggle
dataframe_to_export.to_csv(path_or_buf='submission.csv', index=False)


# ![image.png](attachment:image.png)

# **Summary:**
# * Used 100% of the Training Dataset to build the Model
# * There is no major difference between the accuracy of Linear SVM vs. Non-Linear SVM
# * Our final model is a Non-Linear SVM Model since at times we observed 1% improvement in accuracy
# * For our final Model, optimized values of hyperparameters are C = 10 and Gamma = 0.001
# * For our final Model, accuracy is roughly 95%
# * Predictions made using our final Model got a score of 0.94085 upon submission at Kaggle

# ----
