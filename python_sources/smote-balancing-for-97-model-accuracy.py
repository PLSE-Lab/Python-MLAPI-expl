#!/usr/bin/env python
# coding: utf-8

# The guided approach explained in this notebook will help you to understand how you should design and approach Data Science problems. Though there are many ways to do the same analysis, I have used the codes which I found more efficient and helpful.
# 
# As we can see that dataset is not balanced, we will leverage SMOTE technique to oversample the data and evaluate the impact on model performance.
# 
# **The idea is just to show you the path, try your own ways and share the same with others.**

# <h1 id="tocheading">Table of Contents</h1>
# <div id="toc"></div>

# In[ ]:


get_ipython().run_cell_magic('javascript', '', "$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")


# # What would be the workflow?
# 
# I will keep it simple & crisp.
# 
# This will help you to stay on track. So here is the workflow.
# 
# **Problem Identification**
# 
# **What data do we have?**
# 
# **Exploratory data analysis**
# 
# **Data preparation including feature engineering**
# 
# **Developing a model**
# 
# **Model evaluation**
# 
# **Conclusions**
# 
# That's all you need to solve a data science problem.

# # Problem Identification

#  
# ![Prob%20Ident.png](attachment:Prob%20Ident.png)
# **Best Practice -** The most important part of any project is correct problem identification. Before you jump to "How to do this" part like typical Data Scientists, understand "What/Why" part.  
# Understand the problem first and draft a rough strategy on a piece of paper to start with. Write down things like what are you expected to do & what data you might need or let's say what all algorithms you plan to use. 
# 
# The goal of this kernel is to treat this dataset to explore classification algorithms. The target variable will be converted to a categories such as 'Good' and 'Bad' based on 10 point scale suggested by dataset uploader. 
# 
# So it is a classification problem and you are expected to predict good as 1 and bad as 0.

# # What data do we have?

# 
# ![Data.jpg](attachment:Data.jpg)
# 
# Let's import necessary libraries & bring in the datasets in Python environment first. Once we have the datasets in Python environment we can slice & dice the data to understand what we have and what is missing.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# Import the basic python libraries

import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler, LabelEncoder
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='white', context='notebook', palette='deep')
import warnings
warnings.filterwarnings('ignore')

# Read the datasets
train = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
train.info()


# # EDA & Visualization
# 

# Let's start with finding the number of missing values. 
# 
# Use the groupby/univariate/bivariate analysis method to compare the distribution across Train data

# In[ ]:


# Check missing values in train data set
train_na = (train.isnull().sum() / len(train)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)[:30]
miss_train = pd.DataFrame({'Train Missing Ratio' :train_na})
miss_train.head()


# There are no missing values

# Let's define a function for quick plotting

# In[ ]:


def plot(var):
    fig = plt.figure(figsize = (10,6))
    sns.barplot(x = 'quality', y = var , data = train)
    print(train[[var, "quality"]].groupby(['quality']).mean().sort_values(by=var, ascending=False))


# **fixed acidity**
# 
# Let's look at the distribution.

# In[ ]:


plot('fixed acidity')


# **Volatile Acidity**
# 
# Let's look at the distribution.

# In[ ]:


plot('volatile acidity')


# The chart suggests that low quality wines have high volatile acidity 

# **Citric Acid**

# In[ ]:


plot('citric acid')


# The chart suggests that low quality wines have low citric acid 

# **residual sugar**

# In[ ]:


plot('residual sugar')


# Not very informative feature 

# **chlorides**

# In[ ]:


plot('chlorides')


# Based on data above, better quality wines will have less chlorides

# **free sulfur dioxide**

# In[ ]:


plot('free sulfur dioxide')


# Not very informative, however, it seems average quality wines have higher values of 'free sulphur dioxide'

# **total sulfur dioxide**

# In[ ]:


plot('total sulfur dioxide')


# **density**

# In[ ]:


plot('density')


# Not very informative as all quality groups have almost same average density

# **pH**

# In[ ]:


plot('pH')


# Not very informative as all quality groups have almost same pH

# **Sulphates**

# In[ ]:


plot('sulphates')


# This features seems to be directly related to quality of wine as good quality wines have more sulphates

# **Alcohol**

# In[ ]:


plot('alcohol')


# This features also seems to be directly related to quality of wine as good quality wines have more alcohol

# # Feature engineering
# ![FE.png](attachment:FE.png)

# From Feature Engineering perspective there is not much room to experimen except following two steps - 
# * Convert the 'Quality' column values to 1 and 0 using bins and label encoders
# * Standardize the data for ease of calculations
# 

# Creating binary classification for target variable

# In[ ]:


bins = (2, 6.5, 8)
target_groups = ['BadQ', 'GoodQ']
train['quality'] = pd.cut(train['quality'], bins = bins, labels = target_groups)
label_quality = LabelEncoder()
train['quality'] = label_quality.fit_transform(train['quality'])


# In[ ]:


train['quality'].value_counts()


# Split data into training and test sets.
# 
# First, let's separate our target (y) features from our input (X) features:

# In[ ]:


y = train.quality
X = train.drop('quality', axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)


# In[ ]:


# standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


# # Creating a Model

# In[ ]:


# Import the required libraries
from sklearn.svm import SVC
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier


# **Cross Validation Strategy**
# ![CV.png](attachment:CV.png)
# 
# Cross Validation is one of the most powerful tool in Data Scientist's tool box. It helps you to understand the performance of your model and fight with overfitting. As we all know that Learning the model parameters and testing it on the same data is a big mistake. Such a model would have learned everything about the training data and would give result in a near perfect test score as it has already seen the data. The same model would fail terribly when tested on unseen data. This situation is called overfitting. To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set X_test, y_test. 
# 
# The general approach is as follows:
# 
# 1. Split the dataset into k groups
# 2. For each unique group:
#         a. Keep one group as a hold out or test data set
#         b. Use the remaining groups as training data set
#         c. Build the model on the training set and evaluate it on the test set
#         d. Save the evaluation score 
# 3. Summarize the performance of the model using the sample of model evaluation scores
# 
# You can access following link and read about Cross Validation in detail.
# 
# https://medium.com/datadriveninvestor/k-fold-cross-validation-6b8518070833
# https://www.analyticsvidhya.com/blog/2018/05/improve-model-performance-cross-validation-in-python-r/

# In[ ]:


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)


# Now we have the training and test datasets available and we can start training the model. We will build couple of base models and then will use Grid Search method to optimize the parameters. There are several classification you can select.
# We are trying following to develop a baseline - 
# 
#         1. K Nearest Neighbour
#         2. Linear Discriminant Analysis
#         3. Support Vector Classifier
#         4. Multi-layer Perceptron classifier
#         5. Extra Trees Classifier
#         6. Logistic Regression
#         7. Decision Trees
#         8. Random Forest
#         9. Gradient Boosting Classifier
#         10. AdaBoost Classifier
# 

# In[ ]:


# Modeling differents algorithms. 

random_state = 2
classifiers = []

classifiers.append(KNeighborsClassifier())
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(SVC(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))


cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train_scaled, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,
                       "Algorithm":["SVC",
                                    "AdaBoost",
                                    "ExtraTrees",
                                    "KNeighboors",
                                    "DecisionTree",
                                    "RandomForest",
                                    "GradientBoosting",
                                    "LogisticRegression",
                                    "MultipleLayerPerceptron",
                                    "LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# # Model Evaluation
# ![model.png](attachment:model.png)

# 
# 
# Evaluating multiple models using GridSearch optimization method. 
# 
# Hyper-parameters are key parameters that are not directly learnt within the estimators. We have to pass these as arguments. Different hyper parameters can result in different model with varying performance/accuracy. To find out what paparmeters are resulting in best score, we can use Grid Search method and use the optimum set of hyper parameters to build and select a good model.
# 
# A search consists of:
# 
# 1. an estimator (regressor or classifier)
# 2. a parameter space;
# 3. a method for searching or sampling candidates;
# 4. a cross-validation scheme; and
# 5. a score function.

# **AdaBoost classifier** 
# 
# Adaboost begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

# In[ ]:


# Adaboost
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
                  "base_estimator__splitter" :   ["best", "random"],
                  "algorithm" : ["SAMME","SAMME.R"],
                  "n_estimators" :[1,2],
                  "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsadaDTC.fit(X_train_scaled,y_train)

# Predicting on test data
y_pred_rf = gsadaDTC.predict(X_test_scaled)
print("Training Accuracy - AdaBoost Model: ", gsadaDTC.score(X_train_scaled,y_train))
print('Testing Accuarcy - AdaBoost Model: ', gsadaDTC.score(X_test_scaled, y_test))

# making a classification report
cr = classification_report(y_test,  y_pred_rf)
print(cr)


# ***Using Grid Search, the AdaBoost model gives training accuracy as 100% and testing accuracy as 88%***

# **ExtraTrees Classifier**
# 
# ET is a meta estimator that fits a number of randomized decision trees on various sub-samples of the dataset and then uses averaging method to improve the predictive accuracy and control over-fitting.

# In[ ]:


#ExtraTrees 
ExtC = ExtraTreesClassifier()

## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsExtC.fit(X_train_scaled,y_train)

# Predicting on test data
y_pred_rf = gsExtC.predict(X_test_scaled)
print("Training Accuracy - ExtraTreeClassifier: ", gsExtC.score(X_train_scaled,y_train))
print('Testing Accuarcy - ExtraTreeClassifier: ', gsExtC.score(X_test_scaled, y_test))

# Making a classification report
cr = classification_report(y_test,  y_pred_rf)
print(cr)


# ***Using Grid Search, the ExtraTreeClassifier model gives training accuracy as 100% and testing accuracy as 91%***

# **Random Forest Classifier**
# 
# Similar to Extra Tree Classifier a Random Forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).
# 
# How ET differes from RF - 
# 
# 1) When choosing variables at a split, samples are drawn from the entire training set instead of a bootstrap sample of the training set.
# 
# 2) Splits are chosen completely at random from the range of values in the sample at each split.

# In[ ]:


# RFC Parameters tunning 
RFC = RandomForestClassifier()

## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsRFC.fit(X_train_scaled,y_train)

# Predicting on test data
y_pred_rf = gsRFC.predict(X_test_scaled)
print("Training Accuracy - RandomForestClassifier: ", gsRFC.score(X_train_scaled,y_train))
print('Testing Accuarcy - RandomForestClassifier: ', gsRFC.score(X_test_scaled, y_test))

# Making a classification report
cr = classification_report(y_test,  y_pred_rf)
print(cr)


# ***Using Grid Search, the Random Forest model gives training accuracy as 100% and testing accuracy as 90%***

# **Gradient Boosting**
# 
# Gradient boosting is one of the most powerful techniques for building predictive models. Boosting is a method of converting weak learners into strong learners by building an additive model in a forward stage-wise fashion. In boosting, each new tree is a fit on a modified version of the original data set.

# In[ ]:


# Gradient boosting 
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }
gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsGBC.fit(X_train_scaled,y_train)

# Predicting on test data
y_pred_rf = gsGBC.predict(X_test_scaled)
print("Training Accuracy - Gradient Boosting: ", gsGBC.score(X_train_scaled,y_train))
print('Testing Accuarcy - Gradient Boosting: ', gsGBC.score(X_test_scaled, y_test))

# Making a classification report
cr = classification_report(y_test,  y_pred_rf)
print(cr)


# ***Using Grid Search, the Gradient Boosting model gives training accuracy as 99% and testing accuracy as 90%***

# **Support Vector Machines**
# 
# SVM builds a hyperplane in multidimensional space to separate different classes. SVM generates optimal hyperplane in an iterative manner, which is used to minimize the error. The idea behind SVM is to find a maximum marginal hyperplane(MMH) that best divides the dataset into classes.

# In[ ]:


### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}
gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsSVMC.fit(X_train_scaled,y_train)

# Predicting on test data
y_pred_rf = gsSVMC.predict(X_test_scaled)
print("Training Accuracy - SVC Classifier: ", gsSVMC.score(X_train_scaled,y_train))
print('Testing Accuarcy - SVC Classifier: ', gsSVMC.score(X_test_scaled, y_test))

# Making a classification report
cr = classification_report(y_test,  y_pred_rf)
print(cr)


# ***Using Grid Search, the SVC model gives training accuracy as 98% and testing accuracy as 90%***

# # Applying Over Sampling Techniques Using SMOTE

# In[ ]:


train['quality'].value_counts()


# Post label encoding the train data has imbalanced target data which means there could be a significant difference between the majority and minority classes (1 and 0). In other words, there are too few examples of the minority class (1) for a model to effectively learn the decision boundary. One way to tackle such a situation is to oversample the minority class (GoodQ = 1 in this case). The approach involves simply duplicating the minority class examples. This is a type of data processing for the minority class is referred to as the Synthetic Minority Oversampling Technique or SMOTE for short.
# 
# Almost 10% difference between training and testing accuracy also proves that model is getting overtrained and not performing on test data as expected. In Data Science language we can say that we are experiencing 'High Bias and High Variance' cases. Let's see if SMOTE can help in address this issue as well.

# In[ ]:


# Creating new samples using SMOTE technique
from imblearn.over_sampling import SMOTE
x_resample, y_resample  = SMOTE().fit_sample(X, y.values.ravel())
print("Shape of x_resample :",x_resample.shape)
print("Shape of y_resample :",y_resample.shape)


# In[ ]:


# Testing balanced values
d=pd.DataFrame(y_resample, columns=['a']) 
d['a'].value_counts()


# Now the classes are balanced and model has enough data points to learn and create a decision boundary. Let's further split the new samples into train and test data

# In[ ]:


x_train2, x_test2, y_train2, y_test2 = train_test_split(x_resample, y_resample, test_size = 0.2, random_state = 0)
print("Shape of x_train2 :", x_train2.shape)
print("Shape of y_train2 :", y_train2.shape)
print("Shape of x_test2 :", x_test2.shape)
print("Shape of y_test2 :", y_test2.shape)


# In[ ]:


# standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train2 = sc.fit_transform(x_train2)
x_test2 = sc.transform(x_test2)


# Let's pick up one of the algoritham SVC and see how SMOTE helps in improving the model accuracy

# In[ ]:


SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}
gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsSVMC.fit(x_train2,y_train2)

# Predicting on test data
y_pred_rf = gsSVMC.predict(x_test2)
print("Training Accuracy: ", gsSVMC.score(x_train2, y_train2))
print('Testing Accuarcy: ', gsSVMC.score(x_test2, y_test2))

# Making a classification report
cr = classification_report(y_test2,  y_pred_rf)
print(cr)


# # Conclusion
# ![Conclusion.png](attachment:Conclusion.png)

# As we can clearly see, how balancing of data using SMOTE can help in improving the model accuracy. In this case **testing accuracy improved all the way to 96% from initial 90%.**
# 
# We can also notice that model is performing well on test data as well so we have the '**Low Bias and Low Variance**' case now.

# # **If you liked this notebook, Please upvote and leave a comment**
# ![Good%20Bye.png](attachment:Good%20Bye.png)
