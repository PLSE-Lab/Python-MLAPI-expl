#!/usr/bin/env python
# coding: utf-8

# **Hello, and welcome** to my Kernel which will use Graduate Admissions data and various machine learning models to predict admissions decisions. 
# 
# This kernel is mostly to showcase different skills I have been devloping regarding the implementation of pipelines and scaling techinques. 
# 
# I must point out now that I am going to approach this problem from a categorical viewpoint rather than a continuous one. In the dataset as is stands now, we are attempting to predict a proability of admission. My approach will be to categorize an indivudal as either being "admissable" or not, by assigning a cutoff at the mean admission proability. 
# 
# Let's start off by loading up the modules we need, and some basic inspection. 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/Admission_Predict.csv")
print(df.info())
print(df.describe().to_string())
print(df.columns)


# With this dataset, it is clear we are attempting to predict the probablity of admission. I'm going to take a different route. Instead of using regression to determine the chance of admission, anything above a 72.4% chance of admission (the mean value of Chance of Admit) is an "admit", and anything below is not. 
# 
# Some of the column names have spaces at the end, so be careful when trying to reproduce the results. 
# 
# Let's make that admissible column now:

# In[ ]:


def Admissible(value):
    mean = df['Chance of Admit '].mean()
    if value >= mean:
        return(1)
    else:
        return(0)

df["AdmitOrNot"] = [Admissible(x) for x in df['Chance of Admit '].values]    


# Let's take a moment to inspect the distributions of factors:

# In[ ]:


df.hist()
plt.show()


# All of these distributions appear to have good distributions, but since the relative scales are different, this dataset may be suited for various scaling techniques. All of the values are positive, but some are more categorical in nature such as "Research". 
# 
# Let's check if any are intercorrelated.
# 
# 

# In[ ]:


correlations = df.corr()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
names = df.columns
ticks = np.arange(0,len(names),1)

ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
plt.xticks(rotation=90)
ax.set_yticklabels(names)
plt.show()


# It seems there is some correlation among the "Research" category, so I will exclude it from these algorithms, as many of them require or assume linear independance. 
# 
# Now, we will set up the factors and target.

# In[ ]:


factors = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA']
X = df[factors].values
y = df["AdmitOrNot"]

print(X.shape)
print(y.shape)


# Since the data in the factors vary in their range and scale, a variety of scaling factors will be used to pre-process the data for our ML algorithms. 
# The four scalers I'll use are: 
# * Normalizer - Ccales each row to have unit norm. That is, the sum of sqaures of every element  is one.
# * StandardScaler - Divides each element by a factor such that the mean of elements is zero with unit standard deviation.
# * MinMaxScaler - Similar to Standard Scaler, but divides each element by a factor such that the range of values is specified
# * PCA - Creates a specified number of new factors which are linear combinations of original factors. 

# In[ ]:


from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

scaling = {"Normalizer" : Normalizer(norm="l2"),
           "Standard Scaler" : StandardScaler(),
           "MinMaxScaler" : MinMaxScaler(feature_range=(0,1)),
           "PCA" : PCA(n_components=3, random_state=0, whiten=False)
            }


# Now we can define the classifiyers to use. I will use a suite of classifiers based on the size of the dataset and the categorical targets we are predicting. 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

classifiers = {"K-Nearest Neighbors" : KNeighborsClassifier(n_neighbors=3),
               "Random Forest" : RandomForestClassifier(n_estimators=10, random_state=0),
               #"Support Vector Clf" : LinearSVC(penalty="l2", random_state=0), THIS ONE CAUSES ERRORS IN KAGGLE
               "Logistic Regression" : LogisticRegression(penalty="l2", random_state=0),
               "Perceptron" : Perceptron(penalty="l2", random_state=0),
               "Decision Tree" : DecisionTreeClassifier(random_state=0),
               "Naive Bayes" : GaussianNB()
              }


# I will perform a 5-fold cross validation on each model, and each model will have each type of scaling methodology. I will later store this information in a new dataframe, so I will append the results to a dictionary first. 
# 
# My procedure is as such:
# 1. Instantiate the empty dictionary
# 2. For every classifyer...
# 3. ...Instantiate a new results dictionary
# 4. ...Perform a non-scaled train/test/predict/cross-val and append results to dictionary
# 5. ---For every scaling method---
# 6. ---Instantiate an empty list
# 7. ---Append a scaler to a pipeline, then append the clasifyer to the pipeline
# 8. ---Perform a scaled train/test/predict/cross-val and append results to dictionary
# 
# 
# 

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Let's get learning
fold_splits = 5
rounding_prec = 4 # digits after the decimal
#results_dict = {"Clf Name" : {"Scaling Method":"Accuracy"}}
results_dict = {}
for clfy_name, clfy in classifiers.items(): 

    acc_dict = {}
    acc_dict["Non-scaled"] = round(cross_val_score(clfy, X, y, cv=fold_splits
                                  ).mean(), rounding_prec) 

    for scl_name, sclr in scaling.items(): 
        pipln = []
        pipln.append((scl_name, sclr)) 
        pipln.append(("clf", clfy))

        pip = Pipeline(pipln)
        acc_dict[scl_name] = round(cross_val_score(pip, X, y, cv=fold_splits
                                                  ).mean(), rounding_prec)
    results_dict[clfy_name] = acc_dict

# MAKE DF
df_ML_acc = pd.DataFrame(results_dict)
df_ML_acc.name = "Machine Learning Models Accuracy Scores"
print(df_ML_acc.to_string())
print('---*---Best Results:')
print(df_ML_acc.max())


# From the results, we can see that different models behave differently under different scaling. Overall, the algorithm with the highest accuracy was Logistic Regression with MinMaxScaler scaling procedures with 85% accuracy.  

# There are many paths to take going forward. Each model can be tuned, and different assumptions can be made when scaling data. 
# 
# **Thank you! Please consider leaving a comment. **
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:




