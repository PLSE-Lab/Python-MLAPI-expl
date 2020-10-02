#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random 

random.seed(19)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Task
# 
# * Predict confirmed COVID-19 cases among suspected cases
# 
# Based on the results of laboratory tests commonly collected for a suspected COVID-19 case during a visit to the emergency room, would it be possible to predict the test result for SARS-Cov-2 (positive/negative)?

# In[ ]:


# Load data

df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')


# In[ ]:


pd.set_option('display.max_columns', 110)


# # Exploratory analysis and data manipulation

# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.isna().sum()[1:30]


# In[ ]:


#original_columns = df.columns.values

#for i in range(len(df.index)) :
#    print("Nan in row ", i , " : " ,  df[original_columns[6:110]].iloc[i].isnull().sum())


# We have a high number of missing values. This characteristic is common to almost all dataset lines.
# 
# How to deal with them? 

# In[ ]:


df.groupby("SARS-Cov-2 exam result").count()


# In[ ]:


df.groupby("SARS-Cov-2 exam result").count()


# Given the output, we can observe and discuss a few things:
#    1. The interest variable is proportionally unbalanced. Positive results represent approximately 10% of the total tested (before dealing with NAs).
#    
#    2. We have many variables being mostly composed of missing values. Some of them containing only NAs. (Ex: 'D-Dimner' or 'Mycoplasma pneumoniae')
#    
#    3. Imputing values (such zeros or averages) can cause problems in modeling and results, given the characteristics of the variables.
#    
#    4. The dataset has data variability problems. Because of this, variables that have a not so large number of NAs were selected.
#    
#    5. The priority is to manipulate the data, maintaining the group of variables that have similar NAs. In this case, it is a range of columns between 'Hematocrit' and 'Red blood cell distribution width (RDW)'.

# In[ ]:


set_columns = ['Patient ID', 'Patient age quantile', 'SARS-Cov-2 exam result',
       'Patient addmited to regular ward (1=yes, 0=no)',
       'Patient addmited to semi-intensive unit (1=yes, 0=no)',
       'Patient addmited to intensive care unit (1=yes, 0=no)',
       'Hematocrit', 'Hemoglobin', 'Platelets', 'Mean platelet volume ',
       'Red blood Cells', 'Lymphocytes','Mean corpuscular hemoglobin concentration\xa0(MCHC)',
       'Leukocytes', 'Basophils', 'Mean corpuscular hemoglobin (MCH)',
       'Eosinophils', 'Mean corpuscular volume (MCV)', 'Monocytes',
       'Red blood cell distribution width (RDW)','Respiratory Syncytial Virus','Influenza A',
       'Influenza B','Parainfluenza 1','CoronavirusNL63','Rhinovirus/Enterovirus',
       'Coronavirus HKU1', 'Parainfluenza 3','Chlamydophila pneumoniae', 'Adenovirus', 'Parainfluenza 4',
       'Coronavirus229E', 'CoronavirusOC43', 'Inf A H1N1 2009',
       'Bordetella pertussis', 'Metapneumovirus', 'Parainfluenza 2']


# In[ ]:


df = df[set_columns]
df.head()


# In[ ]:


df.columns = [x.lower().strip().replace(' ','_') for x in df.columns]
df.columns


# Categorical data to dummy:
# 1. Variable of interest: 1 is positive and 0 negative.
# 
# 2. Non-priority variables (previously indicated): 1 represents 'detected' and 0 otherwise.

# In[ ]:


df['sars-cov-2_exam_result'] = [1 if a == 'positive' else 0 for a in df['sars-cov-2_exam_result'].values]

for i in df.columns[20:]:
    df[i] = [1 if a == 'detected' else 0 for a in df[i].values]
    
df.head(20)


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


df = df.dropna()


# In[ ]:


df.describe()


# In[ ]:


df.groupby("sars-cov-2_exam_result").count()


# 1. We have columns of zeros again. They will be dropped.
# 
# 2. We continue with the unbalanced interest variable (before discarding NAs, the positives corresponded to approximately 10%, now it is approximately 13%).
# 
# 3. Some variables have many zeros (An example of this is zero going to the 75th quantile). These variables will be discarded.

# In[ ]:


df = df[df.columns[:20]]


# # Correlagram

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

corrmat = df.corr()
f, ax = plt.subplots(figsize=(16, 9))
sns.heatmap(corrmat, vmax=.8, square=True)


# We can observe moderate to high autocorrelation between the variables: 'hematocrit', 'hemoglobin', 'red_blood_cells', 'mean_corpuscular_hemoglobin (mch)' and 'mean_corpuscular_hemoglobin_(mch)'. Thereby,the first attempt tries to use a model with greater parsimony.

# # Features selection

# Preference: parsimonious models
#     
# Variable selection by correlation. 
# 
# Number of features (10) is ad-hoc.

# In[ ]:


corrmat_v1 = corrmat.nlargest(10, 'sars-cov-2_exam_result')

features = corrmat_v1.index.values.tolist()

sns.heatmap(df[features].corr(), yticklabels=features, xticklabels=features, square=True);


# In[ ]:


# features

print('Features: ',features[1:])
print('Target: ',features[0])


# # Split the data

# In[ ]:


from sklearn.model_selection import train_test_split

X = df[features[1:]]
Y = df[features[0]]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=7)

print('Number of training observations: ',len(X_train))
print('Number of test observations: ',len(X_test))


# # Methods and Metrics

# Methods:
#     1. Logistic
#     2. Decision Tree;
#     3. Decision Tree with CrossValidation;
#     4. RandomForest;
#     5. XGBoost.
#     
# Metrics:
#     1. Accuracy;
#     2. Confusion matrix.

# In[ ]:


# Metrics

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def plot_confusion_matrix(cm):
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax,fmt='g',cmap='Blues');
    ax.set_xlabel('Predict');ax.set_ylabel('True'); 
    ax.set_title('Confusion matrix'); 
    ax.xaxis.set_ticklabels(['Negative', 'Positive']); ax.yaxis.set_ticklabels(['Negative','Positive']);


#  

# ## Logistic

# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

y_pred = logreg.predict(X_test)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(Y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


cf_matrix = confusion_matrix(predictions,Y_test)

plot_confusion_matrix(cf_matrix)


# In[ ]:


print(classification_report(Y_test, predictions))


#  

# ## Decision Tree

# In[ ]:


from sklearn import tree

model_tree = tree.DecisionTreeClassifier()
model_tree = model_tree.fit(X_train, Y_train)

y_pred = model_tree.predict(X_test)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(Y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


fig, ax = plt.subplots(figsize=(15,15))
tm = tree.plot_tree(model_tree, ax=ax)
plt.show()


# In[ ]:


cf_matrix = confusion_matrix(predictions,Y_test)

plot_confusion_matrix(cf_matrix)


# In[ ]:


print(classification_report(Y_test, predictions))


#  

# > ## Decision Tree with CrossValidation

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


tree_cls = tree.DecisionTreeClassifier()
scores = cross_val_score(tree_cls, X, Y,
                        scoring='neg_mean_squared_error', cv=10)

y_pred = cross_val_predict(tree_cls, X, Y, cv=10)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(Y, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


cf_matrix = confusion_matrix(predictions,Y)

plot_confusion_matrix(cf_matrix)


# In[ ]:


print(classification_report(Y, predictions))


#   

# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(max_depth=2, random_state=7)
model_rf = model_rf.fit(X_train, Y_train)

y_pred = model_rf.predict(X_test)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(Y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


cf_matrix = confusion_matrix(predictions,Y_test)

plot_confusion_matrix(cf_matrix)


# The model did not predict positives!

# In[ ]:


#print(classification_report(Y_test, predictions))


#  

# ## XGBoost 

# In[ ]:


from xgboost import XGBClassifier

model_xgb = XGBClassifier()

model_xgb.fit(X_train, Y_train)

y_pred = model_xgb.predict(X_test)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(Y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


cf_matrix = confusion_matrix(predictions,Y_test)

plot_confusion_matrix(cf_matrix)


# In[ ]:


print(classification_report(Y_test, predictions))


#   

# ## Final Comment
# 
# Given the results of the tested methods, we had good accuracy. This result would be expected given the natural proportion 
# of the variable of interest. If confusion matrices are observed, errors are even worse in correctly identifying "positive" (true positives and false negatives).
# 
# Regarding the best method, Logistic had slightly better results than the others, when the precision in relation to the positive (1) and the accuracy is observed.
