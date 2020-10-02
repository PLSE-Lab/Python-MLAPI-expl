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
import time
import csv
import traceback
import pandas as pd
import numpy as np
import dask.dataframe as dd #reading large datasets out of memory

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:



#import the files with appropriate encoding

application = pd.read_csv("../input/application_train.csv")
bureau = pd.read_csv("../input/bureau.csv")
bureau_balance = pd.read_csv("../input/bureau_balance.csv")
credit_card_balance = pd.read_csv("../input/credit_card_balance.csv",encoding='utf-8')
HomeCredit_col = pd.read_csv("../input/HomeCredit_columns_description.csv",encoding='latin-1')
#installments_payments = pd.read_csv("../input/installments_payments.csv")
POS_CASH_balance = pd.read_csv("../input/POS_CASH_balance.csv")
previous_application = pd.read_csv("../input/previous_application.csv")
application_test = pd.read_csv("../input/application_test.csv")


# In[ ]:


application.head()


#  <h3 style="color: blue;"> Have a glanz at the data</h4>

# In[ ]:


application_test.head(2)


# In[ ]:


HomeCredit_col.head(219) #The description file !!!


# <h2 style="color: blue;"> Selecting the most interesting features </h2>
#  
#  Lets select the most interesting features since we cannot possibly analyse everything manually. 
#  
# Idea: Why not using tensorflow in a second part to identify the most important weights using a DNN classifier?

# In[ ]:


def get_description(keyword_list): #Let's make a function that returns the description for selected items only
    
    description = HomeCredit_col[['Row','Table','Description']]   
    df = description.loc[description['Row'].str.contains(keyword_list[0], case=False)]
    for i in range(1, len(keyword_list)):
        df = df.append(description.loc[description['Row'].str.contains(keyword_list[i], case=False)])
    return(df)


# In[ ]:


list(application.columns.values) #Let's see whats in this first application file...


# In[ ]:


#These items sound the most interesting

keyword_list_application = ['SK_ID_CURR','TARGET','CODE_GENDER','DAYS_EMPLOYED','CNT_CHILDREN',
                                  'DAYS_BIRTH','AMT_INCOME_TOTAL','NAME_INCOME_TYPE',
                                  'NAME_HOUSING_TYPE','DAYS_LAST_PHONE_CHANGE']


# In[ ]:


#Check if the description matches with what we thought it would be

get_description(keyword_list_application)


# In[ ]:


keyword_list_bureau = ['SK_ID_CURR', 'CREDIT_ACTIVE','AMT_CREDIT_SUM', 'CREDIT_DAY_OVERDUE']
get_description(keyword_list_bureau)


# In[ ]:


selected_col_bureau = bureau[keyword_list_bureau]


# In[ ]:


#create a new dataframe with the selected features only
selected_col_application = application[keyword_list_application] 


# In[ ]:


list(previous_application.columns.values)


# In[ ]:


keyword_list_previous_application = ['SK_ID_CURR', 'RATE_INTEREST_PRIMARY', 'AMT_CREDIT','CODE_REJECT_REASON' ]
get_description(keyword_list_previous_application)


# In[ ]:


selected_col_prev_app = previous_application[keyword_list_previous_application]


#  <h2 style="color: blue;"> Merging the selected columns</h2>

# In[ ]:


merged = selected_col_application.merge(selected_col_bureau, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='inner')
merged = merged.merge(selected_col_prev_app, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='inner')
merged = merged.drop_duplicates('SK_ID_CURR', keep='first')

merged.head()


#  <h2 style="color: blue;"> Creating new features </h2>

# In[ ]:


merged['Age'] = merged['DAYS_BIRTH']/-365
merged['Phone_age'] = merged['DAYS_LAST_PHONE_CHANGE']/-365
merged.drop('DAYS_LAST_PHONE_CHANGE', axis = 1)
merged.drop('DAYS_BIRTH', axis = 1)

merged.head()


# In[ ]:


merged.describe()


# In[ ]:


merged['Family_size'] = 0 
merged.loc[merged['CNT_CHILDREN'] > 0 , 'Family_size'] = 1 #Normal
merged.loc[merged['CNT_CHILDREN'] > 3 , 'Family_size'] = 2 #Large


# In[ ]:


merged['Income/credit'] = merged['AMT_INCOME_TOTAL']/merged['AMT_CREDIT_SUM']


# In[ ]:


merged['Income_category'] = "Low"
merged.loc[merged['AMT_INCOME_TOTAL'] > 112500 , 'Income_category'] = "Below_average"
merged.loc[merged['AMT_INCOME_TOTAL'] > 157500 , 'Income_category'] = "Above_average"
merged.loc[merged['AMT_INCOME_TOTAL'] > 202500 , 'Income_category'] = "High"
merged.reset_index(drop = True)
merged.head()


#  <h1 style="color: blue;"> II. Anlysing</h1>
#  
# First, whats the matter? The company we're analysing data for is oviously trying to predict the column "TARGET" which is a categorical feature. Target = 1 are clients with payment difficulties. 
#  
# In our analysis, we will focus on the factors that determine more likely that the column TARGET is equal to 1. 
# 
# 
# 
# Let's start with the selected colums first.

# In[ ]:


merged.describe()


# In[ ]:


frequency = merged.describe()['TARGET'][1] # since we have only zeros and ones the mean is the frequency
print("Frequency of Target = 1 is : ", frequency)


# ### Analyze by pivoting features
# To confirm some of our observations and assumptions, we can quickly analyze our feature correlations by pivoting features against each other. We can only do so at this stage for features which do not have any empty values
# 
# - <strong>Income category </strong>: the 4 different salary categories we created earlier have surprisingly all the same correlation, not giving us much informations
# - <strong>Gender </strong> : low correlation, man have a slitly higher risk of defaulting than women
# - <strong>Income type </strong> : low correlation, possible error with "student" and "unemployed", considering removing those
# - <strong>Housing type</strong> : perceptible correlation (>0.1) "Rented appartement" and "living with parents" seem to be positively correlated with defaulting
# - <strong> NB of children </strong> : Missleading, we have an intersting correlation for 6 children but not 7 and 8 and a correlation of 1 for 11 and 9 children, considering dropping this feature
# - <strong> Code reject reason </strong> : SCOFR, LIMIT, SYSTEM, HC have all correlation > 0.1 on one hand and the other reject reasons < 0.094

# In[ ]:


merged[['Income_category', 'TARGET']].groupby(['Income_category'], as_index=False).mean().sort_values(by='TARGET', ascending=False)


# In[ ]:


merged[['CODE_GENDER', 'TARGET']].groupby(['CODE_GENDER'], as_index=False).mean().sort_values(by='TARGET', ascending=False)


# In[ ]:


merged[['NAME_INCOME_TYPE', 'TARGET']].groupby(['NAME_INCOME_TYPE'], as_index=False).mean().sort_values(by='TARGET', ascending=False)


# In[ ]:


merged[['NAME_HOUSING_TYPE', 'TARGET']].groupby(['NAME_HOUSING_TYPE'], as_index=False).mean().sort_values(by='TARGET', ascending=False)


# In[ ]:


merged[['CNT_CHILDREN', 'TARGET']].groupby(['CNT_CHILDREN'], as_index=False).mean().sort_values(by='TARGET', ascending=False)


# In[ ]:


merged[['CODE_REJECT_REASON', 'TARGET']].groupby(['CODE_REJECT_REASON'], as_index=False).mean().sort_values(by='TARGET', ascending=False)


# ### Analyze by visualizing data
# 
# Now we can continue confirming some of our assumptions using visualizations for analyzing the data.
# 
# <strong>Correlating numerical features</strong>
# 
# Histogram chart is useful for analyzing continous numerical variables like Age where banding or ranges will help identify useful patterns. The histogram can indicate distribution of samples using automatically defined bins or equally ranged bands. 
# 
# - <strong>Age </strong>: we see that most payement difficulties are between age 20 and 45
# - <strong>Phone age </strong>: correlation seems relatively low, yet people changing phones the most frequently have a higher defaulting risk
# - <strong>Gender </strong>: On the defaulting side, both men and women seems having the same risk distribution. Yet considering the good payers, women have two hills, with a local minima between 40 and 50 years compared to men who have a continous "blobby formation" formation but degressive after 40 years old 

# In[ ]:


g = sns.FacetGrid(merged, col='TARGET')
g.map(plt.hist, 'Age', bins=20)


# In[ ]:


g = sns.FacetGrid(merged, col='TARGET')
g.map(plt.hist, 'Phone_age', bins=20)


# In[ ]:


grid = sns.FacetGrid(merged, col='TARGET', row='CODE_GENDER', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# In[ ]:


grid = sns.FacetGrid(merged, row='Income_category', size=2.2, aspect=5)
grid.map(sns.pointplot, 'NAME_HOUSING_TYPE', 'TARGET', 'CODE_GENDER', palette='deep')
grid.add_legend()


# ### Correlating categorical and numerical features
# 
# 
# 

# In[ ]:


grid = sns.FacetGrid(merged, row='Family_size', col='TARGET', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'CODE_GENDER', 'AMT_CREDIT_SUM', alpha=.5, ci=None)
grid.add_legend()


#  <h1 style="color: blue;"> III. Predicting </h1>
#  
#  We will use our data set and split it in test and train (again) since we have lots of rows already in the existing train set and since we must have had done the same operations on the test set and the train set. 
#  
#  Go for tensorflow and moreover the high level api KERAS

# In[ ]:


import os
import csv
import traceback
import shutil
import tensorflow as tf
from tensorflow import keras


from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# ### Feature scaling

# In[ ]:


def scale(feature):
    min_x = merged.min()[feature]
    max_x = merged.max()[feature]
    
    merged[feature] = (merged[feature] - min_x) / (max_x - min_x)


# In[ ]:


scale('DAYS_EMPLOYED')
scale('CNT_CHILDREN')
scale('AMT_INCOME_TOTAL')
scale('DAYS_LAST_PHONE_CHANGE')
scale('AMT_CREDIT_SUM')
scale('CREDIT_DAY_OVERDUE')
scale('RATE_INTEREST_PRIMARY')
scale('AMT_CREDIT')
scale('Age')
scale('Family_size')
scale('Income/credit')
scale('Phone_age')
scale('DAYS_BIRTH')


# In[ ]:


cat_columns = merged.select_dtypes(['object']).columns
cat_columns


# In[ ]:


merged[cat_columns] = merged[cat_columns].astype('category')

merged['NAME_HOUSING_TYPE'] = merged['NAME_HOUSING_TYPE'].cat.codes
merged['NAME_INCOME_TYPE'] = merged['NAME_INCOME_TYPE'].cat.codes
merged['CODE_GENDER'] = merged['CODE_GENDER'].cat.codes
merged['CREDIT_ACTIVE'] = merged['CREDIT_ACTIVE'].cat.codes
merged['CODE_REJECT_REASON'] = merged['CODE_REJECT_REASON'].cat.codes
merged['Income_category'] = merged['Income_category'].cat.codes


# In[ ]:


merged.set_index('SK_ID_CURR', inplace=True)


# In[ ]:


merged = merged.dropna() #dropping all null values
#merged = merged.reset_index(drop = True)
merged = merged.drop('RATE_INTEREST_PRIMARY',axis = 1)


# ### Building the model
# 
# We separate the labels from the test data.
# 
# We start 4 well know machine learing algorithms and compare their score before going for the Decision Tree model. 
# 
# We achieve a score of around 85% true positives meaning we can predict correctly 85% of the time correctly if someone is a bad payer, and we predict more than 99% of the time a good payer.
# 
# The model is highly improvable if we would use more of the avilable features, better preanalysis and eventually an other more complex model. 
# 
# <strong> Yet we can be proud of the result since we can eliminate 85% of the bad payers conserving most of the good ones as clients </strong>.

# In[ ]:


np.random.seed(seed=1) #makes result reproducible
msk = np.random.rand(len(merged)) < 0.8
traindf = merged[msk]
evaldf = merged[~msk]


# In[ ]:


X_train = traindf.drop('TARGET', axis = 1)

Y_train = traindf['TARGET']

X_test = evaldf.drop('TARGET', axis = 1)
Y_test = evaldf['TARGET']


# In[ ]:


X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


#KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

acc_knn_test = round(knn.score(X_test, Y_test) * 100, 2)
acc_knn_test

print('Train:',acc_knn, 'Test:', acc_knn_test)


# In[ ]:


#Support vector machine

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

acc_svc_test = round(svc.score(X_test, Y_test) * 100, 2)
acc_svc_test

print('Train:',acc_svc, 'Test:', acc_svc_test)


# In[ ]:





# In[ ]:





# In[ ]:


#random_forest

random_forest = RandomForestClassifier(n_estimators=120)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

acc_random_forest_test = round(random_forest.score(X_test, Y_test) * 100, 2)
acc_random_forest_test

print('Train:',acc_random_forest, 'Test:', acc_random_forest_test)


# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree_Test = round(decision_tree.score(X_test, Y_test) * 100, 2)
acc_decision_tree_Test

print('Train:',acc_decision_tree, 'Test:', acc_decision_tree_Test)


# In[ ]:


submission = pd.DataFrame({
        "Real": evaldf["TARGET"],
        "Prediction": Y_pred
    })


# In[ ]:


submission.head()


# In[ ]:


submission.describe()


# In[ ]:




