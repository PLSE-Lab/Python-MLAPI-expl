#!/usr/bin/env python
# coding: utf-8

# <font color = 'red'>
# <h1>Analysis of Absenteeism<h1>
#     
# <hr>

# # Introduction
# * Feature Engineering Technics
# 
# <font color = 'red'>
# * <h4>If you like, Please don't forget to UPVOTE <h4>
# 
# <br>
# <br>
# <font color = 'blue'>
# <b>Content: </b>
# 
# 1. [Load Libraries](#1)
# 1. [Load Dataset](#2)
# 1. [Basic Data Analysis](#3)
# 1. [Feature Engineering Part-1](#4)
#     * [Remove Constant Features](#5)
#     * [Remove Quasi-Constant Features](#6)
#     * [Remove Duplicate Features](#7)
#     * [Remove Correlation Feature](#8)
#     * [Remove Unnecessary Features](#9)   
# 1. [Observe Reason for Absence](#10)
# 1. [Feature Engineering Part-2](#11)
# 1. [Logistic Regression](#12)
#     * [Standardize The Data](#13)
#     * [Train-Test Split of Data](#14)
#     * [Model](#15)
#     * [Finding The Intercept & Coefficients](#16)
#     * [Save The Model](#17)  
# 
# 
#     
# <hr>
#    

# <a id = "1"></a><br>
# ## Load Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_selection import VarianceThreshold
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <a id = "2"></a><br>
# ## Load Dataset

# In[ ]:


data = pd.read_csv("/kaggle/input/employee-absenteeism-prediction/Absenteeism-data.csv")
data.head()


# <a id = "3"></a><br>
# ## Basic Data Analysis

# In[ ]:


data.describe()


# In[ ]:


data.isnull().sum()


# In[ ]:


# if you want to see all columns and raws:
# pd.options.display.max_columns = None
# pd.options.display.max_rows = None
# display(data)


# In[ ]:


data.info()


# In[ ]:


data.shape


# <a id = "4"></a><br>
# ## Feature Engineering

# <a id = "5"></a><br>
# ## Remove Constant Features

# In[ ]:


# here for simplicity I will use only numerical variables
# select numerical columns:

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data_numerical = data[numerical_vars]
data_numerical.shape


# In[ ]:


# remove constant features
constant_features = [
    feat for feat in data_numerical.columns if data_numerical[feat].std() == 0
]

len(constant_features)


# <a id = "6"></a><br>
# ## Remove Quasi-Constant Features
# * There is no Quasi-Constant Feature

# In[ ]:


# remove quasi-constant features
sel = VarianceThreshold(threshold=0.01) # 0.1 indicates 99% of observations approximately

sel.fit(data_numerical)  # fit finds the features with low variance

sum(sel.get_support()) # how many not quasi-constant?


# <a id = "7"></a><br>
# ## Remove Duplicate Features

# In[ ]:


data.duplicated().sum()


# In[ ]:


data[data.duplicated()]


# In[ ]:


data = data.drop_duplicates(keep="first").reset_index()


# In[ ]:


data.shape


# In[ ]:


data.head()


# <a id = "8"></a><br>
# ## Remove Correlation Feature
# * Drop features if they have 80% correlation 

# In[ ]:


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[ ]:


corr_features = correlation(data, 0.8)
len(set(corr_features))


# <a id = "9"></a><br>
# ## Remove Unnecessary Features

# In[ ]:


data.drop(["ID","index"],axis=1,inplace=True)
data.head()


# <a id = "10"></a><br>
# ## Observe Reason for Absence

# In[ ]:


# Reason for Absence
data["Reason for Absence"].unique()


# * 0: 'Unknown',
# * 1: 'Certain infectious and parasitic diseases',
# * 2: 'Neoplasms',
# * 3: 'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism',
# * 4: 'Endocrine, nutritional and metabolic diseases',
# * 5: 'Mental and behavioural disorders',
# * 6: 'Diseases of the nervous system',
# * 7: 'Diseases of the eye and adnexa',
# * 8: 'Diseases of the ear and mastoid process',
# * 9: 'Diseases of the circulatory system',
# * 10: 'Diseases of the respiratory system',
# * 11: 'Diseases of the digestive system',
# * 12: 'Diseases of the skin and subcutaneous tissue',
# * 13: 'Diseases of the musculoskeletal system and connective tissue',
# * 14: 'Diseases of the genitourinary system',
# * 15: 'Pregnancy, childbirth and the puerperium',
# * 16: 'Certain conditions originating in the perinatal period',
# * 17: 'Congenital malformations, deformations and chromosomal abnormalities',
# * 18: 'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified',
# * 19: 'Injury, poisoning and certain other consequences of external causes',
# * 20: 'External causes of morbidity and mortality',
# * 21: 'Factors influencing health status and contact with health services',
# * 22: 'Patient follow-up',
# * 23: 'Medical consultation',
# * 24: 'Blood donation',
# * 25: 'Laboratory examination',
# * 26: 'Unjustified absence',
# * 27: 'Physiotherapy',
# * 28: 'Dental consultation'

# <a id = "11"></a><br>
# ## Feature Engineering Part-2

# * One Hot Encoding

# In[ ]:


reason_columns = pd.get_dummies(data["Reason for Absence"],drop_first=True)
reason_columns.head()


# In[ ]:


# Make Groups For Reasons
reason_type_1 = reason_columns.loc[:,:14].max(axis=1)
reason_type_2 = reason_columns.loc[:,15:17].max(axis=1)
reason_type_3 = reason_columns.loc[:,18:21].max(axis=1)
reason_type_4 = reason_columns.loc[:,22:].max(axis=1)


# In[ ]:


# concat:
df = pd.concat([data.drop("Reason for Absence",axis=1), reason_type_1, reason_type_2, reason_type_3, reason_type_4],axis=1)
df.head()


# In[ ]:


df.columns.values


# In[ ]:


columns_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours', "Reason_1", "Reason_2", "Reason_3", "Reason_4"]
df.columns = columns_names
df.head()


# In[ ]:


df.columns.values


# In[ ]:


# reorder columns
column_names_reordered = ['Reason_1','Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense', 'Distance to Work', 'Age',
                           'Daily Work Load Average', 'Body Mass Index', 'Education',
                           'Children', 'Pets', 'Absenteeism Time in Hours']
df = df[column_names_reordered]
df.head()


# In[ ]:


# create a chechpoint: to reduce the risk of losing data
df_reason_mod = df.copy()


# In[ ]:


# date:
# df_reason_mod["Date"].apply(lambda x: x.split("/"))
df_reason_mod["Date"] = pd.to_datetime(df_reason_mod["Date"], format="%d/%m/%Y")
df_reason_mod["Date"][0:5]


# In[ ]:


# extract month
list_months = []
for i in range(len(df_reason_mod["Date"])):
    list_months.append(df_reason_mod["Date"][i].month)


# In[ ]:


df_reason_mod["Month Value"] = list_months
df_reason_mod.head()


# In[ ]:


# extract the day of the week:0,1,2,3,4,5,6
df_reason_mod["Date"][0].weekday()


# In[ ]:


def day_to_weekday(date_value):
    return date_value.weekday()


# In[ ]:


df_reason_mod["Day of the Week"] = df_reason_mod["Date"].apply(day_to_weekday)
df_reason_mod.head()


# ## Education
# * 1:High School
# * 2:Graduate
# * 3:Postgraduate
# * 4:Master or Doctor

# In[ ]:


# Education:
df_reason_mod["Education"].unique()


# In[ ]:


df_reason_mod["Education"].value_counts()


# In[ ]:


# 1 => 0
# 2,3,4 => 1
df_reason_mod["Education"] = df_reason_mod["Education"].map({1:0,2:1,3:1,4:1})


# In[ ]:


df_reason_mod["Education"].unique()


# In[ ]:


df_reason_mod["Education"].value_counts()


# <a id = "12"></a><br>
# ## Logistic Regression

# ### Create Targets

# In[ ]:


# create a checkpoint
df_model = df_reason_mod.copy()


# In[ ]:


# use median cut-off hours and making targets 
df_model["Absenteeism Time in Hours"].median()


# ### Moderately Absent <= 3
# ### Excessively Absent > 4

# In[ ]:


targets = np.where(df_model["Absenteeism Time in Hours"] > 3, 1, 0)
targets[0:10]


# In[ ]:


# add to df
df_model["Excessive Absenteeism"] = targets
df_model.head()


# In[ ]:


targets = pd.Series(targets)
targets.value_counts()


# In[ ]:


# drop Absenteeism Time in Hours
df_model.drop(["Absenteeism Time in Hours","Date"],axis=1,inplace=True)


# In[ ]:


df_model is df_reason_mod


# ### Select Inputs

# In[ ]:


unscaled_inputs = df_model.iloc[:,:-1]


# <a id = "13"></a><br>
# ## Standardize The Data
# * Omit Dummy Features

# In[ ]:


# this class is just for selecting features to standardization:

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CustomScaler(BaseEstimator,TransformerMixin): 
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# In[ ]:


unscaled_inputs.columns.values


# In[ ]:


columns_to_scale = ['Month Value','Day of the Week', 'Transportation Expense', 'Distance to Work',
                    'Age', 'Daily Work Load Average', 'Body Mass Index', 'Children', 'Pets']

# Because these features are dummy variable
# columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Education'] 


# In[ ]:


sc = CustomScaler(columns_to_scale)


# In[ ]:


scaled_inputs = sc.fit_transform(unscaled_inputs)
scaled_inputs.head()


# <a id = "14"></a><br>
# ## Train-Test Split of Data

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(scaled_inputs,targets,train_size=0.8,shuffle=True,random_state=7)


# In[ ]:


x_train.shape, y_train.shape


# In[ ]:


x_test.shape, y_test.shape


# <a id = "15"></a><br>
# ## Model

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# ### Training The Model

# In[ ]:


reg = LogisticRegression()


# In[ ]:


reg.fit(x_train,y_train)


# In[ ]:


reg.score(x_train,y_train)


# ### Manually Check The Accuracy

# In[ ]:


model_outputs = reg.predict(x_train)


# In[ ]:


model_outputs == y_train


# In[ ]:


np.sum(model_outputs == y_train)


# In[ ]:


model_outputs.shape


# In[ ]:


np.sum(model_outputs == y_train) / model_outputs.shape[0]


# In[ ]:


np.sum(model_outputs == y_train) / model_outputs.shape[0] == reg.score(x_train,y_train)


# <a id = "16"></a><br>
# ## Finding The Intercept & Coefficients

# In[ ]:


reg.intercept_


# In[ ]:


reg.coef_.T


# In[ ]:


# Make df to show better:
feature_name = unscaled_inputs.columns
summary_table = pd.DataFrame(data=feature_name,columns=["Feature Name"])
summary_table["Coefficient"] = reg.coef_.T
summary_table


# In[ ]:


# add intercept:
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ["Intercept", reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table


# * Whichever weights/coefficient is bigger, its correspending feature is more important.
# 
# <hr> 

# ### Interpreting The Logistic Regression Coeff.

# * log(odds) = intercept + b1x1 + b2x2 + b3x3 + .... + b14x14

# In[ ]:


summary_table["Odds_ratio"] = np.exp(summary_table.Coefficient)
summary_table


# In[ ]:


summary_table.sort_values("Odds_ratio", ascending=False)


# ### A feature is not particularly important:
# * if its coeff is around 0
# * if its odds ratio is around 1
# 
# #### So you can consider to drop these features: Education, Month Value, Distance to Work
# <hr>

# ### Interpreting The Important Features

# * Reason_1 : various diseases
# * Reason_2 : pragnancy and giving birth
# * Reason_3 : poising
# * Reason_4 : light diseases

# ### Test The Model
# * Often the test accuracy is 10-20% lower than the train accuracy (due to overfitting) 

# In[ ]:


reg.score(x_test, y_test)


# In[ ]:


# predict_proba(x) : returns the probability estimates for all possible outputs
predict_proba = reg.predict_proba(x_test)
predict_proba


# In[ ]:


# see sum is 1
0.72033435 + 0.27966565, 0.87854892 + 0.12145108


# In[ ]:


# 1. column: probality of being 0
# 2. column: probality of being 1
predict_proba.shape


# * if the probality is:
#   * below 0.5, it places a 0
#   * above 0.5, it places a 1

# In[ ]:


predict_proba[:,1]


# <a id = "17"></a><br>
# ## Save The Model

# In[ ]:


import pickle


# In[ ]:


# model: file name / wb: write bites / dump:save
with open("model", "wb") as file:
    pickle.dump(reg,file)


# ### Save The Scale

# In[ ]:


with open("model_2", "wb") as file_2:
    pickle.dump(sc,file_2)


# ### See The Files

# In[ ]:


from IPython.display import FileLink, FileLinks
FileLinks('.') #lists all downloadable files on server


# In[ ]:




