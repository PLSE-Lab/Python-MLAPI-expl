#!/usr/bin/env python
# coding: utf-8

# This project aims to perform a useful EDA on the data, suggest some future engineering and finally find the best model for the ML part.

# # 1. Importing libraries and reading data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train = pd.read_csv("../input/home-credit-default-risk/application_train.csv", index_col=0)
test = pd.read_csv("../input/home-credit-default-risk/application_test.csv", index_col=0)


# In[ ]:


display(train.head())
display(train.tail())


# In[ ]:


display(test.head())
display(test.tail())


# # 2. EDA
# Let's do some charting!

# In[ ]:


columns = ["TARGET", "CODE_GENDER", "NAME_CONTRACT_TYPE", "FLAG_OWN_CAR",
           "FLAG_OWN_REALTY", "CNT_CHILDREN","NAME_TYPE_SUITE",
           "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "EMERGENCYSTATE_MODE","FLAG_DOCUMENT_2",
           "FLAG_DOCUMENT_3","FLAG_DOCUMENT_4","FLAG_DOCUMENT_5","FLAG_DOCUMENT_6",
           "FLAG_DOCUMENT_7","FLAG_DOCUMENT_8","FLAG_DOCUMENT_9","FLAG_DOCUMENT_10",
           "FLAG_DOCUMENT_11","FLAG_DOCUMENT_12","FLAG_DOCUMENT_13",
           "FLAG_DOCUMENT_14","FLAG_DOCUMENT_15","FLAG_DOCUMENT_16",
           "FLAG_DOCUMENT_17","FLAG_DOCUMENT_18","FLAG_DOCUMENT_19",
           "FLAG_DOCUMENT_20","FLAG_DOCUMENT_21"]
for column in columns:
  df = pd.DataFrame(train[column].value_counts())
  plt.figure(figsize=(12,12))
  plt.pie(df[column], labels=df.index, autopct='%2.1f%%')
  plt.title(f" {column}")
  plt.show()


# ### As we can see:
# 
#  1- the target column **(labels)** in the training dataset is significantly **imbalanced**
#     
#  2- Around  **66%** of the applicants are **women** and the rest are men
#     
#  3- Around **90%** are the loan applications are **cash loans** and the rest are revolving loans
#     
#  4- **34%** of the applicants already **own a car**
#     
#  5- **30%** of the applicants already **own a home or flat**
#     
#  6- **70%** of the applicants **don't have a child**, **20%** have **1**, and **9%** have **2 children**
#     
#  7- **81%** of the applicant were **living alone** while **13%** were **family**
#  
#  8- Most of the applicants were **laborers (26%)** which were about **twice of the sales staff and core staff**
#  
#  9- **1.4%** of the applicants had an **emergency situation** while applying for the loan
#  
#  10- Most of the **columns named "Documents"** **don't have any useful data** and we can delet them in the data cleaning part
#     

# Let's plot some charts based on the number of those who did or didn't pay their loans:

# ### Gender-based :

# In[ ]:


male_zero = train.loc[(train["CODE_GENDER"]=="M")&(train["TARGET"]==0), :].count()[0]
male_one =  train.loc[(train["CODE_GENDER"]=="M")&(train["TARGET"]==1), :].count()[0]
female_zero =  train.loc[(train["CODE_GENDER"]=="F")&(train["TARGET"]==0), :].count()[0]
female_one =  train.loc[(train["CODE_GENDER"]=="F")&(train["TARGET"]==1), :].count()[0]

plt.figure(figsize=(10,5))
sns.countplot(data=train,x="CODE_GENDER", hue="TARGET", )
plt.text(-0.3,100000,male_zero)
plt.text(0.08,15000,male_one)
plt.text(0.7,190000,female_zero)
plt.text(1.1,20000,female_one)
plt.show()


# As we can see, more than **10%** of the **male applicants** **failed** to pay theri loans while **less than 10%** of **the female applicants** did so.

# ### Owning a Car:

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(data=train,x="FLAG_OWN_CAR", hue="TARGET")
plt.show()


# ### Age-based:

# In[ ]:


plt.figure(figsize=(10,4))
sns.countplot(data=train,x=pd.cut(train.DAYS_BIRTH/-365.25, bins=3,precision=0, right=True,retbins=False), hue="TARGET")
plt.xlabel("YEARS_BIRTH")
plt.show()


# As we can see, there is a **greater chance** for ** youger applicants** to fail to pay their loans which is a good information for taking precautionary measures.

# ### distribution of the employment history:

# In[ ]:


plt.figure(figsize=(10,4))
(train["DAYS_EMPLOYED"]/-365.25).plot.hist(bins=10)
plt.xlabel("YEARS_EMPLOYED")
plt.show()


# ### "Employment history"-based:

# In[ ]:


plt.figure(figsize=(14,10))
sns.countplot(data=train,x=pd.cut(train.DAYS_EMPLOYED/-365.25, bins=5,precision=0, right=True,retbins=False), hue="TARGET")
plt.xlabel("YEARS_EMPLOYED")
plt.show()


# ### Career-based:

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(data=train, x="NAME_INCOME_TYPE", hue="TARGET")
plt.show()


# ### Education-bsed:

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(data=train,x="NAME_EDUCATION_TYPE", hue="TARGET", )
plt.show()


# ### "Marriage status"-based:

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(data=train,x="NAME_FAMILY_STATUS", hue="TARGET", )
plt.show()


# ### based on the last phone number change:

# In[ ]:


plt.figure(figsize=(15,8))
sns.countplot(data=train,x=pd.cut(train.DAYS_LAST_PHONE_CHANGE/-365.25, bins=12,precision=0, right=True,retbins=False), hue="TARGET")
plt.xlabel("YEARS_LAST_PHONE_CHANGE")
plt.show()


# ### Based on Income type:

# In[ ]:


name_income_type=train.NAME_INCOME_TYPE.unique()
for name in name_income_type:
  plt.figure(figsize=(7,5))
  data=train.loc[(train.NAME_INCOME_TYPE==name), :]
  sns.countplot(data = data, x= "NAME_INCOME_TYPE", hue=train["TARGET"])
  plt.show()


# ### Mssing Values:

# In[ ]:


pd.DataFrame(train.isnull().sum(), columns=["MISSING_VALUES"]).sort_values(by="MISSING_VALUES", ascending=False).head(30)


# Wow! what a disaster in data collection! Seems we have lots of work to do!

# In[ ]:


#categorical columns
pd.DataFrame(train.select_dtypes('object').apply(pd.Series.nunique, axis = 0),
             columns=["Categ_Data"]).sort_values(by="Categ_Data", ascending=False)


# ### Categorical values:

# In[ ]:


#categorical data in test data set
pd.DataFrame(train.select_dtypes('object').apply(pd.Series.nunique, axis = 0),
             columns=["Categ_Data"]).sort_values(by="Categ_Data", ascending=False)


# These are the columns which we need to one-hot code before ML model building

# ### Anomalies

# Let's see our data anomalies by plotting some distributions:

# In[ ]:


colomns = ["HOUR_APPR_PROCESS_START", "EXT_SOURCE_1", "APARTMENTS_MODE",
           "YEARS_BEGINEXPLUATATION_MODE","DAYS_LAST_PHONE_CHANGE",
           "REGION_POPULATION_RELATIVE", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
           "DAYS_EMPLOYED", "AMT_ANNUITY", "AMT_CREDIT", "AMT_INCOME_TOTAL"
           ]

for colomn in colomns:
  plt.figure(figsize=(12,6))
  plt.subplot(1,2,1)
  train[colomn].plot(kind = "box")
  plt.subplot(1,2,2)
  train[colomn].plot(kind = "hist")
  plt.show()


# In[ ]:


train.AMT_INCOME_TOTAL.sort_values(ascending=False)


# ## 3. Data Cleaning

# In[ ]:


useless_columns = ["HOUSETYPE_MODE", "WALLSMATERIAL_MODE","FLAG_DOCUMENT_2",
                     "FLAG_DOCUMENT_4","FLAG_DOCUMENT_5","FLAG_DOCUMENT_7",
                     "FLAG_DOCUMENT_9","FLAG_DOCUMENT_10","FLAG_DOCUMENT_12",
                     "FLAG_DOCUMENT_15","FLAG_DOCUMENT_17","FLAG_DOCUMENT_19",
                   "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21","AMT_REQ_CREDIT_BUREAU_HOUR",
                   "YEARS_BUILD_MODE","ELEVATORS_MODE","ENTRANCES_MODE", "FLOORSMAX_MODE",
                     "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI",
                   "FLOORSMAX_MEDI", "FLOORSMIN_MEDI","ELEVATORS_AVG","ENTRANCES_AVG",
                   "FLOORSMAX_AVG","FLOORSMIN_AVG","LIVINGAPARTMENTS_AVG",
                   "NONLIVINGAPARTMENTS_AVG","ORGANIZATION_TYPE"]


# As we saw in the EDA part, some of the columns don't provide any data (as most of "FLAG_DOCUMENT_... ones). Some of the columns also don't contain any useful data (such as the walls materials, the elevator type and area and etc...)

# Let's delet these columns:

# In[ ]:


train = train.drop(columns = useless_columns)
test = test.drop(columns=useless_columns)
train.shape, test.shape


# Our training dataset now has 90 columns which is not too bad!

# ## 4.  Imputation

# Now that we deleted the useless columns, let's impute missing values and anomalies with reasonable values:

# I chose to fill the categorical missing values with the most frequent ones, and the numerical values with their median.
# (I chose median instead of mean to avoid affected by the anomalies)

# In[ ]:


#Fill the missing categorical data with the most frequent velue
train_categorical = train.select_dtypes(include ='object') 
train_categorical = train_categorical.apply(lambda x: x.fillna(x.value_counts().index[0]))

#Fill the missing numerical data with the median 
train_float = train.select_dtypes(include ='float64') 
train_float = train_float.fillna(train_float.median())

train_int = train.select_dtypes(include ='int64') 
train_int = train_int.fillna(train_int.median())

train_final = pd.concat([train_categorical,train_float,train_int], axis=1)
train_final["TARGET"] = train["TARGET"]
train=train_final


# In[ ]:


#fill missing categorical values in test data set with most frequent values in 
#training data set (don't use any info from test data set)

categorical_columns = test.select_dtypes(include ='object').columns
test_categorical = test.select_dtypes(include ='object') 

for column in categorical_columns:
  test_categorical[column] = test_categorical[column].fillna(train[column].value_counts().index[0])

#for numerical missing values in test data set, fill them with median value in 
#the training data set (don't use any info from test data set)
test_float = test.select_dtypes(include ='float64') 
test_float = test_float.fillna(train_float.median())

test_int = test.select_dtypes(include ='int64') 
test_int = test_int.fillna(train_int.median())

test_final = pd.concat([test_categorical,test_float,test_int], axis=1)
test=test_final


# The other point was that I chose to not to impute the test dataset with its own values and get the help of the median values in the training data. that's how we can prevent overfitting

# In[ ]:


train.shape,test.shape


# ### Replacing anomalies with reasonable values

# In[ ]:


train.loc[train.DAYS_EMPLOYED==train.DAYS_EMPLOYED.max(), :].NAME_INCOME_TYPE.unique()


# In[ ]:


train.DAYS_EMPLOYED.max()


# We can see that the anomalies for the employment days were those who were either pensioner or umemployed.
# I chose to impute the pensioners employment history anomalies with 50 (the max years) and the unemployed ones with 0 
# (because the don't have any employment history claimed)

# In[ ]:


#replace anomalies with the extreme value in range of normal data:
#for years of employment, the maximum number should be 50 years for pensioners,
#and 0 for unemployed clients
train_dataframe=train.copy()
test_dataframe = test.copy()
def unemployed_anomaly(x):
  if x==365243:
    x= 0
  return x

def pensioner_anomaly(x):
  if x==365243:
    x=-18263
  return x

train_unemployed = train_dataframe.loc[train_dataframe.NAME_INCOME_TYPE=="Unemployed",:]
train_pensioner = train_dataframe.loc[train_dataframe.NAME_INCOME_TYPE=="Pensioner",:]
train_other = train_dataframe.loc[(train_dataframe.NAME_INCOME_TYPE!="Unemployed")&(train_dataframe.NAME_INCOME_TYPE!="Pensioner"),:]
train_unemployed.DAYS_EMPLOYED = train_unemployed.DAYS_EMPLOYED.apply(unemployed_anomaly)
train_pensioner.DAYS_EMPLOYED=train_pensioner.DAYS_EMPLOYED.apply(pensioner_anomaly)
train_dataframe=pd.concat([train_unemployed,train_pensioner,train_other],axis=0)

test_unemployed = test_dataframe.loc[test_dataframe.NAME_INCOME_TYPE=="Unemployed",:]
test_pensioner = test_dataframe.loc[test_dataframe.NAME_INCOME_TYPE=="Pensioner",:]
test_other = test_dataframe.loc[(test_dataframe.NAME_INCOME_TYPE!="Unemployed")&(test_dataframe.NAME_INCOME_TYPE!="Pensioner"),:]
test_unemployed.DAYS_EMPLOYED=test_unemployed.DAYS_EMPLOYED.apply(unemployed_anomaly)
test_pensioner.DAYS_EMPLOYED=test_pensioner.DAYS_EMPLOYED.apply(pensioner_anomaly)
test_dataframe=pd.concat([test_unemployed,test_pensioner,test_other],axis=0)

display(test_dataframe.head())
display(train_dataframe.head())


# In[ ]:


train_dataframe.DAYS_EMPLOYED.min()/365.25


# We are all good :)

# In[ ]:


train=train_dataframe
test=test_dataframe


# ## 5. Feature Engineering

# I tried to do some feature engineering based on  domain knowledge and not the feature importance

# ### Ratios:

# #### 1- Payment Ratio

# In[ ]:


#RATIOS:
#1- amount of annual payment to annual income:
train_df = train.copy()
train_df["PAYMENT_RATIO"] = train_df["AMT_ANNUITY"]/train_df["AMT_INCOME_TOTAL"]
train_df = train_df.drop(columns=["AMT_ANNUITY"])
test_df = test.copy()
test_df["PAYMENT_RATIO"] = test_df["AMT_ANNUITY"]/test_df["AMT_INCOME_TOTAL"]
test_df = test_df.drop(columns=["AMT_ANNUITY"])


# In[ ]:


train_df.head()


# #### 2- Years instead of days!

# In[ ]:


#Years Instead of Days!
def year_convertor(dataframe):
  df = dataframe.copy()
  df["AGE"] = df["DAYS_BIRTH"].apply(lambda x: np.int(-x/365))
  df["YEARS_EMPLOYED"] = df["DAYS_EMPLOYED"].apply(lambda x: np.int(-x/365))
  df["YEARS_REGISTERED"] = df["DAYS_REGISTRATION"].apply(lambda x: np.int(-x/365))
  df["YEARS_ID_PUBLISHED"] = df["DAYS_ID_PUBLISH"].apply(lambda x: np.int(-x/365))
  df["YEARS_LAST_PHONE_CHANGE"]=df["DAYS_LAST_PHONE_CHANGE"].apply(lambda x: np.int(-x/365))
  df = df.drop(columns = ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH","DAYS_LAST_PHONE_CHANGE"])
  return df


# In[ ]:


training_df = year_convertor(train_df)
testing_df = year_convertor(test_df)
training_df.head()


# In[ ]:


train=training_df
test=testing_df


# ### One-Hot Coding:

# Before one-hot coding, I tried to replace binary categorical values simply with 0 and 1 to prevent more dimention increase:

# In[ ]:


train_1 = train.copy()
test_1 = test.copy()

#Binary Categorical Data (Y/N,F/M):
train_1['FLAG_OWN_CAR'] = train_1['FLAG_OWN_CAR'].replace(to_replace = ['Y', 'N'], value = [1,0] )
train_1['FLAG_OWN_REALTY'] = train_1['FLAG_OWN_REALTY'].replace(['Y', 'N'], [1, 0])
train_1['EMERGENCYSTATE_MODE'] = train_1['EMERGENCYSTATE_MODE'].replace(['Yes', 'No'], [1, 0])
train_1['CODE_GENDER'] = train_1['CODE_GENDER'].replace(['M', 'F'], [1, 0])

test_1['FLAG_OWN_CAR'] = test_1['FLAG_OWN_CAR'].replace(['Y', 'N'], [1, 0])
test_1['FLAG_OWN_REALTY'] = test_1['FLAG_OWN_REALTY'].replace(['Y', 'N'], [1, 0])
test_1['EMERGENCYSTATE_MODE'] = test_1['EMERGENCYSTATE_MODE'].replace(['Yes', 'No'], [1, 0])
test_1['CODE_GENDER'] = test_1['CODE_GENDER'].replace(['M', 'F'], [1, 0])
train_1.head()


# In[ ]:


train=train_1
test=test_1


# And now we are set for the one-hot coding 

# In[ ]:


#one-hot coding on categorical columns:
train = pd.get_dummies(train)
test = pd.get_dummies(test)
train.head()


# In[ ]:


train.shape,test.shape


# Because of more categories in one or more columns in the training dataset, the one-hot coding produced more columns for the training dataset. we can simply align these data sets together

# In[ ]:


#saving the targets firs:
train_labels = train['TARGET']

# Align the training and testing data
train, test = train.align(test, join = 'inner', axis = 1)

# Add the target
train['TARGET'] = train_labels


# In[ ]:


train.shape,test.shape


# Now it's aligned

# ## 6. ML model building

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ### Splitting the data set 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train.drop(columns="TARGET"),
                                                    train["TARGET"], test_size=0.10,
                                                    random_state=101)


# In[ ]:


X_train.shape,X_test.shape


# ###  Logistic Regression:

# In[ ]:


reg_algorithm = LogisticRegression()
reg_algorithm.fit(X_train,y_train)
predictions = reg_algorithm.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# It seems the algorithm doesn't learn any thing from the data. That's probably because of the imbalanced labels.

# ### Decision Tree Classifier:

# In[ ]:


dt_model=DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
dt_pred = dt_model.predict(X_test)
print(confusion_matrix(y_test,dt_pred))
print(classification_report(y_test,dt_pred))


# ### Random Forest Classifier:

# In[ ]:


rf_model= RandomForestClassifier(n_estimators=300)
rf_model.fit(X_train,y_train)
rf_pre=rf_model.predict(X_test)
print(confusion_matrix(y_test,rf_pre))
print(classification_report(y_test,rf_pre))


# ### XGBoost Classifier:

# In[ ]:


xgb_model = XGBClassifier(n_estimators=300)
xgb_model.fit(X_train,y_train)
xg_pred = xgb_model.predict(X_test)
print(confusion_matrix(y_test,xg_pred))
print(classification_report(y_test,xg_pred))


# ## 7. Model Optimization

# ### Grid Search Cross Validation

# In[ ]:


X=train.drop(columns="TARGET")
y=train.TARGET
tuned_parameters = [{'n_estimators': [200,350], 'max_depth' : [5,10]}]
clf = GridSearchCV(XGBClassifier(), tuned_parameters, cv=5, scoring='roc_auc', return_train_score=True)
clf.fit(X, y)
print(clf.classes_)
print ("best score = ", clf.best_score_)
print("Best parameters set found on development set: ", clf.best_params_)
maxDepth=clf.best_params_["max_depth"]
nEstimator = clf.best_params_["n_estimators"]

xgb_model=XGBClassifier(max_depth=maxDepth,n_estimators=nEstimator)
X = train.drop(columns="TARGET")
y = train.TARGET
xgb_model=xgb_model.fit(X,y)
test_labels = xgb_model.predict_proba(test)


# In[ ]:


submission = pd.DataFrame({
    "SK_ID_CURR": test.index,
    "TARGET" : test_labels[:,1]
})
submission.to_csv('submission.csv', index=False)

