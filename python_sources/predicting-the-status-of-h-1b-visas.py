#!/usr/bin/env python
# coding: utf-8

# # Predicting the Status of H-1B Visa Applications

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from statistics import mode
import re
from xgboost import XGBClassifier


# ## H1B VISA and Dataset Kaggle 

# In[ ]:


df = pd.read_csv('../input/h1b_kaggle.csv')


# ## Info about dataset

# In[ ]:


df.info()
df.head()
df.describe()


# In[ ]:


df.rename( columns={'Unnamed: 0':'CASE_ID'}, inplace=True )


# This dataset has 11 columns, from which 1 is a target variable, which in this case is a case_status. So, this data has 1 target variable and 10 independent or explanatory variables.

# In[ ]:


df['CASE_STATUS'].unique()


# The target variable contains 6 different classes
# 
# Converting to binary class classification, we would classify either Certified or Denied. So the first thing that we should do is to convert remaining classes into either denied or certified.

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
df.CASE_STATUS[df['CASE_STATUS']=='REJECTED'] = 'DENIED'
df.CASE_STATUS[df['CASE_STATUS']=='INVALIDATED'] = 'DENIED'
df.CASE_STATUS[df['CASE_STATUS']=='PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED'] = 'DENIED'
df.CASE_STATUS[df['CASE_STATUS']=='CERTIFIED-WITHDRAWN'] = 'CERTIFIED'


# Checking the percentage of Certified and Denied classes in the Dataset.

# In[ ]:


##Drop rows with withdrawn
df.EMPLOYER_NAME.describe()
df = df.drop(df[df.CASE_STATUS == 'WITHDRAWN'].index)

## Storing non null in df w.r.t. case status
df = df[df['CASE_STATUS'].notnull()]
df['CASE_STATUS'].value_counts()


# In[ ]:


94364/(94364+2818282)


# The decline class is only 3.2% of the total dataset, that means you now have approx 96.8% Certified cases in your dataset. This shows that the datset is highly imbalanced. 

# ## Treating missing and NA values 

# In[ ]:


##check count of NAN
count_nan = len(df) - df.count()
print(count_nan)


# 18 Missing Values for Employer name
# 
# Filling Mode value for the Missing Employer name

# In[ ]:


## Filling na in employer name with mode
df['EMPLOYER_NAME'] = df['EMPLOYER_NAME'].fillna(df['EMPLOYER_NAME'].mode()[0])


# To check if any employers are still not null

# In[ ]:


assert pd.notnull(df['EMPLOYER_NAME']).all().all()


# Working on Feature Prevailing wage

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
df.boxplot(column='PREVAILING_WAGE')


# In[ ]:


df.PREVAILING_WAGE.max()


# In[ ]:


np.nanpercentile(df.PREVAILING_WAGE,98)


# In[ ]:


df.PREVAILING_WAGE.median()


# In[ ]:


np.nanpercentile(df.PREVAILING_WAGE,2)


# In[ ]:


## replacing min and max with 2 and 98 percentile
df.loc[df.PREVAILING_WAGE < 34028, 'PREVAILING_WAGE']= 34028
df.loc[df['PREVAILING_WAGE'] > 138611, 'PREVAILING_WAGE']= 138611
df.PREVAILING_WAGE.fillna(df.PREVAILING_WAGE.mean(), inplace = True)


# For the the JOB_TITLE, FULL_TIME_POSITION and SOC_NAME columns, we could fill the missing values with the mode(most occuring value).

# In[ ]:


## Filling na in JOB_TITLE and FULL_TIME_POSITION with mode
df['JOB_TITLE'] = df['JOB_TITLE'].fillna(df['JOB_TITLE'].mode()[0])
df['FULL_TIME_POSITION'] = df['FULL_TIME_POSITION'].fillna(df['FULL_TIME_POSITION'].mode()[0])
df['SOC_NAME'] = df['SOC_NAME'].fillna(df['SOC_NAME'].mode()[0])


# The next feature is FULL_TIME_POSITION

# In[ ]:


df['FULL_TIME_POSITION'].value_counts()


# Y indicates the petitioner has a full time role and N indicates a part time role.

# In[ ]:


foo1 = df['FULL_TIME_POSITION']=='Y'
foo2 = df['CASE_STATUS']=='CERIFIED'
len(df[foo1])/len(df)*100


# Around 85% of the jobs applied for are full time jobs.

# ## Dropping lat and lon columns

# In[ ]:


df = df.drop('lat', axis = 1)
df = df.drop('lon', axis = 1)


# Exploring Employer_Name Feature

# In[ ]:


df['EMPLOYER_NAME'].value_counts()


# EMPLOYER_NAME contains the names of the employers and there are lot of unique employers. It is the company which submits the application for its employee. You cannot use EMPLOYER_NAME directly in the model because it has got many unique string values or categories; more than 500 employers. These employers act as factors or levels. It is not advisable to use this many factors in a single column.
# 
# The top 5 companies submitting the application for their employees are Infosys, TCS, Wipro, Deloitte and IBM. However, if any University is submitting an application then it is more likely to be accepted.

# ## Feature Creation

# If the employer name contains the string 'University' (for instance if a US university is filing a visa petition, then it has more chances of approval for the employee).
# 
# So, if the EMPLOYER_NAME contains 'University', then NEW_EMPLOYER contains the university value.

# In[ ]:


df['NEW_EMPLOYER'] = np.nan
df.shape


# What if the the University string is in upper case in some of the rows and in lower case in some other rows? If you are mapping lowercase university it would then miss the uppercase UNIVERSITY and vice-versa. So in order to correctly map and check the University string, you should first convert all the strings into the same case; either lowercase or uppercase.
# 
# All the strings in EMPLOYER_NAME containing the keyword university will have 'university' as value in the NEW_EMPLOYER column. All the remaining empty rows will be filled with 'non university'.

# In[ ]:


warnings.filterwarnings("ignore")

df['EMPLOYER_NAME'] = df['EMPLOYER_NAME'].str.lower()
df.NEW_EMPLOYER[df['EMPLOYER_NAME'].str.contains('university')] = 'university'
df['NEW_EMPLOYER']= df.NEW_EMPLOYER.replace(np.nan, 'non university', regex=True)


# Similarly for SOC_NAME feature

# In[ ]:


df['SOC_NAME'].value_counts()


# There are lot of values associated with SOC_NAME, so you might want to create a new feature that will contain the important occupation of the applicant, mapping it with the SOC_NAME value. You can create a new variable called OCCUPATION. For example computer, programmer and software are all computer ocupations. This will cover the top 80% of the occupations, and minor and remaining occupations will be categorized as others.

# In[ ]:


# Creating occupation and mapping the values
warnings.filterwarnings("ignore")

df['OCCUPATION'] = np.nan
df['SOC_NAME'] = df['SOC_NAME'].str.lower()
df.OCCUPATION[df['SOC_NAME'].str.contains('computer','programmer')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('software','web developer')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('database')] = 'Computer Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('math','statistic')] = 'Mathematical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('predictive model','stats')] = 'Mathematical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('teacher','linguist')] = 'Education Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('professor','Teach')] = 'Education Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('school principal')] = 'Education Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('medical','doctor')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('physician','dentist')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('Health','Physical Therapists')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('surgeon','nurse')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('psychiatr')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_NAME'].str.contains('chemist','physicist')] = 'Advance Sciences'
df.OCCUPATION[df['SOC_NAME'].str.contains('biology','scientist')] = 'Advance Sciences'
df.OCCUPATION[df['SOC_NAME'].str.contains('biologi','clinical research')] = 'Advance Sciences'
df.OCCUPATION[df['SOC_NAME'].str.contains('public relation','manage')] = 'Management Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('management','operation')] = 'Management Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('chief','plan')] = 'Management Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('executive')] = 'Management Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('advertis','marketing')] = 'Marketing Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('promotion','market research')] = 'Marketing Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('business','business analyst')] = 'Business Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('business systems analyst')] = 'Business Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('accountant','finance')] = 'Financial Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('financial')] = 'Financial Occupation'
df.OCCUPATION[df['SOC_NAME'].str.contains('engineer','architect')] = 'Architecture & Engineering'
df.OCCUPATION[df['SOC_NAME'].str.contains('surveyor','carto')] = 'Architecture & Engineering'
df.OCCUPATION[df['SOC_NAME'].str.contains('technician','drafter')] = 'Architecture & Engineering'
df.OCCUPATION[df['SOC_NAME'].str.contains('information security','information tech')] = 'Architecture & Engineering'
df['OCCUPATION']= df.OCCUPATION.replace(np.nan, 'Others', regex=True)


# In[ ]:


df.OCCUPATION.value_counts()


# Since visa applications majorly depend on State location, you should split the state information from the WORKSITE variable.

# In[ ]:


## Splitting city and state and capturing state in another variable
df['state'] = df.WORKSITE.str.split('\s+').str[-1]


# In[ ]:


df.state.value_counts()


# California has the highest number of petitions cementing its place as the base of choice for IT ops, followed by Texas, New York and New Jersey.

# ## Mapping Target Variables 

# Now, in order to calculate the probabilities, you need to convert the target classes to binary, i.e. 0 and 1. You can use CERTIFIED and DENIED to map it to 0 and 1. 

# In[ ]:


from sklearn import preprocessing
class_mapping = {'CERTIFIED':0, 'DENIED':1}
df["CASE_STATUS"] = df["CASE_STATUS"].map(class_mapping)


# In[ ]:


df.head()


# In[ ]:


test1 = pd.Series(df['JOB_TITLE'].ravel()).unique()
pd.DataFrame(test1)


# ## Dropping columns

# Since, you have now generated new features from the variables below, we can drop them, running horizontally across columns, as axis = 1.

# In[ ]:


# dropping these columns
df = df.drop('EMPLOYER_NAME', axis = 1)
df = df.drop('SOC_NAME', axis = 1)
df = df.drop('JOB_TITLE', axis = 1)
df = df.drop('WORKSITE', axis = 1)
df = df.drop('CASE_ID', axis = 1)


# In[ ]:


df1 = df.copy()


# Changing dtype to category: Now before moving to the modeling part, you should definitely check the data types of the variables. For instance, over here, a few variables should have been used as categories or factors, but instead they are in object string fromat.
# 
# So, you have to change the data type of these variables from object to category, as these are categorical features.

# In[ ]:


df1[['CASE_STATUS', 'FULL_TIME_POSITION', 'YEAR','NEW_EMPLOYER','OCCUPATION','state']] = df1[['CASE_STATUS', 'FULL_TIME_POSITION', 'YEAR','NEW_EMPLOYER','OCCUPATION','state']].apply(lambda x: x.astype('category'))


# In[ ]:


df1.info()


# ## Splitting Data in Training and Test Sets

# In[ ]:


X = df.drop('CASE_STATUS', axis=1)
y = df.CASE_STATUS

seed = 7
test_size = 0.40
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
X_train.columns


# Check if there are any null values in the training set. There should not be any.

# In[ ]:


X_train.isnull().sum()


# ncode X_train and X_test to get them ready for Xgboost, as it only works on numeric data. The function pd.get_dummies() is used to encode the categorical values to integers. It will create a transpose of all the categorical values and then map 1 wherever the value is present or 0 if it's not present. You should definitely try at your end to to print the X_train_encode below to check the transpose.

# In[ ]:


X_train_encode = pd.get_dummies(X_train)
X_test_encode = pd.get_dummies(X_test)


# In[ ]:


y_train.head()


# In[ ]:


X_train_encode.head()


# # XGBoost

# Convert categorical variables into numeric ones using one hot encoding.
# For classification, if the dependent variable belongs to the class factor, convert it to numeric
# as_matrix is quick enough to implement one hot encoding. You can convert the dataset to a matrix format, as shown below.

# In[ ]:


import xgboost
train_X = X_train_encode.as_matrix()
train_y = y_train.as_matrix()


# In[ ]:



gbm=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.5, max_delta_step=0,
       max_depth=3, max_features='sqrt', min_child_weight=1, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=10, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=0.8).fit(train_X, train_y)


# In[ ]:


y_pred = gbm.predict(X_test_encode.as_matrix())


# In[ ]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

