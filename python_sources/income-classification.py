#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing useful libararies
import pandas as pd
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# In[ ]:


#read the Sales Records excel file
df = pd.read_csv(r'../input/income-classification/income_evaluation.csv')


# Lets have a look to our data

# In[ ]:


df.head(5)


# **This Data Set Name is Income Classification**
# **Shape of the data set (32561,15)**
# **NuLL values:** Not Present
# 
# **Variables:** 'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship','race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'.
# 
# There are **9 categorical variables** and **6 Numerical variables.** 
# 
# **Income is the Target Variable**
# 
# 
# By using this dataset, I try to make prediction whether the person is making over 50K or less than 50K. 
# I have implemeted the Random Forest Classification with python. Also, I have also using ensemble learnig to increase the accuracy of our Model.
# 

# ### Correcting the columns names

# In[ ]:


df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship','race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']


# ### Checking missing values

# In[ ]:


df.isna().sum()


# There are no missing values

# #### Checking the data Types of the variables

# In[ ]:


df.dtypes


# ### Seperating categorical and numerical variables apart

# In[ ]:


num = [i for i in df.columns if df[i].dtype!='O']
cat = [i for i in df.columns if df[i].dtype=='O']


# ### EDA on Categorical variables

# In[ ]:


df[cat].nunique().plot(kind='bar')


# ** Observation**
# - This graph sows no of different categories in the categorical variables

# Checking the different categories in each categorical variables

# In[ ]:


for i in df[cat]:
    print(df[i].value_counts())


# **Observation**
# - There are some '?' values instead of null values are present in some variables. we need to remove these values.
# - You can see all different classes of categorical variables with their share or get the count of distribution of values.

# ### ****Replacing all invalid values to NaN values, so we can easily fill that****

# In[ ]:


df.replace(' ?', np.nan, inplace= True)


# See we can easily get the count of NaN values

# In[ ]:


df.isna().sum()


# ##### here you can see the counts of null values in the variables. 
# 
# ### Now using fillna() function we need to replace the the null the previous column value 

# In[ ]:


df.fillna(method = 'bfill', inplace=True)


# In[ ]:


df.isna().sum().sum()


# Now, there is no null or invalid values left in our data

# ### Our target variable income, lets visualize it, with repect to others variables

# In[ ]:


df.income.value_counts()


# **Observation**
# - Only two categories in income variable

# In[ ]:


sea.countplot(x= 'income' ,data =df)


# In[ ]:


df.income.value_counts().plot(kind='pie')


# **Observation**
# - There are more than 75% number of people making less than 50k.

# ### See the income distribution with respect to gender

# In[ ]:


sea.countplot(x="income", hue="sex", data=df)


# **Observation**
# - people who are earning less than 50K are high in numbers also Male are high in numbers

# ### Income distribution w.r.t workclass

# In[ ]:


plt.subplots(figsize=(12, 8))
sea.countplot(x="income", hue="workclass", data=df)


# **Observation**
# - There are more people who employed with private sector, in both categories of income. 

# ### plotting the workclass w.r.t gender.

# In[ ]:


plt.subplots(figsize=(12, 8))
sea.countplot(hue="sex", x="workclass", data=df)


# **Observation**
# - In workclass the highest number of people doing work in private. And in all workclass males are highest. 

# In[ ]:


for i in cat:
    
    print(i, ' contains ', len(df[i].unique()), ' labels')


# ### lets encode the categorical varibles with one hot encoding

# #### Remove the target variable income 

# In[ ]:


y = df.income


# In[ ]:


df


# In[ ]:


y = pd.get_dummies(y,drop_first=True)
#df.drop(['income'],axis =1, inplace=True)


# In[ ]:


x = pd.get_dummies(df[cat])


# In[ ]:


df.drop(df[cat],axis = 1,inplace = True)


# # EDA on Numerical Variables

# In[ ]:


df[num].head()


# #### view the Distribution of Age variable

# In[ ]:


sea.distplot(df.age, bins=10)


# **Observation**
# - most number of people are belong to 20 to 50 age group.

# ### Checking the outliers in Numerical variables.

# In[ ]:


sea.boxplot(df.age)


# **Obseravation**
# 
#  -There are some outliers present in Age variables.

# ### Checking the correlation between numerical variables.

# In[ ]:


df.corr()


# **Observation**
# - there is no correlation beween the variables.

# ### Scalling the Numerical varibales 

# In[ ]:


from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df = scaler.fit_transform(df)


# In[ ]:


df = pd.DataFrame(df)
x = pd.concat([x,df],axis=1)


# ### Splitting the data into test and train

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# ### visualize trainig data

# In[ ]:


X_train.head(2)


# ### Fitting the Logistic Regression model

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# ###  Testing the logistic model by predicting the test data and calculate the accuracy score 

# In[ ]:


y_pred = logreg.predict(X_test)
accuracy_score(y_test, y_pred)


# ### Fitting the Random forest Model with only 10 decision trees 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0,n_estimators=10)
rfc.fit(X_train, y_train)


# ### Testing the random forest model by predicting the test data and calculate the accuracy score 

# In[ ]:


y_pred = rfc.predict(X_test)
accuracy_score(y_test, y_pred)


# ### Fitting the Random forest Model with only 100 decision trees

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0,n_estimators=100)
rfc.fit(X_train, y_train)


# ### Testing the random forest model by predicting the test data and calculate the accuracy score with100 decision tress. 

# In[ ]:


y_pred = rfc.predict(X_test)
accuracy_score(y_test, y_pred)


# ##### Accuracy increase by 0.1 by increasing the number of decision tress.

# ## Bagging method of ensemble learning 

# In[ ]:


from sklearn.ensemble import BaggingClassifier
from sklearn import tree
model = BaggingClassifier(random_state=0)
model.fit(X_train, y_train)


# ### Testing the bagging model and calculate accuracy

# In[ ]:


y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)


# ## Fitting the Extream gradient boosting method with keeping hyperparamerter(learning rate is 0.1 ) then calculate accuracy.

# In[ ]:


#import xgboost as xgb
#model=xgb.XGBClassifier(base_estimator = rfc,random_state=1,learning_rate=0.1)
#model.fit(X_train, y_train)
#model.score(X_test,y_test)


# **Summary**
# - This income prediction classification problem, the logistic regerssion achieved 85.13%.
# - The simple baggging with no base model gives accuracy 84.39%
# - Ensemble learning bagging technique with base model Random forest with 10 decision tress, gives accuracy is 84.95%
# - Ensemble learning bagging technique with base model Random Forest with 100 decision trees, gives accuracy is 85.40%
# - In Ensemble learning Boosting technique with base model random forest, gives accuracy 86.94%
