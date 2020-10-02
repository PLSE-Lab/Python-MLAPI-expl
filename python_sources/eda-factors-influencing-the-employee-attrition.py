#!/usr/bin/env python
# coding: utf-8

# **Introduction:** 
#           In this attrition dataset, will perform the EDA to find out the most important metrics which are all playing major role in employee attrition. 
#     
#     Basic things need to find out from this dataset would be : 
#     1. How satisfied out employees 
#     2. Hourly rate related with education and gender 
#     3. How job invovlment correlated with job satisfaction 

# 
# ![Attrition](http://thecontextofthings.com/wp-content/uploads/2017/01/employee-attrition.jpg)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Read the csv file**

# In[ ]:


data = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")


# In[ ]:


data.head()


# In[ ]:


data.describe()


# **Data overview**

# In[ ]:


import seaborn as sns 
import matplotlib.pyplot as plt 


# In[ ]:


sns.distplot(data['Age'])


# **Total daily rate by department wise**

# In[ ]:


sns.set(style = 'whitegrid')
dailyrate = sns.barplot(x='Department', y='DailyRate' , data = data , estimator = sum  )


# **Average daily rate by Department**

# In[ ]:


from numpy import mean , median 
sns.set(style = 'whitegrid')
avg = sns.barplot(x='Department', y='DailyRate' , data = data , estimator = np.median  )


# ****Median daily rate by gender and**** department wise 

# In[ ]:


sns.set(style = 'whitegrid')
sns.barplot(x='Department', y='DailyRate' , data = data , estimator = np.median, hue = data['Gender'])


# ****Attrition based on business travel****

# In[ ]:


sns.barplot(x = data['BusinessTravel'] , y = data['EmployeeCount'],estimator = np.sum, hue = data['Attrition'])


# ****How satisfied the employees are?****

# In[ ]:


sns.scatterplot(x= data['MonthlyIncome'], y = data['TotalWorkingYears'], hue = data['Attrition'])


# **Proposition of attrition in this dataset**

# In[ ]:


sns.countplot(x= data['Attrition'], data = data, hue = data['Gender'])


# **How marital status impacts the attrition**

# In[ ]:


sns.countplot(x = data['MaritalStatus'],hue=data['Attrition'] , data = data)


# **How many of them satisfied with the job**

# In[ ]:


sns.barplot(y=data['JobRole'], x=data['JobSatisfaction'],estimator = np.mean, data = data)


# **Hourly rate with respect to education field and gender**

# In[ ]:


sns.set(style = 'white')
sns.barplot(x=data['EducationField'], y=data['HourlyRate'], estimator = np.median, hue = data['Gender'])


# **Any Missing values**

# In[ ]:


data.isnull().sum()


# **Treat categorical labels and change into numeric  values**

# Types of techniques for encoding the categorical data
# 1. Replacing values
# 2. Encoding labels
# 3. One-Hot encoding
# 4. Binary encoding
# 5. Backward difference encoding
# 6. Miscellaneous features

# In[ ]:


data.dtypes


# In[ ]:


data_replace = data.copy()


# In[ ]:


obj_df = data.select_dtypes(include=['object']).copy()
obj_df.head()


# **Any missing values in the categorical columns**

# In[ ]:


obj_df[obj_df.isnull().any(axis=1)]


# **There is no missing values in any of the categorical values**

# ****#Find and Replace method in replacing categorical to numerics values** 

# In[ ]:


obj_df['BusinessTravel'].value_counts()


# In[ ]:


obj_df['BusinessTravel'] = obj_df['BusinessTravel'].astype('category')
obj_df.dtypes


# In[ ]:



data['BusinessTravel'] = obj_df['BusinessTravel'].cat.codes
data.head()


# **Checking for converting the categorical to numeric is successful. Now I can apply the same concept to the rest of the variables**

# In[ ]:


obj_df['Department'] = obj_df['Department'].astype('category')
obj_df['EducationField'] = obj_df['EducationField'].astype('category')
obj_df['Gender'] = obj_df['Gender'].astype('category')
obj_df['JobRole'] = obj_df['JobRole'].astype('category')
obj_df['MaritalStatus'] = obj_df['MaritalStatus'].astype('category')
obj_df['Over18'] = obj_df['Over18'].astype('category')
obj_df['OverTime'] = obj_df['OverTime'].astype('category')
obj_df['Attrition'] = obj_df['Attrition'].astype('category')


# In[ ]:


data['Department'] = obj_df['Department'].cat.codes
data['EducationField'] = obj_df['EducationField'].cat.codes
data['Gender'] = obj_df['Gender'].cat.codes
data['JobRole'] = obj_df['JobRole'].cat.codes
data['MaritalStatus'] = obj_df['MaritalStatus'].cat.codes
data['Over18'] = obj_df['Over18'].cat.codes
data['OverTime'] = obj_df['OverTime'].cat.codes
data['Attrition'] = obj_df['Attrition'].cat.codes

data.head()


# ****Correlation plot****

# In[ ]:


data_model = data.copy()


# In[ ]:


data_model.head()


# In[ ]:


data_model.dtypes


# In[ ]:


corr = data_model.corr()
f,ax = plt.subplots(figsize=(16,9))
sns.heatmap(corr, vmax = 0.8,square ='TRUE' )


# ****Take top 10 correlated metrics****

# In[ ]:


k=10
cols=corr.nlargest(k,'Attrition')['Attrition'].index
cm= np.corrcoef(data_model[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar = True ,annot = True,fmt ='.2f',annot_kws ={'size':10}, yticklabels=cols.values, xticklabels=cols.values )
#annot_kws -  provides access to how annotations are displayed, rather than what is displayed
plt.show()


# **Feature selection:** *Using P-value*

# In[ ]:


import statsmodels.api as sm


# In[ ]:


x = data_model.drop(['Attrition'],axis=1)
y = data_model['Attrition']


# In[ ]:


#using backward elimination method 
X_1 = sm.add_constant(x)


# In[ ]:


#Fitting sm.OLS method 
model = sm.OLS(y,X_1).fit()
model.pvalues


# In[ ]:


#Backward elimination 
cols = list(x.columns)
pmax = 1
while (len(cols)>0):
    p =[]
    x_1 = x[cols]
    x_1 = sm.add_constant(x_1)
    model = sm.OLS(y,x_1).fit()
    p= pd.Series(model.pvalues.values[:],index = cols)
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if (pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break 
            
        


# In[ ]:


selected_features_BE = cols
selected_features_BE


# In[ ]:


data_new = data_model[selected_features_BE]
data_new.head()


# **No multicorrelated metrics- so ready to go for modeling  techniques**

# In[ ]:


#Logistic regression
#x = data_model.drop(['Attrition'],axis=1)
#x.head()
X = data_new



# In[ ]:


X.shape


# In[ ]:


Y = y
Y.head()


# **Splitting the dataset into train and test**

# In[ ]:



from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.25,random_state=0)


# **Fitting Logistic regression to the training set**

# In[ ]:


from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state=0)
classifier.fit(train_x,train_y)


# **Predicting the test results**

# In[ ]:


y_pred = classifier.predict(test_x)


# **Making the confusion matrix** 

# In[ ]:


from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(test_y,y_pred)


# In[ ]:


from sklearn.metrics import accuracy_score
accu = accuracy_score(test_y,y_pred)
print("Logistic regression model accuracy is {}".format(accu*100))


# ****Random Forest classification model**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


classifier = RandomForestClassifier(n_estimators = 15, criterion = 'entropy',random_state =0 )
classifier.fit(train_x,train_y)


# In[ ]:


y_random = classifier.predict(test_x)


# In[ ]:


cm_random = confusion_matrix (test_y,y_random)


# In[ ]:


accu_random = accuracy_score(test_y,y_random)
print("Random Forest model accuracy is {}".format(accu_random*100))


# ****Decision tree model**** 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier_dec = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
classifier_dec.fit(train_x,train_y)


# In[ ]:


y_dec = classifier.predict(test_x)


# In[ ]:


cm = confusion_matrix(test_y,y_dec)
accu_dec = accuracy_score(test_y,y_dec)
print("Decision tree model accuracy is {}".format(accu_dec*100))


# ****XG boost model****

# In[ ]:


from sklearn.metrics import mean_squared_error 
import xgboost as xgb
from math import sqrt 


# In[ ]:


def rmse(test_y,y_xg):
    return (np.sqrt(mean_squared_error(test_y,y_xg)))

regr = xgb.XGBRegressor( colsample_bytree = 0.2, 
                        gamma = 0.0,
                        learning_rate = 0.01,
                        max_depth = 4,
                        min_child_weight = 1.5,
                        n_estimators = 7200, 
                        reg_alpha = 0.9 , 
                        reg_lambda = 0.6,
                        subsample = 0.2,
                        seed = 42, 
                        silent = 1 
        )

regr.fit(train_x, train_y )


# In[ ]:


y_xg = regr.predict(test_x)


# In[ ]:


rmse(test_y,y_xg)


# In[ ]:



print("XG boost model accuracy is {}".format((1-rmse(test_y,y_xg))*100))


# **Conclusion:****
#         Logistic regression is performing  better than other models. Statiscally analyzed the data and there is no missing values in this pre-defined non-null values dataset. Lot of things like missing values and outliers are not need in this, so after finishing the basic EDA - directly jump into modeling. 
#          Welcoming thoughts on improving the analysis in the dataset!!! 
