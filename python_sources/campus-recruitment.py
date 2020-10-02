#!/usr/bin/env python
# coding: utf-8

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


data =  pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


data[data['salary'].isnull()]


# In[ ]:


data[data['status']=='Not Placed']


# ## Null value in salary is for those who have not placed.

# In[ ]:


# Making null value as zero.
data.fillna(0,inplace=True)


# In[ ]:


data.isnull().sum()


# **Now As there is no any null value we could move forward for prediction.**
# 
# *First Wether a studet get placed or not*
# 
# *Second if got placed then their expected salary*

# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


# Making a dataset with all precentage columns- 
#['ssc_p', 'hsc_p','degree_p', 'etest_p',  'mba_p','status']


# In[ ]:


modal_data=data[['ssc_p', 'hsc_p','degree_p', 'etest_p',  'mba_p','status']]


# In[ ]:


modal_data.head()


# In[ ]:


# Extracting features and target

X=modal_data.iloc[:,:-1]
y=modal_data.iloc[:,-1]


# In[ ]:


# Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[ ]:


# Spliting data for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3)


# In[ ]:


l_reg = LogisticRegression()


# In[ ]:


l_reg.fit(X_train,y_train)


# In[ ]:


y_pred=l_reg.predict(X_test)


# In[ ]:


metrics.accuracy_score(y_pred,y_test)


# In[ ]:


df = pd.DataFrame(columns=['Actual','Predicted'])
df['Actual']=y_test
df['Predicted']=y_pred


# In[ ]:


df['Actual']=y_test
df['Predicted']=y_pred


# In[ ]:


df


# In[ ]:


# Second Predicting salary
modal_data.head()


# In[ ]:


modal2_data=modal_data[modal_data['status']=='Placed'].iloc[:,:-1]


# In[ ]:


modal2_data['salary']=data[data['salary']>0]['salary']


# In[ ]:


modal2_data.head()


# In[ ]:


X=modal2_data.iloc[:,:-1]
y=modal2_data.iloc[:,-1]


# In[ ]:


# Spliting data for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[ ]:


lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
r2_score(y_test,y_pred)


# In[ ]:


df1 = pd.DataFrame(columns=['Actual','Predicted'])
df1['Actual']=y_test
df1['Predicted']=y_pred
df1


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.scatter(modal2_data['salary'],modal2_data['ssc_p'])


# In[ ]:


plt.scatter(modal2_data['hsc_p'],modal2_data['salary'])


# In[ ]:


plt.scatter(modal2_data['degree_p'],modal2_data['salary'])


# In[ ]:


plt.scatter(modal2_data['etest_p'],modal2_data['salary'])


# In[ ]:


plt.scatter(modal2_data['mba_p'],modal2_data['salary'])


# In[ ]:


data.head()


# In[ ]:


# Seaborn Library.
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


data =  pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# * Gender vs salary
# * Percentage v/s salery
#     1.   Secondary percentage
#     2.   Higher Secondary Percentage
#     3.   Degree Percentage
#     4.   Employment test percentage
#     5.   MBA percentage
# * Board(ssc and hsc) VS  salary
# * Degree Type VS salary
# * MBA specialisation vs salary
# * Work Experience VS salary
# 

# ## Gender

# In[ ]:


data.gender.value_counts()


# In[ ]:


sns.kdeplot(data.salary[ data.gender=="M"],)
sns.kdeplot(data.salary[ data.gender=="F"])
plt.legend(["Male", "Female"])
plt.xlabel("Salary")
plt.show()


# In[ ]:


plt.figure(figsize =(14,6))
sns.boxplot("salary", "gender", data=data)
plt.show()


# * **Male students are getting slightly higher salaries than female students on an average.**

# ## Percentage

# In[ ]:


sns.countplot("ssc_b", hue="status", data=data)
plt.show()


# In[ ]:


sns.scatterplot(x='ssc_p',y='salary',hue='ssc_b',data=data)


# In[ ]:


sns.countplot("hsc_b", hue="status", data=data)
plt.show()


# In[ ]:


sns.scatterplot(x='hsc_p',y='salary',hue='hsc_b',data=data)


# * **There are few instances where central board students are getting higher salaries but those are very small.**
# * **Boards doesn't affect salary much-- we will not consider this for our modal**

# In[ ]:





# **Higher secondary stream**

# In[ ]:


sns.countplot("hsc_s", hue="status", data=data)
plt.show()


# In[ ]:


sns.scatterplot(x='hsc_p',y='salary',hue='hsc_s',data=data)


# **Arts students no is very low. so can't say much. and commerce students are slightly better in placement scenerios.**
# 

# In[ ]:


sns.countplot("degree_t", hue="status", data=data)
plt.show()


# * Commerce and management have better placement chances.

# In[ ]:


sns.scatterplot(x='degree_p',y='salary',hue='degree_t',data=data)


# * **Isn't any clear realation b/w percentage and salary**

# In[ ]:


## Work Experience

sns.countplot("workex", hue="status", data=data)
plt.show()


# * **This affects Placement.** Very few students with work experience not getting placed!

# In[ ]:


plt.figure(figsize =(18,6))
sns.boxplot("salary", "workex", data=data)
plt.show()


# **Ovious having work experience will get you better salary**

# In[ ]:


## Employment test
sns.lineplot("etest_p", "salary", data=data)
plt.show()


# In[ ]:


sns.scatterplot("etest_p", "salary", data=data)


# In[ ]:


# Density plot
sns.kdeplot(data.etest_p[ data.status=="Placed"])
sns.kdeplot(data.etest_p[ data.status=="Not Placed"])
plt.legend(["Placed", "Not Placed"])
plt.xlabel("Employability-test %")
plt.show()


# * **Employability-score doesn't effet much on salary**

# In[ ]:


## POST GRAD SPECIALISATION


# In[ ]:


sns.countplot("specialisation", hue="status", data=data)
plt.show()


# * **Market and finance has better placement record**

# In[ ]:


plt.figure(figsize =(18,6))
sns.boxplot("salary", "specialisation", data=data)
plt.show()


# * **Market and finance has better salary as well**

# In[ ]:


# Post grad percentage
sns.lineplot("mba_p", "salary", data=data)
plt.show()


# In[ ]:


sns.scatterplot("mba_p", "salary", data=data)
plt.show()


# * **Here againg mba percentage does't effect much on salary**

# Taking features in account for modal-
# * Gender
# * All percentage
# * Work exp
# * specialisation(Higher seceondary, degree, post grad)
# 

# In[ ]:


# We have to drop ssc_b and hsc_b, sl.no
data.drop(['ssc_b','hsc_b','sl_no'], axis=1, inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.dtypes


# In[ ]:


# Encoding categorical columns
data["gender"] = data.gender.map({"M":0,"F":1})
data["hsc_s"] = data.hsc_s.map({"Commerce":0,"Science":1,"Arts":2})
data["degree_t"] = data.degree_t.map({"Comm&Mgmt":0,"Sci&Tech":1, "Others":2})
data["workex"] = data.workex.map({"No":0, "Yes":1})
data["status"] = data.status.map({"Not Placed":0, "Placed":1})
data["specialisation"] = data.specialisation.map({"Mkt&HR":0, "Mkt&Fin":1})


# In[ ]:


from sklearn.preprocessing import StandardScaler# for scaling
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# In[ ]:


data.isnull().sum()


# **Salary column has null value but it is for those who havn't placed so can't fill it with any value. Drop**

# In[ ]:


#dropping NaNs (in Salary)
data.dropna(inplace=True)
#dropping Status = "Placed" column
data.drop("status", axis=1, inplace=True)


# In[ ]:


data.head()


# In[ ]:


#Seperating Depencent and Independent Vaiiables
X = data.iloc[:,:-1]
y = data.iloc[:,-1] #Dependent Variable


# In[ ]:


X


# In[ ]:


#Scalizing (Normalization)
X_scaled = StandardScaler().fit_transform(X)


# In[ ]:


X_scaled.shape


# In[ ]:


lr=LinearRegression(fit_intercept=True,normalize=True)
lr.fit(X_scaled,y)
y_pred=lr.predict(X_scaled)
print(f"R2 Score: {r2_score(y, y_pred)}")
print(f"MAE: {mean_absolute_error(y, y_pred)}")


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


model=LinearRegression()
parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
grid = GridSearchCV(model,parameters)
grid.fit(X_scaled, y)
print("r2 / variance : ", grid.best_score_)


# In[ ]:


pred=pd.DataFrame(columns=['salary'])


# In[ ]:


pred['salary']=y_pred


# In[ ]:


sns.kdeplot(y)
sns.kdeplot(pred['salary'])
plt.legend(["Actual", "Predicted"])
plt.xlabel("Salary")
plt.show()


# In[ ]:


# Polynomial regression
from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


poly=PolynomialFeatures(degree=3)
X_poly=poly.fit_transform(X_scaled)


# In[ ]:


lr_poly=LinearRegression()
lr_poly.fit(X_poly,y)


# In[ ]:


y_pred=lr_poly.predict(X_poly)


# In[ ]:


type(X_scaled)


# In[ ]:


X_poly.shape


# In[ ]:


r2_score(y,y_pred)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 19)


# In[ ]:


type(X_test)


# In[ ]:


r2_train=[]
r2_test=[]
for i in range(1,10):
    poly=PolynomialFeatures(degree=i)
    X_poly=poly.fit_transform(X_train)
    lr_poly=LinearRegression().fit(X_poly,y_train)
    y_pred_train=lr_poly.predict(X_poly)
    
    X_test_poly=poly.fit_transform(X_test)
    y_pred=lr_poly.predict(X_test_poly)
    
    r2_train.append(r2_score(y_train,y_pred_train))
    r2_test.append(r2_score(y_test,y_pred))


# In[ ]:


x_ax=np.arange(10)+1


# In[ ]:


r2_train


# In[ ]:


r2_test

