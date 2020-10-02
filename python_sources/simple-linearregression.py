#!/usr/bin/env python
# coding: utf-8

# ----------------------------------SIMPLE LINEAR REGRESSION----------------------------------

# This is a simple Linear Regression Problem on the salary of employees based on their Years of working experience

# **IMPORTING LIBRARIES**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# **LOADING DATASET**

# In[ ]:


df=pd.read_csv('../input/salary-data-simple-linear-regression/Salary_Data.csv')


# In[ ]:


# Displaying the columns of the dataset
df.columns


# There are only two columns in the dataset.

# In[ ]:


# displaying first five entries of the dataset
df.head()


# In[ ]:


# shape of the Dataset
df.shape


# The Shape of the DataSet is 30*2.
# There are 30 rows and 2 columns.

# In[ ]:


# Info about the dataset
df.info()


# In[ ]:


# Checking for the missing values in the dataset
df.isnull().sum()
# There are no missing values in the Dataset


# In[ ]:


df.dtypes


# Both are continous type variables.

# In[ ]:


# Describtion of the Dataset
df.describe()


# **Univariate Analysis**

# In[ ]:


# Checking for the outliers
plt.boxplot(df['YearsExperience'])


# There are no outliers present in this variable.

# In[ ]:


sns.distplot(df['YearsExperience'])


# It is normally distributed.

# In[ ]:


# Checking for the presence of outliers in the dependent variable
plt.boxplot(df['Salary'])


# There are no outliers present in the dependent variable.

# In[ ]:


sns.distplot(df['Salary'])


# Dependent variable is also Normally Distributed.

# **Bivariate Analysis**

# In[ ]:


sns.scatterplot(x='YearsExperience',y='Salary',data=df)


# There is a linear relationship between the dependent and independent variable.

# In[ ]:


plt.title('Correlation Matrix')
sns.heatmap(df.corr(),annot=True)


# The independent and dependent variables are highly correlated with each.

# In[ ]:


X=df.iloc[:,0]
Y=df.iloc[:,1]


# Splitting dataset into train and test

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)


# In[ ]:


print("Shape of X_train is: ",X_train.shape)
print("Shape of X_test is: ",X_test.shape)
print("Shape of Y_train is: ",Y_train.shape)
print("Shape of Y_test is: ",Y_test.shape)


# In[ ]:


#Adding new column
X_train=X_train[:,np.newaxis]
X_test=X_test[:,np.newaxis]


# **Linear Regression**

# In[ ]:


from sklearn.linear_model import LinearRegression
lrg=LinearRegression()


# In[ ]:


lrg.fit(X_train,Y_train)


# In[ ]:


lrg.score(X_train,Y_train)


# 0.9607 is a very good accuracy score 

# In[ ]:


# predicting the output of test dataset
Y_pred=lrg.predict(X_test)


# In[ ]:


data={'Y_test':Y_test,'Y_pred':Y_pred}
pd.DataFrame(data=data)


# In[ ]:


print(lrg.intercept_)
print(lrg.coef_)


# Intercept values is 26049.57                      
# and coefficient of years of experience is 9202.23                                    
# **Salary=26049.57+9202.23*YearsExperience**

# Plotting Residual Plot

# In[ ]:


plt.title('Residual Plot',size=16)
sns.residplot(Y_test,Y_pred,color='r')
plt.xlabel('Y_pred',size=12)
plt.ylabel('Residues',size=12)


# There is no pattern which implies **a good model**

# In[ ]:


from sklearn.metrics import r2_score
r2=r2_score(Y_test,Y_pred)
print('r2_score is: ',r2)


# In[ ]:




