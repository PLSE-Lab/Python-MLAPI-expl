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


df=pd.read_excel("/kaggle/input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx",'Data')


# In[ ]:


df.head(5)


# Personal Loan: This feature is considered as target which describes who has opted for the loan (0) and who has not opted for the loan (1)
# Since ID feature doesn't much contribute to the data set, hence removed from the data set.

# In[ ]:


df=df.drop(['ID'],axis=1)


# In[ ]:


df.head(5)


# In[ ]:


#Find Null values in the data set:
df.isnull().sum()


# None of the columns in the data set have Null values.

# In[ ]:


#Finding of the duplicate values:
df.duplicated().sum()


# None of the duplicate values are available in the data set.

# In[ ]:


#Since all the features in the data set are numerical hence describing the data:
df.describe().transpose()


# Analysis: There is a junk value in the 'Experience' column since there is a minimum value of '-3' which is an incorrect information.
# Before correcting the junk data in the data set let us plot the distribution.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df.columns


# In[ ]:


col=['Age', 'Experience', 'Income', 'CCAvg','Mortgage']

i=3
j=0
plt.figure(figsize=(14,12))
for k in col :
    plt.subplot(i,i,i*(j+1)//i)
    sns.distplot(df[k])
    j=j+1
plt.show()


# **Age**: Mean value of age is 45.33 and the distribution is even across mean and hence normally distributed.
# 
# **Experience**: Even distribution across mean and hence normally distributed.
# 
# **Income**: Positively skewed that is median is lesser than the mean value.
# 
# **CCAvg**: Positively skewed that is median is lesser than the mean value.
# 
# **Mortgage**: Positively skewed that is median is lesser than the mean value.

# In[ ]:


# Replacing negative experience values with the median value in the Experience column:
negexp=df[df['Experience']<0]


# In[ ]:


negexp['Experience'].value_counts()


# In[ ]:


negval=[-3, -2, -1]

for i in negval:
    df['Experience']=df['Experience'].replace(negval,np.median(df['Experience']))


# In[ ]:


df['Experience'].describe()


# Negative values in the Experience column in the data set are replaced with the median value.

# In[ ]:


# Finding Corelation between the features:
cor=df.corr()


# In[ ]:


# Heatmap for Corelation:
plt.figure(figsize=(10,8))
plt.title("Corelation Plot")
sns.heatmap(cor,annot=True)
plt.show()


# Experience and Age are highly positively corelated with eachother and hence one feature can be removed to avoid multi-colinearity issue.

# In[ ]:


plt.figure(figsize=(10,8))
plt.title("Scatter plot for Experience & Age")
sns.scatterplot(x='Age',y='Experience', hue='Personal Loan', data=df)
plt.show()


# Experience & Age are highly positively co-related and so Experience can be dropped.

# In[ ]:


df=df.drop(['Experience'],axis=1)


# In[ ]:


# Plotting Scatter plot for multivariate features:
col=['Income','CCAvg','Mortgage']
plt.figure(figsize=(14,12))
j=3
k=0
for i in col:
    plt.subplot(1,j,j*(k+1)//j)
    sns.scatterplot(x='Age',y=i,hue='Personal Loan', data=df)
    k=k+1
plt.show()


# **Age v/s Income**: People with more income (>100$) seems to have opted for Personal Loan.
# 
# **Age v/s CCAvg**: Also people with high CCAvg seems to have opted for Personal Loan.
# 
# **Age v/s Mortgage**: People who have opted for Personal Loan are not much related with higher Mortgage value but people with Mortgage value greater than 400$ seems to have taken Personal Loan.

# In[ ]:


# Plotting Counts plot for Categorical features:
col=['Securities Account','CD Account','Online','CreditCard']
plt.figure(figsize=(14,12))
j=2
k=0
for i in col:
    plt.subplot(2,j,j*(k+1)//j)
    sns.countplot(x=i,hue='Personal Loan', data=df)
    k=k+1
    plt.grid(True)
plt.show()


# People without the CD Accounts tend to have taken more Personal Loan.
# People with more Online accounts seem to take more Personal Loan, where as very minimal number of people who have Securities Account have opted the Personal Loan.
# Most of the people with the CreditCard seems to have not taken the Personal Loan.

# In[ ]:


df.columns


# In[ ]:


plt.figure(figsize=(9,7))
sns.boxplot(x='Family',y='Income',hue='Personal Loan', data=df)
plt.show()


# People with high Income irrespective of the Family size seems to have opted for Personal Loan.

# In[ ]:


plt.figure(figsize=(12,10))
sns.boxplot(x='Education',y='CCAvg',hue='Personal Loan', data=df)
plt.show()


# Irrespective of Education, people who have good CCAvg > 2.5 seems to have opted out for the Personal Loan.

# In[ ]:


df.columns


# In[ ]:


df=df.drop(['ZIP Code'],axis=1)


# In[ ]:


df1=df


# In[ ]:


df1['Personal Loan'].value_counts()


# In[ ]:


df.head(5)


# In[ ]:


# Checking class balance for Personal Loan:
df['Personal Loan'].value_counts()


# In[ ]:


# Class label has imbalanced data, so this feature needs to be re-balanced using upsample method:
# Splitting major & minor class data frames:
df_majority=df[df['Personal Loan']==0]
df_minority=df[df['Personal Loan']==1]


# In[ ]:


print("Majority calss shape {}".format(df_majority.shape))
print("Minority calss shape {}".format(df_minority.shape))


# In[ ]:


# Upsampling:
from sklearn.utils import resample
df_minority_upsample=resample(df_minority,n_samples=4520)


# In[ ]:


df=pd.concat([df_majority,df_minority_upsample])


# In[ ]:


df['Personal Loan'].value_counts()


# In[ ]:


# Model Building:
x=df.drop(['Personal Loan'],axis=1)
y=df['Personal Loan']


# In[ ]:


# Splitting of Data:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[ ]:


# Decision Tree Model Prediction
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[ ]:


dt.fit(x_train,y_train)


# In[ ]:


y_pred_base=dt.predict(x_test)


# In[ ]:


# Finding Accuracy:
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred_base)
print(acc)


# In[ ]:


# Model validation:
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_test,y_pred_base)


# **Accuracy**: (TP+TN)/(TP+TN+FP+FN)
# 
# **Classification Error**: (FP+FN)/(TP+TN+FP+FN) or 1-Accuracy
# 
# **Sensitivity**: When the actual value is positive, how often is the prediction correct. TP/FN+TP
# 
# **Specificity**: When the actual value is neagative, how often is the prediction correct. TN/TN+FP
# 
# **Precision**: When the positive value is predicted, how often is the prediction correct. TP/TP+FP

# In[ ]:


#Classification Report:
clf_report=classification_report(y_test,y_pred_base)
print(clf_report)


# In[ ]:


# Hyper Parameter Tuning:
from sklearn.model_selection import GridSearchCV
parameters={'criterion':['gini','entropy'],'max_depth':np.arange(1,50),'min_samples_leaf':[1,2,3,6,9,4]}
grid=GridSearchCV(dt,parameters)


# In[ ]:


model=grid.fit(x_train,y_train)


# In[ ]:


grid.best_score_


# In[ ]:


grid.best_params_


# In[ ]:


clf_best=grid.best_estimator_


# In[ ]:


clf_best.fit(x_train,y_train)


# In[ ]:


y_pred_best=clf_best.predict(x_test)


# In[ ]:


accuracy_score(y_test,y_pred_best)


# In[ ]:


# Cross Validation:
from sklearn.model_selection import cross_val_score


# In[ ]:


cross_val=cross_val_score(clf_best,x,y,cv=10)
print(cross_val)


# In[ ]:


np.mean(cross_val)


# In[ ]:


# Visualizg the Tree:
from sklearn import tree
plt.figure(figsize=(16,14))
tree.plot_tree(clf_best)
plt.show()


# In[ ]:


# For the imbalance data set:
x_imbal=df1.drop(['Personal Loan'],axis=1)
y_imbal=df1['Personal Loan']


# In[ ]:


x_train_imbal,x_test_imbal,y_train_imbal,y_test_imbal=train_test_split(x_imbal,y_imbal,test_size=0.3)


# In[ ]:


clf_best.fit(x_train_imbal,y_train_imbal)


# In[ ]:


y_pred_imbal=clf_best.predict(x_test_imbal)


# In[ ]:


accuracy_score(y_test_imbal,y_pred_imbal)

