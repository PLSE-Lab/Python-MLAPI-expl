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


# **Personal loan** : This is our Target variable, where we have data which describe that who have opted for Persnonal Loan(0) and who have not opted for Personal Loan(1).
# 
# Since ID column do not contribute much to this data set we will remove ID column
# 
# 

# In[ ]:


df=df.drop(['ID'],axis=1)


# In[ ]:


df.head(5)


# In[ ]:


#Finding if their are any null values in the data set
df.isnull().sum()


# Any of the columns do not have null values

# In[ ]:


#Finding if any duplicate values
df.duplicated().sum()


# No duplicate value is avaliable in the dataset

# In[ ]:


#Since all the features we have are of integer type we can describe the data
df.describe().transpose()


# **Experience** : We can see minimum of experience is -3, which is junk value, we need to rectify columns which incorrect information.
# 
# Before correcting Experience info let's plot the distribution

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


col=['Age', 'Experience', 'Income','CCAvg','Mortgage']
j=0
i=3
plt.figure(figsize=(14,12))
for k in col :
    plt.subplot(i,i,i*(j+1)//i)
    sns.distplot(df[k])
    j=j+1
plt.show()


# **Age** : Mean value of Age is 45.33 and distribution is even across mean. Hence normally distibuted
# 
# **Experience**: Even distribution across mean hence normally distributed
# 
# **Income** : Positively skewed i.e Median is lesser than mean value
# 
# **CCAvg** : Positively skewed i.e Median is lesser than mean value
# 
# **Mortgage** : Positively skewed i.e Median is lesser than mean value
# 

# In[ ]:


# Replacing negative experience with median value of the Experience column
negexp=df[df['Experience']<0]


# In[ ]:


negexp['Experience'].value_counts()


# In[ ]:


negval=[-3,-2,-1]

for i in negval:
    df['Experience']=df['Experience'].replace(negval,np.median(df['Experience']))


# In[ ]:


df['Experience'].describe()


# Replacing all the negative experience with median value

# In[ ]:


#Finding corelation between features
cor=df.corr()


# In[ ]:


plt.figure(figsize=(10,8))
plt.title("Correlation plot")
sns.heatmap(cor,annot=True)
plt.show()


# Experience and Age looks like highly positively co-related with each other.We can remove one of the variable  to avoide multi-collinearity problem.
# 
# 

# In[ ]:


plt.figure(figsize=(10,8))
plt.title("Scatter plot for Experience and Age")
sns.scatterplot(x='Age',y='Experience',hue='Personal Loan',data=df)
plt.show()


# Experience and Age are highly co-related and we can drop Experience feature

# In[ ]:


#Dropping Experience from the dataset
df=df.drop(['Experience'],axis=1)


# In[ ]:


df.columns


# In[ ]:


#Plotting scatterplot for multivariate variables
col=['Income','CCAvg','Mortgage']
plt.figure(figsize=(14,12))
j=3
k=0
for i in col :
    plt.subplot(1,j,j*(k+1)//j)
    sns.scatterplot(x='Age',y=i,hue='Personal Loan',data=df)
    k=k+1
plt.show()


# **Age Vs Income** : People with more Income(greater than 100$) have opted for Personal loan. 
# 
# **Age Vs CCAvg** : Also ppl with high CCAvg have also opted for Personal loan
# 
# **Age Vs Mortgage** : People who have opted for personal loan does not relate much with higher mortgage value but people with Mortgage greater than 400$ tend to take Personal Loan

# In[ ]:


#Plotting countplot for Categorical variables
col=['Securities Account', 'CD Account', 'Online',
       'CreditCard']
plt.figure(figsize=(14,12))
j=2
k=0
for i in col :
    plt.subplot(2,j,j*(k+1)//j)
    sns.countplot(x=i,hue='Personal Loan',data=df)
    k=k+1
    plt.grid(True)
plt.show()


# People with more Online accounts tends to take more personal loan,whereas very minimal number of people who have Securities account have opted for Personal loan
# Most of the people who have Credit card do not seems to take Personal loan. And people with no CD account are taking more loans than those who have CD account

# In[ ]:


df.columns


# In[ ]:


plt.figure(figsize=(8,6))
sns.boxplot(x='Family',y='Income',hue='Personal Loan',data=df)
plt.show()


# People with high income irrespective family size have opted for laon

# In[ ]:


plt.figure(figsize=(10,8))
sns.boxplot(x='Education',y='CCAvg',hue='Personal Loan',data=df)
plt.show()


# Irrespective of education people who have Credit card avg greater than 2.5 have opted for Personal Loan

# In[ ]:


df.columns


# In[ ]:


df=df.drop(['ZIP Code'],axis=1)


# In[ ]:


df.head(5)


# In[ ]:


#checking class balance for Personal loan
df['Personal Loan'].value_counts()


# In[ ]:


df1=df


# In[ ]:


df1['Personal Loan'].value_counts()


# In[ ]:


#Class label is having imbalance data so we will re-balance the class variable using upsample method
#splitting major and minor class data frames
df_majority=df[df['Personal Loan']==0]
df_minority=df[df['Personal Loan']==1]


# In[ ]:


print("Majority class shape {}".format(df_majority.shape))
print("Minority class shape {}".format(df_minority.shape))


# In[ ]:


from sklearn.utils import resample


# In[ ]:


#Upsampling
df_minority_upsample=resample(df_minority,n_samples=4520)


# In[ ]:


#Joining both dataframes
df=pd.concat([df_majority,df_minority_upsample])


# In[ ]:


df['Personal Loan'].value_counts()


# In[ ]:


#Seperating x and y variables
x=df.drop(['Personal Loan'],axis=1)
y=df['Personal Loan']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[ ]:


y_train.head(5)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dt=DecisionTreeClassifier(criterion="gini")


# In[ ]:


#using entropy
dt_en=DecisionTreeClassifier(criterion="entropy")


# In[ ]:


dt.fit(x_train,y_train)


# In[ ]:


dt_en.fit(x_train,y_train)


# In[ ]:


#Predicting using gini criteria
y_pred_dt_gini=dt.predict(x_test)


# In[ ]:


#Predicting using entropy criteria
y_pred_dt_en=dt_en.predict(x_test)


# In[ ]:


#Checking accuracy of the model
from sklearn.metrics import accuracy_score ,confusion_matrix,classification_report


# In[ ]:


acc=accuracy_score(y_test,y_pred_dt_gini)
print(acc)


# In[ ]:


#Classification error
print(1-acc)


# In[ ]:


acc_en=accuracy_score(y_test,y_pred_dt_en)
print(acc_en)


# Both Entropy and gini Models are showing 99% accuracy

# In[ ]:


#Model validaton for Gini
confusion_matrix(y_test,y_pred_dt_gini)


# In[ ]:


#Model validaton for Entropy
confusion_matrix(y_test,y_pred_dt_en)


# 
# 
# **Accuracy** =(TP+TN)/(TP+TN+FP+FN)
# 
# **Classification error** =(FP+FN)/(TP+TN+FP+FN) or 1-Accuracy
# 
# **Sensitivity: When the actual value is positive, how often is the prediction correct?**
# 
# **Sensitivity** = TP /(FN + TP)
# 
# **Specificity: When the actual value is negative, how often is the prediction correct?**
# 
# **Specificity** = TN / (TN + FP)
# 
# **Precision: When a positive value is predicted, how often is the prediction correct?**
# 
# **Precision** = TP / TP + FP
# 
# 

# In[ ]:


print(classification_report(y_test,y_pred_dt_gini))


# The clear sign of a machine learning overfitting is if its error on testing set is much greater than the error on training set.
# 
# To prevent overfitting, you need to add regularisation in case of Linear and SVM models. Similarly, in decision tree models you we can reduce the maximum depth. While in neural networks, we can introduce dropout layer to reduce overfitting.

# In[ ]:


#Hypertuing Decision Tree
from sklearn.model_selection import GridSearchCV


# In[ ]:


dt_base=DecisionTreeClassifier()


# In[ ]:


parameters={'criterion': ['gini','entropy'],'max_depth' : np.arange(1,50),'min_samples_leaf': [1,2,5,10,13,15]}


# In[ ]:


grid=GridSearchCV(dt_base,parameters)


# In[ ]:


grid.fit(x_train,y_train)


# In[ ]:


best_dt=grid.best_params_
print(best_dt)


# In[ ]:


grid.best_score_


# In[ ]:


model=grid.best_estimator_


# In[ ]:


model.fit(x_train,y_train)


# In[ ]:


y_pred_best=model.predict(x_test)


# In[ ]:


#Cross validation using cross_val_score
from sklearn.model_selection import cross_val_score
a=cross_val_score(model, x, y, cv=10)
print(a)


# In[ ]:


accur = np.mean(a)
print(accur)


# In[ ]:


#Plotting tree
from sklearn import tree
plt.figure(figsize=(20,14))
tree.plot_tree(model)
plt.show()


# In[ ]:


#For imbalance dataset
x_imb=df1.drop(['Personal Loan'],axis=1)
y_imb=df1['Personal Loan']


# In[ ]:


x_train_imb,x_test_imb,y_train_imb,y_test_imb=train_test_split(x_imb,y_imb,test_size=0.3)


# In[ ]:


model.fit(x_train_imb,y_train_imb)


# In[ ]:


y_pred_imb=model.predict(x_test_imb)


# In[ ]:


accuracy_score(y_test_imb,y_pred_imb)


# In[ ]:


#Predict proabilities
probs = model.predict_proba(x_test)


# In[ ]:


probs = probs[:, 1]
print(probs)


# In[ ]:


from sklearn import metrics
auc = metrics.roc_auc_score(y_test, probs)
print(auc)


# In[ ]:


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)


# In[ ]:


print(fpr,tpr)


# In[ ]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[ ]:


plot_roc_curve(fpr, tpr)

