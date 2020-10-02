#!/usr/bin/env python
# coding: utf-8

# In[49]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

# Import statements required for Plotly 
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import missingno as msno
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[50]:


data=pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df=data.copy()


# In[51]:


msno.matrix(df)


# In[52]:


df.head()


# **Checking Count how many employee of Attrite**

# In[53]:


sns.countplot(df['Attrition'])
plt.show()


# > lot shows that around 250 people have left the organization while the organization has been able to retain around 1200 people.

# **Distributing Data with Respect to Important Factors**

# In[54]:


fig,ax = plt.subplots(2,2, figsize=(15,10))               # 'ax' has references to all the four axes
plt.suptitle("Distribution with Respect to Important Factors", fontsize=20)
sns.distplot(df['Age'], ax = ax[0,0])  # Plot on 1st axes
ax[0][0].set_title('Distribution of Age',fontsize=14)
sns.distplot(df['TotalWorkingYears'], ax = ax[0,1])  # Plot on IInd axes
ax[0][1].set_title('Distribution of Total Working Years',fontsize=14)
sns.distplot(df['YearsAtCompany'], ax = ax[1,0])  # Plot on IIIrd axes
ax[1][0].set_title('Employee Years at company',fontsize=14)
sns.distplot(df['YearsInCurrentRole'], ax = ax[1,1])  # Plot on IV the axes
ax[1][1].set_title('Employee Years in Current Role',fontsize=14)
plt.show()  


# In[55]:


sns.barplot(x='Attrition', y='MonthlyIncome', hue= 'Gender',data=df)
plt.show()


# > people with less monthlyincome (around 3000 units) are likely to leave the organization than those with better income (around 5000 units).

# **Gender Income Gap**

# In[56]:


sns.boxplot(df['Gender'], df['MonthlyIncome'])
plt.title('MonthlyIncome vs Gender Box Plot', fontsize=20)      
plt.xlabel('MonthlyIncome', fontsize=16)
plt.ylabel('Gender', fontsize=16)
plt.show()


# In[57]:


age=pd.DataFrame(data.groupby("Age")[["MonthlyIncome","Education","JobLevel","JobInvolvement","PerformanceRating","JobSatisfaction","EnvironmentSatisfaction","RelationshipSatisfaction","WorkLifeBalance","DailyRate","MonthlyRate"]].mean())
age["Count"]=data.Age.value_counts(dropna=False)
age.reset_index(level=0, inplace=True)
age.head()


# In[58]:


plt.figure(figsize=(14,5))
ax=sns.barplot(x=age.Age,y=age.Count, data=df)
plt.xticks(rotation=90)
plt.xlabel("Age")
plt.ylabel("Counts")
plt.title("Age Counts")
plt.show()


# In[59]:


plt.figure(figsize=(14,5))
ax=sns.barplot(x=age.Age,y=age.MonthlyIncome)
#palette = sns.cubehelix_palette(len(age.index))
plt.xticks(rotation=90)
plt.xlabel("Age")
plt.ylabel("Monthly Income")
plt.title("Monthly Income According to Age")
plt.show()


# In[ ]:





# **Plotting Role with respect to their Incomes**

# In[60]:


income=pd.DataFrame(data.groupby("JobRole").MonthlyIncome.mean().sort_values(ascending=False))


# In[61]:


plt.figure(figsize=(14,5))
ax=sns.barplot(x=income.index,y=income.MonthlyIncome)
plt.xticks(rotation=90)
plt.xlabel("Job Roles")
plt.ylabel("Monthly Income")
plt.title("Job Roles with Monthly Income")
plt.show()


# In[62]:


plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df.loc[df['Attrition'] == 'No', 'Age'], label = 'Active Employee')
sns.kdeplot(df.loc[df['Attrition'] == 'Yes', 'Age'], label = 'Ex-Employees')
plt.xlim(left=18, right=60)
plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.title('Age Distribution in Percent by Attrition Status');


# In[63]:


sns.factorplot(data=df,kind='count',x='Attrition',col='Department')


# In[64]:


plt.figure(figsize=(15,6))
ax=sns.countplot(x='JobRole',hue='Attrition', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right" )
plt.show()


# In[65]:


plt.figure(figsize=(15,6))
ax=sns.countplot(x='YearsAtCompany',hue='Attrition', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right" )
plt.show()


# In[66]:


plt.figure(figsize=(15,6))
ax=sns.countplot(x='PercentSalaryHike',hue='Attrition', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right" )
plt.show()


# In[67]:


sns.factorplot(data=df,y='Age',x='Attrition',size=5,aspect=1,kind='box')


# In[68]:


pd.crosstab(columns=[df.Attrition],index=[df.JobSatisfaction],margins=True,normalize='index')


# In[69]:


pd.crosstab(columns=[df.Attrition],index=[df.EnvironmentSatisfaction],margins=True,normalize='index')


# In[70]:


pd.crosstab(columns=[df.Attrition],index=[df.JobInvolvement],margins=True,normalize='index') 


# In[71]:


pd.crosstab(columns=[df.Attrition],index=[df.WorkLifeBalance],margins=True,normalize='index')


# In[72]:


pd.crosstab(columns=[df.Attrition],index=[df.RelationshipSatisfaction],margins=True,normalize='index')


# In[73]:


df1=df.copy()


# In[74]:


dect_gender={'Male':0,'Female':1}
df1['Gender']=df1['Gender'].map(dect_gender)


# In[75]:


dect_attrition={'Yes':0,'No':1}
df1['Attrition']=df1['Attrition'].map(dect_attrition)


# In[76]:


df1.drop(['BusinessTravel','DailyRate','EmployeeCount','EmployeeNumber','HourlyRate','MonthlyRate'
         ,'NumCompaniesWorked','Over18','StandardHours', 'StockOptionLevel','TrainingTimesLastYear',
       'Department','EducationField','OverTime','JobRole','MaritalStatus'],axis=1,inplace=True)


# In[77]:


X = df1.drop('Attrition',axis = 1)
Y = df1['Attrition']


# In[78]:


from sklearn.model_selection import train_test_split


# In[79]:


xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3,random_state=25)
print(xtrain.shape, ytrain.shape)
print(xtest.shape, ytest.shape)


# In[80]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, precision_score, accuracy_score
from sklearn.metrics import recall_score, classification_report, f1_score, roc_curve, auc


# **Decision Tree Classifiers**

# In[81]:


clf_dt = DecisionTreeClassifier(max_depth=3,random_state=38)


# In[82]:


clf_dt.fit(xtrain, ytrain)


# In[83]:


dt_pred = clf_dt.predict(xtest)
dt_pred_prb=clf_dt.predict_proba(xtest)[:,1]


# In[84]:


accuracy_dt = accuracy_score(ytest,dt_pred)
print("Accuracy: {}".format(accuracy_dt))


# In[85]:


print(classification_report(ytest,dt_pred))


# In[86]:


def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(8,6))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='best')


# In[87]:


sns.set_context('poster')
auc_dt=roc_auc_score(ytest,dt_pred_prb)
fpr,tpr,threshold=roc_curve(ytest,dt_pred_prb)
plot_roc_curve(fpr,tpr,label='AUC = %0.3f'% auc_dt)


# In[88]:


clf_dt.feature_importances_


# In[89]:


features_tuple=list(zip(X.columns,clf_dt.feature_importances_))


# In[90]:


features_tuple


# In[91]:


feature_imp=pd.DataFrame(features_tuple,columns=["Feature Names","Importance"])


# In[92]:


feature_imp=feature_imp.sort_values("Importance",ascending=False)


# In[93]:


plt.figure(figsize=(12,6))
sns.barplot(x="Feature Names",y="Importance", data=feature_imp, color='y')
plt.xlabel("Employee Attrition Features")
plt.ylabel("Importance")
plt.xticks(rotation=90)
plt.title("Decision Classifier - Features Importance")


# In[ ]:




