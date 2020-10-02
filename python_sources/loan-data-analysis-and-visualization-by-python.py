#!/usr/bin/env python
# coding: utf-8

# ![](http://resize.hswstatic.com/w_907/gif/loan-personal.jpg)

# # Introduction

# This data set includes customers who have paid off their loans, who have been past due and put into collection without paying back their loan and interests, and who have paid off only after they were put in collection. The financial product is a bullet loan that customers should pay off all of their loan debt in just one time by the end of the term, instead of an installment schedule. Of course, they could pay off earlier than their pay schedule.

# # Import Library

# In[ ]:


import numpy as np 
import pandas as pd

import os
print(os.listdir("../input"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')


# # Exploratory Data Analysis

# In[ ]:


data=pd.read_csv('../input/datasets_1095_1978_Loan payments data.csv')


# In[ ]:


#first five rows in dataset
data.head()


# In[ ]:


#last five rows in dataset
data.tail()


# In[ ]:


print('Data Info\n')
print(data.info())


# In[ ]:


print('Data Describe\n')
print(data.describe())


# In[ ]:


print('Data Age Describe\n')
print(data.age.describe())


# In[ ]:


print('Data Age nad terms Describe')
print(data[['age','terms']].describe())


# In[ ]:


print('Data shape\n')
print(data.shape)


# In[ ]:


print('Data Corr\n')
print(data.corr())


# In[ ]:


print('Data Corr Age Terms\n')
print(data[['age','terms']].corr())


# In[ ]:


print('Data Dtypes\n')
print(data.dtypes)


# In[ ]:


print('Data Size\n')
print(data.size)


# In[ ]:


data.isnull().sum()


# In[ ]:


print('Data Columns\n')
for col in data.columns:
    print(col)


# In[ ]:


print('Rename Data Columns\n')
data=data.rename(columns={'Loan_ID':'Loan_Id','loan_status':'Loan_Status','Principal':'Principal',
                          'terms':'Terms','effective_date':'Effective_Date','due_date':'Due_Date',
                          'paid_off_time':'Paid_Of_Time','past_due_days':'Past_Due_Days','age':'Age',
                          'education':'Education','Gender':'Gender'})
for col in data.columns:
    print(col)


# In[ ]:


print('Loan_ID States\n')    
print('Loan_ID missing sum\n')
print(data.Loan_Id.isnull().sum())


# In[ ]:


print('Loan_ID unique value\n')
print(data.Loan_Id.unique())
print('Loan_Id unqiue sum\n')
print(sum(data.Loan_Id.value_counts()))


# In[ ]:


print('Loan_Status\n')
print('Loan Status Value Counts\n')
print(data['Loan_Status'].value_counts(dropna=False))


# In[ ]:


print('Loan_Status missing sum\n')
print(data.Loan_Status.isnull().sum())


# In[ ]:


print('Loan_ID unqiue value\n')
print(data.Loan_Status.unique())
loan_status_unique=data.Loan_Status.unique().reshape(-1,1)
print(loan_status_unique)


# In[ ]:


paidoff_count=0
collection_count=0
collection_paidoff_count=0

print('Loan status with Loan_Id counts\n')
paidoff_count=len(data[data.Loan_Status==loan_status_unique[0][0]].Loan_Id)
collection_count=len(data[data.Loan_Status==loan_status_unique[1][0]].Loan_Id)
collection_paidoff_count=len(data[data.Loan_Status==loan_status_unique[2][0]].Loan_Id)


# In[ ]:


print('Pricipal State\n')
print('Data Principal value counts\n')
print(data.Principal.value_counts())


# In[ ]:


print('Paidoff and principal all of them whatever\n')   
print(data.Principal.isnull().sum())
print(data.groupby('Loan_Status')['Principal'].value_counts()) 


# In[ ]:


print('Group by every Principal in Data\n')
print(data.groupby('Loan_Status')['Principal'].sum()) 


# In[ ]:


print('Terms is missing value\n')
print(data.Terms.isnull().sum())
print('Terms value unique\n')
print(data.Terms.value_counts())
print('Unique Terms Value\n')
print(data.Terms.unique())


# In[ ]:


print('Groupby Terms and Principal in Data\n')
print(data.groupby('Terms')['Principal'].value_counts())
print('Group sum every Terms in Data')
print(data.groupby('Terms')['Principal'].sum())


# In[ ]:


print('Effective State\n')
print('Effective Date isnull\n')
print(data.Effective_Date.isnull().sum())
print('Effective Value Counts\n')
print(data.Effective_Date.value_counts())
print('Effective group by Terms nad Principal in every Data\n')
print(data.groupby('Effective_Date')[['Terms','Principal']].count())
print('Group sum every Terms and Principal in Data')
print(data.groupby('Effective_Date')[['Terms','Principal']].sum())
print('Group Mean every Terms and Principal in Data')
print(data.groupby('Effective_Date')[['Terms','Principal']].mean())


# In[ ]:


print('Age in Data state\n')
print('Data isnull sum\n')
print(data.Age.isnull().sum())
print('Data unique age\n')
print(data.Age.unique())
print('Data value counts age\n')
print(data.Age.value_counts())
print('Age group by Terms nad Principal in every Data\n')
print(data.groupby('Terms')['Age'].value_counts())
print('Group sum every Terms and Principal in Data')
print(data.groupby('Age')[['Terms','Principal']].sum())


# In[ ]:


data_age=data.Age.value_counts().index
len(data_age)
data_age_list=[]
for age in data_age:
    data_age_list.append(sum(data[data.Age==int(age)].Principal))


# In[ ]:


print('Education State\n')
print('Education Unique\n')
print(data.Education.unique())
print('Education isnull \n')
print(data.Education.isnull().sum())
print('Education value counts\n')
print(data.Education.value_counts())
print('Education group by Count Education and Age in Data\n')
print(data.groupby('Education')['Age'].count())
print('Educaton group by Principal in Data\n')
print(data.groupby('Education')['Principal'].sum())


# In[ ]:


print('Gender State\n')
print('Gender Unique\n')
print(data.Gender.unique())
print('Gender isnull\n')
print(data.Gender.isnull().sum())
print('Value Counts in Data\n')
print(data.Gender.value_counts())
print('Gender group by education in Data\n')
print(data.groupby('Education')['Gender'].value_counts())
print('Gender and Education group by sum Principal in Data\n')
print(data.groupby(['Education','Gender'])['Principal'].sum())
print('Gender group by Principal in Data\n')
print(data.groupby('Gender')['Principal'].sum())


# In[ ]:


print('Effeftive Time State\n')
print('Effeftive Time unique\n')
print(data.Effective_Date.unique())
print('Effective Time isnull\n')
print(data.Effective_Date.isnull().sum())
print('Effective Time value counts\n')
print(data.Effective_Date.value_counts())
print('Effective_Date group by sum Principal in Data\n')
print(data.groupby('Effective_Date')['Principal'].sum())


# In[ ]:


print('Due Date State\n')
print('Due Date unique\n')
print(data.Due_Date.unique())
print('Due Date isnull\n')
print(data.Due_Date.isnull().sum())
print('Due Date value counts\n')
print(data.Due_Date.value_counts())
print('Due Date group by sum Principal in Data\n')
print(data.groupby('Due_Date')['Principal'].sum())


# In[ ]:


print('PaidOff Date State\n')
print('PaidOff Date unique\n')
print(data.Paid_Of_Time.unique())
print('PaidOff Date isnull\n')
print(data.Paid_Of_Time.isnull().sum())
print('PaidOff Date value counts\n')
print(data.Paid_Of_Time.value_counts())
print('PaidOff Date group by sum Principal in Data\n')
print(data.groupby('Paid_Of_Time')['Principal'].sum())


# In[ ]:


print('Past_Due_Days Date State\n')
print('Past_Due_Days Date unique\n')
print(data.Past_Due_Days.unique())
print('Past_Due_Days Date isnull\n')
print(data.Past_Due_Days.isnull().sum())
print('Past_Due_Days Date value counts\n')
print(data.Past_Due_Days.value_counts())
print('Past_Due_Days Date group by sum Principal in Data\n')
print(data.groupby('Past_Due_Days')['Principal'].sum())
print('Past_Due_Days Date group by Gender in Data\n')
print(data.groupby('Past_Due_Days')['Gender'].count())
print('Past_Due_Days Date group by Education in Data\n')
print(data.groupby('Education')['Past_Due_Days'].count())


# # Data Cleaning and Visualization

# In[ ]:


data=data.drop('Loan_Id',axis=1)


# In[ ]:


print(data.isnull().sum())
print('\n')
print(len(data))


# In[ ]:


data.isnull().sum()


# In[ ]:


sns.countplot(data['Loan_Status'])
plt.show()


# In[ ]:


data['Loan_Status'].unique()


# In[ ]:


data[data['Loan_Status']=='PAIDOFF'].groupby('Education')['Gender'].count()


# In[ ]:


sns.barplot(x=data[data['Loan_Status']=='PAIDOFF'].groupby('Education')['Gender'].count().index,
           y=data[data['Loan_Status']=='PAIDOFF'].groupby('Education')['Gender'].count().values)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Loan_Status']=le.fit_transform(data['Loan_Status'])


# In[ ]:


data[data['Principal']==1000].Gender.value_counts()


# In[ ]:


plt.figure(figsize=(5,5))
ax=sns.barplot(x=data[data['Principal']==1000].Gender.value_counts().index,
              y=data[data['Principal']==1000].Gender.value_counts().values,
              palette=sns.cubehelix_palette(120))
plt.xlabel('Principal')
plt.ylabel('Gender')
plt.title('Show Principal & Gender Bar Plot')
plt.show()


# In[ ]:


labels=data['Terms'].value_counts().index
colors=['blue','red','yellow']
explode=[0,0,0.1]
values=data['Terms'].value_counts().values

#visualization
plt.figure(figsize=(7,7))
plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')
plt.title('Ters According Analysis',color='black',fontsize=10)
plt.show()


# In[ ]:


sns.lmplot(x='Terms',y='Age',data=data)
plt.xlabel('Terms Values')
plt.ylabel('Age Values')
plt.title('Terms vs Age Values')
plt.show()


# In[ ]:


sns.kdeplot(data['Age'])
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Age Kde Plot System Analysis')
plt.show()


# In[ ]:


sns.kdeplot(data['Age'],shade=True,color='r')
sns.kdeplot(data['Terms'],shade=True,color='b')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Agevs Terms Kde Plot System Analysis')
plt.show()


# In[ ]:


sns.violinplot(data['Age'])
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Violin Age Score Show')
plt.show()


# In[ ]:


sns.countplot(data['Gender'],hue=data['Education'])
plt.show()


# In[ ]:


plt.figure(figsize=(7,7))
sns.countplot(y=data['Age'],palette="Set2",hue=data['Gender'])
plt.legend(loc=4)
plt.tight_layout()
plt.show()


# In[ ]:


data.head()


# # Please leave your comments / suggestions below,upvote if you liked it
