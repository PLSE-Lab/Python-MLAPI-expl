#!/usr/bin/env python
# coding: utf-8

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


# Fetching the latest data from website and storing it to Latest_covid.csv

# In[ ]:


import urllib.request
from bs4 import BeautifulSoup
import csv

f=open('21-04-2020.csv','w',newline='')
writer=csv.writer(f)

soup=BeautifulSoup(urllib.request.urlopen("https://www.mohfw.gov.in/").read(),'lxml')



tbody=soup('table',{"class":"table table-striped"})[0].find_all("tr")
for rows in tbody:
    cols=rows.findChildren(recursive=False)
    cols=[ele.text.strip() for ele in cols]
    writer.writerow(cols)
    print(cols)


# importing required libraries

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# importing data  and viewing first 5 records

# In[ ]:


data=pd.read_csv('../input/dynamiccovid19india-statewise/21-04-2020.csv')
df=data[:-3]
df.head()


# renaming of column names.........................

# In[ ]:


mapping = {df.columns[2]: 'Confirmed', df.columns[3]:'cured'}
df=df.rename(columns=mapping) 


# viewing dataset   and performing sum of all values(cases)  and using groupby func

# In[ ]:


df2=df.groupby('Name of State / UT')[['Confirmed','cured','Death']].sum()
df.head()


# viewinf info of the dataset

# In[ ]:


df.info()


# **Analysis and Visualizations**

# Pie chart (Each state percentage of confirmed cases over the total cases confirmed**)**

# In[ ]:


import matplotlib.pyplot as plt
perc=[]
for i in df2.Confirmed:
    per=i/len(df2)
    perc.append(i)
plt.figure(figsize=(25,15))    
plt.title('states with confirmed cases (Percentage distribution)',fontsize=40,color="yellow")
plt.pie(perc,autopct='%1.1f%%',)
plt.legend(df2.index,loc='upper left')


# ****Comparing cases of indian Confirmed and Cured and death

# In[ ]:


plt.figure(figsize=(40,35))

plt.suptitle('Comparing cases of indian Confirmed and Cured and death',fontsize=40)

plt.subplot(221)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.barh(df2.index,df2.Confirmed,color='darkmagenta',edgecolor='black',linewidth=3)


plt.subplot(222)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.barh(df2.index,df2.cured,color='blue',edgecolor='black',linewidth=3)



plt.subplot(223)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.barh(df2.index,df2.Death,color='yellow',edgecolor='black',linewidth=3)


# ****axisgrid and pairgrid plot of top 10 states affected****

# In[ ]:


import seaborn as sns
df2=df2.nlargest(10,'Confirmed')
df2['Name of State / UT']=df2.index
sns.pairplot(df2,hue='Name of State / UT')


# ****Plots of top affected countries

# In[ ]:


df2=df2.nlargest(20,'Confirmed')
plt.figure(figsize=(15,10))
plt.title('top 20 states with confirmed cases',fontsize=30)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('State',fontsize=20)
plt.ylabel('Cases',fontsize=20)
plt.plot(df2.index,df2.Confirmed,marker='o',mfc='black',label='Confirmed',markersize=10,linewidth=5)
plt.plot(df2.index,df2.Death,marker='o',mfc='black',label='Deaths',markersize=10,linewidth=5)
plt.plot(df2.index,df2.cured,marker='o',mfc='black',label='Cured',markersize=10,linewidth=5,color='green')
plt.legend(fontsize=20)


# Labelling attribute names

# In[ ]:


lbl=LabelEncoder()
df2['Name of State / UT']=lbl.fit_transform(df2['Name of State / UT'])


# Models for prediction

# In[ ]:


tree=DecisionTreeRegressor()
linear=LinearRegression()
logistic=LogisticRegression()
nb=GaussianNB()
forest=RandomForestClassifier()


# Training,testing of the model

# In[ ]:


x=df2[['Name of State / UT','Confirmed','cured','Death']]
y=df2['Confirmed']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# Predicting of certain models

# In[ ]:


tree.fit(x_train,y_train)
linear.fit(x_train,y_train)
logistic.fit(x_train,y_train)
nb.fit(x_train,y_train)
forest.fit(x_train,y_train)


# In[ ]:


from sklearn.metrics import r2_score

prediction1=logistic.predict(x_test)
score1=r2_score(y_test,prediction1)

prediction2=linear.predict(x_test)
score2=r2_score(y_test,prediction2)


prediction3=forest.predict(x_test)
score3=r2_score(y_test,prediction3)

prediction4=nb.predict(x_test)
score4=r2_score(y_test,prediction4)

prediction5=tree.predict(x_test)
score5=r2_score(y_test,prediction5)


# ******Plotting models with score

# In[ ]:


scores=[score1,score2,score3,score4,score5]
models=['logistic','Linear regression','Random forest','GaussianNB','DecisionTreeRegressor']
plt.figure(figsize=(30,15))
plt.title('Comparing Accuracy of different models',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('models',fontsize=30)
plt.ylabel('Accuracy',fontsize=30)
plt.bar(models,scores,color=['cyan','blue','green'],alpha=0.5,linewidth=3,edgecolor='black')

for i,v in enumerate(scores):
    plt.text(i-.15,v+.03,format(scores[i],'.2f'),fontsize=20)


# **Can view the accuracy of different models****

# 
