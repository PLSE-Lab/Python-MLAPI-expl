#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


cases = pd.read_csv("../input/indonesia-coronavirus-cases/cases.csv")
confirmed_acc = pd.read_csv("../input/indonesia-coronavirus-cases/confirmed_acc.csv")
jabar = pd.read_csv("../input/indonesia-coronavirus-cases/jabar.csv")
jakarta = pd.read_csv("../input/indonesia-coronavirus-cases/jakarta.csv")
keywordtrend = pd.read_csv("../input/indonesia-coronavirus-cases/keywordtrend.csv")
patient = pd.read_csv("../input/indonesia-coronavirus-cases/patient.csv")
province_timeline = pd.read_csv("../input/indonesia-coronavirus-cases/province_timeline.csv")
#province= pd.read_csv("../input/indonesia-coronavirus-cases/province.csv")


# #Cases

# In[ ]:


cases.info()
cases[0:10]


# In[ ]:


fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
cases.groupby('date').mean().sort_values(by='acc_tested', ascending=False)['acc_tested'].plot('bar', color='r',width=0.3,title='Date acc_tested', fontsize=10)
plt.xticks(rotation = 90)
plt.ylabel('acc_tested')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(10)
ax.yaxis.label.set_fontsize(10)
print(cases.groupby('date').mean().sort_values(by='acc_tested', ascending=False)['acc_tested'][[1,2]])
print(cases.groupby('date').mean().sort_values(by='acc_tested', ascending=False)['acc_tested'][[4,5,6]])


# #confirmed_acc

# In[ ]:


confirmed_acc.info()
confirmed_acc[0:10]


# In[ ]:


fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
confirmed_acc.groupby('date').mean().sort_values(by='cases', ascending=False)['cases'].plot('bar', color='r',width=0.3,title='Date cases', fontsize=10)
plt.xticks(rotation = 90)
plt.ylabel('cases')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(10)
ax.yaxis.label.set_fontsize(10)
print(confirmed_acc.groupby('date').mean().sort_values(by='cases', ascending=False)['cases'][[1,2]])
print(confirmed_acc.groupby('date').mean().sort_values(by='cases', ascending=False)['cases'][[4,5,6]])


# In[ ]:


#New confirmed top 
cases.new_confirmed.value_counts().plot(kind='bar')
plt.show()


# In[ ]:


#acc confirmed
cases.acc_confirmed.value_counts().plot(kind='bar')
plt.show()


# In[ ]:


#acc negative
cases.acc_negative.value_counts().plot(kind='bar')
plt.show()


# In[ ]:


#being checked
cases.being_checked.value_counts().plot(kind='bar')
plt.show()


# In[ ]:


#isolated
cases.isolated.value_counts().plot(kind='bar')
plt.show()


# In[ ]:


#new released
cases.new_released.value_counts().plot(kind='bar')
plt.show()


# In[ ]:


#acc released
cases.acc_released.value_counts().plot(kind='bar')
plt.show()


# In[ ]:


#new deceased
cases.new_deceased.value_counts().plot(kind='bar')
plt.show()


# In[ ]:


#acc deceased
cases.acc_deceased.value_counts().plot(kind='bar')
plt.show()


# In[ ]:


#positive rate
cases.positive_rate.value_counts().plot(kind='bar')
plt.show()


# In[ ]:


#Negative rate
cases.negative_rate.value_counts().plot(kind='bar')
plt.show()


# In[ ]:


#decease rate
cases.decease_rate.value_counts().plot(kind='bar')
plt.show()


# In[ ]:


#release r
cases.release_rate.value_counts().plot(kind='bar')
plt.show()


# #Jakarta

# In[ ]:


jakarta.info()
jakarta[0:10]


# #Jakarta : odp_total

# In[ ]:


fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
jakarta.groupby('date').mean().sort_values(by='odp_total', ascending=False)['odp_total'].plot('bar', color='r',width=0.3,title='Date odp_total', fontsize=10)
plt.xticks(rotation = 90)
plt.ylabel('odp_total')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(10)
ax.yaxis.label.set_fontsize(10)
print(jakarta.groupby('date').mean().sort_values(by='odp_total', ascending=False)['odp_total'][[1,2]])
print(jakarta.groupby('date').mean().sort_values(by='odp_total', ascending=False)['odp_total'][[4,5,6]])


# #Jakarta : pdp_total

# In[ ]:


fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
jakarta.groupby('date').mean().sort_values(by='pdp_total', ascending=False)['pdp_total'].plot('bar', color='r',width=0.3,title='Date pdp_total', fontsize=10)
plt.xticks(rotation = 90)
plt.ylabel('pdp_total')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(10)
ax.yaxis.label.set_fontsize(10)
print(jakarta.groupby('date').mean().sort_values(by='pdp_total', ascending=False)['pdp_total'][[1,2]])
print(jakarta.groupby('date').mean().sort_values(by='pdp_total', ascending=False)['pdp_total'][[4,5,6]])


# #JABAR

# In[ ]:


jabar.info()
jabar[0:10]


# In[ ]:


fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
jabar.groupby('date').mean().sort_values(by='positive_total', ascending=False)['positive_total'].plot('bar', color='r',width=0.3,title='Date positive_total', fontsize=10)
plt.xticks(rotation = 90)
plt.ylabel('positive_total')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(10)
ax.yaxis.label.set_fontsize(10)
print(jabar.groupby('date').mean().sort_values(by='positive_total', ascending=False)['positive_total'][[1,2]])
print(jabar.groupby('date').mean().sort_values(by='positive_total', ascending=False)['positive_total'][[4,5,6]])


# #keywordtrend

# In[ ]:


keywordtrend.info()
keywordtrend[0:10]


# #patient

# In[ ]:


patient.info()
patient[0:10]


# In[ ]:


patient.gender.value_counts().plot(kind='bar')
plt.show()


# In[ ]:


patient.nationality.value_counts().plot(kind='bar')
plt.show()


# In[ ]:


patient.gender.value_counts().plot(kind='bar')
plt.show()


# In[ ]:


patient.province.value_counts().plot(kind='bar')
plt.show()


# In[ ]:


patient.hospital.value_counts().plot(kind='bar')
plt.show()


# province_timeline

# In[ ]:


#province_timeline.info()
#province_timeline[0:10]

