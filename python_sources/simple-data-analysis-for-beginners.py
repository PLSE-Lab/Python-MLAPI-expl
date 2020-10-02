#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.
# 
# While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

# Lets see what are the files we have to explore the data.

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


# ## How files are sturtured and link between each files and their content is given in the below image, which helps us to understand probelm well.
# 

# ![](https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png)

# Lets read all the fiels and have a glimpse of data.

# In[ ]:


PATH="../input"

application_train = pd.read_csv(PATH+"/application_train.csv")
application_test = pd.read_csv(PATH+"/application_test.csv")
bureau = pd.read_csv(PATH+"/bureau.csv")
bureau_balance = pd.read_csv(PATH+"/bureau_balance.csv")
credit_card_balance = pd.read_csv(PATH+"/credit_card_balance.csv")
installments_payments = pd.read_csv(PATH+"/installments_payments.csv")
previous_application = pd.read_csv(PATH+"/previous_application.csv")
POS_CASH_balance = pd.read_csv(PATH+"/POS_CASH_balance.csv")


# In[ ]:


application_train.head()


# In[ ]:


application_test.head()


# In[ ]:


bureau.head()


# In[ ]:


bureau_balance.head()


# In[ ]:


credit_card_balance.head()


# In[ ]:


installments_payments.head()


# In[ ]:


previous_application.head()


# In[ ]:


previous_application.head()


# ## Lets check the missing values in each file.

# In[ ]:


def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


# In[ ]:


missing_data(application_train).head(10)


# you can also check missing values like this without need of function.

# In[ ]:


application_train.isnull().sum().sort_values(ascending=False)[:10]


# In[ ]:


missing_data(application_test).head(10)


# In[ ]:


missing_data(bureau)


# In[ ]:


missing_data(bureau_balance)


# In[ ]:


missing_data(credit_card_balance)


# In[ ]:


missing_data(installments_payments)


# # Visuliazation

# In[ ]:


import seaborn as sns #it is my fav and very handy for beginners, plotly is though interactive but hard for beginners i believe.
import matplotlib.pyplot as plt 


# In[ ]:


sns.countplot(application_train.TARGET)
plt.show()
# TARGET value 0 means loan is repayed, value 1 means loan is not repayed.


# from the above image, most of the loans were repayed and less than 5000 were not payed.

# In[ ]:


sns.countplot(application_train.NAME_CONTRACT_TYPE.values,data=application_train)


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_EDUCATION_TYPE.values,data=application_train)
plt.show() #to check what are the differnet categories and their count.


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_FAMILY_STATUS.values,data=application_train)
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_HOUSING_TYPE.values,data=application_train)
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_INCOME_TYPE.values,data=application_train)
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_TYPE_SUITE.values,data=application_train)
plt.show()


# Little advanced visualization 

# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_INCOME_TYPE.values,data=application_train,hue=application_train.TARGET)
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_EDUCATION_TYPE.values,data=application_train,hue=application_train.TARGET)
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_FAMILY_STATUS.values,data=application_train,hue=application_train.TARGET)
plt.show()


# who accompanied client when appliying for the loan/application, and their repayment count given below.

# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_TYPE_SUITE.values,data=application_train,hue=application_train.TARGET)
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_CONTRACT_TYPE.values,data=application_train,hue=application_train.FLAG_OWN_REALTY)
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(application_train.OCCUPATION_TYPE.values,data=application_train)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(previous_application.NAME_CONTRACT_TYPE.values)
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(previous_application.WEEKDAY_APPR_PROCESS_START.values,data=previous_application)
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(previous_application.NAME_CLIENT_TYPE.values,data=previous_application)
plt.show()


# more in pipe line, this is for beginners and all are basic level and self explanatory only.
# 
# if you like it, please upvote for me. 
# 
# Thank you : ) 

# In[ ]:




