#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.
# 
# While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

# Lets see what are the files we have to explore the data.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))


# Lets read all the fiels and have a glimpse of data.

# In[ ]:


PATH="../input"

application_train = pd.read_csv(PATH+"/home-credit-default-risk/application_train.csv")
application_test = pd.read_csv(PATH+"/home-credit-default-risk/application_test.csv")
bureau = pd.read_csv(PATH+"/home-credit-default-risk/bureau.csv")
bureau_balance = pd.read_csv(PATH+"/home-credit-default-risk/bureau_balance.csv")
credit_card_balance = pd.read_csv(PATH+"/home-credit-default-risk/credit_card_balance.csv")
installments_payments = pd.read_csv(PATH+"/home-credit-default-risk/installments_payments.csv")
previous_application = pd.read_csv(PATH+"/home-credit-default-risk/previous_application.csv")
POS_CASH_balance = pd.read_csv(PATH+"/home-credit-default-risk/POS_CASH_balance.csv")


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


# # Lets check the missing values in each file.

# In[ ]:


def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


# In[ ]:


missing_data(application_train).head(15)


# In[ ]:


application_train.isnull().sum().sort_values(ascending=False)[:15]


# In[ ]:


missing_data(application_test).head(15)


# In[ ]:


missing_data(bureau)


# In[ ]:


missing_data(bureau_balance)


# In[ ]:


missing_data(credit_card_balance)


# In[ ]:


missing_data(installments_payments)


# # Visualization

# In[ ]:


sns.countplot(application_train.TARGET)
plt.show()


# from the above image, most of the loans were repayed and less than 5000 were not payed.

# In[ ]:


sns.countplot(application_train.NAME_CONTRACT_TYPE.values,data=application_train)


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_EDUCATION_TYPE.values,data=application_train)
plt.show()


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


# # if you like it, please upvote for me.

# ![](https://www.animatedimages.org/data/media/466/animated-thank-you-image-0078.gif)
