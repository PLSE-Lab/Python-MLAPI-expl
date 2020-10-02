#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


ecom=pd.read_csv("../input/Ecommerce Purchases")


# In[ ]:


ecom.head()


# In[ ]:


#Check the Average Price


# In[ ]:


ecom['Purchase Price'].mean()


# In[ ]:


#Check the Highest and Lowest purchased Price


# In[ ]:


ecom['Purchase Price'].max()


# In[ ]:


ecom['Purchase Price'].min()


# In[ ]:


#How many people have English 'en' as their Language of choice on the website


# In[ ]:


ecom[ecom['Language']=="en"]['Language'].count()


# In[ ]:


#How many people have the job title of "Lawyer" ?


# In[ ]:


len(ecom[ecom['Job']=='Lawyer'].index)


# In[ ]:


#How many people made the purchase during the AM and how many people made the purchase during PM


# In[ ]:


ecom['AM or PM'].value_counts()


# In[ ]:


# ** What are the 5 most common Job Titles? **


# In[ ]:


ecom['Job'].value_counts().head(5)


# In[ ]:


##** Someone made a purchase that came from Lot: "90 WT" , what was the Purchase Price for this transaction? **


# In[ ]:


ecom[ecom['Lot']=='90 WT']['Purchase Price']


# In[ ]:


#What is the email of the person with the following Credit Card Number: 4926535242672853 


# In[ ]:


ecom[ecom['Credit Card']==4926535242672853]['Email']


# In[ ]:


#How many people have American Express as their Credit Card Provider *and made a purchase above $95 ?


# In[ ]:


len(ecom[(ecom['CC Provider']=='American Express')&(ecom["Purchase Price"]>95)].index)


# In[ ]:


#How many people have a credit card that expires in 2025? 


# In[ ]:


len(ecom[ecom['CC Exp Date'].apply(lambda exp:exp[3:]=='25')].index)


# In[ ]:


#What are the top 5 most popular email providers/hosts (e.g. gmail.com, yahoo.com, etc...)


# In[ ]:


ecom['Email'].apply(lambda email:email.split('@')[1]).value_counts().head(5)


# In[ ]:




