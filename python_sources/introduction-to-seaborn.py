#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing necessary modules
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


sns.set(style = 'darkgrid')


# # Relational Plot(Scatter Plot)

# In[3]:


#Loading dataset
df = pd.read_csv("../input/tips.csv")


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


#Default Relational Plot is Scatter Plot
sns.relplot(x = 'total_bill', y = 'tip', data = df)


# In[8]:


#Adding hue semantic
sns.relplot(x = 'total_bill', y = 'tip',data = df, hue = 'smoker')


# In[9]:


#Adding different marker
sns.relplot(x = 'total_bill', y = 'tip', data = df, hue = 'smoker', style = 'smoker' )


# In[10]:


#Changing hue colors
hue_colors = {"Yes": "black","No": "red"}
sns.relplot(x = 'total_bill', y = 'tip', data = df, hue = 'smoker', palette = hue_colors)


# In[11]:


#Setting hue order
sns.scatterplot(x = 'total_bill', y = 'tip', data = df, hue = 'smoker', style = 'smoker', hue_order = ["Yes", "No"] )


# In[12]:


sns.relplot(x="total_bill", y="tip", hue="size", data=df)


# In[13]:


sns.relplot(x="total_bill", y="tip", hue="size", palette="ch:r=-.5,l=.75", data=df)


# In[14]:


sns.relplot(x="total_bill", y="tip", hue="size",size = 'size', data=df)


# In[ ]:




