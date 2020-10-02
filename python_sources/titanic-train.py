#!/usr/bin/env python
# coding: utf-8

# # Titanic Data (Have to find the no. of Survived)

# Import multiple libraries and read data first

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt


# In[ ]:


df=pd.read_csv("../input/titanicdataset-traincsv/train.csv")


# Set display columns , able to see the whole columns

# In[ ]:


pd.set_option("max_columns",40)


# In[ ]:


df.head()


# In[ ]:


df.shape #it has 891 rows and 12 columns


# In[ ]:


df.ndim #it is 2d data


# In[ ]:


df.describe()


# In[ ]:


df.info() #it shows age has some missing values


# In[ ]:


df["Age"].isnull().sum()  #Age has 177 missing values


# In[ ]:


df["Age"].nunique()


# In[ ]:


df["Age"].mode()


# In[ ]:


df["Age"]=df["Age"].fillna(24)


# In[ ]:


df["Age"].isnull().sum()


# In[ ]:


df.boxplot("Age")


# Find the outliers , we do multiple techniques:
# 1) Z-score > 3
# 2) IQR

# z=(df["Age"]-df["Age"].mean/df["Age"].std)

# In[ ]:


u=df["Age"].mean()


# In[ ]:


std=df["Age"].std()


# In[ ]:


otlr=[]
for i in df["Age"]:
    z=(i-u)/std
    if z>3:
        otlr.append(i)


# In[ ]:


print(otlr) # These are the outliers


# In[ ]:


q1=df["Age"].quantile(.25)
q1


# In[ ]:


q3=df["Age"].quantile(.75)
q3


# In[ ]:


iqr=q3-q1


# In[ ]:


iqr


# In[ ]:


outlier=q3+(1.5*iqr)


# In[ ]:


outlier


# In[ ]:


df.groupby(["Sex","Pclass"])["Age"].count()


# In[ ]:


sns.barplot(x="Sex",y="Age",hue="Pclass",data=df)


# In[ ]:


df["Age"].hist()


# In[ ]:


sns.boxplot(x="Sex",y="Age",data=df)


# In[ ]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:


df["SibSp"].unique()


# In[ ]:


df["Parch"].unique()


# In[ ]:


sns.barplot(x="Survived",y="Age",hue="SibSp",data=df)


# In[ ]:


sns.barplot(x="Survived",y="Age",hue="Parch",data=df)


# In[ ]:


df.groupby(["Survived","Sex"])["Age"].count()


# 343 People are Survived in which 233 are Female and 109 are Male ,so there was a gender discrimination during they safe.

# In[ ]:


sns.factorplot('Pclass', 'Survived', hue='Sex', data=df)


# In[ ]:





# In[ ]:




