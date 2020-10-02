#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[ ]:


df=pd.read_csv("../input/coffee-and-code/CoffeeAndCodeLT2018.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.dtypes


# In[ ]:


df.nunique()


# In[ ]:


df.head(2)


# In[ ]:


sns.set(rc={"figure.figsize":(15,7)},style="whitegrid")


# In[ ]:


sns.countplot(x="Gender",data=df,palette="brg")


# In[ ]:


sns.countplot(x="AgeRange",data=df,hue="Gender")


# In[ ]:


sns.countplot(x="CodingHours",data=df)


# In[ ]:


sns.countplot(x="CodingHours",data=df,hue="Gender",palette="Pastel1")


# In[ ]:


sns.countplot(df["CoffeeCupsPerDay"],palette="RdBu_r")


# In[ ]:


sns.countplot(x="CoffeeCupsPerDay",data=df,hue="Gender",palette="Pastel1")


# In[ ]:


sns.countplot(df["CoffeeTime"],palette=sns.color_palette("rainbow",7))


# In[ ]:


sns.countplot(df["CoffeeType"],palette=sns.color_palette("husl",8))


# In[ ]:


df.head()


# In[ ]:


sns.catplot(x="CoffeeTime",y="CodingHours",data=df,aspect=2.5,hue="Gender")


# In[ ]:


sns.catplot(x="CoffeeTime",y="CoffeeCupsPerDay",data=df,aspect=2.5)


# In[ ]:


sns.catplot(x="CoffeeTime",y="CoffeeCupsPerDay",data=df,aspect=2.5,hue="Gender")


# In[ ]:


sns.catplot(x="AgeRange",y="CoffeeCupsPerDay",data=df,hue="Gender",aspect=2.5,kind="point")


# In[ ]:


df.head(2)


# In[ ]:


sns.boxplot(x="CoffeeTime",y="CoffeeCupsPerDay",data=df)


# In[ ]:


sns.boxplot(x="AgeRange",y="CodingHours",data=df)


# In[ ]:


sns.boxplot(x="Gender",y="CodingHours",hue="Gender",data=df,palette="Set1")


# In[ ]:


sns.kdeplot(df["CodingHours"],shade=True)


# In[ ]:


sns.FacetGrid(hue="Gender",data=df,aspect=2.5,height=5).map(sns.kdeplot,"CodingHours",shade=True).add_legend()


# In[ ]:


sns.FacetGrid(hue="AgeRange",data=df,aspect=2.5,height=5).map(sns.kdeplot,"CodingHours",shade=True).add_legend()


# In[ ]:


sns.FacetGrid(hue="CoffeeTime",data=df,aspect=2.5,height=5).map(sns.kdeplot,"CodingHours",shade=True).add_legend()


# In[ ]:


sns.lmplot(x="CoffeeCupsPerDay",y="CodingHours",data=df,aspect=2.5)


# In[ ]:


sns.lmplot(x="CoffeeCupsPerDay",y="CodingHours",data=df,hue="Gender",aspect=2.5)


# In[ ]:


sns.pairplot(df,aspect=2.5)


# In[ ]:


sns.pairplot(df,aspect=2.5,hue="Gender")


# In[ ]:


sns.pairplot(df,aspect=2.5,hue="Gender",kind="reg")


# In[ ]:




