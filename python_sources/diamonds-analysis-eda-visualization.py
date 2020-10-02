#!/usr/bin/env python
# coding: utf-8

# <img src="https://i.ytimg.com/vi/mjUCAMFVjaY/maxresdefault.jpg" style="width:1000px;height:500px;">

# # Information
# Everybody loves diamonds. It is so important thing to show quality level to other people.There is a mistake.When they collect it , They are using African people with too low payment. Everything African people done is to work low salary that is not fair.Somebody takes their home,family,their serenity..I like all people so that i want to show diamonds vis,eda,analysis in this way if you read my information part you can reach what i think about this topic.  

# In[ ]:


## All libraries that i used
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        


# # Data Analysis 

# In[ ]:


dia=pd.read_csv("/kaggle/input/diamonds/diamonds.csv")
dia.head()
#We can reach first 5 information about data.
#I called the dataset is 'dia'


# In[ ]:


#If we wonder what our columns feature -->obeject,categorical,float,int,... we can use 'dtypes' function.
dia.dtypes


# In[ ]:


dia.describe().T
#I think we can find extra and consisten information about data instead of data_name.describe because data_name.describe() function is tall .
#People can understand easier like that.


# In[ ]:


# I check where there are NaN values
dia.isnull().any()


# In[ ]:


#Calculates the number of rows and columns
print(dia.shape)


# In[ ]:


#If we want to see , First of all we can grouping columns so that it shows every columns price mean.
#That has the difference information from other.
dia.groupby(["cut","color"])["price"].mean()


# # Data Visualization

# In[ ]:


# One of the most important visualization styles is histogram
sns.distplot(dia.price, kde = False , bins= 20);


# In[ ]:


sns.catplot(x="cut",y="price",hue="color",kind="point",data=dia);
#Catplot inside Seaborn function is significiant for me because of awesome . It contributes to Visualization.


# In[ ]:


sns.scatterplot(x="table",y="depth",data=dia);
#It gives between relationship table and depth


# In[ ]:


#It is so interesting . I want to demonstrate ratio of carat,color.
# Generally if carat greater than 3 or 2.5 is the outlier value. 
sns.boxplot (x="carat",y="color",data=dia);


# In[ ]:


#LMPLOT i think it offers smooth visuals that is why i am always using that method.I think, This method is very useful for ML algorithms.
sns.lmplot(x="price",y="carat", data=dia);


# In[ ]:


#Heatmap- Right side shows value of color left side shows values that appeal to colors
plt.figure(figsize=(10,10))
plt.title('Correlation Map')
ax=sns.heatmap(dia.corr(),
               linewidth=3.1,
               annot=True,
               center=1)


# In[ ]:


#Finally Line Plot. I want to show almost all graphical things in Seaborn thanks everybody and shows how the relations between table and depth are.
sns.lineplot(x="table",y="depth",data=dia);


#  # WHAT WE DID
# * We can see values of diamonds now.
# * We've reached relationship between columns.
# * Knew effect of price,carat,table,depth 
# * Showed some graphical things
# * Learnt explotary data analysis from data.
# THANKS FOR READING SO FAR.  IF YOU LIKE PLEASE UPVOTE. HAVE A GOOD LUCK ALL PEOPLE.
