#!/usr/bin/env python
# coding: utf-8

# <div id="0">
# In this homework , I will analyze and visualize human behaviours for Black Friday days. Which gender, age , city has most attandency to buy which product  <br>
# 
# 1. [Gender](#1)
# 2. [Age](#2)
# 3. [City Category](#3)
# 4. [Marial Status](#4)
# 5. [Occupation](#5)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import matplotlib.pyplot as plt # visualizing data
import seaborn as sns 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/BlackFriday.csv')


# In[ ]:


df.head(10)


# In[ ]:


#update NaN values with 0 for Product_Category_1 , Product_Category_2 , Product_Category_3
df["Product_Category_1"].fillna(0, inplace=True)
df["Product_Category_2"].fillna(0, inplace=True)
df["Product_Category_3"].fillna(0, inplace=True)


# In[ ]:


df.info()


# I create procedures here which I will use to analyze data in the next steps 

# In[ ]:


# this method draws plot by using dataframe's own plot method. get counts of df[column] for df[group]
def plot(group,column,plot):
    ax=plt.figure(figsize=(6,4))
    df.groupby(group)[column].sum().plot(plot)

# this method draws plot by using sns library. get counts of df[column] for df[group]
def plotUsingSns(group,column):
    fig1, ax1 = plt.subplots(figsize=(8,5))
    sns.countplot(df[group],hue=df[column])
    
# this method draws piechart for counts of df[column]
def pieChartByCounts(df, column):
    fig1, ax1 = plt.subplots(figsize=(8,5))
    sf = df[column].value_counts() #Produces Pandas Series
    explode =()
    for i in range(len(sf.index)):
        if i == 0:
            explode += (0.1,)
        else:
            explode += (0,)
    ax1.pie(sf.values, explode=explode,labels=sf.index, autopct='%1.1f%%', shadow=True, startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')  
    plt.tight_layout()
    plt.legend()
    plt.show()

# this method draws piechart for sf.values for sf.indexes
def pieChartByValues(sf, title, legentTitle):
    
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('small')
    cmap = plt.get_cmap("magma_r")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    fig1, ax1 = plt.subplots(figsize=(8,5))
    
    explode =()
    
    for i in range(len(sf.values)):
        if sf.index[i] == sf.idxmax():
            explode += (0.1,)
        else:
            explode += (0,)
    ax1.pie(sf.values, explode=explode,labels=sf.index, autopct='%1.1f%%', shadow=True, startangle=90, colors=colors, radius =1)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')  
    plt.tight_layout()
    plt.legend(loc='upper center',prop=fontP, bbox_to_anchor=(1.2, 1),title=legentTitle)
    plt.title(title)
    plt.show()


# # <div id="1"> 1. Gender<div/>
#     
# I will search following :
# *  Which gender are more interested in Balck Friday , Female or Male   ?
# *  Which gender interested in which product most ?

# In[ ]:


pieChartByCounts(df, 'Gender' )


# In[ ]:


plot('Gender','Purchase','bar')


# In[ ]:


#Filter data 
df_by_occupation_and_categories = df.groupby(['Gender','Product_Category_1']).count().reset_index('Product_Category_1')

#use filtered data to draw graphs for each index 
for i in list(df_by_occupation_and_categories.index.unique()):
    sf = pd.Series (df_by_occupation_and_categories['Purchase'][i].get_values() , index = df_by_occupation_and_categories['Product_Category_1'][i].get_values())
    pieChartByValues(sf , "Gender {0}".format(i), "Product Category")


# **Conclusion ** : <br>
# We can say that **men** have **more attendency** to buy product on Black Friday <br>
# **Men** are more interested in **Product 1** of Product Category 1 and **Women** are more interested in **Product 5** of Product Category 1

# # <div id="2"> 2. Age<div/>

# In[ ]:


plotUsingSns('Age','Gender')


# In[ ]:


plot('Age','Purchase','bar')


# In[ ]:


pieChartByCounts(df,'Age')


# **Conclusion**<br>
# We can say that age between **26-35** are **most** interestered in Black Friday and ages between **0-17** and **55+** are **least** interested in Black Friday

# # <div id="3"> 3. City Category<div/>

# In[ ]:


pieChartByCounts(df,'City_Category')


# Fpr city category, who are most interested according to "staying in city for years"?

# In[ ]:


plotUsingSns ('City_Category', 'Stay_In_Current_City_Years')


# **Conlusion** <br>
# * City category **B** is **most** interested and city category **A** is **least** interested in Black Friday
# * People staying for **1 year** definitely are most interested in all city_categories . For other years there is no so much difference

# # <div id="4"> 4.  Marial Status<div/>
#     * Which marial status buys which Product most?

# In[ ]:


plotUsingSns ('Marital_Status', 'Product_Category_1')


# **Conclusion**<br>
# **Marial status does not effect**  of buying 'product category 1' habbit . For two marial status Product 5 is #1 , Product 1 is #2 and product 8 is #3

# # <div id="5"> 5.  Occupation<div/>
#     * In which occupation , which Product is sold most?

# In[ ]:


#Filter data
df_by_occupation_and_categories = df.groupby(['Occupation','Product_Category_1']).count().reset_index('Product_Category_1')

# draw on filtered data for each index value of data 
for i in range (len(df_by_occupation_and_categories.index.unique())):
    sf = pd.Series (df_by_occupation_and_categories['Gender'][i].get_values() , index = df_by_occupation_and_categories['Product_Category_1'][i].get_values())
    pieChartByValues(sf , "Occupation {0}".format(i), "Product Category")


# **Conclusion**<br>
# Product_Category_1 product saled in Occupation areas are as : <br>
# * Category 1 is saled in 7 occupation areas
# * Category 5 is saled in 13 occupation areas
# * Category 8 is saled in 1 occupation areas<br>
# 
# The **most populer product category** for Product_Category_1  for Occupations is number  **5**

# 
