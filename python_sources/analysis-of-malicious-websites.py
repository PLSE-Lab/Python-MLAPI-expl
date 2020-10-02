#!/usr/bin/env python
# coding: utf-8

# *Blog for this: https://medium.com/@siddharth.m98/analysis-of-malicious-websites-by-its-characteristics-e202a20d9e38*

# ## 1) Importing required libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Importing all the required libraries


# ## 2) Business Understanding

# 1) WHICH COUNTRY HAS THE MOST NUMBER OF MALICIOUS WEBSITES IN THIS DATASET
# 
# 2) WHAT IS CORRELATION BETWEEN EACH COLUMN IN THE DATASET
# 
# 3) WHAT ARE USEFUL VALUES AVAILABLE TO US IN CASE WE NEED TO GO FOR FURTHER PREDICTIONS

# ## 3) Gathering information from dataset - Data Understanding

# In[ ]:


df = pd.read_csv("../input/dataset.csv")
df


# ## 4) ACCESS

# In[ ]:


df.columns


# In[ ]:


df.describe()


# ## 5) Data Prepration - Clean

# In[ ]:


df = df[df.WHOIS_COUNTRY != 'None']
whois = df.WHOIS_COUNTRY.value_counts()
# filtering out malicious website that dont belong to any country
whois


# In[ ]:


df['WHOIS_COUNTRY'].unique()


# In[ ]:


df


# In[ ]:


df.isnull().sum()
#finding if we have null values in columns


# #### Filtering out bengin websites.

# In[ ]:


df_mal = df[df.Type != 0]  
# 
df_mal


# ## 6) Analyze

# #### Plotting graph for occurances of all website for WHOIS data.

# In[ ]:


(whois/df.shape[0]).plot(kind="bar");
plt.title("WHOIS_COUNTRY");
# plotting bar graph for entire data


# In[ ]:


# function to make plot
def makePlot(city_count,column, title, ylabel, xlabel):
    """
    This function takes in common paramters and produces a plot 
    
    Parameter:
    column(str): name of column from the dataframe
    title(str): title of the chart
    xlabel(str): x-axis title
    ylabel(str): y-axis title

    """

    plt.figure(figsize=(15,5))
    sns.barplot(city_count.index, city_count.values, alpha=0.8)
    plt.title(title)
    plt.ylabel(ylabel, fontsize=15)
    plt.xlabel(xlabel, fontsize=15)
    plt.show()


# #### Filtering it to get the top 10 countries.

# In[ ]:


city_count  = df['WHOIS_COUNTRY'].value_counts()
city_count 
city_count = city_count[:10,]
#malicious data
makePlot(city_count,'WHOIS_COUNTRY', 'Top 10 countries in the World', 'Number of Occurrences','Country')

# Get country data.


# ### Answering the business questions

# #### 1) WHICH COUNTRY HAS THE MOST NUMBER OF MALICIOUS WEBSITES IN THIS DATASET

# In[ ]:


city_count  = df_mal['WHOIS_COUNTRY'].value_counts()
city_count = city_count[:10,]
makePlot(city_count,'WHOIS_COUNTRY', 'Top 10 countries in the World', 'Number of Malcious Websites','Country')

#malicious data


# In[ ]:


df_mal_state = df_mal[df_mal.WHOIS_COUNTRY == 'ES']
city_count  = df_mal_state['WHOIS_STATEPRO'].value_counts()
makePlot(city_count,'WHOIS_STATEPRO', 'States in Spain', 'Number of Malcious Websites','State')

# data of barcelona - malicious


# #### 2) WHAT IS CORRELATION BETWEEN EACH COLUMN IN THE DATASET

# In[ ]:


corrMatrix = df.corr()
plt.figure(figsize = (20,20))
sns.heatmap(corrMatrix,linewidths=2, annot=True)
plt.show()
#plotting the corellation matrix


# #### 3) WHAT ARE USEFUL VALUES AVAILABLE TO US IN CASE WE NEED TO GO FOR FURTHER PREDICTIONS - Trying to remove the columns with very huge null values.

# In[ ]:


sns.heatmap(df.isnull(), cbar=False)


# In[ ]:




