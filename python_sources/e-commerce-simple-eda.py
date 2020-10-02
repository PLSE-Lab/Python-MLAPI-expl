#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#reading Required packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import *
import wordcloud
import os


# In[ ]:


#reading data
data = pd.read_csv('../input/Womens Clothing E-Commerce Reviews.csv')


# In[ ]:


data.head(10)


# In[ ]:


#dropping the unused colm.
data=data.drop(['Unnamed: 0'],axis=1)


# In[ ]:


print('No of columns:',data.shape[0])
print('No of rows:',data.shape[1])


# In[ ]:


data.dtypes


# In[ ]:


#check for missing values
data.isnull().values.any()


# In[ ]:


#finding the missing value count and percentage of all columns.
mss=data.isnull().sum()
columns = data.columns
percent_missing = data.isnull().sum() * 100 / len(data)
missing_value_data = pd.DataFrame({'missing_cnt':mss,
                                 'percent_missing': percent_missing})
missing_value_data


# In[ ]:


#since the missing value percentage is less in variables, we are replacing it with mean and mode.
for column in data.columns:
    data[column].fillna(data[column].mode()[0], inplace=True)


# In[ ]:


data.isnull().values.any()


# In[ ]:


#binning the Age
data['AGE'] = pd.cut(data['Age'], [0,10,20,30,40,50,60], labels=['0-10','11-20','21-30','31-40','41-50','Abv 50'])


# In[ ]:


#plot for age group.
sns.countplot(x ='AGE', data = data)


# In[ ]:


#plot for rating
sns.countplot(x = 'Rating', data = data)


# In[ ]:


#plot for division categories
sns.countplot(x = 'Division Name', data = data)


# In[ ]:


#plot for department categories
sns.countplot(x = 'Department Name', data = data)


# In[ ]:


#plot for age vs division
A=data.groupby(['AGE','Division Name'])['AGE'].count().unstack('Division Name')
A.plot(kind='bar', stacked=True)


# In[ ]:


#plot for division n department
B= data.groupby(['Division Name', 'Department Name'])['Division Name'].count().unstack('Department Name')
B.plot(kind='bar', stacked=True)


# In[ ]:


#boxplot for rating on various departments
sns.boxplot(x="Department Name", y="Rating", data=data,whis="range", palette="vlag")


# In[ ]:


# wordcloud for Title
A=data['Title'].str.cat(sep=' ')
# Create the wordcloud object
wordcloud = WordCloud(width=800, height=480, margin=0).generate(A)
 
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


# In[ ]:


# wordcloud for Review Text
A1=data['Review Text'].str.cat(sep=' ')
# Create the wordcloud object
wordcloud = WordCloud(width=800, height=480, margin=0).generate(A1)
 
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


# Thank You........
