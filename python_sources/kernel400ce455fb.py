#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


data=pd.read_csv("../input/StudentsPerformance.csv")


# In[ ]:


data.head()


# In[ ]:





# In[ ]:


data.shape


# In[ ]:


data.shape


# In[ ]:


data.sample(5)


# In[ ]:


data.sample(frac=0.1)


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.corr()


# In[ ]:


data['gender'].value_counts()


# In[ ]:


data['gender'].unique()


# In[ ]:


fig=plt.figure(figsize=(7,7))
ax=fig.add_subplot(111)


# In[ ]:


ax=sns.barplot(x=data['gender'].value_counts().index, y=data['gender'].value_counts().values,hue=['female','male'])
plt.xlabel('gender')
plt.ylabel('Frequency')
plt.title('Gender Bar Plot')
plt.show()


# In[ ]:


ax=sns.pointplot(x="reading score", y="writing score", hue="gender", data=data, markers=["o","x"],
                linestyle=["-","__"])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


sns.countplot(x='gender',data=data)
plt.ylabel('Frequency')
plt.title('Gender Bar Plot')
plt.show()


# In[ ]:


sns.countplot(x='parental level of education',data=data)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


sns.barplot(x='gender',y='reading score',data=data)
plt.show()


# In[ ]:


sns.barplot(x='gender',y='reading score',hue='race/ethnicity',data=data)
plt.show()


# In[ ]:


sns.distplot(data['writing score'],bins=10,kde=True)
plt.show()


# For Histogram graph

# For Joint Plot

# In[ ]:


sns.jointplot(x='math score',y='gender',data=data)
plot.show()


# In[ ]:


sns.pairplot(data)
plt.show()


# In[ ]:


sns.boxplot(x='gender',y='math score',data=data)
plt.show()


# In[ ]:


sns.boxplot(x='gender',y='writing score',data=data)
plt.show()


# In[ ]:


sns.boxplot(x='race/ethnicity',y='writing score',data=data)
plt.show()


# In[ ]:


data.corr()


# In[ ]:


sns.heatmap(data.corr())


# In[ ]:


g=sns.catplot(x='gender',y='writing score',hue='lunch',col='race/ethnicity',data=data,kind='bar')
plt.show()


# In[ ]:


g=sns.catplot(x='gender',y='writing score',hue='lunch',col='race/ethnicity',col_wrap=3,data=data,kind='bar',height=4,aspect=0.7)
plt.show()


# In[ ]:




