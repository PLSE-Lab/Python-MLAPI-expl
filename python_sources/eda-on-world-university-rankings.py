#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Importing libraries

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# # Loading data

# In[ ]:


df=pd.read_csv('/kaggle/input/world-university-rankings/cwurData.csv')
df.head()


# Checking Shape

# In[ ]:


df.shape


# The dataset has 14 columns and 2200 rows

# In[ ]:


df.info()


# The data has float,int and object types features and only broad_impact has 200 null values

# Now finding total number of countries and displaying them.

# In[ ]:


count=df['country'].nunique()
unique_countries=[]
for i in df['country']:
    if i not in unique_countries:
        unique_countries.append(i)
print("Total number of unique countries are {}".format(count))       
print("*"*116)
print("Unique_countries are::{}".format(unique_countries))      


# Dropping broad_impact column.

# In[ ]:


df.drop(['broad_impact'],axis=1,inplace=True)


# In[ ]:


df.describe()


# Checking for unique values in all columns.

# In[ ]:


df.nunique()


# # Top 10 universities in the world for all years.See how it changes over years.

# In[ ]:


Top10=df[['year','institution','world_rank']].groupby('year').head(10)

plt.figure(figsize=(20,8))
ax=sns.pointplot(data=Top10, x="year", y="world_rank",hue="institution",marker='o')
ax.grid(True)
plt.title('Changes in Top 10 University Ranking across years',fontsize=20,fontweight='bold')
plt.xlabel('Year',fontsize=20)
plt.ylabel('World Rank',fontsize=14)
plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.07), ncol=2)


# Harvard university is in first position for all the years.

# # Top 10 univerities based on quality of education.

# In[ ]:


Top10 = df.groupby('institution')['quality_of_education'].mean().nsmallest(10)
#plt.subplots(figsize=(20,5))
g=sns.barplot(Top10.index, Top10.values,orient='v',palette='coolwarm')
g.set_xticklabels(g.get_xticklabels(), rotation=90)


# # Top 10 universities based on alumini employment.

# In[ ]:


Top10Employment = df.groupby('institution')['alumni_employment'].mean().nsmallest(10)
e=sns.barplot(Top10Employment.index, Top10Employment.values,orient='v',palette='summer')
e.set_xticklabels(g.get_xticklabels(), rotation=90)


# # Note:In the same way you can check for remaining features like petant,cetations etc.

# Now finding correlation using heatmap

# In[ ]:


correlation=df.corr()
f,ax = plt.subplots(figsize=(16,16))
sns.heatmap(correlation,annot=True,linewidths=5,ax=ax)
plt.show()


# Checking the variables having highest correlation with world rank

# In[ ]:


correlation["world_rank"].sort_values(ascending=False)


# As you see publications has the highest correlation with world rank.

# Visually representing the correlation using seaborn regplot.

# In[ ]:


sns.regplot('world_rank','publications',data=df,color='red')


# # Now consider the data regarding to year 2012

# Desplaying 2012 year data

# In[ ]:


year_2012=df[df['year']==2012]
year_2012


# In[ ]:


year_2012.shape


# # Now finding top 10 universites in the year 2012

# In[ ]:


scores=year_2012.groupby(['institution','country'])['world_rank'].first().sort_values().head(10)
g=sns.barplot(scores.index,scores.values,orient='v',palette='coolwarm')
g.set_xticklabels(g.get_xticklabels(), rotation=90)


# # Top 10 universities in USA in the year 2012

# In[ ]:


year_2012_USA=year_2012[year_2012['country']=='USA']
year_2012_USA


# In[ ]:


USA_2012_scores=year_2012_USA.groupby(['institution'])['world_rank'].first().sort_values().head(10)
g=sns.barplot(USA_2012_scores.index,USA_2012_scores.values,orient='v',palette='coolwarm')
g.set_xticklabels(g.get_xticklabels(), rotation=90)


# # Now you can apply the same procedure for all the years and find best univerities based on any particular feature.

# In[ ]:





# In[ ]:




