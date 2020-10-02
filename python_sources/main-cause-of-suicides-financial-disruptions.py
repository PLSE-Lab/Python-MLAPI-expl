#!/usr/bin/env python
# coding: utf-8

# In this short analysis, I would try to see if the predominant driver to suicides is financial disruptions and bankruptcy situations.
# 
# It is widely beleived by many people across developing countries that financial disruptions or bankruptcy are the primary causes for suicides. A very simple analytics can prove that this is not true.

# First we import all the necessary libraries. We set the path to the current directory, and get the dataset. Then we create the pandas dataframe out of it. This provides us with a standard model with which we can work.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt;
import seaborn as sns
import os
os.chdir("../input")
df=pd.read_csv('master.csv')
df.describe()


# Let us first try to figure out which age group has maximum number of suicides against them. If we assume that financial reasons is driving suicides mostly, then people of the older age group must have a greater rate of suicide than that of the younger. As youger age population can easily earn money even in situations of bankruptcy.

# In[ ]:


p=df.groupby(['age']).sum().reset_index('age')
temp_df=p.filter(['age', 'suicides_no'], axis=1)
sns.barplot(x="age", y="suicides_no", data=temp_df)


# As we can clearly see, that contrary to our assumption, the age group which commits maximum number of suicides is the age group 35-54 years and second is from 55-74 years. That might hint towards the fact that maybe our assumptions about suicide are correct which is with increasing age suicides also increase.

# In[ ]:


plt.pie(p.suicides_no, labels=p.age); 
plt.show()


# In[ ]:


p=(df.groupby(['age'])['suicides_no'].sum() / df.groupby(['age'])['population'].sum()).reset_index('age').sort_values(0)


# As we can see from the below visualization, with increasing age, we see an increase in the number of suicides.

# In[ ]:


sns.barplot(x='age', y=0, data=p)


# If we assume that aging reduces the ability of a person to earn money and thus leads to suicide, then to confirm it, we do correlation analysis on the dataset. We assume that there must be a positive coorelation between suicides_no and population, as more population would result in larger suicides, and a negative coorelation between HDI and suicides_no and GDP_per_capita and suicides_no. 
# 
# But we find out that although population and suicides_no go as expected there is a weak positive coorelation between HDI and suicides_no and almost 0 correlation between suicides_no and GDP_per_Captia.

# In[ ]:


coorelation=df.corr();
sns.heatmap(coorelation, xticklabels=coorelation.columns.values, yticklabels=coorelation.columns.values)


# In order to visualize the relationship between GDP_per_capita and suicides_no we plot a barplot in order to look out whether an increasing gdp_per_capita decreases suicides_no.

# In[ ]:


p=df.groupby(['gdp_per_capita ($)']).sum().reset_index('gdp_per_capita ($)')
p=p.filter(['gdp_per_capita ($)', 'suicides_no'], axis=1)
sns.barplot(x='gdp_per_capita ($)', y="suicides_no", data=p)


# As visible there are distinct areas in which the plot shows a hiked suicides_no. More surprisingly, the areas belong to higher gdp_per_capita shows more suicides_no than  the one with low gdp_per_capita.

# In[ ]:


p=df.groupby(['HDI for year']).sum().reset_index('HDI for year')
p=p.filter(['HDI for year', 'suicides_no'], axis=1)
sns.barplot(x='HDI for year', y="suicides_no", data=p)


# Similarly we see that areas or countries with high HDI also shows high suicides_no. Thus, our assumption is wrong and financial disruptions or bankruptcy are in no way a predominant driving factor for suicides.

# In[ ]:




