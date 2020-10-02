#!/usr/bin/env python
# coding: utf-8

# This python notebook uses an Exploratory Data Analysis approach to find insights from the states and their literacy rates. In my process, I wanted to finally see if there is an evidence that would help me conclude that for a better literacy rate amongst all the states, how necessary it is to educate women.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/cities_r2.csv')
df.head()


# Let us first try to see how the sates rank based on population and then delve deeper in the question mentioned.

# In[ ]:


x = df.groupby('state_name')
temp = x.sum().reset_index()
plt.rcParams['figure.figsize'] = (5.5, 5.0)
genre_count = sns.barplot(y='state_name', x='population_total', data=temp, palette="Blues", ci=None)


# From all the other notebooks, it was evident that Maharashtra is the most populated state. Since this plot is self explanatory, I want to move to the next part of analysis.

# Let us now see how the states fare on the basis of the effective literacy rate. Now for my point statistic I have chosen the median because the scale of literacy rate in each state is quite large and choosing median would be the right statistic.

# In[ ]:


x = df.groupby('state_name')
temp = x.median().reset_index()
plt.rcParams['figure.figsize'] = (6.0, 5.0)
genre_count = sns.barplot(y='state_name', x='effective_literacy_rate_total', data=temp, palette="Blues", ci=None)


# Well it was not very evident as to which states performed the best, so I went and sorted them to see the top and the worst performing states in the country based on the literacy rate.

# In[ ]:


temp = temp.sort_values(by='effective_literacy_rate_total', ascending=False)
print (temp[['state_name','effective_literacy_rate_total']].reset_index())


# Great result! More than **70%** of the sates had a median literacy rate of more than 85%! The next step I wanted to take was to see how does the distribution of sex ratio behave in the lower and higher literacy  ranked states.

# Since the states like Mizoram, Himachal Pradesh, Tripura, Meghalaya and Manipur had only *one* observation each, I wasn't able to plot their distributions. The top two states with more than 1 observation were Kerala and Assam while the poorly performing Jammu & Kashmir and Rajasthan have their distributions below:

# In[ ]:


x = df[(df['state_name']=='ASSAM')]
ax = sns.distplot(x['sex_ratio'], color='blue', hist=False, label='ASSAM')
x = df[(df['state_name']=='KERALA')]
ax = sns.distplot(x['sex_ratio'], color='red', hist=False, label='KERALA')
x = df[(df['state_name']=='JAMMU & KASHMIR')]
ax = sns.distplot(x['sex_ratio'], color='black', hist=False, label='JAMMU & KASHMIR')
x = df[(df['state_name']=='RAJASTHAN')]
ax = sns.distplot(x['sex_ratio'], color='green', hist=False, label='RAJASTHAN')


# Another interesting plot! Better literacy performing states have better sex ratio! But we haven't done any statistics yet so that wouldn't be a valid statement. Let us try some though by first seeing if there is a linear fit between sex_ratio and effective_literacy_rate_total:

# In[ ]:


ax = sns.regplot(x="sex_ratio", y="effective_literacy_rate_total", data=df)


# Well it doesn't look like there is a very good evidence of a linear fit between sex ratio and literacy rate. Let us see if there is a correlation between the two and how strong the correlation is:

# In[ ]:


sns.jointplot(x=df['sex_ratio'], y=df['effective_literacy_rate_total'], kind="hex", color="b");


# A good news is that there is a positive correlation between the two. Though not **very** strong, there is still an effective correlation and also the p-value is pretty small which again gives us a good statistic to decide that the correlation is not 0 amongst the two variables.

# In[ ]:


x = df.groupby('state_name')
temp = x.sum().reset_index()
sns.jointplot(x=df['total_graduates'], y=df['effective_literacy_rate_total'], kind="hex", color="b");


# The above plot seems a bit strange where we see that there is hardly any linear correlation between the total number of graduates and the effective literacy rate.

# In[ ]:


sns.jointplot(x=df['effective_literacy_rate_total'], y=df['literates_total'], kind="hex", color="b");


# Again, there is very poor linear correlation between the total number of literates and the effective literacy rate!

# In the next part of the EDA, I wanted to see the extent of literacy gap amongst both the genders and wanted to point out places, where the gap is the worst. First interesting observation I came across was that ***NO*** city in the country had literacy rate of females higher than the men. This could boil down to the fact that the sex ratio is pretty poor in the country and when there are more men, they will have an added advantage on the literacy rate.<br/>
# Next, I did a little bit of data wrangling and plotted the 10 worst places in the terms literacy gaps amongst both the gender.

# In[ ]:


df['education_gap'] = df['effective_literacy_rate_male'] - df['effective_literacy_rate_female']
temp = df.sort_values(by='education_gap', ascending=False)[:10]
x = temp[['name_of_city','state_name']]
x = pd.concat([x]*2, ignore_index=True)
x['sex'] = 'male'
x['sex'][10:] = 'female'
x['literacy'] = 0.0
x['literacy'][:10] = temp['effective_literacy_rate_male']
x['literacy'][10:] = temp['effective_literacy_rate_female']
x['place'] = x['name_of_city']+x['state_name']


# In[ ]:


g = sns.FacetGrid(x, col="place", col_wrap=4, aspect=0.7)
g = g.map(sns.barplot, "sex", "literacy").set_titles("{col_name}")


# Out of the 10 worst performing places, 8 were from Rajasthan! Doesn't speak volumes!

# In[ ]:


One of the final plots I wanted to show was how each state compares to the median graduate for the country:


# In[ ]:


plt.rcParams['figure.figsize'] = (6.0, 5.0)
x = df.groupby('state_name')
temp = x.mean().reset_index()
genre_count = sns.barplot(y='state_name', x='total_graduates', data=temp, palette="Blues", ci=None)
ax = plt.axvline(x = df['total_graduates'].mean(), color='black', linestyle='dashed', linewidth=2)


# The black dashed line represents the median number of graduates for the 500 cities in the dataset and the bar plots show the total number of graduates for each state.

# Now, for the final EDA I wanted to check which states have districts where although the female population is low, the number of female graduates are more than their men counterparts:

# In[ ]:


df[((df['female_graduates']>df['male_graduates']) & (df['population_male']>df['population_female']))]['state_name']


# 10 of the total 11 districts are from Punjab, *'Flying Punjab'*, anyone!?

# In[ ]:




