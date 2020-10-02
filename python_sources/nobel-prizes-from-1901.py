#!/usr/bin/env python
# coding: utf-8

# # Introduction :
#         The Nobel Prize , is a set of annual international awards bestowed in several categories by Swedish and Norwegian institutions in recognition of academic, cultural, or scientific advances.
#         The will of the Swedish scientist Alfred Nobel established the five Nobel prizes in 1895. The prizes in Chemistry, Literature, Peace, Physics, and Physiology or Medicine were first awarded in 1901.The Nobel Prize is widely regarded as the most prestigious award available in the fields of literature, medicine, physics, chemistry, economics and activism for peace.
#         In this kernel(my first kernel in python), lets explore, analyse trends,which country wins the most, youngest /oldest person to won the nobel prize and so on.
#         Please go through and provide your suggestions  and feedback for improvements. 

# ##  Loading Libraries
#         Loading the required packages used in this kernel, package seaborn has used for data visualization, pandas, numpy for data processing and manuipulations.

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set()


# ##  Sourcing the input file
#        Source is the *csv* file which has data from year 1901 to 2016. 

# In[ ]:


nobel = pd.read_csv("../input/archive.csv",parse_dates=True)


# ###  Getting first few rows of data
# Dataset has about 969 rows and 18 coulms, which has details about the nobel prize winners name, birth date, country , and prize details.

# In[ ]:


nobel.head(n=2)


# In[ ]:



nobel.info()


# In[ ]:


# Display the number of (possibly shared) Nobel Prizes handed
# out between 1901 and 2016
display(len(nobel['Prize Share']))

# Display the number of prizes won by male and female recipients.
display(nobel['Sex'].value_counts().head(10))
sex=nobel['Sex'].value_counts()
# Display the number of prizes won by the top 10 nationalities.
ctry=nobel['Birth Country'].value_counts().head(10)
ctry

cat=nobel['Category'].value_counts()


# ## Nobel Prizes by Category from 1901 to 2016

# In[ ]:


year_cat=nobel.groupby(['Year','Category'])['Laureate ID'].count().reset_index()
year_cat
g = sns.FacetGrid(year_cat, col='Category', hue='Category', col_wrap=4, )
g = g.map(plt.plot, 'Year', 'Laureate ID')
g = g.map(plt.fill_between, 'Year', 'Laureate ID', alpha=0.2).set_titles("{col_name} Category")
g = g.set_titles("{col_name}")
# plt.subplots_adjust(top=0.92)
#g = g.fig.suptitle('Evolution of the value of stuff in 16 countries')
 
plt.show()



# ## Nobel Prize by Country/Category/Sex
#       In this section plotting the nobel prize by sex, country and category

# In[ ]:


# and setting the size of all plots.

plt.rcParams['figure.figsize'] = [13, 7]
sns.barplot(x=sex.index,y=sex.values)
plt.xticks(rotation=90)
plt.title('Nobel Prizes by Sex')
plt.show()


# There exists huge gender gap between the male and female prize winners, more than 90 % male has got the nobel prize.
# 
# ## Trend in Nobel Prize

# In[ ]:


year=nobel['Year'].value_counts()

sns.lineplot(x=year.index,y=year.values,color='red')

plt.xticks(rotation=90)
plt.title('Nobel Prizes by Year')
plt.show()


# There is clear trend of ups and down of nobel prizes issued from 1901 to 2016. .  overall its the growtn trend. The prizes are increased every year.
# 
# ## Categorywise - Number of Nobel Prizes

# In[ ]:



sns.barplot(x=cat.index,y=cat.values)
plt.xticks(rotation=90)
plt.title('Nobel Prizes by Category')
plt.show()


# The Nobel Prize is widely regarded as the most prestigious award available in the fields of literature, medicine, physics, chemistry, economics and activism for peace. Economics category there was only 83 laureates, because economics field was established sonce 1969. Medicine field got the highest number of laureates Medicine.

# ## Countrywise - Which Country got the most

# In[ ]:



sns.barplot(x=ctry.index,y=ctry.values)
plt.xticks(rotation=90)
plt.title('Top 10 Countries, which got the nobel prizes the most')
plt.show()


# USA is the dominant country in receiving the prizes,  Next comes United Kingdom, Germany, France, Sweden.

# ## Born City
#        New york city around 48, nobel rpize winners had born as on 2016. Next comes the cities Paris and London.

# In[ ]:


city=nobel['Birth City'].value_counts().head(10)
sns.barplot(x=city.index,y=city.values)
plt.xticks(rotation=90)
plt.title('Top 10 City, in which nobel prize winners born')
plt.show()


# In[ ]:


# Calculating the proportion of USA born winners per decade
nobel['usa_born_winner'] = nobel['Birth Country']=="United States of America"
nobel['decade'] = (np.floor(nobel['Year']/10)*10).astype(int)
prop_usa_winners = nobel.groupby('decade',as_index=False)['usa_born_winner'].mean()

# Display the proportions of USA born winners per decade
display(prop_usa_winners)


# 

# ## USA - Prize Proportion per Decade

# In[ ]:


# Plotting USA born winners 
ax = sns.lineplot(data=prop_usa_winners, x='decade',y='usa_born_winner')

# Adding %-formatting to the y-axis
from matplotlib.ticker import PercentFormatter
ax.yaxis.set_major_formatter(PercentFormatter(1.0))


# ## Women who got the first Nobel Prize
# We all know about Marie Curie, who was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity.
# She was the first woman to win a Nobel Prize, the first person and only woman to win twice, the only person to win a Nobel Prize in two different sciences, and was part of the Curie family legacy of five Nobel Prizes.

# In[ ]:


female=nobel[nobel['Sex']=="Female"].nsmallest(1,'Year')

female[['Year','Category','Full Name','Prize']]


# ## Female Prize winners by decade

# In[ ]:



nobel['female_winner'] = np.where(nobel['Sex']=="Female", True, False)

prop_female_winners = nobel.groupby(['decade','Category'],as_index=False)['female_winner'].mean()


ax = sns.lineplot(x='decade', y='female_winner', hue='Category', data=prop_female_winners)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))


# ##  Male Winners per Decade

# In[ ]:


nobel['male_winner'] = np.where(nobel['Sex']=="Male", True, False)

prop_female_winners = nobel.groupby(['decade','Category'],as_index=False)['male_winner'].mean()


ax = sns.lineplot(x='decade', y='male_winner', hue='Category', data=prop_female_winners)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))


# ##  Repeated Nobel Prize Winners
# Finding the laurates who got nobel price more than once

# In[ ]:



repeat=nobel.groupby(['Category','Full Name']).filter(lambda group : len(group)>=2)
#repeat[repeat[['Year','Category','Full Name','Birth Country','Sex']].groupby(['Year','Category'])['Full Name'].nunique().reset_index()]>=2


# ###  Finding Age 

# In[ ]:


nobel['Birth Year'] = nobel['Birth Date'].str[0:4]
nobel['Birth Year'] = nobel['Birth Year'].replace(to_replace="nan", value=0)
nobel['Birth Year'] = nobel['Birth Year'].apply(pd.to_numeric)


# In[ ]:


nobel['Age']=nobel['Year']- nobel['Birth Year']


# Distribution of Year and Age  with Joint Plot.

# In[ ]:


sns.jointplot(x="Year",
        y="Age",
        kind='reg',
        data=nobel)

plt.show()


# ##  Distribution of Age of Winners
# The average age for receiving nobel prize seems to alwasys above 55 or near to that.For Peace category their exists exceptions , 2 outliers exists which was below 25.
# 
# 

# In[ ]:


sns.boxplot(data=nobel,
         x='Category',
         y='Age')

plt.show()


# ## Age at which Nobel Prize was won
# 
# Find the relationship between the year and the age of nobel prize winner, 

# In[ ]:


# Plotting the age of Nobel Prize winners
sns.lmplot('Year','Age',data=nobel,lowess=True, aspect=2,  line_kws={'color' : 'black'})
plt.show()


# ##  Nobel Category vs Age of Prize Winners

# In[ ]:


sns.lmplot('Year','Age',data=nobel,lowess=True, aspect=2, hue='Category')


# ## Lifespan of Nobel prize Winners
#     Lets find out the lifespan of the nobel prize winners, since we have botht he birth and death date. Also find out will getting nobel prize inceases their age

# In[ ]:


nobel['D Year'] = nobel['Death Date'].str[0:4]
nobel['D Year'] = nobel['D Year'].replace(to_replace="nan", value=0)
nobel['D Year'] = nobel['D Year'].apply(pd.to_numeric)


# In[ ]:


nobel['lifespan']=nobel['D Year']- nobel['Birth Year']


# In[ ]:


sns.boxplot(data=nobel,
         x='Category',
         y='lifespan')

plt.show()


# Above graph clearly shows that , yes getting nobel prize has impact over their lifespan time.

# ## Comparing Lifespan of Male and Female Winners

# In[ ]:


sns.boxplot(data=nobel,
         x='Sex',
         y='lifespan',
           hue='Category')
plt.show()


# In[ ]:


sns.lmplot('Year','lifespan',data=nobel,lowess=True, aspect=2,  line_kws={'color' : 'black'})
plt.show()


# ## Laureate Types

# In[ ]:


sns.countplot(nobel['Laureate Type'])
plt.show()


# ##  Oldest Nobel Prize Winner
# Finding out the oldest nobel prize winners, some of them got there nobel prize in their 90's

# In[ ]:


# The oldest winner of a Nobel Prize as of 2016
old=nobel.nlargest(5,'Age')
display(old[['Category','Full Name','Birth Country','Sex','Age']])


# ## Youngest Nobel Prize Winner
# Malala from pakistan is the youngest one to get the nobel prize at the ag of 17.

# In[ ]:



young=nobel.nsmallest(5,'Age')
display(young[['Category','Full Name','Birth Country','Sex','Age']])


# ## Organization Toppers
# Plotting the top organization , which won the nobel prize the most.

# In[ ]:


org = nobel['Organization Name'].value_counts().reset_index().head(20)

sns.barplot(x='Organization Name',y='index',data=org)
plt.xticks(rotation=90)
plt.ylabel('Organization Name')
plt.xlabel('Count')
plt.show()

