#!/usr/bin/env python
# coding: utf-8

# ## **Introduction**
# 
# **The First question comes in our mind is that "What is the world happiness?"**
# 
# * The World happiness is the global happiness that ranks 156 countries by how happy their citizens perceive themselves to be. 
# 
# * The World Happiness Report is an annual publication of the United Nations Sustainable Development Solutions Network. It contains articles, and rankings of national happiness based on respondent ratings of their own lives, which the report also correlates with various life factors.
# 
# <img src="https://i2.wp.com/balancedachievement.com/wp-content/uploads/2017/03/2017-World-Happiness-Report.jpeg?resize=1170%2C717&ssl=1" width=400 height=400/>
# 
# **Now the next question is "How the happiness calculates from 156 different countries?"**
# 
# * The rankings of national happiness are based on a Cantril ladder survey. Nationally representative samples of respondents are asked to think of a ladder, with the best possible life for them being a 10, and the worst possible life being a 0. They are then asked to rate their own current lives on that 0 to 10 scale. 
# 
# * The report correlates the results with various life factors.
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sb
import warnings  
warnings.filterwarnings('ignore')


# **Which parameters or factors are used for calculate Overall rank (Rank of the country based on the Happiness Score) of a Country?**
# 
# * Score: A metric measured in 2015 by asking the sampled people the question: "How would you rate     your happiness on a scale of 0 to 10 where 10 is the happiest."
# * GDP per capita: The extent to which GDP contributes to the calculation of the Happiness Score.
# * Health life expectancy: The extent to which Life expectancy contributed to the calculation of the   Happiness Score
# * Freedom to make life choices: The extent to which Freedom contributed to the calculation of the     Happiness Score.
# * Perceptions of corruption: The extent to which Perception of Corruption contributes to Happiness Score.
# * Generosity: The extent to which Generosity contributed to the calculation of the Happiness Score.
# 

# In[ ]:


data_15 = pd.read_csv('/kaggle/input/world-happiness/2015.csv')
data_16 = pd.read_csv('/kaggle/input/world-happiness/2016.csv')
data_17 = pd.read_csv('/kaggle/input/world-happiness/2017.csv')
data_18 = pd.read_csv('/kaggle/input/world-happiness/2018.csv')
data_19 = pd.read_csv('/kaggle/input/world-happiness/2019.csv')
data_19


# We have data of total 156 different company with 8 different features.

# In[ ]:


len(data_19['Country or region'].unique())


# In[ ]:


data_16['year'] = 2016

cols = data_17.columns
d17 = data_17.rename(columns={cols[1]:'Happiness Rank',cols[2]:'Happiness Score',cols[5]:'Economy (GDP per Capita)',
                      cols[7]:'Health (Life Expectancy)',cols[10]:'Trust (Government Corruption)'
               })
d17['year'] = 2017

cols = data_18.columns
d18 = data_18.rename(columns={cols[0]:'Happiness Rank', cols[1]:'Country', cols[2]:'Happiness Score', cols[3]:'Economy (GDP per Capita)',
                     cols[5]:'Health (Life Expectancy)',cols[6]:'Freedom',cols[8]:'Trust (Government Corruption)'})
d18['year'] = 2018

cols = data_19.columns
d19 = data_19.rename(columns={cols[0]:'Happiness Rank', cols[1]:'Country', cols[2]:'Happiness Score', cols[3]:'Economy (GDP per Capita)',
                     cols[5]:'Health (Life Expectancy)',cols[6]:'Freedom',cols[8]:'Trust (Government Corruption)'})
d19['year'] = 2019


# In the data only 'Country' is in the text all others features are numerical. So let's calculate mean, standard deviation, five quantiles etc..
# 
# 

# In[ ]:


data_19.describe()


# **How is the Happiness Score is distributed?**
# 
# As you can see below happiness score has values above 2.85 and below 7.76.
# So there is no single country which has happiness score above 8.

# In[ ]:


plt.figure(figsize=(14,7))

plt.title("Distribution of Happiness Score")
sb.distplot(a=data_19['Score']); # here at the end of line semicolon(;) is for hiding printed object name 


# ### **Let's see relationship between different features with happiness score.**
# 
# **1. GDP per capita**
# 
# * Relationship between GDP per capita(Economy of country) has postive strong relationship with happiness score.
# * So If GDP per Capita of a country is high than Happiness Score of that country also more likely to high. 

# In[ ]:


plt.figure(figsize=(14,7))

plt.title("Happiness Score vs GDP per capita")
sb.regplot(data=data_19, x='GDP per capita', y='Score');


# ### **Top 10 Countries with high GDP (Economy)**

# In[ ]:


plt.figure(figsize=(14,7))
plt.title("Top 10 Countries with High GDP")
sb.barplot(data = data_19.sort_values('GDP per capita', ascending= False).head(10), y='GDP per capita', x='Country or region')
plt.xticks(rotation=90);


# **2. Perceptions of corruption**
# 
# Distribution of Perceptions of corruption rightly skewed that means very less number of country has high perceptions of corruption.
# That means most of the country has corruption problem.
# ## Ohhh Corruption is a very big problem for WORLD.
# 
# **How corruption can impact on Happiness Score?**
# * Perceptions of corruption data is highly skewed no wonder why the data has weak linear relationship, but as you can see in scatter plot most of the data points are on left side and most of the countries with low perceptions of corruption has happiness score between 4 to 6.
# 
# * Countries with high perception score has high happiness score above 7.

# In[ ]:


plt.figure(figsize= (15,7))

plt.subplot(1,2,1)
plt.title("Perceptions of corruption distribution")
sb.distplot(a=data_19['Perceptions of corruption'], bins =np.arange(0, 0.45+0.2,0.05))
plt.ylabel('Count')

plt.subplot(1,2,2)
plt.title("Happiness Score vs Perceptions of corruption")
sb.regplot(data=data_19, x='Perceptions of corruption', y='Score');


# ### **Top 10 Countries with high Perceptions of corruption**

# In[ ]:


plt.figure(figsize=(14,7))
plt.title("Top 10 Countries with High Perceptions of corruption")
sb.barplot(data = data_19.sort_values('Perceptions of corruption', ascending= False).head(10), x='Country or region', y='Perceptions of corruption')
plt.xticks(rotation=90);


# **3. Healthy life expectancy**
# 
# * Healthy life expectancy has strong and positive relationship with happiness score.
# * So If country has High life expectancy it can also have high happiness score. This make sense because anyone who has very long healthy life he/she is obviously happy. 
# * I will if i get long healthy life. What about you?

# In[ ]:


plt.figure(figsize=(14,7))

plt.title("Happiness Score vs Healthy life expectancy")
sb.regplot(data=data_19, x='Healthy life expectancy', y='Score');


# ### **Top 10 Countries with high Healthy life expectancy**

# In[ ]:


plt.figure(figsize=(14,7))
plt.title("Top 10 Countries with High Healthy life expectancy")
sb.barplot(data = data_19.sort_values('Healthy life expectancy', ascending= False).head(10), x='Country or region', y='Perceptions of corruption')
plt.xticks(rotation=90);


# **4. Social Support**
# 
# * Social support of countries also have strong and positive relationship with happiness score.
# * Also relationship with happiness score needs to be strong because more you will help socially more you will be happy.
# 

# In[ ]:


plt.figure(figsize=(14,7))

plt.title("Happiness Score vs Social Support")
sb.regplot(data=data_19, x='Social support', y='Score');


# ### **Top 10 Countries with high Social Support**

# In[ ]:


plt.figure(figsize=(14,7))
plt.title("Top 10 Countries with Social Support")
sb.barplot(data = data_19.sort_values('Social support', ascending= False).head(10), x='Country or region', y='Social support')
plt.xticks(rotation=90);


# **5. Freedom to make life choices**
# 
# * Freedom to make life choices has positive relationship with happiness score. This relation make sense because of more you will get freedom to make decision about your life more you will be happy.

# In[ ]:


plt.figure(figsize=(14,7))

plt.title("Happiness Score vs Freedom to make life choices")
sb.regplot(data=data_19, x='Freedom to make life choices', y='Score');


# ### **Top 10 Countries with high Freedom to make life choices**

# In[ ]:


plt.figure(figsize=(14,7))
plt.title("Top 10 Countries with High Freedom to make life choices")
sb.barplot(data = data_19.sort_values('Freedom to make life choices', ascending= False).head(10), x='Country or region', y='Freedom to make life choices')
plt.xticks(rotation=90);


# **6. Generosity**
# 
# Generosity has very weak linear relationship with Happiness score.
# Suddenly we get question in our mind that,
# 
# Why the generosity has not linear relationship with happiness score?
# 
# Generosity score based on the countries which gives the most to nonprofits around the world. Countries which are not generous that does not mean they are not happy.

# In[ ]:


plt.figure(figsize=(14,7))

plt.title("Happiness Score vs Generosity")
sb.regplot(data=data_19, x='Generosity', y='Score');


# ### **Top 10 Countries with high Generosity**

# In[ ]:


plt.figure(figsize=(14,7))

plt.title("Top 10 Countries with High Generosity")
sb.barplot(data = data_19.sort_values('Generosity', ascending= False).head(10), x='Country or region', y='Generosity')
plt.xticks(rotation=90);


# ## **How one feature is related to another feature?**

# In[ ]:


p = sb.PairGrid(data_19)
p.map_diag(plt.hist)
p.map_offdiag(plt.scatter);


# **Below heatmap shows correlation between features.**
# 
# Happiness score is highly correlated with 
# > ***GDP per capita > Social support==Healthy life expectancy>Freedom to make life choices>Perceptions of corruption***
# 
# Happiness score is not much correlated with Generosity.

# In[ ]:


plt.figure(figsize=(14,7))

plt.title("Correlation Heatmap")
sb.heatmap(data=data_19.corr(), annot=True, vmin=0.005,cmap= 'viridis_r');


# In[ ]:


plt.figure(figsize=(15,75))
plt.title('Country vs Happiness Score')
sb.barplot(data=data_19.sort_values('Score', ascending=False), x='Score', y='Country or region');


# ### **Top 10 Countries with high Happiness Score**
# 
# So ***Finland*** is world's happiness country. 

# In[ ]:


plt.figure(figsize=(14,7))
plt.title("Top 10 Countries with High Happiness Score")
sb.barplot(data = data_19.sort_values('Score', ascending= False).head(10), x='Country or region', y='Score')
plt.xticks(rotation=90);


# You can see above world's top 10 countries in happiness they don't have much difference between the happiness score.
# 
# Now Let's compare the top 5 countries in happiness with different features.
# 

# In[ ]:


top_5_country = data_19.sort_values('Score', ascending= False).head(5)['Country or region']
generosity_rank = [np.where(data_19.sort_values('Generosity', ascending= False).reset_index()['Country or region']==i)[0][0] +1 for i in top_5_country]
gdp_rank = [np.where(data_19.sort_values('GDP per capita', ascending= False).reset_index()['Country or region']==i)[0][0] +1 for i in top_5_country]
Social_Support_rank = [np.where(data_19.sort_values('Social support', ascending= False).reset_index()['Country or region']==i)[0][0] +1 for i in top_5_country]
Healthy_life_exp_rank = [np.where(data_19.sort_values('Healthy life expectancy', ascending= False).reset_index()['Country or region']==i)[0][0] +1 for i in top_5_country]
Freedom_choice__rank = [np.where(data_19.sort_values('Freedom to make life choices', ascending= False).reset_index()['Country or region']==i)[0][0] +1 for i in top_5_country]
Perce_corruption_rank = [np.where(data_19.sort_values('Perceptions of corruption', ascending= False).reset_index()['Country or region']==i)[0][0] +1 for i in top_5_country]


# In[ ]:


feature_rank_top_5_country = pd.DataFrame({
    'country':top_5_country,
    'Generosity_rank':generosity_rank,
    'GDP_rank':gdp_rank,
    'Social_Support_rank':Social_Support_rank,
    'Healthy_life_expectancy_rank':Healthy_life_exp_rank,
    'Freedom_make_choices_rank':Freedom_choice__rank,
    'Perceptions_of_corruption_rank':Perce_corruption_rank})
feature_rank_top_5_country 


# In[ ]:


plt.figure(figsize=(15,7))
base_color = sb.color_palette()[0]
for i, col in enumerate(feature_rank_top_5_country.columns[1:]):
        plt.subplot(2,3,i+1)
        sb.barplot(data=feature_rank_top_5_country,y=col, x='country', color=base_color)
        plt.xticks(rotation=15)
    


# As you can see above, The first country in happiness **Finland** has ***high generosity, GDP and healthy life expectancy*** than other 4 countries.

# ### **Now let's build an animation map that shows us all the happiness score from *2016 to 2019* on geographic map.**

# In[ ]:


data_16['year'] = 2016

cols = data_17.columns
d17 = data_17.rename(columns={cols[1]:'Happiness Rank',cols[2]:'Happiness Score',cols[5]:'Economy (GDP per Capita)',
                      cols[7]:'Health (Life Expectancy)',cols[10]:'Trust (Government Corruption)'
               })
d17['year'] = 2017

cols = data_18.columns
d18 = data_18.rename(columns={cols[0]:'Happiness Rank', cols[1]:'Country', cols[2]:'Happiness Score', cols[3]:'Economy (GDP per Capita)',
                     cols[5]:'Health (Life Expectancy)',cols[6]:'Freedom',cols[8]:'Trust (Government Corruption)'})
d18['year'] = 2018

cols = data_19.columns
d19 = data_19.rename(columns={cols[0]:'Happiness Rank', cols[1]:'Country', cols[2]:'Happiness Score', cols[3]:'Economy (GDP per Capita)',
                     cols[5]:'Health (Life Expectancy)',cols[6]:'Freedom',cols[8]:'Trust (Government Corruption)'})
d19['year'] = 2019


# In[ ]:


features = ['year','Country', 'Happiness Rank', 'Happiness Score', 'Economy (GDP per Capita)', 'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)', 'Generosity',]


# In[ ]:


all_year_data = pd.concat([data_16[features], d17[features], d18[features], d19[features]])


# In[ ]:


all_year_data


# In[ ]:


#Reference:- https://plot.ly/python/choropleth-maps/#choropleth-map-with-plotlyexpress
from plotly.offline import iplot,init_notebook_mode
init_notebook_mode(connected=True)
import plotly.express as px
fig = px.choropleth(all_year_data, locations="Country", locationmode='country names',
                     color="Happiness Score",
                     hover_name="Country",
                     animation_frame="year",
                     color_continuous_scale=px.colors.sequential.Plasma)
fig.update_layout(
    title={
        'text': "World Happiness Index 2016-2019",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
iplot(fig)


# <img src="https://frontiermarketnews.files.wordpress.com/2017/03/coverhappy.jpg"/>

# ## If you like this kernel please consider to upvoting it.
