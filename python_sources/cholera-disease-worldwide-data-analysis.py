#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Table Of Context:
# 
# <a href="#eda">Exploratory Analysis</a>
# 
# <a href="#correlation">Correlation Analysis
# </a>
# 
# <a href="#task">Task: See which are the countries that haven't had a Cholera case reported in the past 10 years?
# </a>
# 
# <a href="#spatial">Spatial Analysis of 2016
# </a>
# 
# <a href="#temporal">Temporal Analysis
# </a>
# 
# <a href="#bangladesh">Bangladesh Analysis
# </a>
# 
# <a href="#summary">Summary
# </a>
# 

# # Load the libraries
# 

# In[ ]:


import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
# library for seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
# for map
import plotly.express as px
# set seaborn style
sns.set_style("white")
plt.style.use('fivethirtyeight')


# # Load the data

# In[ ]:


df = pd.read_csv("../input/cholera-dataset/data.csv")


# <a id="eda"></a>

# # Exploratory Data Analysis
# 

# In[ ]:


df.head()


# In[ ]:


# check data types
df.dtypes


# We can see that Country, Number of reported cases of cholera, Number of reported deaths from cholera, Cholera case fatality rate, WHO Region don't' have a correct data type.

# In[ ]:


df[df['Number of reported cases of cholera'] == '3 5']


# In[ ]:


df.isnull().sum()


# In[ ]:


# visualize missing values
msno.matrix(df);


# ## Data Type Correction

# We'll first replace all NaN by 0 in the numeric columns.

# In[ ]:


# replace missing values in numeric columns with 0

df.replace(np.nan, '0', regex = True, inplace = True)
# check missing value count
df.isnull().sum()


# 'Unknown' was used as a missing values. We'll replace it by 0.

# In[ ]:


# there are Unknown in cells which are creating problem
df.replace('Unknown', '0', regex = True, inplace = True)


# There were space between numbers for one row and 0.0 were used twice. We'll remove the space and make 0.0 0.0 to single 0.0.

# In[ ]:


df['Number of reported cases of cholera'] = df['Number of reported cases of cholera'].str.replace('3 5','0')
df['Number of reported deaths from cholera'] = df['Number of reported deaths from cholera'].str.replace('0 0','0')
df['Cholera case fatality rate'] = df['Cholera case fatality rate'].str.replace('0.0 0.0','0')


# In[ ]:


##### correct data types
df.Country = df.Country.astype("string")
df['Year'] = df['Year'].astype("int")
df['Number of reported cases of cholera'] = df['Number of reported cases of cholera'].astype("int64")
df['Number of reported deaths from cholera'] = df['Number of reported deaths from cholera'].astype("int64")
df['Cholera case fatality rate'] = df['Cholera case fatality rate'].astype("float")
df['WHO Region'] = df['WHO Region'].astype("string")


# ## Descriptive Statistics

# In[ ]:


df.describe()


# In[ ]:


df.Country.value_counts()


# In[ ]:


df.Country.nunique()


# There are data only about 162 countries.

# In[ ]:


df.head()


# ## Outlier Detection

# ## Number of reported cases of cholera

# In[ ]:


sns.boxplot('Number of reported cases of cholera',data = df)
plt.title("Boxplot of Number of reported cases of cholera")
plt.xlabel("Cholera Cases");


# ## Number of reported deaths from cholera	

# In[ ]:


sns.boxplot('Number of reported cases of cholera',data = df)
plt.title("Boxplot of Number of Number of reported deaths from cholera")
plt.xlabel("Reported Deaths");


# ## Cholera case fatality rate

# In[ ]:


sns.boxplot('Number of reported cases of cholera',data = df)
plt.title("Boxplot of Number of Cholera case fatality rat")
plt.xlabel("Fatality rat");


# We can see that cholera disease has a very high number of outliers in all 3 measures.

# In[ ]:


# subset data for 2016
df_16 = df[df.Year == 2016]


# ## Number of reported deaths from cholera in 2016 with fatality Rate

# In[ ]:


# we'll exclude the countries with fatality rate 0
fig = px.sunburst(df_16[df_16['Cholera case fatality rate']!=0], path=['WHO Region', 'Country'],                   color = 'Cholera case fatality rate',
                  values='Number of reported deaths from cholera',hover_data=['Cholera case fatality rate'])
fig.show()


# ## Number of reported cases of cholera in 2016 with fatality Rate

# In[ ]:


# we'll exclude the countries with fatality rate 0
fig = px.sunburst(df_16[df_16['Cholera case fatality rate']!=0], path=['WHO Region', 'Country'],                   color = 'Cholera case fatality rate',
                  values='Number of reported cases of cholera',hover_data=['Number of reported cases of cholera'])
fig.show()


# ## Cholera Reported Cases and Reported Death

# In[ ]:


# we'll exclude the countries with fatality rate 0
fig = px.sunburst(df_16[df_16['Cholera case fatality rate']!=0], path=['WHO Region', 'Country'],                   color = 'Number of reported deaths from cholera',
                  values='Number of reported cases of cholera',hover_data=['Number of reported cases of cholera'])
fig.show()


# <a href="" id="correlation"></a>

# # Correlation Analysis
# 

# In[ ]:


# calculate correlaiton
corr = df.drop(["Country","WHO Region"], axis = 1).corr()


# In[ ]:


corr


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
mask = np.triu(np.ones_like(corr, dtype=np.bool))
sns.heatmap(corr,center=0,mask=mask,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});


# We can say that Year has a low negative correlation with all other variables. This means with as we progressed yearly other cases of cholera decreased. 
# 
# The number of reported cases of cholera has a weak positive correlation with the number of reported deaths from cholera and negligible correlation with the Cholera case fatality rate. We can say that cholera death is related to the number of cholera cases but the fatality rate is not related.
# 
# We can see that the Cholera case fatality rate has a positive weak correlation with the number of reported cases.
# 
# In summary cholera cases have decreased over the years. The number of deaths and fatality rate increases with an increasing number of cases.

# <a href="" id="task"></a>

# # See which are the countries that haven't had a Cholera case reported in the past 10 years?
# 

# In[ ]:


df.Year.describe()


# We see that year values range from 1949 to 2016. Considering the latest data of 2016 and the past 10 years from 2016 we'll here find out which countries didn't have any cholera report from 2007 to 2016.

# In[ ]:


# We'll subset data for last 10 years
df_last_ten = df[(df.Year <= 2016) & (df.Year >= 2007)]


# In[ ]:


df_last_ten.Year.nunique()


# In[ ]:


df_last_ten.describe()


# In[ ]:


# subset countries that don't have cholera in last 10 years


# In[ ]:


# count number of cases in last 10 years for each contry
total_ten = df_last_ten.groupby('Country')['Number of reported cases of cholera'].sum()


# In[ ]:


total_ten.sort_values()[0:10].plot(kind= 'bar')
plt.title("Bottom 10 Countries with Least\n amount of Cholera Cases in 2016-2007");


# In[ ]:


total_ten.sort_values(ascending=False)[0:10].plot(kind= 'bar')
plt.title("Top 10 Countries with Most\n amount of Cholera Cases in 2016-2007");


# In[ ]:


total_ten[total_ten == 0]


# In[ ]:


df_last_ten[df_last_ten.Country == "Slovenia"]


# In[ ]:


df[df.Country == "Slovenia"]


# We can see that Slovenia has 0 cases of cholera in last 10 years due to the fact that it had data about cholera about Slovenia.

# ## We can only summarize which countries have the least amount of cholera cases in 2007-2016.

# In[ ]:


# countries with least amount of cholera cases
total_ten.sort_values()[0:10]


# <a href="" id="spatial"></a>

# # Spatial Analysis of Cholera for 2016

# ## Number of reported cases of cholera

# In[ ]:


# subset data for 2016
df_16 = df[df.Year == 2016]


# In[ ]:


fig = px.choropleth(df_16, locations="Country", color='Number of reported cases of cholera',                    locationmode = 'country names',
                    hover_name="Country", animation_frame="Year")
fig.show()


# ## Number of reported deaths from cholera
# 

# In[ ]:


fig = px.choropleth(df_16, locations="Country", color='Number of reported deaths from cholera',                    locationmode = 'country names',
                    hover_name="Country", animation_frame="Year")
fig.show()


# ## Cholera case fatality rate
# 

# In[ ]:


fig = px.choropleth(df_16, locations="Country", color='Cholera case fatality rate',                    locationmode = 'country names',
                    hover_name="Country", animation_frame="Year")
fig.show()


# <a href="" id="temporal"></a>

# # Temporal Analysis

# ## Number of reported cases of cholera

# In[ ]:


df.groupby('Year')['Number of reported cases of cholera'].mean().plot()
plt.title("Average Number of reported cases of cholera");


# ## Number of reported deaths from cholera

# In[ ]:


df.groupby('Year')['Number of reported deaths from cholera'].mean().plot()
plt.title("Average Number of reported deaths from cholera");


# ## Cholera case fatality rate

# In[ ]:


df.groupby('Year')['Cholera case fatality rate'].mean().plot()
plt.title("Average Cholera case fatality rate");


# <a href="" id="bangladesh"></a>

# # Bangladesh Cholera Analysis

# In[ ]:


df_bangladesh = df[df.Country == "Bangladesh"]
df_bangladesh.head()


# In[ ]:


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Cases', color=color)
ax1.plot(df_bangladesh.Year,df_bangladesh[ 'Number of reported cases of cholera'] , color="#fa26a0",        label = 'Cases')
ax1.plot(df_bangladesh.Year,df_bangladesh[ 'Number of reported deaths from cholera'] , color="tab:red",        label="Deaths")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Fatality Rate', color=color)  # we already handled the x-label with ax1
ax2.plot(df_bangladesh.Year, df_bangladesh['Cholera case fatality rate'], color=color,        label = 'Fatality rate')
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
ax1.legend(loc="upper right")
ax2.legend(loc="center right")
plt.title("Bangladesh Cholera Status 1973-2000")
plt.show()


# Bangladesh has data only year up to 2010. Though there is slight increase in the number of cases of cholera we can see that overall Bangladesh has improved its fight from cholera.

# <a href="" id="summary"></a>

# # Summary

# This data set has some data quality issues like missing values, data type mismatch, invalid input of numbers. Which were corrected before in the analysis? We can see that there are a very small number of outliers. We also found that all countries don't have data for all the years.
# 
# In 2016, we can see that the Democratic Republic of the Congo, Somalia, Haiti, the United Republic of Tanzania, Yemen, South Sudan, Kenya, Malawi, Nigeria, Dominican Republic had the highest number of reported deaths. In 2016 Haiti, the Democratic Republic of the Congo, Yemen, Somalia, the United Republic of Tanzania, Kenya, South Sudan, Malawi, Dominican Republic, Mozambique had the highest number of cholera cases.
# 
# In 2016 Niger, Congo, Zimbabwe, Nigeria, Angola, Somalia, Democratic Republic of the Congo, Malawi, Dominican Republic, Uganda had the highest number of fatality rates.
# 
# Over the years all around the years, average number of cases, average death, and the fatality rate have decreased but few countries are mostly hampered by cholera disease. We need to focus on these countries more to eradicate cholera.

# <h1>If you've found it valuable please upvote.</h1>
