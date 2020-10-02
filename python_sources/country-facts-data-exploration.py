#!/usr/bin/env python
# coding: utf-8

# In this kernel i am going to do some basic analysis including top 10 countries per feature and correlation among features.

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd

df = pd.read_csv("../input/countries of the world.csv", decimal=',')
df.head()


# In[ ]:


import seaborn as sns
sns.set(style="whitegrid", font_scale=1.3)


# ## Top 10 List

# **Top 10 countries by GDP**

# In[ ]:


sns.barplot(x="GDP ($ per capita)", y="Country", data=df.sort_values(['GDP ($ per capita)'], ascending=False).reset_index(drop=True)[:10]);


# **Top 10 countries by Population Density**

# In[ ]:


sns.barplot(x="Pop. Density (per sq. mi.)", y="Country", data=df.sort_values(['Pop. Density (per sq. mi.)'], ascending=False).reset_index(drop=True)[:10]);


# **Top 10 Countries by Area**

# In[ ]:


sns.barplot(x="Area (sq. mi.)", y="Country", data=df.sort_values(['Area (sq. mi.)'], ascending=False).reset_index(drop=True)[:10]);


# [Suprisingly USA is bigger than China](https://www.quora.com/Which-is-larger-China-or-America)

# **Top 10 Countries by population**

# In[ ]:


sns.barplot(x="Population", y="Country", data=df.sort_values(['Population'], ascending=False).reset_index(drop=True)[:10]);


# **Top 10 Countries by Literacy Rate**

# In[ ]:


sns.barplot(x="Literacy (%)", y="Country", data=df.sort_values(['Literacy (%)'], ascending=False).reset_index(drop=True)[:10]);


# Alot of Countries seems to have close to 100 % literacy rate. There is also a list on wikepedia : [List of countries by literacy rate](https://en.wikipedia.org/wiki/List_of_countries_by_literacy_rate)

# **Top 10 Countries by Costal Area Ratio**

# In[ ]:


sns.barplot(x="Coastline (coast/area ratio)", y="Country", data=df.sort_values(['Coastline (coast/area ratio)'], ascending=False).reset_index(drop=True)[:10]);


# **Top 10 Countries with highest birthrate**

# In[ ]:


sns.barplot(x="Birthrate", y="Country", data=df.sort_values(['Birthrate'], ascending=False).reset_index(drop=True)[:10]);


# **Top 10 Countries with highest Deathrate**

# In[ ]:


sns.barplot(x="Deathrate", y="Country", data=df.sort_values(['Deathrate'], ascending=False).reset_index(drop=True)[:10]);


# **Top 10 Countries with highest [Infant Mortality Rate](https://en.wikipedia.org/wiki/Infant_mortality)**

# In[ ]:


sns.barplot(x="Infant mortality (per 1000 births)", y="Country", data=df.sort_values(['Infant mortality (per 1000 births)'], ascending=False).reset_index(drop=True)[:10]);


# ## Co-Relations between features

# In[ ]:


import matplotlib.pyplot as plt
df = df.dropna()
print(df.shape)
sns.set(font_scale=1.7)
sns.heatmap(df.corr(), annot=True, fmt=".2f", linewidths=1, cmap='viridis', ax=plt.subplots(figsize=(30,30))[1]);


# Now we plot individual co-relations.

# **GDP and Birthrate**
# 
# Countries with high GDP have low birthrate. 

# In[ ]:


sns.set(font_scale=1.5)
sns.regplot(x='GDP ($ per capita)', y='Birthrate', data=df, logx=True, truncate=True, line_kws={"color": "red"});


# **GDP and Deathrate**
# 
# 

# In[ ]:


sns.regplot(x='GDP ($ per capita)', y='Deathrate', data=df, logx=True, truncate=True, line_kws={"color": "red"});


# **GDP and Climate**
# 
# There is no plausiable relation between climate and GDP.

# In[ ]:


sns.scatterplot(x='GDP ($ per capita)', y='Climate', data=df);


# **GDP and Literacy**
# 
# Countries with high GDP have high Literacy rate.

# In[ ]:


sns.regplot(x='GDP ($ per capita)', y='Literacy (%)', data=df, logx=True, truncate=True, line_kws={"color": "red"});


# **GDP and Phones (per 1000)**
# 
# Number of people using phones is very strongly co-related with GDP

# In[ ]:


sns.regplot(x='GDP ($ per capita)', y='Phones (per 1000)', data=df, truncate=True, line_kws={"color": "red"});


# **GDP and Net Migration**
# 
# Net migration gets to a more positive value as GDP increases. That means more people are willing to relocate to countries with high GDP

# In[ ]:


sns.regplot(x='GDP ($ per capita)', y='Net migration', data=df, truncate=True, line_kws={"color": "red"});


# **Infant Mortality and GDP**
# 
# Infant mortality gets lower as GDP increases.

# In[ ]:


sns.regplot(x='Infant mortality (per 1000 births)', y='GDP ($ per capita)', data=df, logx=True, truncate=True, line_kws={"color": "red"});


# **Infant Mortality and Literacy**
# 
# Infant mortality gets lower as literacy increases. High Literacy rate leads to high GDP(as see earlier) which leads to better medical facilities.
# 

# In[ ]:


sns.regplot(x='Infant mortality (per 1000 births)', y='Literacy (%)', data=df, truncate=True, line_kws={"color": "red"});


# **Infant Mortality and Phones**
# 
# As infant mortality increases, number of people using phones decreases. 

# In[ ]:


sns.regplot(x='Infant mortality (per 1000 births)', y='Phones (per 1000)', data=df, logx=True, truncate=True, line_kws={"color": "red"});


# **Infant Mortality and Birthrate**
# 
# High birthrate corresponds to high infant mortality.

# In[ ]:


sns.regplot(x='Infant mortality (per 1000 births)', y='Birthrate', data=df, truncate=True, line_kws={"color": "red"});


# **Infant Mortality and Agriculture**
# 
# Surprisingly Infant Mortality is co-related with Agriculture. It is hard to derive intutive sense from this.

# In[ ]:


sns.regplot(x='Infant mortality (per 1000 births)', y='Agriculture', data=df,  truncate=True, line_kws={"color": "red"});


# **Literacy and Birthrate**
# 
# Increase in literacy % decreases the birthrate of country.

# In[ ]:


sns.regplot(x='Literacy (%)', y='Birthrate', data=df, truncate=True, line_kws={"color": "red"});


# **Literacy and Agriculture**
# 
# As increase in literacy % of country, overall agriculture decrases.

# In[ ]:


sns.regplot(x='Literacy (%)', y='Agriculture', data=df, truncate=True, line_kws={"color": "red"});


# **Coastline and Net Migration**
# 
# No plausiable co-relation between coastline and net migration

# In[ ]:


sns.regplot(x='Coastline (coast/area ratio)', y='Net migration', data=df, truncate=True, line_kws={"color": "red"});


# **Agriculture and Birthrate**
# 
# Increase in agriculture increases the birthrate of country.

# In[ ]:


sns.regplot(x='Agriculture', y='Birthrate', data=df, truncate=True, line_kws={"color": "red"});


# If anyone knows how **Industry** and **Service** columns are arranged and what values they represent please do let me know in comments. I can see there are some co-relations of Industry and Service with other features.
# 
# Thanks!!

# In[ ]:




