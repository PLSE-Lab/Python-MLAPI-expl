#!/usr/bin/env python
# coding: utf-8

# # Exploring World Economic Freedom Data
# ### Background
# According to https://www.heritage.org/index/about ,
# Economic freedom is based on 12 quantitative and qualitative factors, grouped into four broad categories, or pillars, of economic freedom:
# 
# * Rule of Law (property rights, government integrity, judicial effectiveness)
# * Government Size (government spending, tax burden, fiscal health)
# * Regulatory Efficiency (business freedom, labor freedom, monetary freedom)
# * Open Markets (trade freedom, investment freedom, financial freedom)
# 
# ### There are two objectives for this notebook
# ### 1. Explore the data
# ### 2. Asses the impact of economic freedom on a country's wealth in GDP per capita
# 

# In[ ]:


import pandas as pd
import numpy as np
df = pd.read_csv("../input/the-economic-freedom-index/economic_freedom_index2019_data.csv",encoding = "ISO-8859-1")


# ## 1. Explore the data

# In[ ]:


df.info()


# In[ ]:


def world_compare(chosen_country):
    country = df.loc[df['Country'] == chosen_country]
    print("  World Economic Freedom values for *", chosen_country,"* compared to worldwide averages.\n")
    for col in country:
        country_val = country.iloc[0][col]
        try:
            world_mean = round(df[col].mean(),2)
        except:
            world_mean = "-"
        print("  ",col, " "*(30-len(col)),country_val," "*(30-len(str(country_val))),world_mean)


# In[ ]:


world_compare("Ireland")


# In[ ]:


world_compare("United States")


# In[ ]:


numeric_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
non_numeric_cols = [col for col in df.columns if col not in numeric_cols]


# In[ ]:


print("  Numeric Columns\n  -------------------")
for x in numeric_cols:
    print(" ",x)


# In[ ]:


print("  Non Numeric Columns\n  -------------------")
for x in non_numeric_cols:
    print(" ",x)


# ## 2. Explore the effect of economic freedoms on GDP per capita

# In[ ]:


print("  Converting GDP per Capita (PPP) column to numeric.\n  First i will remove the '$' signs and commas.")
print("  Then convert the string column to the integer datatype.\n")
print("Before:")
print(df['GDP per Capita (PPP)'].values)
df['GDP per Capita (PPP)'] = df['GDP per Capita (PPP)'].str.replace('$', '')
df['GDP per Capita (PPP)'] = df['GDP per Capita (PPP)'].str.replace(',', '')
print("After:")
print(df['GDP per Capita (PPP)'].values)


# In[ ]:


print("  The extraneous info.such as '1700 (2015 est.)' will also be removed.\n")
df['GDP per Capita (PPP)'] = df['GDP per Capita (PPP)'].str.split(' ').str[0]
df['GDP per Capita (PPP)'] = df['GDP per Capita (PPP)'].astype(float)
print(df['GDP per Capita (PPP)'].values)


# In[ ]:


twelve_measures = ["Property Rights","Judical Effectiveness","Government Integrity","Tax Burden",
  "Gov't Spending","Fiscal Health","Business Freedom","Labor Freedom","Monetary Freedom",
  "Trade Freedom","Investment Freedom ","Financial Freedom"]
print("  Correlation of each of the twelve factors with World Economic Freedom Ranking\n")
for col in twelve_measures:
    print("  ",col," "*(30-len(col)),round(df[col].corr(df["World Rank"]),3))


# In[ ]:


print("  In general the lower(better) a country's Economic Freedom Ranking the higher its GDP Per Capita.\n")
print("  Correlation =",round(df["World Rank"].corr(df["GDP per Capita (PPP)"]),3))


# In[ ]:


twelve_measures = ["Property Rights","Judical Effectiveness","Government Integrity","Tax Burden",
  "Gov't Spending","Fiscal Health","Business Freedom","Labor Freedom","Monetary Freedom",
  "Trade Freedom","Investment Freedom ","Financial Freedom"]
print("  Correlation of each of the twelve factors with GDP Per Capita\n")
for col in twelve_measures:
    print("  ",col," "*(30-len(col)),round(df[col].corr(df["GDP per Capita (PPP)"]),3))


# In[ ]:




