#!/usr/bin/env python
# coding: utf-8

# **Data Science in the European Union**
# 
# This kernel focuses on Kaggle survey results for the EU region. Currently it consists 28 countries in which in total over 510 milions of people live. Due to open borders policy this is a huge market for Data Science specialists and it is worth to look inside it more in detail.
# 
# Reading necessary libraries. We will use the following libraries:
# * pandas
# * numpy
# * matplotlib

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
import warnings
from matplotlib_venn import venn3
warnings.filterwarnings('ignore')


# In[ ]:


data = pd.read_csv("../input/multipleChoiceResponses.csv", low_memory=False)


# Creating list of EU contries for filtering data.

# In[ ]:


europe = ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
          "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary",
          "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
          "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
          "Spain", "Sweden", "United Kingdom of Great Britain and Northern Ireland"]


# Let's check what countries are missing.

# In[ ]:


# checking if all EU countries are in database
countries = data["Q3"].drop(0)
countries_unique = countries.value_counts()

cntrs = list(countries_unique.index)

present = []
absent = []
for e in europe:
    if e in cntrs:
        present.append(e)
    else:
        absent.append(e)
print (absent)


# Number of respondents from individual countries.

# In[ ]:


db_EU = data[data["Q3"].isin(europe)]
db_EU["Q3"] = db_EU["Q3"].replace("United Kingdom of Great Britain and Northern Ireland","UK")
Q3 = db_EU["Q3"].value_counts()

plt.figure(figsize=(20,10))
plt.bar(Q3.index,Q3)
plt.xticks(Q3.index,rotation="vertical")
plt.margins(0,0.1)
plt.subplots_adjust(bottom=0.25)
plt.title("Number of respondents from EU contries")
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.rc('axes', axisbelow=True)


# In[ ]:


ages =['22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-69','70-79','80+']

Q2 = db_EU["Q2"].value_counts()
Q2 = Q2.ix[ages]
plt.figure(figsize=(20,10))
plt.bar(Q2.index,Q2, )
plt.xticks(Q2.index,rotation="vertical")
plt.margins(0,0.1)
plt.subplots_adjust(bottom=0.25)
plt.title("Age group of respondents from EU contries")
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.rc('axes', axisbelow=True)


# In[ ]:


salaries = ['Not disclosed','0-10,000','10-20,000','20-30,000','30-40,000','40-50,000','50-60,000','60-70,000','70-80,000','80-90,000','90-100,000','100-125,000','125-150,000','150-200,000','200-250,000','250-300,000','300-400,000','400-500,000','500,000+']

db_EU.loc[:,"Q9"] = db_EU["Q9"].replace("I do not wish to disclose my approximate yearly compensation","Not disclosed")
Q9 = db_EU["Q9"].value_counts()
Q9 = Q9.reindex(salaries)

plt.figure(figsize=(20,10))
ax =plt.bar(Q9.index,Q9)
plt.xticks(Q9.index,rotation="vertical")
plt.margins(0,0.1)
plt.subplots_adjust(bottom=0.25)
plt.title("Salaries ranges of respondents from EU contries [$/annum]")
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.rc('axes', axisbelow=True)
ax[0].set_color('#990000')
#alternative method:
#ax.patches[Q9.index.get_indexer(['Not disclosed'])[0]].set_facecolor('#990000')


# **Analysis of Not Disclosed salaries**
# 
# Distribution of respondents who do not want disclose their salaries by country. Fisrt step - absolute values.

# In[ ]:


not_disclosed = db_EU[db_EU["Q9"]=="Not disclosed"]
nd_countries = not_disclosed["Q3"].value_counts().rename("nd_counts", axis='columns')

plt.figure(figsize=(20,10))
ax = plt.bar(nd_countries.index,nd_countries)

plt.xticks(nd_countries.index,rotation="vertical")
plt.margins(0,0.1)
plt.subplots_adjust(bottom=0.25)
plt.title("Number of respondents who do not want disclose salary \nby country (absolute values)")
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.rc('axes', axisbelow=True)
ax[0].set_color('#f7b45d')
ax[1].set_color('#990000')
ax[2].set_color('#37f250')


# In terms of absolute values three top countries are:
# * Germany
# * UK
# * France
# 
# However absolute values do not tell us the entire story.
# 
# Next step - percentage of respondents from each country who do not want disclose their salary range.

# In[ ]:


df_ndc = nd_countries.to_frame()
df_Q3 = Q3.to_frame()

Q3_nd = df_Q3.merge(df_ndc, how="outer", left_index=True, right_index=True)
Q3_nd["percentage"] = Q3_nd["nd_counts"]/Q3_nd["Q3"]*100
Q3_nd.sort_values("percentage", inplace=True, ascending=False)

plt.figure(figsize=(20,10))
ax = plt.bar(Q3_nd.index,Q3_nd["percentage"])

plt.xticks(Q3_nd.index,rotation="vertical")
plt.margins(0,0.1)
plt.subplots_adjust(bottom=0.25)
plt.title("Number of respondents who do not want disclose salary \nby country (percentage)")
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.rc('axes', axisbelow=True)
ax[4].set_color('#f7b45d')
ax[6].set_color('#990000')
ax[7].set_color('#37f250')


# When looking at the percentage of respondents not disclosing salaries the TOP3 countries from the previous graph are now not on top of the list anymore.
# 
# Three the most common languages in data science are Python, R, SQL. Let's see how many respondents use these languages and how many of them use two or three of them "at once". For this the Venn diagram will be used.

# In[ ]:


langs = db_EU[["Q16_Part_1","Q16_Part_2","Q16_Part_3"]]
pythons = langs[langs["Q16_Part_1"]=="Python"].index.values.tolist()
rs = langs[langs["Q16_Part_2"]=="R"].index.values.tolist()
sqls = langs[langs["Q16_Part_3"]=="SQL"].index.values.tolist()

plt.figure(figsize=(20,10))
plt.title("Venn diagram of languages used by respondents")
venn3([set(pythons),set(rs),set(sqls)],("Python","R","SQL"))


# The venn diagram shows that the biggest number of respondents uses Python. SQL is utilized by relatively big fraction of both Python and R users. There is also a big group of respondents writing with both R and Python.

# UNDER CONSTRUCTION
