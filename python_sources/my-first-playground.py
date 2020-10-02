#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
pd


# In[2]:


df_2015 = pd.read_csv("../input/2015.csv")
df_2016 = pd.read_csv("../input/2016.csv")


# In[3]:


df_2015.describe()


# In[4]:


df_2016.describe()


# In[5]:


df_2015.head()


# In[6]:


df_2016.head()


# In[7]:


df_2015["Region"].unique()


# Region by region analysis (2015)

# In[8]:


df_2015.groupby("Region")["Happiness Rank", "Happiness Score", "Standard Error"].mean().sort_values(by="Happiness Score", ascending=False)


# Region by region analysis (2016)

# In[9]:


df_2016["Standard Error"] = df_2016["Happiness Score"] - df_2016["Lower Confidence Interval"]
df_2016.groupby("Region")["Happiness Rank", "Happiness Score", "Standard Error"].mean().sort_values(by="Happiness Score", ascending=False)


# Country by country analysis in Europe (2015)

# In[10]:


we = df_2015[df_2015["Region"] == "Western Europe"]
cee = df_2015[df_2015["Region"] == "Central and Eastern Europe"]
eu_2015 = pd.concat([we,cee])
eu_2015[["Country", "Happiness Score"]].sort_values(by="Happiness Score", ascending=False)


# Univariate Analysis

# In[11]:


# Let us just impute the missing values with mean values to compute correlation coefficients #
mean_values = df_2015.mean(axis=0)
df_2015_new = df_2015.fillna(mean_values, inplace=True)

# Now let us look at the correlation coefficient of each of these variables #
x_cols = [col for col in df_2015_new.columns if col not in ['Happiness Score', 'Standard Error'] if df_2015_new[col].dtype=='float64']

labels = []
values = []
for col in x_cols:
    labels.append(col)
    values.append(np.corrcoef(df_2015_new[col].values, df_2015_new['Happiness Score'].values)[0,1])
corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
    
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,4))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
#autolabel(rects)
plt.show()


# Correlation among variables

# In[12]:


cols_to_use = corr_df.col_labels.tolist()

temp_df = df_2015[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(8, 8))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=1., square=True)
plt.title("Important variables correlation map", fontsize=15)
plt.show()


# The strongest correlation can be found for the variable 'Economy'

# In[18]:


col = 'Economy (GDP per Capita)'
ulimit = np.percentile(df_2015[col].values, 99.5)
llimit = np.percentile(df_2015[col].values, 0.5)
df_2015[col].ix[df_2015[col] > ulimit] = ulimit
df_2015[col].ix[df_2015[col] < llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=df_2015[col].values, y=df_2015['Happiness Score'].values, size=10, color=color[4])
plt.ylabel('Happiness Score', fontsize=12)
plt.xlabel(col, fontsize=12)
plt.title(col + " Vs Happiness", fontsize=15)
plt.show()


# In[ ]:


The weakest correlation instead is scored by the variable 'Generosity'


# In[21]:


col = 'Generosity'
ulimit = np.percentile(df_2015[col].values, 99.5)
llimit = np.percentile(df_2015[col].values, 0.5)
df_2015[col].ix[df_2015[col] > ulimit] = ulimit
df_2015[col].ix[df_2015[col] < llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=df_2015[col].values, y=df_2015['Happiness Score'].values, size=10, color=color[4])
plt.ylabel('Happiness Score', fontsize=12)
plt.xlabel(col, fontsize=12)
plt.title(col + " Vs Happiness", fontsize=15)
plt.show()

