#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df=pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
df.head()


# I will try to explain how to extract information or make a report using various trend in dataset using visualization
# So let's jump into it

# In[ ]:


df.describe()


# In[ ]:


df.info()


# From the Info method we can see that their is no Null data excluding salary columns those does not have a chance to be placed :)

# First we will create a Groupby for our reference
# 

# This is a table with mean values.

# In[ ]:


df_group=df.groupby(["gender","status","specialisation","hsc_s"])[["hsc_p","degree_p","etest_p","salary"]].mean()
df_group


# Some basic Masking

# In[ ]:


df_p=df[df["status"]=="Placed"]
mask_m=df["gender"]=="M"
mask_f=df["gender"]=="F"
df_np=df[df["status"]=="Not Placed"]
df_p=df[df["status"]=="Placed"]
df_p["gender"].value_counts()


# In[ ]:


plt.figure(figsize=(10,5))
plt.pie([100,48], explode=(0,0.08), labels=["MALE","FEMALE"], autopct='%1.2f%%',
        shadow=True, startangle=100,colors=["yellow","cyan"])
plt.title("MALE VS FEMALE GOT PLACED")


# Depending upon specialization Above plot

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x="gender",data=df_p,hue="specialisation")
plt.title("MALE VS FEMALE GOT PLACED ACCORDING TO SPECILISATION")


# From the above plot we can understand that** MALE students with Mkt&Fin** have greater percentage in the context of placement.

# In[ ]:


sns.countplot(x="gender",data=df_np,hue="specialisation")
plt.title("MALE VS FEMALE NOT PLACED")


# Now some visualisation on Higher Secondary specialization

# In[ ]:


sns.countplot(x="gender",data=df_p,hue="hsc_s")
plt.title("MALE VS FEMALE GOT PLACED")


# From the graph it is clear that
# 
# **For Male:**
# 
# Commerce background students have more chances obviously as it is a Marketing base job.
# 
# **For Female:**
# 
# Science and commerce both have equal percentage for getting placed.
# 

# In[ ]:


sns.countplot(x="gender",data=df_p,hue="degree_t",color="red")
plt.title("MALE VS FEMALE GOT PLACED BASED ON BACHELOR DEGREE")


# Now the SALARY
# 
# We are going to do various comparison on different level 
# lets see

# In[ ]:


import plotly.express as px
grs = df.groupby(["gender"])[["salary"]].mean().reset_index()
fig = px.bar(grs[['gender', 'salary']].sort_values('salary', ascending=False), 
             y="salary", x="gender", color='gender', 
             log_y=True, template='ggplot2')
fig.show()


# Above Plot is interactive you can play with it changing various parameter.

# Males are making way much money than Females :)

# In[ ]:


grgs = df.groupby(["gender","specialisation"])[["salary"]].mean().reset_index()
fig = px.bar(grgs, x="gender", y="salary", color='specialisation', barmode='stack',
             height=400)
fig.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(df['salary'], bins=50, hist=False)
plt.title("SALARY DISTRIBUTION")


# In[ ]:


plt.figure(figsize=(20,7))
sns.violinplot(x=df_p["gender"],y=df_p["salary"],hue=df_p["specialisation"],palette="Set2")


# Almost for all the cases 2lakh is mean

# In[ ]:


plt.figure(figsize=(20,5))
df_pm=df_p[mask_m]
df_pf=df_p[mask_f]
mask_g=["M","F"]
mask_spe=["Mkt&HR","Mkt&Fin"]
#f, axes = plt.subplots(1, 3, figsize=(18,5), sharex=True)
for j in range(len(mask_spe)):
    df_p_=df_p[df_p["specialisation"]==mask_spe[j]]
    for i in mask_g:
        df_p__=df_p_[df_p_["gender"]==i]
        sns.distplot(df_p__["salary"],hist=False,kde_kws = {'shade': True, 'linewidth': 3},label=(mask_spe[j],i))


# Tree Plot
# 
# Making Interactive plots
# 

# In[ ]:


gp = df.groupby(["gender","specialisation"])[["salary"]].mean().reset_index()

fig = px.treemap(gp, path=['gender','specialisation'], values='salary',
                  color='salary', hover_data=['specialisation'],
                  color_continuous_scale='rainbow')
fig.show()


# Another Tree plot.
# 
# 
# Look closely how very easily with groupby we can create this kind of amazing interactive plots.
# 
# Plotting tree Chart of high secondary stream and percentage is a feature.

# In[ ]:


grss = df.groupby(["hsc_b","hsc_s"])[["hsc_p"]].mean().reset_index()

fig = px.treemap(grss, path=['hsc_b','hsc_s'], values='hsc_p',
                  color='hsc_p', hover_data=['hsc_s'],
                  color_continuous_scale='rainbow')
fig.show()


# In[ ]:


grdsp = df.groupby(["degree_t"])[["degree_p"]].mean().reset_index()

fig = px.pie(grdsp,
             values="degree_p",
             names="degree_t",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# In[ ]:


import plotly.express as px
fig = px.scatter_ternary(df, a="ssc_p", b="hsc_p",c="degree_p",color = "status")
fig.show()


# Now it is time to explore other scores deeply

# In[ ]:


sns.violinplot(x="status",y="degree_p",data=df,hue="gender")


# In[ ]:


sns.violinplot(x="status", y="etest_p", hue="gender", data=df)


# In[ ]:


df.tail()


# In[ ]:



sns.catplot(y="degree_p",x="degree_t",col="hsc_s",data=df)


# Above one is very interesting look the graph carefully
# 
# Different background student how choose their degree course and scored well.

# I Will Continue With More Like This 
# # Thank You
