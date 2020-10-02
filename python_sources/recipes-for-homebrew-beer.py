#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sn

from keras.models import Sequential

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


recipe = pd.read_csv("../input/recipeData.csv",encoding="latin1")
print(recipe.head(5))


# In[6]:


print(recipe.info())


# In[5]:


print(recipe.describe())


# In[4]:


style = pd.read_csv("../input/styleData.csv",encoding="latin1")
print(style.head(5))


# In[3]:


missing = recipe.copy()
missing = missing.T
missed = missing.isnull().sum(axis=1)
missing["valid_count"] = (len(missing.columns)-missed) / len(missing.columns)
missing["na_count"] = missed / len(missing.columns)

missing[["na_count","valid_count"]].sort_values("na_count", ascending=True).plot.barh(stacked=True,figsize=(12,10),color=["c","y"])
plt.title("Beer recipe missing data")


# In[7]:


plt.figure(figsize=(12,8))
recipe.Style.value_counts().nlargest(30).sort_values(ascending=False).plot.bar()
plt.title("Recipe based on style")
plt.xlabel("Style")
plt.ylabel("Number of recipes")


# In[8]:


recipe.BrewMethod.value_counts().plot(kind="pie",autopct="%1.1f%%")
plt.title("Recipe based on brew method")


# In[9]:


recipe.SugarScale.value_counts().plot(kind="pie",autopct="%1.1f%%")
plt.title("Recipe based on sugar scale")


# In[10]:


sn.distplot(recipe["Size(L)"],bins=20)


# In[11]:


sn.distplot(recipe["Efficiency"],bins=20)


# In[12]:


sn.distplot(recipe["Color"],bins=20)


# In[21]:


recipe["PrimaryTemp"] = recipe.groupby("StyleID").transform(lambda x: x.fillna(x.mean()))
sn.distplot(recipe["PrimaryTemp"],bins=20)


# In[15]:


recipe["MashThickness"] = recipe.groupby("StyleID").transform(lambda x: x.fillna(x.mean()))
sn.distplot(recipe["MashThickness"],bins=20)


# In[17]:


sn.distplot(recipe["BoilSize"],bins=20)


# In[18]:


sn.distplot(recipe["BoilTime"],bins=20)


# In[19]:


recipe["BoilGravity"] = recipe.groupby("StyleID").transform(lambda x: x.fillna(x.mean()))
sn.distplot(recipe["BoilGravity"],bins=20)


# In[22]:


plt.figure(figsize=(12,8))
sn.pairplot(recipe,vars=["BoilSize","BoilTime","BoilGravity"],hue="BrewMethod")


# In[78]:


brew_grp = recipe.groupby("BrewMethod")["BoilSize","BoilTime","BoilGravity"].mean().sort_values(ascending=False,by="BoilSize")
print(brew_grp)


# In[23]:


boil_grp = recipe.groupby("Style")["BoilSize","BoilTime","BoilGravity"].mean().sort_values(ascending=False,by="BoilSize")[:30]
plt.figure(figsize=(12,8))
sn.pairplot(boil_grp,vars=["BoilSize","BoilTime","BoilGravity"])


# In[24]:


fig, ax = plt.subplots(figsize=(12,8))
plt.subplot(221)
sn.kdeplot(boil_grp["BoilSize"],boil_grp["BoilTime"])
plt.subplot(222)
sn.kdeplot(boil_grp["BoilGravity"],boil_grp["BoilTime"])
plt.suptitle("BoilSize and BoilGravity vs BoilTime")


# In[72]:


fig, ax = plt.subplots(figsize=(14,13))
plt.subplot(231)
sn.distplot(boil_grp["BoilTime"])
plt.ylabel("Average time")
plt.title("Histogram")
plt.subplot(233)
boil_grp["BoilTime"].sort_values(ascending=True).plot.barh(stacked=True)
plt.xlabel("BoilTime")
plt.title("Highest style")
plt.suptitle("BoilTime based on style")


# In[74]:


fig, ax = plt.subplots(figsize=(14,13))
plt.subplot(231)
sn.distplot(boil_grp["BoilGravity"])
plt.ylabel("Average gravity")
plt.title("Histogram")
plt.subplot(233)
boil_grp["BoilGravity"].sort_values(ascending=True).plot.barh(stacked=True)
plt.xlabel("BoilGravity")
plt.title("Highest style")
plt.suptitle("BoilGravity based on style")


# In[75]:


fig, ax = plt.subplots(figsize=(14,13))
plt.subplot(231)
sn.distplot(boil_grp["BoilSize"])
plt.ylabel("Average boiling size")
plt.title("Histogram")
plt.subplot(233)
boil_grp["BoilSize"].sort_values(ascending=True).plot.barh(stacked=True)
plt.xlabel("BoilSize")
plt.title("Highest style")
plt.suptitle("BoilSize based on style")


# In[76]:


sn.distplot(recipe["OG"],bins=10)


# In[77]:


sn.distplot(recipe["FG"],bins=10)


# In[30]:


sn.distplot(recipe["ABV"],bins=20)


# In[31]:


sn.distplot(recipe["IBU"],bins=20)


# In[85]:


eff_grp = recipe.groupby("Efficiency")["OG","FG","ABV","IBU","Color"].mean().sort_values(ascending=False,by="OG")
sn.pairplot(eff_grp)


# In[71]:


temp_grp = recipe.groupby("PrimaryTemp")["OG","FG","ABV","IBU","Color"].mean().sort_values(ascending=False,by="OG")
sn.pairplot(temp_grp)


# In[34]:


style_grp = recipe.groupby("Style")["OG","FG","ABV","IBU","Color"].mean().sort_values(ascending=False,by="OG")[:30]
print(style_grp.index)


# In[35]:


sn.pairplot(style_grp,vars=["OG","FG","ABV","IBU","Color"])


# In[54]:


fig, ax = plt.subplots(figsize=(14,13))
plt.subplot(231)
sn.distplot(style_grp["FG"])
plt.ylabel("Average batch")
plt.title("Histogram")
plt.subplot(233)
style_grp["FG"].sort_values(ascending=True).plot.barh(stacked=True)
plt.title("Highest style")
plt.xlabel("FG")
plt.suptitle("FG based on Style")


# In[55]:


fig, ax = plt.subplots(figsize=(14,13))
plt.subplot(221)
sn.distplot(style_grp["OG"])
plt.ylabel("Average batch")
plt.title("Histogram")
plt.subplot(222)
style_grp["OG"].sort_values(ascending=True).plot.barh(stacked=True)
plt.title("Highest style")
plt.xlabel("OG")
plt.suptitle("OG based on style")


# In[59]:


fig, ax = plt.subplots(figsize=(14,13))
plt.subplot(231)
sn.distplot(style_grp["ABV"])
plt.ylabel("Average batch")
plt.title("Histogram")
plt.subplot(233)
style_grp["ABV"].sort_values(ascending=True).plot.barh(stacked=True)
plt.xlabel("ABV")
plt.title("Highest style")
plt.suptitle("ABV based on style")


# In[61]:


fig, ax = plt.subplots(figsize=(14,13))
plt.subplot(231)
sn.distplot(style_grp["IBU"])
plt.ylabel("Average batch")
plt.subplot(233)
style_grp["IBU"].sort_values(ascending=True).plot.barh(stacked=True)
plt.xlabel("IBU")
plt.title("Highest style")
plt.suptitle("IBU based on style")


# In[63]:


fig, ax = plt.subplots(figsize=(14,13))
plt.subplot(231)
sn.distplot(style_grp["Color"])
plt.ylabel("Average batch")
plt.title("Histogram")
plt.subplot(233)
style_grp["Color"].sort_values(ascending=True).plot.barh(stacked=True)
plt.xlabel("Color")
plt.title("Highest style")
plt.suptitle("Color based on style")


# In[60]:


sn.kdeplot(style_grp["OG"],style_grp["FG"])
plt.title("Distribution between OG and FG")


# In[62]:


fig, ax = plt.subplots(figsize=(12,8))
plt.subplot(221)
sn.kdeplot(style_grp["OG"],style_grp["ABV"])
plt.subplot(222)
sn.kdeplot(style_grp["FG"],style_grp["ABV"])
plt.suptitle("Distribution of OG and FG")


# In[44]:


sn.kdeplot(style_grp["ABV"],style_grp["IBU"])
plt.title("Distribution of ABV and IBU")


# In[45]:


sn.kdeplot(style_grp["IBU"],style_grp["Color"])
plt.title("Distribution of IBU and Color")


# In[40]:


fig, ax = plt.subplots(figsize=(12,8))
plt.subplot(221)
sn.kdeplot(style_grp["OG"],style_grp["IBU"])
plt.subplot(222)
sn.kdeplot(style_grp["FG"],style_grp["IBU"])
plt.suptitle("OG and FG vs IBU")


# In[38]:


ind = np.arange(30)
width = 0.35
fig, ax = plt.subplots(figsize=(12,10))
ax.bar(ind-width/2, style_grp["OG"], width=width, color="SkyBlue", label="Original Gravity")
ax.bar(ind+width/2, style_grp["FG"], width=width, color="IndianRed", label="Final Gravity")
ax.set_xticks(ind)
ax.set_xticklabels(style_grp.index,rotation=90)
ax.set_xlabel("Style")
ax.set_ylabel("Gravity")
ax.legend(["Original gravity","Final gravity"])
ax.set_title("Gravity group by style")


# In[39]:


ind = np.arange(30)
width = 0.5
plt.figure(figsize=(12,10))

p1 = plt.bar(ind, style_grp["FG"], width, color="Teal")
p2 = plt.bar(ind, style_grp["OG"], width, color="Brown", bottom=style_grp["FG"])
p3 = plt.bar(ind, style_grp["ABV"], width, color="LimeGreen", bottom=style_grp["OG"])
p4 = plt.bar(ind, style_grp["IBU"], width, color="Tomato", bottom=style_grp["ABV"])

plt.xticks(ind,style_grp.index,rotation=90)
plt.xlabel("Style")
plt.ylabel("Scale")
plt.legend((p1[0],p2[0],p3[0],p4[0]),("FG","OG","ABV","IBU"))
plt.title("Recipe scaling based on style")


# In[84]:


plt.figure(figsize=(15,12))
sn.pairplot(recipe,vars=["OG","FG","ABV","IBU","Color"],hue="BrewMethod")


# In[94]:


grain = recipe[recipe["BrewMethod"]=="All Grain"]
ag = grain.groupby(["Style","IBU"])["ABV"].mean().unstack()
#ag = ag.sort_values([79.0],ascending=False)
plt.figure(figsize=(12,8))
sn.heatmap(ag,cmap="Reds")


# In[95]:


grain = recipe[recipe["BrewMethod"]=="All Grain"]
ag = grain.groupby(["Style","OG"])["FG"].mean().unstack()
#ag = ag.sort_values([79.0],ascending=False)
plt.figure(figsize=(12,8))
sn.heatmap(ag,cmap="Blues")

