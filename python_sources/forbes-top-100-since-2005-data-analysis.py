#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# # Forbes Top 100 since 2005 Data Analysis

# 1. [Load and Check Data](#1)
# 1. [Variable Description](#2)
# 1. [Univariate Variable Analysis](#3)
#     * [Categorical Variable Analysis](#4)
#     * [Numerical Variable Analysis](#5)
# 1. [Basic Data Analysis](#6)
# 1. [Visualization](#7)
#     * [Correlations between Pay and Year](#8)
#     * [Category - Pay](#9)
#     * [Year - Pay](#10)
# 1. [Feature Engineering](#11)
#     * [First name extraction from Name feature](#12)

# ## Load and Check Data <a id="1"></a>

# In[ ]:


df = pd.read_csv("/kaggle/input/forbes-celebrity-100-since-2005/forbes_celebrity_100.csv")


# In[ ]:


df.columns


# In[ ]:


df.head(10)


# In[ ]:


df.describe()


# * There are 1547 Pay features and 1547 Year features, and that means we have 1547 people.
# * Minimum Pay feature in our data is 1.5 USD Millions, and the maximum is 620 USD Millions.
# * Minimum Year feature in our data is 2005, while maximum is 2019; means that we have started getting a record in 2005, and the last record was in 2019.

# ## Variable Description <a id="2"></a>
# * Name : This feature tells us the person's full name.
# * Pay(USD millions) : This feature tells us the person's earn per year.
# * Year : This feature tells us the year our person showed up in Forbes top 100.
# * Category : This feature tells us the job or the field of our person.

# In[ ]:


df.info()


# * float64(1) : Pay (USD Millions)
# * object(2) : Name, Category
# * int64(1) : Year

# ## Univariate Variable Analysis <a id="3"></a>
# * Categorical Variable : Variables with over 2 categories. (Name,Category)
# * Numerical Variable : Variables with numeric values. (Pay,Year)

# ### Categorical Variable Analysis <a id="4"></a>

# In[ ]:


def barplotall(variable):
    var = df[variable] ## get each value
    varcount = var.value_counts() ## count each value
    
    ## visualization
    plt.figure(figsize=(9,3))
    plt.bar(varcount.index,varcount)
    plt.xticks(varcount.index,varcount.index.values)
    plt.ylabel("Count")
    plt.title(variable)
    plt.show()
    print(f"Count of {variable} variable : \n{varcount}")


# In[ ]:


categorical_features =  ["Name","Category"]
for i in categorical_features:
    barplotall(i)


# * It's too hard to understand the barplot beacuse things are too mixed.
# * But our variable counts will help us.
# * What we understand here is;
#     1. Musicians are majority in our category data.
#     1. Phil Mickelson and Tiger Woods have been in the top 100 list for 16 years.

# ### Numerical Variable Analysis <a id="5"></a>

# In[ ]:


def histoall(variable):
    plt.figure(figsize=(9,3))
    plt.hist(df[variable])
    plt.xlabel(variable)
    plt.ylabel("Count")
    plt.title(f"{variable} Distribution with Histogram")
    plt.show()


# In[ ]:


numerical_features = ["Pay (USD millions)","Year"]
for i in numerical_features:
    histoall(i)


# * From these histograms we can understand that;
#     1. Most of our celebrities earn between 0-100 USD Millions.
#     2. Actually there might not be 100 people in the list, sometimes it can raise and fall.

# ## Basic Data Analysis <a id="6"></a>

# ### Year - Pay

# In[ ]:


df[["Year","Pay (USD millions)"]].groupby("Year",as_index=False).mean().sort_values(by="Pay (USD millions)",ascending=False)


# * As we can see above, the mean of earnings raised year by year.

# ### Category - Pay

# In[ ]:


df[["Category","Pay (USD millions)"]].groupby("Category",as_index=False).mean().sort_values(by="Pay (USD millions)",ascending=False)


# * As we can see above, the directors and producers earn the most money in mean.

# ## Visualization <a id="7"></a>

# ### Correlations between Pay and Year <a id="8"></a>

# In[ ]:


features_list = ["Pay (USD millions)","Year"]
sns.heatmap(df[features_list].corr(),annot=True,fmt=".2f")
plt.show()


# * They have small correlation as 0.20

# ### Category - Pay <a id="9"></a>
# * After this point, I want to turn my category features to numerical values for easier visualization.
# * We will not enumerate all categories.

# In[ ]:


df["Category_nums"] = [1 if i == "Musicians" else 2 if i == "Athletes" else 3 if i == "Personalities" else 4 if i == "Actors" or "Actresses" or "Television actors" or "Television actresses" else 5 for i in df["Category"]]
df["Category_nums"].head()


# In[ ]:


g = sns.factorplot(x="Category_nums",y="Pay (USD millions)",data=df,kind="bar",size=6)
g.set_ylabels("Pay")
plt.show()


# ### Year - Pay <a id="10"></a>

# In[ ]:


g = sns.factorplot(x="Year",y="Pay (USD millions)",data=df,kind="bar",size=6)
g.set_ylabels("Pay")
plt.show()


# ## Feature Engineering <a id="11"></a>

# ### First name extraction from Name feature <a id="12"></a>

# In[ ]:


df["first_names"] = df["Name"]


# * First we will find the non-name values.

# In[ ]:


num_list = [1,2,3,4,5,6,7,8,9,0]
word_list = ["and","the","of","."]
for i in df["first_names"]:
    for w in word_list:
        if str(w) in i:
            df["first_names"].replace(i,np.nan)
    for j in num_list:
        if str(j) in i:
            df["first_names"].replace(i,np.nan)
df["first_names"].dropna(inplace=True)
df["first_names"].head(10)


# * Now we will extract the first names.

# In[ ]:


f_names = []
for i in df["first_names"]:
    f_names.append(i.split()[0])


# * We will append our first names to our data.

# In[ ]:


df["first_names"] = f_names
df["first_names"].head()


# * Lastly, we will visualize them.

# In[ ]:


fig, ax = plt.subplots(figsize=(30,3))
sns.countplot(x="first_names",data=df,ax=ax)
plt.xticks(rotation=90)
plt.show()


# * That's a mess over there :). Because of this, we will count our data.

# In[ ]:


pd.value_counts(df["first_names"])


# * So Jennifer is the most used name.

# # Conclusion
# * We made Forbes Top 100 data understandable in this notebook.
# * Would be nice if you comment your thoughts and upvote.
