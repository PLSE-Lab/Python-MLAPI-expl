#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/googleplaystore.csv")
df.head(2)


# # Get the count of app for various Generes

# In[ ]:


plt.figure(figsize=(16,4))
df.Genres.value_counts()[:50].plot(kind="bar")
plt.show()


# # Get the top rated apps count

# ### find min and max rated value

# In[ ]:


df.Rating.max()


# In[ ]:


df.Rating.min()


# #### Find unique values in rating

# In[ ]:


df.Rating.unique()


# In[ ]:


df1 =df[df.Rating.notnull()][["Rating","Type"]]
plt.figure(figsize=(16,4))
sns.boxplot(df1.Type,df1.Rating,)


# In[ ]:


# ABove figure tell that we have one value which is completely outlier for this data (i.e Rating 19). Lets remove them and plot again
df1 =df[(df.Rating.notnull()) & (df.Rating <19)][["Rating","Type"]]
plt.figure(figsize=(16,4))
sns.boxplot(df1.Type,df1.Rating)


# ### Category counts

# In[ ]:


plt.figure(figsize=(16,4))
df.Category.value_counts().plot(kind="bar")
plt.show()


# ### Lets find out the ratings of Top category Apps

# In[ ]:


df.head(1)


# In[ ]:


df.Category.value_counts()[:5]


# #### We will filter out Ratings values for those top rated (first 5 Category) Apps

# In[ ]:


Family_4_star = df[(df.Category == "FAMILY") & (df.Rating >= 3)].Category.count()
GAME_4_star = df[(df.Category == "GAME") & (df.Rating >= 3)].Category.count()
TOOLS_4_star = df[(df.Category == "TOOLS") & (df.Rating >= 3)].Category.count()
MEDICAL_4_star = df[(df.Category == "MEDICAL") & (df.Rating >= 3)].Category.count()
BUSINESS_4_star = df[(df.Category == "BUSINESS") & (df.Rating >= 3)].Category.count()
plt.figure(figsize=(16,4))
plt.bar("Family_4_star",Family_4_star)
plt.bar("GAME_4_star",GAME_4_star)
plt.bar("TOOLS_4_star",TOOLS_4_star)
plt.bar("MEDICAL_4_star",MEDICAL_4_star)
plt.bar("BUSINESS_4_star",BUSINESS_4_star)
plt.show()


# ### Analyze Reviews of app (find the highly reviewed app)
# This column data type is Object hence we need to convert to Integer so that we can do further processing

# In[ ]:


df.Reviews.dtype


# In[ ]:


# pd.to_numeric(df.Reviews) 

# After analyzing thsi data, we see that their are some numbers 
# which has "M" for million, so it is not feasible to convert all of them to numeric. 
# !!!Someone can try here to add extra logic !!!


# ### Lets find out total Paid and free apps

# #### Find unique (categorical values and latyer their counts)

# In[ ]:


df.Type.unique()


# In[ ]:


df.Type.value_counts()


# In[ ]:


plt.figure(figsize=(16,4))
df.Type.value_counts().plot(kind="bar")


# In[ ]:


df.head(1)


# ## Lets find out the price variations based on ratings of Apps (considering only 3 and above rated Apps)

# In[ ]:


rating_price_var = df[df.Rating >=3]


# In[ ]:


rating_price_var.Price.unique()


# ### Above uniq values tell that we have some strings present in price columns, Since some apps price  shows it is for everyone then we will replace "Everyone" -> 0

# In[ ]:


rating_price_var.Price.replace({"Everyone":0},inplace=True)
# Ignore the warning


# In[ ]:


rating_price_var.Price.unique()
# Now we can see "Everyone" is not present


# ### We have to replace all "$" symbols and convert all teh string values to Integer type to proceed further

# In[ ]:


rating_price_var.Price.replace({"\$":""},regex=True,inplace=True)
# Now we can see "Everyone" is not present


# In[ ]:


rating_price_var.Price.unique()
# Now no $ sign present


# ### Now, convert all to integet type

# In[ ]:


rating_price_var.Price = pd.to_numeric(rating_price_var.Price)
# Now no $ sign present


# ### Check the type of data, it is not float64 (as expected)

# In[ ]:


rating_price_var.Price.dtype


# In[ ]:


plt.figure(figsize=(16,4))
rating_price_var.Rating.sort_values().value_counts().plot(kind="bar")
plt.show()


# In[ ]:


# plt.figure(figsize=(16,4))
rate_price = rating_price_var[["Rating","Price"]].sort_values(by="Rating",ascending=False)
rate_price.head(2)
# plt.show()


# In[ ]:


plt.figure(figsize=(16,4))
plt.scatter(rate_price.Rating[1:],rate_price.Price[1:],alpha = .2,color="g")
plt.xticks(np.linspace(3,5,len(rate_price.Rating[1:]))[::400])
plt.show()


# ### This plot will highlight the apps which are highly priced. It show their rating as well

# In[ ]:


plt.figure(figsize=(16,4))
rating = rate_price.Rating[1:].tolist()
price = rate_price.Price[1:].tolist()
for i in range(rate_price.Rating[1:].size):
    plt.scatter(rating[i],price[i],alpha = .2,color="g")
    if price[i] >=250:
        plt.scatter(rating[i],price[i],alpha = .2,s=.012*price[i]**2,color="m",marker="*")
        plt.text(rating[i],price[i]+3,(rating[i],price[i]))
# plt.xticks(np.linspace(3,5,len(rate_price.Rating[1:]))[::400])
plt.show()

