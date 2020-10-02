#!/usr/bin/env python
# coding: utf-8

# **Men vs Women**
# 
# Myth: Women shop more than men.
# 
# Fact: Men shop more than women.
# 
# *Let's begin to see how.*

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/BlackFriday.csv")
df.head()


# **As expected, women are winning in the first 5 entries**

# In[ ]:


df.info()


# **Men vs Women: Who should be labelled shopaholic?**

# In[ ]:


columns = ['User_ID','Gender']
df_subset = pd.DataFrame(df, columns=columns)
type(df_subset)


# In[ ]:


df_unique_buyers = df_subset.drop_duplicates(['User_ID','Gender'])
df_unique_buyers[df_unique_buyers.Gender == 'M']
uniq_male = len(df_unique_buyers[df_unique_buyers.Gender == 'M'])
uniq_female = len(df_unique_buyers[df_unique_buyers.Gender == 'F'])
print("Total unique samples = " + str(len(df_unique_buyers)))
print("Unique Male buyers = " + str(uniq_male))
print("Unique Female buyers = " + str(uniq_female))

labels = 'Male', 'Female'
sizes = [(uniq_male / len(df_unique_buyers)) * 100, (uniq_female / len(df_unique_buyers)) * 100]
explode = (0.1, 0)
colors = ['#1e90ff', '#FF69B4']
plt.pie(sizes, explode = explode , labels = labels, colors = colors, autopct = '%1.1f%%', shadow = True, startangle = 90)
plt.axis('equal')
plt.show()


# **Evidently, less than 50% of the male shoppers consitute female buyers in this sample, roughly equivalent to an approximate 3:1 ratio.**

# **Now let's take a look at the spending pattern of the men and women in this sample**

# In[ ]:


columns = ['User_ID','Gender', 'Purchase']
df_subset = pd.DataFrame(df, columns=columns)
type(df_subset)


# In[ ]:


df_purchase = df_subset.groupby(['User_ID','Gender']).sum().reset_index()
df_purchase.head()


# **Did you notice the flip from df to df_purchase in terms of counts of men and women buyers in the first 5 samples?**

# In[ ]:


uniq_male = df_purchase[df_purchase.Gender == 'M']
uniq_female = df_purchase[df_purchase.Gender == 'F']


# In[ ]:


uniq_male["Purchase"].max(), uniq_male["Purchase"].min(), uniq_male["Purchase"].mean(), uniq_male["Purchase"].median()


# In[ ]:


sns.distplot(uniq_male["Purchase"])


# In[ ]:


uniq_female["Purchase"].max(), uniq_female["Purchase"].min(), uniq_female["Purchase"].mean(), uniq_female["Purchase"].median()


# In[ ]:


sns.distplot(uniq_female["Purchase"])


# In[ ]:


sns.catplot(x="Gender", y="Purchase", kind="violin", split=True,
                palette="pastel",data=df_purchase)


# From the above investigations, we could conclude than men shop more expensive items than women. Women have fatter violin width in the previous violin plot which indicates that women tend to buy cheaper products as compared to men. Therefore, the median for women is closer to that of men (while men still leading) but the high-end quality expenditures of men pulled the median up by ~34%.
# 
# **I wonder....**

# In[ ]:


sns.countplot(x='Gender', data=df, palette= sns.color_palette("husl", 2))


# **Among the total number of entries and considering the count of items shops, men are leading....**

# In[ ]:


sns.countplot(x='Marital_Status', data=df, palette= sns.color_palette("husl", 2))


# **Getting married did not stop men from shopping more and more....**

# In[ ]:


sns.countplot(x='Gender',hue="City_Category", data=df, palette=sns.cubehelix_palette(3))


# **Wow, men are unstoppable. They purchased items twice as much as women in all types of cities.....**

# In[ ]:


plt.figure(figsize = (20,10))
sns.countplot(x='Gender',hue="Product_Category_1", data=df, palette=sns.cubehelix_palette(18))


# **Notice how the product category 1 buying pattern for men and women are similar.  Can you fnd a product for which men have been buying less than women because I CAN'T.**

# **Conclusions**
# 
# 1. The sample is skewed, for example, the data is acquired from showrooms which is well-known for menswear.
# 2. The men have been buying the products for the women and so there are some hidden facts in the depths of this data that we have no clue about.
# 
# However, the myth is brutally busted by this dataset.
# 
# *Disclaimer: This was a light-hearted exploration of the BlackFriday dataset. Please do not take this seriously except for the statistics.*
