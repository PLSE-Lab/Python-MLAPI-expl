#!/usr/bin/env python
# coding: utf-8

# # Black Friday
# ### A study of sales trough consumer behaviours
# https://www.kaggle.com/mehdidag/black-friday
# 
#  Dataset of 550 000 observations about the black Friday in a retail store, it contains different kinds of variables either numerical and categorical.
# 

# **This Kernel is still in progress, and we appreciate any constructive insights. If you found this helpful, please give this Kernel an upvote, as it keeps up motivated to continue to progress and share with the community.**

# ### Libraries
# 
# We will be using the Pandas, Numpy, Seaborn, and Matplotlib Python libraries for this analysis.

# In[ ]:


# Warnings
import warnings
warnings.filterwarnings('ignore')

# Data and analysis
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))

sns.set(style='darkgrid')
plt.rcParams["patch.force_edgecolor"] = True


# ***

# # Data Import and Feature Exploration
# 
# Let's import the data into a Pandas dataframe and check out some of its broader aspects to see what we're working with.

# In[ ]:


df = pd.read_csv('../input/BlackFriday.csv')


# In[ ]:


# First 5 rows:
df.head(5)


# In[ ]:


print(df.info())
print('Shape: ',df.shape)


# ***

# ## <font color='blue'>Missing Values</font>
# 
# 

# In[ ]:


total_miss = df.isnull().sum()
perc_miss = total_miss/df.isnull().count()*100

missing_data = pd.DataFrame({'Total missing':total_miss,
                            '% missing':perc_miss})

missing_data.sort_values(by='Total missing',
                         ascending=False).head(3)


# Since most products will belong to only one category, it makes sense for less products to have a second category, let alone a third one.

# ***

# ## <font color='blue'>Unique Values</font>

# Lets now explore the unique values in some of the features. Remember there is a total of 537577 entries:

# In[ ]:


print('Unique Values for Each Feature: \n')
for i in df.columns:
    print(i, ':',df[i].nunique())


# In[ ]:


# Info about products
print('Number of products:',df['Product_ID'].nunique())
print('Number of categories:',df['Product_Category_1'].unique().max())
print('Highest and lowest purchase:',
      df['Purchase'].max(),',',df['Purchase'].min())


# In[ ]:


# Info about shoppers
print('Number of shoppers:',df['User_ID'].nunique())
print('Years in city:',df['Stay_In_Current_City_Years'].unique())
print('Age Groups:',df['Age'].unique())


# ***

# ## <font color='blue'>Gender</font>
# Lets first find whether the data is uniformly distributed by gender by looking at how many entries belong to each one:

# In[ ]:


count_m = df[df['Gender']=='M'].count()[0]
count_f = df[df['Gender']=='F'].count()[0]


# In[ ]:


print('Number of male clients:',count_m)
print('Number of female clients:',count_f)


# We can see that the number of male clients recorded exceeds the number of female clients recorded by almost 4 times. For this reason, it will be much more informational to analyze **Gender** by using ratios instead of counting each entry. Lets see how much each gender spent in regards to eachself:

# In[ ]:


print('Female Purchases:',round(df[df['Gender']=='F']['Purchase'].sum()/count_f,3))
print('Male Purchases:',round(df[df['Gender']=='M']['Purchase'].sum()/count_m,3))


# In[ ]:


plt.pie(df.groupby('Gender')['Product_ID'].nunique(),labels=['Male','Female'],
       shadow=True, autopct='%1.1f%%',colors=['steelblue','cornflowerblue'])
plt.title('Unique Item Purchases by Gender')
plt.show()


# Although almost even, women did purchase a slightly wider array of products than men did. Now, lets analyze the proportions of each gender's purchase in terms of the product categories:

# In[ ]:


# Individual groupby dataframes for each gender
gb_gender_m = df[df['Gender']=='M'][['Product_Category_1','Gender']].groupby(by='Product_Category_1').count()
gb_gender_f = df[df['Gender']=='F'][['Product_Category_1','Gender']].groupby(by='Product_Category_1').count()

# Concatenate and change column names
cat_bygender = pd.concat([gb_gender_m,gb_gender_f],axis=1)
cat_bygender.columns = ['M ratio','F ratio']

# Adjust to reflect ratios
cat_bygender['M ratio'] = cat_bygender['M ratio']/df[df['Gender']=='M'].count()[0]
cat_bygender['F ratio'] = cat_bygender['F ratio']/df[df['Gender']=='F'].count()[0]

# Create likelihood of one gender to buy over the other
cat_bygender['Likelihood (M/F)'] = cat_bygender['M ratio']/cat_bygender['F ratio']

cat_bygender['Total Ratio'] = cat_bygender['M ratio']+cat_bygender['F ratio']


# In[ ]:


cat_bygender.sort_values(by='Likelihood (M/F)',ascending=False)


# This table tells us a lot about how likely a type of product is to be bought in regards of gender. For instance, men are almost 3 times as likely to buy an item in category 17, while women are almost 2 times as likely to buy a product in category 14.

# ***

# ## <font color='blue'>Age</font>
# Since as of now, **Age** values are strings, lets encode each group so they can be represented with an integer value which a machine learning algorithm can understand:

# In[ ]:


# Encoding the age groups
df['Age_Encoded'] = df['Age'].map({'0-17':0,'18-25':1,
                          '26-35':2,'36-45':3,
                          '46-50':4,'51-55':5,
                          '55+':6})


# In[ ]:


prod_byage = df.groupby('Age').nunique()['Product_ID']

fig,ax = plt.subplots(1,2,figsize=(14,6))
ax = ax.ravel()

sns.countplot(df['Age'].sort_values(),ax=ax[0], palette="Blues_d")
ax[0].set_xlabel('Age Group')
ax[0].set_title('Age Group Distribution')
sns.barplot(x=prod_byage.index,y=prod_byage.values,ax=ax[1], palette="Blues_d")
ax[1].set_xlabel('Age Group')
ax[1].set_title('Unique Products by Age')

plt.show()


# It's quite apparent that the largest age group amongst the customers is 26-35. Interestingly, the distribution of product purchase, in terms of quantity, does not vary greatly amongst the age groups. This means that, though the 26-35 age group is the most popular, the other age groups purchase almost as many unique items as them. But does this mean that the amount of money spent amongst the age groups is the same? Let's see...

# In[ ]:


spent_byage = df.groupby(by='Age').sum()['Purchase']
plt.figure(figsize=(12,6))

sns.barplot(x=spent_byage.index,y=spent_byage.values, palette="Blues_d")
plt.title('Mean Purchases per Age Group')
plt.show()


# Our data clearly shows that the amount of money made from each age group correlates proportionally with the amount of customers within the age groups. This can be valuable information for the store, as it might want to add more products geared towards this age group in the future, or perhaps work on marketing different items to increase a broader diversity in the age groups of their customers.

# ***

# ## <font color='blue'>Occupation</font>

# This sections draws some insights on our data in terms of the occupation of the customers.

# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(df['Occupation'])
plt.title('Occupation Distribution')
plt.show()


# In[ ]:





# In[ ]:


plt.figure(figsize=(12,6))
prod_by_occ = df.groupby(by='Occupation').nunique()['Product_ID']

sns.barplot(x=prod_by_occ.index,y=prod_by_occ.values)
plt.title('Unique Products by Occupation')
plt.show()


# In[ ]:


spent_by_occ = df.groupby(by='Occupation').sum()['Purchase']
plt.figure(figsize=(12,6))

sns.barplot(x=spent_by_occ.index,y=spent_by_occ.values)
plt.title('Total Money Spent per Occupation')
plt.show()


# Once again, the distribution of the mean amount spent within each occupation appears to mirror the distribution of the amount of people within each occupation. This is fortunate from a data science perspective, as we are not working with odd or outstanding features. Our data, in terms of age and occupation seems to simply make sense.

# ## <font color='blue'>Products</font>

# Here we explore the products themselves. This is important, as we do not have labeled items in this dataset. Theoretically, a customer could be spending $5,000 on 4 new TVs, or 10,000 pens. This difference matters for stores, as their profits are affected. Since we do not know what the items are, let's explore the categories of the items.

# In[ ]:


plt.figure(figsize=(12,6))
prod_by_cat = df.groupby('Product_Category_1')['Product_ID'].nunique()

sns.barplot(x=prod_by_cat.index,y=prod_by_cat.values, palette="Blues_d")
plt.title('Number of Unique Items per Category')
plt.show()


# Category labels 1, 5, and 8 clearly have the most items within them. This could mean the store is known for that item, or that the category is a broad one.

# In[ ]:


category = []
mean_purchase = []


for i in df['Product_Category_1'].unique():
    category.append(i)
category.sort()

for e in category:
    mean_purchase.append(df[df['Product_Category_1']==e]['Purchase'].mean())

plt.figure(figsize=(12,6))

sns.barplot(x=category,y=mean_purchase)
plt.title('Mean of the Purchases per Category')
plt.xlabel('Product Category')
plt.ylabel('Mean Purchase')
plt.show()


# Interestingly enough, our most popular categories are not the ones making the most money. This appears to be a big store, and they may be aware of this. Yet this same form of analysis can be used in the case of a smaller store that might not be aware, and it could be very useful.

# ## <font color='blue'>Estimate of price and quantity of purchase
# </font>
# Since the **Purchases** feature alludes to how much a customer paid for an unknown amount of a certain item, let's make a bold assumption that the lowest purchase paid by product is the price of said item:

# In[ ]:


# Dictionary of product IDs with minimum purchase
prod_prices = df.groupby('Product_ID').min()['Purchase'].to_dict()


# Now, the purchase value for each item ID is grouped by what one could asume is the amount of items purchased. Then, the prices and quantities calculated below are an estimate, but it is a very good one:

# In[ ]:


def find_price(row):
    prod = row['Product_ID']
    
    return prod_prices[prod]


# In[ ]:


df['Price'] = df.apply(find_price,axis=1)


# In[ ]:


df['Amount'] = round(df['Purchase']/df['Price']).astype(int)


# In[ ]:





# In[ ]:





# In Progress...

# In[ ]:





# In[ ]:




