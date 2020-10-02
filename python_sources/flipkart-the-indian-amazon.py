#!/usr/bin/env python
# coding: utf-8

# # Flipkart Data Exploration

# # Contents

# - [Data](#Data)
# - [Objectives](#Objectives)
# - [Imports](#Imports)
# - [Cleaning and Prep](#Cleaning-and-Prep)
# - [Sales](#Sales)
# - [Discounted Sales](#Discounted-Sales)
# - [Conclusion](#Conclusion)
# - [Feedback](#Feedack-is-appreciated!!-Thank-you.)

# # Data

# [Flipkart](https://www.flipkart.com) is the largest Indian Ecommerce site. Started in 2007 by two ex Amazon employees, [Sachin Bansal](https://en.wikipedia.org/wiki/Sachin_Bansal) and [Binny Bansal](https://en.wikipedia.org/wiki/Binny_Bansal), Flipkart started off by selling books and eventually moving on to other goods, such as clothing, electronics and other consumables. For more information about Flipkart, check out there [Wikipedia](https://en.wikipedia.org/wiki/Flipkart) page. Courtesy to [Promptcloud](https://www.promptcloud.com/datastock-access-ready-to-use-datasets) and Kaggle for making this dataset possible!

# # Objectives

#  - What does the data look like? Missing? Duplicates? Nan values? 
#  - Does the Data need preparation? Datetime? Better categories? 
#  - Which months/years have the most sales?
#  - Which categories have the most sales?
#  - Which categories have the best discounts by percentage?

# # Imports

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('../input/flipkart_com-ecommerce_sample.csv')


# In[3]:


df.info()


# In[4]:


df.head()


# # Cleaning and Prep

# The cleaning and prep section will consist of searching for missing data, looking for duplicate data, converting the time column to a pandas Timestamp object and sectioning off the timestamp column into a month column and a year column.

# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(df.isnull(),
            cmap='plasma',
            yticklabels=False,
            cbar=False)
plt.title('Missing Data?',fontsize=20)
plt.xticks(fontsize=15)
plt.show()


# The Brand column is missing a hefty amount of data. Everything else looks relatively complete.

# In[6]:


df.duplicated().value_counts()


# No duplicate data

# In[7]:


#make this column into a datetime type for workability

df['crawl_timestamp'] = pd.to_datetime(df['crawl_timestamp'])


# In[8]:


df['crawl_year'] = df['crawl_timestamp'].apply(lambda x: x.year)
df['crawl_month'] = df['crawl_timestamp'].apply(lambda x: x.month)


# Now let's look at the product_category_tree column. This is a juicy column that can be divided up into subcategory columns. Below is how the column looks like and how we will divide it.

# In[9]:


print(df.product_category_tree[1])
print('\n')

for i in df.product_category_tree[1].split('>>'):
    print(i)


# In[10]:


df.product_category_tree[10].split('>>')[1][1:]


# In[11]:


#This .apply(lambda) will create a main category column from the first item in the product_category_tree column

df['MainCategory'] = df['product_category_tree'].apply(lambda x: x.split('>>')[0][2:])


# In[12]:


#These functions will be .apply() to the df. These functions will draw the second, third and fourth items from the product_category_tree
#try except statements because an index error occurs when there is no second/third/fourth item in the product_category_tree.

def secondary(x):
    try:
        return x.split('>>')[1][1:]
    except IndexError:
        return 'None '
    
def tertiary(x):
    try:
        return x.split('>>')[2][1:]
    except IndexError:
        return 'None '
    
def quaternary(x):
    try:
        return x.split('>>')[3][1:]
    except IndexError:
        return 'None '


# In[13]:


df['SecondaryCategory'] = df['product_category_tree'].apply(secondary)
df['TertiaryCategory'] = df['product_category_tree'].apply(tertiary)
df['QuaternaryCategory'] = df['product_category_tree'].apply(quaternary)


# # Sales

# Now that we have spruced up our data, let's visualize some sales!! First by month and year. After visualizing by time, we will visualize by category.

# In[47]:


plt.figure(figsize=(12,9))
df.groupby('crawl_month')['crawl_month'].count().plot(kind='bar')
plt.title('Sales Count by Month',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel('Month',fontsize=12)
plt.ylabel('Sales Count',fontsize=12)
plt.show()
print(df.groupby('crawl_month')['crawl_month'].count())


# Does anyone else see something funny going on here? Where is the data for July - November? Hmm.

# In[48]:


plt.figure(figsize=(10,6))
df.groupby('crawl_year')['crawl_year'].count().plot(kind='bar')
plt.title('Sales Count by Year',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel('Year',fontsize=12)
plt.ylabel('Sales Count',fontsize=12)
plt.show()
print(df.groupby('crawl_year')['crawl_year'].count())


# Only two years worth of data here. Nothing very insightful here.

# In[49]:


plt.figure(figsize=(12,8))
df['MainCategory'].value_counts()[:20].sort_values(ascending=True).plot(kind='barh')
plt.title('Main Category',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.show()
print('Top Ten Main Categories by Sales.\n')
print(df['MainCategory'].value_counts()[:10])


# Clothing, Jewellery, Shoes, electronics and automotive are the main category winners!

# In[50]:


plt.figure(figsize=(12,8))
df['SecondaryCategory'].value_counts()[:20].sort_values(ascending=True).plot(kind='barh')
plt.title('Secondary Category',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.show()
print('Top Ten Secondary Categories by Sales.\n')
print(df['SecondaryCategory'].value_counts()[:10])


# This is interesting. Women's Clothing is more than double of men's clothing. It is clear now that a lot of the users on Flipkart are Women, though there is no gender column. As if that is not enough, look at 'Necklaces', 'Women's Footwear', 'Rings' and 'Kid's Clothing'.

# In[51]:


plt.figure(figsize=(12,8))
df['TertiaryCategory'].value_counts()[:20].sort_values(ascending=True).plot(kind='barh')
plt.title('Tertiary Category',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.show()
print('Top Ten Tertiary Categories by Sales.\n')
print(df['TertiaryCategory'].value_counts()[:10])


# Western Wear? What does that mean?? Does that mean I want to look like a cowboy/cowgirl or does that mean I want to look like I live in Los Angeles?

# In[19]:


df[df['TertiaryCategory']=='Western Wear ']['product_name'][60:70]


# [Here](https://www.google.com/search?q=Megha+Casual+Short+Sleeve+Printed+Women%27s+Top&safe=off&client=firefox-b-1&source=lnms&tbm=isch&sa=X&ved=2ahUKEwirgMLc-draAhUL8IMKHd7YDZ4Q_AUoAXoECAAQAw&biw=1440&bih=735) is one google image search result. After doing a few more, I'm going to guess Western wear to mean California wear over the wild west. 

# In[52]:


plt.figure(figsize=(12,8))
df['QuaternaryCategory'].value_counts()[:20].sort_values(ascending=True).plot(kind='barh')
plt.title('Quaternary Category',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.show()
print('Top Ten Quaternary Categories by Sales.\n')
print(df['QuaternaryCategory'].value_counts()[:10])


# The Quaternary graph has lost a lot of insight. Nearly 6000 'None' values.

# Out of curiousity, what is the most expensive item in Flipkart during this time???

# In[21]:


df['retail_price'].max()


# In[22]:


df[df['retail_price']==571230.000000]


# A wrist watch!!!

# # Discounted Sales

# For the discounted sales section, we are going to have to do some data manipulation beforehand. First, we are going to create a discounted percentage column by subtracting the discount price from the retail price and dividing that amount by the retail price. 

# In[23]:


#discount percent = ((retail - sale) / retail) * 100

df['discount_%'] = round(((df['retail_price'] - df['discounted_price']) / df['retail_price'] * 100),1) 


# In[24]:


df[['product_name','retail_price','discounted_price','discount_%']].head()


# Second, we are going to create a few new dataframes that contain the product by category, average discounted percentages and count of each product.

# In[25]:


MainCategoryDiscount = pd.DataFrame(df.groupby('MainCategory').agg({
    'discount_%':[(np.mean)],
    'MainCategory':['count']
}))

SecondaryCategoryDiscount = pd.DataFrame(df.groupby('SecondaryCategory').agg({
    'discount_%':[np.mean],
    'SecondaryCategory':['count']
}))

TertiaryCategoryDiscount = pd.DataFrame(df.groupby('TertiaryCategory').agg({
    'discount_%':[np.mean],
    'TertiaryCategory':['count']
}))

QuaternaryCategoryDiscount = pd.DataFrame(df.groupby('QuaternaryCategory').agg({
    'discount_%':[np.mean],
    'QuaternaryCategory':['count']
}))


# In[26]:


MainCategoryDiscount.head()


# Third, we are going to combine the levels of the columns for visualization purposes.

# In[27]:


MainCategoryDiscount.columns = ['_'.join(col) for col in MainCategoryDiscount.columns]
SecondaryCategoryDiscount.columns = ['_'.join(col) for col in SecondaryCategoryDiscount.columns]
TertiaryCategoryDiscount.columns = ['_'.join(col) for col in TertiaryCategoryDiscount.columns]
QuaternaryCategoryDiscount.columns = ['_'.join(col) for col in QuaternaryCategoryDiscount.columns]


# In[28]:


MainCategoryDiscount.head()


# In[29]:


MainCategoryDiscount = MainCategoryDiscount.sort_values(by=['MainCategory_count'],ascending=False)[:20]
SecondaryCategoryDiscount = SecondaryCategoryDiscount.sort_values(by=['SecondaryCategory_count'],ascending=False)[:20]
TertiaryCategoryDiscount = TertiaryCategoryDiscount.sort_values(by=['TertiaryCategory_count'],ascending=False)[:20]
QuaternaryCategoryDiscount = QuaternaryCategoryDiscount.sort_values(by=['QuaternaryCategory_count'],ascending=False)[:20]


# In[53]:


plt.figure(figsize=(12,8))
MainCategoryDiscount['discount_%_mean'].sort_values(ascending=True).plot(kind='barh')
plt.title('Main Category by Discount',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel('MainCategory',fontsize=12)
plt.show()
print('Main Category by Discount (Percentage)\n')
print(MainCategoryDiscount['discount_%_mean'].sort_values(ascending=False)[:8])


# Automotive, clothing and electronics were some of the most discounted categories.

# In[54]:


plt.figure(figsize=(12,8))
SecondaryCategoryDiscount['discount_%_mean'].sort_values(ascending=True).plot(kind='barh')
plt.title('Secondary Category by Discount',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel('SecondaryCategory',fontsize=12)
plt.show()
print('Secondary Category by Discount (Percentage)\n')
print(SecondaryCategoryDiscount['discount_%_mean'].sort_values(ascending=False)[:8])


# For the secondary category, once again electronics and accessories. Some clothing, spare parts and jewellery. Coffee mugs? None is present.

# In[55]:


plt.figure(figsize=(12,8))
TertiaryCategoryDiscount['discount_%_mean'].sort_values(ascending=True).plot(kind='barh')
plt.title('Tertiary Category by Discount',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel('Tertiary Category',fontsize=12)
plt.show()
print('Tertiary Category by Discount (Percentage)\n')
print(TertiaryCategoryDiscount['discount_%_mean'].sort_values(ascending=False)[:8])


# Automotive parts on top again, followed by mostly jewellery and clothing. None is present. 

# In[ ]:


plt.figure(figsize=(12,8))
QuaternaryCategoryDiscount['discount_%_mean'].sort_values(ascending=True).plot(kind='barh')
plt.title('Quaternary Category by Discount',fontsize=20)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel('QuaternaryCategory',fontsize=12)
plt.show()
print('Quaternary Category by Discount (Percentage)\n')
print(QuaternaryCategoryDiscount['discount_%_mean'].sort_values(ascending=False)[:8])
#MainCategoryDiscount


# The Quaternary category is dominated by mostly clothing, with come automotive and electric categories.

# # Conclusion

# Does the following analysis satisfy any of the objectives??
#  
#  
# - The data looks complete, with little missing data. No Duplicates. 
#  
# - The data needed to be converted for datetime processing. More descriptive subcategories were created.
#  
# - The months section was missing about half of the years worth of data. No sales between July and November. The Year category only represented two years, both of which look very similar.
# 
# - The categories with the most sales were clothing (women's), electronics, jewellery and automotive. 
# 
# - Automotive, electronics, accessories and clothing held the highest discounts.

# # Feedack is appreciated!! Thank you.

# In[ ]:




