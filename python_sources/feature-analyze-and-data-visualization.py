#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This Python notebook explore Mercari's data set. Try to give us a intuitive feeling of the data set. 
# 
# 1. Deal with the missing data
# 2. Price distribution for all product
# 3. Take a look at the "shipping" feature
# 4. Take a look at the "item_condition_id" feature
# 5. Analyze the "name" and "description" feature and explore on how to combine text feature to other feature
# 6. Take a look at "category_name" feature, break them into 3 level of categories
# 7. Take a look at "brand_name" feature
# 
# This notebook take https://www.kaggle.com/huguera/mercari-data-analysis as a reference. Let's Salute the original author.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

import string
import time

get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading data

# In[ ]:


df_train = pd.read_csv('../input/train.tsv', sep='\t')


# ## First look of the data set

# In[ ]:


df_train.head()


# In[ ]:


print('Shape of train data set: ', df_train.shape)


# ## Deal with missing data
# 
# We will fill the missing data of "item_description" with "No description yet", just like the original data set.

# In[ ]:


df_train['item_description'].fillna(value="No description yet", inplace=True)


# In[ ]:


frac1 = 100 * df_train[df_train['price'] <= 0].shape[0] / df_train.shape[0]
print('%0.2f%% percent of product have 0 price. We may need drop them when we train our model.' % frac1)
df_train = df_train[df_train['price'] > 0]


# ## Price distribution

# In[ ]:


df_train['price'].describe()


# In[ ]:


def price_hist(price, bins=100, r=[0,200], label='price', title='Price Distribution', **argv):
    plt.figure(figsize=(20, 15))
    plt.hist(price, bins=bins, range=r, label=label, **argv)
    plt.title(title, fontsize=15)
    plt.xlabel(label, fontsize=15)
    plt.ylabel('Samples', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.show()


# In[ ]:


price_hist(df_train['price'])


# In[ ]:


price_hist(np.log1p(df_train['price']), r=[0, 10])


# Seems that the price for most of the product is less than 100 USD. The log of the price is satisfied **Gaussian distribution**.

# ## Shipping
# 
# How many product with free shipping? What's the price distribution with/without shipping?

# In[ ]:


free_shipping = df_train[df_train['shipping']==1]
print('%0.2f%% percent of the product with free shipping' % (100 * len(free_shipping)/len(df_train)))


# In[ ]:


def price_double_hist(price1, price2, label1='price 1', label2='price 2',
                      bins=100, r=[0,200], title='Double Price Distribution', **argv):
    plt.figure(figsize=(20, 15))
    plt.hist(price1, bins=bins, range=r, label=label1, **argv)
    plt.hist(price2, bins=bins, range=r, label=label2, **argv)
    plt.title(title, fontsize=15)
    plt.xlabel('Price', fontsize=15)
    plt.ylabel('Samples', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.show()


# In[ ]:


price_double_hist(price1=df_train[df_train['shipping']==1]['price'], 
                  price2=df_train[df_train['shipping']==0]['price'],
                  label1='Price with shipping',
                  label2='Price without shipping',
                  normed=True, alpha=0.6)


# Seems the lower price have more chance to get free shipping. The boundary price is about 15 USD.

# ## item_condition_id
# 
# How is the "item_condition_id" related to it's price?

# In[ ]:


df = df_train[df_train['price']<100]
df.boxplot(column='price', by='item_condition_id', grid=True, figsize=(20, 15), return_type='dict');


# Seems that the condition id is not related too much to it's price. But we still need to further explore, maybe there are second hand products, which highly related to condition id.

# # Name and Description
# 
# Name and description of the product are the text features. We expect to get a lot of information from them. We will combine Name and Description together, since Name is some kind of summary of Description.

# ## Word cloud

# In[ ]:


start = time.time()
cloud = WordCloud(width=1440, height=1080).generate(" ".join(df_train['name'] + " " + df_train['item_description']))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
print('Time to compute and show word cloud: %0.2fs' % (time.time() - start))


# ### Has description or not
# 
# We learn from word cloud that, the "description yet" is a significant keyword. Does it has description or not will affect the price distribution?

# In[ ]:


df_train['has_description'] = 1
df_train.loc[df_train['item_description']=='No description yet', 'has_description'] = 0


# In[ ]:


price_double_hist(price1=df_train[df_train['has_description']==1]['price'], 
                  price2=df_train[df_train['has_description']==0]['price'],
                  label1='Price with description',
                  label2='Price without description',
                  normed=False, alpha=0.6)


# In[ ]:


price_double_hist(price1=df_train[df_train['has_description']==1]['price'], 
                  price2=df_train[df_train['has_description']==0]['price'],
                  label1='Price with description',
                  label2='Price without description',
                  normed=True, alpha=0.6)


# The first diagram is the same to the second one, except that the second one is normed. From the second normed diagram, it seems that, The low price product have more chance that do not have description, the boundary price is about 15USD. This meet our daily life experience.

# ### Has price or not
# 
# We know that the price in description will be removed and replace with a placeholder "[rm]". We can check does the description have a price or not will affect the price distribution or not.

# In[ ]:


df_train['has_price'] = 0
df_train.loc[df_train['item_description'].str.contains('[rm]', regex=False), 'has_price'] = 1
df_train.loc[df_train['name'].str.contains('[rm]', regex=False), 'has_price'] = 1


# In[ ]:


price_double_hist(price1=df_train[df_train['has_price']==0]['price'], 
                  price2=df_train[df_train['has_price']==1]['price'],
                  label1='Price without price in name/description',
                  label2='Price with price in name/description',
                  normed=False, alpha=0.6)


# In[ ]:


with_price = df_train[df_train['has_price']==1]
print('%0.2f%% percent of the product have price marked in name/description' % (100 * len(with_price)/len(df_train)))


# In[ ]:


price_double_hist(price1=df_train[df_train['has_price']==0]['price'], 
                  price2=df_train[df_train['has_price']==1]['price'],
                  label1='Price without price in name/description',
                  label2='Price with price in name/description',
                  normed=True, alpha=0.6)


# With normed hist diagram, it seems that, the low price (~8 USD) product have more chance to mark it's price in name/description.

# # TF-IDF
# 
# TF-IDF maybe the good way to deal with name and description. We will combine name and description together to compute TFIDF.

# In[ ]:


tfidf = TfidfVectorizer(
                        min_df=2, lowercase =True,
                        analyzer='word', token_pattern=r'\w+', use_idf=True, 
                        smooth_idf=True, sublinear_tf=True, stop_words='english')

vect_tfidf = tfidf.fit_transform(df_train['name'] + " " + df_train['item_description'])


# In[ ]:


df_train['tfidf'] = vect_tfidf.sum(axis=1)


# In[ ]:


plt.figure(figsize=(20, 15))
plt.scatter(df_train['tfidf'], df_train['price'])
plt.title('Train price X name/item_description TF-IDF', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('TF-IDF', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15);


# It seems that, the higher TFIDF sum value, the lower price. Also, we note that, simply sum up TFIDF of name and description may not a good approach. We may have another good method to transform TFIDF from sparse data(TFIDF vector) to dense data. This need to dig into deeper.
# 
# General speaking, there are 3 ways to combine text features with other features:
# 
# 1. Perform dimensionality reduction (such as LSA via TruncatedSVD) on your sparse data to make it dense and combine the features into a single dense matrix to train your model(s).
# 2. Add your few dense features to your sparse matrix using something like scipy's hstack into a single sparse matrix to train your model(s).
# 3. Create a model using only your sparse text data and then combine its predictions (probabilities if it's classification) as a dense feature with your other dense features to create a model (ie: ensembling via stacking). If you go this route remember to only use CV predictions as features to train your model otherwise you'll likely overfit quite badly (you can make a quite class to do this all within a single Pipeline if desired).
# 
# Take a reference on https://datascience.stackexchange.com/questions/987/text-categorization-combining-different-kind-of-features or google "combine text feature with other features".

# ## Categorys
# 
# Most of the product belong to 3 level of categorys. How much product for each category? What's the price distribution for each category?

# ### Missing data for category
# 
# Check how many empty category names. Also check how many non-three-level category names.

# In[ ]:


frac1 = 100 * df_train['category_name'].isnull().sum() / df_train.shape[0]
print('%0.2f percent empty category name' % frac1)


# In[ ]:


def transform_category_name(category_name):
    try:
        main, sub1, sub2 = category_name.split('/')
        return main, sub1, sub2
    except:
        return np.nan, np.nan, np.nan

df_train['category_main'], df_train['category_sub1'], df_train['category_sub2'] = zip(*df_train['category_name'].apply(transform_category_name))


# In[ ]:


frac1 = 100 * df_train['category_main'].isnull().sum() / df_train.shape[0]
print('%0.2f%% percent of the product do not fit 3 level category structure' % frac1)


# ### Product for main category

# In[ ]:


main_categories = [c for c in df_train['category_main'].unique() if type(c)==str]
categories_sum=0
for c in main_categories:
    categories_sum+=100*len(df_train[df_train['category_main']==c])/len(df_train)
    print('{:<25}{:>10.4f}% of training data'.format(c, 100*len(df_train[df_train['category_main']==c])/len(df_train)))
print('{:<25}{:>10.4f}% of training data'.format('nan', 100-categories_sum))


# "Women" and "Beauty" got more than a half of the products.

# In[ ]:


df = df_train[df_train['price']<100]
df.boxplot(column='price', by='category_main', grid=True, figsize=(20, 15));


# It seems that the main category have diferent price level. Category "Men" have the higher price compare to others.

# ### Product for second category

# In[ ]:


print('%d type of 2nd categories.' % len(df_train['category_sub1'].unique()))


# In[ ]:


def mean_price(groupby='category_sub1', cnt=20, top=True):
    df = df_train.groupby([groupby])['price'].agg(['size','sum'])
    df['mean_price']=df['sum']/df['size']
    df.sort_values(by=['mean_price'], ascending=(not top), inplace=True)
    df = df[:cnt]
    df.sort_values(by=['mean_price'], ascending=top, inplace=True)
    return df

def price_barh(df, title, ylabel):
    plt.figure(figsize=(20, 15))
    plt.barh(range(0,len(df)), df['mean_price'], align='center', alpha=0.5)
    plt.yticks(range(0,len(df)), df.index, fontsize=15)
    plt.xticks(fontsize=15)
    plt.title(title, fontsize=15)
    plt.xlabel('Price', fontsize=15)
    plt.ylabel(ylabel, fontsize=15)


# In[ ]:


df = mean_price(cnt=50)
price_barh(df, 'highest mean price sorted by 2nd category', '2nd category')


# In[ ]:


df = mean_price(cnt=50, top=False)
price_barh(df, 'lowest mean price sorted by 2nd category', '2nd category')


# ### Product for 3rd level category

# In[ ]:


print('%d type of 3rd categories.' % len(df_train['category_sub2'].unique()))


# In[ ]:


df = mean_price(cnt=50)
price_barh(df, 'highest mean price sorted by 3nd category', '3nd category')


# In[ ]:


df = mean_price(cnt=50, top=False)
price_barh(df, 'lowest mean price sorted by 3nd category', '3nd category')


# We should take category as input features, since it's highly related to it's price.

# ## Brand name
# 
# The price should highly related to Brand. Let's dig into it.

# In[ ]:


brands = df_train['brand_name'].unique()
print('There are totaly %d brand names' % len(brands))


# ### The most expensive brand

# In[ ]:


df = mean_price(groupby='brand_name', cnt=50, top=True)
price_barh(df, 'Most expensive product', 'brand')


# ### The most cheap brand

# In[ ]:


df = mean_price(groupby='brand_name', cnt=50, top=False)
price_barh(df, 'Most cheap product', 'brand')


# ### Has brand name or not
# 
# What's the percentage of the product have brand name? What's the price distribution of product with/without brand name?

# In[ ]:


df_train['has_brand'] = 1
df_train.loc[df_train['brand_name'].isnull(), 'has_brand'] = 0

product_without_brand_name = df_train[df_train['has_brand']==0]
print('%0.4f%% percent of the product do not have brand name' % (100 * len(product_without_brand_name) / len(df_train)))


# For my personal feeling, most of the product should have a brand name. But it's not true here. I saw some product do not fill the brand name in "brand_name" column, instead, it write the brand name in the "name" column. We may can work out a way to transfer brand name from "name" to "brand_name".

# In[ ]:


price_double_hist(price1=df_train[df_train['has_brand']==0]['price'], 
                  price2=df_train[df_train['has_brand']==1]['price'],
                  label1='Price without brand name',
                  label2='Price with brand name',
                  normed=False, alpha=0.6)


# From this hist diagram, we can see that more samples of the lower price product do not have brand name. The boundary price seems to be in 10USD. This meet our daily life experience.

# In[ ]:


boundary = 10
below_boundary = df_train[df_train['price']<=boundary]
above_boundary = df_train[df_train['price']>boundary]

product_without_brand_name = below_boundary[below_boundary['has_brand']==0]
print('%0.4f%% percent of the product below price boundary(%dUSD) do not have brand name' % ((100 * len(product_without_brand_name) / len(below_boundary)), boundary))

product_without_brand_name = above_boundary[above_boundary['has_brand']==0]
print('%0.4f%% percent of the product above price boundary(%dUSD) do not have brand name' % ((100 * len(product_without_brand_name) / len(above_boundary)), boundary))


# In[ ]:


price_double_hist(price1=df_train[df_train['has_brand']==0]['price'], 
                  price2=df_train[df_train['has_brand']==1]['price'],
                  label1='Price without brand name',
                  label2='Price with brand name',
                  normed=True, alpha=0.6)


# It seems that, the expensive product have more chance to have brand name. This meet our life experience.

# In[ ]:


df = df_train[df_train['price'] > 100]
frac1 = 100 * df['brand_name'].isnull().sum() / df.shape[0]
print('There are still %0.4f%% percent of product do not have brand name with price above 100 USD' % frac1)


# In[ ]:


price_double_hist(price1=df[df['has_brand']==0]['price'], 
                  price2=df[df['has_brand']==1]['price'],
                  label1='Price without brand name for > $100',
                  label2='Price with brand name for > $100',
                  normed=False, alpha=0.6)


# Thanks for your time to look at this kernel. If you like it, please upvote it, this will add to your favorite kernel in case you want to take a sencond glance.

# In[ ]:




