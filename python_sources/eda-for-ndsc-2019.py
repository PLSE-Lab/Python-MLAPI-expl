#!/usr/bin/env python
# coding: utf-8

# # **A Simple Exploratory Data Analysis**

# In[ ]:


import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.markers
import os
import seaborn as sns
import pprint
import string
import re

from tqdm import tqdm

plt.style.use('ggplot')
tqdm.pandas()


# # **Start off by loading all the necessary files**
# 
# And map the categories and product type back to the dataset, for ease of exploratory analysis.

# In[ ]:


df_train = pd.read_csv('../input/train.csv')


# In[ ]:


df_train.shape


# In[ ]:


df_train.info()


# In[ ]:


df_train.head()


# In[ ]:


# import the json file to view the categories

with open('../input/categories.json', 'rb') as handle:
    cat_details = json.load(handle)


# In[ ]:


pprint.pprint(cat_details)


# In[ ]:


category_mapper = {}
product_type_mapper = {}

for category in cat_details.keys():
    for key, value in cat_details[category].items():
        category_mapper[value] = key
        product_type_mapper[value] = category


# In[ ]:


# Display category mapper

category_mapper


# In[ ]:


# Display product mapper

product_type_mapper


# In[ ]:


# Apply the mapper to get new columns - category_type and product_type

df_train['Category_type'] = df_train['Category'].map(category_mapper)
df_train['Product_type'] = df_train['Category'].map(product_type_mapper)


# # **Check out the distributions of product types and category types**

# In[ ]:


plt.figure(figsize=(12,6))
plot = sns.countplot(x='Product_type', data=df_train)
plt.title('Product Type %', fontsize=20)
ax = plot.axes

for p in ax.patches:
    ax.annotate(f'{p.get_height() * 100 / df_train.shape[0]:.2f}%',
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', 
                va='center', 
                fontsize=11, 
                color='black',
                xytext=(0,7), 
                textcoords='offset points')


# In[ ]:


for product in cat_details.keys():
    plt.figure(figsize=(20,6))
    plot = sns.countplot(x='Category_type', 
                         data = df_train.loc[df_train['Product_type'] == product, :], 
                         order = df_train.loc[df_train['Product_type'] == product, 'Category_type'].value_counts().index)
    plt.xticks(rotation=90)
    plt.title(f'Category breakdown ({product})', fontsize=20)
    ax = plot.axes

    for p in ax.patches:
        ax.annotate(f'{p.get_height() * 100 / df_train.shape[0]:.2f}%',
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', 
                    va='center', 
                    fontsize=11, 
                    color='black',
                    xytext=(0,7), 
                    textcoords='offset points')
    plt.show()


#  # **Lets explore the 'title' feature. (e.g. Frequency Distribution, Word Cloud)**
#  
#  Do standard text processing stuff like removal of symbols and numbers (which are unlikely to have any predictive power)
#  Will only remove word with length = 1 (contrary to standard pre-processing step of removing any words with length < 3), because certain short words like 'gb' 'bb' have high predictive power for certain product type.

# In[ ]:


from nltk import FreqDist
from nltk.corpus import stopwords
from wordcloud import WordCloud


# In[ ]:


df_train.head()


# In[ ]:


def preprocessing(titles_array):
    
    processed_array = []
    
    for title in titles_array:
        
        # remove digits and other non-alphabets symbols with space (i.e. keep only alphabets and whitespaces)
        processed_title = re.sub('[^a-zA-Z ]', '', title.lower())
        words = processed_title.split()
        
        # remove words that have length of 1
        processed_array.append([word for word in words if len(word) > 1])
    
    return processed_array


# In[ ]:


def get_freqdist_wc(titles, product_type, num_words=30):
    
    freq_dist = FreqDist([word for title in titles for word in title])
    wordcloud = WordCloud(background_color='White').generate_from_frequencies(freq_dist)
    
    plt.figure(figsize=(22,6))
    plt.subplot2grid((1,5),(0,0),colspan=2)
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis('off')

    plt.subplot2grid((1,5),(0,2),colspan=3)
    plt.title(f'Frequency Distribution ({product_type}, Top {num_words})', fontsize=20)
    freq_dist.plot(num_words, marker='|', markersize=20)

    plt.tight_layout()
    plt.show()


# # **Start off with Product Type for a macro view**

# In[ ]:


mobile_titles = df_train.loc[df_train['Product_type'] == 'Mobile','title'].values
fashion_titles = df_train.loc[df_train['Product_type'] == 'Fashion','title'].values
beauty_titles = df_train.loc[df_train['Product_type'] == 'Beauty','title'].values

mobile_titles_p = preprocessing(mobile_titles)
fashion_titles_p = preprocessing(fashion_titles)
beauty_titles_p = preprocessing(beauty_titles)


# In[ ]:


get_freqdist_wc(mobile_titles_p, 'Mobile')


# In[ ]:


get_freqdist_wc(fashion_titles_p, 'Fashion')


# In[ ]:


get_freqdist_wc(beauty_titles_p, 'Beauty')


# # **A further drill-down to look at Frequency Distribution by Category**

# In[ ]:


def process_and_plot(cat_type, num_words = 10):
    titles = df_train.loc[df_train['Category_type'] == cat_type,'title'].values
    processed_titles = preprocessing(titles)
    print(f'{cat_type}\'s total counts:\t {len(titles)}')
    print(f'{len(titles) * 100/ df_train.shape[0]:.2f}% of the training set.')
    get_freqdist_wc(processed_titles, cat_type, num_words)
    


# In[ ]:


for category in tqdm(list(category_mapper.values())):
    process_and_plot(category)


# # **Summary of findings on the feature 'Title'**
# 1. For 'Mobile', the sub-categories have distinct keywords. Which means 'Title' alone should be a strong predictor of sub-categories.
# 2. For 'Fashion', the sub-categories have overlapping keywords (e.g. wanita - which means 'women' in Bahasa, dress), this is the one that require image recognition to differentiate its sub-categories.
# 3. For 'Beauty', it is a little in between 'Mobile' and 'Fashion', certain sub-categories like 'lipstick', 'other lip cosmetics', 'lip liner' are like sub-categories of each other and share similar keywords.  
# 4. Overall, 'Title' seems to be a robust predictor for categories (product_type), can be used as an intermediate model to provide information to other models.

# In[ ]:




