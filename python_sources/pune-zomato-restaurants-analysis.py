#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import seaborn as sns
import matplotlib.pyplot as plt 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
from wordcloud import WordCloud
from geopy.geocoders import Nominatim
from folium.plugins import HeatMap
import folium
import warnings
warnings.filterwarnings('ignore')
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Reference - 
# 
# 1. https://www.kaggle.com/shahules/zomato-complete-eda-and-lstm-model
# 
# 2. https://www.kaggle.com/parthsharma5795/finding-the-best-restaurants-in-bangalore

# Breakdown of this notebook:
# Loading the dataset: Load the data and import the libraries.
# **Data Cleaning:**
# 1. Cleaning column values.
# 2. Extracting digits for certain columns like Reviews, cost.
# 3. Dropping duplicates values.
# 4. Data Visualization: Using plots to find relations between the features.
# 

# In[ ]:


zomato = pd.read_csv('/kaggle/input/pune-restaurants-zomato/zomato_outlet_final.csv', delimiter=',')
zomato.head(n=2)


# # Data Cleaning

# In[ ]:


'Cleaning Data'

'Function to remove special characters from name'
def clean_data(cols, str_to_replace):
           
    for col in cols:
        zomato[col] = [str(x).replace(str_to_replace,"") for x in zomato[col]]    
        
    return zomato

def extract_digit(cols):
    
    for col in cols:
        zomato[col] = zomato[col].str.extract(r'(\d+)', expand=True)
        
    return zomato
'Calling our functions'
zomato = clean_data(['rest_name'], '\r\r\n')
zomato = clean_data(['cost', 'dine_reviews', 'delivery_reviews'], ",")
zomato = extract_digit(['cost', 'dine_reviews', 'delivery_reviews'])


# In[ ]:


zomato.head(n=3)


# In[ ]:


'Checking for duplicates'
zomato.drop_duplicates("link",keep='first',inplace=True)
zomato.reset_index(drop=True,inplace=True)
zomato.shape


# In[ ]:


print("Percentage null or na values in df")
((zomato.isnull() | zomato.isna()).sum() * 100 / zomato.index.size).round(2)


# In[ ]:


'Checking data type of columns'
zomato.info()


# In[ ]:


'Converting Reviews and cost to integer'
zomato.replace('NA', np.nan)
def convert_cols(cols):
    
    for col in cols:
        zomato[col] = zomato[col].astype(float)
        
    return zomato

zomato = convert_cols(['dine_reviews', 'delivery_reviews', 'cost'])


# # ** Exploratory Data Analysis**

# # Which chain has maximum no of outlets in Pune?

# In[ ]:


plt.figure(figsize=(10,7))
chains=zomato['rest_name'].value_counts()[:20]
sns.barplot(x=chains,y=chains.index,palette='Set2')
plt.title("Restaurants having maximum no of outlets in Pune")
plt.xlabel("Number of outlets")


# # Food Street in Pune?

# In[ ]:


plt.figure(figsize=(20,5))
locations=zomato['loc'].value_counts()[:20]
g = sns.barplot(locations.index,locations,palette="Set1")
g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")
g
plt.title("No of restaurants in Locality")


# # Most preferred restaurant type in Pune?
# 

# In[ ]:


r_type =zomato['rest_type'].value_counts()[:10]
sns.barplot(x=r_type,y=r_type.index,palette='Set2')
plt.title("Most preferred restaurant type in Pune")
plt.xlabel("Number of restaurants")


# In[ ]:


'Defining a function to plot graphs for top 10 attributes'


def bar_plot_h(cols, titles):
    print(cols)
    n=len(cols)
    f, axes = plt.subplots(1, 2, figsize=(15, 7))
    for col, ax, i  in zip(cols, axes.flatten()[:n], range(0,n)):
        c=zomato[col].value_counts()[:10]
        sns.barplot(c, c.index,ax = ax, palette = 'Set2').set_title(titles[i])
        
    
       
    
bar_plot_h(['cuisine', 'liked'], ["Cuisine liked by most PUNEKARS??? ", "What do PUNEKARS love???"])  


# # Generating wordclouds for most liked dish 

# In[ ]:


'Converting dish liked into 1 text item'
text = " ".join(str(dish) for dish in zomato.liked)
print ("There are {} words in the combination of all review.".format(len(text)))
stopwords = ['NaN']
wordcloud = WordCloud( stopwords = stopwords, background_color="white").generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# # Forming wordclouds for 5 restaurant type
# 

# In[ ]:


'Printing Restuarnat types'
rest=zomato['rest_type'].value_counts().index
print(rest)


# In[ ]:


rest=zomato['rest_type'].value_counts()[:5].index
def produce_wordcloud(df,rest):
    stopwords = ['NaN']
    plt.figure(figsize=(20,30))
    for i,r in enumerate(rest):
        plt.subplot(3,3,i+1)
        corpus=df[df['rest_type']==r]['liked'].tolist()
        corpus=','.join(str(x)  for x in corpus )
        wordcloud = WordCloud(stopwords = stopwords,max_font_size=None, background_color='white', collocations=False,
                      width=1500, height=1500).generate(corpus)
        plt.imshow(wordcloud)
        plt.title(r)
        plt.axis("off")
        

        
        
produce_wordcloud(zomato,rest)


# # Most voted restaurants in Pune?
# 

# In[ ]:



def bar_plot(cols, titles):
    print(cols)
    n=len(cols)
    f, axes = plt.subplots(1, 2, figsize=(15, 6))
    for col, ax, i  in zip(cols, axes.flatten()[:n], range(0,n)):
        c=zomato[['rest_name',col]].sort_values(by = col,ascending = False)[:8]
        g = sns.barplot(x = c[col], y =c['rest_name'],ax = ax, palette = 'Set2').set_title(titles[i])
        'For showing values in barplot'
        for p in ax.patches:
            width = p.get_width()
            ax.text(width -1.5  ,
                p.get_y()+p.get_height()/2. + 0.2,
                '{:1.2f}'.format(width),
                ha="right")

bar_plot( [ 'cost', 'dine_reviews'], ["Most Expensive Restaurant?", "Most reviewed ?"])           


# In[ ]:


'Cost Distribution '
fig, ax = plt.subplots(figsize=[16,4])
sns.distplot(zomato['cost'],ax=ax)
ax.set_title('Cost Distrubution for all restaurants')


# # Inferences - 
# 
# More than 50% restaurants have rating between 3.5 and 4

# In[ ]:


zomato.describe()


# In[ ]:


'--------------Rating Distribution----------'

plt.figure(figsize=(7,6))
rating=zomato['dine_rating'].value_counts()
sns.barplot(x=rating.index,y=rating)
plt.xlabel("Ratings")
plt.ylabel('count')


# # Localities for Most Expensive 20 restaurants?

# In[ ]:


most_voted = zomato[['loc','cost']].sort_values(by = 'cost', ascending = False)[:20]
xy = most_voted['loc'].value_counts()
sns.barplot(x = xy, y = xy.index)


# # Most Popular Cuisine for a locality?

# In[ ]:


popular_cuis=zomato.groupby(['loc','cuisine']).agg('count')
data=popular_cuis.groupby(['loc'],
                as_index=False).apply(lambda x : x.sort_values(by="rest_name",ascending=False).head(3))['rest_name'].reset_index().rename(columns={'rest_name':'count'})

data.head(n=10)


# # Best cheap Restaurants?
# 
# Criteria - 
# 1. Cost < 1000
# 2. Rating (dine or delivery) > 4.3
# 3. No of reviews( dine or delivery) > 4000
# 

# In[ ]:


cheap_rest=zomato[['rest_name','cost', 'loc','rest_type','cuisine', 'delivery_rating', 'dine_rating', 'delivery_reviews', 'dine_reviews']]
cheap_rest=cheap_rest[(cheap_rest['cost'] <1000) & ( (cheap_rest['dine_rating'] > 4.3 )| (cheap_rest['delivery_rating'] > 4.3)) &   ((cheap_rest['delivery_reviews'] > 4000) | (cheap_rest['dine_reviews'] >4000))]
cheap_rest.head(n=10)


# # Highly Rated and Highly reviewed Expensive restaurants

# In[ ]:


exp_rest=zomato[['rest_name','cost', 'loc','rest_type','cuisine', 'delivery_rating', 'dine_rating', 'delivery_reviews', 'dine_reviews']]
exp_rest=exp_rest[(exp_rest['cost'] >2500) & ( (exp_rest['dine_rating'] > 4.0 )| (exp_rest['delivery_rating'] > 4.0))&   ((exp_rest['delivery_reviews'] > 400) | (exp_rest['dine_reviews'] >400))].sort_values(by=['dine_reviews'], ascending = False)
exp_rest.head()


# # Interesting thing to note in Expensive restaurants is that none of them provide delivery. 

# In[ ]:




