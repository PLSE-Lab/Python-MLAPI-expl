#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objs as go
from wordcloud import WordCloud
import geopandas as gpd


# # Zomato Chennai Listings as of May 27th 2020
# ---
# This dataset was created by scraping their webiste hence all the data used here belongs to Zomato and it's restaurants alone. This kernel is an attempt at analyzing Chennai's food scene through Data and how it's urban inhabitants are using the new-age services like Delivery & Cloud Kitchens. 
# 
# > Note that - this does not represent the entire spectrum of restaurants in Chennai since Zomato's main competition - Swiggy also has exclusive access to some restaurant data. And like any other Indian City, quite a lot of eateries fall into the unorganized/unlisted category - the ones you can find on the roadside which is where most of a city's population transact when we define 'eating out'.
# 
# ### Contents
# 1. [Loading Data and Pre-Processing](https://www.kaggle.com/phiitm/chennai-zomato-eda/#Loading-and-Pre-Processing-Data)
# 2. [Chennai Location Wise Distribution](https://www.kaggle.com/phiitm/chennai-zomato-eda/#Chennai-Location-Wise-Distribution)
# 3. [Most Number of Franchises](https://www.kaggle.com/phiitm/chennai-zomato-eda/#Most-Number-of-Franchises)
# 4. [Rating Distribution Location wise](https://www.kaggle.com/phiitm/chennai-zomato-eda/#Rating-Distribution-Location-wise)
# 5. [Price Comparision Location wise](https://www.kaggle.com/phiitm/chennai-zomato-eda/#Price-Comparision-Location-wise)
# 6. [What does Chennai Eat?](https://www.kaggle.com/phiitm/chennai-zomato-eda/#What-does-Chennai-eat?)
# 7. [Chennai and Vegetarianism - Myths debunked](https://www.kaggle.com/phiitm/chennai-zomato-eda/#Chennai-and-Vegetarianism---Myths-debunked)
# 8. [Most Popular Restaurants in Chennai](https://www.kaggle.com/phiitm/chennai-zomato-eda/#Most-Popular-Restaurants-in-Chennai)

# <div id="prep"> </div>
# # Loading and Pre-Processing Data 

# In[ ]:


df = pd.read_csv('../input/chennai-zomato-restaurants-data/Zomato Chennai Listing 2020.csv')
df.head()


# In[ ]:


df.replace(to_replace = ['None','Invalid','Does not offer Delivery','Does not offer Dining','Not enough Delivery Reviews','Not enough Dining Reviews'], value =np.nan,inplace=True)
df.isnull().sum()


# In[ ]:


df['name of restaurant'] = df['Name of Restaurant'].apply(lambda x: x.lower())
df['Top Dishes'] = df["Top Dishes"].astype(str)
df['Top Dishes'] = df['Top Dishes'].apply(lambda x:x.replace('[','').replace(']','').replace("'",'').replace('  ','').split(','))
df['Cuisine'] = df["Cuisine"].astype(str)
df['Cuisine'] = df['Cuisine'].apply(lambda x:x.replace('[','').replace(']','').replace("'",'').replace('  ','').split(','))
df['Features'] = df['Features'].apply(lambda x:x.replace('[','').replace(']','').replace("'",'').replace('  ','').split(','))
df['Dining Rating Count'] = df['Dining Rating Count'].astype("Float32")
df['Delivery Rating Count'] = df['Delivery Rating Count'].astype("Float32")


# In[ ]:


def locsplit(x):
    if len(x.split(','))==2:
        return x.split(',')[1].replace(' ','')
    else:
        return x

df['Location_2'] = df['Location'].apply(lambda x: locsplit(x))


# In[ ]:


print(len(df['Location'].unique()))
print(len(df['Location_2'].unique()))


# In[ ]:


print(df['Location_2'].unique().tolist())


# ## Unique Restaurant Features

# In[ ]:


feat_list = [feat.lower() for feats in df['Features'].tolist() for feat in feats]
print(len(set(feat_list)))
print(list(set(feat_list)))


# <div id="prep"></div>
# # Chennai Location Wise Distribution

# In[ ]:


fig = go.Figure(data=[go.Bar(
                x = df['Location_2'].value_counts()[:20].index.tolist(),
                y = df['Location_2'].value_counts()[:20].values.tolist())])

fig.show()


# <div id="fran"></div>
# ## Most Number of Franchises

# In[ ]:


df['name of restaurant'].value_counts()[:25]


# <div id="rate"></div>
# # Rating Distribution Location wise
# ## Dining only Ratings

# In[ ]:


bins_r = [0,2.5,4,5]
group_r = ['bad','good','best']
df['Dining Rating'] = df['Dining Rating'].astype(float)
df['Dine_Verdict'] = pd.cut(df['Dining Rating'],bins_r,labels=group_r)
yv = df['Dine_Verdict'].value_counts().tolist()
colors = ['blue','green','red']
fig = go.Figure(data=[go.Bar(x=group_r,y=yv,marker_color=colors)])
fig.show()


# In[ ]:


loc_price2 = pd.crosstab(df['Location_2'],df['Dine_Verdict'],margins=True,margins_name='Total') 
loc_price3 = loc_price2.sort_values('Total',ascending=False)[1:26]
loc_price3.drop(columns=['Total'],inplace=True)
loc_price3.div(loc_price3.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,figsize=(20, 10),color = ['g','y','b','r'])


# ### - Nungabakkam, Adyar, Anna Nagar East and Mylapore have the highest percentage of restaurants rated > 4 by Zomato's dining ratings

# ## Delivery only Ratings

# In[ ]:


bins_r = [0,3.5,4,5]
group_r = ['bad','good','best']
df['Delivery Rating'] = df['Delivery Rating'].astype(float)
df['Delivery_Verdict'] = pd.cut(df['Delivery Rating'],bins_r,labels=group_r)
yv = df['Delivery_Verdict'].value_counts().tolist()
colors = ['blue','green','red']
fig = go.Figure(data=[go.Bar(x=group_r,y=yv,marker_color=colors)])
fig.show()


# In[ ]:


loc_price4 = pd.crosstab(df['Location_2'],df['Delivery_Verdict'],margins=True,margins_name='Total') 
loc_price5 = loc_price4.sort_values('Total',ascending=False)[1:26]
loc_price5.drop(columns=['Total'],inplace=True)
loc_price5.div(loc_price5.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,figsize=(20, 10),color = ['g','y','b','r'])


# ### - Anna Nagar East, Kilpauk, Nungabakkam, Mogappair & Adyar have the highest percentage of ratings > 4 as sourced from Zomato's delivery orders

# <div id="price"></div>
# # Price Comparision Location wise

# In[ ]:


bins = [0,500,1000,2500,float("inf")]
groups = ['cheap','moderate','pricey','expensive']
df['Cost'] = pd.cut(df['Price for 2'], bins,labels=groups)
yc = df['Cost'].value_counts().tolist()
colors = ['green','orange','blue','red']
fig = go.Figure(data=[go.Bar(x=groups,y=yc,marker_color=colors)])
fig.show()


# In[ ]:


loc_price0 = pd.crosstab(df['Location_2'],df['Cost'],margins=True,margins_name='Total') 
loc_price1 = loc_price0.sort_values('Total',ascending=False)[1:26]
loc_price1.drop(columns=['Total'],inplace=True)
loc_price1.div(loc_price1.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,figsize=(20, 10),color = ['g','y','b','r'])


# ### - As expected Nungambakkam has the most number of expensive restaurants

# <div id="food"></div>
# # What does Chennai eat?

# In[ ]:


dishes = ' '.join(dish for dish_list in df['Top Dishes'].tolist() for dish in dish_list if dish != np.nan)
wordcloud = WordCloud(background_color='white',stopwords=['nan']).generate(dishes)
figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ### Clearly this breaks the stereotype of Chennai predominantly being a 'Idly Dosa' place and Chicken is hands-down the most eaten dish/food

# ## Most Popular Cuisines being served in Chennai restaurants

# In[ ]:


cuisines = ' '.join(dish for dish_list in df['Cuisine'].tolist() for dish in dish_list if dish != 'Invalid')
wordcloud = WordCloud(background_color='white').generate(cuisines)
figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# <div id="veg"></div>
# # Chennai and Vegetarianism - Myths debunked

# In[ ]:


def veg_status(feat_list):
    if 'Vegetarian Only' in feat_list:
        return 'Yes'
    elif ' Vegetarian Only' in feat_list:
        return 'Yes'
    else:
        return 'No'


# In[ ]:


df['Vegetarian Status'] = df['Features'].apply(lambda x: veg_status(x))
df['Vegetarian Status'].value_counts()


# ### Only 1830 restaurants listed on Zomato have declared themselves 'Vegetarian Only' which means only about 15% of them are classified Vegetarian which debunks the myth that Chennai is a 'Vegetarian' majority

# ## Dominant Vegetarian Restaurant Franchises

# In[ ]:


fig = go.Figure(data=[go.Bar(
                x = df.loc[df['Vegetarian Status'] == 'Yes']['name of restaurant'].value_counts()[:10].index.tolist(),
                y = df.loc[df['Vegetarian Status'] == 'Yes']['name of restaurant'].value_counts()[:10].values.tolist())])

fig.show()


# ## Locations with Maximum Vegetarian Restaurants

# In[ ]:


fig = go.Figure(data=[go.Bar(
                x = df.loc[df['Vegetarian Status'] == 'Yes']['Location'].value_counts()[:10].index.tolist(),
                y = df.loc[df['Vegetarian Status'] == 'Yes']['Location'].value_counts()[:10].values.tolist())])

fig.show()


# <div id="pop"></div>
# # Most Popular Restaurants in Chennai

# In[ ]:


df.loc[df['Dining Rating Count'].nlargest(10).index][['Name of Restaurant','Location_2','Dining Rating Count','Delivery Rating Count']]


# ### - Coal Barbeques, BBQ Nation and Onesta are the top 3 most popular restaurants as per the number of Zomato Dining Ratings

# In[ ]:


df.loc[df['Delivery Rating Count'].nlargest(10).index][['Name of Restaurant','Location_2','Delivery Rating Count','Dining Rating Count']]


# ### - Guntur Gongura, Hotelkaar Biriyani and Supriya Andhra Restaurants are the top 3 most popular delivery focussed restaurants as per the number of Zomato Delivery Ratings

# ## Don't forget to upvote this Kernel and the Dataset :)
