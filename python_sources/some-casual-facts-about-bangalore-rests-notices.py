#!/usr/bin/env python
# coding: utf-8

# # Some casual facts about Bangalore Rests (notices)

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-poster')
get_ipython().run_line_magic('matplotlib', 'inline')

from wordcloud import WordCloud

import nltk
from nltk.stem import WordNetLemmatizer
# nltk.download('stopwords')
from nltk.corpus import stopwords

from geopy.geocoders import Nominatim
from folium.plugins import HeatMap
import folium

import os

import warnings
warnings.filterwarnings('ignore') 


# In[ ]:


from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')


# In[ ]:


PATH_TO_DATA = '../input'


# In[ ]:


df = pd.read_csv(os.path.join(PATH_TO_DATA, 'zomato.csv'))
df = df.drop(['url', 'address', 'phone', 'menu_item'], 1)


# In[ ]:


cloud = WordCloud(background_color='white',
                            max_font_size=400,
                            width=1500,
                            height=700,
                            max_words=50,
                            stopwords = stopwords.words('english')
                            ).generate(' '.join(df.name))

plt.figure(figsize=(15,7))
plt.axis('off')
plt.imshow(cloud, interpolation="bilinear")
plt.title('Biryani is the most popular ethnic dish in Bangalore (judging by name)\n', size=21)
plt.show()


# In[ ]:


plt.figure(figsize=(15,9))
names_of_restaurants = df.name.value_counts()[:20]
sns.barplot(x=names_of_restaurants,
            y=names_of_restaurants.index, 
            palette='Blues_d')
plt.title("Italian, Indian, American and Chinese food are the most popular (judging by name)")
plt.xlabel("Number of outlets")
plt.show()


# In[ ]:


df = df.dropna(subset=['cuisines'])

cloud = WordCloud(background_color='white',
                            max_font_size=400,
                            width=1500,
                            height=700,
                            max_words=50,
                            stopwords = stopwords.words('english')
                            ).generate(' '.join(df.cuisines))

plt.figure(figsize=(15,7))
plt.axis('off')
plt.imshow(cloud, interpolation="bilinear")
plt.title('Indian and Chinese food are the most popular (judging by cuisines)\n')
plt.show()


# In[ ]:


df.rate = df.rate.str.replace('/5$', '')
df.rate = df.rate.replace('NEW', np.nan)
df.rate = df.rate.replace('-', np.nan)
df = df.dropna(subset=['rate'])
df.rate = df.rate.astype('float')
df = df.dropna(subset=['rest_type'])


# In[ ]:


plt.figure(figsize=(15,9))
type_of_restaurants = df.rest_type.value_counts()[:10]
sns.barplot(x=type_of_restaurants,
            y=type_of_restaurants.index, 
            palette='Blues_d')
plt.title("Fast food is the most popular type of restaurants in Bangalore")
plt.xlabel("Number of restaurants")
plt.show()


# In[ ]:


df.dish_liked = df.dish_liked.fillna('Uknown')

cloud = WordCloud(background_color='white',
                            max_font_size=400,
                            width=1500,
                            height=700,
                            max_words=50,
                            stopwords = stopwords.words('english')
                            ).generate(' '.join(df.dish_liked[df.dish_liked != 'Uknown']))

plt.figure(figsize=(15,7))
plt.axis('off')
plt.imshow(cloud, interpolation="bilinear")
plt.title('The most popular dish is a chicken in various states of aggregation (dish_liked)\n')
plt.show()


# In[ ]:


df = df.rename(columns={'approx_cost(for two people)': 'approx_cost_for_two_people',
                        'listed_in(type)': 'listed_in_type',
                        'listed_in(city)': 'listed_in_city'})
df = df.dropna(subset=['approx_cost_for_two_people'])
df.approx_cost_for_two_people = df.approx_cost_for_two_people.str.replace(',', '')
df.approx_cost_for_two_people = df.approx_cost_for_two_people.astype('float')


# In[ ]:


df['cleantext'] = df.reviews_list.str.lower()
df.cleantext = df.cleantext.str.replace('[-_]', ' ')
df.cleantext = df.cleantext.str.replace('[^a-z ]+', '')

morph = WordNetLemmatizer()
def text_normalization(input_text):
    return ' '.join([morph.lemmatize(item) for item in input_text.split(' ')])
df.cleantext = df.cleantext.apply(text_normalization)

stop_words = set(stopwords.words('english'))
def remove_stops(input_text):
        return ' '.join([w for w in input_text.split() if not w in stop_words])
df.cleantext = df.cleantext.apply(remove_stops)
df.cleantext = df.cleantext.str.replace('rated ratedn ', '')
df.cleantext = df.cleantext.str.replace('wa', '')

df.reviews_list = df.pop('cleantext')


# In[ ]:


cloud = WordCloud(background_color='white',
                            max_font_size=400,
                            width=1500,
                            height=700,
                            max_words=50,
                            stopwords = stopwords.words('english')
                            ).generate(' '.join(df.reviews_list[df.rate < np.quantile(df.rate, q=0.01)]))

plt.figure(figsize=(15,7))
plt.axis('off')
plt.imshow(cloud, interpolation="bilinear")
plt.title('The main problem is service (reviews from 1% of the worst places)\n')
plt.show()


# In[ ]:


g = sns.catplot(y="listed_in_type",x="rate",data=df, kind="point", height = 10, )
plt.title('Deliveries have the lowest rate',size = 20)
plt.show()


# In[ ]:


cloud = WordCloud(background_color='white',
                            max_font_size=400,
                            width=1500,
                            height=700,
                            max_words=50,
                            stopwords = stopwords.words('english')
                            ).generate(' '.join(df.reviews_list[df.rate > np.quantile(df.rate, q=0.99)]))

plt.figure(figsize=(15,7))
plt.axis('off')
plt.imshow(cloud, interpolation="bilinear")
plt.title('Reviews from 1% of the best places: Non veg\n')
plt.show()


# In[ ]:


cuisines = []
rate = []
for i in range(len(df.cuisines)):
    n = df.cuisines.values[i].split(',')
    cuisines.extend(n)
    rate.extend(len(n)*[df.rate.values[i]])
df_cuisines_rate = pd.DataFrame({'cuisines': cuisines, 
                                 'rate': rate})
df_cuisines_rate.cuisines = df_cuisines_rate.cuisines.str.replace('^ ', '')
df_cuisines_rate_top = df_cuisines_rate[df_cuisines_rate.cuisines.isin                                        (df_cuisines_rate.cuisines.value_counts().index[0:10])]


# In[ ]:


g = sns.catplot(y="cuisines",x="rate",data=df_cuisines_rate_top, kind="point", height = 10, )
plt.title('Italian and Continental food have the highest rate (among top 10 mass ones)',size = 20)
plt.show()


# In[ ]:


location_biryani = df.location[df.cuisines.str.contains('Biryani')]
location_continental = df.location[df.cuisines.str.contains('Continental')]


# In[ ]:


geo_codes_biryani = pd.DataFrame({'name': location_biryani.value_counts().index,
                                  'num_of_outlets': location_biryani.value_counts().values}) 
geo_codes_biryani.name = 'Bangalore '+geo_codes_biryani.name
geo_codes_biryani['Location'] = geo_codes_biryani.name.apply(lambda x: Nominatim(user_agent="app").geocode(x))
geo_codes_biryani['lat'] = geo_codes_biryani.Location[-geo_codes_biryani.Location.isna()].apply(lambda x: x[-1][0])
geo_codes_biryani['lon'] = geo_codes_biryani.Location[-geo_codes_biryani.Location.isna()].apply(lambda x: x[-1][1])
geo_codes_biryani = geo_codes_biryani.dropna()

map_bangalore = folium.Map(location=[12.97, 77.59], control_scale=True, zoom_start=11)

HeatMap(geo_codes_biryani[['lat','lon','num_of_outlets']].values.tolist(), radius=14).add_to(map_bangalore)


# ## Biryani is more in the south

# In[ ]:


map_bangalore


# In[ ]:


geo_codes_continental = pd.DataFrame({'name': location_continental.value_counts().index,
                                  'num_of_outlets': location_continental.value_counts().values}) 
geo_codes_continental.name = 'Bangalore '+geo_codes_continental.name
geo_codes_continental['Location'] = geo_codes_continental.name.apply(lambda x: Nominatim(user_agent="app").geocode(x))
geo_codes_continental['lat'] = geo_codes_continental.Location[-geo_codes_continental.Location.isna()].apply(lambda x: x[-1][0])
geo_codes_continental['lon'] = geo_codes_continental.Location[-geo_codes_continental.Location.isna()].apply(lambda x: x[-1][1])
geo_codes_continental = geo_codes_continental.dropna()

HeatMap(geo_codes_continental[['lat','lon','num_of_outlets']].values.tolist(),
        radius=14).add_to(map_bangalore)


# ## Continental food is more in the center

# In[ ]:


map_bangalore


# In[ ]:


cuisines = []
cost = []
for i in range(len(df.cuisines)):
    n = df.cuisines.values[i].split(',')
    cuisines.extend(n)
    cost.extend(len(n)*[df.approx_cost_for_two_people.values[i]])
df_cuisines_cost = pd.DataFrame({'cuisines': cuisines, 
                                 'cost': cost})
df_cuisines_cost.cuisines = df_cuisines_cost.cuisines.str.replace('^ ', '')
df_cuisines_cost_top = df_cuisines_cost[df_cuisines_cost.cuisines.isin                                        (df_cuisines_cost.cuisines.value_counts().index[0:10])]


# In[ ]:


g = sns.catplot(y="cuisines",x="cost",
                data=df_cuisines_cost_top, kind="point", height = 10, )
plt.title('Italian and Continental are the most expensive cuisines (among top 10 mass ones)',size = 20)
plt.show()


# In[ ]:


plt.figure(figsize=[15, 9])
plt.plot(df.rate, df.approx_cost_for_two_people, '.')
plt.plot(df.rate, len(df.rate)*[df.approx_cost_for_two_people.median()], label='median cost')
plt.plot(len(df.rate)*[df.rate.median()], df.approx_cost_for_two_people, label='median rate')
plt.legend(title_fontsize=18)
plt.xlabel('rate')
plt.ylabel('approx_cost_for_two_people')
plt.title('Very expencive doesn`t always mean very good')
plt.show()

