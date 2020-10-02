#!/usr/bin/env python
# coding: utf-8

# # Investigating names in the USA for the past century (1910-2017) via Social Security applications
# ## All sourced from Data.gov

# ## Setting up the environment and loading libraries

# In[181]:


# Environment defined by docker image: https://github.com/kaggle/docker-python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.graph_objs as go
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image    # to import the image
import warnings; warnings.simplefilter('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
import os


# ### Using Big Query Helper to query the data from Google BigQuery via the Kernel

# In[182]:


import bq_helper
from bq_helper import BigQueryHelper
usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")
usa_names.list_tables()


# ### There are two datasets to query, so we can query both if desired
# #### would be nice to look at the most complete data if possible

# In[183]:


QUERY_thru_2013 = "SELECT year, gender, name, sum(number) as year_total FROM `bigquery-public-data.usa_names.usa_1910_2013` group by year, gender, name"


# In[184]:


QUERY_thru_current = "SELECT year, gender, name, sum(number) as year_total FROM `bigquery-public-data.usa_names.usa_1910_current` group by year, gender, name"


# ### How large is the larger query?

# In[185]:


usa_names.estimate_query_size(QUERY_thru_current)


# ### Less than 1/5 of a Gigabyte so using pandas and the larger table should certainly be fine

# In[186]:


all_names = usa_names.query_to_pandas(QUERY_thru_current)


# ### Appears there are 612k names applied for in US through SS since 1910...
# #### There are 1 million according to US census so we capture more than half

# In[187]:


all_names['gender'].describe()


# In[188]:


all_names.loc[all_names['gender']=='F'].describe()


# In[189]:


all_names.loc[all_names['gender']=='M'].describe()


# ### There are more unique names for females as shown below

# In[190]:


sns.countplot(x='gender',data=all_names, linewidth=5, edgecolor=sns.color_palette("Set3"))


# ### Because we want to be conciencious stewards of data, we should check to see there is a 50/50 split overall for all applications
# #### Appears that there are 7.5% less applications for females, this could be counter intuitive, so possible data quality concern
# #### Although looks like https://en.wikipedia.org/wiki/List_of_countries_by_sex_ratio gives a figure of 5% less females for USA

# In[191]:


all_names.loc[all_names['gender']=='F'].year_total.sum()  / all_names.loc[all_names['gender']=='M'].year_total.sum() * 100


# ### Curious of generational break downs; the join data will need to be created
# #### Adding a lookup DF with Pew Reserch Centers Generational Definition
# http://pewrsr.ch/2GRbL5N

# In[192]:


generations = pd.DataFrame([], columns=['year','generation'])
generations['year'] = pd.Series(range(1910,2017))
generations.loc[(generations['year'] < 1928), 'generation'] = 'Pre-Silent'
generations.loc[(generations['year'] >= 1928) & (generations['year'] < 1946), 'generation'] = 'Silent'
generations.loc[(generations['year'] >= 1946) & (generations['year'] < 1965), 'generation'] = 'Baby Boomers'
generations.loc[(generations['year'] >= 1965) & (generations['year'] < 1982), 'generation'] = 'Generation X'
generations.loc[(generations['year'] >= 1982) & (generations['year'] < 1997), 'generation'] = 'Millenials'
generations.loc[(generations['year'] >= 1997), 'generation'] = 'Post-Millenials'
generations.sample(10)


# ### Join generations to data

# In[193]:


all_names_gen = pd.merge(all_names,generations, on='year')
all_names_gen


# ### Aggregate names according to generations

# In[194]:


gen_group = all_names_gen.groupby(['generation','name'])
gen_stats = gen_group['year_total'].agg([np.sum,np.std,np.mean])
gen_stats.loc['Millenials']


# ### Reviewing 'flattened' list of top year-over-year Millenial names applied on average 

# In[195]:


" ".join(gen_stats.loc['Millenials'].nlargest(100,'mean').index.tolist())


# ### Instead of sampling the data, we can get an accurate representation of the top USA Millenial name applications by looking at the most common names over the years within the generation and including a normalized frequency
# #### We can think of these as the names defining the generation

# In[207]:


top_millenial = list(zip(gen_stats.loc['Millenials'].nlargest(100,'mean')['sum'].index,
                         gen_stats.loc['Millenials'].nlargest(100,'mean')['sum']))

norm = [float(i)/sum([x[1] for x in top_millenial]) for i in [x[1] for x in top_millenial]]

top_millenial_norm = list(zip(gen_stats.loc['Millenials'].nlargest(100,'mean')['sum'].index,
         norm))


# ### Building a word cloud with a masking image to visualize the names for the generational subset born between 1982 and 1996

# In[197]:


gi_mask = np.array(Image.open("../input/millenial.png"))

wordcloud = WordCloud( max_font_size=75,
                       mask = gi_mask,
                       background_color='white',
                       width=1000, height=450
                     ).fit_words(dict(top_millenial_norm))
image_colors = ImageColorGenerator(gi_mask)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")

plt.title("Wordcloud for Top Names in Millenial Generation", fontsize=35)

plt.show()


# ### To create customized wordclouds for the remaining generations, utilize the function below with genaration, name stats DF, and image
# #### This way, we don't have to recode for each word cloud

# In[198]:


def generation_cloud(gen_id,gen_stats,png_mask) :
    top_ = list(zip(gen_stats.loc[gen_id].nlargest(250,'mean')['sum'].index,
                         gen_stats.loc[gen_id].nlargest(250,'mean')['sum']))

    norm = [float(i)/sum([x[1] for x in top_]) for i in [x[1] for x in top_]]

    top_norm = list(zip(gen_stats.loc[gen_id].nlargest(100,'mean')['sum'].index,
         norm))

    image_mask = np.array(Image.open(png_mask))

    wordcloud = WordCloud( max_font_size=120,
                       mask = image_mask,
                       background_color='black',
                       width=1600, height=400
                     ).fit_words(dict(top_norm))
    
    image_colors = ImageColorGenerator(image_mask)
    
    plt.figure(figsize=(30,10))
    plt.title("Wordcloud for Top Names in " + gen_id + " Generation" , fontsize=35)
    plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis("off")
    plt.show()


# In[199]:


generation_cloud('Pre-Silent',gen_stats,"../input/gi.png")


# In[200]:


generation_cloud('Silent',gen_stats,"../input/silent.png")


# In[201]:


generation_cloud('Baby Boomers',gen_stats,"../input/boomer.png")


# In[202]:


generation_cloud('Generation X',gen_stats,"../input/genx.png")


# In[203]:


generation_cloud('Post-Millenials',gen_stats,"../input/gi.png")


# # Name popularity predictions with Prophet

# ### How has the name 'Sage' risen or fallen in popularity?

# In[225]:


line_dat = pd.DataFrame(all_names_gen.loc[(all_names_gen['name'] == 'Sage') & (all_names_gen['gender'] == 'F')]).sort_values(by='year')
plt.plot( 'year','year_total',data = line_dat)


# ### Using 'Prophet' developed by Facebook create a simple time series forecasting model for # of applications

# In[244]:


from fbprophet import Prophet

line_dat['year'] = pd.to_datetime(line_dat['year'],format = '%Y')
pd.to_numeric(line_dat['year_total'])
ts_single = line_dat[['year','year_total']]
ts_single['cap'] = 2* max(ts_single['year_total'])

sns.set(font_scale=1) 
ts_date_index = ts_single
ts_date_index = ts_date_index.set_index('year')
ts_prophet = ts_date_index.copy()
ts_prophet.reset_index(drop=False,inplace=True)
ts_single.columns = ['ds','y','cap']
ts_single.head()


# In[245]:


model = Prophet(changepoint_prior_scale=10, growth = 'logistic').fit(ts_single)

future = model.make_future_dataframe(periods=10,freq='Y')
future['cap'] = 2* max(ts_single['y'])
forecast = model.predict(future)
fig = model.plot(forecast) 

