#!/usr/bin/env python
# coding: utf-8

# # EDA - Zomato Restaurants Data

# ####    In this article, I have given an attempt to find some relevant information in Indian food business through resturants data provided by some of the fastest growing Indian startup 'Zomato'. These kinds of startups have changed old business approaches and schemes in ready-made and fast-food industries not just by expediting food delivery process but also, introducing attractive  services for food lovers across and beyond the nation. 
# ##### Few business questions that I have understood at the end of data analysis are:
#            - Few resturants has aggregate zero - rating. Is this rating TRUE or FALSE?
#            - Does customer ratings have any impact on average cost charged by the resturants?
# 

# ### Load libraries

# In[ ]:


from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook

import matplotlib.pyplot as plt
from scipy.stats import skew
import pandas as pd
import numpy as np
import seaborn as sns

output_notebook()

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings


# In[ ]:


data = pd.ExcelFile('../input/zomato/zomato.xlsx').parse()


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


numerical_features   = data.select_dtypes(include = [np.number]).columns
categorical_features = data.select_dtypes(include= [np.object]).columns


# In[ ]:


numerical_features


# In[ ]:


categorical_features


# #### There are 17 features available for this dataset, however, all of them does not seem to be relevant. Most of them are also redundant.
#         - 'City' and '(Longitude & Latitude)' provides same information. You can directly search a city on a map or use (Latitude, Longitude) combination to locate a city on the graph.
#         - 'Restaurant ID', 'Country Code', 'Restaurant Name', 'City' are nominal data hence, does not carry any pertinent information. 
#         - I believe 'Rating color', 'Rating text', 'Aggregate rating', 'Votes' represents same piece of information. Let's cross verify the same

# In[ ]:


col = ['Rating color', 'Rating text', 'Aggregate rating', 'Votes']
data[col].head(2)


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 5))
sns.countplot('Rating color', data = data, ax = ax1)
sns.countplot('Rating text', data = data, hue = 'Rating color', ax= ax2,)


#     - Above visualisations clearly shows, "rating text' and 'rating color' are holding same information in different format. 
#      Each 'rating text' have only one 'rating color'. I will be using 'rating text' here onwards, in my analysis.
#      
# There is a key insight that can not be ignored. More than 2000 resturants are not rated at all while, majority resturants have received average rating

# In[ ]:


p = figure(plot_width=800, plot_height=400, title = "Votes Vs. Aggregate Voting")
p.xaxis.axis_label = 'Aggregate Rating'
p.yaxis.axis_label = 'Votes'

colormap = {'Excellent': 'green', 'Very Good': 'blue', 'Good': 'orange', 'Average': 'yellow', 
            'Not rated': 'black', 'Poor': 'red'}
colors = [colormap[x] for x in data['Rating text']]

p.asterisk(x = data['Aggregate rating'], y = data['Votes'], size=20, color=colors, alpha=0.7,)

show(p)


#     - observations having zero aggregate rating has minimal votes given. Note: zero aggr. rating could have two meaning: either no one has rated these resturants or these resturants have actually received zero rating. we'll dig into this further.
#     - Aggr. rating is not linearly proportional to votes. This means that both of these features have individual importance as well as collective importance. 

# ### Question1.  Zero ratings are true or false?

# In[ ]:


zero_rated_resturants = data[data['Aggregate rating'] == 0]
rated_resturants = data[data['Aggregate rating'] > 0]


# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize = (20, 5))
ax1.set_xlabel('Country Code')
ax1.set_ylabel('No. of resturants')
ax1.set_yscale('symlog')
current_palette_4 = sns.color_palette("Greys_r", 4)
sns.set_palette(current_palette_4)
sns.countplot('Country Code', data = zero_rated_resturants, ax = ax1).set_title("Countries having no rated resturants")


# In[ ]:


fig, ax = plt.subplots(2, 2, figsize = (20, 9), sharey=False, sharex=False)
ax[0][0].set_yscale('symlog')
palett = sns.color_palette("Blues_r")
sns.set_palette(palette=palett)
sns.countplot('City', data = zero_rated_resturants[zero_rated_resturants['Country Code']==1], ax= ax[0][0],).set_title("Country code 1")
sns.countplot('City', data = zero_rated_resturants[zero_rated_resturants['Country Code']==30], ax= ax[0][1]).set_title("Country code 30")
sns.countplot('City', data = zero_rated_resturants[zero_rated_resturants['Country Code']==215], ax= ax[1][0]).set_title("Country code 215")
sns.countplot('City', data = zero_rated_resturants[zero_rated_resturants['Country Code']==216], ax= ax[1][1]).set_title("Country code 216")


# In[ ]:


zero_rated_resturants[zero_rated_resturants['Country Code']== 1].head(30)


# In[ ]:


rated_resturants[rated_resturants['Country Code']== 1].head(20)


# In[ ]:


print(zero_rated_resturants['Has Table booking'].unique())
print(zero_rated_resturants['Has Online delivery'].unique())
print(zero_rated_resturants['Is delivering now'].unique())
print(zero_rated_resturants['Switch to order menu'].unique())
print(zero_rated_resturants['Price range'].unique())


# ### Answer:
#     All these non-rated resturants have one common properties - does not provide "Switch to order menu" facility
#     
#  - Out of 4 countries (India, Brazil, United Kingdom and United states), India stands on the top having non-rated or zero rated resturants. It should be valid for, zomato has large business expansion in India.
#  - Indian cities Faridabad, Noida, New Delhi, Gurgoan and Gaziabad has non-rated resturants along with rated resturants.
# Comming to the point "IS zero-rated values true or false?"
#     - Non-rated resturants are in following places: Boarder, Chowk, Malls, Colonies, small market places, city stalls
#             - People for these hostels are mostly are migrating customers. Hence, it is possible that these customers have not rated the food quality as they would have not got enough experiance to rate the quality.
#     - Rated resturants are in places like canttines, Cinema Complex, Tourist places, Big hotels, etc.
#             - These kinds of locations have mostly a bunch of fixed customers who visit frequently. Based on routine visits people have given ratings to these hotels.
#   - According to me ratings could false as well as true because just business locations alone is not sufficient to answer this questions. Given the following data could help us in better ways to reach to conclusion:
#               - Date of establishment of the resturants: If zero rated resturants are opened long ago and still running. This means people have not rated at all knowingly. zero-rating is a false value here. Date information can also help us to understand if these resturants need to improve there service quality to attract customers ratings and reviews.

# ### Question2. Is average cost  propotional to avg. resturants rating by customer?

# In[ ]:


# Reference table
country_code = pd.ExcelFile('../input/zomato/Country-Code.xlsx').parse()
country_code


# In[ ]:


from IPython.display import IFrame
IFrame('https://public.tableau.com/profile/rageeni#!/vizhome/ZomatoResturantsAnalysis/ZomatoResturantAnalysis?publish=yes', 
       width=1100, height=1000)


# ### Answer:
#     Note - Refer 'country_code' table to find corresponding countries names.
# - As it can be seen major business for zomato is established in India while expanding across world. 
# - Jakarta, Bandu, Ankara, and Colombia are few major cities having average cost for two much higher than other cities in world.
# - Resturant'Lincoln' from 'United States' has highest rating given though, avg. cost for two is negligible compared to Banda
# - Among Indian market, Panchkula has highest average staying cost for two persons.
# - Resturants in Panchkula has average rating similar to metro cities of India.
# - "Avg. Aggregate rating Vs. Avg. Cost for two" visualisation clear shows that average cost is not directly depended on customers rating for the resturants.
