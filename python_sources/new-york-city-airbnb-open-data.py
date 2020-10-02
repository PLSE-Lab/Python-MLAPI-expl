#!/usr/bin/env python
# coding: utf-8

# This kernel is a reconstruction of https://www.kaggle.com/dgomonov/data-exploration-on-nyc-airbnb for educational purposes.

# Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present more unique, personalized way of experiencing the world. Today Airbnb has become a kind of a service that is used and preferred all over the world. Data analysis on huge number of listing provided through Airbnb is crucial for the company. This huge amount of listings generate a lot of data that can be used to understand the customers, hosts, behaviour and performance etc on the platform.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing
import os
import seaborn as sns
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Loading and printing the first few rows 

# In[ ]:


data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.head()


# In[ ]:


print('Total number of observations in the dataset : {}'.format(data.shape[0]))


# ## About the dataset
# The dataset contains about 49000 rows of NYC Airbnb Data.

# In[ ]:


data.dtypes


# By checking the dtypes of the dataframe we can see that the data provides rich amount of features to explore the data. 
# Next We need to check for the null values in the data and make sense out of them i.e. whether the null values are due to data collection error or the null values signify something.

# # Data Wrangling

# In[ ]:


data.isnull().sum()


# The missing values in the above output doesn't seem like the need special consideration. The `name`,`host_name` are not required for our analysis. The null values counts in `last_review` and `reviews_per_month` are equal. This makes sense as the `last_review` is a date and if there are no reviews for a particular entry, then it will be empty. Same is the case with `reviews_per_month`.

# We can just drop the columns like `last_review`,`id` and `host_name` as the dont necessarily add any significant information to our analysis. Also we can fill the nulls in `reviews_per_month` with 0.

# In[ ]:


data = data.drop(['id','host_name','last_review'],axis=1)
data.head()


# In[ ]:


data.fillna({'reviews_per_month':0},inplace=True)
data.reviews_per_month.isnull().sum()


# In[ ]:


data.neighbourhood_group.unique()


# In[ ]:


data.neighbourhood.unique()


# In[ ]:


data.room_type.unique()


# Understanding the unique values and the categorical data that we had in out dataset was the last step we had to do. It looks like for those column values we will be doing some mapping to prepare the dataset for predictive analysis.

# # Exploring and Visualizing the data

# Now that we are ready for an exploration of our data, we can make a rule that we are going to be working from left to right. The reason for this maybe that with this approach, no matter how big the dataset is, we will remember to explore each and every feature in the data to learn as much as we can about the data.

# In[ ]:


#lets see which host ids have the most listings on Airbnb
top_host = data.host_id.value_counts().head(10)
top_host


# In[ ]:


#coming back to our dataset, we can confirm out findings with already existing column called 'calculated_host_listings_count'
top_host_check = data.calculated_host_listings_count.max()
top_host_check


# In[ ]:


#setting the figsize for future visualizations
sns.set(rc={'figure.figsize':(10,8)})


# In[ ]:


viz_1 = top_host.plot(kind='bar')
viz_1.set_title('Hosts with the most listings in NYC')
viz_1.set_ylabel('Count of listings')
viz_1.set_xlabel('Host IDs')
viz_1.set_xticklabels(viz_1.get_xticklabels(), rotation=45)


# We can see the distribution of listing between the top 10 hosts. The top host has 300+ listings.

# # Prices by neighbourhood_groups

# In[ ]:


data.groupby('neighbourhood_group')['price'].describe().T


# The above table shows the summary statistics for prices of each of the `neighbourhood_group`. It can be observed that there are a lot of outliers on the higher side of the price.<br>
# We can cut the outliers from the data for better visualization.

# In[ ]:


sub_df = data[data.price <500]

viz_2 = sns.violinplot(data=sub_df,x='neighbourhood_group',y='price')
viz_2.set_title('Density and distribution of prices for each neighbourhood_group')


# With the statistical table and the violinplot we can definitely observe a couple of things about the distribution of prices for Airbnb in NYC. First, we can see that Manhattan has the highest median price of \$150  per night, followed by Brooklyn at \$90 per night. Queens and Staten Island have similar distribution and Bronx is the cheapers of all.
# This distribution and densities comes as no surprise as it is a known fact that manhattan is one of the most expensive places on earth, where as Bronx appears to have lower standards of living

# In[ ]:


# as we saw earlier there are too many unique values in neighbourhood columns. So lets get the top 10.

top_nbh = data.neighbourhood.value_counts().head(10).index


# In[ ]:


#now lets combine the neighbourhoods and room types for richer visualizations
subdf_1 = data.loc[data['neighbourhood'].isin(top_nbh)]

viz = sns.catplot(x='neighbourhood',hue='neighbourhood_group',col='room_type',data=subdf_1,kind='count')
viz.set_xticklabels(rotation=90)
plt.show()


# Lets breakdown what we can see from this plot. First, we can see that our plot consists of 3 subplots - that is the power of using catplot.
# With such output we can easily proceed with comparing distributions among interesting attributes. Y and X axes stay exactly the same for each subplot, Y-axis represents a count of observations and X-axis observations we want to count. However, there are 2 more important elements; column and hue; those 2 differentiate subplots. After we specify the column and determined hue we are able to observe and compare our Y and X axes among specified column as well as color-coded. <br>
# The observation that is definitely contrasted the most is that `Shared room` tyype Airbnb listing is barely available among 10 most listing populated neighbourhooods. Then, we can see that for these 10 neighbourhood only 2 boroughs are represented: Manhattan and Brooklyn; that was somewhat expected as Manhattan and Brooklyn are one of the most travelled destinations, therefore would have the most listing availability. We can also observe that Bedford-Stuyvesant and Williamsburg are the most popular for Manharran borough, and Harlem for Brooklyn

# In[ ]:


#lets see what we can do with our given longitude and latitude columns

#lets check how the scatterplot will come out
viz_3 = sub_df.plot(kind='scatter',x='longitude',y='latitude',label='availability_365', c='price',cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))
viz_3.legend()


# The scatterplot worked just fine to output our latitude and longitude points. However, it would be nice to have a map below for fully immersive heatmap in our case - let's see what we can do!

# In[ ]:


import urllib

plt.figure(figsize=(10,8))

i= urllib.request.urlopen('https://upload.wikimedia.org/wikipedia/commons/e/ec/Neighbourhoods_New_York_City_Map.PNG')
nyc_img=plt.imread(i)

plt.imshow(nyc_img,zorder=0,extent=[-74.258,-73.7,40.49,40.92])
ax=plt.gca()

sub_df.plot(kind='scatter',x = 'longitude',y='latitude',label='availability_365',c='price',ax=ax,cmap=plt.get_cmap('jet'),colorbar=True,alpha=0.4,zorder=5)
plt.legend()
plt.show()


# In[ ]:


#lets check the names columns now

names =[]

for name in data.name:
    names.append(name)
    
def split_names(name):
    spl=str(name).split()
    return spl

names_for_count=[]

for x in names:
    for word in split_names(x):
        word = word.lower()
        names_for_count.append(word)


# In[ ]:


from wordcloud import WordCloud

wc = WordCloud(width=800,height=800,background_color='white',min_font_size=10).generate(str(names_for_count))
plt.imshow(wc)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# We can see how the host  are simply describing their listing in a short form with very specific terms for easier search by a potential traveller. Such words are 'room', 'bedroom', 'private', 'apartment', 'beautiful' etc. This shows that there are no catchphrases or 'popular/trending' terms that are used for names; hosts use very simple terms for describing the space and the area where the listing is. This technique was somewhat expected as dealing with multilingual customers can be tricky and you definitely want ot describe your space in a concise and understood form as much as possible.

# In[ ]:


#lastly we will look at 'number_of_reviews'

#lets grab 10 most reviewed listings in NYC
top_reviewed_listings=data.nlargest(10,'number_of_reviews')
top_reviewed_listings


# In[ ]:


price_avg = top_reviewed_listings.price.mean()
print('Average price per night: {}'.format(price_avg))


# From this output we can understand that top reviewed listing have an average price of \$65 with most of the listing under \$50 and 9/10 of them are 'Private room' type; top reviewed listing has 629  reviews

# # Conclusion
# 
# This Airbnb dataset for 2019 appeared to be a very rich dataset with a variety of columns that allowed us to do deep data exploration on each significant column presented. First we have found hosts that take good advantage of the Airbnb platform and provide the most listings; we found that out top host has 327 listings. After that, we proceeded with analyzing the boroughs and neighbourhood listing densities and what areas were more popular than another. Next we put good use of our latitude and longitude columns and used to create a geographical heatmap color-coded by price of listings. Further, we came back to the first column with name strings and had to do a bit more coding to parse each title and analyze existinh trends on how listings are named as well as what was the count for the most used words by hosts. Lastly we found the most reviewed listings and analyzed some additional attributed. For our data exploration purposes, it also would be nice to have a couple additional features, such as positive and negative numeric reviews or 0-5 star average review for each listing; addition of these features would help to determine the best-reviewed hosts for NYC along with 'number_of_review' column that is provided. Overall, we discovered a very good number of interesting relationships between features and explained each step of the process.

# In[ ]:




