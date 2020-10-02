#!/usr/bin/env python
# coding: utf-8

# # Hi all! 
# ### In this notebook I'll demonstrate a simple exploratory analysis of Airbnb Data for New York City through 2019. This dataset is very accessible and presents us with great opportunities to analyze and visualize the data. I welcome advice and comments from all!
# 
# ### Access the data set from the [Kaggle source page](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data).
# 
# ### Before we dive in, a quick note on the data from the data source page:
# 
# ### Context  
# ### Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present more unique, personalized way of experiencing the world. This dataset describes the listing activity and metrics in NYC, NY for 2019.
# 
# ### Content  
# ### This data file includes all needed information to find out more about hosts, geographical availability, necessary metrics to make predictions and draw conclusions.

# Let's begin!

# # **1. Upload the Data**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from matplotlib import cm
from collections import OrderedDict

# Any results you write to the current directory are saved as output.


# # 2. Quick Look & Assessment of the Data

# In[ ]:


# I like to keep a version of the original data set unchanged should I ever need to
# look at it or compare to an edited dataset as I make changes
original_df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

# for cleaning purposes
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')


# In[ ]:


# Let's take a quick look at the dataset and get a feel for what we're working with.
# Already I notice some missing values and am considering which columns may be dropped
# for either redundancy or lack of relevance to our analysis
df.head()


# In[ ]:


# Let's get more information on the number of rows (48,895), columns (16), and data types.
df.info()


# In[ ]:


df['room_type'].unique()
# we find out that Airbnb rentals are classified as one of three distinct room types
# Private room
# Entire home/ apartment
# Shared Room


# In[ ]:


# let's do the same for neighborhood_group
df['neighbourhood_group'].unique()


# Further, we've got four columns that provide information based on location:
# * neighborhood_group
# * neighborhood
# * latitude 
# * longitude  
# 
# We'll deal with the first two in a bit. Latitude and longitude will be best utilized for some fun geolocation visualization towards the end.
# 
# 
# 
# 

# # 3. Cleaning the Data for Analysis

# ## 3.1 Missing Values

# In[ ]:


# Let's see which columns have missing values, the kinds of missing values they have, and how many.
df.isnull().sum()


# A total of four columns have missing values, but we don't have to adjust all of them:
# 1. 'name'
#     Unless you'd like to do a text based analysis of the rental names, this column won't add too much to your analysis.  
# **Decision: DROP**
# 2. 'host_name'
#     Here again, this column contains personal information and doesn't add much to our analysis. Even if for some reason we did need to use it, its use would be limited as in most cases only the first name is provided.   
# **Decision: DROP**    
# 3. 'last_review'
#     Since these are all dates, a missing value most likely means that there has been no review for this rental. I won't be using this for any analysis or visualization.  
#     **Decision: Drop **   
# 
# 4. 'reviews_per_month'
#     We might be able to do some analysis with this. This is the kind of situation where own intuition supplements the analysis. In this column, missing values most likely do not mean that the data points were not recorded but that they don't exist, a contrast I've learned to appreciate when dealing with missing values.   
# **Decision: REPLACE missing values with 0's**  
# 

# In[ ]:


df.drop(['name','host_name','last_review'],axis = 1,inplace=True)

# we now have 13 columns to work with
df.head()


# In[ ]:


# Now to address the missing values from the 'reviews_per_month' page
df['reviews_per_month'].fillna(0,inplace=True)


# In[ ]:


# After addressing these four columns, let's verify that we don't have any 
# missing values left:
df.isnull().sum()


# ## 3.2 Renaming Columns
# The thirteen remaining columns will be easy to work with.  
# 
# One thing I would like to do is rename the neighborhood_group column, which actually corresponds to what is known in New York as the five boroughs. As I'm a native New Yorker, this is a personal choice :)

# In[ ]:


# rename the column using the rename() method
df.rename(columns = {'neighbourhood_group':'borough'}, inplace = True)
# Let's verify the change
df.head()


# I found this cleaning process refreshingly smooth. Now on to the next step.

# # 4. Explore & Visualize the Data
# *Quick Notes*:   
# 1. For larger and more complex datasets, these steps would be more clearly divided. Given the accessibility and modest size of this dataset, my intution was that these steps could be done together. 
# 2. Everyone approaches a dataset from a different angle and for different purposes. While a few questions may be commonplace, your curiosity may take you in different directions. The following presents only one set of analysis and visualizations. Feel free to follow your curiousity and questions as well.
# 

# ### Let's begin with a few barplots that will help us visualize the quantity, kind and distribution of listings across New York City
# 

# In[ ]:


# Let's look at some statistics by the three kinds of room types available onn Airbnb
by_room = df.groupby('room_type').agg(['count'])
by_room.head()


# In[ ]:


by_room.index
# this index call will come in handy in a barchart that separates values by room type. It will go on the x axis


# In[ ]:


by_room['id']['count']
# this count will go on our y axis


# In[ ]:


sns.set_style('darkgrid')

plt.rcParams['axes.axisbelow'] = True
plt.grid(b=True,which ='major',axis='y')
my_colors =['g','b','m']

# This is the primary code guiding the structure of the plot. Notice how we specified the values for both axes.
plt.bar(x=by_room.index, height=by_room['id']['count'], color=my_colors, alpha=.7)

# Label the data
plt.xlabel('Type of Rental',color='r',fontsize=13)
plt.ylabel('Number of Rentals',color='r',fontsize=13)
plt.title('Airbnb Rentals by Room Type',color='r',fontsize='20',fontweight='bold')

# I set the upper y limit to 30000 in order to get a little more room at the top of the plot. 
# Otherwise, it would be a little tight 
plt.ylim(0,30000)


# We can see that entire home/apt and private rooms make up the bulk of the listed 2019 airbnb rentals

# In[ ]:


# Next, let's see how the listings are distributed across the five boroughs
by_borough = df.groupby('borough').agg(['count'])
by_borough.head()


# In[ ]:


# Here I sorted the data from highest to lowest before plotting it below
by_borough_sorted = by_borough['id']['count'].sort_values(ascending=False)


# In[ ]:


sns.set_style('darkgrid')

plt.grid(b=True,which ='major',axis='y')
my_colors =['g','b','m','c','y']

plt.bar(x=by_borough.index,height=by_borough_sorted,color=my_colors,alpha=.7)

plt.xlabel('Borough',color='r',fontsize=13)
plt.ylabel('# of Airbnb Rentals',color='r',fontsize=13)
plt.title('Airbnb Rentals by Borough',color='r',fontsize='20',fontweight='bold')

# I reset the upper y limit here again
plt.ylim(0,25000)


# ### Next, let's turn our attention to price distributions

# In[ ]:


# It would be great to break things down by borough AND room type. There are several ways to do so.
# A pivot table is a great place to start
room_pivot = pd.pivot_table(df,'price',['borough','room_type'],aggfunc=np.mean)
room_pivot


# This information will be useful in many ways. Now, let's visualize it. 

# In[ ]:


# Let's prepare some data through indexing, grouping and sorting before we plot it
avg_by_borough = df.groupby(['borough']).mean()
avg_price_by_borough_sorted= avg_by_borough['price'].sort_values(ascending=False)
avg_price_by_borough_sorted


# In[ ]:


# Here I decided to state the colors ahead of time as a named list and pass that list when plotting the bar
sns.set_style('darkgrid')

le_colors =['g','b','m','r','y']
plt.bar(x=avg_price_by_borough_sorted.index, height=avg_price_by_borough_sorted, color=le_colors)

plt.title('Average Rental Price Across Boroughs',color='r',fontsize=15,fontweight='bold')
plt.xlabel('Borough',color='r',fontsize=13)
plt.ylabel('Average Price',color='r',fontsize=13)
plt.ylim(0,250)

# i want to add the values for the bars


# ### Let's say that this wasn't enough and we wanted a more detailed breakdown of rental price distribution across the boroughs.   A boxplot proves mighty useful.

# In[ ]:


sns.set_style('darkgrid')

box_price = sns.boxplot(x='borough',y='price',data=df,palette='rainbow',showfliers=False)
box_price.set_title('Price Distribution Across Boroughs\n'+'price < 500',fontsize=15,color='r', fontweight='bold')
box_price.set_ylabel('Price',color='r', fontsize=13)
box_price.set_xlabel('-- Borough --',color='r',fontsize=13)


# ### Important note: I intentionally removed the outliers in the boxplot. Manhattan, for example, had many outliers that would have hurt the figure's visual appeal and usefulness. This will change depending on the context. 

# ### The following plot took me some time to figure out. Instead of only showing the code for the final plot, I've left several lines of code below that show several ways of extracting information through indexing, particularly using the pivot table from above. Read the code to figure out the thinking process and see how it plays into the final plot, which indexed things a little differently.

# In[ ]:


bx_bar = list(room_pivot[0:3]['price'])
bx_bar


# In[ ]:


bk_bar = list(room_pivot[3:6]['price'])
bk_bar


# In[ ]:


man_bar = list(room_pivot[6:9]['price'])
man_bar


# In[ ]:


qns_bar = list(room_pivot[9:12]['price'])
qns_bar


# In[ ]:


si_bar = list(room_pivot[12:15]['price'])
si_bar


# ### Quick hack: If you don't know how to make a plot, find an example and reverse engineer it. Here, I grabbed an example from the Matplotlib documentation and applied it to the data.

# In[ ]:


sns.set_style('darkgrid')

N = 5
entire = list(room_pivot['price'][[0,3,6,9,12]])

ind = np.arange(N)  # the x locations for the groups
width = .25       # the width of the bars

fig, ax = plt.subplots(figsize=(10,7))

rects1 = ax.bar(ind, entire, width, color='r')


private = list(room_pivot['price'][[1,4,7,10,13]])

rects2 = ax.bar(ind + width, private, width, color='y')


shared = list(room_pivot['price'][[2,5,8,11,14]])
rects3 = ax.bar(ind + width+width, shared, width, color='b')


# add some text for labels, title and axes ticks
ax.set_title('Average Price by Room Type\n'+'Across Boroughs',color='r',fontsize=15, fontweight='bold')
ax.set_ylabel('Price',color='r',fontsize=13)
ax.set_xlabel('-- Borough --',color='r',fontsize=13)
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island'))

ax.legend((rects1[0], rects2[0], rects3[0]), ('Entire Home/Apt', 'Private Room','Shared Room'))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

#Let's reset the y-axis limits to give some buffer room for that 
#tall $249-priced column
plt.ylim(0,300)


plt.show()


# ### With enough information on prices, let's explore other facets of the data. 
# 
# ### Let's say that Airbnb wanted to know the availability of listings throughout the year. One column in particular will help.

# In[ ]:


df['availability_365'].head(10)
# The column lists the amount of days that the listing was available for rental throughout the 2019 calendar yea


# In[ ]:


# One useful way of visualizing this would be to separate the values into ranges of days,
# say into 4 groups as follows:
a_availability = df[df['availability_365']<=90].count()[0]
b_availability = df[(df['availability_365']>90) & (df['availability_365']<=180)].count()[0]
c_availability = df[(df['availability_365']>180) & (df['availability_365']<=270)].count()[0]
d_availability = df[(df['availability_365']>270)].count()[0]


# In[ ]:


# With our ranges created using indexing and '&' operators, the data can be plotted in a pie chart

labels = 'Less than 90 days','Between 90 & 180 days','Between 180 & 270 days','Greater than 270 days'
sizes = a_availability,b_availability,c_availability,d_availability
explode = (.1,.1,.1,.1)

availability_pie = plt.pie(sizes,labels=labels,explode=explode,shadow=True,startangle=90,autopct='%1.1f%%',radius=1.1)
plt.title('Availability of Airbnb Rentals\n'+'As a % of the Calendar Year 2019')


# ### While pie charts don't tell us specific numbers, they're good at capturing the overall percentages. Here, we see that nearly 60% of Airbnb listings are available for less then a quarter of the year.
# 
# (Quick note: Because of the way I created the four ranges, the 4th range, "Greater   than 270 days" has five more days then the other ranges since there are 365 days in the year. )

# ### Next, what if Airbnb wanted to get some data on the top ten or twenty hosts with the most listings over the 2019 calendar year? 

# In[ ]:


df['host_id'].value_counts().head(20)


# In[ ]:


# As before, I find it mentally easier to first clarify what our x and y values will be: 
a = df['host_id'].value_counts()
top_host_values = a[0:20]
top_host_index = a[0:20].index


# In[ ]:


host_chart = sns.barplot(y=top_host_values,x=top_host_index,order=top_host_index)
sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(7,7)})


# the following line sets the contentn of the xtick labels and their rotation. 
host_chart.set_xticklabels(host_chart.get_xticklabels(), rotation=55)


host_chart.set_title('Top Hosts Across New York City',color='r',fontsize=16,fontweight='bold')
host_chart.set_ylabel('Count of listings',color='r',fontsize=14)
host_chart.set_xlabel('Host IDs',color='r',fontsize=14)
plt.ylim(0,350)


# ### Similarly, what if Airbnb wanted to find what the top ten most popular neighborhoods for rentals are across New York City?

# In[ ]:


# Let's plot the neighborhoods in nyc with the most listings
a = df['neighbourhood'].value_counts()
a


# In[ ]:


top_neighbourhood_values = a[0:10]
top_neighbourhood_values


# In[ ]:


top_neighbourhood_index = a[0:10].index
top_neighbourhood_index


# In[ ]:


# Finally, we'll use are pre-indexed x and y axis to create the plot
sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(7,5)})


neighbourhood_chart = sns.barplot(y=top_neighbourhood_values,x=top_neighbourhood_index,order=top_neighbourhood_index,alpha=.8)

# Because some of the neighborhood names were long, I found that setting the rotation less then 80 degrees distorted the position of the xtick labels. 
neighbourhood_chart.set_xticklabels(neighbourhood_chart.get_xticklabels(), rotation=85)


neighbourhood_chart.set_title('Top Neighbourhoods Across New York City',color='r',fontsize=14,fontweight='bold')
neighbourhood_chart.set_ylabel('Number of listings',color='r',fontsize=14)
neighbourhood_chart.set_xlabel('Neighbourhood',color='r',fontsize=14)


# ### Finally, let's make use of the provided latitude and longitude coordinates.

# In[ ]:


# First, plot the entire dataset using the location coordinates on the x and y axis
df.plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price',
                  cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))


# ### The result is a bit overcrowded. We can decrease the density by limiting the image to prices less then a certain threshhold. 

# In[ ]:


price_plot = df[df.price < 500]
price_plot.plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price',
                  cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))


# ### The result is more visually appealing and allows one to compare it to the following image that only maps rentals with prices above $500. 

# In[ ]:


price_plot_500 = df[df.price > 500]
price_plot_500.plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price',
                  cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))


# ### As expected from previous price visualizations, the points are concentrated in Manhattan with the second most points across the river in Brooklyn. 

# ### It's also possible to set an extra Boolean statement to narrow down to one borough.

# In[ ]:


price_plot_manhattan = df[(df.price < 500) & (df.borough=='Manhattan')]
price_plot_manhattan.plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price',
                  cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))


# ### Note: You can also download and import a background of New York City or a specific borough that matches your image size (you can configure the size easily).
# 
# ### You can also reset the image to show the locations of all listings, and instead of showing price variations, just label the points by borough.

# In[ ]:


plt.figure(figsize=(10,10))
sns.set_style('darkgrid')
sns.scatterplot(df.longitude,df.latitude,hue=df.borough)
plt.ioff()


# # ----- E N D -----

# ### I hope you've enjoyed this guide. There's always more to learn and each person will approach a given dataset from a different angle. I hope you found this walkthrough helpful.
# 
# ### #fortheloveofdatascience
# 
# Sergio A. Galeano

# In[ ]:




