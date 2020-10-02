#!/usr/bin/env python
# coding: utf-8

# # WiT Data Science!

# # Useful Data Science imports:
# 
# * **numpy** is a package for math things, including linear algebra which we will be using. It's also a base for things like pandas and other libraries
# * **pandas** is for data processing and handling things like data input/output. It can handle CSV, Excel, and even pulling from a web page
# 
# * **matplotlib** is for making graphs, it is a base for things like seaborn and other more complex image creation
# * **seaborn** makes very pretty graphs
# * **cufflinks** makes other very pretty graphs
# * **plotly** makes interactive graphs! neato!
# 
# You can import things as you need them, so you are only importing things you need, or do what I do and just import everything at the beginning. 
# 
# This first block of code was created by Kaggle when I started a brand new kernel and starts with some useful things. I've added several imports:

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: 
# https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra and other math things
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # graphing more complicated things
import matplotlib.pyplot as plt  # more graph things

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) 
# will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # numpy: linear algebra using arrays
# ## Slicing (and dicing) to find the data you need within an array
# 
# First we just need an array to play with, so just run this cell:

# In[ ]:


my_array = [[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24]]


# To select a row type the row number in brackets after the array name.

# In[ ]:


my_array[1]


#   This selected the *2nd* row. Why? Indices in Python start at 0 so the first row would be 0 and the 2nd row is 1, etc. Same things for the column.
#   
#   Select a specific data value:  The first number is the row, the 2nd is the column. Let's try to get the number 8 out of our array:

# In[ ]:


my_array[1][3]


# There are plenty of other ways to slice and dice numerical data, but what do we do about more complicated things? Let's move on to a more powerful tool. The main take-away from this section is that **arrays start at an index of 0** (so the first row is index 0, the second is 1, etc.)

# # Pandas and DataFrames
# ## A powerful library for dealing with all kinds of data
# 
# Let's just read in the first file:

# In[ ]:


listings = pd.read_csv('/kaggle/input/berlin-airbnb-data/listings_summary.csv')


# ## Let's use some ways to look at the data.  
# 
# Now that we have it all in a variable we can use what are called methods on it. To do this, append a method to what you're trying to use the method on. The method must have parentheses (some methods take additional information, which would go inside those parentheses).

# In[ ]:


# This gives us the first 5 rows by default. 
# You can enter a number in the parentheses to give you more or less.
# This is a great way to get a quick look at your data -- with some caveats
listings.head()

# You can look at the last rows by using .tail() instead:
#listings.tail()

# Take a minute to look over the information below. 
# Is there anything interesting to you?  
# Is there anything that you have questions about?


# There are some columns we're not seeing. Can we find out what all the columns are?

# In[ ]:


listings.columns


# In[ ]:


# Let's check out a text field. 
# Remember we just wanted to select a column? 
# Super easy with a data frame:
listings['city'].head()


# In[ ]:


# More useful let's find out how many unique ones there are:
listings['city'].unique()


# In[ ]:


# Neat. How many listings in each of those cities?
listings['city'].value_counts()


# Oh now that's interesting - look how many are the same thing but listed slightly differently. Will this be a problem for what you're trying to do?

# In[ ]:


# Let's check out a numerical field. One thing we can do are look at the 
# distribution:
listings['bedrooms'].plot.hist()


# In[ ]:


#What if I want some statistics?
listings.bedrooms.describe()

# NOTE: Be careful with .describe() because it will try even if it doesn't 
# make sense with the data!


# In[ ]:


# What happens if we try to describe a text field?

listings.neighbourhood.describe()


# In[ ]:


# How can I tell if a field is in fact numerical? One way might just be
# to look at the header. 
# If it doesn't show up in the full list if I know the column name 
# I can easily look at it by adding it as an index. Notice that DataFrames
# assume you are looking for a column unless you specify you want a row.
listings['review_scores_rating'].head()


# ## Q: Pick a numerical field and do some basic looking into it:
# Write your code in the cell below. Here are some ideas for the "requires license" field:
# 
# listings['requires_license'].head()
# 
# listings['requires_license'].tail()
# 
# listings['requires_license'].describe()
# 
# listings['requires_license'].nunique() tells me how many unique values (e.g. 2 in this case)
# 
# listings['requires_license'].unique()  gives me an array containing the unique values in a column
# 
# NOTE: This notebook will only show you the last thing in a cell, you will need to have separate cells if you want to see more than one result. To add another cell, hover the mouse below a cell and click "code" (Markdown is for text).

# In[ ]:


# Write your code here. 


# In[ ]:


# Write more code here. 


# I'm curious about the "requires license' field. How many are true and how many are false?  We can count the values:

# In[ ]:


listings['requires_license'].value_counts()


# Which ones don't require a license? We can look at them (only 8) using the following block of code:

# In[ ]:


listings[listings['requires_license']=='f']


# 
# It might not make sense but you can think of it like this, working your way from inside to the outside:
# 
# You want listings where the 'requires_license' column is false. To get that column, you need listings['requires_license']=='f'. 
# 
# Then you want the listings of that column, hence wrapping listings[] around the column statement. 
# 
# You can have all kinds of logic inside! But if it gets too complex it might get a little hard to read and you might want to define a function to call instead.

# # Processing the data
# ## How do I go about "fixing" some of the problems to make things easier to use?
# * Do I really want all of these columns? Are there some I don't care about?
# * Deal with missing values?
# * Change 't' and 'f' to 1 and 0?
# 
# ## Dropping columns entirely

# In[ ]:


# Who needs the URL? Let's just drop it, right?
listings.drop('listing_url', axis=1)  # axis=1 means a column


# BAM! GONE! Right? Oh no what if I need it???

# In[ ]:


listings.head()


# BUT WAIT IT IS STILL ACTUALLY THERE.  
# 
# This is important: data frames never delete data unless you make them do it. If you REALLY want to drop something, you need **inplace=True** inside the method. What's something we *actually* don't care about? license maybe?

# In[ ]:


listings.drop('license', axis=1, inplace=True)  # True must be capitalized


# In[ ]:


listings.head()  # all gone

# You can drop rows too if you want (axis=0) but let's not worry about that now.


# # Q: What is a field you might want to drop?
# Try it here:

# In[ ]:





# I'm going to do a couple of quick things to reduce the amount of data we're dealing with. Just run the next cell. (You can change the columns you keep if you want.)

# In[ ]:


columns_of_interest = ['id','host_has_profile_pic','host_since','neighbourhood_cleansed', 'neighbourhood_group_cleansed',
                   'host_is_superhost','description',
                   'latitude', 'longitude','is_location_exact', 'property_type', 'room_type', 'accommodates', 'bathrooms',  
                   'bedrooms', 'bed_type', 'amenities', 'price', 'cleaning_fee',
                   'review_scores_rating','reviews_per_month','number_of_reviews',
                   'review_scores_accuracy','review_scores_cleanliness','review_scores_checkin',
                   'review_scores_communication','review_scores_location','review_scores_value',
                   'security_deposit', 'extra_people', 'guests_included', 'minimum_nights',  
                   'instant_bookable', 'is_business_travel_ready', 'cancellation_policy','availability_365']

df = listings[columns_of_interest].set_index('id')


# ## Fixing the 't' and 'f' parts:
# Let's try to fix the "Is Location Exact" column

# In[ ]:


# Are there any NaNs in there?
df['is_location_exact'].isna().sum()


# In[ ]:


#Ok cool. There are a bunch of ways to make 't' and 'f' something 
# useful (we'll use 1 and 0) but let's try a map:
df['is_location_exact'] = df['is_location_exact'].map({'f':0,'t':1})


# In[ ]:


df['is_location_exact'].unique()


# # Q: What are some other fields you might want to try to change to 0 and 1?

# In[ ]:


# Try one here:


# ## Wow there is a lot of cleaning to do! Dang! Now what?
# 
# # Those pesky "Not A Number" fields -- missing data. 
# ## This is the hard part because you have to make some real choices.

# In[ ]:


# This will give us a count of how many in each column are missing (null, aka NaN)
df.isnull().sum()


# # Q: Discuss: What would make sense to fill the following missing data:
# * host_has_profile_pic 
# * bedrooms
# * bathrooms

# # Q: Discuss: Find two fields with missing values that don't make sense to do anything with. Why?

# In[ ]:





# All right I've cleaned up the data. Run the next four cells:

# In[ ]:


# These three additional columns just need to replace 'f' and 't' with 0 and 1
df['host_is_superhost'] = df['host_is_superhost'].map({'f':0,'t':1})
df['is_business_travel_ready'] = df['is_business_travel_ready'].map({'f':0,'t':1})
df['instant_bookable'] = df['instant_bookable'].map({'f':0,'t':1})


# In[ ]:


# This column needs to first replace the NaN with 'f' and then replace the 'f' and 't' with 0 and 1.
df['host_has_profile_pic'].fillna('f',inplace=True)
df['host_has_profile_pic'] = df['host_has_profile_pic'].map({'f':0,'t':1})
df['host_has_profile_pic'].value_counts()


# In[ ]:


# Let's make a quick bar plot of that profile pic column:
sns.countplot(x='host_has_profile_pic',data=df)


# In[ ]:


# Gotta clean up those prices, remove the $ and , (replace with empty string) 
# and then turn them from strings into numbers:
df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
df['cleaning_fee'] = df['cleaning_fee'].str.replace('$', '').str.replace(',', '').astype(float)
df['security_deposit'] = df['security_deposit'].str.replace('$', '').str.replace(',', '').astype(float)
df['extra_people'] = df['extra_people'].str.replace('$', '').str.replace(',', '').astype(float)


# The above four need some kind of special treatment:
# 
# # Q: What should we do with these values?  Replace with...?
# 
# Possible choices:
# * mean (average)
# * median
# * min
# * max
# * something specifc
# * something *calculated*

# In[ ]:


# Examples, either run this or replace with your own choice
df['cleaning_fee'].fillna(df['cleaning_fee'].median(), inplace=True)
df['security_deposit'].fillna(df['security_deposit'].mean(), inplace=True)
df['extra_people'].fillna(df['extra_people'].min(), inplace=True)


# Let's look at the distribution of prices, basically a different way of doing a histogram using seaborn:

# In[ ]:


sns.distplot(df['price'], kde=True)


# In[ ]:


df['price'].isnull().sum()


# In[ ]:


df['price'].describe()


# Might be interesting to see if prices vary based on room type

# In[ ]:


sns.countplot(x='room_type',data=df)


# Oh cool not too many variants of room types. Maybe I'm not interested in anything over a price of 250. I'm going to make a couple more plots selecting for < 250 (remember that bit earlier where I selected specific things? I can add logic!). First just that selection and then also adding if the room_type is 'Entire home/apt'. I want to plot the price, so that goes in the second set of []. Then I'll make a 3rd plot where I select the 'Private room' option instead.

# In[ ]:


sns.distplot(df[df['price']<250]['price'], kde=True)


# In[ ]:


sns.distplot((df[(df['price']<250) & (df['room_type']=='Entire home/apt')]['price']), kde=True)


# In[ ]:


sns.distplot((df[(df['price']<250) & (df['room_type']=='Private room')]['price']), kde=True)


# Or we could try to do this all on one graph! Check it out:

# In[ ]:


g = sns.FacetGrid(data=df,col='room_type')
g.map(plt.hist,'price', range=(0,200))


# I'm going to borrow a function quick to calculate distance from the center of Berlin, and add a column to the data frame with that information.
# https://www.kaggle.com/kpriyanshu256/airbnb-price has this calculation and some more fun ideas (including some machine learning at the end).
# 
# Then a few more fun graphs!
# 
# Feel free to just run the rest of the notebook:

# In[ ]:


from math import sin, cos, sqrt, atan2, radians

def haversine_distance_central(row):
    berlin_lat,berlin_long = radians(52.5200), radians(13.4050)
    R = 6373.0
    long = radians(row['longitude'])
    lat = radians(row['latitude'])
    
    dlon = long - berlin_long
    dlat = lat - berlin_lat
    a = sin(dlat / 2)**2 + cos(lat) * cos(berlin_lat) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

df['distance'] = df.apply(haversine_distance_central,axis=1)


# In[ ]:


sns.jointplot(x='distance',y='price',data=df,kind='scatter', xlim=(0,15), ylim=(0,250))


# In[ ]:


df['distance'].plot.hist()


# In[ ]:


sns.heatmap(df.corr(),cmap='coolwarm')


# # So now I'm going to turn you loose and let you play with the whole thing.
# If you need something to think about what might you do with the zero prices? What questions might you want to ask about the data? You can get pretty complex if you want even if you don't know how to solve it!  See what you can discover. If you're interested in some other graphs you can try:
# 
# **Scatter plots:**
# 
# df.plot.scatter(x='a',y='b',c='red',s=50,figsize=(12,3))  (this is a plot of y vs x, where 'a' and 'b' are the column names.
# 
# **Basic histograms**
# 
# plt.style.use('ggplot')   (this prettifies the regular histogram a little. Or you can use the seaborn (sns) one I used above (distplot)
# df['a'].plot.hist(alpha=0.5,bins=25)
# 
# df3['a'].plot.kde()  Plots a smoothed curve instead of the usual bins for a histogram. This is like the kde overlaid on the seaborn plot above.
# 
# **Box plots**
# 
# df[['a','b']].plot.box()  (Makes a box plot of columns 'a' and 'b')
# 
# **Area plots**
# 
# df.ix[0:30].plot.area(alpha=0.4)   An area plot of all columns (you might want to be picky about choosing fewer. Extra bonus: the .ix[0:30] just plots the first 30 rows. 

# **Other ways to fill NaN**
# 
# df.fillna(value='FILL VALUE')
# 
# df['a'].fillna(value=df['a'].mean())

# In[ ]:




