#!/usr/bin/env python
# coding: utf-8

# # Dataset
# <br/>
# The dataset contains information about apps on the Play store. It provides verious information about the apps like their category, price, size, etc.
# <br/>
# This dataset can be used to get information about the present android market to understand the behaviour of the customer on certain apps.
# <br/>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
sns.set_style('whitegrid')


# In[ ]:


df = pd.read_csv('../input/googleplaystore.csv')
df.sample(5)


# ## Data Cleaning

# - Removing duplicate and null values.
# <br/>
# - Converting sizes to MB
# <br/>
# - Remove certain characters from the string and convert it into usable format.

# In[ ]:


#Keeping apps with type as either free or paid
df = df[(df['Type'] == 'Free') | (df['Type'] == 'Paid')]

#Removing null values
df = df[(df['Android Ver'] != np.nan) & (df['Android Ver'] != 'NaN')]

#Remove anomalies where rating is less than 0 and greater than 5
df = df[df['Rating'] < 5]
df = df[df['Rating'] > 0]

#Convert all sizes to MB and removing 'M' and 'k'
df['Size'] = df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(str(x).replace('k', ''))/1000 if 'k' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(x))

#Remove ' and up' to get the minimum android version
df['Android Ver'] = df['Android Ver'].apply(lambda x: str(x).replace(' and up', '') if ' and up' in str(x) else x)

#Remove '$' from the price to convert it to float
df['Price'] = df['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else x)
df['Price'] = df['Price'].apply(lambda x: float(x))

#Convert number of reviews to int
df['Reviews'] = df['Reviews'].apply(lambda x: int(x))


# In[ ]:


print('Number of apps in the dataset: ', len(df))


# ## App categories

# In[ ]:


trace = [go.Pie(
    values = df['Category'].value_counts(),
    labels = df['Category'].value_counts().index)]

layout = go.Layout(title = 'Distribution of app categories in the market')

fig = go.Figure(data = trace, layout = layout)
iplot(fig)


# A large segment of market is owned by family and game apps.
# <br/>
# Interestingly, medical and business apps are at 4th and 5th positions in the most installed apps category.

# ## Ratings
# Let's check out the ratings of the apps in the dataset

# In[ ]:


print('The average rating of the apps in the dataset is: ',np.mean(df['Rating']))

trace = [go.Histogram(
    x = df['Rating'],
    xbins=dict(start = 0.0, end = 5.0, size = 0.1))]

layout = go.Layout(title = 'Distribution of the ratings of apps in the dataset')

fig = go.Figure(data = trace, layout = layout)
iplot(fig)


# Most apps have fairly good rating, between 4.1 and 4.6

# ## Reviews
# Let's check the ratings of the top 100 most reviewed apps

# In[ ]:


most_reviewed = df.sort_values('Reviews', ascending=False).head(100)

print('Average rating of top 100 most reviewed apps: ',np.mean(most_reviewed['Rating']))

trace = [go.Scatter(
    x = most_reviewed['Reviews'],
    y = most_reviewed['Rating'],
    mode = 'markers')]

layout = go.Layout(title = 'Ratings of top 100 most reviewed apps')

fig = go.Figure(data = trace, layout = layout)
iplot(fig)


# The most reviewed apps generally have a rating around 4.3

# ## Size
# Let's check out the sizes of the apps in the dataset

# In[ ]:


print('The average rating of the apps in the dataset is: ',np.mean(df['Size']))

trace = [go.Histogram(x = df['Size'])]

layout = go.Layout(title = 'Distribution of the size of apps in the dataset')

fig = go.Figure(data = trace, layout = layout)
iplot(fig)


# Most apps have sizes less than 40 MB
# <br/>
# <br/>
# **Let's check out the effect of size on ratings**

# In[ ]:


sns.jointplot('Size', 'Rating', data=df)
plt.show()


# Majority of the top rated apps have sizes less than 40 MB

# ## Free and Paid apps

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,6))

ax1.hist(df[df['Type'] == 'Free']['Rating'], bins = 50)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title('Free apps rating')

ax2.hist(df[df['Type'] == 'Paid']['Rating'], bins = 50)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_title('Paid apps rating')

plt.show()


# Paid apps don't really have better rating than free apps

# In[ ]:


trace = [go.Scatter(
    x = df[df['Type'] == 'Paid']['Price'],
    y = df[df['Type'] == 'Paid']['Rating'],
    mode = 'markers')]

layout = go.Layout(title = 'Rating based on price')

fig = go.Figure(data = trace, layout = layout)
iplot(fig)


# There are a few apps with prices more than $250! Let's remove them to get a better look at the chart

# In[ ]:


trace = [go.Scatter(
    x = df[(df['Type'] == 'Paid') & (df['Price'] < 50)]['Price'],
    y = df[df['Type'] == 'Paid']['Rating'],
    mode = 'markers')]

layout = go.Layout(title = 'Rating based on price')

fig = go.Figure(data = trace, layout = layout)
iplot(fig)


# Most paid apps have prices between 0.99 and $20

# Let's look at the apps with price more than $250!

# In[ ]:


df[df['Price'] > 250][['App', 'Category', 'Price']]


# These are the spam apps intended just to earn money

# **Pricing in top 6 categories**

# In[ ]:


top_categories = df['Category'].value_counts()[:6].index
top_apps = df[df['Category'].isin(top_categories)]
top_apps = top_apps[top_apps['Price'] < 50]

trace = [go.Scatter(
    x = top_apps['Price'],
    y = top_apps['Category'],
    mode = 'markers')]

layout = go.Layout(title = 'Pricing in top 6 categories')

fig = go.Figure(data = trace, layout = layout)
iplot(fig)


# Medical apps are generally priced higher than other apps

# ## Installs

# - Installs of free apps

# In[ ]:


trace = [go.Pie(
    values = df[df['Type'] == 'Free']['Installs'].value_counts(),
    labels = df[df['Type'] == 'Free']['Installs'].value_counts().index)]

layout = go.Layout(title = 'Installs of free apps')

fig = go.Figure(data = trace, layout = layout)
iplot(fig)


# - Installs of paid apps

# In[ ]:


trace = [go.Pie(
    values = df[df['Type'] == 'Paid']['Installs'].value_counts(),
    labels = df[df['Type'] == 'Paid']['Installs'].value_counts().index)]

layout = go.Layout(title = 'Installs of paid apps')

fig = go.Figure(data = trace, layout = layout)
iplot(fig)


# Free apps have significantly more installs than paid apps.
# <br/>
# Only 3 paid apps have 10M+ installs

# ## Content Rating
# Number of installs by content rating

# In[ ]:


x = df[['Content Rating', 'Installs', 'Type']].copy()
x.dropna(inplace = True)
x = pd.DataFrame(x.groupby(['Content Rating', 'Installs'])['Type'].count())
x.reset_index(inplace=True)
x = x.pivot('Installs', 'Content Rating', 'Type')

for y in x:
    x[y] = x[y].apply(lambda z: int(str(z).replace('nan', '0')) if 'nan' in str(z) else z)

fig, ax = plt.subplots(figsize=(12,12))
ax = sns.heatmap(x, annot=True, cmap=sns.light_palette("green"), fmt='.0f')
ax.set_title('Number of installs by content rating')

plt.show()


# ## Summary

# - Categories containing the most number of apps on the play store are Family and games. Interestingly, medicine and business are catching up.
# <br/><br/>
# - Most apps recieve a good rating, typically between 4.1 and 4.6
# <br/><br/>
# - Most apps from the top 100 most reviewed apps have a very good rating.
# <br/><br/>
# - Most apps with good rating have sizes less than 40 MB. Hence, users prefer apps with sizes less than 40MB
# <br/><br/>
# - Paid apps don't have better rating than free apps. There are many paid apps that have rating less than 4.0
# <br/><br/>
# - Most paid apps cost less than \\$20. There are a few spam paid apps that cost more than \\$300
# <br/><br/>
# - Medical apps cost more than other categories.
# <br/><br/>
# - Unsurprisingly, paid apps have fewer installations than free ones. There are very few paid apps that have very high number of installs.
# <br/><br/>
# - Most apps on the play store are rated for Everyone.
