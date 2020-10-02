#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import cufflinks as cf
import sklearn
from sklearn import svm, preprocessing 
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import os


# ## **[1] Reading Data**

# In[ ]:


df = pd.read_csv('../input/googleplaystore.csv')
df.info()


# **Observations:** Every feature is of type object, that means we'll have to change some data types and also check for NaN values. 

# In[ ]:


df = df.drop_duplicates(subset=['App'], keep = 'first')  # Removing duplicates by app names.
df.head()


# ### **Observations:** - Size and Installs feature need some preprocessing. 

# ## **[2] Feature Analysis**

# ## **[2.1] Category** 

# In[ ]:


print(df['Category'].unique())


# In[ ]:


df = df[df.Category != '1.9']


# In[ ]:


# finding which categories have more apps
df_catr = df.groupby('Category').count()['App'].sort_values(ascending = False).reset_index()


# In[ ]:


data = [go.Pie(
            values=df_catr['App'],
            labels=df_catr['Category'],
#             orientation = 'h'
)]
layout = go.Layout(
    title = 'Number of Apps in the store per Caterory', 
)
fig = go.Figure(data = data , layout = layout)
iplot(fig, filename='Pie')


# ## **[2.2] RATING**

# In[ ]:


# checking for null values. 
print(df['Rating'].isna().sum())


# In[ ]:


df['Rating'].mean()


# ### Now we can fill the Null values with the overall mean of Ratings, but as the number of Null values are very high, this can distort the original distribution. hence we'll fill the null values with the mean of their respectve Categories, so it wont be same for every row. 

# In[ ]:


df['Rating'] = df.groupby('Category').transform(lambda x: x.fillna(x.mean()))['Rating']


# In[ ]:


df['Rating'].hist()
plt.show()


# ### **Observations:** It is clear that users often dont give negative ratings as much they give positive ratings.  That is also obvious, as most of the times when we dont like an app, we directly uninstall it rather giving it a review. 

# In[ ]:


df.Rating.describe()


# In[ ]:





# ## **[2.3] Reviews**

# In[ ]:


# checking for Nan values 
df['Reviews'].isna().sum()


# In[ ]:


df['Reviews'] = pd.to_numeric(df['Reviews'])
df['Reviews'].describe()


# In[ ]:


df['Reviews'].hist()


# ### Mostly apps have very low number of reviews, but there are some apps having over 50M reviews. lets see which apps these are.

# In[ ]:


df[df['Reviews'] >= 50000000]


# ### **Observations:** So it can be seen that the max reviews apps in the store are mostly Social media/texting apps. 

# ## **[2.4] Installs**

# In[ ]:


# First we need to preprocess the installs column and convert those strings into integer values. 


# In[ ]:


df['Installs'] = df['Installs'].str.replace(',', '')
df['Installs'] = df['Installs'].str.replace('+', '')
df['Installs'] = pd.to_numeric(df['Installs'])


# In[ ]:


df['Installs'].describe()


# In[ ]:


df[df['Installs'] <100]['App'].count()


# In[ ]:


df['Installs'].count()


# In[ ]:


data = [go.Histogram(x=df['Installs'], nbinsx = 500, marker=dict(color='#10BD22'))]
layout = go.Layout(
    title = 'Distribution of number of Installs. ',
    xaxis=dict(
        title='#Installs'
    ),
    yaxis=dict(
        title='Count'
    )
)
fig = go.Figure(data = data , layout = layout)

iplot(fig, filename='histogram')


# ### So there are 20 Apps which have the maximum number of installs(over 1B), lets see which categories these apps belogn to. 

# In[ ]:


top_installed = df.sort_values(by = ['Installs'], ascending = False)[:20]


# In[ ]:


top_installed.groupby('Category')['App'].count().sort_values().plot.bar()


# ### Texting and social media apps have max number of downloads overall. 

# In[ ]:


top_installed.groupby('Content Rating')['App'].count().sort_values().plot.bar()
plt.show()


# ## **[2.5] Type**

# In[ ]:


df['Type'].unique()


# In[ ]:


df.drop(df[df['Type'].isna()].index, inplace = True)


# In[ ]:


sns.set(style="darkgrid")
sns.countplot(x="Type", data = df)
plt.show()


# In[ ]:


top_paid = df[df['Type'] == 'Paid'].groupby('Category').count()['App'].sort_values(ascending = True).reset_index()


# In[ ]:


data = [go.Bar(
            x=top_paid['App'],
            y=top_paid['Category'],
            orientation = 'h'
)]
layout = go.Layout(
    title = 'Distribution of Paid apps in Categories', 
    xaxis = dict(
        title = 'Count')
)
fig = go.Figure(data = data , layout = layout)
iplot(fig, filename='horizontal-bar')


# In[ ]:


data = [go.Histogram(x=df[df['Type'] == 'Paid']['Installs'], nbinsx = 500, marker=dict(color='#D389EC'))]
iplot(data, filename='histogram')


# In[ ]:


df[df['Type'] == 'Paid'].sort_values(by = 'Installs', ascending = False)[:20].groupby('Category').count()['App'].sort_values().plot.bar()


# ## **[2.6] Price**

# In[ ]:


df['Price'] = df['Price'].str.replace('$', '')
df['Price'] = pd.to_numeric(df['Price'])


# In[ ]:


data = [go.Histogram(x=df[df['Price']>0]['Price'],  nbinsx = 200, marker=dict(color='#37CDEA'))]
layout = go.Layout(
    title = 'Distribution of price of Paid apps', 
    xaxis = dict(
        title = 'Price in $')
)
fig = go.Figure(data = data , layout = layout)
iplot(fig, filename='histogram')


# ### Apps which have price less than 15\$ have the most number of installs, but there are some apps having much higher price (>350\****$) lets see what these are. 

# In[ ]:


df[df['Price'] > 300]


# ### On looking up, these apps are basically trolls, they have no content but exceptionally high price.The idea behind these apps is that if someone has a lot of money they will buy these apps to show that. What really interesting is that people do download these apps. smh
# 

# ## **[2.7] SIZE**

# ### We have some Sizes that varies with device and some are constant. lets see whats the distribution of fixed size anyways. 

# In[ ]:


df[df['Size'] == 'Varies with device'].count()['App']  # number of apps having variable size


# In[ ]:


fixed_size = df[df['Size'] != 'Varies with device']


# In[ ]:


# source : https://stackoverflow.com/questions/39684548/convert-the-string-2-90k-to-2900-or-5-2m-to-5200000-in-pandas-dataframe
fixed_size['Size'] = fixed_size['Size'].str.replace(r'[kM]+$', '', regex = True).astype(float) * fixed_size['Size'].str.extract(r'[\d\.]+([kM]+)', expand = False).fillna(1).replace(['k','M'], [10**0, 10**3]).astype(int)


# In[ ]:


fixed_size['Size'].describe()


# In[ ]:


data = [go.Histogram(x = fixed_size['Size'],  nbinsx = 50, marker=dict(color='#FCB461'))]
layout = go.Layout(
    title = 'Distribution of app size', 
    xaxis = dict(
        title = 'Size in KB')
)
fig = go.Figure(data = data , layout = layout)
iplot(fig, filename='histogram')


# In[ ]:


fixed_size['Size'].describe()


# In[ ]:





# In[ ]:


plt.figure(figsize=(15,5))
sns.scatterplot(x="Size", y="Installs", hue = 'Content Rating',size = 'Content Rating', data=fixed_size)
plt.show()


# ## **[2.8] Rating**

# In[ ]:


df_rating = df.groupby('Content Rating').count()['App'].reset_index()
df_rating


# In[ ]:


data = [go.Pie(
            values=df_rating['App'],
            labels=df_rating['Content Rating'],
#             orientation = 'h'
)]
layout = go.Layout(
    title = 'Number of Apps in different Content Rating', 
)
fig = go.Figure(data = data , layout = layout)
iplot(fig, filename='Pie')


# ### Most of the are for everyone, although 18+ and unrated have least number of apps. 

# ## **[3] Conclusion:** 
# 1. Family and Gaming categories dominate the play store. 
# 2. Most of the apps have average rating of 4.5
# 3. Social media and texting application have most number of downloads and reviews. 
# 4. Except some troll apps, average pricing is below 15 \$, and these apps are mainly under gaming category.
# 5. Mean app size is 20MB, number of apps significantly decrease as size decreases. 
# 6. Most of the apps are of 'Everyone' Rating. Although, 'Teen' apps have significantly high number of Installs. 
