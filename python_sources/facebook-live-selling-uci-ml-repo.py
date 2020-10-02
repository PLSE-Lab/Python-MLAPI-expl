#!/usr/bin/env python
# coding: utf-8

# ## Introduction.

# * Facebook Live has recently become a popular direct selling medium, notably in East and South-East Asia.
# * This feature offers small, self-employed sellers unseen level of consumer reach and involvement and is reinventing direct selling.
# * The data was extracted from Thai Facebook sellers selling fashion and beauty, other popular products online.

# ## Preliminary Steps.

# **Import Libraries.**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from IPython.display import display
import os


# **Some Notebook Settings.**

# In[ ]:


warnings.filterwarnings('ignore') # ignore warnings.
get_ipython().run_line_magic('config', 'IPCompleter.greedy = True # autocomplete feature.')
pd.options.display.max_rows = None # set maximum rows that can be displayed in notebook.
pd.options.display.max_columns = None # set maximum columns that can be displayed in notebook.
pd.options.display.precision = 2 # set the precision of floating point numbers.


# **Read Data.**

# In[ ]:


df = pd.read_csv('../input/Live.csv', encoding='utf-8')
df.drop_duplicates(inplace=True) # drop duplicates if any.
df.shape # num rows x num columns.


# <hr>

# ## Data Preparation.

# Check for missing values.

# In[ ]:


miss_val = (df.isnull().sum()/len(df)*100).sort_values(ascending=False)
miss_val[miss_val>0]


# Remove columns with missing values.

# In[ ]:


df.drop(labels=['Column1', 'Column2', 'Column3','Column4'], axis=1, inplace=True)


# Let's take a look at the data.

# In[ ]:


df.head()


# ID column is no longer required.

# In[ ]:


df.drop('status_id', axis=1, inplace=True)


# Let's convert `status_type` to a 0-1 categorical column, where 0 is photo and 1 is video.

# In[ ]:


df['status_type_isvideo'] = df['status_type'].map(lambda x:1 if(x=='video') else 0)
df.drop('status_type', axis=1, inplace=True)


# `status_published` has a string data type. Let's convert it to datetime.

# In[ ]:


df['status_published'] = pd.to_datetime(df['status_published'])


# Let's derive year, month, weekday, hour from the `status_published` column.

# In[ ]:


df['year'] = df['status_published'].dt.year
df['month'] = df['status_published'].dt.month
df['dayofweek'] = df['status_published'].dt.dayofweek # 0 is Monday, 7 is Sunday.
df['hour'] = df['status_published'].dt.hour


# <hr>

# In[ ]:


reaction = ['num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas',
            'num_sads', 'num_angrys'] # reaction of users.


# Since Facebook Live wwas launched in August 2015, Let's split the data on Year, before and after 2016.

# In[ ]:


before2016 = df[df['year']<=2015]
after2016 = df[df['year']>2015]


# Let's compare reactions before and after facebook live.

# In[ ]:


before2016[reaction].describe()


# In[ ]:


after2016[reaction].describe()


# <hr>

# ## EDA.

# Let's analyse data before 2016. We'll try to visualise the data using PCA.

# In[ ]:


before2016.groupby('status_type_isvideo')[reaction].mean()


# The reactions and likes are almost double if content is video. Comments and shares are also higher for video content, though only slightly.

# In[ ]:


sns.heatmap(before2016[reaction].corr(), cmap='coolwarm', annot=True)


# Shares and Comments have good correlation.

# <hr>

# In[ ]:


from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
before2016_s = before2016[reaction]
before2016_s = standard_scaler.fit_transform(before2016_s) # s in before2016_s stands for scaled.


# In[ ]:


# Improting the PCA module.

from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized', random_state=123)


# In[ ]:


# Doing the PCA on the data.
pca.fit(before2016_s)


# In[ ]:


# Making the screeplot - plotting the cumulative variance against the number of components

fig = plt.figure(figsize = (10,5))

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

plt.show()


# In[ ]:


# what percentage of variance in data can be explained by first 2,3 and 4 principal components respectively?
(pca.explained_variance_ratio_[0:2].sum().round(3),
pca.explained_variance_ratio_[0:3].sum().round(3),
pca.explained_variance_ratio_[0:4].sum().round(3))


# In[ ]:


# we'll use first 2 principal components to visualise feature importance.

loadings = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'Feature':reaction})
loadings


# In[ ]:


# we can visualize what the principal components seem to capture.

fig = plt.figure(figsize = (6,6))
plt.scatter(loadings.PC1, loadings.PC2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
for i, txt in enumerate(loadings.Feature):
    plt.annotate(txt, (loadings.PC1[i],loadings.PC2[i]))
plt.tight_layout()
plt.show()


# The num_wows, hahas, angry, sad, loves are all 0 on both the principal components as they aren't available features before 2016. The first principal componet shows that likes, reactions are more important features, then comments and then shares.
# The second principal component shows that num_shares are important and then comments.
# <br>It seems that first principal component captures the non-video theme. And, second principal component capture the video theme.

# <hr>

# In[ ]:


before2016.groupby('year').sum()[reaction]


# In[ ]:


before2016.groupby('year').sum()[reaction].plot(figsize=(12,5))


# Over the years, before 2016 and for photo content, reactions and likes have seen steady increase.

# In[ ]:


before2016.groupby(['year', 'status_type_isvideo']).sum()[reaction]


# In[ ]:


plt.figure(1)
before2016[before2016['status_type_isvideo']==0].groupby('year').sum()[reaction].plot(
    figsize=(10,5), title='photo content')

plt.figure(2)
before2016[before2016['status_type_isvideo']==1].groupby('year').sum()[reaction].plot(
    figsize=(10,5), title='video content')


# Likes and reactions increased for photo content from 2014 to 2015.<br>
# And for video content, they were high in the year of 2014.

# In[ ]:


plt.figure(1)
before2016[before2016['status_type_isvideo']==0].groupby('month').sum()[reaction].plot(
    figsize=(10,5), title='photo content')

plt.figure(2)
before2016[before2016['status_type_isvideo']==1].groupby('month').sum()[reaction].plot(
    figsize=(10,5), title='video content')


# Both for photo and video content, likes and reactions seem to show a wavy pattern.<br>
# Although trend is decreasing for video content, and for photo, it seems winters also recieve less likes and reactions.

# In[ ]:


plt.figure(1)
before2016[before2016['status_type_isvideo']==0].groupby('dayofweek').sum()[reaction].plot(
    figsize=(10,5), title='photo content')

plt.figure(2)
before2016[before2016['status_type_isvideo']==1].groupby('dayofweek').sum()[reaction].plot(
    figsize=(10,5), title='video content')


# Fridays have most likes for photo content.<br>
# And, Sundays and Saturdays have most likes for video content.

# In[ ]:


plt.figure(1)
before2016[before2016['status_type_isvideo']==0].groupby('hour').sum()[reaction].plot(
    figsize=(10,5), title='photo content')

plt.figure(2)
before2016[before2016['status_type_isvideo']==1].groupby('hour').sum()[reaction].plot(
    figsize=(10,5), title='video content')


# For photo content, 10 pm peaks in terms of likes, and night time 1 am to 4-5 am also sees maximum likes.<br>
# For video content, 12 pm to 5 am sees maximum amount of likes.

# <hr>

# Let's analyse data after 2016. We'll try to visualise the data using PCA.

# In[ ]:


after2016.groupby('status_type_isvideo')[reaction].mean()


# The reaction (hahas, comments, shares) are greater for the video content. They are thus more engaging than photos.

# In[ ]:


sns.heatmap(after2016[reaction].corr(), cmap='coolwarm', annot=True)


# shares, comments and loves have good correlation.<br>
# reactions, loves, shares and likes have good correlation as well.

# <hr>

# In[ ]:


from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
after2016_s = after2016[reaction]
after2016_s = standard_scaler.fit_transform(after2016_s) # s in before2016_s stands for scaled.


# In[ ]:


# Improting the PCA module.

from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized', random_state=123)


# In[ ]:


# Doing the PCA on the data.
pca.fit(after2016_s)


# In[ ]:


# Making the screeplot - plotting the cumulative variance against the number of components

fig = plt.figure(figsize = (10,5))

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

plt.show()


# In[ ]:


# what percentage of variance in data can be explained by first 2,3 and 4 principal components respectively?
(pca.explained_variance_ratio_[0:2].sum().round(3),
pca.explained_variance_ratio_[0:3].sum().round(3),
pca.explained_variance_ratio_[0:4].sum().round(3))


# In[ ]:


# we'll use first 2 principal components to visualise feature importance.

loadings = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'Feature':reaction})
loadings


# In[ ]:


# we can visualize what the principal components seem to capture.

fig = plt.figure(figsize = (6,6))
plt.scatter(loadings.PC1, loadings.PC2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
for i, txt in enumerate(loadings.Feature):
    plt.annotate(txt, (loadings.PC1[i],loadings.PC2[i]))
plt.tight_layout()
plt.show()


# Likes, reactions and wows are similar and important as they are clusterd together and their loadings arer high in both the principal components. See top right corner of the graph.<br>
# Loves, shares have high loadings on first principal component, alongwith likes, reactions.<br>
# Hahas, angrys, and sads sort of weigh low on both the principal components.<br>
# As, we had established earlier that more shares and comments are the indicators of video content, thus it seems that the first principal component captures the video theme and the second principal component captures the photo theme.

# In[ ]:


after2016.groupby('year').sum()[reaction]


# In[ ]:


after2016.groupby('year').sum()[reaction].plot(figsize=(12,5))


# Over the years, all the reactions have seen increase in reactions, with the most prominent being comments. Before 2016, it used to be likes and num_reactions.

# In[ ]:


after2016.groupby(['year', 'status_type_isvideo']).sum()[reaction]


# In[ ]:


plt.figure(1)
after2016[after2016['status_type_isvideo']==0].groupby('year').sum()[reaction].plot(
    figsize=(10,5), title='photo content')

plt.figure(2)
after2016[after2016['status_type_isvideo']==1].groupby('year').sum()[reaction].plot(
    figsize=(10,5), title='video content')


# This can be observed again, photo content has most likes, video content has most comments and shares.

# In[ ]:


plt.figure(1)
after2016[after2016['status_type_isvideo']==0].groupby('month').sum()[reaction].plot(
    figsize=(10,5), title='photo content')

plt.figure(2)
after2016[after2016['status_type_isvideo']==1].groupby('month').sum()[reaction].plot(
    figsize=(10,5), title='video content')


# Summers see maximum likes for photo content. Wavy pattern that was earlier observed for year before 2016 is not present here. Lowest likes are in the months of July, September.<br>
# May,September and December see peaks in comments for video contentlier observed for year less than is present here as well.

# In[ ]:


plt.figure(1)
after2016[after2016['status_type_isvideo']==0].groupby('dayofweek').sum()[reaction].plot(
    figsize=(10,5), title='photo content')

plt.figure(2)
after2016[after2016['status_type_isvideo']==1].groupby('dayofweek').sum()[reaction].plot(
    figsize=(10,5), title='video content')


# Wednesdays and Sundays see maximum amount of likes for photo content, although the likes seem fairly constant throught th week.<br>
# Saturdays see the maximum amount of comments, and similar to trend observed in photo content, video content also sees constant trend throught the week in terms of all reactions.

# In[ ]:


plt.figure(1)
after2016[after2016['status_type_isvideo']==0].groupby('hour').sum()[reaction].plot(
    figsize=(10,5), title='photo content')

plt.figure(2)
after2016[after2016['status_type_isvideo']==1].groupby('hour').sum()[reaction].plot(
    figsize=(10,5), title='video content')


# Form late night to morning, 5-8 am, the likes are maximum for photo content.
# <br>For video content, 1 am late night and 8 am in the morning sees maximum peaks in shares and comments. It is fair to assume, thats when the live selling happens. 

# <hr>
