#!/usr/bin/env python
# coding: utf-8

# # YouTube Statistics Analysis
# 
# This data science project analyzes Indian YouTube statistics. 
# It was created for the UE18CS203 (B. Tech CSE third semester Introduction to Data Science course) project. A few of the charts were taken from [YouTube Trending Videos Analysis (More Than 40,000 Videos)](https://www.kaggle.com/ammar111/youtube-trending-videos-analysis).
# 
# ## Dirtying the Dataset
# 
# We need 3%-5% of the cells to be NaN. Since this dataset contains a relatively negligible number of NaN cells, we'll manually dirty it by replacing 24,000 cells (4% of the cells) with NaN. 12,000 cells from column 4 (a categorical column), and 12,000 cells from column 10 (a numerical column) will be replaced. Since there are more than 36,000 rows, we'll replace a cell in every third row.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from scipy.stats import norm

import datetime
import json
from math import sqrt

numeric_cols = 'views', 'likes', 'dislikes', 'comment_count'

    
def inform(df, msg):
    """Prints the <msg> (<str>) about null cells in the <df> (<pd.DataFrame>)."""
    print('NaN CELLS {}:\n{}\n'.format(msg, df.isna().sum()))


def write_dataset(df, name):
    """Saves the <df> (<pd.DataFrame>) with the file <name> (<str>)."""
    path = '/kaggle/working/' + name
    df.to_csv(path, index=False)
    print('Modified dataset saved to', path)


with open('/kaggle/input/youtube-new/INvideos.csv', 'r') as f:
    df = pd.read_csv(f)
inform(df, 'BEFORE DIRTYING')
for i in range(1, 36000, 3):
    df.loc[[i], df.columns[4]] = np.nan
    df.loc[[i], df.columns[10]] = np.nan
inform(df, 'AFTER DIRTYING')
write_dataset(df, 'dirty.csv')


# ## Cleaning the Dataset
# 
# All the NaN cells for categorical columns will be replaced with their previous row's value. All the NaN cells for numerical columns will be replaced with their column's average.

# In[ ]:


avg = df[df.columns[10]].mean()
for i in range(1, 36000, 3):
    df.loc[[i], df.columns[10]] = avg
df.fillna(method='ffill', inplace=True)
inform(df, 'AFTER CLEANING')
write_dataset(df, 'clean.csv')


# ## Hypothesis Testing
# 
# - Null hypothesis: There is one rating (1 like or dislike) for every 25 views.
# - Alternate hypothesis: There isn't one rating for every 25 views.

# In[ ]:


print('Null hypothesis: Trending videos receive an average of over 1 million views each')
print('Alternate hypothesis: Trending videos receive less than or equal to 1 million views each')
alpha = 0.05
x = df['views'].mean()
mu = 1000 * 1000
std = df['views'].std()
z = (x-mu) / std
print('alpha: {}\nmu: {}\nx: {}\nstd: {}\nz: {}'.format(alpha, mu, x, std, z))


# ## Normalization and Standardization
# 
# We'll normalize the numerical columns in order to make the mean 0, and the variance 1.
# 
# ### Why is normalization important?
# 
# Normalization is important because it brings all the values of numerical columns to a common scale.
# 
# ### How does normalization affect the dataset?
# 
# It affects the dataset by making all the elements lie between 0 and 1.

# Here's what the dataset looks like before normalization.

# In[ ]:


def distplot():
    sns.distplot(df['views'])

    
def hist():
    plt.hist(df['comment_count'], alpha=.3)
    sns.rugplot(df['comment_count'])
    
    
distplot()


# In[ ]:


hist()


# Here's the dataset after normalization.

# In[ ]:


def print_stats(df, is_before):
    """State whether this <is_before> (<bool>) the <df>'s (<pd.DataFrame>) modification."""
    print('{} MODIFICATION:'.format('BEFORE' if is_before else 'AFTER'))
    for col in numeric_cols:
        print('{0} mean = {1}\n{0} variance = {2}'.format(col, df[col].mean(), df[col].var()))
    print()


print_stats(df, is_before=True)
for col in numeric_cols:
    values = df[[col]].values.astype(float)
    df[col] = pd.DataFrame(MinMaxScaler().fit_transform(values))
print_stats(df, is_before=False)
write_dataset(df, 'normalized.csv')


# In[ ]:


distplot()


# In[ ]:


hist()


# In[ ]:


pd.DataFrame({col: [df[col].mean()] for col in numeric_cols}).plot(kind='bar')


# ## Insights From Visualizations
# 
# ### Which video category has the highest number of trending videos?

# In[ ]:


with open("/kaggle/input/youtube-new/IN_category_id.json") as f:
    categories = json.load(f)["items"]
cat_dict = {}
for cat in categories:
    cat_dict[int(cat["id"])] = cat["snippet"]["title"]
df['category_name'] = df['category_id'].map(cat_dict)
cdf = df["category_name"].value_counts().to_frame().reset_index()
cdf.rename(columns={"index": "category_name", "category_name": "No_of_videos"}, inplace=True)
fig, ax = plt.subplots()
_ = sns.barplot(x="category_name", y="No_of_videos", data=cdf, 
                palette=sns.cubehelix_palette(n_colors=16, reverse=True), ax=ax)
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
_ = ax.set(xlabel="Category", ylabel="No. of videos")


# 1. **Insight**
# 
#     The most popular (by far) category was _Entertainment_.
# 
#     **Explanation**
# 
#     This makes sense since YouTube started out primarily as an entertainment site.
# 
#     **Action**
# 
#     YouTubers should aim to create entertainment videos since the majority of the audience uses YouTube during their downtime.
# 1. **Insight**
# 
#     The next most popular categories were _News & Politics_, _Music_, and _Comedy_. 
# 
#     **Explanation**
# 
#     India is a boiling pot of cultures, and there is no shortage of news for avid citizens. Popular news channels such as _The Times of India_ have highly active YouTube channels where people agedr 20-60 watch regularly. Teenagers flock to YouTube for free music from around the world. It is a fact that YouTube has recognized their popularity in the music industry since they recently launced _YouTube Music_. Comedy is yet another popular category, which can be accepted as a fact since there are a great number of ads for comedians on the service.
# 
#     **Action**
# 
#     - News sites should maintain YouTube channels.
#     - Musicians should accompany their tracks with music videos.
#     - Stand-up comedians should embrace online platforms such as YouTube to gain extra revenue from their live shows.

# ### Trending Videos and Publish Time

# In[ ]:


df["publishing_day"] = df["publish_time"].apply(
    lambda x: datetime.datetime.strptime(x[:10], "%Y-%m-%d").date().strftime('%a'))
df["publishing_hour"] = df["publish_time"].apply(lambda x: x[11:13])
df.drop(labels='publish_time', axis=1, inplace=True)

cdf = df["publishing_day"].value_counts().to_frame().reset_index().rename(
    columns={
        "index": "publishing_day", 
        "publishing_day": "No_of_videos"
    }
)
fig, ax = plt.subplots()
_ = sns.barplot(
    x="publishing_day", 
    y="No_of_videos", 
    data=cdf, 
    palette=sns.color_palette(
        ['#003f5c', '#374c80', '#7a5195', '#bc5090', '#ef5675', '#ff764a', '#ffa600'],
        n_colors=7
    ), 
    ax=ax
)
_ = ax.set(xlabel="Publishing Day", ylabel="No. of videos")

cdf = df["publishing_hour"].value_counts().to_frame().reset_index()        .rename(columns={"index": "publishing_hour", "publishing_hour": "No_of_videos"})
fig, ax = plt.subplots()
_ = sns.barplot(x="publishing_hour", y="No_of_videos", data=cdf, 
                palette=sns.cubehelix_palette(n_colors=24), ax=ax)
_ = ax.set(xlabel="Publishing Hour", ylabel="No. of videos")


# 1. **Insight**
# 
#     People watch more videos during the end of the week, with the highest being on Friday. Fewer videos are watched during the beginning of the week, the lowest being on Sunday.
#     
#     **Explanation**
#     
#     People usually work more during the beginning of the week, and wind down towards the beginning of the weekend.
#     
#     **Action**
#     
#     Videos should be published right before the start of the weekend.
# 1. **Insight**
# 
#     Indians usually watch YouTube during the afternoon or morning.
#     
#     **Explanation**
#     
#     A lot of news is watched during the morning, as well as entertainment videos while traveling to college.
#     
#     **Action**
#     
#     Live video streams should be done during the afternoons or mornings so as to to engage with the maximum number of users.

# ### How many trending videos have their comments or ratings disabled?

# In[ ]:


value_counts = df["comments_disabled"].value_counts().to_dict()
fig, ax = plt.subplots()
_ = ax.pie(x=[value_counts[False], value_counts[True]], labels=['No', 'Yes'], 
           colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'})
_ = ax.axis('equal')
_ = ax.set_title('Comments Disabled?')

value_counts = df["ratings_disabled"].value_counts().to_dict()
fig, ax = plt.subplots()
_ = ax.pie([value_counts[False], value_counts[True]], labels=['No', 'Yes'], 
            colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'})
_ = ax.axis('equal')
_ = ax.set_title('Ratings Disabled?')


# 1. **Insight**
# 
#     Disabling comments or ratings severely impacts the ability of videos to become popular.
#     
#     **Explanation**
#     
#     Viewers feel that videos with their ratings or comments disabled are untrustworthy.
#     
#     **Action**
#     
#     Have comments and ratings enabled on all your videos.
# 1. **Insight**
# 
#     Videos with disabled ratings or comments occasionally become popular.
#     
#     **Explanation**
#     
#     Although videos with disabled comments or ratings have a significantly lower chance of becoming popular, we still see such videos occasionally becoming popular. This can be attributed to the fact that controversial videos have their ratings or comments disabled later on. Many a times videos are viewed in large numbers simply because of the absurdity of their content, and not their actual value to the customer.
#     
#     **Action**
#     
#     Do not disable comments or ratings on your video to prevent the potential for negative publicity. Videos which have shown otherwise are an exception, not the rule.
#     
# ## Correlation

# In[ ]:


df.corr(method='pearson')


# In[ ]:


df.plot.scatter(x='views', y='likes')
df.plot.scatter(x='views', y='dislikes')
df.plot.scatter(x='views', y='comment_count')


# We can see that the `views` and `likes` columns are correlated the most: the higher the number of views, the higher the number of likes.
# From this, we can **infer** that better videos get viewed more.

# In[ ]:


df.plot.scatter(x='comment_count', y='views')
df.plot.scatter(x='comment_count', y='likes')
df.plot.scatter(x='category_id', y='dislikes')


# We can see that the `category_id` and `dislikes` columns have the least correlation. 
# The **reason** behind this is because certain categories of videos (e.g., entertainment) get significantly more views. Hence, the ratings such as dislikes varies widely across categories of videos.
