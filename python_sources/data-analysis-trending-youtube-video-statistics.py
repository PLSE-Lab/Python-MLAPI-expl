#!/usr/bin/env python
# coding: utf-8

# <b>Inspiration</b><br>
# Possible uses for this dataset could include:<br>
# 
# - Sentiment analysis in a variety of forms<br>
# - Top 10 most ____ videos <br>
# - Categorising YouTube videos based on their comments and statistics.<br>
# - Relationships in the dataset
# - Statistical analysis over time.<br>
# - Analysing what factors affect how popular a YouTube video will be.<br>

# In this analysis, only the data from the US is used.

# In[ ]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[ ]:


from pylab import *
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import datetime


# In[ ]:


sns.set()


# In[ ]:


# import data

df = pd.read_csv("../input/youtube-new/USvideos.csv", )


# # Sentimental analysis using WordCloud

# ## w.r.t. 'title'

# In[ ]:


# preprocess
df["title"] = df["title"].apply(lambda s: s.lower())
text = '\n'.join(df["title"].tolist())
word_cloud_img = WordCloud(width=400*4, height=200*4, collocations=False).generate(text)

# plot
plt.figure(figsize=(20, 8))
plt.imshow(word_cloud_img)
plt.axis("off");


# - many titles contain words such as 'official', 'trailer', 'video', 'new', 'vs', 'makeup', 'audio', 'music', 'ft', ...

# ## w.r.t. 'tags'

# In[ ]:


# preprocess
df["tags"] = df["tags"].apply(lambda s: s.lower())
text = '\n'.join(df["tags"].tolist())
word_cloud_img = WordCloud(width=400*4, height=200*4, collocations=False).generate(text)

# plot
plt.figure(figsize=(20, 8))
plt.imshow(word_cloud_img)
plt.axis("off");


# - many tags contain words such as 'new', 'video', 'funny', 'makeup', 'music', 'movie', 'yotube', 'comedy', 'trailer', ...
# - Seeing title and tag categories, movie trailers and makeup videos are very popular.

# ## w.r.t. 'description'

# In[ ]:


# preprocess
df["description"] = df["description"].apply(lambda s: str(s).lower())
text = '\n'.join(df["description"].tolist())
word_cloud_img = WordCloud(width=400*4, height=200*4, collocations=False).generate(text)

# plot
plt.figure(figsize=(20, 8))
plt.imshow(word_cloud_img)
plt.axis("off");


# - many descriptions contain words such as 'https', 'facebook', 'twitter', 'instagram', ...
# - people tend to put some links or their SNS accounts (facebook ,twitter, instagram) in the description section.

# # Data Cleaning

# In[ ]:


# remove columns of 'video_id', 'thumbnail_link', 'category_id'
# because they are not really useful

df = df.drop(columns=["video_id", "category_id", "thumbnail_link"])


# In[ ]:


# check if there's any Nan or rows with typos

# ...


# In[ ]:


df["comments_disabled"] = df["comments_disabled"].apply(lambda comment: int(comment))
df["ratings_disabled"] = df["ratings_disabled"].apply(lambda comment: int(comment))


# In[ ]:


def cvt_date(sdate):
    year, day, month = re.findall(r"(\d+).(\d+).(\d+)", sdate)[0]
    year = '20' + year
    year, day, month = int(year), int(day), int(month)
    return datetime.datetime(year=year, month=month, day=day)

df["trending_date"] = df["trending_date"].apply(lambda sdate: cvt_date(sdate))


# In[ ]:


def cvt_pubtime(pubtime):
    year, month, day = re.findall(r"(\d+)-(\d+)-(\d+)", pubtime)[0]
    year, month, day = int(year), int(month), int(day)
    return datetime.datetime(year=year, month=month, day=day)

df["publish_time"] = df["publish_time"].apply(lambda pubtime: cvt_pubtime(pubtime))


# # Top 10 most ____ videos

# In[ ]:


def print_top_videos(category: str):
    df_temp = df[["title", category]].sort_values(by=category, ascending=False)
    
    titles = []
    rank = 1
    for title, views in df_temp.values:
        if title not in titles:
            titles.append(title)
            print("Rank {:<5} | title: {:<70} | {}: {:<20}".format(rank, title, category, views))  # printing alignment
            rank += 1
            if rank == 11:
                break


# ## top 10 most viewed videos

# In[ ]:


print_top_videos("views")


# ## top 10 most liked videos

# In[ ]:


print_top_videos("likes")


# ## top 10 most disliked videos

# In[ ]:


print_top_videos("dislikes")


# # Categorising YouTube videos based on their comments and statistics

# ## Comment count

# In[ ]:


sns.distplot(df.comment_count, bins=300, kde=False);
plt.xlim(-100, 75000);


# ## Categorizing using K-Means

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


df_temp = df[["title", "comment_count"]]
kmeans = KMeans(n_clusters=3, random_state=1).fit(df_temp[["comment_count"]])
df_temp["labels"] = kmeans.labels_

def get_group(label):
    return df_temp[df_temp["labels"] == label]


# ### label group: 0

# In[ ]:


get_group(0)


# In[ ]:


sns.distplot(get_group(0)["comment_count"], bins=10, kde=False);
plt.xlim(-100, 750000);


# - In the label group 0, videos with low 'comment_count' are grouped

# ### label group: 1

# In[ ]:


get_group(1)


# In[ ]:


sns.distplot(get_group(1)["comment_count"], bins=30, kde=False);
plt.xlim(-100, 750000);


# - In the label group 1, videos with medium 'comment_count' are grouped

# ### label group: 1

# In[ ]:


get_group(2)


# In[ ]:


sns.distplot(get_group(2)["comment_count"], bins=90, kde=False);
plt.xlim(-100, 750000);


# - In the label group 1, videos with high 'comment_count' are grouped

# # Relationships in the dataset

# ## relation btn 'views', 'likes', 'dislikes', 'comment_count'

# In[ ]:


# Correlation matrix

corr = df[["views", "likes", "dislikes", "comment_count"]].corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[ ]:


sns.pairplot(df[["views", "likes", "dislikes", "comment_count"]], height=1.5, 
             plot_kws={"linewidth": 0, "alpha": 0.2});


# - the more 'view', the more 'likes' (very strong relationship)
# - the more 'view', the more 'dislikes' (not as much as 'likes' tho)
# - the more 'likes', the more 'dislikes'
# - the more 'views', 'likes', and 'dislikes', the more 'comment_count'

# ## relation btn 'comments_disabled', 'ratings_disabled' and 'views'

# In[ ]:


counter = Counter(df["comments_disabled"])
df_temp = pd.DataFrame({"category": list(counter.keys()), "count": list(counter.values())})
df_temp.plot.pie(y="count")
plt.legend(['comment_abled', 'comment_disabled']);

counter = Counter(df["ratings_disabled"])
df_temp = pd.DataFrame({"category": list(counter.keys()), "count": list(counter.values())})
df_temp.plot.pie(y="count")
plt.legend(['ratings_abled', 'ratings_disabled']);


# - there are small numbers of videos with 'comment_disabled' and 'ratings_diabled' in the dataset

# In[ ]:


sns.catplot("comments_disabled", "views", data=df, col="ratings_disabled",
            alpha=0.2, height=4);


# - 'comments_disabled' leads to a lower viewership
# - 'ratings_disabled' leads to a lower viewership
# -  when both 'comments_disabled' and 'ratings_disabled' are applied, 'views' is very low

# ## realtion between 'publish_time' and others

# In[ ]:


plt.hist(df.publish_time, bins=50);
plt.xlabel("publish_time")
plt.ylabel("count");


# - most of 'publish_time' are in 2018
#  

# In[ ]:


plt.figure(figsize=(10, 3))
sns.scatterplot(x="publish_time", y="trending_date", data=df, 
                linewidth=0, alpha=0.2);
plt.ylim(datetime.datetime(2017, 1, 1), )
plt.xlim(datetime.datetime(2006, 1, 1), );


# - In 2018, 'publish_time' is quite proportional to 'trending_date'

# In[ ]:


plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(df.publish_time, df.views, 'o', alpha=0.2)
plt.xlim(datetime.datetime(2017, 1, 1), )
plt.xlabel("publish_time")
plt.ylabel("views");

plt.subplot(2, 1, 2)
plt.plot(df.publish_time, df.likes, 'o', alpha=0.2)
plt.xlim(datetime.datetime(2017, 1, 1), )
plt.xlabel("publish_time")
plt.ylabel("likes")

plt.tight_layout();


# - as the year passes, there are more videos with much more likes. 
# - It may tell that more and more people are using Youtube.

# In[ ]:


plt.figure(figsize=(15, 3))
y_jitter = np.random.uniform(-1, 1, size=df.comments_disabled.size) / 20
plt.plot(df.publish_time, df.comments_disabled+y_jitter, 'o', alpha=0.5)
plt.xlim(datetime.datetime(2017, 1, 1), )
plt.xlabel("publish_time")
plt.ylabel("comments_disabled");

plt.figure(figsize=(15, 3))
y_jitter = np.random.uniform(-1, 1, size=df.ratings_disabled.size) / 20
plt.plot(df.publish_time, df.comments_disabled+y_jitter, 'o', alpha=0.5)
plt.xlim(datetime.datetime(2017, 1, 1), )
plt.xlabel("publish_time")
plt.ylabel("ratings_disabled");


# - there was (almost) no video with 'comments_disabled' and 'ratings_disabled'
# - as there are much more videos in 2018 than before, there are more videos with 'comments_disabled' and 'ratings_disabled'
# <br><br>
# 
# - I suppose many people don't really like being JUDGED by ratings or comments in 2018

# # Analysing what factors affect how popular a YouTube video will be.

# ## Proprocess the dataset

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[ ]:


# preprocess the dataset

def convert2unixtime(date_time):
    """Convert datetime to unixtime"""
    unixtime = date_time.timestamp()
    return unixtime

df2 = df.copy()

df2["trending_date"] = df2["trending_date"].apply(lambda d: convert2unixtime(d))
df2["publish_time"] = df2["publish_time"].apply(lambda d: convert2unixtime(d))

df2 = df2.drop(columns=["title", "channel_title", "tags", "description"])  # drop unprocessable columns


# In[ ]:


# split data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df2[["trending_date", "publish_time", "likes", "dislikes", "comment_count","comments_disabled", "ratings_disabled"]], 
                                                    df2[["views"]], 
                                                    test_size=0.2, random_state=1)


# ## Build ML model (RandomForest)

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
rfr.fit(X_train, y_train)


# ## Test

# In[ ]:


print("r-squared score on training dataset: {:0.3f}".format(rfr.score(X_train, y_train)))
print("r-squared score on test dataset: {:0.3f}".format(rfr.score(X_test, y_test)))


# ## Feature importance analysis

# In[ ]:


df_temp = pd.DataFrame({'category': list(X_train.columns), 'feature_importance': rfr.feature_importances_})
df_temp = df_temp.sort_values(by="feature_importance", ascending=False)

sns.catplot("category", "feature_importance", data=df_temp, kind="bar", aspect=2);


# - most viewed videos are affected by 'likes' the most
