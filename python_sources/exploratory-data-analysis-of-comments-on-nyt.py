#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis of the comments made on NYT articles

# The [dataset here](https://www.kaggle.com/aashita/nyt-comments) comprises of comments made on articles in New York Times in March 2018 and Jan-Feb 2017. Here we explore the features of the dataset and in particular study their relationship with the feature `recommendations` that counts the number of upvotes a comment has received. 

# # Loading data and getting to know features

# First we import python modules:

# In[25]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings 
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# Next we import the dataframe containing all the comments on New York Times articles published in March 2018.

# In[26]:


df = pd.read_csv('../input/nyt-comments/NYTCommentsMarch2018.csv')


# In[27]:


df.sample(5)


# In[22]:


df.shape


# There are 246,946 comments in total with 33 features.

# In[ ]:


df.info()


# The text of the first comment in the dataframe:

# In[23]:


df.commentBody.loc[0]


# The feature `commentTitle` is not useful as it only contains $\text{<br/>}$ values.

# In[24]:


df.commentTitle.value_counts()


#  We drop `commentTitle` along with the columns that contain only null values, as seen by the info() function above.

# In[ ]:


df.drop(['commentTitle', 'recommendedFlag', 'reportAbuseFlag', 'userURL'], axis=1, inplace=True)


# We first get the statistical summary for the categorical or string variables:

# In[ ]:


df.describe(include=['O']).transpose()


# Next numeric variables:

# In[ ]:


df.describe().transpose()


# In[ ]:


df.dtypes


# Next we look into the missing values in each column

# In[ ]:


df.isnull().sum()


# We fill the missing values:

# In[ ]:


df.parentUserDisplayName.fillna('Unknown', inplace=True)
df.sectionName.fillna('Unknown', inplace=True)
df.userDisplayName.fillna('Unknown', inplace=True)
df.userLocation.fillna('Unknown', inplace=True)
df.userTitle.fillna('Unknown', inplace=True)


# In[ ]:


df.columns


# We write the two functions that is used often:

# In[ ]:


def print_largest_values(s, n=5):
    s = sorted(s.unique())
    for v in s[-1:-(n+1):-1]:
        print(v)
    print()
    
def print_smallest_values(s, n=5):
    s = sorted(s.unique())
    for v in s[:n]:
        print(v)
    print()


# We understand the features by exploring the values they take and how do they relate to other features, and most importantly their relationship to the feature `recommendations` that counts the number of upvotes a comment has received.

#   # Understanding the distribution of upvotes on comments and removing outliers

# We plot the number of upvotes on a random selection of 500 comments and notice that the count of upvotes vary a lot.

# In[ ]:


mpl.rcParams['figure.figsize'] = (16, 8)
sns.barplot(x='commentID', y='recommendations', data=df.sample(2000)); 


# The following graph shows that the distribution of upvotes on the comments is highly skewed to the right. So, the mean of the number upvotes on comments is significantly higher than the median.

# In[ ]:


mpl.rcParams['figure.figsize'] = (10, 8)
sns.distplot(df.recommendations);


# The top 5 highest number of upvotes are all above 3000 and they can be considered as outliers.

# In[ ]:


print_largest_values(df.recommendations)


# 99% of comments have fewer than 317 upvotes and 90% of comments have 62 or fewer upvotes.

# In[ ]:


df.recommendations.quantile(0.99), df.recommendations.quantile(0.95)


# We plot the quantiles for the number of upvotes after discarding the 100th quantile.

# In[ ]:


percs = np.linspace(0,100,40)
qn = np.percentile(df.recommendations, percs)
plt.scatter(percs[:-1], qn[:-1]);


# We plot the distribution of upvotes for the comments that are in the bottom 95% in terms of upvotes:

# In[ ]:


sns.distplot(df.loc[df.recommendations<=df.recommendations.quantile(0.95), 'recommendations']);


# Keeping outliers in our dataframe will give misleading averages for the exploratory data analysis. So here we first discard them by restricting the number of upvotes to 2500.

# In[ ]:


df[df.recommendations > 2500].shape


# There are 37 such comments in total that we are discarding.

# In[ ]:


df = df[df.recommendations < 2500]


# # Extracting and analyzing data about Articles 

# First we group the dataframe by `articleID` and then derive features such as the count of comments on each article, the median number of upvotes, etc. 

# In[28]:


grouped = df.groupby('articleID')
grouped_articles = pd.concat([grouped.commentID.count(), grouped.recommendations.median(),
           grouped[['editorsSelection', 'sharing', 'timespeople', 'trusted']].mean()], 
          axis=1).reset_index().rename(columns = {'commentID': 'commentsCount'})
grouped_articles.sample(5)


# We collect other features that were unique to each article such as word count, page on which it was printed, section name, etc.

# In[29]:


articles = df[['articleID', 'articleWordCount', 'newDesk', 'printPage', 'sectionName']].copy()
articles = articles.drop_duplicates()
articles.sample(5)


# In[30]:


articles.shape, grouped_articles.shape


# We merge the two dataframes to get `articles` dataframe:

# In[ ]:


articles = articles.merge(grouped_articles, on='articleID')
articles.sample(5)


# In[ ]:


articles.describe().transpose()


# In[ ]:


articles.describe(include=['O']).transpose()


# The distribution of the number of comments on the articles is skewed right meaning a few articles have a large number of comments where as most articles don't.

# In[ ]:


mpl.rcParams['figure.figsize'] = (16, 8)
sns.distplot(articles.commentsCount);


# In[ ]:


print("Top 5 articles with the highest number of comments have the following count of comments: ")
print_largest_values(articles.commentsCount)


# The distribution of the word counts of the articles is more closer to normal distribution with a some skewness to the right:

# In[ ]:


sns.distplot(articles.articleWordCount);


# The top 5 lengthiest articles contains the following number of words

# In[ ]:


print("Top 5 lengthiest articles contains the following number of words: ")
print_largest_values(articles.articleWordCount)
print("Top 5 shortest articles contains the following number of words: ")
print_smallest_values(articles.articleWordCount)


# The distribution of the average number of upvotes on comments on the articles is far less skewed as compared to the distribution of the number of upvotes on comments themselves. The skewness is smoothen out because the mean of the number upvotes on comments is considered.

# In[ ]:


sns.distplot(articles.recommendations);


# In[ ]:


print("Top 5 articles in terms of the highest number of median upvotes on the comments have the following count of upvotes: ")
print_largest_values(articles.recommendations)
print("Top 5 articles in terms of the least number of median upvotes on the comments have the following count of upvotes: ")
print_smallest_values(articles.recommendations)


# # Analyzing different features and their relationship with the number of upvotes in a comment. 

# ## Editor's pick

# Every article on NYT that accepts comments displays a selected collection of comments, called NYT's (or Editor's) pick. 

# In[ ]:


df.editorsSelection.value_counts()


# Very few comments are selected as Editor's pick:

# In[ ]:


mpl.rcParams['figure.figsize'] = (6, 4)
sns.countplot(x="editorsSelection", data=df);


# The number of upvotes on the comments selected as Editor's pick are significantly higher on average with a mean close to 230 upvotes:

# In[ ]:


sns.barplot(x='editorsSelection', y='recommendations', data=df);


# Even though the comments that are selected as Editor's picks have more upvotes more on average, the distribution of upvotes is still highly skewed to the right. 

# In[ ]:


mpl.rcParams['figure.figsize'] = (16, 8)
sns.distplot(df.loc[df.editorsSelection==1, 'recommendations']);


# This explains the not-so-high correlation coefficient between the two features - Editor's pick and the number of upvotes:

# In[ ]:


df.editorsSelection.corr(df.recommendations)


# ## Features related to replies to comments

# ### Types of comments

# There are three types of comments - usual comments, replies made to other comments/replies and reporter's replies to a comment/reply.

# In[ ]:


df.commentType.value_counts()


# In[ ]:


mpl.rcParams['figure.figsize'] = (8, 5)
sns.countplot(x="commentType", data=df);


# In[ ]:


sns.barplot(x='commentType', y='recommendations', data=df);


# ### Depth

# The depth of 2 would mean that the comment was posted as a reply to another comment whereas a depth of 3 would mean that comment was in reply to a comment that was itself a reply. Most of the comments have a depth of 1 as shown below:

# In[ ]:


df.depth.value_counts()


# In[ ]:


sns.countplot(x="depth", data=df);


# The comments that were replies have significantly less upvotes on average than the original comments. Similarly, the nested comments have the least average number of upvotes.

# In[ ]:


mpl.rcParams['figure.figsize'] = (10, 6)
sns.barplot(x='depth', y='recommendations', data=df);


# ### Interplay between **the features - `depth` and `commentType`** in terms of number of upvotes 

# From the description above, it is clear that the features - `depth` and `commentType` are very closely related to each other. The following two graphs plot the number of upvotes taking both features into account simultaneously:

# In[ ]:


sns.barplot(x='depth', y='recommendations', hue='commentType', data=df);


# In[ ]:


sns.barplot(x='commentType', y='recommendations', hue='depth', data=df);


# ### Count of replies

# For the number of replies to a comment, the average number of upvotes on the comment is plotted below:

# In[ ]:


mpl.rcParams['figure.figsize'] = (20, 8)
sns.barplot(x='replyCount', y='recommendations', data=df);


# ## Features related to the articles of the comments

# ### Page on which the article was printed

# The printPage give the number of the page in print newspaper on which the article was published. Here, page 0 means the page for the article is unknown. 

# In[ ]:


sns.countplot(x="printPage", data=df);


# The number of comments of Page 1 may be higher because more articles from Page 1 were open to comments as shown in the graph below:

# In[ ]:


sns.distplot(articles.printPage);


# Despite the highly skewed distribution of the number of upvotes on comments from articles on various pages, the average number of upvotes received is more uniform:

# In[ ]:


sns.barplot(x='printPage', y='recommendations', data=df);


# ### Article Desk

# Number of comments in each Desk

# In[ ]:


mpl.rcParams['figure.figsize'] = (16, 16)
sns.countplot(y="newDesk", data=df, order=df.newDesk.value_counts().index);


# Average number of upvotes for the comments in each Desk with the Desks arranged in the same order as above:

# In[ ]:


mpl.rcParams['figure.figsize'] = (16, 16)
sns.barplot(y='newDesk', x='recommendations', data=df, order=df.newDesk.value_counts().index);


# First we select the top four Desks where most comments are made and then we plot the number of upvotes in a sample of 2000 comments from those Desk:

# In[ ]:


top_desk = set(df.newDesk.value_counts()[:4].index)
top_desk


# In[ ]:


sample_frequent_newDesk = df.loc[df.newDesk.isin(top_desk),
                                 ['newDesk', 'recommendations']].sample(2000)

sample_frequent_newDesk.newDesk = sample_frequent_newDesk.newDesk.astype('object')
sns.swarmplot(x='newDesk', y='recommendations', data=sample_frequent_newDesk);


# ### Section of news articles

# Number of comments in each section

# In[ ]:


sns.countplot(y="sectionName", data=df, order=df.sectionName.value_counts().index);


# Average number of upvotes for the comments in each section with the sections arranged in the same order as above:

# In[ ]:


sns.barplot(y='sectionName', x='recommendations', data=df, order=df.sectionName.value_counts().index);


# ### Word count of articles

# The median number of upvotes on comments based on the word count of articles is plotted below. Some articles are likely to attract more upvotes on most of its comments and hence the median upvotes for comments based on the word counts of the article has spikes.

# In[ ]:


mpl.rcParams['figure.figsize'] = (16, 10)
sns.barplot(x='articleWordCount', y='recommendations', data=articles, order=articles.articleWordCount);


# ## Features related to the timeline of the comments

# The three features - create date, approve date and update date follow a very similar pattern.

# In[ ]:


fig, ax = plt.subplots()
sns.distplot(df.createDate, ax=ax);
sns.kdeplot(df.createDate, ax=ax);
sns.kdeplot(df.approveDate, ax=ax);
sns.kdeplot(df.updateDate, ax=ax);


# ## Miscellaneous features

# ### Sharing

# In[ ]:


df.sharing.value_counts()


# In[ ]:


mpl.rcParams['figure.figsize'] = (6, 4)
sns.countplot(x="sharing", data=df);


# In[ ]:


sns.barplot(x='sharing', y='recommendations', data=df);


# ### Trusted

# In[ ]:


df.trusted.value_counts()


# In[ ]:


sns.countplot(x="trusted", data=df);


# In[ ]:


sns.barplot(x='trusted', y='recommendations', data=df);


# ### Timespeople

# In[ ]:


sns.countplot(x="timespeople", data=df);


# In[ ]:


sns.barplot(x='timespeople', y='recommendations', data=df);


# ### User title

# In[ ]:


df.userTitle.value_counts()


# In[ ]:


mpl.rcParams['figure.figsize'] = (12, 6)
sns.barplot(x="userTitle", y='recommendations', data=df);


# # Conclusion

# We have explored the numerical and categorical features so far using graphs and descriptive statistics. The most central features in the dataset are textual for example the `commentBody`. The next kernel that is a work in progress will be a starter kernel for model building using the textual data. Meanwhile, the [kernel here](https://www.kaggle.com/aashita/word-clouds) makes word clouds out of most common words in certain features of the dataset.
