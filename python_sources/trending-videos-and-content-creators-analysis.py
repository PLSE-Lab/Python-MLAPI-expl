#!/usr/bin/env python
# coding: utf-8

# **Trending, time series, and correlation data analysis of Youtube trending videos dataset.**
# 
# ** My first data analysis project presented at BootCampSpot (Data Analysis 2019)** 
# 
# **Inspirations:**
# + YouTube Trending Videos Project
#     https://www.kaggle.com/datasnaek/youtube-new
#     
# + Alexis Gardin Analysis
#     https://www.kaggle.com/alexisgardin/youtube-video-analysis-si4/report
#     
# + Yanni Papadakis Analysis
#     https://www.kaggle.com/yanpapadakis/trending-youtube-video-metadata-analysis    
#     
# ____________

# > ### Dependencies
# ______

# In[ ]:


import pandas as pd
import json
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import numpy as np
import statsmodels.api as sm

#nltk.download('punkt')
#nltk.download('stopwords')

from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from scipy import stats

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


columns =  ['views', 'likes', 'dislikes', 'comment_count', 'trending_date', 'category_name', 'duration', 
            'tags', 'tags_rate', 'title', 'channel_title', 'tags_positive', 'tags_neutral', 'tags_negative',
            'title_rate', 'title_positive', 'title_neutral', 'title_negative', 'publish_time']

trends = pd.read_csv('../input/youtube-new/USvideos.csv', parse_dates=['publish_time'], index_col='video_id')
duration = pd.read_csv('../input/extras/USvideos_duration.csv',usecols=['video_id', 'duration'], index_col='video_id')
sentimental = pd.read_csv('../input/extras/US_sentimental.csv', index_col='video_id')
creators = pd.read_csv('../input/extras/USchannels.csv', index_col='video_id',)

df = pd.merge(trends, duration, how='outer', on=['video_id'])
df = df.merge(sentimental, how='outer', on=['video_id'])

df.head(5)


# ### Data Cleanup Process
# ________________
# 

# In[ ]:


# import category description
category_file = '../input/youtube-new/US_category_id.json'

map_category = {}
with open(category_file) as jsonfile:
    categories = json.load(jsonfile)
    
for item in categories['items']:
    map_category[int(item['id'])] = item["snippet"]["title"]

df['category_name'] = df['category_id'].map(map_category)
df['category_name'] = df['category_name'].astype('category')

df.dropna(inplace=True)

df.info()


# In[ ]:


# fix trendind_date field
df['trending_date'] = df['trending_date'].apply(lambda dt: datetime.datetime.strptime(dt, '%y.%d.%m'))

df.info()

#df['trending_date'].head(10)
#dt='17.14.11'

#print (datetime.datetime.strptime(dt, '%y.%d.%m'))



# In[ ]:


# Considering only the last record for each video for the summary analysis. 
df_unique = df[columns].sort_values(by='trending_date',ascending=False).groupby(by='video_id').first()

df_unique.head(5)


# In[ ]:


# adding tag length column
df_unique['tags_length'] = df_unique['tags'].map(lambda tag: len(tag.split('|')))
df_unique.head(5)


# In[ ]:


### Remove NaN values from creators Dataframe

creators.dropna(inplace=True)


# #### Data Analysis Process
# ____________

# #### Summary Information
# > ______________
# High-level summary information related to videos trending dataset.

# In[ ]:


total_videos = df_unique.shape[0]

max_views = df_unique['views'].max()
sum_views = df_unique['views'].sum()
avg_views = df_unique['views'].mean()
med_views = df_unique['views'].median()
min_views = df_unique['views'].min()
std_views = df_unique['views'].std()
trend_min = df_unique['trending_date'].min()
trend_max = df_unique['trending_date'].max()
channels  = df_unique['channel_title'].nunique()

summary_data = [{
    'Videos'             : total_videos,
    'Channels'           : channels,
    'Max of Views'       : max_views,    
    'Average of Views'   : avg_views,
    'Median of Views'    : med_views,
    'Minimum of Views'   : min_views,
    'Standard Deviation' : std_views,
    'Start date'         : trend_min,
    'End date'           : trend_max,
    'Total of Views'     : sum_views
}]

videos_summary = pd.DataFrame(data=summary_data, columns=list(summary_data[0].keys()))
videos_summary['Videos'] = videos_summary['Videos'].map('{:,}'.format)
videos_summary['Channels'] = videos_summary['Channels'].map('{:,}'.format)
videos_summary['Max of Views'] = videos_summary['Max of Views'].map('{:,}'.format)
videos_summary['Total of Views'] = videos_summary['Total of Views'].map('{:,}'.format)
videos_summary['Average of Views'] = videos_summary['Average of Views'].map('{:,.0f}'.format)
videos_summary['Median of Views'] = videos_summary['Median of Views'].map('{:,.0f}'.format)
videos_summary['Minimum of Views'] = videos_summary['Minimum of Views'].map('{:,.0f}'.format)
videos_summary['Standard Deviation'] = videos_summary['Standard Deviation'].map('{:,.0f}'.format)

videos_summary


# **Observation :** The dataset provides a total of 6.351 videos and 2.198 channels. The timeframe of the dataset covers around eleven months (between Nov-2017 until June-2018). The dataset has the potential to connect our efforts in the marketing campaign with almost 2M people(views) on average. A standard deviation of 7M in additional with the enormous difference between the average and the median gives us the inside how spread out the dataset is. The maximum number of views also give us another inside for potential outlier(s) inside of the dataset. The total number of views is almost 38 times higher than US current population.

# #### Univariate Views Distribution Versus Quanties Distribution
# ______
# In-deep analysis of the dataset[](http://)

# In[ ]:


# Views distribution

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 1, 1)

sns.set_style("dark")
sns.set(font_scale=0.9)

ax = sns.distplot(a=np.log10(df_unique['views'].values),label='Number of Views',hist=True, kde=True, rug=False, bins=100,ax=ax)

ax.set_title('Univariate Views Distribution', fontsize=15)
ax.set_xlabel('Views\n(log10 base)')
ax.set_ylabel('Observations')
ax.set_yscale('linear')
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)

#plt.savefig('../../../../resources/images/plot_001.png')
plt.tight_layout()
plt.show()


# In[ ]:


quanties = [0.01, 0.10, 0.20, 0.30, 0.40, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
series_quanties = df_unique['views'].quantile(quanties)

record = {}

for item in series_quanties.items():    
    record[str(int(item[0]*100)) + '%'] = f'{item[1]:,.0f}'
    
pd.DataFrame(data=[record], columns=list(record.keys())) 


# **Observation :** Comparing the probability distribution of the total number of views versus the quantiles split in a range of 10%, we can start predicting some more realistic potential numbers of people that the marketing campaign might achieve using Youtube. Something to notice is the fact the from 70% to 80%, the difference between the total of numbers of views is minimum. Another important fact is 99% of the views represents a total of 25M. Therefore, a total of 225M views is an estimate that should be considered as an outlier.

# #### Trending Videos Analysis over Time
# __________
# Time-Series Analysis of Trending Videos behaviors.

# In[ ]:


sns.set(font_scale=0.9)

ts_trend = df_unique['trending_date'].value_counts().reset_index().rename(columns={'index' : 'Date', 'trending_date' : 'Videos'})
ts_trend['rate_change'] = ts_trend['Videos'].diff()

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)


ax1 = sns.lineplot(x=ts_trend['Date'], y=ts_trend['Videos'], color='blue', ax=ax1)
ax2 = sns.lineplot(x=ts_trend['Date'], y=ts_trend['rate_change'], color='darkred', ax=ax2)

ax1.set_xlim(auto=True)
ax1.set_ylim(auto=True)

ax1.set_title('Total of Trending Video Over Time', fontsize=15)
ax1.set_xlabel('Trending Date')
ax1.set_ylabel('Total of Videos')

ax2.set_xlim(auto=True)
ax2.set_ylim(auto=True)

ax2.set_title('Rate of Change Over Time', fontsize=15)
ax2.set_xlabel('Trending Date')
ax2.set_ylabel('Total of Videos')

plt.tight_layout()
#plt.savefig('../../../../resources/images/plot_002.png')
plt.show()


# #### Days Taken for Video Classification
# _______
# 

# In[ ]:


df_publish = df_unique[['trending_date', 'publish_time']].copy(deep=True)
df_publish['time_taken'] = (df_publish['trending_date'] - df_publish['publish_time']).dt.days
cond = df_publish['time_taken'] > 0

df_publish_summary = df_publish[cond].groupby(by='trending_date', as_index=False).agg({
                                                                                    'time_taken' : ['min', 'mean', 'max']
                                                                                })
fig = plt.figure(figsize=(20, 5))
sns.set(font_scale=0.8)

ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)


ax1 = sns.lineplot(x=df_publish_summary['trending_date'], y=df_publish_summary['time_taken']['min'], color='yellow', ax=ax1)
ax1.set_xlim(auto=True)
ax1.set_ylim(auto=True)

ax1.set_title('Minimum Tike Taken', fontsize=15)
ax1.set_xlabel('Trending Date')
ax1.set_ylabel('Days')

ax2 = sns.lineplot(x=df_publish_summary['trending_date'], y=df_publish_summary['time_taken']['mean'], color='blue', ax=ax2)
ax2.set_xlim(auto=True)
ax2.set_ylim(auto=True)

ax2.set_title('Average Tike Taken', fontsize=15)
ax2.set_xlabel('Trending Date')
ax2.set_ylabel('Days')

ax2 = sns.lineplot(x=df_publish_summary['trending_date'], y=df_publish_summary['time_taken']['max'], color='red', ax=ax3)
ax2.set_xlim(auto=True)
ax2.set_ylim(auto=True)

ax2.set_title('Max Tike Taken', fontsize=15)
ax2.set_xlabel('Trending Date')
ax2.set_ylabel('Days')

plt.tight_layout()
#plt.savefig('../../../../resources/images/plot_003.png')
plt.show()


# **Observation :** Once the status of a video change from publishing to trending, it means that the potential to reach more people increase on an exponential scale. Observing this behavior can give us insights to plans for deployment dates for the marketing campaign. A higher number of trending videos, higher are the chances to reach more people. No trending videos at all mean that we need to find new ways to reach people. January, April, and May seems to be months of more activity for trending videos. It's important also to observe the rate of change of the trending video. The activity tends to remains constant over time. The numbers of days required for a video change the status from published to trended in December through February in average is huge. There is definitely something causing this phenomenon for these months. January presents two behaviors: a higher number of trending videos and higher time taken for video status change.

# #### Views per Category
# __________
# Identity the performance of the categories areas.

# In[ ]:


df_category = df_unique[['category_name', 'views']].groupby(by='category_name', as_index=False).agg({'views' : ['count', 'sum']})

fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

sns.set(font_scale=0.7)

ax1.bar(df_category['category_name'], df_category['views']['sum'],align="center", alpha=0.6, orientation='vertical',log=True)
ax1.set_title('Max Views per Category', fontsize=15)
ax1.set_xlabel('Category')
ax1.set_ylabel('Views')
ax1.set_xticklabels(df_category['category_name'], rotation='vertical')
ax1.set_xlim(auto=True)
ax1.set_ylim(auto=True)


ax2.pie(x=df_category['views']['count'], labels=df_category['category_name'], shadow=True, startangle=90, autopct='%1.1f%%',labeldistance=1.0,pctdistance=0.7)
ax2.set_title('Total of Videos Per Category', fontsize=15)
ax2.axis('equal')
ax2.set_xlim(auto=True)
ax2.set_ylim(auto=True)

plt.tight_layout()
#plt.savefig('../../../../resources/images/plot_004.png')
plt.show()


# **Observation :** Music and entertainment categories dominated the total number of views but comparing these two categories with the total number of trending videos, the game change. Even though the Entertainment category has a higher number of trending videos, this fact does not help in making this category to dominate also the total of views.

# #### Views Distribution per Category
# ______
# Identity how the numbers of views are distributed across the categories.

# In[ ]:


fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(1, 1, 1)

ax = sns.boxplot(x=df_unique['category_name'], y=df_unique['views'], width=0.5,palette="colorblind", data=df_unique,showfliers=True,showmeans=True,ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_xlim(auto=True)
ax.set_ylim(auto=True)

ax.set_title('Number of Views per Category Distribution', fontsize=15)
ax.set_xlabel('Category')
ax.set_ylabel('Views')
ax.set_yscale('log')


plt.tight_layout()
#plt.savefig('../../../../resources/images/plot_005.png')
plt.show()


# **Observation :** Analyzing the distribution for each category, we can observe that Shows presents what can be considered a normal distribution. Music and Entertainment and Film & Animations categories present a considerable number of outliers. Therefore, predicting data for these categories might be difficult due to the presence of a higher number of outliers.

# Correlation between metrics
# ______________________
# Observe the correlation between Views, Like, Dislikes, Comments, Duration.

# In[ ]:


columns = ['views', 'likes', 'dislikes', 'comment_count', 'duration', 'tags_length']
df_corr = df_unique[columns].corr()

fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(1, 1, 1)

ax.set_title('Correlation coefficient between Views, Likes, Dislikes, Comments, Duration\nCoefficient Range : [-1.0 to 1.0]', fontsize=15)

ax = sns.heatmap(df_corr, annot=True, fmt='.2f',vmin=-1.0, vmax=1.0, cmap='YlGnBu', center=0, linewidths=.5)

plt.tight_layout()
#plt.savefig('../../../../resources/images/plot_012.png')

plt.show()


# **Observation:** Views and Likes presents are a strong correlation which was expected. Not surprisingly the correlation between views and comments is not so strong due to the fact that not the majority of the viewers make comments of a video. Another not so strong correlation is between Views and Dislikes which tells us that might have more factors involved that make a person dislike a video. The biggest surprise in this correlation analysis is the weaker correction between Views and Video Duration. The result was unexpected by all members of the group. All additional studies that were performed by the group indicated us a strong correlation between Views and Video duration but this hypothesis proved to be wrong at least for trending videos.

# #### Ratios per Categories
# ________________
# Ratios across several metrics.

# In[ ]:


df_ratio = df_unique.groupby(by='category_name',as_index=False).agg({
    'likes'         : ['sum'],
    'dislikes'      : ['sum'],
    'views'         : ['sum'],
    'comment_count' : ['sum']
})

fig = plt.figure(figsize=(15, 10))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)


ax1.bar(df_ratio['category_name'], df_ratio['likes']['sum'] / df_ratio['dislikes']['sum'],align="center", alpha=0.6, orientation='vertical',log=False)
ax1.set_title('Likes - Dislikes Ratio', fontsize=15)
ax1.set_xlabel('Category')
ax1.set_ylabel('Likes / Dislikes')
ax1.set_xticklabels(df_ratio['category_name'], rotation='vertical')
ax1.set_xlim(auto=True)
ax1.set_ylim(auto=True)


ax2.bar(df_ratio['category_name'], df_ratio['dislikes']['sum'] / df_ratio['views']['sum'],align="center", alpha=0.6, orientation='vertical',log=True, color='r')
ax2.set_title('Dislikes - Views Ratio', fontsize=15)
ax2.set_xlabel('Category')
ax2.set_ylabel('Disikes / Views')
ax2.set_xticklabels(df_ratio['category_name'], rotation='vertical')
ax2.set_xlim(auto=True)
ax2.set_ylim(auto=True)

ax3.bar(df_ratio['category_name'], df_ratio['likes']['sum'] / df_ratio['views']['sum'],align="center", alpha=0.6, orientation='vertical',log=True, color='g')
ax3.set_title('Likes - Views Ratio', fontsize=15)
ax3.set_xlabel('Category')
ax3.set_ylabel('Likes / Views')
ax3.set_xticklabels(df_ratio['category_name'], rotation='vertical')
ax3.set_xlim(auto=True)
ax3.set_ylim(auto=True)

ax4.bar(df_ratio['category_name'], df_ratio['views']['sum'] / df_ratio['comment_count']['sum'],align="center", alpha=0.6, orientation='vertical',log=False, color='y')
ax4.set_title('Views - Comments Ratio', fontsize=15)
ax4.set_xlabel('Category')
ax4.set_ylabel('Views / Comments')
ax4.set_xticklabels(df_ratio['category_name'], rotation='vertical')
ax4.set_xlim(auto=True)
ax4.set_ylim(auto=True)

plt.tight_layout()
#plt.savefig('../../../../resources/images/plot_006.png')
plt.show()


# **Observation:** The ratio between Likes and Dislikes presents the expected results. Pets & Animals and Shows categories present the higher ratios scores. News & Politics and Nonprofits & Blogs categories present the lowest ratios scores. Another interesting ratio result can be observed in between Views and Comments. Autos & Vehicles, Travel & Events, and Shows categories presents higher ratio scores. Nonprofits & Activism and News & Politics present the lowest ratios score. Higher Likes and Dislikes ratios can be used to measure people satisfaction. More Likes can bring more Views. Higher Views and Comments ratios can be used to explore more direct communication with your audiences. Hearing your audiences opinion is crucial for the business.

# #### Used Tags
# _______________
# Top keywords (tags) used per video category.

# In[ ]:


fig = plt.figure(figsize=(25, 12))
stopwords_list = list(stopwords.words('english'))

def normalize_data(tag):    
    tag_word = tag.str.lower().str.cat(sep=' ')
    tag_word = re.sub('[^A-Za-z]+', ' ', tag_word)
    tokens = word_tokenize(tag_word)
    filter_sentence = [word for word in tokens if not word in stopwords_list]
    filter_one_chr = [word for word in filter_sentence if len(word) > 2]
    
    return [word for word in filter_one_chr if not word.isdigit()]
    
def plot_word_cloud(word_data, x, y, i, title):
    ax = fig.add_subplot(x, y, i)
    
    cloud = WordCloud(background_color = 'white', max_words = 100,  max_font_size = 50)
    cloud.generate(' '.join(word_data))
    
    ax.imshow(cloud,interpolation='bilinear')
    ax.set_title(label=title, fontsize=15)
    ax.axis('off')

index = 1

for category in df_unique['category_name'].unique():    
    cond = df_unique['category_name'] == category
    plot_word_cloud(normalize_data(df_unique[cond]['tags']), 4,4, index, category)
    index = index + 1
    
plt.tight_layout()
#plt.savefig('../../../../resources/images/plot_007.png')
plt.show() 


# **Observation:** According to YouTube, tagging is one of the most important ways to rank your video in YouTube search results: Tags help users find your video when they search the site. When users type keywords related to your tags your video will appear in their search results. Therefore, knowing the most used tags per category can provide excellent insights in order to achieve a higher number of people. Some keywords across the categories were expected such as dog or cat for Pets & Animals categories or Fornite and Pokemon for Gaming categories. Discovering the keyword iPhone as one of the most used keywords for the Shows category still remains a secret for the group members.

# #### Sentimental Analysis
# ________
# Sentimental analysis for videos titles and tags across categories.

# In[ ]:


df_sentimental = df[['category_name','tags_rate', 'views']].groupby(by=['tags_rate', 'category_name'], as_index=False).count()

cond1 = df_sentimental['tags_rate'] == 'Positive'
cond2 = df_sentimental['tags_rate'] == 'Neutral'
cond3 = df_sentimental['tags_rate'] == 'Negative'

df_positive = df_sentimental[cond1][['category_name', 'views']] 
df_positive.set_index(keys=['category_name'], inplace=True)
df_positive.fillna(value=0, inplace=True)

df_neutral = df_sentimental[cond2][['category_name', 'views']] 
df_neutral.set_index(keys=['category_name'], inplace=True)
df_neutral.fillna(value=0, inplace=True)

df_negative = df_sentimental[cond3][['category_name', 'views']] 
df_negative.set_index(keys=['category_name'], inplace=True)
df_negative.fillna(value=0, inplace=True)

df_total = df_positive + df_neutral + df_negative

df_total['perc_positive'] = (df_positive['views'] / df_total['views'])*100
df_total['perc_neutral'] = (df_neutral['views'] / df_total['views'])*100
df_total['perc_negative'] = (df_negative['views'] / df_total['views'])*100
df_total.reset_index(drop=False, inplace=True)

fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(1, 2, 1)

ax1.bar(df_total['category_name'], df_total['perc_positive'], align="center", width=0.8, alpha=0.6, orientation='vertical',log=False, label='Positive', color='b')
ax1.bar(df_total['category_name'], df_total['perc_neutral'], align="center", width=0.8, alpha=0.6, orientation='vertical',log=False, label='Neutral', color='y', bottom=df_total['perc_positive'])
ax1.bar(df_total['category_name'], df_total['perc_negative'], align="center", width=0.8, alpha=0.6, orientation='vertical',log=False, label='Negative', color='r', bottom=df_total['perc_positive']+df_total['perc_neutral'])

ax1.set_title('Sentimental Analysys of Videos Tags', fontsize=15)
ax1.set_xlabel('Category')
ax1.set_ylabel('Sentimental Count')
ax1.set_xticklabels(df_total['category_name'], rotation='vertical')
ax1.set_xlim(auto=True)
ax1.set_ylim(auto=True)
ax1.legend()


df_sentimental_title = df[['category_name','title_rate', 'views']].groupby(by=['title_rate', 'category_name'], as_index=False).count()

cond1 = df_sentimental_title['title_rate'] == 'Positive'
cond2 = df_sentimental_title['title_rate'] == 'Neutral'
cond3 = df_sentimental_title['title_rate'] == 'Negative'

df_positive_title = df_sentimental_title[cond1][['category_name', 'views']] 
df_positive_title.set_index(keys=['category_name'], inplace=True)
df_positive_title.fillna(value=0, inplace=True)

df_neutral_title = df_sentimental_title[cond2][['category_name', 'views']] 
df_neutral_title.set_index(keys=['category_name'], inplace=True)
df_neutral_title.fillna(value=0, inplace=True)

df_negative_title = df_sentimental_title[cond3][['category_name', 'views']] 
df_negative_title.set_index(keys=['category_name'], inplace=True)
df_negative_title.fillna(value=0, inplace=True)

df_total_title = df_positive_title + df_neutral_title + df_negative_title

df_total_title['perc_positive'] = (df_positive_title['views'] / df_total_title['views'])*100
df_total_title['perc_neutral'] = (df_neutral_title['views'] / df_total_title['views'])*100
df_total_title['perc_negative'] = (df_negative_title['views'] / df_total_title['views'])*100
df_total_title.reset_index(drop=False, inplace=True)

ax2 = fig.add_subplot(1, 2, 2)

ax2.bar(df_total_title['category_name'], df_total_title['perc_positive'], align="center", width=0.8, alpha=0.6, orientation='vertical',log=False, label='Positive', color='b')
ax2.bar(df_total_title['category_name'], df_total_title['perc_neutral'], align="center", width=0.8, alpha=0.6, orientation='vertical',log=False, label='Neutral', color='y', bottom=df_total_title['perc_positive'])
ax2.bar(df_total_title['category_name'], df_total_title['perc_negative'], align="center", width=0.8, alpha=0.6, orientation='vertical',log=False, label='Negative', color='r', bottom=df_total_title['perc_positive']+df_total_title['perc_neutral'])

ax2.set_title('Sentimental Analysys of Videos Title', fontsize=15)
ax2.set_xlabel('Category')
ax2.set_ylabel('Sentimental Count')
ax2.set_xticklabels(df_total['category_name'], rotation='vertical')
ax2.set_xlim(auto=True)
ax2.set_ylim(auto=True)
ax2.legend()

plt.tight_layout()
#plt.savefig('../../../../resources/images/plot_008.png')
plt.show()


# **Observation:** Examining the sentimental used by videos titles and tags per category can provide us directions regarding how to make to a selected product association with a specific channel or category. Howto & Style and Pets & Animals categories present higher positives score for tags (higher than 60%). In another hand, Nonprofits & Activism category presents the higher positive score but this score is below 50%. This initial analysis brings the following question: Is there any correction between titles and tags sentiments with the number of views?

# #### Correlation between Title Sentiment and Views
# __________________

# In[ ]:


cond = df_unique['title_rate'] == 'Positive'
df1 = df_unique[cond][['title_positive','views', 'title_rate']]
df1.rename(columns={ 'title_positive' : 'score', 
                     'title_rate' : 'sentiment'
                   },inplace=True)

cond = df_unique['title_rate'] == 'Neutral'
df2 = df_unique[cond][['title_neutral','views', 'title_rate']]
df2.rename(columns={ 'title_neutral' : 'score', 
                     'title_rate' : 'sentiment'
                   },inplace=True)

cond = df_unique['title_rate'] == 'Negative'
df3 = df_unique[cond][['title_negative','views', 'title_rate']]
df3.rename(columns={ 'title_negative' : 'score', 
                     'title_rate' : 'sentiment'
                   },inplace=True)

sns.set(font_scale=2)
df_regplot = pd.concat([df1, df2, df3])

kws = dict(s=50, linewidth=.5, edgecolor="w")

ax = sns.lmplot(x='score', y='views', col='sentiment', data=df_regplot, aspect=1.3, height=8,
                legend=True, hue='sentiment', markers='+',col_wrap=3)

ax.set(xlim=(0, 101))
ax.set(xscale='linear', yscale='log')
#plt.savefig('../../../../resources/images/plot_013.png')


plt.tight_layout()
plt.show()


# In[ ]:


# y={0:.1f}x+{1:.1f}

summary_data = []

slope, intercept, r_value, p_value, std_err = stats.linregress(x=np.log10(df1['score']), y=np.log10(df1['views']))
summary_data.append({'Score': 'Positive', 'Slope' : slope, 'Y-Intercept': intercept})

slope, intercept, r_value, p_value, std_err = stats.linregress(x=np.log10(df2['score']), y=np.log10(df2['views']))
summary_data.append({'Score': 'Neutral', 'Slope' : slope, 'Y-Intercept': intercept})

slope, intercept, r_value, p_value, std_err = stats.linregress(x=np.log10(df3['score']), y=np.log10(df3['views']))
summary_data.append({'Score': 'Negative', 'Slope' : slope, 'Y-Intercept': intercept})


pd.DataFrame(data=summary_data, columns=list(summary_data[0].keys()))


# #### OLS Regression for Positive Sentiment

# In[ ]:


model = sm.OLS(endog=np.log10(df1['views']), exog=np.log10(df1['score']))
fit = model.fit()
fit.summary()


# #### OLS Regression for Neutral Sentiment

# In[ ]:


model = sm.OLS(endog=np.log10(df2['views']), exog=np.log10(df2['score']))
fit = model.fit()
fit.summary()


# #### OLS Regression for Negative Sentiment

# In[ ]:


model = sm.OLS(endog=np.log10(df3['views']), exog=np.log10(df3['score']))
fit = model.fit()
fit.summary()


# **Observation:** As we can see, the title sentiment has a strong correlation. As the positive or negative score increases, the number of views also increase. R-squared score presents a result of 0.97 indicating a strong relationship between the two variables. Another really good indicator is the p-value score which is below 0.001. It means that the results were highly significant. Or, in other words, they're very unlikely to have occurred by chance alone.

# #### Correlation between Tags Sentiment and Views
# _______

# In[ ]:


cond = df_unique['tags_rate'] == 'Positive'
df1 = df_unique[cond][['tags_positive','views', 'tags_rate']]
df1.rename(columns={ 'tags_positive' : 'score', 
                     'tags_rate' : 'sentiment'
                   },inplace=True)

cond = df_unique['tags_rate'] == 'Neutral'
df2 = df_unique[cond][['tags_neutral','views', 'tags_rate']]
df2.rename(columns={ 'tags_neutral' : 'score', 
                     'tags_rate' : 'sentiment'
                   },inplace=True)

cond = df_unique['tags_rate'] == 'Negative'
df3 = df_unique[cond][['tags_negative','views', 'tags_rate']]
df3.rename(columns={ 'tags_negative' : 'score', 
                     'tags_rate' : 'sentiment'
                   },inplace=True)

sns.set(font_scale=2)
df_regplot = pd.concat([df1, df2, df3])

kws = dict(s=50, linewidth=.5, edgecolor="w")

ax = sns.lmplot(x='score', y='views', col='sentiment', data=df_regplot, aspect=1.3, height=8,
                legend=True, hue='sentiment', markers='+',col_wrap=3)

ax.set(xlim=(0, 101))
ax.set(xscale='linear', yscale='log')


plt.tight_layout()
#plt.savefig('../../../../resources/images/plot_014.png')

plt.show()


# In[ ]:


# y={0:.1f}x+{1:.1f}

summary_data = []

cond = (df1['score'] > 0) & (df1['views'] > 0)
slope, intercept, r_value, p_value, std_err = stats.linregress(x=np.log10(df1[cond]['score']), y=np.log10(df1[cond]['views']))
summary_data.append({'Score': 'Positive', 
                     'Slope' : slope, 
                     'Y-Intercept': intercept,
                     'P-Value' : p_value,
                     'Standard Error' : std_err
                    })

cond = (df2['score'] > 0) & (df2['views'] > 0)
slope, intercept, r_value, p_value, std_err = stats.linregress(x=np.log10(df2[cond]['score']), y=np.log10(df2[cond]['views']))
summary_data.append({'Score': 'Neutral', 
                     'Slope' : slope, 
                     'Y-Intercept': intercept,
                     'P-Value' : p_value,
                     'Standard Error' : std_err
                    })

cond = (df3['score'] > 0) & (df3['views'] > 0)
slope, intercept, r_value, p_value, std_err = stats.linregress(x=np.log10(df3[cond]['score']), y=np.log10(df3[cond]['views']))
summary_data.append({'Score': 'Negative', 
                     'Slope' : slope, 
                     'Y-Intercept': intercept,
                     'P-Value' : p_value,
                     'Standard Error' : std_err
                    })

pd.DataFrame(data=summary_data, columns=list(summary_data[0].keys()))


# **Observation:** The correlation between tags sentiment and views presented not satisfactorily results. A higher score for p-values and standard errors can be examined for this analysis. The hypothesis-test of the existence of a correlation between the sentiment of the tags and the number of views was rejected.

# #### Top 10 Creators & Categories
# _______

# In[ ]:


fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
sns.set(font_scale=0.7)


df_top = df_unique[['views', 'title', 'category_name', 'channel_title' ]].nlargest(n=10,columns=['views'])

ax1.pie(x=df_top['views'], labels=df_top['title'], shadow=True, startangle=90, autopct='%1.1f%%',labeldistance=1.0,pctdistance=0.7)
ax1.set_title('Total 10 Videos Views', fontsize=12)
ax1.axis('equal')
ax1.set_xlim(auto=True)
ax1.set_ylim(auto=True)
sns.set(font_scale=0.9)

df_top_cat = df_top[['category_name', 'views']].groupby('category_name', as_index=False).sum()
ax2.pie(x=df_top_cat['views'], labels=df_top_cat['category_name'], shadow=True, startangle=90, autopct='%1.1f%%',labeldistance=1.0,pctdistance=0.7)
ax2.set_title('Total 10 Categories', fontsize=12)
ax2.axis('equal')
ax2.set_xlim(auto=True)
ax2.set_ylim(auto=True)
plt.tight_layout()
#plt.savefig('../../../../resources/images/plot_009.png')
plt.show()


# #### Music & Entertainment Categories Versus Remainder Population
# ____________

# In[ ]:


cond = (df_unique['category_name'] == 'Music') | (df_unique['category_name'] == 'Entertainment')
total_music_enter = df_unique[cond]['views'].sum()

cond = (df_unique['category_name'] != 'Music') & (df_unique['category_name'] != 'Entertainment')
total_others = df_unique[cond]['views'].sum()

summary_data = [{
        'Category' : 'Music & Entertainment',
        'Views'    : total_music_enter},
    {
        'Category' : 'All Others',
        'Views'    : total_others
    }
]

df_compare = pd.DataFrame(data=summary_data, columns=list(summary_data[0].keys()))

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 1, 1)

explode= (0.1, 0)
colors = ['darksalmon','royalblue']


ax1.pie(x=df_compare['Views'], labels=df_compare['Category'], shadow=True, startangle=90, autopct='%1.1f%%',
        labeldistance=1.0, pctdistance=0.7, explode=explode, colors=colors)
ax1.set_title('Music & Entertainment Total Views versus All Others Categories', fontsize=12)
ax1.axis('equal')
ax1.set_xlim(auto=True)
ax1.set_ylim(auto=True)

plt.tight_layout()
#plt.savefig('../../../../resources/images/plot_010.png')

plt.show()


# **Observation:** Youtube is a territory dominated by Music & Entertainment. Any kind of association with these categories has the potential to bring an enormous number of views. As we can see, people tend to love to use Youtube to watch Music and Entertainment videos. It's clear also to see that they tend to watch the same video several times. Retaining people attention is difficult and knowing where to explore this kind of behavior can potentially lead to success.

# #### Videos Duration
# ______________________
# Patterns of video duration

# In[ ]:


size_bins = [-1, 60, 120, 300, 600]
group_names = ['< 1 min', '1-2 min(s)', '2-5 mins', '5-10 mins', '> 10 mins']
color = ['black', 'red', 'green', 'blue', 'cyan']
size_bins.append(df_unique['duration'].max()+1)
video_columns = ['duration', 'views']

df_videos = df_unique[video_columns].copy(deep=True)
df_videos['Videos Range'] = pd.cut(df_videos['duration'],size_bins, labels=group_names)

df_videos_agg  = df_videos.groupby(by='Videos Range', as_index=False).agg({
                                        'views' : ['sum']
                                    })

fig = plt.figure(figsize=(14, 5))
ax = fig.add_subplot(1, 1, 1)

ax.bar(df_videos_agg['Videos Range'], df_videos_agg['views']['sum'], align="center", alpha=0.6, orientation='vertical', color=color, edgecolor='black')

ax.set_title('Total of Views per Video Duration', fontsize=15)
ax.set_ylabel('Views')
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
ax.set_yscale('linear')

plt.tight_layout()
#plt.savefig('../../../../resources/images/plot_011.png')

plt.show()


# **Observation:** As we can see, the video duration falls into a range duration of 2-5 minutes. Knowing how long should YouTube videos be is crucial for marketing campaign and producer efforts. Shorter videos demonstrated to be the key to success. Keep people attention for more than 5 minutes is a challenging task. Any decision in terms of video duration must take into consideration shorted videos rather than longer ones.

# #### Correlation between Total of Subscribers & Total of Uploads and Views
# ________________

# In[ ]:


cond = (creators['viewCount'] > 0) & (creators['videoCount'] > 0) & (creators['subscriberCount'] > 0)

creators_corr = creators[cond][['channel_id', 'subscriberCount', 'videoCount', 'viewCount']].groupby('channel_id').last()

df1 = creators_corr[['subscriberCount', 'viewCount']].copy(deep=True)
df1.rename(columns={'subscriberCount' : 'correlation_count', 'viewCount' : 'views'}, inplace=True)
df1['correlation_type'] = 'subscribers'

df2 = creators_corr[['videoCount', 'viewCount']].copy(deep=True)
df2.rename(columns={'videoCount' : 'correlation_count', 'viewCount' : 'views'}, inplace=True)
df2['correlation_type'] = 'uploads'

df_creators_plot = pd.concat([df1, df2])

ax = sns.lmplot(x='correlation_count', y='views', col='correlation_type', data=df_creators_plot, aspect=1.3, height=8,
                legend=True, hue='correlation_type', markers='+',col_wrap=2)

ax.set(xscale='log', yscale='log')

plt.tight_layout()
##plt.savefig('../../../../resources/images/plot_015.png')

plt.show()


# In[ ]:


# y={0:.1f}x+{1:.1f}
summary_data = []

slope, intercept, r_value, p_value, std_err = stats.linregress(x=np.log10(df1['correlation_count']), y=np.log10(df1['views']))
summary_data.append({'Correlation': 'subscribers', 
                     'Slope' : slope, 
                     'Y-Intercept': intercept,
                     'P-Value' : p_value,
                     'Standard Error' : std_err
                    })

slope, intercept, r_value, p_value, std_err = stats.linregress(x=np.log10(df2['correlation_count']), y=np.log10(df2['views']))
summary_data.append({'Correlation': 'uploads', 
                     'Slope' : slope, 
                     'Y-Intercept': intercept,
                     'P-Value' : p_value,
                     'Standard Error' : std_err
                    })

pd.DataFrame(data=summary_data, columns=list(summary_data[0].keys()))


# #### OLS Regression for Subscribers

# In[ ]:


model = sm.OLS(endog=np.log10(df1['views']), exog=np.log10(df1['correlation_count']))
fit = model.fit()
fit.summary()


# #### OLS Regression for Uploads

# In[ ]:


model = sm.OLS(endog=np.log10(df2['views']), exog=np.log10(df2['correlation_count']))
fit = model.fit()
fit.summary()


# **Observation:** As expected, the correlation between subscribers and views is strong. The same behavior can be examined for the correlation between uploads and views. R-squared score presents a result of 0.98 indicating a strong relationship between the two variables. Another really good indicator is the p-value score which is below 0.001. Therefore, the number of subscriber of a channel and the total of uploads has a direct relationship with the number of views. As one variable increase, the other also increases.

# In[ ]:




