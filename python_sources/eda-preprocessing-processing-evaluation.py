#!/usr/bin/env python
# coding: utf-8

#  **Stock Analysis**

# ![](https://st.depositphotos.com/1760261/1348/i/950/depositphotos_13484306-stock-photo-data-analysis.jpg)

# <a id="0"></a> <br>
# ## Kernel Headlines
# 1. [Introduction and RoadMap](#1)
# 2. [Market Data Analysis](#2)
#     1.  [time](#3)
# 	2.  [assetCode_violin](#4)
# 	3.  [assetName_wordcloud](#5)
# 	4.  [universe_pi-chart](#6)
# 	5.  [volume_violin](#7)
# 	6.  [close_line](#8)
# 	7.  [open_line](#9)
# 	8.  [returns_boxplot](#10)
#     
# 3. [News Data Analysis](#15)
#      1.  [time_line](#16)
# 	 2.  [sourceTimestamp_line](#17)
# 	 3.  [firstCreated_line](#18)
# 	 4.  [sourceId_qpercentile](#19)
# 	 5.  [headline_wordcloud](#20)
# 	 6.  [urgency_pi_chart](#21)
# 	 7.  [takeSequence](#22)
# 	 8.  [provider_barchart](#23)
# 	 9.  [subjects_wordcloud](#24)
# 	 10.  [audience_wordcloud](#25)
# 	 11.  [word_sentences_describe](#26)
# 	 12.  [sentimentStatus_barchart](#27)
# 	 13.  [novelties_pichart](#28)
# 	 14.  [volumes_violin](#29)
# 	 15.  [headlineTag_wordcloud](#30)
# 	 16.  [marketCommentary_pichart](#31)
# 	 17.  [sentimentWordcount_histogram](#32)
# 	 18.  [assetName_wordcloud](#33)
# 	 19.  [assetCode_head](#34)
# 	 20.  [sentimentClass_pichart](#35)
# 	 21.  [relevance_violon](#36)
# 	 22.  [firstMentionSentence_describe](#37)
#      
# 4. [COMPOUND FEATURES ANALYSIS](#38)
# 

# <a id="1"></a> <br>
# #  1-INTRODUCTION AND ROADMAP
# 
# In this challenge we will deal with the stock data. There are two datasets. Marketdata_sample and News_sample. contains financial market information. Features like opening price and closing price and this sort of things are existed in dataset. News dataset contains information about news articles/alerts published about assets. Attributes like asset details are located in this dataset.
# The main goal of the competition is how we can use the content of news analytics to predict stock price performance.  
# Now, we will try to analyze both of datasets feature by feature.
# 
#    **1. In first step we will do study on market data and try to visualize features and extract information from data.**
#     
#    **2. In the second step we will concentrate on news data and it's features.**
#     
#    **3. In the third section relation between two dataframes will be validated. Features correlations will be discussed there.**
#     
#    **4. Finally in forth step we will try to address the competition challenge goal and do submission based on the EDA we will have in this kernel. **
#    
# Any comment, idea or hint will be appreciated. 
# 
# **Your upvote will be motivation for me for continuing the kernel ;-)**

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import collections
import warnings
from kaggle.competitions import twosigmanews
from datetime import datetime
from wordcloud import WordCloud

warnings.filterwarnings('ignore')
#getting environment for accessing full data
env = twosigmanews.make_env()


# Reading the data and understanding the data.

# In[ ]:


market_train_full_df = env.get_training_data()[0]
sample_market_df = pd.read_csv("../input/marketdata_sample.csv")
sample_news_df = pd.read_csv("../input/news_sample.csv")


# Checking the dimentions of data.

# In[ ]:


"market_train_full_df dimention:{}".format(market_train_full_df.shape)


# <a id="2"></a> <br>
# #  2-MARKET_DATA ANALYSIS

# In[ ]:


market_train_full_df.head(5)


# Now lets read the data description. 
# 
# As it is mentioned in in description, the Market data (2007 to present) contains financial market information such as opening price, closing price, trading volume, calculated returns, etc.
# Now, lets take a glance to data description and understanding the attributes.

# In[ ]:


market_train_full_df.columns


# Going deeper to market_df attributes and investigate feature by feature.

# <a id="3"></a> <br>
# * **A. TIME**
# 
# time(datetime64[ns, UTC]) - the current time 

# In[ ]:


market_train_full_df.time.describe()


# In[ ]:


# pd.to_datetime(market_train_full_df.time).apply(lambda x: pd.Series({"daily":datetime.date(x)}))["daily"].value_counts()
fig,axes = plt.subplots(1,1,figsize=(15,10))
axes.set_title("Time Distro")
axes.set_ylabel("# of records")
axes.set_xlabel("date")
axes.plot(market_train_full_df.time.dt.date.value_counts().sort_index().index, market_train_full_df.time.dt.date.value_counts().sort_index().values)


# As it can be concluded that we have a missing values in last month of 2014. 
# 
# It is Time dependent operation. So, timeseries analyzing is one of the possible action in this competition. We will to try to do it in next steps.

# <a id="4"></a> <br>
# * **B. ASSETCODE**
# 
#  a unique id of an asset

# In[ ]:


market_train_full_df.assetCode.describe()


# In[ ]:


market_train_full_df.assetCode.value_counts().describe()


# In[ ]:


list(market_train_full_df.assetCode)[:5]


# There are 3780 assetCodes.  the relation between these codes and the correspondence assetCode which is in news_df can bring considerable information. we will check it in next sections.
# 
# As it can be seen, there are assetsCode which repeated in 2498 records. The violion chart maybe represents more clear information about assetsCode distributions.

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
axes.set_title("Daily assetCodes Violin")
axes.set_ylabel("Repetition")
axes.violinplot(list(market_train_full_df.assetCode.value_counts().values),showmeans=False,showmedians=True)


# <a id="5"></a> <br>
# * **C. ASSETNAME**
# 
# Or category, the name that corresponds to a group of assetCodes. These may be "Unknown" if the corresponding assetCode does not have any rows in the news data.

# In[ ]:


market_train_full_df.assetName.describe()


# As you can see, the 'Unknown' assetName has 3511 unique values and  'Unknown' is the assetName that has most frequency in train data.

# In[ ]:


list(market_train_full_df.assetName)[:10]


# WordCloud can represnets more detailed information.

# In[ ]:


from wordcloud import WordCloud
# Create the wordcloud object
wordcloud = WordCloud(width=1024, height=1024, margin=0).generate(" ".join(market_train_full_df.assetName))
 
# Display the generated image:
fig,ax = plt.subplots(1,1,figsize=(20,10))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
ax.margins(x=0, y=0)
plt.show()


# assetNames with more repetition can be easily seen in the wordcloud.

# <a id="6"></a> <br>
# * **D. UNIVERSE**
# 
# (float64) - a boolean indicating whether or not the instrument on that day will be included in scoring. This value is not provided outside of the training data time period. The trading universe on a given date is the set of instruments that are avilable for trading (the scoring function will not consider instruments that are not in the trading universe). The trading universe changes daily.

# In[ ]:


market_train_full_df.universe.value_counts()


# In[ ]:


univers_df_dict = dict(collections.Counter(list(market_train_full_df.universe)))
percent_univers_df_dict = {k: v / total for total in (sum(univers_df_dict.values()),) for k, v in univers_df_dict.items()}
explode=(0,0.1)
labels ='notUniverse','isUniverse'
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.set_title("Univere Status")
ax.pie(list(percent_univers_df_dict.values()), explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)

market_train_full_df.universe.value_counts()


# The distribution of universe in train data is provided above.

# <a id="7"></a> <br>
# * **E. VOLUME**
# 
# (float64) - trading volume in shares for the day

# In[ ]:


market_train_full_df.volume.describe()


# There is considerable std in the volume feature. On the other hand, the 75% percentile represents that the gap between the last quater of volume is very bigger than first quater. As a result, if we plot the violin diagram, It has considerable mass in first quater. Lets validate it...

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
axes.set_title("Volume Violin")
axes.set_ylabel("Volume")
axes.violinplot(list(market_train_full_df["volume"].values),showmeans=False,showmedians=True)


# Our assumption has been approved. ;-)

# In[ ]:


fig, axes = plt.subplots(figsize=(20,10))
axes.set_title("Volume")
axes.set_ylabel("volume")
axes.set_xlabel("records")
axes.plot(market_train_full_df["volume"])


# The volume in some records has considerable peak in comparison to other ones. We will analyse it with more details in next sections.

# <a id="8"></a> <br>
# * **F. CLOSE**
# 
# (float64) - the close price for the day (not adjusted for splits or dividends)

# In[ ]:


market_train_full_df.close.describe()


# In[ ]:


fig, axes = plt.subplots(figsize=(20,10))

axes.set_title("Close Price")
axes.set_ylabel("close price")
axes.set_xlabel("records")
axes.plot(market_train_full_df["close"])


# <a id="9"></a> <br>
# * **G. OPEN**
# 
# 
# the open price for the day (not adjusted for splits or dividends)

# In[ ]:


fig, axes = plt.subplots(figsize=(20,10))
axes.set_title("Open Price")
axes.set_ylabel("open price")
axes.set_xlabel("records")
axes.plot(market_train_full_df["open"])


# By discarding the two peaks which are obviously detectable and comparing two above diagrams (close and open price) the low difference between these two diagram has been revealed.. We will discuss in more details in next sections. But for now, it is enough to see the same rate in both of the diagrams.

# <a id="10"></a> <br>
# * **F. RETURNS**
# 
# Returns are calculated based on different timespans. creating 

# In[ ]:


market_returns_df = pd.concat(
    [
        market_train_full_df["returnsClosePrevRaw1"].describe(),
        market_train_full_df["returnsOpenPrevRaw1"].describe(),
        market_train_full_df["returnsClosePrevMktres1"].describe(),
        market_train_full_df["returnsOpenPrevMktres1"].describe(),
        market_train_full_df["returnsClosePrevRaw10"].describe(),
        market_train_full_df["returnsOpenPrevRaw10"].describe(),
        market_train_full_df["returnsClosePrevMktres10"].describe(),
        market_train_full_df["returnsOpenPrevMktres10"].describe(),
        market_train_full_df["returnsOpenNextMktres10"].describe()
        ],
        axis=1
    )
market_returns_df


# Dropping NaN attributes for more clearly representation.

# In[ ]:


market_returns_df.drop(["returnsClosePrevMktres1","returnsOpenPrevMktres1","returnsClosePrevMktres10","returnsOpenPrevMktres10"],axis=1,inplace=True)
market_returns_df


# There are some points can be concluded from the table:
# 
#  **1- 25% percentiles of whole the returnPeriods are negative.**
#  
#  **2- By increasing the period of returns, both of the open and close values are encoutered with larger std**
#  
#  **3- whole the periods have negative returns (min)**

# Is there any relation between returnsClosePrevRaw1 and returnsOpenPrevRaw1 ?!

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
axes.set_title("Box Plot")
axes.set_ylabel("Returns")
market_train_full_df.boxplot(column=['returnsClosePrevRaw1', 'returnsOpenPrevRaw1', "returnsClosePrevRaw10","returnsOpenPrevRaw10","returnsOpenNextMktres10"])    


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8),sharey=True)
axes[0].set_title("One-day difference in close and open returns")
axes[0].set_ylabel("difference")
axes[0].violinplot(list((market_train_full_df["returnsClosePrevRaw1"] - market_train_full_df["returnsOpenPrevRaw1"]).values),showmeans=False,showmedians=True,widths=0.9, showextrema=True)
axes[1].set_title("10-day difference in close and open returns")
axes[1].set_ylabel("difference")
axes[1].violinplot(list((market_train_full_df["returnsClosePrevRaw10"] - market_train_full_df["returnsOpenPrevRaw10"]).values),showmeans=False,showmedians=True,widths=0.9, showextrema=True)


# Two important hints can be concluded:
# 
# **1- The difference of both of the periods is approximately concentrated on -8000 .**

# Doing the similar analyze on returnsOpenNextMktres10, we have:

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9),sharey=True)
axes.set_title("returnsOpenNextMktres10 violon")
axes.set_ylabel("")
axes.violinplot(list((market_train_full_df["returnsOpenNextMktres10"]).values),showmeans=False,showmedians=True,widths=0.9, showextrema=True)


# In next sections we will discussed more about market_df and its relation with news data.
# Now lets move on to News data and do similar studies.

# For handling probably memory heaps, we delete the market_train_full_df data and load the news_train_full_df. After finishing EDA on news_train_full_df, we will load both of them for analyzing compound features.

# In[ ]:


del axes
del market_train_full_df


# In[ ]:


news_train_full_df = env.get_training_data()[1]


# <a id="15"></a> <br>
# #  3-NEWS_DATA ANALYSIS
# 

# In[ ]:


news_train_full_df.head()


# Taking a glance to columns for undestranding the features...

# In[ ]:


news_train_full_df.columns


# <a id="16"></a> <br>
# * **A. TIME**
# 
# 
# time(datetime64[ns, UTC]) - UTC timestamp showing when the data was available on the feed (second precision)
# 
# lets define new temporary dataframe for studying on time parameters.

# In[ ]:


news_train_full_df.time.describe()


# In[ ]:


fig,axes = plt.subplots(1,1,figsize=(20,10))
axes.set_title("Time Distro")
axes.set_ylabel("# of records")
axes.set_xlabel("date")
axes.plot(news_train_full_df.time.dt.date.value_counts().sort_index().index, news_train_full_df.time.dt.date.value_counts().sort_index().values)


# There is interesting point in the above chart. As you can see in december months (every year) there is a local minimum. And the reason is people was preparing themselves for christmas. They have ignored the stocks obviously.

# <a id="8"></a> <br>
# * **B. SOURCE_TIMESTAMP**
# 
# sourceTimestamp represents the time when the news was created.
# Lets do a studying like a previous feature on this attribute.
# 

# In[ ]:


news_train_full_df.sourceTimestamp.describe()


# In[ ]:


fig,axes = plt.subplots(1,1,figsize=(20,10))
axes.set_title("sourceTimestamp Distro")
axes.set_ylabel("# of records")
axes.set_xlabel("date")
axes.plot(news_train_full_df.sourceTimestamp.dt.date.value_counts().sort_index().index, news_train_full_df.sourceTimestamp.dt.date.value_counts().sort_index().values)


# Similarity between the features "sourceTimestamp" and "time" can be concluded.

# <a id="18"></a> <br>
# * **C. FIRST_CREATED**
# 
# 
# firstCreated(datetime64[ns, UTC]) - UTC timestamp for the first version of the item
# last two previous time-based features, lets do similar study.

# In[ ]:


news_train_full_df.firstCreated.describe()


# In[ ]:


fig,axes = plt.subplots(1,1,figsize=(20,10))
axes.set_title("firstCreated Distro")
axes.set_ylabel("# of records")
axes.set_xlabel("date")
axes.plot(news_train_full_df.firstCreated.dt.date.value_counts().sort_index().index, news_train_full_df.firstCreated.dt.date.value_counts().sort_index().values)


# <a id="19"></a> <br>
# * **D. SOURCE_ID**
# 
# 
# An Id for each news item.

# In[ ]:


news_train_full_df.sourceId.describe()


# In[ ]:


news_train_full_df.sourceId.value_counts().describe()


# So, in contrast to time-based features, sourceIds have repetitions. There are 9328750 sourceIds and most repetition is 43 times. Correlation between this features and other ones will be discussed on next steps.
# 
# 75% of news items have repetition less than one time.

# <a id="20"></a> <br>
# * **E. HEADLINE**
# 
# The item's headline

# In[ ]:


news_train_full_df.headline.describe()


# Similar to sourceIds, there are repetitions. There are 5532379 unique headline. 
# Lets go deeper to headlines ;-)

# Drawing wordcloud ...
# Using a section of data beacause of avoiding memory overflows...

# In[ ]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
 

# Create a list of word
# text=("Python Python Python Matplotlib Matplotlib Seaborn Network Plot Violin Chart Pandas Datascience Wordcloud Spider Radar Parrallel Alpha Color Brewer Density Scatter Barplot Barplot Boxplot Violinplot Treemap Stacked Area Chart Chart Visualization Dataviz Donut Pie Time-Series Wordcloud Wordcloud Sankey Bubble")
 
# Create the wordcloud object
wordcloud = WordCloud(width=1024, height=1024, margin=0).generate(( " ".join(list(news_train_full_df.head(200000).headline))))
 
# Display the generated image:
fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
ax.margins(x=0, y=0)
plt.show()


# Most repetitive word have been seen above ;-).

# <a id="21"></a> <br>
# * **F. URGENCY**
# 
# differentiates story types (1: alert, 3: article,2:unknown)

# In[ ]:


news_train_full_df.urgency.value_counts()


# There is only one record which represents the alerting. The rest of 99 items are representing article record. You can easily draw the distribution using pi-chart.

# In[ ]:


urgency_df_dict = dict(collections.Counter(list(news_train_full_df.urgency)))
percent_urgency_df_dict = {k: v / total for total in (sum(urgency_df_dict.values()),) for k, v in urgency_df_dict.items()}
explode=(0,0.1,0.1)
labels ="article","alert", "unknown"
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.set_title("Urgency Status")
ax.pie(list(percent_urgency_df_dict.values()), explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)


# It seems that unknow is a miss in data. Except it's small occurences, it also doesnt mentioned in data describtion.

# <a id="22"></a> <br>
# * **G. TAKE_SEQUENCE**
# 
# The take sequence number of the news item, starting at 1. For a given story, alerts and articles have separate sequences.

# In[ ]:


news_train_full_df.takeSequence.value_counts().head(20)


# <a id="23"></a> <br>
# * **H. PROVIDER**
# 
# 
# dentifier for the organization which provided the news item (e.g. RTRS for Reuters News, BSW for Business Wire)

# In[ ]:


fig,ax = plt.subplots(1,1,figsize=(15,10))
ax.set_xlabel("name")
ax.set_ylabel("#")
news_train_full_df.provider.value_counts().plot(kind="bar",legend="provider",color="tan")


# 'Reuturs' has the most repetition in our dataset.

# <a id="24"></a> <br>
# * **I. SUBJECTS**
# 
# Topic codes and company identifiers that relate to this news item. Topic codes describe the news item's subject matter. These can cover asset classes, geographies, events, industries/sectors, and other types.

# In[ ]:


news_train_full_df.head(5).subjects


# In[ ]:


from collections import Counter
tmp_list = []
for i in news_train_full_df.head(200000).subjects:
    tmp_list += i.replace("{","").replace("}","").replace(" ","").split(",")
# Counter(tmp_list)
# fig,ax = plt.subplots(1,1,figsize=(30,10))
# ax.set_xticklabels(dict(Counter(tmp_list)).keys(),rotation=90)
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 10}
# plt.rc('font', **font)
# ax.bar(dict(Counter(tmp_list)).keys(),dict(Counter(tmp_list)).values())


# Another userful visualization for this feature can be wordcloud.

# In[ ]:


text =" ".join(tmp_list).replace("'","")
 # Create the wordcloud object
wordcloud = WordCloud(width=1024, height=1024, margin=0).generate(text)
 
# Display the generated image:
fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
ax.margins(x=0, y=0)
plt.show()


# <a id="25"></a> <br>
# * **J. AUDIENCES**
# 
# Identifies which desktop news product(s) the news item belongs to. They are typically tailored to specific audiences. (e.g. "M" for Money International News Service and "FB" for French General News Service)

# In[ ]:


news_train_full_df.head(5).audiences


# Similar visualization like previous feature can be done on the 'audiences' feature.

# In[ ]:


from collections import Counter
tmp_list = []
for i in news_train_full_df.head(200000).audiences:
    tmp_list += i.replace("{","").replace("}","").replace(" ","").split(",")
# Counter(tmp_list)

# fig,ax = plt.subplots(1,1,figsize=(30,10))
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 22}
# plt.rc('font', **font)
# ax.set_xticklabels(dict(Counter(tmp_list)).keys(),rotation=90)
# ax.bar(dict(Counter(tmp_list)).keys(),dict(Counter(tmp_list)).values())


# In[ ]:


text =" ".join(tmp_list).replace("'","")
 # Create the wordcloud object
wordcloud = WordCloud(width=1024, height=1024, margin=0).generate(text)
 
# Display the generated image:
fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
ax.margins(x=0, y=0)
plt.show()


# <a id="26"></a> <br>
# * **K. BODY_SIZE, COMPANY_COUNT,SENTENCE_COUNT,WORD_COUNT**
# 
# bodySize represents the size of the current version of the story body in characters
# companyCount represents the number of companies explicitly listed in the news item in the subjects field
# sentenceCount represents the total number of sentences in the news item. Can be used in conjunction with firstMentionSentence
# wordCount represents the total number of lexical tokens (words and punctuation) in the news item

# In[ ]:


pd.concat([news_train_full_df.bodySize.describe(),news_train_full_df.companyCount.describe(),news_train_full_df.sentenceCount.describe(),news_train_full_df.wordCount.describe()],axis=1)


# news bodies have averagely 3082 character. Atleast one record have no body.
# in companyCount feature we can see there is (are) which referenced 9 companies in it's subject field.

# <a id="27"></a> <br>
# * **L. SENTIMENT_NEGATIVE, SENTIMENT_NEUTRAL, SENTIMENT_POSITIVE**
# 
# sentimentNegative, sentimentNeutral and sentimentPositive respectively represents the probability that the sentiment of the news item was negative, neutral or positive for the asset.

# In[ ]:


pd.concat([news_train_full_df.sentimentNegative.describe(),news_train_full_df.sentimentNeutral.describe(),news_train_full_df.sentimentPositive.describe()],axis=1)


# In[ ]:


fig , axes = plt.subplots(1,1,figsize=(20,8))
news_train_full_df.sentimentNegative.head(100).plot(kind="bar",legend="Negative",colormap="brg")
news_train_full_df.sentimentPositive.head(100).plot(colormap="Set2",linewidth=2,legend="Postive")
news_train_full_df.sentimentNeutral.head(100).plot(colormap="RdGy",linewidth=0.8,linestyle='dashed',legend="Neutral")
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}
axes.set_xticklabels(news_train_full_df.head(100).index,rotation=90)
legend = axes.legend(loc='upper left', shadow=True, fontsize='x-large')
plt.rc('font', **font)


# The comparison between negative, positive and neutral probabilities represents the summation of these three probabilities is more than 1.0 in most cases and one of them is higher one. On the other hand there is no significant relation between these probabilities. For example in record number 83,84 and 85 the negativeEffect and PositiveEffects are low but the neutral effect is high. In contrast in records 53 and 54 the positiveEffect is major one.

# <a id="28"></a> <br>
# * **M. NOVELTY_COUNT12H, NOVELTY_COUNT_24H, NOVELTY_COUNT_3D, NOVELTY_COUNT_5D, NOVELTY_COUNT_7D**
# 
# 
# NoveltyCounts represents the novelty of the content within a news item on a particular asset. It is calculated by comparing it with the asset-specific text over a cache of previous news items that contain the asset.

# In[ ]:


pd.concat([news_train_full_df.noveltyCount12H.describe(),news_train_full_df.noveltyCount24H.describe(), news_train_full_df.noveltyCount3D.describe(),
          news_train_full_df.noveltyCount5D.describe(),news_train_full_df.noveltyCount7D.describe()],axis=1)


# In[ ]:


fig,axes = plt.subplots(3,2,figsize=(10,15))

noveltyCount12H_dict = dict(collections.Counter(list(news_train_full_df.noveltyCount12H)))
percent_noveltyCount12H_dict = {k: v / total for total in (sum(noveltyCount12H_dict.values()),) for k, v in noveltyCount12H_dict.items()}
sizes = list(percent_noveltyCount12H_dict.values())
axes[0][0].set_title("noveltyCount12H",loc="left")
axes[0][0].pie(sizes,  autopct='%1.1f%%',shadow=False, startangle=90)


noveltyCount24H_dict = dict(collections.Counter(list(news_train_full_df.noveltyCount24H)))
percent_noveltyCount24H_dict = {k: v / total for total in (sum(noveltyCount24H_dict.values()),) for k, v in noveltyCount24H_dict.items()}
sizes = list(percent_noveltyCount24H_dict.values())
axes[0][1].set_title("noveltyCount24H",loc="left")
axes[0][1].pie(sizes,  autopct='%1.1f%%',shadow=False, startangle=90)

noveltyCount3D_dict = dict(collections.Counter(list(news_train_full_df.noveltyCount3D)))
percent_noveltyCount3D_dict = {k: v / total for total in (sum(noveltyCount3D_dict.values()),) for k, v in noveltyCount3D_dict.items()}
sizes = list(percent_noveltyCount3D_dict.values())
axes[1][0].set_title("noveltyCount3D",loc="left")
axes[1][0].pie(sizes,  autopct='%1.1f%%',shadow=False, startangle=90)


noveltyCount5D_dict = dict(collections.Counter(list(news_train_full_df.noveltyCount5D)))
percent_noveltyCount5D_dict = {k: v / total for total in (sum(noveltyCount5D_dict.values()),) for k, v in noveltyCount5D_dict.items()}
sizes = list(percent_noveltyCount5D_dict.values())
axes[1][1].set_title("noveltyCount5D",loc="left")
axes[1][1].pie(sizes,  autopct='%1.1f%%',shadow=False, startangle=90)

noveltyCount7D_dict = dict(collections.Counter(list(news_train_full_df.noveltyCount7D)))
percent_noveltyCount7D_dict = {k: v / total for total in (sum(noveltyCount7D_dict.values()),) for k, v in noveltyCount7D_dict.items()}
sizes = list(percent_noveltyCount7D_dict.values())
axes[2][0].set_title("noveltyCount7D",loc="left")
axes[2][0].pie(sizes,  autopct='%1.1f%%',shadow=False, startangle=90)


overal_dict = pd.concat([news_train_full_df.noveltyCount12H,news_train_full_df.noveltyCount24H, news_train_full_df.noveltyCount3D,
          news_train_full_df.noveltyCount5D,news_train_full_df.noveltyCount7D],axis=0)
noveltyOveral_dict = dict(collections.Counter(list(overal_dict)))
percent_overal_dict = {k: v / total for total in (sum(noveltyOveral_dict.values()),) for k, v in noveltyOveral_dict.items()}
sizes = list(percent_overal_dict.values())
axes[2][1].set_title("overalNovelty",loc="left")
axes[2][1].pie(sizes,  autopct='%1.1f%%',shadow=False, startangle=90)
print()


# <a id="29"></a> <br>
# * **N. VOLUME_COUNT_12H, VOLUME_COUNT_24H, VOLUME_COUNT_3D, VOLUME_COUNT_15D, VOLUME_COUNT_7D**
# 
# volumeCounts represents the volume of news for each asset. A cache of previous news items is maintained and the number of news items that mention the asset within each of five historical periods is calculated.

# In[ ]:


pd.concat([news_train_full_df.volumeCounts12H.describe(),news_train_full_df.volumeCounts24H.describe(), news_train_full_df.volumeCounts3D.describe(),
          news_train_full_df.volumeCounts5D.describe(),news_train_full_df.volumeCounts7D.describe()],axis=1)


# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15),squeeze=False)
axes[0][0].set_title("volumeCounts12H")
axes[0][0].violinplot(list(news_train_full_df["volumeCounts12H"].values))

axes[0][1].set_title("volumeCounts24H")
axes[0][1].violinplot(list(news_train_full_df["volumeCounts24H"].values))

axes[1][0].set_title("volumeCounts3D")
axes[1][0].violinplot(list(news_train_full_df["volumeCounts3D"].values))

axes[1][1].set_title("volumeCounts5D")
axes[1][1].violinplot(list(news_train_full_df["volumeCounts5D"].values))

axes[2][0].set_title("volumeCounts7D")
axes[2][0].violinplot(list(news_train_full_df["volumeCounts7D"].values))

fig.delaxes(axes[2][1])


# It is clear that by increasing the timestamp (going from 12H to 7D) the volumeCounts range is increased. By the way the most frequent values have belonged to the minimum percentiles in whole cases.

# <a id="30"></a> <br>
# * **O. HEADLINE_TAG**
# 
# The Thomson Reuters headline tag for the news item

# In[ ]:


text=""
text =" ".join(list(news_train_full_df.headlineTag))
wordcloud = WordCloud(width=1024, height=1024, margin=0).generate(text)
 
# Display the generated image:
fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
ax.margins(x=0, y=0)
plt.show()


# As it is cleared in the word_cloud some combination of texts are repetitive. For example, both of the 'ROUNDUP RESEARCH' and 'RESEARCH ROUNDUP' exist in headlineTag

# <a id="31"></a> <br>
# * **P. MARKET_COMMUNITY**
# 
# Boolean indicator that the item is discussing general market conditions, such as "After the Bell" summaries

# In[ ]:


news_train_full_df.marketCommentary.value_counts()


# In[ ]:


market_commentary_df_dict = dict(collections.Counter(list(news_train_full_df.marketCommentary)))
percent_commentary_df_dict = {k: v / total for total in (sum(market_commentary_df_dict.values()),) for k, v in market_commentary_df_dict.items()}
explode=(0,0.1)
labels ='False','True'
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.set_title("Representing Genral Market Conditions")
ax.pie(list(percent_commentary_df_dict.values()), explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)


# Only 5 percent of the dataset have information about general conditions.

# <a id="32"></a> <br>
# * **Q. SENTIMEN_WORD_COUNT**
# 
# The number of lexical tokens in the sections of the item text that are deemed relevant to the asset. This can be used in conjunction with wordCount to determine the proportion of the news item discussing the asset.

# In[ ]:


news_train_full_df.sentimentWordCount.describe()


# In[ ]:


fig,ax = plt.subplots(1,1,figsize=(8,8))
ax.set_title("Hist(log sentimentWordCount)")
ax.set_xlabel("Log(sentimentWordCount)")
np.log10(news_train_full_df.sentimentWordCount).hist(ax=ax,)


# Averagely, there are approximately 200 lexical tokens which are related to assets.

# <a id="33"></a> <br>
# * **R. ASSET_NAME**
# 
# name of the asset

# In[ ]:


news_train_full_df.assetName.head(5)


# In[ ]:


tmp_list = []
for i in news_train_full_df.head(200000).assetName:
    tmp_list += i.replace("{","").replace("}","").replace(" ","").split(",")
text =" ".join(tmp_list).replace("'","")
 # Create the wordcloud object
wordcloud = WordCloud(width=1024, height=1024, margin=0).generate(text)
 
# Display the generated image:
fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
ax.margins(x=0, y=0)
plt.show()


# <a id="34"></a> <br>
# * **S. ASSET_CODE**
# 
# list of assets mentioned in the item

# In[ ]:


news_train_full_df.assetCodes.head(10)


# <a id="35"></a> <br>
# * **T. SENTIMENT_CLASS**
# 
# Indicates the predominant sentiment class for this news item with respect to the asset. The indicated class is the one with the highest probability.

# In[ ]:


news_train_full_df.sentimentClass.value_counts()


# In[ ]:


news_train_full_df.sentimentClass.value_counts()


# In[ ]:


sentiment_df_dict = dict(collections.Counter(list(news_train_full_df.sentimentClass)))
percent_univers_df_dict = {k: v / total for total in (sum(sentiment_df_dict.values()),) for k, v in sentiment_df_dict.items()}
explode=(0.0,0.025,0.05)
labels ='1','0',"-1"
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.set_title("sentimentClass")
ax.pie(list(percent_univers_df_dict.values()), explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)


# The distribution of sentimentClass is approximately uniform.

# <a id="36"></a> <br>
# * **U. RELEVANCE**
# 
# A decimal number indicating the relevance of the news item to the asset. It ranges from 0 to 1. If the asset is mentioned in the headline, the relevance is set to 1. When the item is an alert (urgency == 1), relevance should be gauged by firstMentionSentence instead.

# In[ ]:


news_train_full_df.relevance.describe()


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
axes.set_title("Volume Violin")
axes.set_ylabel("Volume")
axes.violinplot(list(news_train_full_df["relevance"].values),showmeans=False,showmedians=True)


# The most repetition of relevance is on One.

# <a id="37"></a> <br>
# * **V. FIRST_MENTION_SENTENCE**
# 
# The first sentence, starting with the headline, in which the scored asset is mentioned.
#         1: headline
#         2: first sentence of the story body
#         3: second sentence of the body, etc
#         0: the asset being scored was not found in the news item's headline or body text. As a result, the entire news item's text (headline + body) will be used to determine the sentiment score.

# In[ ]:


news_train_full_df.firstMentionSentence.describe()


# <a id="38"></a> <br>
# #  4-COMPOUND FEATURES ANALYSIS

# Now, we have considerable information about all the attributes. So, we are ready to start deeper analysis on each dataset and also the relation between these two dataset.  In the next step, we will focus on relation between features.

# **In progress ...**
# 
# **Be in touch to get last commits ...**
# 
# **I'll try to complete it as soon as possible**
# 
