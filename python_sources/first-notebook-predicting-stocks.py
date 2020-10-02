#!/usr/bin/env python
# coding: utf-8

# **Workflow of the project**
# 1. EDA
# 2. Feature Selection and Feature Engineering
# 3. Train Models
# 4. Evaluate Models

# In[ ]:


# Creating Environment 
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
print("Data Loaded")


# In[ ]:


market_train_df, news_train_df = env.get_training_data()


# In[ ]:


print("We have {:,} market samples and {} features in the training dataset.".format(market_train_df.shape[0], market_train_df.shape[1]))
print("We have {:,} news samples and {} features in the training dataset.".format(news_train_df.shape[0], news_train_df.shape[1]) )


# | **Variable** | **Definition** | **Key** |
# | ---------                    | ---------                                                                                                                                                |
# | time                         | the current time                                                                                                                                 | datatime64[ns, UTC] |
# | assetCode               | Unique id of an asset                                                                                                                        | object                           |
# | assetName              | Name that corresponds to a group of assetCodes                                                                       | category                       |
# | universe                   | a boolean indicating whether or not the instrument on that day will be included in scoring | float64                         |
# | volume               | trading volume in shares for the day| float64 |
# | close               | the close prize for the day | float64 |
# | open               | the open prize for the day | float64 |
# | returnsClosePrevRaw1               | Returns calculated close-to-close for raw data | float64 |
# | returnsOpenPrevRaw1               | Returns calculated open-to-open for raw data | float64 |
# | returnsClosePrevMktres1               | Returns calculated close-to-close for market-residualized (MKtres) for one day | float64 |
# | returnsOpenPrevMktres1               | Returns calculated open-to-open for market-residualized (MKtres) for one day | float64 |
# | returnsClosePrevRaw10               | Returns calculated close-to-close for raw data  for previous 10 days | float64 |
# | returnsOpenPrevRaw10               | Returns calculated open-to-open for raw data  for previous 10 days | float64 |
# | returnsClosePrevMktres10               | Returns calculated close-to-close for market-residualized (MKtres) for 10 days | float64 |
# | returnsOpenPrevMktres10               | Returns calculated open-to-open for market-residualized (MKtres) for previous 10 days | float64 |
# | returnsOpenNextMktres10               | Returns calculated open-to-open for market-residualized (MKtres) for next 10 days | float64 |
# 

# In[ ]:


market_train_df.head()


# From the first 5 rows of data, we can see that the data is a mixture of 
# * Numerical
#     * Volume
#     * close
#     * open
#     * returnsClosePrevRaw1
#     * returnsOpenPrevRaw1
#     * returnsClosePrevMktres1
#     * returnsOpenPrevMktres1
#     * returnsClosePrevRaw10
#     * returnsOpenPrevRaw10
#     * returnsClosePrevMktres10
#     * returnsOpenPrevMktres10
#     * returnsOpenNextMktres10
# * Time   
#     * time
# * Categorical
#     * universe - 0 or 1
# * text
#     * assetName
#     * assetCode - unique id
#     

# In[ ]:


news_train_df.head()


# | **Variable** | **Definition** | **Key** |
# | ---------  | ---------  |
# | time  | UTC timestamp of this news item when it was created  | datatime64[ns, UTC] |
# | firstCreated  | UTC timestamp for the first version of the item | datatime64[ns, UTC]|
# | sourceId | an Id for each news item| object   |
# | headline  | the item's headline | object|
# | urgency  | differentiates story types (1: alert, 3: article) | int8 |
# | takeSequence  | the take sequence number of the news item, starting at 1. For a given story, alerts and articles have separate sequences. | float64 |
# | provider | identifier for the organization which provided the news item (e.g. RTRS for Reuters News, BSW for Business Wire) | category |
# | subjects | topic codes and company identifiers that relate to this news item. Topic codes describe the news item's subject matter. These can cover asset classes, geographies, events, industries/sectors, and other types. | category |
# | audiences |  identifies which desktop news product(s) the news item belongs to. They are typically tailored to specific audiences. (e.g. "M" for Money International News Service and "FB" for French General News Service) | category |
# | bodySize | the size of the current version of the story body in characters | int32 |
# | companyCount | the number of companies explicitly listed in the news item in the subjects field | int8 |
# | headlineTag | the Thomson Reuters headline tag for the news item | object |
# | marketCommentary | boolean indicator that the item is discussing general market conditions, such as "After the Bell" summaries | bool |
# | sentenceCount |  the total number of sentences in the news item. Can be used in conjunction with firstMentionSentence to determine the relative position of the first mention in the item. | int16 |
# | wordCount | the total number of lexical tokens (words and punctuation) in the news item | int32 |
# | assetCodes | list of assets mentioned in the item | category |
# | assetName |  name of the asset | category |
# | firstMentionSentence | the first sentence, starting with the headline, in which the scored asset is mentioned. 1: headline, 2: first sentence of the story body, 3: second sentence of the body, etc, 0: the asset being scored was not found in the news item's headline or body text. As a result, the entire news item's text (headline + body) will be used to determine the sentiment score.  | int16 |
# | relevance | a decimal number indicating the relevance of the news item to the asset. It ranges from 0 to 1. If the asset is mentioned in the headline, the relevance is set to 1. When the item is an alert (urgency == 1), relevance should be gauged by firstMentionSentence instead. | float32 |
# | sentimentClass | indicates the predominant sentiment class for this news item with respect to the asset. The indicated class is the one with the highest probability. | int8 |
# | sentimentNegative | probability that the sentiment of the news item was negative for the asset | float32 |
# | sentimentNeutral | probability that the sentiment of the news item was neutral for the asset | float32 |
# | sentimentPositive | probability that the sentiment of the news item was positive for the asset | float32 |
# | sentimentWordCount| the number of lexical tokens in the sections of the item text that are deemed relevant to the asset. This can be used in conjunction with wordCount to determine the proportion of the news item discussing the asset. | int32 |
# | noveltyCount12H | The 12 hour novelty of the content within a news item on a particular asset. It is calculated by comparing it with the asset-specific text over a cache of previous news items that contain the asset. | int16 |
# | noveltyCount24H| same as above, but for 24 hours | int16 |
# | noveltyCount3D| same as above, but for 3 day | int16 |
# | noveltyCount5D| same as above, but for 5 day | int16 |
# | noveltyCount7D| same as above, but for 7 day | int16 |
# | volumeCounts12H| the 12 hour volume of news for each asset. A cache of previous news items is maintained and the number of news items that mention the asset within each of five historical periods is calculated. | int16 |
# | volumeCounts24H| same as above, but for 24 hours | int16 |
# | volumeCounts3D| same as above, but for 3 days | int16 |
# | volumeCounts5D| same as above, but for 5 days | int16 |
# | volumeCounts7D| same as above, but for 7 days | int16 |

# From the first 5 rows of data, we can see that the data is a mixture of 
# * Numerical
#     * bodySize
#     * companyCount
#     * open
#     * sentenceCount
#     * wordCount
#     * relevance
#     * sentimentNegative
#     * sentimentNeutral
#     * sentimentPositive
#     * sentimentWordCount
#     * noveltyCount12H
#     * noveltyCount24H
#     * noveltyCount3D
#     * noveltyCount5D
#     * noveltyCount7D
#     * volumeCounts12H
#     * volumeCounts24H
#     * volumeCounts3D
#     * volumeCounts5D
#     * volumeCounts7D
# * Time   
#     * time
#     * sourceTimestamp
#     * firstCreated
# * Categorical
#     * urgency
#     * takeSequence
#     * provider
#     * subjects
#     * audiences
#     * headlineTag
#     * marketCommentary
#     * urgency
#     * firstMentionSentence
#     * sentimentClass
# * text
#     * headline
#     * assetName

# **EDA:**
# This is used to explore the target and features. Looking at the dataset, the priliminary questions are
# 1. What do variables look like ? Are they numerical or categorical ? If numerical, what is their distribution. If categorical, how many are they in different categories?
# 2. Are numerical variables correlated?

# In[ ]:


# What do variables look like?
market_train_df.describe()


# All the numerical variables are verified from the above. The only exception here is 'universe' which from the description is the boolean. 

# In[ ]:


market_train_df.describe(include=['O'])


# All the object variables, you can see how many categories are in each variable from the "unique" row. But from the above it can be said that 'assetCode' from the description is the unique id of an asset. 

# In[ ]:


news_train_df.describe()


# Exceptions are 
# 1. urgency which describes 1 for alert and 3 for article.
# 2. takeSequence which starts at 1 and varies depending on alerts, articles etc. 
# 3. firstMentionSentence which start with 0 and ends in 1 depending on headline etc. 
# 4. sentimentClass - also is categorical

# In[ ]:


news_train_df.describe(include=['O'])


# Exceptions are 
# 1. sourceId is a unique id for each news item
# 2. headline - is a text field.
# 3. headlineTag - is also a text field.

# **Histograms of the Numerical Variables**
# Histogram is a good visualization technique to check the distribution of numerical data. 

# In[ ]:


# import visulization packages
import matplotlib.pyplot as plt

# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (20, 30)

# make subplots
fig, axes = plt.subplots(nrows = 6, ncols = 2)

# Specify the features of interest
num_features = ['volume', 'close', 'open','returnsClosePrevRaw1','returnsOpenPrevRaw1','returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
                'returnsClosePrevRaw10','returnsOpenPrevRaw10','returnsClosePrevMktres10', 'returnsOpenPrevMktres10','returnsOpenNextMktres10']
xaxes = num_features
yaxes = ['Counts', 'Counts', 'Counts','Counts', 'Counts','Counts', 'Counts','Counts', 'Counts','Counts', 'Counts','Counts']

# draw histograms
axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.hist(market_train_df[num_features[idx]].dropna(), bins=40)
    ax.set_xlabel(xaxes[idx], fontsize=20)
    ax.set_ylabel(yaxes[idx], fontsize=20)
    ax.tick_params(axis='both', labelsize=15)


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
sns.set(color_codes=True)
sns.distplot(market_train_df['volume'], hist=False, rug=True);


# In[ ]:




