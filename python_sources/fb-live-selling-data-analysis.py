#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# 
# The 'Facebook Live Sellers in Thailand' is a dataset curated in UCI Machine Learning Datasets. The data contains 7050 observations and twelve attributes. The data is about live selling feature on the Facebook platform. Each record consists of information about the time live information of sale is posted to Facebook and engagements in the data. The engagements are regular Facebook interactions such as share and emotion rection. Details and academic publications relating to the data is available from the source https://archive.ics.uci.edu/ml/datasets/Facebook+Live+Sellers+in+Thailand. 
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import warnings
warnings.simplefilter(action='ignore')

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn as sl


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Data Analysis

# In[ ]:


data = pd.read_csv("/kaggle/input/facebook-live-sellers-in-thailand-uci-ml-repo/Live.csv")


# In[ ]:


data.head(2)


# The columns Column1,Column2,Column3,Column4 are not part of the original data. These colums might have appeared in the data due to format conversion. We will exclude these columns from the analysis. 

# In[ ]:


data = data[data.columns[:-4]]
data.head()


# In[ ]:


data.info()


# From the pandas DataFrame meta information, it is evident that the data is complete w.r.f to the description. There are 7050 entries, and no null values are reported here. Let's proceed to expalore the data!
# 

# In[ ]:


data.nunique()


# From the unique value counts, it is evident that from the 7050 observations, only 6997 is unique live selling status. There are four types of status available in the data. From this, we can infer that around 53 observations may be duplicated or some other business phenomena are involved in the status_id column.

# In[ ]:


duplicated_data = data[data['status_id'].duplicated() == True]


# In[ ]:


duplicated_data.head()


# In[ ]:


duplicated_data.tail()


# In[ ]:


data[data.status_id == '246675545449582_326883450762124']


# In[ ]:


data[data.status_id == '819700534875473_1002372733274918']


# In[ ]:


data[data.status_id == '819700534875473_955149101330615']


# In[ ]:


data[data.status_id == '819700534875473_951614605017398']


# From the samples evaluated, it is evident that the 53 observations are duplicate. We will proceed and remove the duplicated by the status_id column. 

# In[ ]:


data_ndp = data.drop_duplicates(subset='status_id',
                               keep='last')
data_ndp.shape


# Now we have only 6997 observation in the dataset. Let's explore the status type and other attributes in the data to gain further insights. 

# In[ ]:


st_ax = data_ndp.status_type.value_counts().plot(kind='bar',
                                        figsize=(10,5),
                                        title="Status Type")
st_ax.set(xlabel="Status Type", ylabel="Count")


# Most of the sellers seem to be using a photo or video as status for the selling. A tiny portion of the users is depending on text status or URL/link for posting an advertisement. 

# In[ ]:


data_ndp.head(2)


# The num_reaction column seems to be a sum of following colums.
# 
# *  num_reaction = sum(num_likes, num_loves,num_wows,num_hahas,num_sads,num_angrys)
# 
# 
# Let's validate the assumption.

# In[ ]:


data_ndp['all_reaction_count'] = data_ndp.iloc[:,-6:].sum(axis=1)


# In[ ]:


data_ndp['reactio_match'] = data_ndp.apply(lambda x: x['num_reactions'] == x['all_reaction_count'],
                                           axis=1)


# In[ ]:


data_react_mismatch = data_ndp[data_ndp.reactio_match == False]
data_react_mismatch.shape


# There are nine observations where the assumption mentioned above is invalid. Let's examine the difference and reasons behind this. Since only nine observations are there, we can even remove these observations from the data due to inconsistency issues. But it is worthwhile to examine the reason. 

# In[ ]:


data_react_mismatch["diff_react"] = data_react_mismatch.num_reactions - data_react_mismatch.all_reaction_count


# In[ ]:


data_react_mismatch


# Let's check if the duplicate records cause the mismatch. We created a subset data consists only the duplicated values. Let's run a quick search by the status_id! 

# In[ ]:


data_react_mismatch[data_react_mismatch['status_id'].isin(list(duplicated_data.status_id.values))]


# And by looking at the numbers, it is evident that comments or shares do not contribute it. Values of those attributes are higher than the difference, and some of the status_is's are not even shared. 
# 
# As there is no data available to verify the correctness, we can go for
# 
# * Correct the value based on the interactions.
# * Drop the nine observations. 
# 
# I prefer to correct the values as part of this experiment before we proceed further. 

# In[ ]:


data_ndp.num_reactions = data_ndp.all_reaction_count


# In[ ]:


data_ndp['reactio_match'] = data_ndp.apply(lambda x: x['num_reactions'] == x['all_reaction_count'],
                                           axis=1)


# In[ ]:


data_ndp[data_ndp.reactio_match == False]


# Now all the reactions_count is matching based on the calculation logic. 

# In[ ]:


data_ndp.head(2)


# Let's create two variables to understand the reactions to comment and share ratio. Comments and shares show people may be interested and inquiring or maybe complaining. Shares activity indicates that users found it interesting, hence sharing it for other's benefits. 

# In[ ]:


data_ndp['react_comment_r'] = data_ndp.num_reactions/data_ndp.num_comments
data_ndp['react_share_r'] = data_ndp.num_reactions/data_ndp.num_shares
data_ndp.head()


# In[ ]:


data_ndp.react_comment_r.plot(kind='line',
                             figsize=(16,5))


# From the graph, we can see that there are many NaN or Inf values and extreme values in the reactions to comments ratio. The ratio becomes inf while the comments or shares are zero in the count.  The extreme values are something exciting. It may be an indication of data error or a trend in the data and worth investigating. 

# In[ ]:


data_ndp.replace([np.inf, -np.inf],
                 0.0,
                inplace=True)


# In[ ]:


data_with_p_reaction = data_ndp[(data_ndp.react_comment_r > 0) &
        (data_ndp.react_comment_r <= 2)]
data_with_p_reaction = data_with_p_reaction[["num_reactions","num_comments","react_comment_r"]]


# In[ ]:


data_with_p_reaction.shape
data_with_p_reaction.head()


# In[ ]:


data_with_p_reaction.react_comment_r.min(),data_with_p_reaction.react_comment_r.max()


# When comments are less than ten, the reaction to comment ratio becomes higher. It means it created impressions but may not be enough interest in the customer base. At the same time, we can see that the three interaction types in the data 'haha', 'angry,' and 'sad' are there. Knowing Facebook as a social platform, these reactions are expressed in extreme emotions or disappointed by the product. It is work exploring the positive reactions 'likes,' 'loves,' and 'wows.' We can create positive reactions and adverse reactions summary here. 
# 
# Positive Reactions = sum('likes,' 'loves,' and 'wows.' )
# 
# Negative Reactions = sum('haha', 'angry,' and 'sad' )
# 
# With the variables mentioned above, we can check if the reaction to comment ratio is higher for selling attempts with positive comments or negative comments. 

# In[ ]:


data_ndp.head(2)


# In[ ]:


data_ndp['postive_reactions'] = data_ndp.iloc[:,-10:-7].sum(axis=1)
data_ndp.head(2)


# In[ ]:


data_ndp['negative_reactions'] = data_ndp.iloc[:,-8:-5].sum(axis=1)
data_ndp.head(2)


# In[ ]:


data_ndp.plot.scatter(x='num_comments',
                      y='negative_reactions',
                     figsize=(16,5),
                     title="Number of Comments v.s Negative Reactions")


# In[ ]:


data_ndp.plot.scatter(x='num_comments',
                      y='postive_reactions',
                     figsize=(16,5),
                     title="Number of Comments v.s Positive Reactions")


# In[ ]:


data_ndp.num_comments.min(),data_ndp.num_comments.max()


# It looks like low comments and otherwise negative and positive, and reactions are there. Comments to positive responses are much higher than comments to negative. If we extract the respective comments and study the intent and sentiment, that could lead us to fascinating insights. 

# # Data Quality Issues and Resolutions
# 
# We found the following data quality issues and appropriate remedy implemented. 
# 
# 1) Duplicate records - There were 53 records duplicated, and we preserved the last records. 
# 
# 2) Calculated Columns value Mismatch  - The column [num_reactions](http://) column is created by summing the columns num_likes, num_loves,num_wows,num_hahas,num_sads,num_angrys. There was nine instanced where the values are not matching. The values were replaced with correct calculations. 
# 
# 
# # New Features and Rationale
# 
# As part of the analysis, we created six new features. They are :
# 
# all_reaction_count, reactio_match, react_comment_r, react_share_r, postive_reactions, negative_reactions.
# 
# all_reaction_count: This feature was generated to check the validity of data 'num_reactions'.  The logic used to create the column is num_reaction = sum(num_likes, num_loves,num_wows,num_hahas,num_sads,num_angrys) .
# 
# reactio_match: This is a bool column. If the values are False that means num_reactions and all_reactions_count values are different. 
# 
# react_comment_r: reactions to comments ratio. The logic for creating this variable is num_reactions/num_comments 
# 
# react_share_r: Reactiont to share ratio. The logic to create the variable is num_reactions/num_shares.
# 
# postive_reactions: This is the overall positve reaction count. Logic to generate the column : positive_reactions = sum(num_likes,num_loves,num_wows)
# 
# negative_reactions: This variable represents overall negative reactions. Logic to generate the columsn :  negative_reactions = sum(num_hahas, num_sads, num_angrys)
# 
# 
# # Clean Data
# 
# From the final data, we will exclude the columns all_reaction_count, reactio_match. These columns are created for verification and validation. The rest of the new columns can be removed based on the use case we are framing from the data. 

# In[ ]:


clean_data = data_ndp.drop(['all_reaction_count','reactio_match'],
                          axis=1)
clean_data.head()


# In[ ]:


clean_data.to_csv("clean_data_v1.0.csv",
                  index=False)


# # More to Analyze! 
# 
# One interesting factor that can be explored form the data is the impact of time in engagements. From the status_published, we can create a segment of a day such as early morning, morning, noon, evening, and night. Another new variable idea will be the days, such as working day or weekend. Generally, Facebook live sellers may be trying to earn a side income and selecting free time from work. The sellers may have a hunch on the best time they are getting audience and revenue. By creating this variable, we may be able to frame a use case. One can create a couple of new Machine Learning use cases from this new data. 
# 
# 
# The curators of the dataset, Nassim Dehouche, and Apiradee Wongkitrungrueng, used this data in a paper they presented in a Marketing Conference. The paper details are
# 
# Nassim Dehouche and Apiradee Wongkitrungrueng. Facebook Live as a Direct Selling Channel, 2018, Proceedings of ANZMAC 2018: The 20th Conference of the Australian and New Zealand Marketing Academy. Adelaide (Australia), 3-5 December 2018.
# 
# Before proceeding with the use cases, it is worth to read this paper. 
