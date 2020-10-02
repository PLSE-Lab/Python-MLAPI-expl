#!/usr/bin/env python
# coding: utf-8

# The purpose of this kernel is to show how to load the data, and perform basic analysis. Let's get started.

# **Note:** I use Altair for the visualizations for this kernel. On the public side, it seems that the plots don't render. You might have to fork and run the kernel yourself to see all of the plots.

# In[ ]:


import pandas as pd
import numpy as np
import scipy as sp
import altair as alt
import matplotlib.pyplot as plt
import time
import pandasql as ps

import os
os.chdir("/kaggle/input")
print(os.listdir())


# In[ ]:


#Lets make our console outputs more nice, by applying some settings.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
alt.renderers.enable('notebook')
alt.data_transformers.enable('default', max_rows=None)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Support functions for the analysis go here:
#Solution comes from: https://stackoverflow.com/questions/34122395/reading-a-csv-with-a-timestamp-column-with-pandas
def date_parser(string_list):
    return [time.ctime(float(x)) for x in string_list]


# In[ ]:


#First, lets load the data:
#The header throws off our parser, so just ignore and write manually.
acroDF = pd.read_csv("./reddit-scitech-acronyms/acronyms.csv",skiprows=1,parse_dates=[1],date_parser=date_parser,
                     names=["commID","time","user","subreddit","acronym"])

#Nice date formats!
acroDF.head(10)


# In[ ]:


acroDF.dropna(inplace=True) #There are some NA acronyms.
acroDF.count() #Now every row is fully defined. 
acroDF.describe() #Some basic information.


# So we have 142965 acronyms, and 13212 are unique. There are 55 subreddits, and 51538 users logged in the 
# table. Timestamps are almost unique. There are >1000 of them that are not considered unique by our descriptor
# function. Lets check this out first.
# 
# We construct a query to find timestamps that are not unique, to see their associated information.

# In[ ]:


q1 = """SELECT t1.time,t1.user,t2.user FROM acroDF AS t1 INNER JOIN acroDF AS t2 ON (t1.time = t2.time) AND (t1.commID != t2.commID) """
timeQueryDF = ps.sqldf(q1, locals())
timeQueryDF.head(10)


# Clearly, two users can both post acronyms within a 1s interval. Also, since the timestamp is mined from the posting of the comment itself, any user that posts more than one acronym in the same post will also show up in the dataset above.

# Lets make a quick histogram of the total acronyms per subreddit. First, lets make an aggregated dataframe, 
# followed by an altair plot.

# In[ ]:


subredCount = acroDF.groupby("subreddit",as_index=False).count()
subredCount.drop(["commID","time","user"],axis=1,inplace=True)
subredCount.head(5)


# In[ ]:


srList = (subredCount.sort_values("acronym", ascending=False))["subreddit"].tolist()
alt.Chart(subredCount).mark_bar().encode(
alt.X('subreddit:N',sort=srList),
alt.Y('acronym:Q'))


# The BitcoinMarkets and Hardware subreddits have abnormally high counts. 
# 
# Are the users just prone to acronym dropping? 
# 
# What factors would affect an acronym being mentioned in a random post? 
# 
# First, lets assume that there are no causal factors, and that we have a random phenomenon with some underlying distribution. The distribution of a random acronym appearing in the next comment is p(X), where X ~ Poisson($\lambda$), with lambda being a very small number. Then three factors would matter for the **total number of acronyms observed.** 
# 
# 1) Age of subreddits: if an acronym is randomly dropped in conversation, then over a long time frame we should see more of them.
# 
# 2) Number of Subscribers: for similar reasons, if more subscribers can issue more comments, then we would see higher total counts.
# 
# 3) Number of Posts: '' ''
# 
# Note that we can get (1) and (2) for each subreddit using the PRAW library, but it is not easy to get (3). To access submissions, you need to use a ListingGenerator. We would have to exhaust a listing generator to get an estimate for this, which we won't attempt here. I have gathered (1) and (2) for each subreddit, outside this kernel. The dataset is loaded below, and a ratio column is derived and appended to the dataframe:
# 

# In[ ]:


#will put things in alphabetical order, by default
subredDF = acroDF.groupby(by="subreddit",as_index=False).count()
#dont need these columns.
subredDF.drop(["commID","time","user"],axis=1,inplace=True)
#subredDF = subredDF.sort_values(by="user",ascending=False)

srStatDF = pd.read_csv("./subredditstats/subredditstats.csv",skiprows=1,parse_dates=[2],
                     date_parser=date_parser,names=["subreddit","subscribers","utc_created"])

srStatDF.sort_values("subreddit",ascending=True,inplace=True)

#when we add columns, they are series. So entries will be matched by indices. Looking at the two columns above,
#one has non-ascending indices, so lets reset them
srStatDF.reset_index(drop=True,inplace=True)

srStatDF["acroCount"] = subredDF["acronym"]
#add the acronym count column.
#derive  an acronym ratio.
def f(x,y): #assume we dont have y=0!
    return (x/y)
#we use the clever unpacking notation (*x), because x is typed as a 2ple. we do this down the columns
srStatDF['acroRatio'] = srStatDF[['acroCount','subscribers']].apply(lambda x: f(*x), axis=1)
#display the full datatable
srStatDF.head(10)


# Now lets make some Altair charts. Lets plot histograms by subscriber counts, and acroRatio.

# In[ ]:


#Lets make our altair histogram for this.
#We need to sort our chart in altair by counts.
#I can't seem to get the chart sorted correctly, so we will have to provide a list of subreddits explicitly,
#as per: https://altair-viz.github.io/user_guide/encoding.html?highlight=alt%20order#ordering-marks
#srStatDF.sort_values("subscribers", ascending=False).head(5)
srList = (srStatDF.sort_values("subscribers", ascending=False))["subreddit"].tolist()

alt.Chart(srStatDF, title="Number of Subscribers per Subreddit").mark_bar().encode(
x=alt.X('subreddit:N',sort=srList),
y=alt.Y('subscribers:Q'))

srList = (srStatDF.sort_values("acroRatio", ascending=False))["subreddit"].tolist()
alt.Chart(srStatDF, title="Ratio of Acronyms to Subscribers, per Subreddit").mark_bar().encode(
x=alt.X('subreddit:N',sort=srList),
y=alt.Y('acroRatio:Q'))


# We see a huge range in number of subscribers (a few hundred to 20M+). Our acronym ratio column reveals that some subreddits have high numbers of acronyms, relative to their size. This debunks the "Poisson Distribution" idea - we should see fairly constant ratios if it were true. 
# 
# This indicates that the (i) individual subscribers, (ii) their intereactions, and (iii) the topic of the subreddit probably has something to do with acronym counts. 
# 
# (iii) Is the easiest to tackle. The topics in question just happen to involve a lot of acronyms. Looking at the Suppliments and Programming Languages subreddits, one can easily confirm this by eye. Its not enriching to ask "why do the topics of QuantumPhysics and Suppliments have so many acronyms" - there are just a lot of complicated terms and acronyms are used to make conversation more brief.
# 
# Our dataset doesn't say much about (i) and (ii) however. 
# 

# Next, let's look at acronyms per subreddit. Lets choose three subreddits: genetics, neuralnetworks and Nootropics

# In[ ]:


#Support local function: our chained calls are too long.

def getcountdf(aDF,subreddit):
    temp = aDF.groupby("acronym",as_index=False).count()
    return temp.sort_values("subreddit",ascending=False).reset_index(drop=True)

acroFocus = acroDF.copy(deep=True) #leave intact for now.
acroFocus.drop(['commID','time',"user"], axis=1, inplace=True)
acroGroup = acroFocus.groupby("subreddit",as_index=False)

#Dict -> subDF return type.
#DataFrame -> Group -> DataFrame -> Dataframe
geneticsCountDF = getcountdf(acroGroup.get_group("genetics"), "genetics")
btcCountDF = getcountdf(acroGroup.get_group("BitcoinMarkets"), "BitcoinMarkets")
nnCountDF = getcountdf(acroGroup.get_group("neuralnetworks"), "neuralnetworks")

geneticsCountDF.head(5)


# In[ ]:


#Visualization: Bar Chart with top 75 acronyms for each subreddit.
#Same sorting issues as last time, lets do an explicit encoding.

nameList = (geneticsCountDF.sort_values("subreddit", ascending=False))["acronym"].tolist()
limit = 30

genChart = alt.Chart(geneticsCountDF[0:limit],title="Top Acronyms for Genetics Subreddit").mark_bar().encode(
y=alt.Y("acronym:N",sort=nameList[0:limit]),
x=alt.X("subreddit:Q")).properties(width=300)

nameList = (btcCountDF.sort_values("subreddit", ascending=False))["acronym"].tolist()

btcChart = alt.Chart(btcCountDF[0:limit],title="Top Acronyms for BitcoinMarkets Subreddit").mark_bar().encode(
y=alt.Y("acronym:N",sort=nameList[0:limit]),
x=alt.X("subreddit:Q")).properties(width=300)

nameList = (nnCountDF.sort_values("subreddit", ascending=False))["acronym"].tolist()

nnChart = alt.Chart(nnCountDF[0:limit],title="Top Acronyms for Neural Networks Subreddit").mark_bar().encode(
y=alt.Y("acronym:N",sort=nameList[0:limit]),
x=alt.X("subreddit:Q")).properties(width=300)

#Here we see the power of the grammar of graphics and Altair :)
genChart | btcChart | nnChart


# So above, we see the most popular terms in a given subreddit. You might have noticed some of the words (particularly in the BitcoinMarkets Chart) are not acronyms. Terms like "IMO" , "LOL" are general abbreviations that are *not topic specific*. Also, people may write words in all caps out of anger, or to express a strong opinion. 

# **In summary:** before a subset of this data is taken, a list of general acronyms and other small words (including conjunctions, prepositions, articles, pejoratives...) needs to be supplied to filter out non-topic specific terms.  

# END
