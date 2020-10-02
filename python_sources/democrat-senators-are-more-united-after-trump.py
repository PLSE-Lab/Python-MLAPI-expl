#!/usr/bin/env python
# coding: utf-8

# #### Table of Contents
# * [1. Introduction](#intro)
# * [2. Data scraping with BeautifulSoup](#scraping)
# * [3. Data Preparation and Cleaning](#cleaning)
# * [4. Clustering and visualization](#clustering)
# * [5. US Senator's clustering comparison in age of Obama vs Trump](#compare)
# 
# ### 1. Introduction <a name="intro"></a>
# [**K-Means Clustering**](https://en.wikipedia.org/wiki/K-means_clustering) is a common approach in clustering data into similar groups. It aims to partition `n` observations into `k` groups, in which each observation belongs to the group with the nearest mean.
# 
# In this project, I'd like to use this approach to cluster United States Senators into two clusters. [US Senate](https://www.senate.gov/) composes of 100 members, each 50 states being equally represented by two senators. There are three political party affiliations in the Senate: Democrats, Independatnts, and Republicans. The current US Congress is the 115th one, which runs from Jan 03, 2017 until Jan 03, 2019.
# 
# Per US Senate:
# > A roll call vote is a vote in which each senator votes "yea" or "nay" as his or her name is called by the clerk, so that the names of senators voting on each side are recorded.
# 
# This data is [available online](https://www.congress.gov/roll-call-votes).
# 
# In this project, first I will use data scraping approaches to extract the votes data, then I will apply K-Means clustering approach to cluter senators based on their voting behaviour. The rest of this Kernel is organized as follows:
# 
# 

# ### 2. Data scraping with BeautifulSoup <a name="scraping"></a>

# In[ ]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import requests
from bs4 import BeautifulSoup


# [`BeautifulSoup`](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) library of Python is a powerful tool for extracting online data from HTML and XML sources.
# 
# Here, we first use `Request` library to load the xml data into our kernel. However, before we start doing that, by looking at the urls on senate.gov, I realized they only change in the last 5 digits. For instance, this is the url for the 15th roll call votes: 
# https://www.senate.gov/legislative/LIS/roll_call_votes/vote1152/vote_115_2_00015.xml
# 
# and this the one for 14th roll call votes:  
# https://www.senate.gov/legislative/LIS/roll_call_votes/vote1152/vote_115_2_00014.xml
# 
# Therefore, we can create a list that contains all the urls for first and second session of 115th Congress. In order to do so, I will regular expressions (regex) to replace the last two digits and create a list of urls. I recommend explorig [regex101.com](regex101.com).
# 
# #### 2.1. Creating list of urls

# In[ ]:


urls=[]

#first url in the first session of 115th Congress. It runs from Jan 03 untill Dec 21, 2017.
first_url_1stsession='https://www.senate.gov/legislative/LIS/roll_call_votes/vote1151/vote_115_1_00001.xml'

for i in range(1,326):
    url=re.sub('0+([0-9]+).xml$',str(i).zfill(5)+'.xml',first_url_1stsession)
    urls.append(url)


# And now the remaining votes during 2nd session (Year 2018):

# In[ ]:


first_url_2ndsession='https://www.senate.gov/legislative/LIS/roll_call_votes/vote1152/vote_115_2_00001.xml'

for i in range(1,16):
    url=re.sub('0+([0-9]+).xml$',str(i).zfill(5)+'.xml',first_url_2ndsession)
    urls.append(url)

# First 3 urls of final list    
urls[:3]


# #### 2.2. Data scraping
# 
# Here, I define a fuction that loads each url, creates an instance of BeautifulSoup, and then uses the xml tags to extract the year and number of roll call vote. Then creates a unique name for that voting day (year_number). Eventually, this function reads the names of senators and their votes and stores them in a dictionary.
# 
# During this process, I realized a few of the urls cause errors. I used try and except to flag those problamatic urls.

# In[ ]:


def votes_scraper(urls):
    votes_dict={}
    for url in urls:
        try:
            names=[]
            votes=[]
            doc_name=''
            
            #loading url contents 
            page=requests.get(url)
            
            #instatiating BeautifulSoup
            soup=BeautifulSoup(page.content, 'html.parser')
            
            #creating unique keys for dictionary
            cong_year=soup.find('congress_year').text
            vote_num=soup.find('vote_number').text
            doc_name=cong_year+'_'+vote_num.zfill(3)
            
            #loading senator names and their votes
            names_tag=soup.find_all('member_full')
            votes_tag=soup.find_all('vote_cast')
            names=[names_tag[i].text for i in range(len(names_tag))]
            votes=[votes_tag[i].text for i in range(len(votes_tag))]
        
            #storing data in a dictionary
            votes_dict[doc_name]={k:v for k, v in zip(names, votes)}
        except:
            print(url)
            pass
        
    return votes_dict


# Now let's call our function on list of urls. I have done it and stored the data in `votes_115thCongress.csv`. This file is available in the data tab of this Kernel.

# In[ ]:


#Note: at the time of this analysis, I recieve an error on getting access to the urls. The error...
#...says there is a potential security risk. I assume it is becuase this function loads a few...
#...urls in a short perio of time.

#Uncomment in your own code
#votes=votes_scraper(urls)


# ### 3. Data preparation <a name="cleaning"></a>

# First, I'd like to store this data in a data frame. Pandas does a pretty good job in tranforming a dictionary to a dataframe. In this case, from OCT 2017 and beyond, there has been three changes in the seats. Pandas will include the new members and replace the voting values of old senate members to nan.  
# Also senators names and their party affiliations are in the index column. Let's add a column with their party affiliations.

# In[ ]:


# Feel free to uncomment following lines after executing the above cell. 
#votes_df=pd.DataFrame(votes)

votes_df=pd.read_csv('../input/../input/votes_115thCongress.csv')
votes_df.head()


# In[ ]:


# replacing null values with Not Voting
votes_df=votes_df.replace('Not Voting',np.nan)

# extracting party affiliations 
votes_df['Party']=votes_df['index'].apply(lambda x: re.findall('\(([A-za-z])',x)[0])

# reseting the index. This will add a new column with our old index's.
votes_df=votes_df.set_index('index',drop=True)

#Percent of missing data in each each row.
num_na_row=votes_df.isnull().sum(axis=1).sort_values(ascending=False)/votes_df.shape[1]*100
num_na_row.head()


# A little bit of Googling, Jones join Senate recently (Big news for Alabama). Smith is in the same situation. Jeff Sessions left Senate to serve as Attorney General back in 2017. The voting history for these 3 senators are not alot. Therefore, I will drop these from our dataframe.

# In[ ]:


drop_sens=['Jones (D-AL)','Smith (D-MN)','Sessions (R-AL)']
votes_df=votes_df.drop(drop_sens,axis=0)


# In[ ]:


votes_df.shape


# Now, I'd like to map the text votes to numerical votes, where 1 represents Yea, 0 Nay, and 0.5 for Abstain. Please note, somewhere along doing this analysis, I realized there are some 'Present' values in votes. I consider these as Abstain (and hence convert them to 0.5) as there are not a lot of them (I actually checked).

# In[ ]:


map_dict={'Yea':1,
         np.nan:0.5,
         'Present':0.5,
         'Nay':0}
votes_df_numeric=votes_df.replace(map_dict)

#Let's check if there is any non numerical values in our dataframe!
votes_df_numeric.select_dtypes(include=['object']).head()


# Sweet! The only nun-numerical values are index and party! let's start the clustering process.

# ### <a names="clustering"></a>4. Clustering and visualization
# 
# First, let's take look at the numner of senators in each party.

# In[ ]:


votes_df_numeric['Party'].value_counts()


# Our goal is to cluster US Senators based on their voting behaviour. I will use K-means clustering with k=2, as there are two major political party in the US.

# In[ ]:


from sklearn.cluster import KMeans

X=votes_df_numeric.iloc[:,1:(votes_df_numeric.shape[1]-1)]
kmeans= KMeans(n_clusters=2, random_state=1)

senator_distances=kmeans.fit_transform(X)

#labeling each senator based on the kmeans algorithm.
labels=kmeans.labels_

pd.crosstab(votes_df_numeric['Party'],labels)


# Interestingly, Independents behave as Democrats. I'd like to visualize the distances of each senator in these two clusters.

# In[ ]:


dis=pd.DataFrame(senator_distances)
dis.columns=['Distance from 1st cluster','Distance from 2nd cluster']
dis['Actual_Party']=votes_df_numeric.reset_index()['Party']


# In[ ]:


sns.set_style("dark")
sns.set_context("talk")
sns.lmplot(x='Distance from 1st cluster',y='Distance from 2nd cluster',data=dis,
           hue='Actual_Party',scatter=True,fit_reg=False,size=6, scatter_kws={"s": 100},
           legend=False,palette=sns.color_palette(['red','blue','gray']))
plt.title('Distance from clusters',fontsize=20)
plt.legend(loc=0,title='Actual Party',fontsize =12)
plt.show()


# Let's see who are the extremists. I will do that by creating a rating system :  
# `extremism rating=(distance from 1st cluster)^3 + (distance from 2nd cluster)^3`

# In[ ]:


extremism=((senator_distances)**3).sum(axis=1)
votes_df_numeric['extremism']=extremism
votes_df_numeric.sort_values('extremism',inplace=True,ascending=False)
votes_df_numeric.head(10)


# Wow! It seems like Democrats are acting more extreme in their voting behaviours. Look who's at the top of the list. Gillibrand, Warren, and Sanders (an Independent senator).

# ### <a names="compare"></a>5. US Senator's clustering comparison in age of Obama vs Trump

# A Similar analysis could be done for the behaviour of senator voters for before and after the new administration takes over the office. Let's extract the data from the senate.gov website, and perform the same exact analysis.

# In[ ]:


urls_obama=[]

first_url_1stsession_113='https://www.senate.gov/legislative/LIS/roll_call_votes/vote1131/vote_113_1_00001.xml'
for i in range(1,292):
    url=re.sub('0+([0-9]+).xml$',str(i).zfill(5)+'.xml',first_url_1stsession_113)
    urls_obama.append(url)

first_url_2ndsession_113='https://www.senate.gov/legislative/LIS/roll_call_votes/vote1132/vote_113_2_00001.xml'
for i in range(1,367):
    url=re.sub('0+([0-9]+).xml$',str(i).zfill(5)+'.xml',first_url_2ndsession_113)
    urls_obama.append(url)
    
first_url_1stsession_114='https://www.senate.gov/legislative/LIS/roll_call_votes/vote1141/vote_114_1_00001.xml'
for i in range(1,340):
    url=re.sub('0+([0-9]+).xml$',str(i).zfill(5)+'.xml',first_url_1stsession_114)
    urls_obama.append(url)

first_url_2ndsession_114='https://www.senate.gov/legislative/LIS/roll_call_votes/vote1142/vote_114_2_00001.xml'
for i in range(1,164):
    url=re.sub('0+([0-9]+).xml$',str(i).zfill(5)+'.xml',first_url_2ndsession_114)
    urls_obama.append(url)


# ** Note**:  
# Unfortunately, at the time of this analysis, the same security issue occured extracting the data. Roll call votes are publicly available, but I assume security issue is due to extracting so many data at the same time.
# 
# You can compare the results of clustering 115th Congeress with the one published [here](https://www.dataquest.io/blog/k-means-clustering-us-senators/) for 114 Congeress. It is important to note that senators tend to unite more after the new administation took the office last year.

# In[ ]:




