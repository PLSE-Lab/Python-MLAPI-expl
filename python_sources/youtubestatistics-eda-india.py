#!/usr/bin/env python
# coding: utf-8

# ### Exploratory Data analysis on Trending Youtube statistics data - INDIA 
# 
# * Please Note : It will be extremely motivating if you guys find it good . So please UPVOTE , if you like ; Also if possible , correct my mistakes and recommend some alternatives to improve this presentation . 

# #### Import the libraries

# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px

from datetime import datetime


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/youtube-new/INvideos.csv')


# ### Descriptive statistics 

# In[ ]:


# Peek at the data 
data.head(5)


# **Observation** : Following dataset looks a bit complicated with categorical , numerical , boolean and time series data . 

# In[ ]:


# Check the dimension of our data , dtypes etc...

print('Number of observations in hand : {}'.format( data.shape[0] ))
print('Number of columns : {}'.format( data.shape[1] )) 

print('Dtypes \n' , data.dtypes)


# In[ ]:


# statistical summary 

data.describe( include = 'all' ).T


# **Observation** : We could see a lot of Object datatypes dominating in our summary . Plus we could smell missing value on last column .

# In[ ]:


# missing value check

print(data.isnull().sum() )
print('Percentage share of Nullity in description  : ' , (561 / data.shape[0])*100 ,'%'  )


# In[ ]:


sns.heatmap( data.isnull() , cmap = 'viridis' , yticklabels= False , cbar = True )


# **Observation** : There are 561 missing values in column - 'description' . 

# In[ ]:


data['category_id'].value_counts()


# **Observation** : Category id 24 has the most occurences in our dataset followed by id '25'.

# In[ ]:


#since description and title almost matches in context , i would rather remove the description column .
#It already has great deal of null values , its better we drop it . No harm done :-)

Columns2Delete = [ 'thumbnail_link' , 'description' , 'tags'   ]
data4EDA = data.drop( labels = Columns2Delete , axis = 1  )
data4EDA.head()


# In[ ]:


data4EDA['publish_time'] = pd.to_datetime( data4EDA.publish_time , format= '%Y-%m-%dT%H:%M:%S' )
print( data4EDA.publish_time.dtypes )
data4EDA.head()


# #### Binning of likes , dislikes , views , comments for better interpretation

# In[ ]:


like_bins = [-1, 100000, 500000, 1000000, 5000000]
view_bins = [-1, 300000, 5000000, 10000000, 500000000]
dislike_bins = [-1, 100000, 500000, 1000000, 5000000]
comment_bins = [-1, 10000, 50000, 500000 , 1000000]


data4EDA['like_BandWidth'] = pd.cut( data4EDA.likes  , labels= ['Poor','Improving', 'Good', 'Very Good'] , bins= like_bins )
data4EDA['view_BandWidth'] = pd.cut( data4EDA.views  , labels= ['Poor','Improving', 'Good', 'Very Good'] , bins= view_bins )
data4EDA['dislikes_BandWidth'] = pd.cut( data4EDA.dislikes  , labels= ['Normal','Critical', 'Bad', 'Worse'] , bins= dislike_bins )
data4EDA['comment_BandWidth'] = pd.cut( data4EDA.comment_count  , labels= ['Poor','Improving', 'Good', 'Very Good'] , bins= comment_bins ) 


# In[ ]:


data4EDA['comment_BandWidth'] = data4EDA['comment_BandWidth'].astype('object') 
data4EDA.loc[ data4EDA['comment_BandWidth'].isna() , 'comment_BandWidth' ] = data4EDA.loc[ data4EDA['comment_BandWidth'].isna() , 'comment_BandWidth' ].astype('object').replace( np.NaN , 'Disabled' )


# **When you bin the comments , you will observe Nan values in Comment_bandwidth for values in comments_disabled == True . So that was a concern we had to deal with .**

# In[ ]:


data4EDA.head()


# **Data4EDA becomes our dataset for exploration**

# In[ ]:


# To check out what content is most avail in our dataset .
data4EDA['title'].value_counts()


# In[ ]:


data4EDA['video_id'].value_counts()


# **Column - 'video_id' is hard to keep track off**

# #### Visualisation

# In[ ]:


data4EDA['like_BandWidth'] = data4EDA['like_BandWidth'].astype('object') 
data4EDA['view_BandWidth'] = data4EDA['view_BandWidth'].astype('object') 
data4EDA['dislikes_BandWidth'] = data4EDA['dislikes_BandWidth'].astype('object') 

data4EDA.dtypes


# In[ ]:


data4EDA.isnull().sum()


# In[ ]:


sns.heatmap( data4EDA.corr() , annot = True )


# **Seems like there's correlation among few columns**

# In[ ]:


# distribution of views 
plt.figure( figsize= (30,10) )
sns.kdeplot( data = data4EDA.views , label = 'views' , shade = True ) 

# sclale = 10 crore 


# ### Observations from views
# * This looks like a Log distibution
# * Area under the pdf curve looks significant in the interval ( 0.0 , 0.2 ) , which means most of our observation has views lying in these ranges , Only very few datapoints have beyond that scale . 

# In[ ]:


plt.figure( figsize= (30,10) )
sns.kdeplot( data = data4EDA.likes , label = 'likes in million' , shade = True  ) 
sns.kdeplot( data = data4EDA.dislikes , label = 'dislikes in million' , shade = True  ) 

#scale : 1 million


# #### Plotting Likes vs Dislikes
# 
# * seems like area under pdf is very small for dislikes category compared to the likes column 
# * Its a log normal distribution

# In[ ]:


plt.figure( figsize= (30,10) )
sns.kdeplot( data = data4EDA.views , label = 'views' , shade = True ,  ) 
sns.kdeplot( data = data4EDA.likes , label = 'likes in million' , shade = True  ) 
sns.kdeplot( data = data4EDA.dislikes , label = 'dislikes in million' , shade = True  ) 

plt.xlim( ( 0 , 3e6) )
plt.ylim( ( 0 , 20e-8 ) )


# #### Likes x Dislikes x Views 
# * None of them is gaussian . 
# * Area under pdf of Views curve is very much leading compared to the other two variables , which indicates likes and views got to have a good relationship each other while dislike curve dies out .

# In[ ]:


# like_bins = [0, 100000, 500000, 1000000, 5000000] : { 'for reference' }
# labels= ['Poor','Improving', 'Good', 'Very Good']

px.box( data_frame= data4EDA , x = 'like_BandWidth' , y = 'views'  , color = 'video_error_or_removed' )


# #### Observations 
# * When the Like_bandwidth goes high , that is the our content getting more likes , Number of views are also getting better 
# * Most videos that get removed are the ones that received lesser attention in terms of likes . Mostly indicates that content doesnt deserve much likes . So one inference is improve the quality of the video content enough to grab people's likes . 

# In[ ]:


#dislike_bins = [0, 100000, 500000, 1000000, 5000000]
#labels= ['Normal','Critical', 'Bad', 'Worse']

px.box( data_frame= data4EDA , x = 'dislikes_BandWidth' , y = 'views' , color = 'video_error_or_removed' )


# **Observation** 
# * We could observe a fact here ! 
# * As long as as the number of views gets increasing , till 60M views , the share of dislikes distribution is absolute zero . But for videos with exceptional number of views have not invited any critics out there in public . 

# In[ ]:


# Double click of the legends to isolate the plots you wish 

px.line( data_frame= data4EDA , x = 'trending_date' , y = 'views' , color= 'category_id' )


# **Observations** 
# 
# * cat_id 1 : there's good peak in no. of views on the descent of November , followed after that there is a gentle demand in this video .
# 
# * cat_id 24 : the highes peak in views . After the initial surge , it roughly crossed 20 M views .
# 
# * cat_id 25 : We had highest count in cat_id 25 from descriptive statistics session . There's a growth in views during early 2018 compared to other time period .
# 
# 
# 

# In[ ]:


px.line( data_frame= data4EDA , x = 'publish_time' , y = 'views' , color= 'category_id' )


# In[ ]:


from wordcloud import WordCloud, STOPWORDS
text = data4EDA.title.values
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure( figsize = (30, 20),facecolor = 'k', edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# **Observation** 
# 
# * Most of the frequent words seen in our dataset is mostly native , words like verma , parmish , allu makes it clear . 

# In[ ]:


px.violin( data_frame = data4EDA , y = 'views' ) 


# **Observation** 
# 
# * Distribution of views is significantly large in interval ( 4024 , 1.8M ) , with a median of 304 k views . 
# * Thus we could divide our dataset above 304k views trending ,  anything below that could be treated as 'Fading' for easier interpretation .
# * This assumption is strictly subjected to our dataset .

# In[ ]:


ColumnsWeCareAbout = ['title','view_BandWidth','like_BandWidth','dislikes_BandWidth','comment_BandWidth'] 
data4EDA[ColumnsWeCareAbout]


# In[ ]:


pd.crosstab( index = [ data4EDA.like_BandWidth , data4EDA.dislikes_BandWidth , data4EDA.comment_BandWidth ] , 
           columns = data4EDA.view_BandWidth ).style.background_gradient(cmap='summer_r')


# #### Summary 
# 
# * Despite the like bandwidth is poor , but if the dislike amount is normal and less comments , There is a good chance that video is most probably very poor in reach or improving . 
# 
# * But is like bandwidth is improving , you could observe the video content to be acquiring good reach in internet . 
# 
# * Anomaly : if your video is very good in reaching exceptional number of likes , despite having bad stats in dislikes and comments , there is a chance your video get good attention . 
# 
# * So Likes over rule the dislikes and comments section . 

# In[ ]:


pd.crosstab( index = [ data4EDA.like_BandWidth , data4EDA.comments_disabled , data4EDA.ratings_disabled ] , 
           columns = data4EDA.view_BandWidth ).style.background_gradient(cmap='summer_r')


# #### Observation :
# 
# * So Most likely the guy might have disabled comments and ratings , indicated the video quality is poor . 
