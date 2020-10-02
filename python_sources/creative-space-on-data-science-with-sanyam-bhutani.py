#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import Image
Image("../input/ctds-poster/ctds_1.png")


#  <font size="+2" color="darkgreen"><i><b><center> "Dream big, </center></b></i></font> <font size='+2' color='blue'><i><b><center>work hard,</center></b></i></font>  <font size='+2' color='red'><i><b><center>stay positive,</center></b></i></font>  <font size='+2' color='indigo'><i><b><center>and live the journey of Data Science......"</center></b></i></font>

# # About the Host and CTDS.show :
#     
#    ### [Sanyam Bhutani](https://www.kaggle.com/init27) is a Machine Learning Engineer and AI content creater at [H20.ai](https://www.h2o.ai/)
#   > **LinkedIn Id** :      
#    https://www.linkedin.com/in/sanyambhutani/
#    
#   > **GitHub Id :**
#    https://github.com/init27
#    
#   > **Chai Time Data Science web link :**      
#    https://sanyambhutani.com/tag/chaitimedatascience/   
#    
#    **CHAI TIME DATA SCIENCE SHOW | CTDS.show :**
#              
#   > ***This show is a podcast series show by Sanyam Bhutani. He interviews with Kagglers, Researchers and    Data Science Practitioners.
#    All the episodes are available as video,audio,blog posts.***
#    
#         

# ### About me : 
# > #### This is my first kernel notebook and if you love my first kernel and EDA (Thanks to CTDS.show Datasets) so 
#    <font size="+0.1" color=chocolate ><b>please appreciate me by your UPVOTE.</b></font>

# <font size="+2" color="brown"><b>Libraries Needed For EDA :</b></font><br><br>
# > libraries used are plotly, seaborn, numpy, pandas, matplotlib, rcparams, re, string, nltk, counter, operator, stopwords, warnings.
# 
#  

# In[ ]:


import plotly
import plotly.express as pv
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style ="whitegrid")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import plotly.graph_objects as go
figg = go.Figure()
import re
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import operator




# <font size="+2" color="brown"><b>Importing datasets of CTDS.show</b></font><br><br>
# 

# In[ ]:


CTDS_episodes = pd.read_csv('../input/chai-time-data-science/Episodes.csv',parse_dates=['recording_date','release_date'])

YouTube_thumbnails = pd.read_csv('../input/chai-time-data-science/YouTube Thumbnail Types.csv')

Anchor_thumbnails = pd.read_csv('../input/chai-time-data-science/Anchor Thumbnail Types.csv')

Description_details = pd.read_csv('../input/chai-time-data-science/Description.csv')


#  <font size="+3" color="black"><i><b><center> CHAI TIME DATA SCIENCE statistics starts here...... </center></b></i></font>

# In[ ]:


length = len(CTDS_episodes["episode_id"])
print(f'Total number of episodes Till Now : {length}','\n')
length1 = len(set(CTDS_episodes['heroes_location']))
print(f'Total number of countries from where  Guest belongs : {length1}','\n')
country = list(set(CTDS_episodes['heroes_location']))
print(f'list of countries are given here below :\n \n{ country }','\n')


# In[ ]:


CTDS_episodes.head()


# # <font size="+2" color="darkyellow"><b>Gender Distribution details of ML Heroes:</b></font><br><br>  
# 
# #### How many are from male category and female category?
# #### How many missing values are in "heroes_gender" column ? 
# 
# > let's check through code and visualization below :)

# In[ ]:


# check for missing values in 'heroes_gender' column
gender = CTDS_episodes["heroes_gender"].isnull().sum()
print(f'Missing values in "heroes_gender" column is : {gender}')


# In[ ]:


CTDS_episodes['heroes_gender'].value_counts()


# In[ ]:


#plotting
rcParams["figure.figsize"] = 10,8
sns.countplot(x = CTDS_episodes["heroes_gender"],hue=CTDS_episodes["heroes_gender"])


# ### We can see through above bar_graph that there is more than 75% males are guest in the CTDS.show 
# ### Also 11 gender category (in numbers) are missing from the dataset so might be there is no episode of interviews related.
# 
# #### Now check following through code and visualization :-
# 1. how many people are from 'Kaggle'?
# 2. how many people are from 'Research'?
# 3. how many people are from 'Industry'?
# 4. What is in 'Other' category?
#    
#     Answer is below :)

# # <font size="+2" color="Darkyellow"><b>Category Distribution details of ML Heroes:</b></font><br><br>

# In[ ]:


#check for missing values in category column

category = CTDS_episodes["category"].isnull().sum()
print(f'Missing values in "category" column is : {category}')


# In[ ]:


CTDS_episodes['category'].value_counts()


# In[ ]:


#plotting

rcParams["figure.figsize"] = 10,8
sns.countplot(x = CTDS_episodes["category"],hue = CTDS_episodes["heroes_gender"]) 


#  #### As we see from graph that 'Other' part category is not there. Why?
#  #### Reason is simple because according to me, No person is interviewed in the 'Other' category part.There may be different topic for 11 episodes rather than to have a interview with person.
#  
# ### Also from visualizing the graph, we found that most of the `MALE` gender is from either Kaggle or Industry. Few of them are from Research. NO `FEMALE` is from Kaggle.
# 

# # <font size="+2" color="darkyellow"><b>Flavours of Tea Distribution details:</b></font><br><br>

# <h2 class="list-group-item list-group-item-action active"  aria-controls="home"><center>Part 1 --> Which is the most consumed 'Tea Flavour'?</center></h2>
# 

# In[ ]:


fig = pv.bar(data_frame=CTDS_episodes,x ="flavour_of_tea",color="flavour_of_tea",title = "Bar_Graph : ' Distribution Of Flavour of Tea'")
fig.show()


# ### We can see through plot that 'Masala Chai' and 'Ginger Chai' are consumed more than other flavours of tea available in dataset.

# <h2 class="list-group-item list-group-item-action active"  aria-controls="home"><center>Part 2 --> Analysing `Flavours Of Tea` with `Recording Time` of CTDS.show</center></h2>
# 

# In[ ]:


rcParams["figure.figsize"] = 20,10
sns.countplot(hue = CTDS_episodes["recording_time"],x = CTDS_episodes["flavour_of_tea"])


# ### `We found that 'Kesar Rose Chai' flavour is the only flavour consumed in night time.`

# # <font size="+2" color="darkyellow"><b>Heroes Location Distribution details:</b></font><br><br>

# In[ ]:


fig = pv.bar(CTDS_episodes,x = "heroes_location", title="Bar_Graph : ML heroes location info check")
fig.show()


# ###  We see from the plot that most of the ML heroes are from USA who interviwed by [Sanyam Bhutani](https://www.kaggle.com/init27) on CTDS.show

# <h2 class="list-group-item list-group-item-action active"  aria-controls="home"><center>---Let's check how many heroes are from USA---</center></h2>
# 

# In[ ]:


#counting ml heroes and identifying how many are from USA.

location_data = Counter(CTDS_episodes["heroes_location"])
sorted_location_data = dict(sorted(location_data.items(), key=operator.itemgetter(1),reverse=True))


# In[ ]:


fig = pv.funnel_area(names = list(sorted_location_data.keys()),values = list(sorted_location_data.values()),title="Heroes Location in Percentage") 
fig.show()


# ### *We figured out that `43.5%` (i.e. 37) heroes are from USA.
# #### Through this, Also we figured out that `12.9%`  locations are missing from the dataset.*

# <h2 class="list-group-item list-group-item-action active"  aria-controls="home"><center>---'Recording Time' Of the Show w.r.t `Heroes Location`---</center></h2>
# 

# In[ ]:


fig = pv.bar(CTDS_episodes,x = "heroes_location",color = 'recording_time', title="Bar_Graph : Recording Time Analysis")
fig.show()


# ### *We figure out that most of the episodes were recorded during night due to different time zones as most of the heroes are from USA.*
# #### I figured out some points listed below:
#    > 1. CTDS.show is a online interview show performed through videocalling with ML heroes. 
#    > 2. Flavour of Tea available only for host.
#    > 3. We see most of the episodes recorded in night so CTDS show is perfrom and recorded in India.
#    > 4. Host loves 'kesar rose chai' during night interviews, also he loves Ginger and Masala chai too in other interviews.
# 

# # <font size="+2" color="darkyellow"><b> YouTube Subscriber details:</b></font><br><br>

# In[ ]:


#checking for missing subscribers in dataset. 
ys = CTDS_episodes["youtube_subscribers"].isnull().sum()
print(f'Missing values in "youtube_subscribers" column is : {ys}')


# <h2 class="list-group-item list-group-item-action active"  aria-controls="home"><center>Which  hero's interview  is having the highest number of  increase in youtube subscribers ?  </center></h2>
# 

# In[ ]:


#sorting youtube subscribers in descending order.
highest_subs = CTDS_episodes.sort_values(by="youtube_subscribers",ascending=False)


# In[ ]:


rcParams["figure.figsize"] = 20,10
fig = pv.bar(x="heroes",y="youtube_subscribers",data_frame=highest_subs[:10],color = 'youtube_views',title="Number of Subscribers of CTDS show on Youtube")
fig.show()


# ### *Among all the heroes interviewed, a hero named `JEREMY HOWARD's` episode having the highest number of increase in youtube subscribers on CTDS show channel. Total youtube views on his episode is `4502`.*
# ## &
# #### *According to dataset given, `Parul Pandey` and `Abhishek Thakur` are at the 2nd and 3rd position respectively.*
# 
# 

# # <font size="+2" color="darkyellow"><b> Highest YouTube Impression Views :</b></font><br><br>

# In[ ]:


#sorting youtube impression views in descending order.
highest_views = CTDS_episodes.sort_values(by="youtube_impression_views",ascending=False)


# In[ ]:


rcParams["figure.figsize"] = 20,10
fig = pv.bar(x="heroes",y="youtube_impression_views",color = 'release_date',data_frame=highest_views[:10],title="Number of impression views of ML heroes")
fig.show()


# ### *In this case also,a hero named `JEREMY HOWARD's` episode having the highest number of youtube impression views on episode of CTDS.show channel released on 08-12-2019.*
# 

# # <font size="+2" color="darkyellow"><b> Highest YouTube Likes :</b></font><br><br>

# In[ ]:


#sorting youtube likes in descending order.
highest_likes = CTDS_episodes.sort_values(by="youtube_likes",ascending=False)


# In[ ]:


rcParams["figure.figsize"] = 20,10
fig = pv.bar(x="episode_id",y="youtube_likes",color= 'heroes',data_frame=highest_views[:10],title="Number of YouTube Likes of CTDS.show on YouTube (Episodes comparison) ")
fig.show()


# ### *`Episode 27` `(E27)` or`Jeremy Howard's` episode has the highest number of youtube likes on CTDS.show channel.* 
# 

# # <font size="+2" color="darkyellow"><b> Counting on YouTube Dislikes :</b></font><br><br>
# 

# In[ ]:


rcParams["figure.figsize"] = 20,10
fig = pv.line(CTDS_episodes,y= "youtube_dislikes" ,x = 'episode_id', title="YouTube Episode's dislike graph ")
fig.show()


# #### We see that most of the episodes have zero or one dislike by viewer.
# #### It means that the episodes content are valuable and attractive.

# # <font size="+2" color="darkyellow"><b> Subscriber's Growth Rate on CTDS youtube channel :</b></font><br><br>
# 

# In[ ]:


fig = pv.line(data_frame=CTDS_episodes,x = "release_date",y="youtube_subscribers",title="Growth Rate of Subscribers w.r.t Release date on youtube channel")
fig.show()


# In[ ]:


# total subscribers on youtube channel
tot_subs = CTDS_episodes['youtube_subscribers'].sum()
print(f'Total subcscribers on youtube channel is : {tot_subs}' )


# # <font size="+2" color="darkyellow"><b> Episode's Duration details:</b></font><br><br>
# 

# In[ ]:


fig = pv.bar(CTDS_episodes,y = "episode_duration",x = "episode_id",color = 'category', title="Episode duration distribution w.r.t category and episode ID")
fig.show()


# ### Episode 23 from kaggle category have the longest interview duration. This suppose to be a good interview session for Sanyam Sir.
# ##### From the above bar plots, we can't able to see any constant pattern or same duration bars in episode duration plot , It depends on interview session, how well it goes matter.

# # <font size="+2" color="darkyellow"><b> Episodes Generating CTR:</b></font><br><br>
# 

# ### Let's find out which episode have highest Click Through Rate

# In[ ]:


#sorting and analysing through visualising the graph
highest_ctr = CTDS_episodes.sort_values(by = 'youtube_ctr',ascending = False)

pv.bar(data_frame=highest_ctr,x="episode_id",y="youtube_ctr",color = 'category',title="bar_graph : 'Episodes with CTR'")


# #### `Episode 19` of `'Industry'` category has the highest youtube CTR.
# #### It means that Episode 19 have well performing keywords and ad which attract viewers.
#         CTR = (`CLICKS`/ `IMPRESSIONS`)

# # <font size="+2" color="darkyellow"><b> Average Watch Duration of Episodes:</b></font><br><br>
# 

# In[ ]:


aw_duration = CTDS_episodes.sort_values(by = 'youtube_avg_watch_duration',ascending = False)
pv.bar(data_frame= aw_duration,x="episode_id",y="youtube_avg_watch_duration",color = 'category',title="bar_graph : 'Average Watch Duration")


# #### Episode 63 of `INDUSTRY` category have highest youtube average watch duration of 584. 

# # <font size="+2" color="darkyellow"><b> Podcast Streaming Platforms details:</b></font><br><br>
# 

# <h2 class="list-group-item list-group-item-action active"  aria-controls="home"><center>Anchor Plays comparison :</center></h2>
# 

# In[ ]:


fig = pv.area(data_frame=CTDS_episodes,x="release_date",y="anchor_plays",title =' Anchor plays according to release date')
fig.show()


# In[ ]:


fig = pv.bar(data_frame=CTDS_episodes.sort_values(by = 'anchor_plays',ascending = False)[:25],x="heroes",y="anchor_plays",title = 'Anchor plays according to Heroes name',color = 'anchor_plays')
fig.show()


# ### Anchor plays shows that `JEREMY HOWARD` has more popularity than any other hero.

# <h2 class="list-group-item list-group-item-action active"  aria-controls="home"><center>Spotify comparison :</center></h2>
# 

# ## `Spotify listeners with respect to heroes name: ` 

# In[ ]:


fig = pv.bar(data_frame=CTDS_episodes.sort_values(by = 'spotify_listeners',ascending = False)[:25],x="heroes",y="spotify_listeners",title = 'Analysing Spotify listeners w.r.t heroes name',color = 'spotify_listeners')
fig.show()


# ### Abhishek Thakur has highest number of spotify listeners that is 456. Thus have more popularity on Spotify.
# 

# <h2 class="list-group-item list-group-item-action active"  aria-controls="home"><center>Apple Podcast details :</center></h2>
# 

# In[ ]:


fig = pv.bar(data_frame=CTDS_episodes.sort_values(by = 'apple_listeners',ascending = False)[:25],x="heroes",y="apple_listeners",title = 'Analysing Apple listeners w.r.t heroes name',color = 'apple_listeners')
fig.show()


# ### Jeremy Howard has highest number of Apple listeners that is 96. Thus have more popularity on Apple Podcast.
# 

# In[ ]:


fig = pv.bar(data_frame=CTDS_episodes.sort_values(by = ['apple_avg_listen_duration'],ascending = False),x="episode_id",y="apple_avg_listen_duration",title = 'Analysing Apple Average Listen duration w.r.t Episodes',color = 'category')
fig.show()


# ### Episode 23 has longest duration on Apple Podcast that is 5122.
# 

# # <font size="+2" color="darkyellow"><b> Text Data Analysis details :</b></font><br><br>
# 

# In[ ]:


Description_details.head(10)


#  #### Now we will clean text by removing special characters and numbers from the description dataset.

# #### `Make all the text words in  lowercase, remove the text in square brackets,links,punctuations and words containing numbers.`

# In[ ]:


def cleaner(text):
    text = text.lower()
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    return text
Description_details["description"] = Description_details["description"].apply(cleaner)


# ### cleaning and parsing the words below here :

# In[ ]:


def text_preprocessing(text):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    nopunc = cleaner(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    complete_txt = ' '.join(remove_stopwords)
    return complete_txt


# In[ ]:


Description_details["description"] = Description_details["description"].apply(text_preprocessing)
Description_details.head(10)


# ### `THAT's PERFECT.`

# <h2 class="list-group-item list-group-item-action active"  aria-controls="home"><center>Most common words in the given text data :</center></h2>
# 

# In[ ]:


Description_details['new_list'] = Description_details['description'].apply(lambda x:str(x).split())
top_words = Counter([item for subset in Description_details['new_list'] for item in subset])
temp_data = pd.DataFrame(top_words.most_common(20))
temp_data.columns = ['Common_words','count']
temp_data.style.background_gradient(cmap='YlOrBr')


# #### *Through Visualization :* 

# In[ ]:


fig = pv.bar(temp_data, x="count", y="Common_words", title='Most Commmon Words in Description Data',color='Common_words')
fig.show()


# ### The word 'Podcast' is the most common word in the given description data.

# # <font size="+2" color="darkyellow"><b> Generating Word Cloud  :</b></font><br><br>
# 

# In[ ]:


def generate_word_cloud(text):
    wordcloud = WordCloud(width = 2000,height = 1000,background_color = 'lightyellow').generate(str(text))
    fig = plt.figure(figsize = (30, 20),facecolor = 'k',edgecolor = 'k')
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
description_text = Description_details.description[:100].values
generate_word_cloud(description_text)


# ### <font size="+2" color="indigo"><b> Thanks for seeing my first kernel notebook with patiently.</b></font><br><br>
# 
# ### <font size="+3" color="green"><b> *Happy Kaggling..!!!!*</b></font><br><br>
# 
# 

# In[ ]:




