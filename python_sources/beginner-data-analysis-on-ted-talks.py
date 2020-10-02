#!/usr/bin/env python
# coding: utf-8

# # Data Analysis on Ted Talks

# ## Introduction
# 
# <b>TED talks</b> is a platform for speakers to present their ideas mainly <b>technology, entertainment, and design</b> (<b>TED</b>) at TED Conference. <br>
# The maximum length of each talk is limited to 18minutes. <[Source](https://whatis.techtarget.com/definition/TED-talk)><br>
# The dataset used has 2550 rows of Ted Talks data until 2017. <br>
# Exploratory data analysis is performed on the dataset to get insights about the Ted Talks. <br>
# As I'm still a beginner in data analysis, therefore I'm welcoming any suggestion to help improve the data analysis of this notebook.  Your suggestions will be greatly appreciated.

# ## Import libraries and dataset

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import datetime
from collections import Counter
import ast
import re
from PIL import Image
from wordcloud import WordCloud
import requests
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import os
from os import path


# In[ ]:


# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
# d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()


# In[ ]:


df = pd.read_csv('../input/ted-talks/ted_main.csv')
df.head()


# Check number of rows and column in ted_main dataset:

# In[ ]:


df.shape


# Identify missing values within the dataset:

# In[ ]:


df.isnull().sum()


# The dataset has no any missing value

# View basic statistical details of ted_main dataset:

# In[ ]:


df.describe()


# Identify the datatype of each feature:

# In[ ]:


df.dtypes


# ## Data Exploration

# <li><b>name</b>: The official name of the TED Talk. Includes the title and the speaker. <br>
# <li><b>title</b>: The title of the talk <br>
# <li><b>description</b>: A blurb of what the talk is about. <br>
# <li><b>main_speaker</b>: The first named speaker of the talk. <br>
# <li><b>speaker_occupation</b>: The occupation of the main speaker. <br>
# <li><b>num_speaker</b>: The number of speakers in the talk. <br>
# <li><b>duration</b>: The duration of the talk in seconds. <br>
# <li><b>event</b>: The TED/TEDx event where the talk took place. <br>
# <li><b>film_date</b>: The Unix timestamp of the filming. <br>
# <li><b>published_date</b>: The Unix timestamp for the publication of the talk on TED.com <br>
# <li><b>comments</b>: The number of first level comments made on the talk. <br>
# <li><b>tags</b>: The themes associated with the talk. <br>
# <li><b>languages</b>: The number of languages in which the talk is available. <br>
# <li><b>ratings</b>: A stringified dictionary of the various ratings given to the talk (inspiring, fascinating, jaw dropping, etc.) <br>
# <li><b>related_talks</b>: A list of dictionaries of recommended talks to watch next. <br>
# <li><b>url</b>: The URL of the talk. <br>
# <li><b>views</b>: The number of views on the talk. <br>

# Convert film_date and published_date into datetime format

# In[ ]:


df['film_date'] = df['film_date'].apply(lambda x: datetime.date.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
df['published_date'] = df['published_date'].apply(lambda x: datetime.date.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))


# In[ ]:


df.sample(5)


# In[ ]:


df['film_date'], df['published_date'] = pd.to_datetime(df['film_date']), pd.to_datetime(df['published_date'])


# ### How many Ted Talks have been filmed and published?

# In[ ]:


#Number of ted talks published or filmed by year
pub_year=df['published_date'].dt.year.value_counts().sort_index()
film_year=df['film_date'].dt.year.value_counts().sort_index()

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(15,6))
film_year.plot(kind='bar', ax=ax1)
pub_year.plot(kind='bar', ax=ax2)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
ax1.set_xlabel('Filmed Year')
ax2.set_xlabel('Published Year')
for i, v in enumerate(film_year):
    ax1.text(i-0.25,v+2, str(v),color='black',fontweight='bold')
for i, v in enumerate(pub_year):
    ax2.text(i-0.15,v+2, str(v),color='black',fontweight='bold')
ax1.title.set_text('Number of Ted Talks Filmed By Year')
ax2.title.set_text('Number of Ted Talks Published By Year')
plt.subplots_adjust(bottom=0, top=2)
plt.show()


# <li>It can be observed that the ted talks videos starts filming since 1972, but only start to publish at 2006. <br>
# <li>Number of filmed ted talks increased drastically since 2009.<br>
# <li>Most ted talks videos were published on year 2012.

# ### Which Ted Talks videos have the most first level comments?

# In[ ]:


# Which video received greater number of first level comments
df_comm = df[['main_speaker','title','published_date','comments']].sort_values(by=['comments']).reset_index(drop=True)
fig,ax=plt.subplots(figsize=(15,6))
plt.barh(df_comm['title'].tail(), df_comm['comments'].tail())
for i, v in enumerate(df_comm['comments'].tail()):
    ax.text(v/v,i, str(v),color='white',fontweight='bold')
plt.title('Ted Talks with Most First Level Comments')
plt.xlabel('Number of Comments')
plt.ylabel('Title')
plt.show()

print(df_comm.tail())


# "Militant atheism" shows highest first level comments followed by "Do schools kill creativity?". <br>
# These 2 ted talks videos were published since year 2006 and year 2007 respectively, in which they have been published for longer years compared to the other ted talks with higher first level comments. <br>
# It is worth noting that "How do you explain consciousness?" have received 2673 first level comments since publishing on year 2014 until year 2017.

# ### Quantile on Ted Talks views

# In[ ]:


#Views by quantile 
com_quantile = pd.qcut(df['views'], q=4).value_counts().sort_index()
plt.figure(figsize=(10,6))
com_quantile.plot(kind='bar')
plt.xticks(rotation=0)
plt.xlabel('Number of views')
plt.title('Quantile on Views')
plt.show()


# 25% of Ted Talks videos have their view counts at least 1.7mil, while 50% of Ted Talks videos have their views at least 1.1mil

# ### What are the Top 10 Ted Talks events with most published videos?

# In[ ]:


# Top 10 ted talk events with most published videos
top10_event = df['event'].value_counts().sort_values(ascending=False).head(10)
fig, ax=plt.subplots(figsize=(15,6))
top10_event.plot(kind='bar')
plt.xlabel('Events')
for i, v in enumerate(top10_event):
    ax.text(i-0.05,v+1, str(v),color='black',fontweight='bold')
plt.xticks(rotation=0)
plt.title('Top 10 Ted Talks Events with Most Published Videos')
plt.show()


# More Ted Talks videos were published on TED events rather than TEDGlobal events.<br>
# Although year 2012 has the higher published Ted Talks videos with 306 counts, but there is just a single event in 2012 fitted into Top 10 Ted Talks events with most published videos.

# In[ ]:


events = df['event'].value_counts().sort_values(ascending=False)
event_2012 = [(i,v) for i,v in events.iteritems() if('2012' in i)]

event_2012_tag = [tag[0] for tag in event_2012]
event_2012_val = [val[1] for val in event_2012]

fig, ax = plt.subplots(figsize=(15,6))
plt.barh(event_2012_tag,event_2012_val)
plt.xlabel('Events')
for i, v in enumerate(event_2012_val):
    ax.text(v,i, str(v),color='black',fontweight='bold')
plt.title('Ted Talks Events on 2012')
plt.show()

print('Total number of videos published on Ted Talks events with 2012 labelling: ',sum(val for val in event_2012_val))


# As we go deeper into Ted Talks events with 2012 label, there are only 178 published videos instead of 306 published videos.<br>
# 'TED2012' and 'TEDGlobal 2012' have most of the videos published (135 videos) among the 178 published videos. 

# ### What is the top 10 common tags used in Ted Talks?

# In[ ]:


# What is the top 10 common tags in ted talks
flat_list=[]
for index, row in df.iterrows():
    tag = ast.literal_eval(row['tags'])
    for item in tag:
        flat_list.append(item)

tag_count = Counter(flat_list)
print('Total types of tags:',len(tag_count))


# In[ ]:


tag_cat = [tag[0] for tag in tag_count.most_common(10)]
tag_val = [tag[1] for tag in tag_count.most_common(10)]

fig, ax = plt.subplots()
plt.pie(tag_val, labels=tag_val, autopct='%1.1f%%', shadow=True, startangle=90)
ax.axis('equal') 
plt.title('Top 10 Most Common Tags in Ted Talks')
plt.legend(tag_cat,bbox_to_anchor=(1.5,1), fontsize=10, bbox_transform=plt.gcf().transFigure)
plt.subplots_adjust(bottom=0, top=1.3)
plt.show()


# Among 416 tags, the top 10 most common tags used in Ted Talks videos are technology, science, global issues, culture, TEDx, design, business, entertainment, health, and innovation. Well, it fits with the TED(technology, entertainment, and design) main theme.

# In[ ]:


#Create word cloud of tags

# read the mask image
# taken from https://cdn.freebiesupply.com/images/large/2x/ted-logo-white.png
d = '../input/word-cloud-mask/'
ted_talk_mask = np.array(Image.open(d + "ted-logo-white.png"))

wc = WordCloud(mask=ted_talk_mask, background_color="white",width=800, height=400, contour_width=1).generate_from_frequencies(tag_count)

# show
plt.figure(figsize=(15,8))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud of Tags')
plt.show()


# ### Which Ted Talks tag categories viewed most by audiences?

# In[ ]:


#Which tag viewed most by audiences
tag_cat_view = []
for tag in tag_count:
    view_counts = 0
    for i in range(len(df)):
        #Match the token
        if(re.search("'"+tag+"'",df['tags'][i])):
            view_counts = view_counts + df['views'][i]
    #Append into list for visualization
    tag_cat_view.append((tag,view_counts))
    
# Sort it in descending order
tag_cat_view.sort(key=lambda x:x[1], reverse=True)


# In[ ]:


tag_cat_view_cat = [x[0] for x in tag_cat_view[:10]]
tag_cat_view_view = [x[1] for x in tag_cat_view[:10]] 

fig,ax=plt.subplots(figsize=(15,6))
plt.barh(tag_cat_view_cat, tag_cat_view_view)
for i, v in enumerate(tag_cat_view_view):
    ax.text(v/v,i, str(v),color='white',fontweight='bold')
plt.xticks(rotation=0)
plt.ylabel('Categories')
plt.xlabel('Number of Views')
plt.title('Top 10 Categories Viewed By Audiences')
plt.show()


# Culture, technology, and science tags videos received most views by audiences.<br>
# To improve the views of their Ted Talks videos, speakers should present more ideas about culture, technology and science.

# ### Which speakers gave the most Ted Talks?

# In[ ]:


# Who gave the most ted talks
df_most_active_speaker = df.groupby(['main_speaker','speaker_occupation']).agg(
    counts=('speaker_occupation', 'count'), average_views=('views','mean')).reset_index(
).sort_values(by='counts',ascending=False)
print(df_most_active_speaker[df_most_active_speaker['counts'] >= 5])


# In[ ]:


top_5_active_speaker = df_most_active_speaker[df_most_active_speaker['counts'] >= 5]
fig,ax=plt.subplots(figsize=(15,6))
plt.bar(top_5_active_speaker['main_speaker'], top_5_active_speaker['counts'])
for i, v in enumerate(top_5_active_speaker['counts']):
    ax.text(i-0.05,v/v-0.8, str(v),color='white',fontweight='bold')
plt.xlabel('Speakers')
plt.xticks(rotation=45)
plt.title('Most Active Speakers in Ted Talks')
plt.show()


# Hans Rosling given the most Ted Talks (9 times), followed by Juan Enriquez (7 times), and Marco Tempest (6 times) and Rives (6 times)

# In[ ]:


print(df[df['main_speaker'] == 'Hans Rosling'][['main_speaker','title','tags']])


# Most of the Ted Talks given by Hans Rosling revolving around global issues, economics, and health.

# ### What are the main speakers occupations in Ted Talks?

# Speakers occupations in ted_talks dataset are very dirty, where different people have their own filling style. Some of them fill their multiple occupations through slash(/), 'AND', plus(+), comma(,), and even dash(-). Some terms even have different namings such as co-founder, cofounder, or cofounders. Furthermore, there are some occupations cannot be separatede even though they contain 'AND', slash(/), such '9/11 MOTHERS', 'SURVEILLANCE AND CYBERSECURITY COUNSEL', etc. <br><br>
# 
# Through scrutinizing on the speakers occupations, these are the occupations that should not be processed:<br> 
# ('SURVEILLANCE AND CYBERSECURITY COUNSEL', 'NEUROSCIENCE AND CANCER RESEARCHER', 'FOOD AND AGRICULTURE EXPERT', 'SCULPTOR OF LIGHT AND SPACE', 'PLANETARY AND ATMOSPHERIC SCIENTIST', 'HEALTH AND TECHNOLOGY ACTIVIST', 'ENVIRONMENTAL AND LITERACY ACTIVIST', 'PROFESSOR OF MOLECULAR AND CELL BIOLOGY','HIV/AIDS FIGHTER', '9/11 MOTHERS')<br><br>
# 
# Some occupations that should be processed in more details including:<br>
# ('FOUNDER','COO','DIRECTOR', and some symbols such as ellipsis(...), semicolon(;), etc)
# 
# *<i>P/S: The code below may look inefficient, if there are any suitable method for cleaning the data, hopefully can provide some suggestions on it.</i>

# In[ ]:


ignore_process_occupation = ['SURVEILLANCE AND CYBERSECURITY COUNSEL', 'NEUROSCIENCE AND CANCER RESEARCHER', 
                             'FOOD AND AGRICULTURE EXPERT', 'SCULPTOR OF LIGHT AND SPACE', 'PLANETARY AND ATMOSPHERIC SCIENTIST', 
                             'HEALTH AND TECHNOLOGY ACTIVIST', 'ENVIRONMENTAL AND LITERACY ACTIVIST', 
                             'PROFESSOR OF MOLECULAR AND CELL BIOLOGY','HIV/AIDS FIGHTER', '9/11 MOTHERS']
occupations = []
for index, row in df.iterrows():
    speaker_occupation = str(row['speaker_occupation']).upper().strip()
    if(re.search(r'FOUNDER', speaker_occupation)):
        
        speaker_occupation = re.sub(r'COFOUNDER|CO-FOUNDERS','CO-FOUNDER', speaker_occupation)
        if('CO-FOUNDER' in speaker_occupation):
            occupations.append('CO-FOUNDER')
        if('BLOGGER' in speaker_occupation): #BLOGGER; CO-FOUNDER, SIX APART
            occupations.append('BLOGGER')
        if('EXECUTIVE DIRECTOR' in speaker_occupation):
            occupations.append('EXECUTIVE DIRECTOR')
        if('CEO' in speaker_occupation):
            occupations.append('CEO')
        if('DESIGNER' in speaker_occupation):
            occupations.append('DESIGNER')
        if('FOUNDER' in speaker_occupation):
            occupations.append('FOUNDER')
    elif(re.search(r'COO', speaker_occupation)):
        occupations.append(speaker_occupation.split(',')[0])
    elif(re.search(r'DIRECTOR', speaker_occupation)):
        if(' AND ' in speaker_occupation):
            occupations.extend(speaker_occupation.split(' AND '))
        elif('DIRECTOR OF' in speaker_occupation):
            occupations.append(speaker_occupation.split(',')[0])
        elif(',' in speaker_occupation):
            speaker_occupation = re.sub(r'/',', ', speaker_occupation)
            speaker_occupation = re.sub(r';',',', speaker_occupation)
            speaker_occupation = speaker_occupation.replace(', IDEO','')
            speaker_occupation = speaker_occupation.replace(', THE INSTITUTE FOR GLOBAL HAPPINESS','')
            speaker_occupation = speaker_occupation.replace(', NSA','')
            occupations.extend(speaker_occupation.split(','))
    elif(re.search(r' AND |[+;.,/]', speaker_occupation)):
        if(speaker_occupation in ['EXECUTIVE CHAIR, FORD MOTOR CO.']): #SINGER-SONGWRITER
            occupations.append(speaker_occupation.split(',')[0])
        elif(speaker_occupation in ignore_process_occupation):
            occupations.append(speaker_occupation)
        else:
            speaker_occupation = re.sub(r' AND |[/]',', ', speaker_occupation)
            speaker_occupation = speaker_occupation.replace(' + ',', ')
            speaker_occupation = re.sub(r';',',', speaker_occupation)
            speaker_occupation = speaker_occupation.replace(' ...','')
            if('SINGER-SONGWRITER' == speaker_occupation):
                speaker_occupation = speaker_occupation.replace('-',', ')

            occupations.extend(speaker_occupation.split(', '))
        


# Display the counts of each speaker occupation

# In[ ]:


occupations_counts = Counter(occupations)
print(occupations_counts)


# In[ ]:


#Counts of Top 10 speakers occupations in ted talks
occupations_counts_cat = [occ[0] for occ in occupations_counts.most_common(10)]
occupations_counts_val = [occ[1] for occ in occupations_counts.most_common(10)]

fig, ax = plt.subplots(figsize=(15,6))
plt.bar(occupations_counts_cat,occupations_counts_val)
for i, v in enumerate(occupations_counts_val):
    ax.text(i-0.1,v/v, str(v),color='white',fontweight='bold')
plt.title('Top 10 Speakers Occupations in Ted Talks')
plt.xlabel('Speakers Occupations')
plt.ylabel('Counts')
plt.show()


# From the data that have been processed, most of the speakers occupations in Ted Talks are author, followed by activist and writer.

# ### Which speakers have the higher average views of their Ted Talks?

# In[ ]:


#Which speakers have higher average views per talk
top_ten_most_average_views_speaker = df_most_active_speaker.sort_values(by='average_views').tail(10)
fig, ax = plt.subplots(figsize=(15,6))
plt.barh(top_ten_most_average_views_speaker['main_speaker'],top_ten_most_average_views_speaker['average_views'])
for i, v in enumerate(top_ten_most_average_views_speaker['average_views']):
    ax.text(v/v, i, str(v), color='white', fontweight='bold')
plt.xlabel('Views')
plt.ylabel('Speakers')
plt.title('Top 10 Average Views by Speakers')
plt.show()

print(top_ten_most_average_views_speaker.tail(10))


# Amy Cuddy has the highest average views (43mil) of her Ted Talks, and with just a single talk. This shows that her talk probably bring some positive impacts and very inspiring to audiences.<br>
# **If you are interested on how good is her talk, you may click on the link to watch it: [Amy Cuddy: 'Your body language may shape who you are'](https://www.ted.com/talks/amy_cuddy_your_body_language_may_shape_who_you_are?language=en)

# ### Which Ted Talks videos have most views?

# In[ ]:


#Top 10 up to date most views videos
df_most_views = df[['main_speaker', 'title', 'views','published_date']].sort_values(
    by='views').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8,6))
plt.barh(df_most_views['title'].tail(10),df_most_views['views'].tail(10))
for i, v in enumerate(df_most_views['views'].tail(10)):
    ax.text(v/v, i, str(v), color='white', fontweight='bold')
plt.title('Top 10 Most Views Ted Talks')
plt.xlabel('Views')
plt.ylabel('Titles')
plt.show()

df_most_views.tail(10)


# Ken Robinson with 'Do schools kill creativity?' has the most views in Ted Talks, followed by Amy Cuddy with 'Your body language may shape who you are'.<br>
# 'Do schools kill creativity?' has been published for a longer time compared to 'Your body language may shape who you are'.<br>
# **If you are interested on 'Do schools kill creativity?', you may click on the link to watch it: [Sir Ken Robinson: Do schools kill creativity'](https://www.ted.com/talks/sir_ken_robinson_do_schools_kill_creativity)

# ### What are the ratings for each Ted Talks?

# Make a copy of main dataframe

# In[ ]:


df_ext = df.copy()


# Get the ratings for each ted talk

# In[ ]:


for index, row in df.iterrows():
    rates = ast.literal_eval(row['ratings'])
    for item in rates:
        if(index == 0): #to create new column
            df_ext[item['name']] = item['count']
        else:
            df_ext[item['name']][index] = item['count']

df_ext.head(5)        


# In[ ]:


df_ext.columns


# drop 'ratings' column from dataframe, as it has been splitted into multiple columns.

# In[ ]:


df_ext.drop('ratings',axis=1, inplace=True)


# ### Which Ted Talks videos received most ratings by audiences?

# In[ ]:


#Which ted talks received higher number of ratings
rating_col = ['Funny', 'Beautiful', 'Ingenious', 'Courageous', 'Longwinded',
       'Confusing', 'Informative', 'Fascinating', 'Unconvincing', 'Persuasive',
       'Jaw-dropping', 'OK', 'Obnoxious', 'Inspiring']

df_ext['sum_ratings'] = df_ext[rating_col].sum(axis=1)

df_ext_sum_rate_sort = df_ext[['name','sum_ratings']].sort_values(by='sum_ratings').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10,6))
plt.barh(df_ext_sum_rate_sort['name'][-10:],df_ext_sum_rate_sort['sum_ratings'][-10:])
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
for i, v in enumerate(df_ext_sum_rate_sort['sum_ratings'][-10:]):
    ax.text(v/df_ext_sum_rate_sort['sum_ratings'][i], i, str(v), color='white', fontweight='bold')
plt.title('Ted Talks with Higher Number of Ratings')
plt.xlabel('Total Ratings')
plt.ylabel('Titles')
plt.show()


# 'Do schools kill creativity?' with the most views and one of the earliest Ted Talks published received highest number of ratings, which is total of 93850 ratings, followed by 'My stroke of insight' (70665 ratings) and 'Your body language may shape who you are' (65968 ratings). <br>
# Although 'My stroke of insight' only has 21mil views compared to 'Do schools kill creativity?'(47mil views) and 'Your body language may shape who you are' (43mil views), but it has the second most highest number of ratings.<br>
# **If you are interested on 'My stroke of insight', you may click on the link to watch it: [Jill Bolte Taylor: My stroke of insight'](https://www.ted.com/talks/jill_bolte_taylor_my_stroke_of_insight)

# ### What are the main ratings of Ted Talks videos?

# In[ ]:


df_ext['main_rating'] = df_ext[rating_col].idxmax(axis=1)

most_rel_rat = df_ext['main_rating'].value_counts()
fig, ax = plt.subplots(figsize=(15,6))
plt.bar(most_rel_rat.index,most_rel_rat.values)
for i, v in enumerate(most_rel_rat):
    ax.text(i-0.2, v+10, str(v), color='black', fontweight='bold')
plt.title('Ted Talks Main Ratings')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.xlabel('Main Rating')
plt.show()


# Most of the Ted Talks videos have main rating that show positive, such as inspiring, informative, fasicnating, funny, etc.<br>
# However, there are some Ted Talks videos with main rating that show negative, such as unconvincing, longwinded, obnoxious and confusing. <br>
# Even though there are nagative main rating of Ted Talks videos, but these videos are just made up a small part. <br>
# Hence, Ted Talks is still a good platform for audiences to gain inspiring and informative knowledge.

# ### What are the Ted Talks videos that show negative main rating?

# In[ ]:


# Which ted talks considered as poor rating
for rate in ['Confusing','Obnoxious','Longwinded','Unconvincing']:
    print(df_ext[['main_rating','name',rate,'sum_ratings']][df_ext['main_rating'].isin([rate])].reset_index
          (drop=True))


# ### Correlation Analysis

# In[ ]:


fig, ax = plt.subplots(figsize=(20,20))
ax = sns.heatmap(df_ext.corr(), annot = True)


# Based on the correlation of the features: <br>
# <li>Comments shows irrelevant to number of speakers, and less imapct on video duration.<br>
# <li>Ted Talks that are longwinded shows positive correlation to video duration, which implies that longwinded videos normally have longer video duration.
# <li>Ted Talks that have positive ratings normally show higher views compared to Ted Talks with negative ratings. 

# #### Hope that this notebook could help generates more insights on data analysis on Ted Talks. 

# ### Small notes
# 
# ====================================================================<br>
# For simple WordCloud
# pip install wordcloud or conda install -c conda-forge wordcloud
# 
# To import and install WordCloud with mask effect
# 
# git clone https://github.com/amueller/word_cloud.git
# cd word_cloud
# pip install .
# 
# ====================================================================<br>
# For row filtering
# https://cmdlinetips.com/2018/02/how-to-subset-pandas-dataframe-based-on-values-of-a-column/
# 
# ====================================================================<br>
# For annotation on bar plot
# https://stackoverflow.com/questions/52182746/matplotlib-horizontal-bar-plot-add-values-to-bars
# 
# ====================================================================<br>
# Avoid overlapping of legend on pie chart
# https://stackoverflow.com/questions/23577505/how-to-avoid-overlapping-of-labels-autopct-in-a-matplotlib-pie-chart
# 

# In[ ]:




