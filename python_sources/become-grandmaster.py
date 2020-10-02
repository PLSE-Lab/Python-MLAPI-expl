#!/usr/bin/env python
# coding: utf-8

# <font size="+2" color="brown"><i><b><center>"A King isn't born.He is Made"</center></b></i></font>

# <font size="+3" color=purple ><b> <center><u>Secret of Success - Become Grandmaster</u></center></b></font>

# ![](https://media.istockphoto.com/photos/gold-crown-picture-id160231072?k=6&m=160231072&s=612x612&w=0&h=epxgLSCvRxBvn-IZv39fz2SmzqUg7OYuoOS6BLCgPss=)

# <font size="+3" color="blue"><b>1. Objective</b></font><br><a id="top"></a>

# The aim of this kernel is to understand the lifestyle/activity of Grandmasters in Kaggle over Competition,Notebooks & Discussions.My analysis would be completely about the way of life of all grandmasters in Kaggle platform.Their approaches and activities will definetly give us a success path to become like them.I am still working/exploring on it.
# 
# Usually we may have some questions in our mind when it comes to their achievement.Few of them are:
# 
# * **How long do they take to become GM?**
# * **How consistent are they in each areas?**
# * **Where do they concentrate?**
# * **What strategies do they follow?**
# * **What is difference between a GM and an other Tier kaggler?**
# 
# *(GM - Grandmaster)*
# 
# I have tried to bring answers to such question and many more in my below analysis with some interesting EDAs.I have listed few keys as part of conclusion at end of kernel.So please go through all sections with patience.And yes if you like it,please appreciate me with an <font size=+1 color="red"><b>Upvote</b></font>.

# **Note:**
# * This kernel will fetch you basic lifestyles,principles and strategies of GMs.
# * The whole analysis is about how all GMs work in Kaggle.(Strictly It is not about which GM is better than the other)
# * I have skipped Kaggle Team GMs to ensure that it is exclusive to users.
# * Kaggle had improvised ranking in 2015-2016,so we might have seen a lot of new GMs at that time.(Honestly I do not know what was the process before).I have evaluated based on the current data which kaggle has provided in [meta-kaggle dataset](https://www.kaggle.com/kaggle/meta-kaggle).

# <font size=+2 color="purple"><i>Wanna become GrandMaster... Let's get started!!!</i></font>

# This notebook holds 4 main sections:
# 
# * [Kaggle Grandmasters Overview](#3)
# * [Competition Grandmaster](#4)
# * [Notebook Grandmaster](#5)
# * [Discussion Grandmaster](#6)

# <font size="+3" color="blue"><b>2. Libraries & Data</b></font><br><a id="2"></a>

# I have utilized data from meta-kaggle data and some of my own predefined data to save some computational time.

# In[ ]:


# Basic libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import itertools
import statistics
import re
import json
from numpy import arange,array,ones
from scipy import stats
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

# Datetime and word
from datetime import datetime
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

# Maps
from geopandas.tools import geocode
import folium
from folium import Marker
from folium.plugins import MarkerCluster

# Scrab web Data
import requests
from bs4 import BeautifulSoup
import time

# Plots
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from plotly.colors import n_colors
from IPython.display import display, HTML
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools
from plotly.subplots import make_subplots

import gc
import warnings
warnings.filterwarnings("ignore")

KAGGLE_BASE_URL = "https://www.kaggle.com/"
TWITTER_BASE_URL = "https://twitter.com/"
GITHUB_BASE_URL = "https://github.com/"

pd.options.display.max_colwidth = 250


# In[ ]:


#Users
Users=pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')
UserAchievements=pd.read_csv('/kaggle/input/meta-kaggle/UserAchievements.csv')
Tags=pd.read_csv('/kaggle/input/meta-kaggle/Tags.csv')

#Competitions
Competitions=pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')
Submissions=pd.read_csv('/kaggle/input/meta-kaggle/Submissions.csv')
TeamMemberships=pd.read_csv('/kaggle/input/meta-kaggle/TeamMemberships.csv')
Teams=pd.read_csv('/kaggle/input/meta-kaggle/Teams.csv')

#Kernels
KernelVotes=pd.read_csv('/kaggle/input/meta-kaggle/KernelVotes.csv')
KernelVersionOutputFiles=pd.read_csv('/kaggle/input/meta-kaggle/KernelVersionOutputFiles.csv')
KernelVersionKernelSources=pd.read_csv('/kaggle/input/meta-kaggle/KernelVersionKernelSources.csv')
KernelVersionCompetitionSources=pd.read_csv('/kaggle/input/meta-kaggle/KernelVersionCompetitionSources.csv')
KernelTags=pd.read_csv('/kaggle/input/meta-kaggle/KernelTags.csv')
Kernels=pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv')
KernelVersions=pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv')
KernelLanguages=pd.read_csv('/kaggle/input/meta-kaggle/KernelLanguages.csv')

#Discussion
ForumMessageVotes=pd.read_csv('/kaggle/input/meta-kaggle/ForumMessageVotes.csv')
ForumTopics=pd.read_csv('/kaggle/input/meta-kaggle/ForumTopics.csv')
Forums=pd.read_csv('/kaggle/input/meta-kaggle/Forums.csv')
ForumMessages=pd.read_csv('/kaggle/input/meta-kaggle/ForumMessages.csv')

#Predefined data
comp_merged=pd.read_csv('/kaggle/input/competition-gm/competition_gm.csv')
non_comp_merged=pd.read_csv('/kaggle/input/competition-gm/non_competition_gm.csv') 
combined_note=pd.read_csv('/kaggle/input/competition-gm/combined_note.csv') 
note_user_merged=pd.read_csv('/kaggle/input/competition-gm/note_user_merged.csv') 
all_gm_loc=pd.read_csv('/kaggle/input/competition-gm/all_gm_loc.csv') 
dis_user_merged=pd.read_csv('/kaggle/input/competition-gm/dis_user_merged.csv') 
country_code=pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')


# <font size="+3" color="blue"><b>3. Overview</b></font><br><a id="3"></a>

# <a id="3.1"></a>
# <font size="+2" color="indigo"><b>3.1 Kagglers by Tier</b></font><br>

# Number of current Kagglers from all tiers

# In[ ]:


tier_df_count=Users.loc[(Users['PerformanceTier']!=5)]['PerformanceTier'].value_counts().to_frame().reset_index().rename(columns={'index':'PerformanceTier','PerformanceTier':'count'})
fig = go.Figure()
fig.add_trace(go.Indicator(
    mode = "number",
    value = int(tier_df_count.loc[tier_df_count['PerformanceTier']==4]['count']),
    title = {'text': "GrandMaster",'font': {'color': 'gold','size':26}},
    number={'font':{'color': 'gold','size':71}},
    domain = {'row': 0, 'column': 0}
))
fig.add_trace(go.Indicator(
    mode = "number",
    value = int(tier_df_count.loc[tier_df_count['PerformanceTier']==3]['count']),
    title = {'text': "Master",'font': {'color': 'orange','size':23}},
    number={'font':{'color': 'orange','size':48}},
    domain = {'row': 0, 'column': 1}
))

fig.add_trace(go.Indicator(
    mode = "number",
    value = int(tier_df_count.loc[tier_df_count['PerformanceTier']==2]['count']),
    title = {'text': "Expert",'font': {'color': 'darkviolet','size':20}},
    number={'font':{'color': 'darkviolet','size':37}},
    domain = {'row': 0, 'column': 2}
))

fig.add_trace(go.Indicator(
    mode = "number",
    value = int(tier_df_count.loc[tier_df_count['PerformanceTier']==1]['count']),
    title = {'text': "Contributer",'font': {'color': 'deepskyblue','size':18}},
    number={'font':{'color': 'deepskyblue','size':30}},
    domain = {'row': 0, 'column': 3}
))

fig.add_trace(go.Indicator(
    mode = "number",
    value = int(tier_df_count.loc[tier_df_count['PerformanceTier']==0]['count']),
    title = {'text': "Novice",'font': {'color': 'green','size':15}},
    number={'font':{'color': 'green','size':15}},
    domain = {'row': 0, 'column': 4}
))
fig.update_layout(
    grid = {'rows': 1, 'columns': 5, 'pattern': "independent"})
fig.show()


# <a id="3.2"></a>
# <font size="+2" color="indigo"><b>3.2 All 3x & 2x GrandMasters</b></font><br>

# Let us see all our present 3x & 2x Grandmasters.I have utilized below codes from an awesome kernel of [sahid](https://www.kaggle.com/sahidvelji/meet-the-kaggle-team).

# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# In[ ]:


def display_html(df, cols=None, index=False, na_rep='', num_rows=0):
    if num_rows == 0:
        df_table = df.to_html(columns=cols, index=index, na_rep=na_rep, escape=False, render_links=True)
        display(HTML(df_table))
    else:
        df_table = df.head(num_rows).to_html(columns=cols, index=index, na_rep=na_rep, escape=False, render_links=True)
        display(HTML(df_table))


# In[ ]:


three_x=UserAchievements.loc[UserAchievements['Tier']==4]['UserId'].value_counts().to_frame().reset_index().rename(columns={'index':'Id','UserId':'count'})
all_three_two_x=three_x.loc[three_x['count']>1]['Id']
print("We have {} - 2x & 3x Grandmasters".format(len(all_three_two_x)))
gms_2x_3x= Users[Users['Id'].isin(list(all_three_two_x))].copy()
gms_2x_3x.reset_index(drop=True, inplace=True)
gms_2x_3x.drop(columns=['PerformanceTier'], inplace=True)


# In[ ]:


def scrap_data(gms):
    for index in gms.index:
        time.sleep(1)
        row = gms.iloc[index]
    
        username = row.UserName
        profile_url = '{}{}'.format(KAGGLE_BASE_URL, username)
        displayname = row.DisplayName

        result = requests.get(profile_url)
        src = result.content
        soup = BeautifulSoup(src, 'html.parser')
        soup = soup.find_all("div", id="site-body")[0].find("script")

        user_info = re.search('Kaggle.State.push\(({.*})', str(soup)).group(1)
        user_dict = json.loads(user_info)
    
        city = user_dict['city']
        region = user_dict['region']
        country = user_dict['country']
        avatar_url = user_dict['userAvatarUrl']
        occupation = user_dict['occupation']
        organization = user_dict['organization']
        github_user = user_dict['gitHubUserName']
        twitter_user = user_dict['twitterUserName']
        linkedin_url = user_dict['linkedInUrl']
        website_url = user_dict['websiteUrl']
        last_active = user_dict['userLastActive']
        num_followers = user_dict['followers']['count']
        num_following = user_dict['following']['count']

        num_posts = user_dict['discussionsSummary']['totalResults']
        num_datasets = user_dict['datasetsSummary']['totalResults']
        num_kernels = user_dict['scriptsSummary']['totalResults']
        num_comps = user_dict['competitionsSummary']['totalResults']


        gms.loc[index, 'Image'] = '<a href="{}" target="_blank" title="{}"><img src="{}" width="100" height="100"></a>'.format(
            profile_url, displayname, avatar_url)
        gms.loc[index, 'NumFollowers'] = num_followers
        gms.loc[index, 'NumFollowing'] = num_following
        gms.loc[index, 'NumPosts'] = num_posts
        gms.loc[index, 'NumDatasets'] = num_datasets
        gms.loc[index, 'NumKernels'] = num_kernels
        gms.loc[index, 'NumCompetitions'] = num_comps
        gms.loc[index, 'Country'] = country
    return gms


# In[ ]:


gms_2x_3x = gms_2x_3x.convert_dtypes()
display_html(scrap_data(gms_2x_3x), cols=['Image','UserName', 'DisplayName','RegisterDate','NumFollowers','Country'])


# In[ ]:


del gms_2x_3x
gc.collect()


# <a id="3.3"></a>
# <font size="+2" color="indigo"><b>3.3 Grandmaster by Areas</b></font><br>

# Let us visualize number of Grandmasters by all areas 

# In[ ]:


# Competitions
competition_gm=UserAchievements.loc[(UserAchievements['Tier']==4)&(UserAchievements['AchievementType']=='Competitions')]
non_competition_gm=UserAchievements.loc[(UserAchievements['Tier']!=4)&
                                        ((UserAchievements['TotalGold']!=0)|(UserAchievements['TotalSilver']!=0)|(UserAchievements['TotalBronze']!=0))&
                                        (UserAchievements['AchievementType']=='Competitions')]

# noteboooks
notebook_gm=UserAchievements.loc[(UserAchievements['Tier']==4)&(UserAchievements['AchievementType']=='Scripts')]

non_notebook_gm=UserAchievements.loc[(UserAchievements['Tier']!=4)&(Users['PerformanceTier']!=5)&
                                        ((UserAchievements['TotalGold']!=0)|(UserAchievements['TotalSilver']!=0)|(UserAchievements['TotalBronze']!=0))&
                                        (UserAchievements['AchievementType']=='Scripts')]

# Discussion - Removing Kaggle Team GM
a=set(UserAchievements.loc[(UserAchievements['Tier']==4)&(UserAchievements['AchievementType']=='Discussion')]['UserId'])
b=set(Users[(Users['Id'].isin(list(a)))&(Users['PerformanceTier']!=5)]['Id'])
all_discussion_gm=a.intersection(b)
discussion_gm=UserAchievements[(UserAchievements['Tier']==4)&(UserAchievements['AchievementType']=='Discussion')&(UserAchievements['UserId'].isin(list(all_discussion_gm)))]
non_discussion_gm=UserAchievements.loc[(UserAchievements['Tier']!=4)&(Users['PerformanceTier']!=5)&
                                        ((UserAchievements['TotalGold']!=0)|(UserAchievements['TotalSilver']!=0)|(UserAchievements['TotalBronze']!=0))&
                                        (UserAchievements['AchievementType']=='Discussion')]


# In[ ]:


fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  competition_gm.shape[0],
    domain = {'x': [0.25, 1], 'y': [0.08, 0.25]},
    title = {'text': "Competitions GMs",'font':{'color': 'teal','size':17}},
     number={'font':{'color': 'teal'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 200]},
        'bar': {'color': "gold"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = notebook_gm.shape[0],
    domain = {'x': [0.25, 1], 'y': [0.4, 0.6]},
    title = {'text': "Notebooks GMs",'font':{'color': 'darkmagenta','size':17}},
    number={'font':{'color': 'darkmagenta'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,50]},
        'bar': {'color': "gold"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = discussion_gm.shape[0],
    domain = {'x': [0.25, 1], 'y': [0.7, 0.9]},
    title = {'text' :"Discussions GMs",'font':{'color': 'sienna','size':17}},
     number={'font':{'color': 'sienna'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,50]},
        'bar': {'color': "gold"}}
))
fig.update_layout(height = 400 , margin = {'t':0, 'b':0, 'l':0},title="All Areas - Grandmasters")
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * We have **more competition GMs(186)** which is expected.It seems like kagglers concentrate more towards competition.
# * We have only **20** and **28** GMs for discussion/notebook.

# <a id="3.4"></a>
# <font size="+2" color="indigo"><b>3.4 How long did they take?</b></font><br>

# This question is very common among all kagglers.How long will it take one to become GM.So I have made a gantt chart to see timeline taken by each GM.
# 
# In this sections,we will see the answer for above question at high level.Then going gorward in each area section, we will dig deeper.
# 
# **Note:**<br>
# *There is an important thing to understand here that all Grandmasters listed below have their own lifestyle and way of participation.So we should never conclude/assume that if a GM takes long time ,so he/she is performing weak.I feel quality of their work matters not quantity of their time taken*
# 

# In[ ]:


def merge_user_achieve(df):
    filter_df=Users[Users['Id'].isin(list(df['UserId']))]
    filter_df=filter_df.rename(columns={'Id':'UserId'})
    select_df=filter_df[['UserId','RegisterDate','DisplayName']]
    merged_df=pd.merge(df,select_df,on='UserId',how="left")
    merged_df['diff_days']=(pd.to_datetime(merged_df['TierAchievementDate'])-pd.to_datetime(merged_df['RegisterDate'])).dt.days 
    merged_df['total_medals']=merged_df['TotalGold']+merged_df['TotalSilver']+merged_df['TotalBronze']
    return merged_df


# In[ ]:


comp_user_merged=merge_user_achieve(competition_gm)
non_comp_user_merged=merge_user_achieve(non_competition_gm)
#note_user_merged=merge_user_achieve(notebook_gm)
non_note_user_merged=merge_user_achieve(non_notebook_gm)
#dis_user_merged=merge_user_achieve(discussion_gm)
non_dis_user_merged=merge_user_achieve(non_discussion_gm)
def plot_duration(data,color,height,title):
    scaler = MinMaxScaler()
    data[['diff_days']] = scaler.fit_transform(data[['diff_days']])*100
    df=[]
    for i in data.index:
        df.append(dict(Task=data['DisplayName'][i],Start=data['RegisterDate'][i],Finish=data['TierAchievementDate'][i],
                   Complete=data['diff_days'][i]))
    df=pd.DataFrame(df)
    df['Start']=pd.to_datetime(df['Start'], format='%m/%d/%Y')
    df['Finish']=pd.to_datetime(df['Finish'], format='%m/%d/%Y')
    df=df.sort_values('Complete',ascending=False)
    fig = ff.create_gantt(df,colors=color, index_col='Complete',showgrid_x=True, show_colorbar=True,showgrid_y=True,height=height,title=title)
    fig.update_layout(title_x=0.5)
    fig.show()
    


# In[ ]:


plot_duration(comp_user_merged,"Blackbody",3500,"Competition GM Duration")
plot_duration(note_user_merged,"Blackbody",650,"Notebook GM Duration")
plot_duration(dis_user_merged,"Blackbody",600,"Discussion GM Duration")


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * Duration of days varied in all three areas.Most commonly it took **1-3 years** for one to achieve competition GM.Here another fact is that competition arent so frequent.They come in clusters and go in clusters.
# * We also noticed that **1-2 years** for notebook and discussion GM.

# <a id="4">
# <font size="+3" color="blue"><b>4. Competition GM</b></font>

# <a id="4.1"></a>
# <font size="+2" color="indigo"><b>4.1 Competitions & GMs over Years</b></font><br>

# In[ ]:


awardable_competitions=Competitions.loc[(Competitions['CanQualifyTiers']==True)]
awardable_competitions['year']=pd.to_datetime(awardable_competitions['EnabledDate']).dt.year
comp_year=awardable_competitions['year'].value_counts().reset_index().rename(columns={'index':"year","year":'count'})
print("Number of Medal Awardable competitions in Kaggle : {}".format(Competitions.loc[(Competitions['CanQualifyTiers']==True)].shape[0]))

comp_merged["days"]=(pd.to_datetime(comp_merged['TierAchievementDate'])-pd.to_datetime(comp_merged['RegisterDate'])).dt.days
comp_merged['tier_year']=pd.to_datetime(comp_merged['TierAchievementDate']).dt.year
tier_year=comp_merged['tier_year'].value_counts().reset_index().rename(columns={"index":"tier_year","tier_year":"count"})


# In[ ]:


fig = make_subplots(rows=2, cols=1,subplot_titles=("# of Competitions over years", "# of GM over years"))
fig.add_trace(
    go.Bar(x=comp_year['year'], y=comp_year['count'],name="Competitions",marker_color='blue',text=comp_year['count'],textposition="outside"),
    row=1, col=1
)
fig.add_trace(
    go.Bar(x=tier_year['tier_year'], y=tier_year['count'],name="GM",marker_color='teal',text=tier_year['count'],textposition="outside"),
    row=2, col=1
)
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_xaxes(title_text="Year", row=1, col=1)
fig.update_xaxes(title_text="Year", row=2, col=1)
fig.update_yaxes(title_text="Number of Competitions", row=1, col=1)
fig.update_yaxes(title_text="Number of Grandmasters", row=2, col=1)
fig.update_layout(title_text="Competitions & GrandMasters",title_x=0.5,height=1000)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * After 2011,we have been facing more than **20+** competitions every year.
# * From second plot,we have seen a **linear growth** in 2017,2018 & 2019.This is great.Kaggle is **facing more problem solvers**.
# * We have **10** new grandamsters this year and it is still growing.

# <a id="4.2"></a>
# <font size="+2" color="indigo"><b>4.2 How many days will it take?</b></font><br>

# We will see how many days did GMs take to become GM.

# In[ ]:


fig = go.Figure(data=[go.Histogram(x=comp_merged["days"],nbinsx=100,marker_color='teal',xbins=dict(size=180),
    opacity=1)])
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text='Days to become GM',
    xaxis_title_text='# of Days',
    yaxis_title_text='# of GM',
    title_x=0.5)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * I have binned this plot with size of 180 to visualize half yearly growth.We could see a **slight bimodal** distribution.
# * Anyhow it takes almost **2-3 years** to be a competition GM

# <a id="4.3"></a>
# <font size="+2" color="indigo"><b>4.3 How many Competitions to participate?</b></font><br>

# We will see how much competitions did it take for one to become GM.

# In[ ]:


#def get_totalcomp(id):
 #   total= Submissions.loc[Submissions['SubmittedUserId']==id]['TeamId'].nunique()
  #  return total
#comp_merged['total_comp']=comp_merged['UserId'].apply(lambda x:get_totalcomp(x))

#def get_totalcomp_before(id,data):
 #   Submissions['SubmissionDate']=pd.to_datetime(Submissions['SubmissionDate'])
  #  achieved_date=pd.to_datetime(data[data['UserId']==id]['TierAchievementDate'].to_string().split(" ")[-1])
   # total= Submissions.loc[(Submissions['SubmittedUserId']==id)&(Submissions['SubmissionDate']<achieved_date)]['TeamId'].nunique()
    #return total

#comp_merged['total_comp_before']=comp_merged['UserId'].apply(lambda x:get_totalcomp_before(x,comp_user_merged))
#non_comp_merged['total_comp']=non_comp_merged['UserId'].apply(lambda x:get_totalcomp(x,non_comp_user_merged))


# In[ ]:


fig = go.Figure(data=go.Histogram(x=comp_merged['total_comp_before'],nbinsx=50,marker = dict(color = 'teal')))
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text='Distribution of Competition Participation',
    xaxis_title_text='#Competitions', 
    title_x=0.5)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * We can split this graph into two part of people 
#     - 1.Those who concentrate on particulars. (first peak) 
#     - 2.Who explore everything. (short bars after peak)

# <a id="4.4"></a>
# <font size="+2" color="indigo"><b>4.4 Competitons & Medals</b></font><br>

# We know that competition & medals achieved for GM are always linear.But this will test their excellence.I wanted to see how much did they lose more than they win?

# In[ ]:


#def get_totalmedal(id,data):
 #   Submissions['SubmissionDate']=pd.to_datetime(Submissions['SubmissionDate'])
  #  achieved_date=pd.to_datetime(data[data['UserId']==id]['TierAchievementDate'].to_string().split(" ")[-1])
   # total_teams= list(Submissions.loc[(Submissions['SubmittedUserId']==id)&(Submissions['SubmissionDate']<achieved_date)]['TeamId'].unique())
    #total_medals=Teams[(Teams['Id'].isin(list(total_teams)))&(Teams['Medal'].notnull())].shape[0]
    #return total_medals
#comp_merged['total_med_before']=comp_merged['UserId'].apply(lambda x:get_totalmedal(x,comp_user_merged))


# In[ ]:


import scipy.stats
corr_vote=round(scipy.stats.pearsonr(comp_merged['total_comp_before'], comp_merged['total_med_before'])[0],4)
slope, intercept, r_value, p_value, std_err = stats.linregress(comp_merged['total_comp_before'],comp_merged['total_med_before'])
line = slope*comp_merged['total_comp_before']+intercept

fig=go.Figure()
fig.add_traces(go.Scatter(
                  x=comp_merged['total_comp_before'],
                  y=comp_merged['total_med_before'],
                  mode='markers',
                  text=comp_merged['DisplayName'],
                  marker = dict(
                  color = 'teal'
                  )))

fig.add_traces(go.Scatter(
                  x=comp_merged['total_comp_before'],
                  y=line,
                  mode='lines',
                  marker=go.Marker(color='red'),
                  name='Fit'
                  ))
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text='# of Competitions Vs # of Medals',
    xaxis_title_text='Total Competitions participated', # xaxis label
    yaxis_title_text='Total Medals achieved', # yaxis label
    title_x=0.5,showlegend=False,annotations=[
        dict(
            x=150,
            y=35,
            xref="x",
            yref="y",
            text="Correlation:{}".format(corr_vote),
            showarrow=False,
            arrowhead=7,
            ax=0,
            ay=-40,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#76EE00",
            #opacity=0.8
        )
    ])

fig.show()


# In[ ]:


fig = go.Figure(data=[go.Scatter3d(x=comp_merged['total_comp_before'], y=comp_merged['total_med_before'], z=comp_merged['days'],

    mode='markers',
    marker=dict(
        size=12,
        color=comp_merged['diff_days'],             
        colorscale='portland',   # choose a colorscale
        opacity=0.8
    ),
)])

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),
    scene = dict(xaxis_title='Total Competitions Participated',
                 yaxis_title='Medals Achieved',
                zaxis_title='Days taken to become GM'))
fig.show()


# In[ ]:


fig = go.Figure(data=[go.Mesh3d(x=comp_merged['total_comp_before'], y=comp_merged['total_med_before'], z=comp_merged['days'],
                   opacity=0.5,
                   color='rgba(244,22,100,0.6)'
                  )])


fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),
    scene = dict(xaxis_title='Total Competitions Participated',
                 yaxis_title='Medals Achieved',
                zaxis_title='Days taken to become GM'))

fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * From first plot we could see the correlation is around **0.73**.We can observe here that they have missed few medals too(1-0.73=**0.27**).So takeaway here is even winners lose at times.**Never Give Up**
# * From second plot ,we could see more of points are clubbed towards low number of competitions,low number of medals and less than half of usual time spent by one to become GM.They are smart.

# <a id="4.5"></a>
# <font size="+2" color="indigo"><b>4.5 Do they believe in Teaming?</b></font><br>

# I did min-max scaler on days taken by each GM to achieve tier and splited them into three categories.

# In[ ]:


#def teaming_percentage(id):
 #   all_team=TeamMemberships[TeamMemberships['UserId']==id]['TeamId'].unique()
 #  all_unique_teams=TeamMemberships[TeamMemberships['TeamId'].isin(list(all_team))].shape[0]
 #   return ((all_unique_teams-len(all_team))/all_unique_teams)*100
#comp_merged['teaming_percent']=comp_merged['UserId'].apply(lambda x:teaming_percentage(x))
#non_comp_merged['teaming_percent']=non_comp_merged['UserId'].apply(lambda x:teaming_percentage(x))

#comp_merged['teaming_frequecy']=np.where(comp_merged['teaming_percent']<=20,"Less Dependent",
#    np.where((comp_merged['teaming_percent']>20)&(comp_merged['teaming_percent']<=40),"Moderate Dependent",
# np.where(comp_merged['teaming_percent']>40,"Highly Dependent","Nan"))


# In[ ]:


hist_data = [comp_merged['teaming_percent']]
group_labels = ['Teaming Percentage'] 
colors=["teal"]
fig=go.Figure()
fig=ff.create_distplot(hist_data, group_labels,bin_size=2,colors=colors)
fig.update_layout(title_text="Distribution of Teaming Dependency",title_x=0.5)
fig.show()


# In[ ]:


gm_team=comp_merged['teaming_frequecy'].value_counts().to_frame().reset_index().rename(columns={'index':'teams','teaming_frequecy':'count'})
colors = ['dodgerblue','lightblue','darkblue']
fig=go.Figure(data=go.Pie(labels=list(gm_team['teams']), values=list(gm_team['count']), hoverinfo='label+percent', 
               textinfo='value+percent',marker=dict(colors=colors)))
fig.update_layout( title_text="Distribution of Teaming Dependency",title_x=0.5)

fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * First plot reveals **three modals**.So i did a pie chart with categories.
# * From second plot,we could see there are only few more dependent GMs for teaming.

# <a id="4.6"></a>
# <font size="+2" color="indigo"><b>4.6 Median Competition Ranking</b></font><br>

# There might be some question why not mean instead of median.Few of competitions with high rank collapsed their actual aggression value.So it was ideal to take median over mean to avoid biased value towards maximum ranks.

# In[ ]:


#def get_all_medianrank(id):
 #   all_comp= Submissions.loc[Submissions['SubmittedUserId']==id]['TeamId'].unique()
  #  all_ranks=[]
   # for i in all_comp:   
    #    all_ranks.append(float(Teams.loc[Teams['Id']==i]['PrivateLeaderboardRank']))
    #return np.nanmedian(all_ranks)

#comp_merged['median_ranks']=comp_merged['UserId'].apply(lambda x:get_all_medianrank(x))
#non_comp_merged['median_ranks']=non_comp_merged['UserId'].apply(lambda x:get_all_medianrank(x))


# In[ ]:


fig = make_subplots(rows=1, cols=2,subplot_titles=("GM Ranks", "Non GM Ranks"))

fig.add_trace(go.Histogram(x=comp_merged['median_ranks'],marker = dict(
                                color = 'teal'
                                )),row=1,col=1)

fig.add_trace(go.Histogram(x=non_comp_merged['median_ranks'],marker = dict(
                                color = 'grey'
                                )),row=1,col=2)
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_xaxes(title_text="Year", row=1, col=1)
fig.update_xaxes(title_text="Year", row=1, col=2)
fig.update_yaxes(title_text="Number of Grandmasters", row=1, col=1)
fig.update_yaxes(title_text="Number of Kagglers", row=1, col=2)

fig.update_layout(title_text='Overall Median Ranks', 
    title_x=0.5,showlegend=False)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * We observe that all GMs have been easily finished **median value inside 280**(median value throughout their past kaggle life).This plot reveals their knowledge.

# <a id="4.7"></a>
# <font size="+2" color="indigo"><b>4.7 Average Submission Vs Ranks</b></font><br>

# In[ ]:


#def submission(id):
 #   all_team=TeamMemberships[TeamMemberships['UserId']==id]['TeamId']
 #  all_sub=[]
 #   for i in all_team:
  #      all_sub.append(Submissions[Submissions['TeamId']==i].shape[0])
  #  return np.nanmean(all_sub)
# comp_merged['avg_submission']=comp_merged['UserId'].apply(lambda x:submission(x))


# In[ ]:


x=comp_merged['avg_submission']
y=comp_merged['median_ranks']
fig = go.Figure()
fig.add_trace(go.Histogram2dContour(
        x = x,
        y = y,
        colorscale = 'gray',
        reversescale = True,
        xaxis = 'x',
        yaxis = 'y'
    ))
fig.add_trace(go.Scatter(
        x = x,
        y = y,
        xaxis = 'x',
        yaxis = 'y',
        mode = 'markers',
        marker = dict(
            color = 'teal',  #'rgba(0,0,0,0.3)',
            size = 4
        )
    ))
fig.add_trace(go.Histogram(
        y = y,
        xaxis = 'x2',
        marker = dict(
            color = 'rgba(0,0,0,1)'
        )
    ))
fig.add_trace(go.Histogram(
        x = x,
        yaxis = 'y2',
        marker = dict(
            color = 'rgba(0,0,0,1)'
        )
    ))

fig.update_layout(
    autosize = False,
    xaxis = dict(
        zeroline = False,
        domain = [0,0.85],
        showgrid = False
    ),
    yaxis = dict(
        zeroline = False,
        domain = [0,0.85],
        showgrid = False
    ),
    xaxis2 = dict(
        zeroline = False,
        domain = [0.85,1],
        showgrid = False
    ),
    yaxis2 = dict(
        zeroline = False,
        domain = [0.85,1],
        showgrid = False
    ),
    height = 600,
    width = 600,
    bargap = 0,
    hovermode = 'closest',
    showlegend = False,
    title_text="Median Ranks vs Avg Submission",title_x=0.5,
    xaxis_title="Average Submssion",
    yaxis_title="Median Ranks"
)

fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * We see the clustered zone where GMs with **less than 50 average submissions achieve come inside less than 70 ranks**.
# * Takeaways,they dont do trial and error,infact **work a lot behind every submission**.

# <a id="4.8"></a>
# <font size="+2" color="indigo"><b>4.8 Daily Activity/Submission</b></font><br>

# In[ ]:


cats = ['Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
all_gm_sub=Submissions[Submissions['SubmittedUserId'].isin(comp_merged['UserId'])]['SubmissionDate'].value_counts().to_frame().reset_index().rename(columns={'index':'Date','SubmissionDate':'count'})
all_gm_sub['Date']=pd.to_datetime(all_gm_sub['Date'],format="%m/%d/%Y")
all_gm_sub = all_gm_sub.groupby([all_gm_sub['Date'].dt.date]).mean().reset_index()
all_gm_sub['weekday'] = pd.to_datetime(all_gm_sub['Date']).dt.strftime('%A')  
day_gm=all_gm_sub.groupby('weekday').median().reindex(cats).reset_index()

non_all_gm_sub=Submissions[Submissions['SubmittedUserId'].isin(non_comp_merged['UserId'])]['SubmissionDate'].value_counts().to_frame().reset_index().rename(columns={'index':'Date','SubmissionDate':'count'})
non_all_gm_sub['Date']=pd.to_datetime(non_all_gm_sub['Date'],format="%m/%d/%Y")
non_all_gm_sub = non_all_gm_sub.groupby([non_all_gm_sub['Date'].dt.date]).mean().reset_index()
non_all_gm_sub['weekday'] = pd.to_datetime(non_all_gm_sub['Date']).dt.strftime('%A')  
day_non_gm=non_all_gm_sub.groupby('weekday').median().reindex(cats).reset_index()


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=non_all_gm_sub['Date'], y=np.log(non_all_gm_sub['count']),marker = dict( color = 'grey'),             
                    name='Non GM'))
fig.add_trace(go.Scatter(x=all_gm_sub['Date'], y=np.log(all_gm_sub['count']),marker = dict( color = 'teal'),  
                    name='GM'))
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text='Daily Submission',
    xaxis_title_text='Year', 
    yaxis_title_text='Count', 
    title_x=0.5,showlegend=True)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * We see an equal pattern between GM and Non GM.But there are few peaks from GMs which reveals their hardwork

# <a id="4.9"></a>
# <font size="+2" color="indigo"><b>4.9 Weekdays Activity/Submission</b></font><br>

# In[ ]:


fig = make_subplots(rows=1, cols=2,subplot_titles=("GM Submissions", "Non GM Submissions"))
fig.add_trace(
    go.Bar(x=day_gm['weekday'], y=day_gm['count'],name="GM",marker_color='teal'),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=day_non_gm['weekday'], y=day_non_gm['count'],name="Non Gm",marker_color='grey'),
    row=1, col=2
)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_xaxes(title_text="Days", row=1, col=1)
fig.update_xaxes(title_text="Days", row=1, col=2)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=2)


fig.update_layout(title_text="Weekdays Submission",title_x=0.5)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * These pattern look alike almost 99% by naked eye.
# * As we see some dips in saturdays and sundays,it seems like all kagglers **enjoy/rest weekends** too.

# In[ ]:


del comp_merged,non_comp_merged,Competitions,Submissions,UserAchievements,TeamMemberships,Teams,tier_df_count,tier_year
gc.collect()


# <a id="5">
# <font size="+3" color="blue"><b>5. Notebook GM</b></font>

# In[ ]:


#note_gm_data=Kernels[Kernels['AuthorUserId'].isin(list(notebook_gm['UserId']))]
#note_user_merged['days']=(pd.to_datetime(note_user_merged['TierAchievementDate'])-pd.to_datetime(note_user_merged['RegisterDate'])).dt.days
#note_user_merged['year']=pd.to_datetime(note_user_merged['TierAchievementDate']).dt.year


# <a id="5.1"></a>
# <font size="+2" color="indigo"><b>5.1 Notebooks & GMs over Years</b></font><br>

# In[ ]:


Kernels['year']=pd.to_datetime(Kernels['MadePublicDate']).dt.year
all_nb_year_df=Kernels['year'].value_counts().to_frame().reset_index().rename(columns={'index':'year','year':'count'})
nb_year_df=note_user_merged['year'].value_counts().to_frame().reset_index().rename(columns={'index':'year','year':'count'})


# In[ ]:


fig = make_subplots(rows=2, cols=1,subplot_titles=("#Notebooks over years", "#GM over years"))

fig.add_trace(
    go.Bar(x=all_nb_year_df['year'], y=all_nb_year_df['count'],name="Notebooks",marker_color='blue',text=all_nb_year_df['count'],textposition="outside"),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=nb_year_df['year'], y=nb_year_df['count'],name="GMs",marker_color='darkmagenta',text=nb_year_df['count'],textposition="outside"),
    row=2, col=1
)



fig.update_xaxes(title_text="Year", row=1, col=1)
fig.update_xaxes(title_text="Year", ticktext=["2018", "2019", "2020"],tickvals=["2018", "2019", "2020"], row=2, col=1)
fig.update_yaxes(title_text="Number of Notebooks", row=1, col=1)
fig.update_yaxes(title_text="Number of Grandmasters", row=2, col=1)

fig.update_layout(title_text="Notebooks & GrandMasters",title_x=0.5,height=1000)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * First plot we observe that there is a **dip in 2018**.A downhill of almost 10000 kernels can be found here.
# * The growth rate for GM in 2020 is high. We are in mid 2020,still we have almost got the numbers of last year.

# <a id="5.2"></a>
# <font size="+2" color="indigo"><b>5.2 How many days will it take?</b></font><br>

# I did min-max scaler on days taken by each GM to achieve notebook GM tier and splited them on three categories.

# In[ ]:


#note_user_merged['days_category']=np.where(note_user_merged['diff_days']<=30,"Fast (<30% of days)",
 #                                          np.where((note_user_merged['diff_days']>30)&(note_user_merged['diff_days']<=70),"Moderate (30%-70% of days)",
  #                                        np.where((note_user_merged['diff_days']>70),"Slow (>70% of days)","nan"
   #                                                )))


# In[ ]:


nb_categ=note_user_merged['days_category'].value_counts().to_frame().reset_index().rename(columns={'index':'category','days_category':'count'})
colors = ['dodgerblue','lightblue','darkblue']
fig=go.Figure(data=go.Pie(labels=list(nb_categ['category']), values=list(nb_categ['count']), hoverinfo='label+percent', 
               textinfo='value+percent',marker=dict(colors=colors)))
fig.update_layout( title_text="Distribution of Days ",title_x=0.5)

fig.show()

fig=go.Figure([go.Histogram(x=note_user_merged["days"],nbinsx=100,marker_color='darkmagenta',xbins=dict( 
        size=100
    ))])
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout( title_text="Distribution of Days",title_x=0.5, xaxis_title_text='# of days', 
    yaxis_title_text='# of Grandmasters')
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * **Almost 70%** of kagglers have taken less than 30% of time to become GM comparative to all.
# * We could see first two where majority of notebook GMs fall.

# <a id="5.3"></a>
# <font size="+2" color="indigo"><b>5.3 Number of Notebooks</b></font><br>

# In[ ]:


#def get_total_note_before(id,data):
 #   Kernels['MedalAwardDate']=pd.to_datetime(Kernels['MedalAwardDate'])
  #  achieved_date=pd.to_datetime(data[data['UserId']==id]['TierAchievementDate'].to_string().split(" ")[-1])
   # total= Kernels.loc[(Kernels['AuthorUserId']==id)&(Kernels['MedalAwardDate']<achieved_date)]['Id'].nunique()
   # return total
#note_user_merged['total_note_before']=note_user_merged['UserId'].apply(lambda x:get_total_note_before(x,note_user_merged))

#def get_total_note_medal(id,data):
#    Kernels['MedalAwardDate']=pd.to_datetime(Kernels['MedalAwardDate'])
#    achieved_date=pd.to_datetime(data[data['UserId']==id]['TierAchievementDate'].to_string().split(" ")[-1])
#    total= Kernels.loc[(Kernels['AuthorUserId']==id)&(Kernels['MedalAwardDate']<achieved_date)]['Id'].unique()
#    total_medals=Kernels[(Kernels['Id'].isin(list(total)))&(Kernels['Medal'].notnull())].shape[0]
#    return total_medals
#note_user_merged['total_med_before']=note_user_merged['UserId'].apply(lambda x:get_total_note_medal(x,note_user_merged))


# In[ ]:


colorsIdx = {'Fast (<30% of days)': 'green', 'Moderate (30%-70% of days)': 'orange','Slow (>70% of days)':'red'}

cols= note_user_merged['days_category'].map(colorsIdx)
fig = go.Figure(data=go.Scatter(x=note_user_merged['total_note_before'],y=note_user_merged['DisplayName'] ,mode="markers",marker = dict(size=10,
                                color = cols
                                )))

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text='# of Notebook Created to achieve GM',
    xaxis_title_text='# of Notebooks', 
    title_x=0.5,height=650)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * Green dot represents the kagglers which achieved GM less than 30% of entire time.Then orange goes with 30-70% and red with greater than 70%.
# * We could obseve that most of kagglers have written **less than 40 notebooks**.

# <a id="5.4"></a>
# <font size="+2" color="indigo"><b>5.4 Notebooks & Days</b></font><br>

# In[ ]:


fig = go.Figure(go.Histogram2d(
        x=note_user_merged['days'],
        y=note_user_merged['total_note_before'],
    colorscale='portland'
    ))

fig.update_layout(title_text='# of Days vs # of Notebooks',
    xaxis_title_text='Total Days', # xaxis label
    yaxis_title_text='Total Notebooks ', # yaxis label
    title_x=0.5,showlegend=False)

fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * This is continuation of first plot,we could see more people have made kernels between **20 and 40 with less than 3 years**.

# <a id="5.5"></a>
# <font size="+2" color="indigo"><b>5.5 Views & Votes</b></font><br>

# This graph will try to reveal the quality of kernel they make.For instance,do every user make an upvote when they see their kernel.(1 View=1 Upvote).

# In[ ]:


#note_gm_data['UserId']=note_gm_data['AuthorUserId']
#note_gm_data['created_year']=pd.to_datetime(note_gm_data['CreationDate']).dt.year
#combined_note=pd.merge(note_gm_data,note_user_merged,on="UserId",how="left")

#total_views_nb=Kernels[Kernels['AuthorUserId'].isin(list(notebook_gm['UserId']))]['']
#total_votes_nb=Kernels[Kernels['AuthorUserId'].isin(list(notebook_gm['UserId']))]['TotalVotes']
#total_com_nb=Kernels[Kernels['AuthorUserId'].isin(list(notebook_gm['UserId']))]['TotalComments']
import scipy.stats
corr_vote=round(scipy.stats.pearsonr(combined_note['TotalViews'], combined_note['TotalVotes'])[0],4)
slope, intercept, r_value, p_value, std_err = stats.linregress(combined_note['TotalViews'],combined_note['TotalVotes'])
line = slope*combined_note['TotalViews']+intercept

fig=go.Figure()

fig.add_traces(go.Scatter(
                  x=combined_note['TotalViews'],
                  y=combined_note['TotalVotes'],
                  mode='markers',
                  text=combined_note['DisplayName'],
                  marker = dict(
                  color = 'darkmagenta'
                  )))

fig.add_traces(go.Scatter(
                  x=combined_note['TotalViews'],
                  y=line,
                  mode='lines',
                  marker=go.Marker(color='rgb(31, 119, 180)'),
                  name='Fit'))

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text='# of Views Vs # of Votes',
    xaxis_title_text='Total Views', 
    yaxis_title_text='Total Votes', 
    title_x=0.5,showlegend=False,annotations=[
        dict(
            x=1000000,
            y=2000,
            xref="x",
            yref="y",
            text="Correlation:{}".format(corr_vote),
            showarrow=False,
            arrowhead=7,
            ax=0,
            ay=-40,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#76EE00",
            #opacity=0.8

        )
    ])

fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * We see a strong correlation of **0.83**.This states that they produce real quality kernels.

# <a id="5.6"></a>
# <font size="+2" color="indigo"><b>5.6 Comments by Year</b></font><br>

# In[ ]:


comment_df=combined_note[['DisplayName',"TotalComments",'created_year']]
comment_df=comment_df.groupby(['DisplayName','created_year'])['TotalComments'].sum().reset_index()


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Bar(
            x=comment_df[comment_df['created_year']==2015]['TotalComments'],
            y=comment_df[comment_df['created_year']==2015]['DisplayName'],
            name="2015",
            orientation='h'))
fig.add_trace(go.Bar(
            x=comment_df[comment_df['created_year']==2016]['TotalComments'],
            y=comment_df[comment_df['created_year']==2016]['DisplayName'],
            name="2016",
            orientation='h'))
fig.add_trace(go.Bar(
            x=comment_df[comment_df['created_year']==2017]['TotalComments'],
            y=comment_df[comment_df['created_year']==2017]['DisplayName'],
            name="2017",
            orientation='h'))
fig.add_trace(go.Bar(
            x=comment_df[comment_df['created_year']==2018]['TotalComments'],
            y=comment_df[comment_df['created_year']==2018]['DisplayName'],
            name="2018",
            orientation='h'))
fig.add_trace(go.Bar(
            x=comment_df[comment_df['created_year']==2019]['TotalComments'],
            y=comment_df[comment_df['created_year']==2019]['DisplayName'],
            name="2019",
            orientation='h'))

fig.add_trace(go.Bar(
            x=comment_df[comment_df['created_year']==2020]['TotalComments'],
            y=comment_df[comment_df['created_year']==2020]['DisplayName'],
            name="2020",
            orientation='h'))

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text="Total Comments by Authors",barmode='stack',
    xaxis_title_text='Total Comments', # xaxis label
    yaxis_title_text='Author', # yaxis label
    title_x=0.5,height=600)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * Most User have been active during 2018.

# <a id="5.7"></a>
# <font size="+2" color="indigo"><b>5.7 Trio - Votes,Views,Comments</b></font><br>

# In[ ]:


colorsIdx = {'Fast (<30% of days)': 'green', 'Moderate (30%-70% of days)': 'orange','Slow (>70% of days)':'red'}
cols= combined_note['days_category'].map(colorsIdx)

fig = go.Figure(data=[go.Scatter3d(x=combined_note['TotalViews'], y=combined_note['TotalVotes'], z=combined_note['TotalComments'],
    mode='markers',
    marker=dict(
        size=10,
        color=cols,             
        opacity=0.8
    ),
)])

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),title='Trio',
    scene = dict(xaxis_title='Total Views',
                 yaxis_title='Total Votes',
                zaxis_title='Total Comments'))
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * This is similar plot which we saw in competition section.Most of data points are **clustered at one area**.

# <a id="5.8"></a>
# <font size="+2" color="indigo"><b>5.8 Time Gap between kernel publications</b></font><br>

# This plot will reveal time difference between their kernels publications by each kaggler.

# In[ ]:


combined_note['CreationDate']=pd.to_datetime(combined_note['CreationDate'])
kernel_diff=combined_note[['DisplayName','CreationDate']]


# In[ ]:


temp_list=[]
names=note_user_merged['DisplayName']
for i in names:
    temp_df=kernel_diff[kernel_diff['DisplayName']==i].sort_values(by=['CreationDate'])
    temp_df['diff']=(temp_df['CreationDate']-temp_df['CreationDate'].shift()).dt.days
    temp_list.append((temp_df['diff'].dropna()))
final_arr=np.array(temp_list)


# In[ ]:


colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', 25, colortype='rgb')

fig = go.Figure()
for data_line, color,n in zip(final_arr, colors,names):
    fig.add_trace(go.Violin(x=data_line, line_color=color,name=n))

fig.update_traces(orientation='h', side='positive', width=2, points=False)
fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=True,height=1200)
fig.show()
del kernel_diff,final_arr,temp_list,names


# <a id="5.9"></a>
# <font size="+2" color="indigo"><b>5.9 Versions</b></font><br>

# In[ ]:


#def find_versions(id):
#    if(KernelVersions[(KernelVersions['ScriptId']==id)].shape[0]==1):
#        return KernelVersions[(KernelVersions['ScriptId']==id)].shape[0]
#    else:
#        return KernelVersions[(KernelVersions['ScriptId']==id)&(KernelVersions['VersionNumber'].notnull())].shape[0]
#combined_note['total_versions']=combined_note['Id_x'].apply(lambda x:find_versions(x))

#def find_lines_changed(id):
#    if(KernelVersions[(KernelVersions['ScriptId']==id)&(KernelVersions['VersionNumber'].notnull())].shape[0]>1):
#        return KernelVersions[(KernelVersions['ScriptId']==id)]['LinesChangedFromPrevious'].sum()
#combined_note['lines_changed']=combined_note['Id_x'].apply(lambda x:find_lines_changed(x))


# In[ ]:


combined_note['total_versions'].describe().to_frame()


# In[ ]:


fig=make_subplots(1,2,subplot_titles=("Versions vs Lines Changed","Versions Vs Votes"))
fig.add_trace(go.Scatter(
                  x=combined_note['total_versions'],
                  y=combined_note['lines_changed'],
                  mode='markers',
                  text=combined_note['DisplayName'],
                  marker = dict(
                  color = 'darkmagenta'
                  )),row=1,col=1)
fig.add_trace(go.Scatter(
                  x=combined_note['total_versions'],
                  y=combined_note['TotalVotes'],
                  mode='markers',
                  text=combined_note['DisplayName'],
                  marker = dict(
                  color = 'darkmagenta'
                  )),row=1,col=2)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_xaxes(title_text="Total Versions", row=1, col=1)
fig.update_xaxes(title_text="Total Versions",  row=1, col=2)
fig.update_yaxes(title_text="Total Lines changed", row=1, col=1)
fig.update_yaxes(title_text="Total Votes", row=1, col=2)

fig.update_layout(title_text="Versions",title_x=0.5,showlegend=False)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * They make an average of **5 versions per kernel**.
# * From plot 2,there is n**o big impact of votes in addition of versions.**
# 

# <a id="5.10"></a>
# <font size="+2" color="indigo"><b>5.10 Tags</b></font><br>

# In[ ]:


def get_tags(id):
    temp_list=[]
    tags=KernelTags[KernelTags['KernelId']==id]['TagId']
    for j in tags:
        temp_list.append(Tags[Tags['Id']==j]['Name'].to_string().split("  ")[-1])
    return temp_list
combined_note['tags']=combined_note['Id_x'].apply(lambda x:get_tags(x))


# In[ ]:


temp_list=list(combined_note['tags'])
temp_list_rem=[ele for ele in combined_note['tags'] if ele != []] 
merged = list(itertools.chain(*temp_list_rem))
temp_dict = dict(Counter(merged))
temp_df=pd.DataFrame(temp_dict.items(),columns=['word','count'])
temp_df=temp_df.sort_values(by="count",ascending=False)[:20]


# In[ ]:


fig = go.Figure(data=[go.Bar(
            x=temp_df['word'], y=temp_df['count'],marker=dict(color="darkmagenta"),
            text=temp_df['count'],
            textposition='outside',
        )])

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text="Top 20 tags used by Authors",
    xaxis_title_text='Tags', # xaxis label
    yaxis_title_text='Count', # yaxis label
    title_x=0.5,height=600)

fig.show()


# In[ ]:


fig, (ax1) = plt.subplots(1,1,figsize=[17, 10])
wordcloud1 = WordCloud( background_color='black',colormap="PuRd_r",
                        width=600,
                        height=400).generate(" ".join(merged))
ax1.imshow(wordcloud1,interpolation='bilinear')
ax1.axis('off')
ax1.set_title('Most Used Tags',fontsize=35);


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * **"gpu"** is the most common word.This reveals that they make more competition oriented kernels which fetch them votes.
# *  Other tags are **"eda","data visualization","beginner"**. They also make kernels for beginners.Good to know that.

# <a id="5.11"></a>
# <font size="+2" color="indigo"><b>5.11 Title</b></font><br>

# In[ ]:


get_title=[]
for i in combined_note['Id_x']:
    temp=list(KernelVersions[KernelVersions['ScriptId']==i]['Title'].unique())
    get_title.append(" ".join(temp))
all_get_title = [y for x in get_title for y in x.split(' ')]

#Remove stopwords and punct
stop_words = set(stopwords.words('english'))
filtered_sentence = [w for w in all_get_title if not w in stop_words] 
final_title= [word for word in filtered_sentence if word.isalnum()]

# Making Dataframe
temp_title_dict = dict(Counter(final_title))
temp_title_df=pd.DataFrame(temp_title_dict.items(),columns=['word','count'])
temp_title_df=temp_title_df.sort_values(by="count",ascending=False)[:20]


# In[ ]:


fig = go.Figure(data=[go.Bar(
            x=temp_title_df['word'], y=temp_title_df['count'],marker=dict(color="darkmagenta"),
            text=temp_title_df['count'],
            textposition='outside',
        )])
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text="Top 20 words in Titles",
    xaxis_title_text='Title words', # xaxis label
    yaxis_title_text='Count', # yaxis label
    title_x=0.5,height=600)

fig.show()


# In[ ]:


fig, (ax2) = plt.subplots(1,1,figsize=[17, 10])
wordcloud2 = WordCloud( background_color='black',colormap="BuPu_r",
                        width=600,
                        height=400).generate(" ".join(final_title))
ax2.imshow(wordcloud2,interpolation='bilinear')
ax2.axis('off')
ax2.set_title('Most Used Words in Title',fontsize=35);


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * Again **"EDA"** has topped in the list.Their kernel will surely have EDA works.

# <a id="5.12"></a>
# <font size="+2" color="indigo"><b>5.12 Language War</b></font><br>

# In[ ]:


#def get_lang(id):
#    lang_id=KernelVersions[KernelVersions['ScriptId']==id][:1]['ScriptLanguageId']
#    temp_split= KernelLanguages[KernelLanguages['Id']==int(lang_id)]['DisplayName'].to_string().split(" ")
#    temp_list=[x for x in temp_split if x!=""]
#    temp_list=temp_list.pop()
#    return temp_list
#combined_note['language']=combined_note['Id_x'].apply(lambda x:get_lang(x))
#combined_note.at[16,'language']="Python" # This row had id 7 which didnot exist now.

#note_user_merged.to_csv("note_user_merged.csv",index=False)
#combined_note.to_csv("combined_note.csv",index=False)


# In[ ]:


temp_pie=combined_note['language'].value_counts().to_frame().reset_index().rename(columns={"index":"language","language":"count"})
colors=["darkblue",'dodgerblue']
fig = go.Figure(data=[go.Pie(labels=temp_pie['language'], values=temp_pie['count'], textinfo='label+percent',
                             insidetextorientation='radial',marker=dict(colors=colors)
                            )])
fig.update_layout(title_text="Language War",
    title_x=0.5,height=600)
fig.show()
del temp_pie


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * Expected one.Python wins the race

# <a id="5.13"></a>
# <font size="+2" color="indigo"><b>5.13 Monthly Creation</b></font><br>

# In[ ]:


cats = ['January','February', 'March', 'April', 'May', 'June', 'July','August','September','October','November','December']
all_note_gm_sub=Kernels[Kernels['AuthorUserId'].isin(note_user_merged['UserId'])]['MadePublicDate'].value_counts().to_frame().reset_index().rename(columns={'index':'Date','MadePublicDate':'count'})
all_note_gm_sub['Date']=pd.to_datetime(all_note_gm_sub['Date'],format="%m/%d/%Y")
all_note_gm_sub = all_note_gm_sub.groupby([all_note_gm_sub['Date'].dt.date]).sum().reset_index()

#Weekday Aggregate
day_note_gm=all_note_gm_sub
day_note_gm['monthly'] = pd.to_datetime(all_note_gm_sub['Date']).dt.strftime('%B')  
day_note_gm=all_note_gm_sub.groupby('monthly').sum().reindex(cats).reset_index()

#Weekly Aggregate
all_note_gm_sub['Date']=pd.to_datetime(all_note_gm_sub['Date'])
all_note_gm_sub=all_note_gm_sub.resample('M', on='Date').sum().reset_index()
all_note_gm_sub=all_note_gm_sub.dropna()

non_all_note_gm_sub=Kernels[Kernels['AuthorUserId'].isin(non_note_user_merged['UserId'])]['MadePublicDate'].value_counts().to_frame().reset_index().rename(columns={'index':'Date','MadePublicDate':'count'})
non_all_note_gm_sub['Date']=pd.to_datetime(non_all_note_gm_sub['Date'],format="%m/%d/%Y")
non_all_note_gm_sub = non_all_note_gm_sub.groupby([non_all_note_gm_sub['Date'].dt.date]).sum().reset_index()

#Weekday Aggregate
day_note_non_gm=non_all_note_gm_sub
day_note_non_gm['monthly'] = pd.to_datetime(non_all_note_gm_sub['Date']).dt.strftime('%B')  
day_note_non_gm=non_all_note_gm_sub.groupby('monthly').sum().reindex(cats).reset_index()

#Weekly Aggregate
non_all_note_gm_sub['Date']=pd.to_datetime(non_all_note_gm_sub['Date'])
non_all_note_gm_sub=non_all_note_gm_sub.resample('M', on='Date').sum().reset_index()


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=non_all_note_gm_sub['Date'], y=np.log(non_all_note_gm_sub['count']),marker=dict(color="grey"),
                    #mode='lines',
                    name='Non GM'))
fig.add_trace(go.Scatter(x=all_note_gm_sub['Date'], y=np.log(all_note_gm_sub['count']),marker=dict(color="darkmagenta"),
                    #mode='lines',
                    name='GM'))
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text='Creation of kernels',
    xaxis_title_text='Year', 
    yaxis_title_text='Count', 
    title_x=0.5,showlegend=True)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * This is a cool fact that GMs maintain creation of **4 kernels per month** averaging one per week.This is  **complete consistence**

# <a id="5.14"></a>
# <font size="+2" color="indigo"><b>5.14 Active Month</b></font><br>

# In[ ]:


fig = make_subplots(rows=1, cols=2,subplot_titles=("GM Submissions", "Non GM Submissions"))
fig.add_trace(
    go.Bar(x=day_note_gm['monthly'], y=day_note_gm['count'],name="GM",marker_color='darkmagenta'),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=day_note_non_gm['monthly'], y=day_note_non_gm['count'],name="Non Gm",marker_color='grey'),
    row=1, col=2
)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_xaxes(title_text="Days", row=1, col=1)
fig.update_xaxes(title_text="Days", row=1, col=2)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=2)


fig.update_layout(title_text="Monthly Creation",title_x=0.5)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * There is downfall in mid of every year.Non GM users are less active in winter too.
# * We see that GMs are more active than Non GM users even during **winters**.

# <a id="6">
# <font size="+3" color="blue"><b>6. Discussion GM</b></font>

# In[ ]:


all_gm_discussion_df=ForumMessages[ForumMessages['PostUserId'].isin(list(dis_user_merged['UserId']))]
all_gm_discussion_df=all_gm_discussion_df.reset_index()

def find_name(id):
    name=dis_user_merged[dis_user_merged['UserId']==id]['DisplayName'].to_string().split("  ")[-1]
    return name
all_gm_discussion_df['DisplayName']=all_gm_discussion_df['PostUserId'].apply(lambda x:find_name(x))


# <a id="6.1"></a>
# <font size="+2" color="indigo"><b>6.1 Discussions & GMs over Years</b></font><br>

# In[ ]:


ForumMessages['year']=pd.to_datetime(ForumMessages['PostDate']).dt.year
dis_user_merged['year']=pd.to_datetime(dis_user_merged['TierAchievementDate']).dt.year

all_discussion_year_df=ForumMessages['year'].value_counts().to_frame().reset_index().rename(columns={'index':'year','year':'count'})
discussion_gm_year_df=dis_user_merged['year'].value_counts().to_frame().reset_index().rename(columns={'index':'year','year':'count'})


# In[ ]:


fig = make_subplots(rows=2, cols=1,subplot_titles=("#Discussions over years", "#GM over years"))

fig.add_trace(
    go.Bar(x=all_discussion_year_df['year'], y=all_discussion_year_df['count'],name="Discussions",marker_color='blue',text=all_discussion_year_df['count'],
           textposition='outside'),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=discussion_gm_year_df['year'], y=discussion_gm_year_df['count'],name="GMs",marker_color='sienna',text=discussion_gm_year_df['count'],
           textposition='outside'),
    row=2, col=1
)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_xaxes(title_text="Year", row=1, col=1)
fig.update_xaxes(title_text="Year", ticktext=["2018", "2019", "2020"],tickvals=["2018", "2019", "2020"], row=2, col=1)
fig.update_yaxes(title_text="Number of Notebooks", row=1, col=1)
fig.update_yaxes(title_text="Number of Grandmasters", row=2, col=1)

fig.update_layout(title_text="Discussions & GrandMasters",title_x=0.5,height=1000)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * We see the **consistent growth on number discussions over years**.
# * Discussion Gm are less comparative to Notebook GM.Also the rate of increase is bit low.

# <a id="6.2"></a>
# <font size="+2" color="indigo"><b>6.2 How many days will it take?</b></font><br>

# I did min-max scaler on days taken by each GM to achieve discussion GM tier and splited them on three categories.

# In[ ]:


#dis_user_merged['days']=(pd.to_datetime(dis_user_merged['TierAchievementDate'])-pd.to_datetime(dis_user_merged['RegisterDate'])).dt.days
#dis_user_merged['days_category']=np.where(dis_user_merged['diff_days']<=30,"Fast (<30% of days)",
#                                          np.where((dis_user_merged['diff_days']>30)&(dis_user_merged['diff_days']<=70),"Moderate (30%-70% of days)",
#                                        np.where((dis_user_merged['diff_days']>70),"Slow (>70% of days)","nan"
#                                                )))


# In[ ]:


discussion_day_category_df=dis_user_merged['days_category'].value_counts().to_frame().reset_index().rename(columns={'index':'category','days_category':'count'})

colors = ['dodgerblue','lightblue','darkblue']
fig=go.Figure(data=go.Pie(labels=list(discussion_day_category_df['category']), values=list(discussion_day_category_df['count']), hoverinfo='label+percent', 
               textinfo='value+percent',marker=dict(colors=colors)))
fig.update_layout( title_text="Categories",title_x=0.45)

fig.show()

fig=go.Figure([go.Histogram(x=dis_user_merged["days"],nbinsx=100,marker_color='sienna',xbins=dict( 
        size=100
    ),
    opacity=0.75)])
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout( title_text="Distribution of Days",title_x=0.5,xaxis_title="# of Days",yaxis_title="# of GMs")
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * There are more fast and moderate people here.They are not aggressive as Notebook GMs.Still the rate of speed to attain GM(days) is comparatively slow

# <a id="6.3"></a>
# <font size="+2" color="indigo"><b>6.3 How much discussions are required?</b></font><br>

# In[ ]:


#def get_total_discussion_before(id,data):
#    ForumMessages['MedalAwardDate']=pd.to_datetime(ForumMessages['MedalAwardDate'])
#    achieved_date=pd.to_datetime(data[data['UserId']==id]['TierAchievementDate'].to_string().split(" ")[-1])
#    total= ForumMessages.loc[(ForumMessages['PostUserId']==id)&(ForumMessages['MedalAwardDate']<achieved_date)]['Id'].nunique()
#    return total

#dis_user_merged['total_discussion_before']=dis_user_merged['UserId'].apply(lambda x:get_total_discussion_before(x,dis_user_merged))


# In[ ]:


colorsIdx = {'Fast (<30% of days)': 'green', 'Moderate (30%-70% of days)': 'orange','Slow (>70% of days)':'red'}
cols= note_user_merged['days_category'].map(colorsIdx)
fig = go.Figure(data=go.Scatter(x=dis_user_merged['total_discussion_before'],y=dis_user_merged['DisplayName'] ,mode="markers",marker = dict(size=10,
                                color = cols
                                )))
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text='Comments made to be GMs',
    xaxis_title_text='# of Discussions', 
    title_x=0.5,height=600)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * The points are **outspread**.It is difficult to judge the number of discussions required as all categories are also spread out. 

# <a id="6.4"></a>
# <font size="+2" color="indigo"><b>6.4 Days vs Discussions</b></font><br>

# In[ ]:


#def get_total_discussion_before(id,data):
#    ForumMessages['MedalAwardDate']=pd.to_datetime(ForumMessages['MedalAwardDate'])
#    achieved_date=pd.to_datetime(data[data['UserId']==id]['TierAchievementDate'].to_string().split(" ")[-1])
#    total= ForumMessages.loc[(ForumMessages['PostUserId']==id)&(ForumMessages['MedalAwardDate']<achieved_date)]['Id'].nunique()
#    return total
#dis_user_merged['total_discussion_before']=dis_user_merged['UserId'].apply(lambda x:get_total_discussion_before(x,dis_user_merged))

#def get_total_discussion_medal(id,data):
#    ForumMessages['MedalAwardDate']=pd.to_datetime(ForumMessages['MedalAwardDate'])
#    achieved_date=pd.to_datetime(data[data['UserId']==id]['TierAchievementDate'].to_string().split(" ")[-1])
#    total= ForumMessages.loc[(ForumMessages['PostUserId']==id)&(ForumMessages['MedalAwardDate']<achieved_date)]['Id'].unique()
#    total_medals=ForumMessages[(ForumMessages['Id'].isin(list(total)))&(ForumMessages['Medal'].notnull())].shape[0]
#    return total
#dis_user_merged['total_med_discussion_before']=dis_user_merged['UserId'].apply(lambda x:get_total_discussion_medal(x,dis_user_merged))


# In[ ]:


fig = go.Figure(go.Histogram2d(
        x=dis_user_merged['days'],
        y=dis_user_merged['total_discussion_before'],
    colorscale='portland'
    ))

fig.update_layout(title_text='Days vs Discussions',
    xaxis_title_text='Total Days', # xaxis label
    yaxis_title_text='Total Notebooks', # yaxis label
    title_x=0.5,showlegend=False)

fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * Three red blocks with **non linearity(mixtures of arrangement)** shows there is no much relationship between days and discussions.

# <a id="6.5"></a>
# <font size="+2" color="indigo"><b>6.5 Word Length</b></font><br>

# In[ ]:


def remove_unwanted(text):
    line=re.sub('<.*?>',"",text)
    line=re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',"",line)
    line=re.sub('[.*?]',"",line)
    line=re.sub(r'\n','',line)
    line=re.sub('[^A-Za-z0-9]+',' ',line)
    return line
temp_text=[]
for i in all_gm_discussion_df['Message']:
    temp_text.append(remove_unwanted(str(i)))


# In[ ]:


all_gm_discussion_df['text']=pd.Series(temp_text)
all_gm_discussion_df['word_len']=all_gm_discussion_df['text'].str.split().map(lambda x:len(x))

temp_dis_df=all_gm_discussion_df[['DisplayName','word_len']]
diss_gm_mean=temp_dis_df.groupby('DisplayName').mean().reset_index().rename(columns={'word_len':"mean"})
diss_gm_median=temp_dis_df.groupby('DisplayName').median().reset_index().rename(columns={'word_len':"median"})
diss_word_gm=pd.merge(diss_gm_mean,diss_gm_median,on="DisplayName")


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(x=diss_word_gm['DisplayName'],
               y=diss_word_gm['mean'],
                name='Word Length - Mean',
                marker_color='lightcoral',
                text=round(diss_word_gm['mean']),
               textposition='outside',
                ))
fig.add_trace(go.Bar(x=diss_word_gm['DisplayName'],
                y=diss_word_gm['median'],
                name='Word Length - Median',
                marker_color='indianred',
                text=diss_word_gm['median'],
                textposition='outside',
                ))
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text="Mean/Median Word Length",
    xaxis_title_text='GMs', # xaxis label
    yaxis_title_text='Mean & Median', # yaxis label
    title_x=0.5,height=600)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * Most of authors write **around 20-40 word length** comment.We heva authors who use less than 10 words too.

# <a id="6.6"></a>
# <font size="+2" color="indigo"><b>6.6 Most Word Used</b></font><br>

# In[ ]:


all_message = [y.lower() for x in all_gm_discussion_df['text'] for y in x.split(' ')]
stop_words = set(stopwords.words('english'))
filtered_message = [w for w in all_message if not w in stop_words] 
final_message= [word for word in filtered_message if word.isalnum()]

temp_msg_dict = dict(Counter(final_message))
temp_msg_df=pd.DataFrame(temp_msg_dict.items(),columns=['word','count'])
temp_msg_df=temp_msg_df.sort_values(by="count",ascending=False)[:20]


# In[ ]:


fig = go.Figure(data=[go.Bar(
            x=temp_msg_df['word'], y=temp_msg_df['count'],marker_color="sienna",
            text=temp_title_df['count'],
            textposition='outside',
        )])
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text="Top 20 Words in Commments",
    xaxis_title_text='Comment Words', # xaxis label
    yaxis_title_text='Count', # yaxis label
    title_x=0.5,height=600)

fig.show()


# In[ ]:


fig, (ax2) = plt.subplots(1,1,figsize=[17, 10])
wordcloud2 = WordCloud( background_color='black',colormap="BrBG",  #OrRd_r
                        width=600,
                        height=400).generate(" ".join(final_message))
ax2.imshow(wordcloud2,interpolation='bilinear')
ax2.axis('off')
ax2.set_title('Most Used Words in Discussion',fontsize=35);


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * **"data","lb" ,"model","competition"** are most common words which reveals that they can be more of competition oriented.
# * Other words like **"score","image","feature"** are commonly used.
# * **Thanking gratitude** exists in kaggle.

# <a id="6.7"></a>
# <font size="+2" color="indigo"><b>6.7 Do they reply?</b></font><br>

# In[ ]:


#def total_discussion(id):
 #   return all_gm_discussion_df[all_gm_discussion_df['PostUserId']==id].shape[0]
#def total_replies(id):
#    return all_gm_discussion_df[(all_gm_discussion_df['ReplyToForumMessageId'].notnull())&(all_gm_discussion_df['PostUserId']==id)].shape[0]

#dis_user_merged['total_discussions']=dis_user_merged['UserId'].apply(lambda x:total_discussion(x))
#dis_user_merged['total_replies']=dis_user_merged['UserId'].apply(lambda x:total_replies(x))
#dis_user_merged['reply_percent']=(dis_user_merged['total_replies']/dis_user_merged['total_discussions'])*100
#dis_user_merged.to_csv("dis_user_merged.csv",index=False)
gm_reply=round(all_gm_discussion_df[all_gm_discussion_df['ReplyToForumMessageId'].notnull()]['DisplayName'].shape[0]/all_gm_discussion_df.shape[0]*100)


# In[ ]:


fig = go.Figure(go.Indicator(
    mode = "number",
    value = gm_reply,
    title = {"text": "Overall Reply Percentage"},
    domain = {'y': [0, 1], 'x': [0.50, 0.50]}))

fig.add_trace(go.Histogram(x=dis_user_merged['reply_percent'],nbinsx=20,marker_color='sienna',
    opacity=0.75))
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text='Distribution of Replies',
    xaxis_title_text='# of GM',
    yaxis_title_text='Replying percentage',
    title_x=0.5)

fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * We observe **three modal**.And it is uniformaly distrbuted.It is not ideal to judge this with very few data.But with given,we conclude this is so random that they reply.
# * Overall average reply percentage is **52%**.Even the density plot proves this with the randomness

# <a id="6.8"></a>
# <font size="+2" color="indigo"><b>6.8 Weekly Post</b></font><br>

# In[ ]:


cats = ['Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

all_dis_gm_sub=ForumMessages[ForumMessages['PostUserId'].isin(dis_user_merged['UserId'])]['PostDate'].value_counts().to_frame().reset_index().rename(columns={'index':'Date','PostDate':'count'})
all_dis_gm_sub['Date']=pd.to_datetime(all_dis_gm_sub['Date'],format="%m/%d/%Y %H:%M:%S")
all_dis_gm_sub = all_dis_gm_sub.groupby([all_dis_gm_sub['Date'].dt.date]).sum().reset_index()

#Weekday Aggregate
day_diss_gm=all_dis_gm_sub
day_diss_gm['weekday'] = pd.to_datetime(all_dis_gm_sub['Date']).dt.strftime('%A')  
day_diss_gm=all_dis_gm_sub.groupby('weekday').mean().reindex(cats).reset_index()

# Weekly Aggregate
all_dis_gm_sub['Date']=pd.to_datetime(all_dis_gm_sub['Date'])
all_dis_gm_sub=all_dis_gm_sub.resample('W-Mon', on='Date').mean().reset_index()
all_dis_gm_sub=all_dis_gm_sub.dropna()


non_all_dis_gm_sub=ForumMessages[ForumMessages['PostUserId'].isin(non_dis_user_merged['UserId'])]['PostDate'].value_counts().to_frame().reset_index().rename(columns={'index':'Date','PostDate':'count'})
non_all_dis_gm_sub['Date']=pd.to_datetime(non_all_dis_gm_sub['Date'],format="%m/%d/%Y  %H:%M:%S")
non_all_dis_gm_sub = non_all_dis_gm_sub.groupby([non_all_dis_gm_sub['Date'].dt.date]).sum().reset_index()

# Weekday Aggregate
day_diss_non_gm=non_all_dis_gm_sub
day_diss_non_gm['weekday'] = pd.to_datetime(non_all_dis_gm_sub['Date']).dt.strftime('%A')  
day_diss_non_gm=day_diss_non_gm.groupby('weekday').mean().reindex(cats).reset_index()

# Weekly Aggregate
non_all_dis_gm_sub['Date']=pd.to_datetime(non_all_dis_gm_sub['Date'])
non_all_dis_gm_sub=non_all_dis_gm_sub.resample('W-Mon', on='Date').mean().reset_index()


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=pd.to_datetime(all_dis_gm_sub['Date']), y=np.log(all_dis_gm_sub['count']),marker_color="sienna",
                    name='GM'))
fig.add_trace(go.Scatter(x=pd.to_datetime(non_all_dis_gm_sub['Date']), y=np.log(non_all_dis_gm_sub['count']),marker_color="grey",
                    name='Non GM'))
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text='Weekly Message count',
    xaxis_title_text='Year', 
    yaxis_title_text='Count', 
    title_x=0.5,showlegend=True)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * Weekly posts have been grown to 3-4 in last two years.This could be biased as most of users may be writing more comments.

# <a id="6.9"></a>
# <font size="+2" color="indigo"><b>6.9 Weekdays Post</b></font><br>
# 

# In[ ]:


fig = make_subplots(rows=1, cols=2,subplot_titles=("GM Discussions", "Non GM Discussions"))
fig.add_trace(
    go.Bar(x=day_diss_gm['weekday'], y=day_diss_gm['count'],name="GM",marker_color='sienna'),
    row=1, col=1)
fig.add_trace(
    go.Bar(x=day_diss_non_gm['weekday'], y=day_diss_non_gm['count'],name="Non Gm",marker_color='grey'),
    row=1, col=2)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_xaxes(title_text="Days", row=1, col=1)
fig.update_xaxes(title_text="Days", row=1, col=2)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=2)


fig.update_layout(title_text="Average Weekdays Discussions",title_x=0.5)
fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * We observe same pattern in both GMs and Non GMs.Except on wednesday,GMs have contributed less comparative to Non GMs.

# <font size="+3" color="blue"><b>7. Where are they from?</b></font><br><a id="7"></a>

# In[ ]:


#all_gm_list=UserAchievements.loc[(UserAchievements['Tier']==4)]['UserId']
#all_gm=Users[Users['Id'].isin(list(all_gm_list))].copy()
#all_gm.reset_index(drop=True, inplace=True)
#all_gm.drop(columns=['PerformanceTier'], inplace=True)
#all_gm = all_gm.convert_dtypes()
#display_html(scrap_data(all_gm), cols=['UserName', 'DisplayName','Country'])
#all_gm.to_csv("all_gm_loc.csv",index=False)

country_code=country_code.rename(columns={'COUNTRY':'Country'})
gm_location=all_gm_loc['Country'].value_counts().to_frame().reset_index().rename(columns={'index':'Country','Country':'gm_count'})
gm_location=pd.merge(gm_location,country_code,on="Country")


# In[ ]:


fig = go.Figure(data=go.Choropleth(
    locations = gm_location['CODE'],
    z = gm_location['gm_count'],
    text = gm_location['Country'],
    colorscale = 'mint',
    autocolorscale=False,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    #colorbar_tickprefix = '$',
    colorbar_title = '# GM',
))

fig.update_layout(
    title_text='GMs all over the world',title_x=0.5,
    geo=dict(
        showframe=True,
        showcoastlines=False,
        projection_type='equirectangular'
    )
)

fig.show()


# <font size="+1" color="green"><b>Observation:</b></font><br>
# 
# * Most of GMs are from **Unites States,Russia and China**.

# <font size="+3" color="blue"><b>8. Findings/Takeaways</b></font><br><a id="8"></a>

# <font color="teal">
# <b><u>Competition GM Takeaways:</u></b>
#     
# * Need to be a good problem solver.
# * Days doesnt matter.
# * Competitions: Number of competitions is not important.Most kagglers explore it to learn.
# * Frequency: Active participation is more important.
# * Teams : Teaming is good.GMs utilize more teamings
# * Submission : Quality submission is better than Trial and Error. 
# * Time off : Weekends are important too.
# 
# 
# <font color="darkmagenta">
# <b><u>Notebook GM Takeaways:</u></b>
#     
# * Days doesnt matter.
# * Kernels published frequently.
# * Make quality kernels.
# * Versions dont really increase votes.
# * Competition oriented kernels are the gain more attention.
# * Titles : Simple title matters most.
# * Language : Python is attracted to most.
# * Publication : 3-4 per month is very ideal.
# * Active time : June - September,the noise is less which can be good time to get attention from the rest of people.
# 
# <font color="sienna">
# <b><u>Discussion GM Takeaways:</u></b>
#     
# * Be Active daily
# * Days doesnt matter.
# * Discussion Length: Explanation in comment with 20-40 words.
# * Keywords: "data" ,"lb","model" - Give your research idea on them.
# * Reply or not : Reply for valid questions.
# 

# <font size="+3" color="blue"><b>9. EndNotes</b></font><br><a id="9"></a>

# We are into end now.Throughout this analysis i could see the effort which a grandmaster takes to achieve GM position.Definitely this is serious hardwork which they do daily on the kaggle platform.So those who wish to become a grandmaster,just apply their process which i mentioned above.
# 
# **Next things in this kernel:** I will keep on researching to bring new insights on how they work and their success story.So keep it in your favourite list by upvoting it.
# 
# *All the best! Happy Kaggling*

# <font size=+1><b>I hope you enjoyed the story and lifestyle of Grandmasters in our platform.I wish that I made your time worth to read.If you really want to appreciate me,please give an </b></font><font size=+1 color="red"><b>Upvote</b></font><font size=+1><b> which encourages me to do more quality kernels and feel free to comment about what you liked.</b></font>

# <font size="+2" color="chocolate"><b>My Other Kernels</b></font><br>
# 
# Click on the buttons to view kernel...
# 
# <a href="https://www.kaggle.com/raenish/don-t-shoot" class="btn btn-primary" style="color:white;">Dont Shoot</a>
# 
# <a href="https://www.kaggle.com/raenish/cheatsheet-100-plotly-part-1-basic" class="btn btn-primary" style="color:white;">100+ Plotly Basic</a>
# 
# <a href="https://www.kaggle.com/raenish/cheatsheet-100-plotly-part-2-advanced" class="btn btn-primary" style="color:white;">100+ Plotly Advanced</a>
# 
# <a href="https://www.kaggle.com/raenish/cheatsheet-date-helpers" class="btn btn-primary" style="color:white;">Cheatsheet Date Helpers</a>
# 
# <a href="https://www.kaggle.com/raenish/cheatsheet-text-helper-functions" class="btn btn-primary" style="color:white;">Cheatsheet Text Helpers</a>
# 
# <a href="https://www.kaggle.com/raenish/tweet-sentiment-insight-eda/" class="btn btn-primary" style="color:white;">Tweet Sentiment Extraction</a>
# <br>
# <br>
# ### If these kernels impress you,give them an <font size="+2" color="red"><b>Upvote</b></font>.<br>
# 
# 

# <a href="#top" class="btn btn-success btn-lg active" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to Colors">Go to TOP</a>
