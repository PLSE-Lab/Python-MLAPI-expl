#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">Meta Kaggle: What happened to the team size?</font></center></h1>
# 
# <h3><center><font size="4">An investigation in team size evolution for winning teams in Kaggle competitions</font></center></h3>  
# <br>
# 
# 
# <img src="https://lh3.googleusercontent.com/4IWRQx5munaM7OjDxHecuxHxjaL8YJz_2v3nGw_PX1h-vrhvC0f_cmlQI0hOuQ2rBQWlwVYR-SJd6PAcepi4lgzIN1mgaLS5QhqqUKnsix8vmCRvi6ZOILgd0haPbYhlXgLerNKl" width=600></img>
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>  
# - <a href='#2'>Competitions</a>  
#   - <a href='#21'>Check the data</a>   
#   - <a href='#22'>Competition types</a>   
#   - <a href='#23'>Number of competitions, grouped by year and Max Team Size</a>
#   - <a href='#24'>Number of competitions, grouped by year, Max Team Size and Host Segment Title</a>
#   - <a href='#25'>Competition reward</a>
# - <a href='#3'>Teams</a>   
#   - <a href='#31'>Check the data</a>   
#   - <a href='#32'>Teams per year and teams per year and team size</a>   
#   - <a href='#33'>Number of teams per team size and year heatmap (all competitions)</a>   
#   - <a href='#34'>Number of teams for team size and year heatmap (no InClass competitions)</a>   
#   - <a href='#35'>Time variation of number of teams vs. team size (with plotly and blobby)</a>   
#   - <a href='#36'>Time variation of number of winning teams vs. team size (with plotly and blobby)</a>   
#       - <a href='#361'>Featured competitions</a>   
#       - <a href='#362'>Research competitions</a>   
#   - <a href='#37'>Teams size and teams rankings</a>     
#   - <a href='#38'>Teams size and teams rankings per year</a>    
# - <a href='#4'>Conclusions</a>     
#  - <a href='#41'>Final remarks - and a possible explanation of the perception</a>    
# - <a href='#5'>References</a>   
# 
# 

# # <a id="1">Introduction</a>  
# 
# This Kernel objective is to investigate how the team sizes (in some cases limited by competitions, in other cases free) evolved in time.   
# 
# We will use the data from **Meta Kaggle**<a href='#4'>[1]</a>, a dataset with meta-information from Kaggle about Datasets, Competitions, Users, Teams. We will focus in this Kernel on Competitions, Teams and Team Membership information.
# 
# We will try to understand how many competitions limited the team size (**MaxTeamSize**) each year. The type of the competition is also an important factor and will show the results as well grouped on competition type (**HostSegmentTitle**).   
# 
# Then, we will look to the number of teams per year and the number of teams, grouped by year and team size.    
# 
# **Important note**: originally, the data was analyzed in **Sept, 2018**; the data is last updated on **September, 2019**.   
# 
# The dataset is constantly updated by Kaggle team, therefore I will update this Kernel regularly, so, stay tuned!
# 

# ## Load packages

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from bubbly.bubbly import bubbleplot 
from __future__ import division
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode()
IS_LOCAL = False
import os
if(IS_LOCAL):
    PATH="../input/meta-kaggle"
else:
    PATH="../input/meta-kaggle"
print(os.listdir(PATH))


# ## Read the data

# In[ ]:


competition_df = pd.read_csv(os.path.join(PATH,"Competitions.csv"))
teams_df = pd.read_csv(os.path.join(PATH,"Teams.csv"))
team_membership_df = pd.read_csv(os.path.join(PATH,"TeamMemberships.csv"))


# ## Check the data

# In[ ]:


print("Meta Kaggle competition data -  rows:",competition_df.shape[0]," columns:", competition_df.shape[1])
print("Meta Kaggle teams data -  rows:",teams_df.shape[0]," columns:", teams_df.shape[1])
print("Meta Kaggle team memberships data -  rows:",team_membership_df.shape[0]," columns:", team_membership_df.shape[1])


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# # <a id="2">Competitions</a>

# ## <a id="21">Check the data</a>
# 
# Let's inspect the competition data. We will also look to the columns for missing data.

# In[ ]:


competition_df.head()


# In[ ]:


competition_df.describe()


# We will extract the **Deadline Year** from the **Deadline Date**.

# In[ ]:


competition_df["DeadlineYear"] = pd.to_datetime(competition_df['DeadlineDate']).dt.year


# ## <a id="22">Competition types</a>
# 
# Let's verify how many competitions types are. There are two fields,  **CompetitionTypeId** and **HostSegmentTitle**. As indicated by [James Trotman,](https://www.kaggle.com/jtrotman) the second is more meaningful for our purpose. Let's visualize the host segment title distribution.

# In[ ]:


tmp = competition_df.groupby('HostSegmentTitle')['Id'].nunique()
df = pd.DataFrame(data={'Competitions': tmp.values}, index=tmp.index).reset_index()
trace = go.Bar(
    x = df['HostSegmentTitle'],y = df['Competitions'],
    name="Competitions",
    marker=dict(color="blue"),
    text=df['HostSegmentTitle']
)
data = [trace]
layout = dict(title = 'Competitions per type',
          xaxis = dict(title = 'Competitioon Type', showticklabels=True), 
          yaxis = dict(title = 'Number of competitions'),
          hovermode = 'closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='competition-types')


# Most of the competitions are of type **InClass** (1045), on second position are the **Featured** then **Research** and **Playground** and **Recruitment**. 
# 
# Let's look closer to some of the features more relevant for our subject.
# 
# **Note**: from Sept, 2018 to March, 2019, the number of InClass competitions raised with more than 500.  
# 
# **Note**: due to a persistent error, the Kernels using **Meta Kaggle** dataset appears to have the dataset no longer available.

# In[ ]:


var = ["DeadlineDate", "DeadlineYear", "CompetitionTypeId", "HostSegmentTitle", "TeamMergerDeadlineDate", "TeamModelDeadlineDate", "MaxTeamSize", "BanTeamMergers"]
competition_df[var].head(5)


# Let's also check missing data.

# In[ ]:


def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))
missing_data(competition_df[var])


# [](http://)We can see that in 84.4% (as of September 2019) of the cases, **MaxTeamSize** is not set. This means that the team size is not restricted.
# 
# Let's replace not defined **MaxTeamSize** with **-1**.

# In[ ]:


competition_df.loc[competition_df['MaxTeamSize'].isnull(),'MaxTeamSize'] = -1


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# ## <a id="23">Number of competitions, grouped by year and Max Team Size</a>
# 
# 
# Let's show the number of competitions having a certain MaxTeamSize, grouped by year.

# In[ ]:


tmp = competition_df.groupby('DeadlineYear')['MaxTeamSize'].value_counts()
df = pd.DataFrame(data={'Competitions': tmp.values}, index=tmp.index).reset_index()


# In[ ]:


dataset = df[df['MaxTeamSize']>-1]
max_team_sizes = (dataset.groupby(['MaxTeamSize'])['MaxTeamSize'].nunique()).index
data = []
for max_team_size in max_team_sizes:
    dts = dataset[dataset['MaxTeamSize']==max_team_size]
    trace = go.Bar(
        x = dts['DeadlineYear'],y = dts['Competitions'],
        name=max_team_size,
        text=('Max team size:{}'.format(max_team_size))
    )
    data.append(trace)
    
layout = dict(title = 'Number of competitions with size of max team set per year',
          xaxis = dict(title = 'Year', showticklabels=True), 
          yaxis = dict(title = 'Number of competitions'),
          hovermode = 'closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='competition-types')


# In[ ]:


tmp = competition_df[competition_df['MaxTeamSize']>-1]['DeadlineYear'].value_counts()
df = pd.DataFrame(data={'Competitions': tmp.values}, index=tmp.index).reset_index()
trace = go.Bar(
    x = df['index'],y = df['Competitions'],
    name='Competition',
    marker=dict(color="red")
)
data = [trace]
    
layout = dict(title = 'Total number of competitions with size of max team set per year',
          xaxis = dict(title = 'Year', showticklabels=True), 
          yaxis = dict(title = 'Number of competitions'),
          hovermode = 'closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='competition-types')


# In[ ]:


tmp = competition_df['DeadlineYear'].value_counts()
df = pd.DataFrame(data={'Competitions': tmp.values}, index=tmp.index).reset_index()
trace = go.Bar(
    x = df['index'], y = df['Competitions'],
    name='Competition',
    marker=dict(color="blue")
)
data = [trace]    
layout = dict(title = 'Total number of competitions per year',
          xaxis = dict(title = 'Year', showticklabels=True), 
          yaxis = dict(title = 'Number of competitions'),
          hovermode = 'closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='competition-types')


# We can observe few interesting things:    
# * In 2012, 2014 and 2015 there were competitions (one per year) with maximum team size of 10;  
# * In 2012 there was a competition with maximum team size of 8;
# * In 2014 there was also a competition with maximum team size of 7;   
# * In 2017, the number of competitions increased to more than double the number in the previous year, 2016;  
# * Also in 2017 the number of competition limiting the number of team members increased to more than double  the number in the previous year, 2016; also the number of competitions limiting to only one team member in a team was very large (70% of all competitions);   
# * In 2018, the number of competitions was larger (until **Sept 1**, when the data was updated) than in 2017 (for whole year); in the same time, the number of competitions with limited number of team members decreased to a number smaller than the one in 2016.  Update: there was a competition with maximum team size of 8;  
# * in 2019, the data is incomplete; we should disregard the results for 2019, since they are really just partial, from the first few days of the year.  In this year there is already a competition with 8 maximum team size;
# 
# Let's look also to the distribution of the competitions grouped on year, max team size and host segment title.  
#  **MaxTeamSize** = <font color="red">**-1**</font> means that there is no Max Team Size set (we replaced NaN with -1).
#  
#  <a href="#0"><font size="1">Go to top</font></a>  
#  
#  
#  ## <a id="24">Number of competitions, grouped by year, Max Team Size and Host Segment Title</a>
# 
# 

# In[ ]:


tmp = competition_df.groupby(['DeadlineYear','MaxTeamSize'])['HostSegmentTitle'].value_counts()
df = pd.DataFrame(data={'Competitions': tmp.values}, index=tmp.index).reset_index()
df['CompLog'] = np.log(df['Competitions'] + 1)
hover_text = []
for index, row in df.iterrows():
    hover_text.append(('Year: {}<br>'+
                      'Competitions: {}<br>'+
                      'Max Team Size: {}<br>'+
                      'Competition type: {}').format(row['DeadlineYear'],
                                                row['Competitions'],
                                                row['MaxTeamSize'],
                                                row['HostSegmentTitle']
                                            ))
df['hover_text'] = hover_text
competition_type = (df.groupby(['HostSegmentTitle'])['HostSegmentTitle'].nunique()).index
data = []
for comptype in competition_type:
    dfL = df[df['HostSegmentTitle']==comptype]
    trace = go.Scatter3d(
        x = dfL['DeadlineYear'],y = dfL['CompLog'],
        z = dfL['MaxTeamSize'],
        name=comptype,
        marker=dict(
            symbol='circle',
            sizemode='area',
            sizeref=0.01,
            size=dfL['CompLog'] + 2,
        ),
        mode = "markers",
        text=dfL['hover_text'],
    )
    data.append(trace)
    
layout = go.Layout(title = 'Number of competitions per year and maximum team size, grouped by competition type',
         scene = dict(
                xaxis = dict(title = 'Year'),
                yaxis = dict(title = 'Competitions [log scale]'), 
                zaxis = dict(title = 'Maximum Team Size'),
                hovermode = 'closest',  
         )
                  )
fig = dict(data = data, layout = layout)
iplot(fig, filename='competitions-year-comp_type-maxteamsize')


# We can make few observations on the competitions maximum team size, grouped per year:  
# 
# * In 2010 and 2011 all competitions were with maximum team size field not set;  
# * In 2012 there were one competition with maximum team size of 8 and another one with maximum team size of 10 (of type *Prospect*);  
# * In 2014 there was a **InClass** competition with the **MaxTeamSize** set to 10;  
# * In 2015  there was a **InClass** competition with the **MaxTeamSize** set to 10;  
# We can see that most of the competitions with **MaxTeamSize** not set (-1) are **InClass** competitions since 2016. In 2017 and 2018 allmost all competitions are **InClass**.    
# 
# If we look to the competitions that are not **InClass**, we can see that we do have majority of competitions either **Featured** and **Research** and most are with no **MaxTeamSize** set.    
# 
# From the competitions with deadline in 2018, there is only one **Featured** competition with **MaxTeamSize** set to 3. All the rest of **Featured** competitions with the deadline in 2018 have not a **MaxTeamSize** set.

# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# ## <a id="25">Competition Reward</a>
# 
# We will analyze the competition reward, grouped by year and competition type. We exclude the competitions of type **InClass** (with 0 reward). We show the number of competitions, grouped by competition type, for each year, separately.

# In[ ]:


tmp = competition_df[competition_df['HostSegmentTitle']!='InClass']
tmp = tmp.groupby(['DeadlineYear','RewardQuantity'])['HostSegmentTitle'].value_counts()
df = pd.DataFrame(data={'Competitions': tmp.values}, index=tmp.index).reset_index()
df['RewardLog'] = np.log(df['RewardQuantity'] + 1)
hover_text = []
for index, row in df.iterrows():
    hover_text.append(('Year: {}<br>'+
                      'Competitions: {}<br>'+
                      'Reward: {}<br>'+
                      'Competition type: {}').format(row['DeadlineYear'],
                                                row['Competitions'],
                                                row['RewardQuantity'],
                                                row['HostSegmentTitle']
                                            ))
df['hover_text'] = hover_text
competition_type = (df.groupby(['HostSegmentTitle'])['HostSegmentTitle'].nunique()).index
data = []
for comptype in competition_type:
    dfL = df[df['HostSegmentTitle']==comptype]
    trace = go.Scatter3d(
        x = dfL['DeadlineYear'],y = dfL['Competitions'],
        z = dfL['RewardLog'],
        name=comptype,
        marker=dict(
            symbol='circle',
            sizemode='area',
            sizeref=0.01,
            size=dfL['RewardLog'] + 2,
        ),
        mode = "markers",
        text=dfL['hover_text'],
    )
    data.append(trace)
    
layout = go.Layout(title = 'Number of competitions per year and reward amount, grouped by competition type',
         scene = dict(
                xaxis = dict(title = 'Year'),
                yaxis = dict(title = 'Competitions'), 
                zaxis = dict(title = 'Reward Quantity [log scale]'),
                hovermode = 'closest',  
         ))
fig = dict(data = data, layout = layout)
iplot(fig, filename='competitions-year-comp_type-reward')


# Let's see what is the total amount per year for the rewards, grouped by competition.   
# We exclude **InClass** competitions.   
# The year **2019** has only partial data.
# 

# In[ ]:


#exclude "inClass"
tmp = competition_df[competition_df['HostSegmentTitle']!='InClass']
tmp = tmp.groupby(['DeadlineYear','HostSegmentTitle'])['RewardQuantity'].sum()
df = pd.DataFrame(data={'Total amount': tmp.values}, index=tmp.index).reset_index()


# In[ ]:


host_segment_titles = (df.groupby(['HostSegmentTitle'])['HostSegmentTitle'].nunique()).index
data = []
for host_segment_title in host_segment_titles:
    dts = df[df['HostSegmentTitle']==host_segment_title]
    trace = go.Bar(
        x = dts['DeadlineYear'],y = dts['Total amount'],
        name=host_segment_title,
        text=('Competition type:{}'.format(host_segment_title))
    )
    data.append(trace)

layout = dict(title = ('Total amount of reward per year, grouped by Competition type'),
          xaxis = dict(title = 'Year', showticklabels=True), 
          yaxis = dict(title = 'Total amount'),
          hovermode = 'closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='competition-types')


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# # <a id="3">Teams</a>   
# 
# We continue with exploration of team information.
# 
# ## <a id="31">Check the data</a>
# 
# Let's inspect now the team and team membership datasets.

# In[ ]:


teams_df.head(5)


# In[ ]:


teams_df.describe()


# Let's check as well the missing data.

# In[ ]:


missing_data(teams_df)


# In[ ]:


print(f"Teams: {teams_df.shape[0]} different teams: {teams_df.Id.nunique()}")


# We can see that we have over **2.8M** teams registered in more than 10K competitions, with more than 10M submissions.   
# Let's look now to the team membership.   
# We can merge team data with competition data (we do not have missing **CompetitionId**, which is the merge field.
# 
# **Late note**: since the last run of this Notebook, with the updated data, we only see **2.0 M** teams registered.

# In[ ]:


team_membership_df.head(5)


# In[ ]:


team_membership_df.describe()


# Let's check as well the missing data.

# In[ ]:


missing_data(team_membership_df)


# 1. **RequestDate** is missing in less than 0.5% of the cases.  
# 
# We can merge team membership data with team data (we do not have missing **TeamId**, which is the merge field.

# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# 
# ## <a id="32">Teams per year and teams per year and team size</a>
# 
# Let's check now the number of teams per year. We will merge Competitions, Teams and Team Membership data.

# In[ ]:


comp_team_df = competition_df.merge(teams_df, left_on='Id', right_on='CompetitionId', how='inner')
comp_team_membership_df = comp_team_df.merge(team_membership_df, left_on='Id_y', right_on='TeamId', how='inner')


# Let's plot the number of teams per year and also the number of teams per year and per number of team members.  
# We prepare the dataframe with the number of teams per year and team size.

# In[ ]:


tmp = comp_team_membership_df.groupby(['DeadlineYear','TeamId'])['Id'].count()
df = pd.DataFrame(data={'Teams': tmp.values}, index=tmp.index).reset_index()
tmp = df.groupby(['DeadlineYear','Teams']).count()
df2 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()
df2.columns = ['Year', 'Team size','Teams']


# In[ ]:


def plot_heatmap_count(data_df,feature1, feature2, color, title):
    matrix = data_df.pivot(feature1, feature2, 'Teams')
    fig, (ax1) = plt.subplots(ncols=1, figsize=(16,6))
    sns.heatmap(matrix, 
        xticklabels=matrix.columns,
        yticklabels=matrix.index,ax=ax1,linewidths=.1,linecolor='darkblue',annot=True,cmap=color)
    plt.title(title, fontsize=14)
    plt.show()


# Let's show now the number of teams grouped by year.

# In[ ]:


tmp = comp_team_df['DeadlineYear'].value_counts()
df = pd.DataFrame(data={'Teams': tmp.values}, index=tmp.index).reset_index()
trace = go.Bar(
    x = df['index'], y = df['Teams'],
    name='Team',
    marker=dict(color="blue")
)
data = [trace]    
layout = dict(title = 'Total number of teams per year',
          xaxis = dict(title = 'Year', showticklabels=True), 
          yaxis = dict(title = 'Number of teams'),
          hovermode = 'closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='team-types')


# Please note that the results for 2019 are just partial.
# 
# Let's also show the number of teams grouped by year and by competition type.

# In[ ]:


tmp = comp_team_df.groupby('DeadlineYear')['HostSegmentTitle'].value_counts()
df = pd.DataFrame(data={'Competitions': tmp.values}, index=tmp.index).reset_index()


# In[ ]:


host_segment_titles = (df.groupby(['HostSegmentTitle'])['HostSegmentTitle'].nunique()).index
data = []
for host_segment_title in host_segment_titles:
    dts = df[df['HostSegmentTitle']==host_segment_title]
    trace = go.Bar(
        x = dts['DeadlineYear'],y = dts['Competitions'],
        name=host_segment_title,
        text=('Competition type:{}'.format(host_segment_title))
    )
    data.append(trace)

layout = dict(title = ('Number of teams per year, grouped by HostSegmentTitle (Competition type)'),
          xaxis = dict(title = 'Year', showticklabels=True), 
          yaxis = dict(title = 'Competitions'),
          hovermode = 'closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='competition-types')


# We can see that most of the teams are for the **Featured** competitions.

# Let's show now the number of teams per year and per number of team members.
# 
# 
# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# ## <a id="33">Number of teams per team size and year heatmap (all competitions)</a>

# In[ ]:


df2.head()


# In[ ]:


plot_heatmap_count(df2,'Team size','Year', 'Greens', "Number of teams grouped by year and by team size")


# We can see that large teams were not restricted to 2018. The largest team were actually in:
# 
# * **2012** (**40** and **23** team members);  
# * **2017** (**34** team members);
# * **2014** (**24**, **25** team members);   
# * **2013** (**24** team members);  
# 
# What happens in 2017 and 2018 is that sudden increases the number of teams (2017) and of medium-sized teams (4-8 team members).
# 
# When checking the number of competition per year we also notice that what happens in 2018 is that the number of competitions without limit of team size increased, as a percent from the total number of competitions. This will explain in part the pattern we observed, that we do have more and more teams (with large size) in 2018. Of course, these findings will have to be revisited after Meta Kaggle is updated with all 2018 data.  
# 
# Note: the results for 2019 are just partial.
# 
# Let's remove the **InClass** competitions and plot again the number of teams grouped by year and team size.
# 
# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# 
# ## <a id="34">Number of teams for team size and year heatmap (no InClass competitions)</a>
# 

# In[ ]:


no_inclass_df = comp_team_membership_df[comp_team_membership_df['HostSegmentTitle']!='InClass']
tmp = no_inclass_df.groupby(['DeadlineYear','TeamId'])['Id'].count()
df = pd.DataFrame(data={'Teams': tmp.values}, index=tmp.index).reset_index()
tmp = df.groupby(['DeadlineYear','Teams']).count()
df2 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()
df2.columns = ['Year', 'Team size','Teams']
plot_heatmap_count(df2,'Team size','Year', 'Blues', "Number of teams grouped by year and by team size (no InClass comp.)")


# As we expected, by removing **InClass** competitions we obtained very similar result as for all competitions, since majority of the teams are formed for **Featured** competitions.
# 
# <a href="#0"><font size="1">Go to top</font></a>  
# 
# ## <a id="35">Time variation of number of teams vs. team size (with plotly and blobby)</a>
# 
# Let's represent now on a single graph, using **blobby** <a href='#4'>[2]</a><a href='#4'>[3]</a> (bubble plot using **plotly** <a href='#4'>[4]</a>) the time variation (yearly) of **Teams** (number of teams) vs. **Team size**. For each **Host Segment Title**  - i.e. type of competition and **Team size** there is a separate bubble displayed. The bubble size is proportional with the team size (on sqrt scale).  The number of teams scale is logarithmic. The plot is animated, with one plot frame for each **year**.

# In[ ]:


tmp = comp_team_membership_df.groupby(['DeadlineYear','TeamId', 'HostSegmentTitle'])['Id'].count()
df3 = pd.DataFrame(data={'Teams': tmp.values}, index=tmp.index).reset_index()
tmp = df3.groupby(['DeadlineYear','HostSegmentTitle','Teams']).count()
df4 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()
df4.columns = ['Year', 'Host Segment Title','Team size','Teams']
df4['TeamsSqrt'] = np.sqrt(df4['Teams'] + 2)


# In[ ]:


figure = bubbleplot(dataset=df4, x_column='Team size', y_column='Teams', color_column = 'Team size',
    bubble_column = 'Host Segment Title', time_column='Year', size_column = 'TeamsSqrt',
    x_title='Team size', y_title='Number of Teams [log scale]', 
    title='Number of Teams vs. Team size - time variation (years)', 
    colorscale='Rainbow', colorbar_title='Team size', 
    x_range=[-5,41], y_range=[-0.4,7], y_logscale=True, scale_bubble=5, height=650)
iplot(figure, config={'scrollzoom': True})


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# ## <a id="36">Time variation of number of winning teams vs. team size (with plotly and blobby)</a>
# 
# ### <a id="361">Featured competitions</a>
# 
# Let's focus now on the winning teams (teams with bronze, silver or gold medals). We will only select **Featured** competitions.

# In[ ]:


feature_df = comp_team_membership_df[comp_team_membership_df['HostSegmentTitle']=='Featured']


# We represent the wining teams grouped by medal (**Gold**, **Silver** and **Bronze**) and team size.   
# In the graph, on one axis we have the team size (x-axis) and on the other axis we have the number of teams (y-axis).

# In[ ]:


tmp = feature_df.groupby(['DeadlineYear','TeamId', 'Medal'])['Id'].count()
df3 = pd.DataFrame(data={'Teams': tmp.values}, index=tmp.index).reset_index()
tmp = df3.groupby(['DeadlineYear','Medal','Teams']).count()
df4 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()
df4.columns = ['Year', 'Medal','Team size','Teams']
df4['Rank'] = (df4['Medal'] - 1) / 2
df4['Size'] = 4 - df4['Medal']


# In[ ]:


#create the bins for gold, silver and bronze
bins = [-0.01, 0.49, 0.99, np.inf]
names = ['Gold', 'Silver', 'Bronze']
df4['MedalName'] = pd.cut(df4['Rank'], bins, labels=names)


# In[ ]:


figure = bubbleplot(dataset=df4, x_column='Team size', y_column='Teams', color_column = 'Rank',
    bubble_column = 'MedalName', time_column='Year', size_column = 'Size', 
    x_title='Team size', y_title='Number of Teams [log scale]', 
    colorscale = [[0, "gold"], [0.5, "silver"], [1,"brown"]],
    title='Number of Winning Teams vs. Team size - time variation (years)', 
    x_range=[-5,41], y_range=[-0.4,4], y_logscale=True, scale_bubble=0.2, height=650)
iplot(figure, config={'scrollzoom': True})


# We can observe that the largest teams winning a medal were in each year:
# * 2010: 1 team winning **gold**, with **4** members;  
# * 2011: 1 team winning **gold**, with **12** members;  
# * 2012: 1 team winning **bronze**, with **40** members;  
# * 2013: 2 teams winning **gold**, with **24** members;  
# * 2014: 1 team winning **bronze**, with **6** members;   
# * 2015: 1 team winning **bronze**, with **18** members;  
# * 2016: 1 team winning **gold**, with **13** members;  
# * 2017: 1 team winning **bronze**, with **34** members;  
# * 2018: 1 team winning **silver**, with **23** members;  
# * 2019: 2 teams winning **gold** with **8** members and 4 teams winning **silver** with **8** members; please note that for 2019 the results are still partial.
# 
# 
# 

# In[ ]:


df5 = df4[df4['Medal']==1.0]
plot_heatmap_count(df5,'Team size','Year', 'Greens', "Number of Gold winning teams grouped by year and by team size (Featured competitions)")


# We can observe several things:
# * In **2018** the number of **gold** winning teams increased only for the teams with **2**, **5** and **7** members;  
# * The largest teams winning gold were in **2013** (**24**, **10** members), **2012** (**23**, **15**, **12** members), **2011** (**12** members), **2016** (**13** and **11** members) and **2017** (**10** members);  
# **Note**: the results for 2019 are partial.
# 
# Therefore the perception that only recently the number of large teams in Featured competition increased is false. Actually the largest team winning a medal was in **2012** (**40** team members) and in **2013** were **two** teams with **24** members winning **gold**!
# 

# ### <a id="362">Research competitions</a>
# 
# Let's focus now on the winning teams (teams with bronze, silver or gold medals). We will only select **Research** competitions.

# In[ ]:


research_df = comp_team_membership_df[comp_team_membership_df['HostSegmentTitle']=='Research']


# In[ ]:


tmp = research_df.groupby(['DeadlineYear','TeamId', 'Medal'])['Id'].count()
df3 = pd.DataFrame(data={'Teams': tmp.values}, index=tmp.index).reset_index()
tmp = df3.groupby(['DeadlineYear','Medal','Teams']).count()
df4 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()
df4.columns = ['Year', 'Medal','Team size','Teams']
df4['Rank'] = (df4['Medal'] - 1) / 2
df4['Size'] = 4 - df4['Medal']
bins = [-0.01, 0.49, 0.99, np.inf]
names = ['Gold', 'Silver', 'Bronze']
df4['MedalName'] = pd.cut(df4['Rank'], bins, labels=names)


# In[ ]:


figure = bubbleplot(dataset=df4, x_column='Team size', y_column='Teams', color_column = 'Rank',
    bubble_column = 'MedalName', time_column='Year', size_column = 'Size', 
    x_title='Team size', y_title='Number of Teams [log scale]', 
    colorscale = [[0, "gold"], [0.5, "silver"], [1,"brown"]],
    title='Number of Winning Teams vs. Team size - time variation (years) - Research competitions', 
    x_range=[-5,25], y_range=[-0.3,3], y_logscale=True, scale_bubble=0.2, height=650)
iplot(figure, config={'scrollzoom': True})


# We can observe that the largest teams winning a medal in Research competitions were in each year:
# 
# * 2012: 1 team winning **gold** and one **silver**, with **11** members;  
# * 2013: 1 team winning **bronze**, with **9** members;  
# * 2014: 1 team winning **bronze**, with **24** members;   
# * 2015: 1 team winning **silver**, with **8** members;  
# * 2016: 1 team winning **silver**, with **8** members;  
# * 2017: 4 teams winning **bronze**, with **8** members;  
# * 2018: 1 team winning **gold**, with **9** members;    
# 
# **Note**: for 2019 we have only partial results.
# 
# **Conclusion** is that the large teams winning bronze, silver or gold medals in Research competitions is not something recent. Already in 2012 were teams with 11 members winning gold and silver. In 2017 were 4 teams winning bronze.
# 

# In[ ]:


df5 = df4[df4['Medal']==1.0]
plot_heatmap_count(df5,'Team size','Year', 'Reds', "Number of Gold winning teams grouped by year and by team size (Research competitions)")


# We notice that the number of teams winning golds with multiple teams members (as well as with one team member) increased in 2019 compared with previous years (2015 to 2018), although 2019 has still partial results. 

# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# ## <a id="37">Teams size and teams rankings</a>  
# 
# 
# We select the teams for **Featured** competitions and we will check if the teams size is correlated in any way with the teams rankings.   
# 
# 
# For this, we count the number of teams members for each team and then we merge back the result with the **teams_df** dataset, to have in one dataset the number of team members per team and the public leaderboard rank.
# 

# In[ ]:


tmp = feature_df.groupby(['TeamId'])['Id'].count()
df = pd.DataFrame(data={'Team Size': tmp.values}, index=tmp.index).reset_index()
#merge back df with teams_df
df2 = df.merge(teams_df, left_on='TeamId', right_on='Id', how='inner')
var = ['Team Size', 'PublicLeaderboardRank', 'PrivateLeaderboardRank']
teams_ranks_df = df2[var]


# In[ ]:


corr = teams_ranks_df.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,
            cmap="YlGnBu",linewidths=.1,annot=True,vmin=-1, vmax=1)
plt.show()


# We can observe that while there is an obvious strong correlation between the public and private leaderboard rank, there is no correlation (values under 0.1 and negative) between the team size and the public or private leaderboard rank. The negative very small correlation coeficient prevent us to draw any conclusion about existence of an (inverse) correlation.   
# 
# Let's check if this factor changes significantly over years.   
# 
# <a href="#0"><font size="1">Go to top</font></a>  
# 
# ## <a id="38">Teams size and teams rankings per year</a>  
# 
# The results shown here are only for **Featured** competitions.

# In[ ]:


df2["Year"] = pd.to_datetime(df2['LastSubmissionDate']).dt.year
var = ['Team Size', 'PublicLeaderboardRank', 'PrivateLeaderboardRank' ]
years = df2['Year'].unique()
years = np.sort(years[~np.isnan(years)])


# In[ ]:


f, ax = plt.subplots(5,2, figsize=(12,22))
for i, year in enumerate(years):
    teams_ranks_df = df2[df2['Year']==year]
    corr = teams_ranks_df[var].corr()
    labels = ['Size', 'Public', 'Private']
    axi = ax[i//2, i%2]
    s1 = sns.heatmap(corr,xticklabels=labels,yticklabels=labels,
                     cmap="YlGnBu",linewidths=.1,annot=True,vmin=-1, vmax=1,ax=axi)
    s1.set_title("Year: {}".format(int(year)))
plt.show()


# Although a very small negative value (in the range of **no inverse correlation**), we can observe the following for the  value of correlation between Private Leaderboard Rank & Public Leaderboard Rank with the Team Size:  
# 
# *  It is a inverse very small correlation factor (i.e. teams size increases with the lower value of rank or, the closer to the top, the larger the teams);  
# * Values are between -0.02 and -0.12 (very small inverse correlation values) and increased (in absolute values) over time;
# * When different, in general inverse correlations for Public are larger i.e. teams tend to be larger for higher positions on the public leaderboard;   
# 
# 

# <a href="#0"><font size="1">Go to top</font></a>  
# 
# # <a id="4">Conclusions</a>    
# 
# Analyzing the competitions and teams data we understood that large teams winning medals were equaly frequent in past years, with very large size teams winning competitions as early as 2012.
# 
# A team of 40 members won a bronze medal in 2012 and a team with 13 members won gold in 2016, both in Featured competitions. Comparatively, one team won silver in 2018 with 23 members.
# 
# We can observe that while there is an obvious strong correlation between the public and private leaderboard rank, there is no correlation (values under 0.1 and negative) between the team size and the public or private leaderboard rank.   
# 
# Final conclusion will be that, although there was a recent perception of increase in frequency of large size teams formed to win medals, this is not a recent phenomena; there were larger teams in the past winning medals. What changed dramatically recently is actually the number of Kagglers.  See in the following this dynamic.
# 
# ## <a id="41">Final remarks - and a possible explanation of the perception</a>
# 
# Let's check the user dynamics. We will show the number of new users registered every year.
# 
# 

# In[ ]:


users_df = pd.read_csv(os.path.join(PATH,"Users.csv"))


# In[ ]:


users_df.head()


# In[ ]:


users_df['RegisterYear'] = pd.to_datetime(users_df['RegisterDate'], format='%m/%d/%Y').dt.year


# In[ ]:


tmp = users_df['RegisterYear'].value_counts()
df = pd.DataFrame(data={'Users': tmp.values}, index=tmp.index).reset_index()
trace = go.Bar(
    x = df['index'],y = df['Users'],
    name='Users',
    marker=dict(color="red")
)
data = [trace]
    
layout = dict(title = 'Total number of new users per year',
          xaxis = dict(title = 'Year', showticklabels=True), 
          yaxis = dict(title = 'Number of new users'),
          hovermode = 'closest',
         width = 600, height = 600
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='competition-types')


# This dynamics shows that **70%**  of the current users of Kaggle were not actually around two years ago (this includes me, actually).   
# 
# The increase in the new users number is really exponential. Therefore what is perceived as the community memory might be a biased image, since for the majority of the Kagglers were not around to experience large teams just few years ago.  
# 
# 
# <a href="#0"><font size="1">Go to top</font></a>  

# # <a id="5">References</a>  
# 
# [1] Meta Kaggle, https://www.kaggle.com/kaggle/meta-kaggle  
# [2] <a href="https://www.kaggle.com/aashita">Aashita Kesarwani</a>, https://www.kaggle.com/aashita/guide-to-animated-bubble-charts-using-plotly  
# [3] <a href="https://www.kaggle.com/aashita">Aashita Kesarwani</a>,  https://github.com/AashitaK/bubbly/blob/master/bubbly/bubbly.py  
# [4] Plotly, https://community.plot.ly/   
# [5] Plotly tutorial, https://www.kaggle.com/gpreda/plotly-tutorial-120-years-of-olympic-games  
# [6] Data Science Glossary on Kaggle, https://www.kaggle.com/shivamb/data-science-glossary-on-kaggle  
# [7] Winning Solutions of Kaggle Competitions,  https://www.kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions   
# 
# 
# <a href="#0"><font size="1">Go to top</font></a>  
