#!/usr/bin/env python
# coding: utf-8

# # IPL Data 2008-2017 (GUI Select Match Report Analysis)

# In this kernal you will be able to see the complete batting report for selected season and match between 2008 -2017
# 
# Packages i have used -
# * Pandas
# * Numpy
# * Seaborn
# * Matplotlib
# * ipywidgets
# * plotly
# 
# 
# 

# This will allow you to select one specific season and match and show the complete match analysis 
# * Man of the match
# * winner
# * batsman run with strike rate
# * contribution of each batsman in an innings (pie chart)
# * Per over runs with bowler name (bar chart)
# * Line chart (Between two teams per over )

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sms
#from plotly import __version__
#import cufflinks as cf
#from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
#init_notebook_mode(connected=True)
#cf.go_offline()


# In[ ]:


import plotly.express as px
from ipywidgets import Dropdown
import asyncio


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import ipywidgets as widgets
import plotly.graph_objects as go


# In[ ]:


deliveries = pd.read_csv("../input/ipl/deliveries.csv")
matches = pd.read_csv("../input/ipl/matches.csv")


# In[ ]:


matches.head(2)


# In[ ]:


clean_del = deliveries.replace(['Kolkata Knight Riders', 'Royal Challengers Bangalore',
       'Chennai Super Kings', 'Kings XI Punjab', 'Rajasthan Royals',
       'Delhi Daredevils', 'Mumbai Indians', 'Deccan Chargers',
       'Kochi Tuskers Kerala', 'Pune Warriors', 'Sunrisers Hyderabad',
       'Rising Pune Supergiant', 'Rising Pune Supergiants', 'Gujarat Lions'],['kkr','rcb','csk','kxip','rr','dd','mi','dc','ktk','pw','sh','rps','rps','gl'])


# In[ ]:


clean_mat = matches.replace(['Kolkata Knight Riders', 'Royal Challengers Bangalore',
       'Chennai Super Kings', 'Kings XI Punjab', 'Rajasthan Royals',
       'Delhi Daredevils', 'Mumbai Indians', 'Deccan Chargers',
       'Kochi Tuskers Kerala', 'Pune Warriors', 'Sunrisers Hyderabad',
       'Rising Pune Supergiant', 'Rising Pune Supergiants', 'Gujarat Lions'],['kkr','rcb','csk','kxip','rr','dd','mi','dc','ktk','pw','sh','rps','rps','gl'])


# In[ ]:


clean_del.head(2)


# In[ ]:


clean_mat.head(2)


# In[ ]:


clean_mat['season'].unique()


# In[ ]:


clean_mat = clean_mat.drop('umpire3', axis=1)
clean_del = clean_del.fillna(value='0')


# # Automated Ipl Analysis

# From below code you can select any season from 2008-2017.
# You can copy the code and run it in your jupyter notebook.
# I need to comment the GUI part so that it could run successfully once committed.
# You can try out copying the GUI part and try it in your jupyter notebook.
# 

# In[ ]:


#def wait_for_change(widget,w):
    #future = asyncio.Future()
    #def on_change(change):
        #if change['type'] == 'change' and change['name'] == 'value':
            #print ("changed to %s" % change['new'])
            #global season1
            #season1 = change['new']
    #w.observe(on_change)
    #return future
#my_list = list(clean_mat['season'].unique())
#my_list.sort()
#my_list

#w = widgets.Dropdown(
    #options=my_list,
    #value=my_list[0],
    #description='Select Season:',
#)
#display(w)
#async def f():
    #print("abhi")
    #x = await wait_for_change(Dropdown,w)
    #print(x)


#asyncio.ensure_future(f())
#Dropdown


# I will manually choose 2017 season and sh -vs- rcb <match no- 1>

# In[ ]:


season1 = 2017


# In[ ]:


season=clean_mat[clean_mat['season']==season1]
new_list=list(season['id'].unique())
#new_list
list_tm = ["select below"]
match = 1
for i,j in (zip(season['team1'],season['team2'])):
    #print(i)
    #print(j)
    Str = i + str(' -vs- ') + j +str(' <match no- ')+str(match)+str('>')
    list_tm.append(Str)
    #print(Str)
    match +=1


# Below code dynamically selects all the matchs played in that season.

# In[ ]:


#def wait_for_change(widget,s):
    #future = asyncio.Future()
    #def on_change(change):
        #if change['type'] == 'change' and change['name'] == 'value':
            #print ("changed to %s" % change['new'])
            #global index_value
            #index_value = change['new']
    #s.observe(on_change)
    #return future
#my_list = list(clean_mat['season'].unique())
#my_list.sort()
#my_list

#s = widgets.Dropdown(
    #options=list_tm,
    #value=list_tm[0],
    #description='Select no:',
#)
#display(s)
#async def g():
    #y = await wait_for_change(Dropdown,s)
    #print(y)

#asyncio.ensure_future(g())
#Dropdown


# In[ ]:


index_value = 'sh -vs- rcb <match no- 1>'
#list_tm


# In[ ]:


n = list_tm.index(index_value)
n-=1
match_id = new_list[n]
match1 = clean_del[clean_del['match_id']==match_id]


# In[ ]:


match1.head()


# In[ ]:


team_list = []
team_list = match1['batting_team'].unique()
team1 = team_list[0]
team2 = team_list[1]


# In[ ]:


match_details = clean_mat[clean_mat['id']==match_id]


# complete match details

# In[ ]:


match_details


# In[ ]:


team_1 = match1[match1['batting_team']==team1]
team_2 = match1[match1['batting_team']==team2]


# In[ ]:


total_run_team1 = team_1['total_runs'].sum()
total_run_str1 = str(total_run_team1)
total_run_team2 = team_2['total_runs'].sum()
total_run_str2 = str(total_run_team2)


# In[ ]:


team_1_bats_man=team_1[['batsman','batsman_runs']]
team_1_list = team_1_bats_man['batsman'].unique()
team_2_bats_man=team_2[['batsman','batsman_runs']]
team_2_list = team_2_bats_man['batsman'].unique()


# Both team batsman runs count function 

# In[ ]:


list_name = []
list_runs = []
list_balls = []


# In[ ]:


for x in team_1_list:
    runs = team_1_bats_man.loc[team_1_bats_man['batsman']==x , ['batsman','batsman_runs']]
    total_run = runs['batsman_runs'].sum()
    balls = runs['batsman_runs'].count()
    bt_runs = runs['batsman_runs'].value_counts()
    list_name.append(x)
    list_runs.append(total_run)
    list_balls.append(balls)
    #dist_team1.update({x:total_run})
#dist_team1  


# In[ ]:


list_name1 = []
list_runs1 = []
list_balls1 = []


# In[ ]:


for y in team_2_list:
    runs1 = team_2_bats_man.loc[team_2_bats_man['batsman']==y , ['batsman','batsman_runs']]
    total_run1 = runs1['batsman_runs'].sum()
    balls1 = runs1['batsman_runs'].count()
    bt_runs = runs1['batsman_runs'].value_counts()
    list_name1.append(y)
    list_runs1.append(total_run1)
    list_balls1.append(balls1)


# In[ ]:


demo_df1 = pd.DataFrame(columns=['Dot','One','Two','Three','Four','Six'])
demo_df2 = pd.DataFrame(columns=['Dot','One','Two','Three','Four','Six'])


# Boundary and runs scored by each batsman function

# In[ ]:


def runs_count(team_list,team_bats,team_df):   
    count= 0
    for x in team_list:
            count+=1
    #print("count",count)
    for i, y in zip(team_list, range(count)):
        runs1 = team_bats.loc[team_bats['batsman']==i , ['batsman','batsman_runs']]
        #print(runs1)
        balls=runs1['batsman_runs'].value_counts()
        #print(balls)
        abhi = list(balls.index.values)
        if 0 in abhi:
            team_df.at[y,'Dot'] = balls[0]
            #print("dot", balls[0])
        else:
            team_df.at[y,'Dot'] = "0"
        if 1 in abhi:
            team_df.at[y,'One'] = balls[1]
        else:
            team_df.at[y,'One'] = "0"
        if 2 in abhi:
            team_df.at[y,'Two'] = balls[2]
        else:
            team_df.at[y,'Two'] = "0"
        if 3 in abhi:
            team_df.at[y,'Three'] = balls[3]
        else:
            team_df.at[y,'Three'] = "0"
        if 4 in abhi:
            team_df.at[y,'Four'] = balls[4]
        else:
            team_df.at[y,'Four'] = "0" 
        if 6 in abhi:
            team_df.at[y,'Six'] = balls[6]
        else:
            team_df.at[y,'Six'] = "0"
    return team_df
    


# In[ ]:


team_one_counts = runs_count(team_1_list,team_1_bats_man,demo_df1)


# In[ ]:


team_two_counts = runs_count(team_2_list,team_2_bats_man,demo_df2)


# In[ ]:


basic_df1 = pd.DataFrame({'name':list_name,
                         'runs':list_runs,
                         'balls':list_balls})
basic_df2 = pd.DataFrame({'name':list_name1,
                         'runs':list_runs1,
                         'balls':list_balls1})


# Strike rate count

# In[ ]:


basic_df1['strike_rate'] = basic_df1['runs']*100/basic_df1['balls']
team_first_df1 = pd.concat([basic_df1,team_one_counts],axis=1)
basic_df2['strike_rate'] = basic_df2['runs']*100/basic_df2['balls']
team_second_df2 = pd.concat([basic_df2,team_two_counts],axis=1)


# Team 1 batting stats

# In[ ]:


team_first_df1


# In[ ]:


team_second_df2


# # Bar plot for both the teams

# 1st team batting analysis --- 
# Total runs scored by each player
# 

# In[ ]:


sms.set(style='darkgrid')
fig=plt.gcf()
fig.set_size_inches(18.5,10.5)
plt.xticks(fontsize=0)
plt.yticks(rotation=90,fontsize=15)
X = np.arange(team_first_df1['name'].count())
plt.bar(X+0.00,team_first_df1['runs'],color=['#CD202D','#EF2920','#D4480B','#7698f5','#FFFF48','#EF2920',
               '#FFFF48','#FFFF48','#292734','#FFFF48','#ECC5F2','#EF2920',
               '#292734','#15244C','#005DB7','#005DB7','#292734','#15244C',
               '#FFFF48','#CD202D'], width = 0.25)

plt.bar(X+0.25,team_first_df1['balls'], color = 'b', width = 0.25)


count=0
for i in team_first_df1['runs']:
    plt.text(count,18,team_first_df1['name'][count]+'  :  '+str(i),rotation=90,color='black',fontsize=15)
    count+=1
count1=0
for p in team_first_df1['balls']:
    plt.text(count1+0.25,18,'Balls played  - ' +str(p),rotation=90,color='black',fontsize=15)
    count1+=1
count2=0
for j in team_first_df1['strike_rate']:
    plt.text(count2+0.50,18,'Strike Rate  - ' "%.2f "% round(j,2),rotation=90,color='r',fontsize=15)
    count2+=1
    #plt.text(count,10,team_first_df['strike_rate'][count],color='black',fontsize=15)
plt.title('Total runs scored by '+team1+' team batting first '+total_run_str1+'.',fontsize=16)
plt.xlabel(team1+ ' players',fontsize=14)
plt.ylabel('Total runs',rotation=90,fontsize=14)
plt.show()


# 2nd team battin analysis --- total runs scored 

# In[ ]:


sms.set(style='darkgrid')
fig=plt.gcf()
fig.set_size_inches(18.5,10.5)
plt.xticks(fontsize=0)
plt.yticks(rotation=90,fontsize=15)
Y = np.arange(team_second_df2['name'].count())
plt.bar(Y+0.00,team_second_df2['runs'],color=['#CD202D','#EF2920','#D4480B','#7698f5','#FFFF48','#EF2920',
               '#FFFF48','#FFFF48','#292734','#FFFF48','#ECC5F2','#EF2920',
               '#292734','#15244C','#005DB7','#005DB7','#292734','#15244C',
               '#FFFF48','#CD202D'],width = 0.25)
plt.bar(Y+0.25,team_second_df2['balls'], color = 'b', width = 0.25)

count=0
for i in team_second_df2['runs']:
    plt.text(count,10,team_second_df2['name'][count]+'  :  '+str(i),rotation=90,color='black',fontsize=15)
    count+=1
count1=0
for p in team_second_df2['balls']:
    plt.text(count1+0.25,10,'Balls played  - ' +str(p),rotation=90,color='black',fontsize=15)
    count1+=1
count2=0
for j in team_second_df2['strike_rate']:
    plt.text(count2+0.50,10,'Strike Rate  - ' "%.2f "% round(j,2),rotation=90,color='r',fontsize=15)
    count2+=1
plt.title('Total runs scored by '+team2+' team batting secound '+total_run_str2+'.',fontsize=16)
plt.xlabel(team2+ ' players',fontsize=14)
plt.ylabel('Total runs',rotation=90,fontsize=14)
plt.show()


# # Match Summary Complete

# In[ ]:


n = match_details['city'].index.values[0]
print('Match between - {0} vs {1} '.format(team1,team2))
print('City - {0} & Venue - {1}'.format(match_details['city'][n],match_details['venue'][n]))
print('Date - {0}'.format(match_details['date'][n]))
print('Toss winner - {0}'.format(match_details['toss_winner'][n]))
print('Winner of the match - {0}'.format(match_details['winner'][n]))
print('Man of the match - {0}'.format(match_details['player_of_match'][n]))


# Each player contribution to total score in percentage (pie chart)

# In[ ]:


fig = go.Figure(data=[go.Pie(labels=team_first_df1['name'], values=team_first_df1['runs'])])
fig.show()


# In[ ]:


fig = go.Figure(data=[go.Pie(labels=team_second_df2['name'], values=team_second_df2['runs'])])
fig.show()


# In[ ]:


team_1['bowling_team'].unique()[0]


# Each over total runs with bowler name count function

# In[ ]:


def match_overview(df_bowling,team_1,team_2):
    for i in np.arange(1,21):
        p=i
        p-=1
        bowling_team_name = team_1['bowling_team'].unique()[0]
        over_stat = team_1[team_1['over']==i]
        total_runs_per_over=over_stat['total_runs'].sum()
        
        over_stat1 = team_2[team_2['over']==i]
        total_runs_per_over1=over_stat1['total_runs'].sum()
        
        bowler_name_per_over=over_stat['bowler'].unique()[0]
        
        df_bowling.at[p,'bowling_team1']=bowling_team_name
        df_bowling.at[p,'bowler_team1']=bowler_name_per_over
        df_bowling.at[p,'over_team1']=i
        df_bowling.at[p,'total_runs_team1']=total_runs_per_over
        df_bowling.at[p,'bowling_team2']=team_2['bowling_team'].unique()[0]
        df_bowling.at[p,'bowler_team2']=over_stat1['bowler'].unique()[0]      
        df_bowling.at[p,'over_team2']=i
        df_bowling.at[p,'total_runs_team2']=over_stat1['total_runs'].sum()
        
        #df_bowling.at[p,'player_dismissed']=out_player_name
    return df_bowling        
        


# In[ ]:


df_bowling = pd.DataFrame(columns=['bowling_team1','bowler_team1','over_team1','total_runs_team1','bowling_team2','bowler_team2','over_team2','total_runs_team2'])
df_bowling


# In[ ]:


df_bowling=match_overview(df_bowling,team_1,team_2)


# In[ ]:


df_bowling


# Bar plot for each over with bowler name and runs scored in that over for both the teams

# In[ ]:


data_team1 = df_bowling
fig = px.bar(data_team1, x='over_team1', y='total_runs_team1',
             hover_data=['over_team1', 'total_runs_team1','bowler_team1'], color='bowling_team1',
             labels={'bowler_team1':'Bowler of the over'}, height=400)
fig.show()


# In[ ]:


data_team2 = df_bowling
fig = px.bar(data_team2, x='over_team2', y='total_runs_team2',
             hover_data=['over_team2', 'total_runs_team2','bowler_team2'], color='bowling_team2',
             labels={'bowler_team2':'Bowler of the over'}, height=400)
fig.show()


# In[ ]:


array1=np.array(df_bowling['total_runs_team1'])
array2=np.array(df_bowling['total_runs_team2'])
temp = []
temp1 = []


# In[ ]:


sum = 0
for x,j in zip(array1,np.arange(20)):
    sum = sum + x
    temp.insert(j,sum)   


# In[ ]:


sum1 = 0
for y,z in zip(array2,np.arange(20)):
    sum1 = sum1 + y
    temp1.insert(z,sum1)
    


# In[ ]:


over_new=list(np.arange(1,21))
over_new1=list(np.arange(1,21))
team_new1 = [df_bowling['bowling_team2'][0]]*20 #mi
team_new2 = [df_bowling['bowling_team1'][0]]*20 #rps
team_new3 = team_new1 + team_new2
temp3 =temp + temp1
over_new3=over_new + over_new1


# In[ ]:


df_both_teams = pd.DataFrame({'team':team_new3,
                             'over':over_new3,
                             'runs':temp3})
#df_both_teams


# Complete run analysis per over for both the teams 

# In[ ]:


gapminder = px.data=df_both_teams
fig = px.line(gapminder, x="over", y='runs',color='team')
fig.show()


# In[ ]:




