#!/usr/bin/env python
# coding: utf-8

# 
# <font size="4.5">Title: Texas Tech football team performance analysis under coach Kingsbury</font>

# **Introduction:**
# 
# <font size="2">This is a project that I did as an example of "Story telling with data". I used Texas Tech Football Team performance data under coach Kingsbury. Sidenote: NFL MVP Patrick Mahomes aslo played under coach Kingsbury and we can see his crazy numbers (as a team).</font>
# 
# 

# 
# **Key data attributes: **
# * Rushing Yards
# * Rushing Attempts
# * Rushing Yards per Attempts(rpa)
# * Passing Yards
# * Passing Attempts
# * Passing Yards per Attempts(ppa)
# * Score

# In[ ]:


from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# Generating Key Words for WordCloud:
text = 'Texas Tech Offense Defense BIG12 Touchdowns Interception Yards Fumble Lubbock Data Analysis Elyas Pass Rush Punt Kick Kliff Bowman Duffy Wesley QB RB Tackle Return Wide Receiver RedZone FieldGoal Score GunsUp'


# In[ ]:


#mask used for the shape of wordcloud
from IPython.display import Image
Image("../input/texastech-football-data/t.jpeg")


# In[ ]:


from PIL import Image
img = np.array(Image.open("../input/texastech-football-data/t.jpeg"))
# Create and generate a word cloud image:
wordcloud = WordCloud(background_color='black',mask=img,colormap="cool").generate(text)

# Display the generated image:
fig = plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/texastech-football-data/2013_2018_data.csv')
## remove white spaces in cloumns index and Vis Team coloumn
data.columns = data.columns.str.strip()
data['Vis Team'] = data['Vis Team'].str.strip()
## ignore warnings
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# In[ ]:


data_record = { 'Year':[2013,2014,2015,2016,2017,2018],
                 'Win':[8,4,7,5,6,5],
                'Lose':[5,8,6,7,7,2]}
record_fr = pd.DataFrame(data_record)
record_fr['Win Percentage'] = record_fr['Win']/(record_fr['Win'] + record_fr['Lose'])


# In[ ]:


from matplotlib import style
#style.use('ggplot')
plt.style.use('fivethirtyeight')
fig1 = plt.figure(figsize=(10,9))
plt.bar(record_fr['Year'],record_fr['Win'],align='center',label='Win',color='red',width=0.5)
plt.bar(record_fr['Year'],-record_fr['Lose'],align='center',label='Lose',color='black',width=0.5)
plt.plot(record_fr['Year'],[6,6,6,6,6,6],label='Bowl Game Eligible',color='blue')
plt.legend()
plt.xlabel('Year');plt.ylabel('Number of Games');plt.title('Coach Kingsburys record')


# Coach made three bowl games with TTU since cutoff or team needs to have 6 wins to qualify for bowl game. Here I used negitave numbers for the number of games coach lost. As you can see, 2014 was his worst performance year!

# In[ ]:


from matplotlib import style
#style.use('ggplot')
plt.style.use('fivethirtyeight')
fig2 = plt.figure(figsize=(15,10))
plt.plot(record_fr['Year'],record_fr['Win Percentage'],color='blue',linewidth=2,marker='o')
plt.annotate('Bowl Game Win Vs ASU', xy=(2013, 0.62), xytext=(2013.2, 0.65), fontsize = 12,
            arrowprops=dict(facecolor='grey', shrink=0.05, linewidth = 2))
plt.annotate('Bowl Game Loss Vs LSU', xy=(2015, 0.54), xytext=(2015.2, 0.56), fontsize = 12,
            arrowprops=dict(facecolor='grey', shrink=0.05, linewidth = 2))
plt.annotate('Bowl Game Loss Vs USF', xy=(2017, 0.47), xytext=(2016, 0.5), fontsize = 12,
            arrowprops=dict(facecolor='grey', shrink=0.05, linewidth = 2))
plt.annotate('Only 7 Games Played', xy=(2018, 0.72), xytext=(2016.5, 0.66), fontsize = 12,
            arrowprops=dict(facecolor='grey', shrink=0.05, linewidth = 2))
plt.xlabel('Year');plt.ylabel('Win Percentage');plt.title('Coach Kingsbury win percentage')
plt.legend()
plt.show()


# Coach only won one bowl game which is against Arizona State University. Also note that I only had first 7 games of 2018 in my analysis, later games he lost them all (Thats why he hot fired at the end of the season).

# In[ ]:


avg_off = np.zeros((1,6), dtype=float)
avg_def = np.zeros((1,6), dtype=float)
Years = [[2013,2014,2015,2016,2017,2018]]
j = -1
Years 
plt.style.use('fivethirtyeight')
for year in range (2013,2018+1):
  # print(year)
   # print(data[data['Vis Team'] == 'Texas Tech'][data['Date']==year][data['Location']=='Away'].Score.mean())
    j = j + 1
    avg_off [0, j] = data[data['Vis Team'] == 'Texas Tech'][data['Date']==year].Score.mean()
    avg_def [0, j] = data[data['Vis Team'] != 'Texas Tech'][data['Date']==year].Score.mean()

avg_off = np.round(avg_off ,2)
avg_def = np.round(avg_def ,2)

fig5 = plt.figure(figsize=(15,10))
plt.plot(np.transpose(Years),np.transpose(avg_off),color='r',linewidth=0.5,marker='o',label='Offense')
plt.plot(np.transpose(Years),np.transpose(avg_def),color='b',linewidth=0.5,marker='^',label='Defense')
plt.xlabel('Year');plt.ylabel('Points');plt.title('Offense VS Defense')
plt.legend()
plt.show()


# TTU has a good offense, at the same time they had horrible defense.

# In[ ]:


avg_score_away = np.zeros((1,6), dtype=float)
avg_score_home = np.zeros((1,6), dtype=float)
Years = [[2013,2014,2015,2016,2017,2018]]
j = -1
Years 
for year in range (2013,2018+1):
  # print(year)
   # print(data[data['Vis Team'] == 'Texas Tech'][data['Date']==year][data['Location']=='Away'].Score.mean())
    j = j + 1
    avg_score_away [0, j] = data[data['Vis Team'] == 'Texas Tech'][data['Date']==year][data['Location']=='Away'].Score.mean()
    avg_score_home [0, j] = data[data['Vis Team'] == 'Texas Tech'][data['Date']==year][data['Location']=='Home'].Score.mean()

avg_score_away = np.round(avg_score_away ,2)
avg_score_home = np.round(avg_score_home ,2)

fig3 = plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(np.transpose(Years),np.transpose(avg_score_home),color='r',linewidth=2,marker='o',label='Home')
plt.plot(np.transpose(Years),np.transpose(avg_score_away),color='k',linewidth=2,marker='^',label='Away')
plt.annotate('QB:Pat Mahomes', xy=(2015, 60), xytext=(2014.5, 45), fontsize = 12,
            arrowprops=dict(facecolor='grey', shrink=0.05, linewidth = 2))
plt.annotate('QB:Pat Mahomes', xy=(2016, 49), xytext=(2014.5, 45), fontsize = 12,
            arrowprops=dict(facecolor='grey', shrink=0.05, linewidth = 2))
plt.xlabel('Year');plt.ylabel('Points');plt.title('Scoring Offense')
plt.legend()
#########
avg_defense_away = np.zeros((1,6), dtype=float)
avg_defense_home = np.zeros((1,6), dtype=float)
Years = [[2013,2014,2015,2016,2017,2018]]
j = -1
Years 
for year in range (2013,2018+1):
  # print(year)
   # print(data[data['Vis Team'] == 'Texas Tech'][data['Date']==year][data['Location']=='Away'].Score.mean())
    j = j + 1
    avg_defense_away [0, j] = data[data['Vis Team'] != 'Texas Tech'][data['Date']==year][data['Location']=='Away'].Score.mean()
    avg_defense_home [0, j] = data[data['Vis Team'] != 'Texas Tech'][data['Date']==year][data['Location']=='Home'].Score.mean()

avg_defense_away = np.round(avg_defense_away ,2)
avg_defense_home = np.round(avg_defense_home ,2)

plt.subplot(1,2,2)
plt.plot(np.transpose(Years),np.transpose(avg_defense_home),color='r',linewidth=2,marker='o',label='Home')
plt.plot(np.transpose(Years),np.transpose(avg_defense_away),color='k',linewidth=2,marker='^',label='Away')
plt.xlabel('Year');plt.ylabel('Points');plt.title('Scoring Defense')
plt.legend()
plt.show()


# TTU offense performed better at home than away, the highest two year offense are 2015&2016 when Patrick Mahomes played was QB. Now, he is the NFL MVP!!!

# In[ ]:


data1 = data[data['Vis Team']=='Texas Tech']
import seaborn as sns
plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
sns.boxplot(x=data1['Date'],y=data1['Passing Yards'],data=data1)
sns.swarmplot(x=data1['Date'],y=data1['Passing Yards'],data=data1,color=".25")
plt.xlabel('Year')
plt.subplot(1,2,2)
sns.boxplot(x=data1['Date'],y=data1['Rushing Yards'],data=data1)
sns.swarmplot(x=data1['Date'],y=data1['Rushing Yards'],data=data1,color=".25")
plt.xlabel('Year')
plt.show()


# TTU was heavy on passing game than rushing, Air raid offense! 

# In[ ]:


data1['rpa']=data1['Rushing Yards']/data1['Rushing Attempts']
data1['ppa']=data1['Passing Yards']/data1['Passing Attempts']
data1['tco']=data1['3rd Down Conversions']/data1['3rd Down Attempts']
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
sns.scatterplot(x=data1['Rushing Attempts'], y=data1['Date'],size=data1['rpa'],sizes=(50, 200),hue=data1['rpa'],palette='Blues',legend='brief',data=data1)
plt.ylabel('Year')
plt.subplot(2,1,2)
sns.scatterplot(x=data1['Passing Attempts'], y=data1['Date'],size=data1['ppa'],sizes=(50, 200),hue=data1['ppa'],palette='Greens',legend='brief',data=data1)
plt.ylabel('Year')
plt.show()


# This graph shows how efficient TTU's passing and running game. Again more passing plays also gained more yards per play!

# In[ ]:


import plotly
import plotly as py
import plotly.graph_objs as go
py.offline.init_notebook_mode(connected=True)


# In[ ]:


penalties = np.zeros((6,12))
tdcomps = np.zeros((6,12))
Years = [[2013,2014,2015,2016,2017,2018]]
j = -1
Years 
for year in range (2013,2018+1):
  # print(year)
   # print(data[data['Vis Team'] == 'Texas Tech'][data['Date']==year][data['Location']=='Away'].Score.mean())
    j = j + 1
    #print(np.array(data1[data1['Location'] != 'Bowl'][data1['Date']==year]['tco']))
    if year != 2018:
        penalties [j,:] = np.array(data1[data1['Location'] != 'Bowl'][data1['Date']==year]['Penalty Yards'])
        tdcomps [j,:] = np.array(data1[data1['Location'] != 'Bowl'][data1['Date']==year]['tco'])
    else:
        penalties [j,0:7] = np.array(data1[data1['Location'] != 'Bowl'][data1['Date']==year]['Penalty Yards'])
        tdcomps [j,0:7] = np.array(data1[data1['Location'] != 'Bowl'][data1['Date']==year]['tco'])

penalties = np.round(penalties ,2)
tdcomps = 100*np.round(tdcomps ,2)


# In[ ]:


penalties[penalties==0] = np.nan
tdcomps[tdcomps==0] = np.nan

layout = go.Layout(
    title='Penalties Heatmap',
    xaxis = dict(title='Game'),
    yaxis = dict(title='Year')
)
    
trace = go.Heatmap(z=penalties,
                   x=[1,2,3,4,5,6,7,8,9,10,11,12],
                   y=[2013,2014,2015,2016,2017,2018])

fig = go.Figure(data=[trace], layout=layout)
py.offline.iplot(fig)


# TTU lost lots of yards due to penalty. 

# In[ ]:


des = data1.describe()
pdat =des[['Rushing Yards','Rushing Attempts','rpa','Passing Yards','Passing Attempts','ppa','Penalty Yards','Score']].loc[['mean','std','min','25%','50%','75%','max']]


# In[ ]:


plt.style.use('ggplot')
plt.figure(figsize=(13,6))
sns.heatmap(pdat,annot=True,fmt="f",linecolor="white",linewidths=2)
plt.title("Data Summary")
plt.show()


# In[ ]:


layout = go.Layout(
    title='3rd down completions',
    xaxis = dict(title='Game'),
    yaxis = dict(title='Year')
)
    
trace = go.Surface(z=tdcomps,
                   contours=go.surface.Contours(
            z=go.surface.contours.Z(
              show=True,
              usecolormap=True,
              highlightcolor="#42f462",
              project=dict(z=True)
                                )
                                                ),
        x=[1,2,3,4,5,6,7,8,9,10,11,12],
        y=[2013,2014,2015,2016,2017,2018])

fig = go.Figure(data=[trace], layout=layout)
py.offline.iplot(fig)


# I like this dynamic graph, here you can rotate it and see which games TTU had good 3rd conversions.

# In[ ]:


from IPython.display import Image
Image("../input/otherfigs/excelfig.png")


# This figure is made from Excel. TTU offense was above BIG12 average most of the years, but their deffense was also below average (here large numbers means number of points gave up)

# In[ ]:


Image("../input/otherfigs/jmpfig.png")


# JMP figure example. TTU never lost to University of Kansas, but never won against University of Oklahoma
# under coach Kinsbury...

# 
# <font size="5">Conclusions:</font>
# <font size="4">TTU Football Team Performance under coach Kingsbury</font>
# * Offense was great!
# * Defense was horrible, but show improvement after 2015.
# * Penalty was a big problem.
# * Need to balance passing and running game.
# * Poor bowl game performance.
# * Best record against KU, worst against OU.
# 
# 
# Data Source:
# 
# Each teams and players stats, updates frequently!   
# http://sports.snoozle.net/search/fbs/index.jsp
# 
# BIG 12 conference game statistics 
# http://www.big12sports.com/ViewArticle.dbml?DB_OEM_ID=10410&ATCLID=1514232
# 
# Team VS Team Stats
# http://www.winsipedia.com/
