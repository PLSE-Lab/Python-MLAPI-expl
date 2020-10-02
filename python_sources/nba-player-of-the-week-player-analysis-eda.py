#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
Importing all the necessary libraries
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


'''
Load the NBA Player of the week file from Kaggle
'''
df=pd.read_csv('../input/NBA_player_of_the_week.csv')


# ## Check for the info of the dataset (head,describe, and info)

# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


'''
decribe() function is use to visualize the numeric value of the dataset
'''
df.describe()


# In[ ]:


'''
As you can see Height and Weight data wasn't clean
The units are'nt the same.
So we clean the data.
'''
df[['Height','Weight']]


# In[ ]:


'''
function of cleaning the Weight Column
'''

def weight_in_lbs(weight):
    try:
        return int(weight)
    except:
        pass
    finally:
        if len(weight.split('k'))==2:
            return int(weight.split('k')[0])*2.23
        


# In[ ]:


df['Weight']=df['Weight'].apply(weight_in_lbs)


# In[ ]:


df['Weight']


# In[ ]:


'''
Cleaning the height column
'''
def height_in_cm(height):
    try:
        return (int(height.split('-')[0])*30.48) + (int(height.split('-')[1])*2.54)
    except:
        pass
    finally:
        if len(height.split('c'))==2:
            return int(height.split('c')[0])


# In[ ]:


df['Height']=df['Height'].apply(height_in_cm)


# In[ ]:


df['Height']


# In[ ]:


'''
This returns the smallest and the tallest player.
'''
print(df['Player'][df['Height'].idxmax()],df['Height'].max(),'cm','The Tallest Player to ever entitiled a Player of the Week Award')
print(df['Player'][df['Height'].idxmin()],df['Height'].min(),'cm','The Smallest Player to ever entitiled a Player of the Week Award')


# In[ ]:


'''
This returns the lightest and the heaviest player.
'''

print(df['Player'][df['Weight'].idxmax()],df['Weight'].max(),'lbs','The Heaviest Player to ever entitiled a Player of the Week Award')
print(df['Player'][df['Weight'].idxmin()],df['Weight'].min(),'lbs','The Lightest Player to ever entitiled a Player of the Week Award')


# In[ ]:


'''
Now we have to deal with missing data!
For the conferrence column we have to fill those missing data
'''
df['Conference']


# In[ ]:


'''
Providing NBA Team to Missing data on Conference
'''

NBA_team=[]
for team in df['Team']:
    NBA_team.append(team)
NBA_team=set(NBA_team)

est_team=['Atlanta Hawks',
 'Boston Celtics',
 'Brooklyn Nets',
 'Charlotte Bobcats',
 'Charlotte Hornets',
 'Chicago Bulls',
 'Cleveland Cavaliers','Detroit Pistons','Indiana Pacers', 'Miami Heat',
 'Milwaukee Bucks','New Jersey Nets', 'New York Knicks', 'Orlando Magic','Philadelphia Sixers','Seattle SuperSonics',
 'Toronto Raptors']

wst_team=['Dallas Mavericks',
 'Denver Nuggets','Golden State Warriors',
 'Houston Rockets','Los Angeles Clippers',
 'Los Angeles Lakers',
 'Memphis Grizzlies','Minnesota Timberwolves','New Orleans Hornets',
 'New Orleans Pelicans','Oklahoma City Thunder','Phoenix Suns',
 'Portland Trail Blazers',
 'Sacramento Kings',
 'San Antonio Spurs','Utah Jazz',
 'Washington Bullets',
 'Washington Wizards']


# In[ ]:


'''
This function returns whether the team is on the east or west. 
'''
def input_conf(team):
    for h in team:
        for i in est_team:
            if i == team:
                return('East')
            for j in wst_team:
                if j==team:
                    return ('West')


# In[ ]:


df['Conference']=df['Team'].apply(input_conf)


# In[ ]:


df['Conference']


# In[ ]:


plt.figure(figsize=(20,10))
df['Conference'].value_counts().plot(kind='bar')


# In[ ]:


'''
Real Value:If two awards given at the same week [East & West] the player got 0.5, else 1 point.
'''
plt.figure(figsize=(20,10))
sns.countplot(x='Real_value',hue='Conference',data=df)


# In[ ]:


'''
Visualization for Age and Weight.
'''
sns.jointplot(x='Age',y='Weight',data=df)


# In[ ]:


'''
Visualization for Age and Height.
'''
sns.jointplot(x='Age',y='Height',data=df)


# In[ ]:


'''
Visulatization of Seasons in league and Age
'''
sns.jointplot(x='Seasons in league',y='Age',data=df)


# In[ ]:


print(df['Player'][df['Seasons in league'].idxmax()],', receive player of the week award early in his career')
print(df['Player'][df['Seasons in league'].idxmin()],', receive player of the week award late in his career')


# > Let us now compare the top 5 most awarded players based of their awards and it's Real Value

# In[ ]:


plt.figure(figsize=(20,10))
df['Player'].value_counts().head().plot(kind='bar')
plt.title("Top 5 players")


# In[ ]:


'''
Create a separate dataset for each player
'''
Malone=pd.DataFrame(df[df['Player']=='Karl Malone'])
MJ=pd.DataFrame(df[df['Player']=='Michael Jordan'])
KD=pd.DataFrame(df[df['Player']=='Kevin Durant'])
Kobe=pd.DataFrame(df[df['Player']=='Kobe Bryant'])
LeBron=pd.DataFrame(df[df['Player']=='LeBron James'])


# In[ ]:


'''
Visualization comparison of each players. 
'''
sns.lmplot(data=MJ,x='Age',y='Season short',hue='Team',col='Real_value',fit_reg=False)
sns.lmplot(data=Malone,x='Age',y='Season short',hue='Team',col='Real_value',fit_reg=False)
sns.lmplot(data=Kobe,x='Age',y='Season short',hue='Team',col='Real_value',fit_reg=False)
sns.lmplot(data=LeBron,x='Age',y='Season short',hue='Team',col='Real_value',fit_reg=False)
sns.lmplot(data=KD,x='Age',y='Season short',hue='Team',col='Real_value',fit_reg=False)


# In[ ]:


sns.countplot(data=Malone,x='Real_value')
plt.title("Malone's Real Value")


# In[ ]:


sns.countplot(data=KD,x='Real_value')
plt.title("KD's Real Value")


# In[ ]:


sns.countplot(data=Kobe,x='Real_value')
plt.title("Kobe's Real Value")


# In[ ]:


sns.countplot(data=LeBron,x='Real_value')
plt.title("Lebron's Real Value")


# In[ ]:


sns.countplot(data=MJ,x='Real_value')
plt.title("MJ's Real Value")


# > **Summary:**
# > If based of the data if the real impact on the game will be based of the real value **MJ** > any other player.

# In[ ]:




