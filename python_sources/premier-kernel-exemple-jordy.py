#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')
df_tournaments=pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')


# In[ ]:


df=df[(df['Season']>=2013)&(df['Season']<=2015)]
df_2015=df_tournaments[df_tournaments['Season']==2015]


# In[ ]:


points_pos_team=(df.groupby('WTeamID')['WScore'].sum()-df.groupby('WTeamID')['LScore'].sum())
points_neg_team=(df.groupby('LTeamID')['LScore'].sum()-df.groupby('LTeamID')['WScore'].sum())
diff_per_game=(points_pos_team+points_neg_team)/(df.groupby('LTeamID')['LScore'].count()+df.groupby('WTeamID')['LScore'].count())


# In[ ]:


diff_per_game


# In[ ]:


team_in_tournaments=np.unique(list(df_2015['WTeamID'].unique())+list(df_2015['LTeamID'].unique()))


# In[ ]:


dico_proba={'team1_team2':[],'diff_avg_points':[]}
for team1 in team_in_tournaments:
    for team2 in team_in_tournaments:
        if team1!=team2:
            dico_proba['team1_team2'].append(str(team1)+'_'+str(team2))
            dico_proba['diff_avg_points'].append(diff_per_game[team1]-diff_per_game[team2])
df_proba=pd.DataFrame(dico_proba)


# In[ ]:


df_proba['proba']=df_proba['diff_avg_points'].rank(pct=True)
df_proba['proba2']=(df_proba['diff_avg_points']-df_proba['diff_avg_points'].min())/(df_proba['diff_avg_points'].max()-df_proba['diff_avg_points'].min())


# In[ ]:


table_proba=df_proba.set_index('team1_team2')['proba']


# In[ ]:





# In[ ]:


df_2015['probaWteam']=(df_2015['WTeamID'].apply(lambda x:str(x))+'_'+df_2015['LTeamID'].apply(lambda x:str(x))).apply(lambda x:table_proba[x])


# In[ ]:


-df_2015['probaWteam'].apply(lambda x:np.log(x)).mean()


# In[ ]:


df_2015

