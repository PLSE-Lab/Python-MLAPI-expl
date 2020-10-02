#!/usr/bin/env python
# coding: utf-8

# #THIS NOTEBOOK IS BASED ON THE STATISTICS ON REGULAR SEASON TO PREDICT NCAA TOURNEY MATCHES.
# 
# f_ denotes first team with low ID.
# 
# s_denotes second team with high ID.
# 
# It may look like containing several for loops but the data wasn't so tiddy.
# 

# In[ ]:


#Importing libraries
import pandas as pd#data manipulation
pd.set_option('display.max_columns', None)
import numpy as np # mathematical operations
import scipy as sci # math ops
import seaborn as sns # visualizations
import matplotlib.pyplot as plt # for plottings
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Suppressing filters
import warnings
warnings.filterwarnings("ignore")


# # TEAMS

# In[ ]:


teams = pd.read_csv("../input/Teams.csv")


# In[ ]:


#Glimpsing data
teams.head()


# In[ ]:


#Showing which teams aren't in this year's regular season
mask1 = (teams['LastD1Season'] != 2018 )
print("teams which aren't in 351 teams are :",list(teams[mask1]['TeamName']))


# # SEASONS

# In[ ]:


seasons = pd.read_csv("../input/Seasons.csv")


# In[ ]:


seasons.head()


# In[ ]:


seasons.shape


# # SEED

# In[ ]:


seed = pd.read_csv("../input/NCAATourneySeeds.csv")


# In[ ]:


seed.tail()


# In[ ]:


seed.shape


# In[ ]:


seed.groupby(by='Season').count().tail(10)


# In[ ]:


#Separating seed into no and region
seed['region'] = seed['Seed'].apply(lambda x: x[0])
seed['no'] = seed['Seed'].apply(lambda x: x[1:])


# In[ ]:


seed.head()


# In[ ]:


print("no of unique characters in region column in seed data=",len(set(seed['region'])))


# In[ ]:


print("no of unique characters in no column in seed data=",len(set(seed['no'])))


# In[ ]:


print("unique seed values are",set(seed['no']))


# In[ ]:


seed['no_len'] = seed['no'].apply(lambda x: len(x))


# In[ ]:


seed_first_four = seed[seed['no_len']>2]


# In[ ]:


seed_first_four.tail()


# # RegularSeasonCompactResults

# In[ ]:


#Reading data
rscr = pd.read_csv('../input/RegularSeasonCompactResults.csv')


# In[ ]:


rscr.head()


# In[ ]:


rscr.shape


# In[ ]:


len(set(rscr['WTeamID']))


# In[ ]:


print("percentage of games where Overtime was played = ",round(len(rscr[rscr['NumOT']>0])/len(rscr)*100,4))


# In[ ]:


(rscr['WLoc'].value_counts()/len(rscr)).plot(kind='bar',title='Total wins since 1985 (H:HOME,A:VISITING,N:NEUTRAL)')


# # #NCAA Tourney Compact Results

# In[ ]:


#Tournament results
ntcr = pd.read_csv('../input/NCAATourneyCompactResults.csv')


# In[ ]:


ntcr.head()


# In[ ]:


len(ntcr[ntcr['Season'] ==2000])


# In[ ]:


ntcr.shape


# In[ ]:


ntcr['Season'].value_counts().head(8)


# In[ ]:


len(set(ntcr['WLoc']))


# ALL NCAA PLAYS ARE PLAYED ON A NEUTRAL FIELD.

# In[ ]:


years = sorted(list(set(ntcr['Season'])))


# In[ ]:


years[:5]


# In[ ]:


year_team_dict = {}


# In[ ]:


for i in years:
    year_team_dict[str(i)] = list(set(list(set(ntcr[ntcr['Season'] ==i]['WTeamID'])) + list(set(ntcr[ntcr['Season'] ==i]['LTeamID']))))


# In[ ]:


len(year_team_dict['2017'])


# In[ ]:


[int(x) for x in list(year_team_dict.keys())][:5]


# In[ ]:


year_team_dict['2017'][:5]


# # REGULAR SEASON DETAILED RESULTS

# In[ ]:


rsdr = pd.read_csv('../input/RegularSeasonDetailedResults.csv')


# In[ ]:


rsdr.head()


# In[ ]:


#How many teams in 2017 in regular season
len(set(rsdr[(rsdr['Season'] ==2017)]['LTeamID']))


# In[ ]:


#2 point throws for winning team
rsdr['WFGM2'] = rsdr['WFGM'] - rsdr['WFGM3']
rsdr['WFGA2'] = rsdr['WFGA'] - rsdr['WFGA3']
#2 point throws for losing teamm
rsdr['LFGM2'] = rsdr['LFGM'] - rsdr['LFGM3']
rsdr['LFGA2'] = rsdr['LFGA'] - rsdr['LFGA3']


# In[ ]:


rsdr.shape


# In[ ]:


len(set(rsdr['WTeamID']))


# In[ ]:


#Creating list of columns
clms=['Score','EScore','FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF','FGM2','FGA2']


# In[ ]:


#Creating empty dataframes
df_2003 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2003]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2003]['WTeamID'])))
df_2004 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2004]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2004]['WTeamID'])))
df_2005 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2005]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2005]['WTeamID'])))
df_2006 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2006]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2006]['WTeamID'])))
df_2007 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2007]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2007]['WTeamID'])))
df_2008 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2008]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2008]['WTeamID'])))
df_2009 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2009]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2009]['WTeamID'])))
df_2010 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2010]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2010]['WTeamID'])))
df_2011 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2011]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2011]['WTeamID'])))
df_2012 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2012]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2012]['WTeamID'])))
df_2013 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2013]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2013]['WTeamID'])))
df_2014 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2014]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2014]['WTeamID'])))
df_2015 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2015]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2015]['WTeamID'])))
df_2016 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2016]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2016]['WTeamID'])))
df_2017 = pd.DataFrame(np.zeros((len(set(rsdr[rsdr['Season'] ==2017]['WTeamID'])),17)),columns=clms,index=list(set(rsdr[rsdr['Season'] ==2017]['WTeamID'])))


# In[ ]:


#Merging these data frames in a list
df_list = [df_2003,df_2004,df_2005,df_2006,df_2007,df_2008,df_2009,df_2010,df_2011,df_2012,df_2013,df_2014,df_2015,df_2016,df_2017]


# In[ ]:


get_ipython().run_cell_magic('time', '', "#Taking statistics for each team for each year\nyear = 2003\nfor m in df_list:\n    for i in list(set(rsdr[rsdr['Season'] ==year]['LTeamID'])):\n        klm = pd.DataFrame()\n        klm = rsdr[(rsdr['Season']==year)&((rsdr['WTeamID'] ==i)|(rsdr['LTeamID'] ==i))]\n        for j in clms:\n            if j=='EScore':\n                m.loc[i,j] = (klm[klm['WTeamID'] == i]['LScore'].values.sum() + klm[klm['LTeamID'] == i]['WScore'].values.sum() ) / len(klm)\n            else:\n                m.loc[i,j] = (klm[klm['WTeamID'] == i]['W'+j].values.sum() + klm[klm['LTeamID'] == i]['L'+j].values.sum() ) / len(klm)\n    year = year + 1")


# In[ ]:


#Assigning zeros to seed values
df_2003['Seed'],df_2004['Seed'],df_2005['Seed'],df_2006['Seed'],df_2007['Seed'],df_2008['Seed'],df_2009['Seed'],df_2010['Seed'],df_2011['Seed'],df_2012['Seed'],df_2013['Seed'],df_2014['Seed'],df_2015['Seed'],df_2016['Seed'],df_2017['Seed']=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0


# In[ ]:


df_2017.head()


# In[ ]:


#Taking the statistics only for tournament games
year = 2003
r = 0
for i in df_list:
    m = i.loc[pd.Series(seed[seed['Season'] ==year]['TeamID']).sort_values(ascending=True),:]
    for j in list(m.index):
        m.loc[j,'Seed'] = list(seed[(seed['Season'] ==year)&(seed['TeamID'] ==j)]['Seed'])[0]
    df_list[r] = m    
    year = year + 1
    r = r+1


# In[ ]:


df_list[14].shape


# In[ ]:


#Splitting seed column into 2: no and region
k = 0
for i in df_list:
    i['Seed_no'] = i['Seed'].map(lambda x: int(x[1:3]))
    i['Seed_region'] = i['Seed'].map(lambda x: x[0])
    df_list[k] = i
    k = k+1


# In[ ]:


#Deleting seed column
k = 0
for i in df_list:
    del i['Seed']
    df_list[k] = i
    k = k + 1


# In[ ]:


#We are using the below for loops because some teams only won and didn't lose in regular season. It caused some errors.
#Therefore I assigned for 2 special rows in 2014 for team 1455 and in 2015 for team 1246
for i in clms:
    if i =='EScore':
        df_list[11].loc[1455,i] = rsdr[(rsdr['Season'] ==2014)&(rsdr['WTeamID'] ==1455)]['LScore'].mean()
    else:
        df_list[11].loc[1455,i] = rsdr[(rsdr['Season'] ==2014)&(rsdr['WTeamID'] ==1455)]['W'+i].mean()
for i in clms:
    if i =='EScore':
        df_list[12].loc[1246,i] = rsdr[(rsdr['Season'] ==2015)&(rsdr['WTeamID'] ==1246)]['LScore'].mean()
    else:
        df_list[12].loc[1246,i] = rsdr[(rsdr['Season'] ==2015)&(rsdr['WTeamID'] ==1246)]['W'+i].mean()


# In[ ]:


df_list[0].head()


# In[ ]:


#Putting the diff between average scored points and average eaten points.
for m in df_list:
    m['diff'] = m['Score'] - m['EScore']


# In[ ]:


df_list[0].shape


# # NCAA DETAILED

# In[ ]:


ntdr = pd.read_csv('../input/NCAATourneyDetailedResults.csv')


# In[ ]:


ntdr.head()


# In[ ]:


#Create output column indicating which team win
def winning(data):
    if data['WTeamID'] < data['LTeamID']:
        return 1
    else:
        return 0


# In[ ]:


ntdr['winning'] = ntdr.apply(winning,axis=1)


# In[ ]:


#2 point throws
ntdr['WFGM2'] = ntdr['WFGM'] - ntdr['WFGM3']
ntdr['WFGA2'] = ntdr['WFGA'] - ntdr['WFGA3']
#2 point throws
ntdr['LFGM2'] = ntdr['LFGM'] - ntdr['LFGM3']
ntdr['LFGA2'] = ntdr['LFGA'] - ntdr['LFGA3']


# In[ ]:


#Creating lists for column names
f_columns = ['f_'+str(x) for x in list(df_list[14].columns.values)]
s_columns = ['s_'+str(x) for x in list(df_list[14].columns.values)]
t_columns = f_columns + s_columns


# In[ ]:


#Creating empty data frame for past ncaa tourney matches
nc_2003 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2003].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2003][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2004 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2004].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2004][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2005 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2005].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2005][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2006 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2006].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2006][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2007 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2007].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2007][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2008 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2008].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2008][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2009 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2009].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2009][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2010 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2010].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2010][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2011 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2011].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2011][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2012 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2012].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2012][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2013 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2013].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2013][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2014 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2014].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2014][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2015 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2015].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2015][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2016 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2016].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2016][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)
nc_2017 = pd.concat([pd.DataFrame(np.zeros((ntdr[ntdr['Season'] ==2017].shape[0],df_list[0].shape[1]*2)),columns= t_columns),ntdr[ntdr['Season'] ==2017][['WTeamID','LTeamID','winning']].reset_index(drop=True)],axis=1)


# In[ ]:


nc_2017.shape


# In[ ]:


nc_2017.head()


# In[ ]:


#Merging dataframes in a list
nc_list = [nc_2003,nc_2004,nc_2005,nc_2006,nc_2007,nc_2008,nc_2009,nc_2010,nc_2011,nc_2012,nc_2013,nc_2014,nc_2015,nc_2016,nc_2017]


# In[ ]:


len(nc_list)


# In[ ]:


get_ipython().run_cell_magic('time', '', "r = 0\nfor i in nc_list:\n    for j in range(len(i)):\n        for m in range(len(f_columns)):\n            i.iloc[j,m] = df_list[r].loc[min(i.loc[j,'LTeamID'],i.loc[j,'WTeamID']),f_columns[m][2:]]\n        for m in range(len(f_columns),len(f_columns)*2):\n            i.iloc[j,m] = df_list[r].loc[max(i.loc[j,'LTeamID'],i.loc[j,'WTeamID']),s_columns[m-len(f_columns)][2:]]\n    nc_list[r] = i\n    r = r+1")


# # COACHES

# In[ ]:


#Reading data
coaches= pd.read_csv('../input/TeamCoaches.csv')


# In[ ]:


#Creating lists for teamsID for each year
l_2003 = list(set(coaches[coaches['Season'] ==2003]['TeamID']))
l_2004 = list(set(coaches[coaches['Season'] ==2004]['TeamID']))
l_2005 = list(set(coaches[coaches['Season'] ==2005]['TeamID']))
l_2006 = list(set(coaches[coaches['Season'] ==2006]['TeamID']))
l_2007 = list(set(coaches[coaches['Season'] ==2007]['TeamID']))
l_2008 = list(set(coaches[coaches['Season'] ==2008]['TeamID']))
l_2009 = list(set(coaches[coaches['Season'] ==2009]['TeamID']))
l_2010 = list(set(coaches[coaches['Season'] ==2010]['TeamID']))
l_2011 = list(set(coaches[coaches['Season'] ==2011]['TeamID']))
l_2012 = list(set(coaches[coaches['Season'] ==2012]['TeamID']))
l_2013 = list(set(coaches[coaches['Season'] ==2013]['TeamID']))
l_2014 = list(set(coaches[coaches['Season'] ==2014]['TeamID']))
l_2015 = list(set(coaches[coaches['Season'] ==2015]['TeamID']))
l_2016 = list(set(coaches[coaches['Season'] ==2016]['TeamID']))
l_2017 = list(set(coaches[coaches['Season'] ==2017]['TeamID']))
#l_2018 = list(set(coaches[coaches['Season'] ==2018]['TeamID']))
l_list = [l_2003,l_2004,l_2005,l_2006,l_2007,l_2008,l_2009,l_2010,l_2011,l_2012,l_2013,l_2014,l_2015,l_2016,l_2017]


# In[ ]:


#Creating empty dicts and merging them
d_2003,d_2004,d_2005,d_2006,d_2007,d_2008,d_2009,d_2010,d_2011,d_2012,d_2013,d_2014,d_2015,d_2016,d_2017={},{},{},{},{},{},{},{},{},{},{},{},{},{},{}
d_list = [d_2003,d_2004,d_2005,d_2006,d_2007,d_2008,d_2009,d_2010,d_2011,d_2012,d_2013,d_2014,d_2015,d_2016,d_2017]


# In[ ]:


get_ipython().run_cell_magic('time', '', "#The below code calculates which team has its coach for how many years\nyear = 2003\nr = 0\nfor i in l_list:\n    for j in i:\n        temp = coaches[(coaches['Season']==year)&(coaches['TeamID'] ==j)]\n        coach = temp.tail(1)['CoachName']\n        coach = list(coach)[0]\n        yr = len(coaches[(coaches['Season'] <= year) & (coaches['CoachName'] ==coach)&(coaches['TeamID'] ==j)])\n        d_list[r][str(j)] =yr\n    r = r+1\n    year = year+1")


# In[ ]:


get_ipython().run_cell_magic('time', '', "k=0\nfor i in d_list:\n    tempo = pd.Series(list(i.values()),index= list( i.keys() ) )\n    nc_list[k]['f_coach'],nc_list[k]['s_coach'] = 0,0\n    for j in range(len(nc_list[k])):\n        nc_list[k]['f_coach'][j] = tempo[str(min(nc_list[k].iloc[j]['WTeamID'],nc_list[k].iloc[j]['LTeamID']))]\n        nc_list[k]['s_coach'][j] = tempo[str(max(nc_list[k].iloc[j]['WTeamID'],nc_list[k].iloc[j]['LTeamID']))]\n    k = k+1 ")


# # RANKINGS

# In[ ]:


massey = pd.read_csv("../input/MasseyOrdinals.csv")


# In[ ]:


massey.shape


# In[ ]:


massey.tail()


# In[ ]:


#Filtering data according to years
m_2003 = massey[massey['Season'] ==2003]
m_2004 = massey[massey['Season'] ==2004]
m_2005 = massey[massey['Season'] ==2005]
m_2006 = massey[massey['Season'] ==2006]
m_2007 = massey[massey['Season'] ==2007]
m_2008 = massey[massey['Season'] ==2008]
m_2009 = massey[massey['Season'] ==2009]
m_2010 = massey[massey['Season'] ==2010]
m_2011 = massey[massey['Season'] ==2011]
m_2012 = massey[massey['Season'] ==2012]
m_2013 = massey[massey['Season'] ==2013]
m_2014 = massey[massey['Season'] ==2014]
m_2015 = massey[massey['Season'] ==2015]
m_2016 = massey[massey['Season'] ==2006]
m_2017 = massey[massey['Season'] ==2017]

m_list = [m_2003,m_2004,m_2005,m_2006,m_2007,m_2008,m_2009,m_2010,m_2011,m_2012,m_2013,m_2014,m_2015,m_2016,m_2017]


# In[ ]:


#Which systemnames are in which years
s_2003 = list(set(m_2003['SystemName']))
s_2004 = list(set(m_2004['SystemName']))
s_2005 = list(set(m_2005['SystemName']))
s_2006 = list(set(m_2006['SystemName']))
s_2007 = list(set(m_2007['SystemName']))
s_2008 = list(set(m_2008['SystemName']))
s_2009 = list(set(m_2009['SystemName']))
s_2010 = list(set(m_2010['SystemName']))
s_2011 = list(set(m_2011['SystemName']))
s_2012 = list(set(m_2012['SystemName']))
s_2013 = list(set(m_2013['SystemName']))
s_2014 = list(set(m_2014['SystemName']))
s_2015 = list(set(m_2015['SystemName']))
s_2016 = list(set(m_2016['SystemName']))
s_2017 = list(set(m_2017['SystemName']))


# In[ ]:


s_list = [s_2003,s_2004,s_2005,s_2006,s_2007,s_2008,s_2009,s_2010,s_2011,s_2012,s_2013,s_2014,s_2015,s_2016,s_2017]


# In[ ]:


big_list = []


# In[ ]:


get_ipython().run_cell_magic('time', '', "#We are taking evaluations only after 117th day for each team.\nr = 0\nfor m in m_list:\n    middle_dict = {}\n    for l in l_list[r]:\n        small_list = []\n        temp = m[(m['TeamID'] ==l)&(m['RankingDayNum'] >117)]\n        for s in s_list[r]:\n            mn = temp[temp['SystemName'] ==s]['OrdinalRank'].mean()\n            small_list.append(mn)\n        middle_dict[str(l)] = small_list\n    big_list.append(middle_dict)\n    r = r+1")


# In[ ]:


#Averaging evaluations
r = 0
for b in big_list:
    b = pd.DataFrame(b).T
    big_list[r] =b
    r = r+1
#Converting np.array to pd.Series
r = 0
for b in big_list:
    i = b.index
    b = np.nanmean(b,axis=1)
    big_list[r] = pd.Series(b,index=i)
    r = r + 1
#Filling NaN values with mean. Some teams aren't evaluated after 117th day.
r = 0
for i in big_list:
    k = i.mean()
    big_list[r] = i.fillna(k)
    r = r+1    


# In[ ]:


get_ipython().run_cell_magic('time', '', "#The code below is rounding the results. If the rank is equal to 34.2, it was rounded to 34.\nr = 0\nfor i in nc_list:\n    i['f_rank'],i['s_rank'] = 0,0\n    for j in range(len(i)):\n        a = min(i.iloc[j]['WTeamID'],i.iloc[j]['LTeamID'])\n        b = max(i.iloc[j]['WTeamID'],i.iloc[j]['LTeamID'])\n        a = big_list[r].loc[str(a)]\n        b = big_list[r].loc[str(b)]\n        i['f_rank'][j] = a \n        i['s_rank'][j] = b\n    nc_list[r] = i\n    r = r + 1")


# In[ ]:


#Assigning season to rows
j=2003
for i in nc_list:
    i['Season'] = j
    j = j + 1


# In[ ]:


nc_list[0].head()


# # MODELLING

# In[ ]:


#Merging all years data
X = pd.concat([nc_list[0],nc_list[1],nc_list[2],nc_list[3],nc_list[4],nc_list[5],nc_list[6],nc_list[7],nc_list[8],nc_list[9],nc_list[10],nc_list[11],nc_list[12],nc_list[13],nc_list[14]])


# In[ ]:


X = X.reset_index(drop=True)


# In[ ]:


X.shape


# In[ ]:


#f_ denotes first team with lower id. s_ denotes second team with higher id.
X.head()


# In[ ]:


x_columns = list(X.columns.values)


# In[ ]:


x_columns = [x for x in x_columns if x not in ['f_Seed_region','s_Seed_region','winning','Season']]


# In[ ]:


len(x_columns)


# In[ ]:


y = X['winning']
X = X.drop('winning',axis=1)


# In[ ]:


#Creating train and test data
X_train = X[:713]
y_train = y[:713]
X_test = X[713:]
y_test = y[713:]
#Dividing test data into 2014,2015,2016,2017
t_2014 = X_test[:67]
t_2015 = X_test[67:134]
t_2016 = X_test[134:201]
t_2017 = X_test[201:268]


# In[ ]:


teams_2014 = sorted(list(set(list(set(t_2014['WTeamID'])) + list(set(t_2014['LTeamID'])))))
teams_2015 = sorted(list(set(list(set(t_2015['WTeamID'])) + list(set(t_2015['LTeamID'])))))
teams_2016 = sorted(list(set(list(set(t_2016['WTeamID'])) + list(set(t_2016['LTeamID'])))))
teams_2017 = sorted(list(set(list(set(t_2017['WTeamID'])) + list(set(t_2017['LTeamID'])))))


# In[ ]:


teams_list = [teams_2014,teams_2015,teams_2016,teams_2017]


# In[ ]:


row_num = int(((68*67)/2)*4)
col_num = 2


# In[ ]:


sub_df = pd.DataFrame(np.zeros((row_num,col_num)),columns=['FirstID','SecondID'])


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in range(0,row_num,2278):\n    l = 0 + i\n    u = 67 + i\n    r = 0\n    m = 67\n    k = teams_list[int(i/2278)]\n    while(0 < m):\n        for j in range(l,u):\n            sub_df['FirstID'][j] = k[r]\n        r = r+1\n        l = u   \n        u = u + m - 1\n        m = m - 1")


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in range(0,row_num,2278):\n    l = 0 + i\n    u = 67 + i\n    r = 0\n    m = 67\n    k = teams_list[int(i/2278)]\n    while(0 < m):\n        t = 0\n        for j in range(l,u):\n            sub_df['SecondID'][j] = k[r+t+1]\n            t = t + 1\n        r = r+1\n        l = u   \n        u = u + m - 1\n        m = m - 1   ")


# In[ ]:


sub_df.tail()


# In[ ]:


#Creating year column.
sub_df['year'] = 0
sub_df['year'][:2278]=2014
sub_df['year'][2278:int(2278*2)]=2015
sub_df['year'][int(2278*2):int(2278*3)]=2016
sub_df['year'][int(2278*3):]=2017


# In[ ]:


#creating an empty dataframe
testing = pd.DataFrame(np.zeros((len(sub_df),len(x_columns))) ,columns=x_columns)


# In[ ]:


#Renaming columns
testing = testing.rename(columns={'WTeamID':'FirstID','LTeamID':'SecondID'})


# In[ ]:


testing.head()


# In[ ]:


testing.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in range(len(sub_df)):\n    season = int(sub_df.iloc[i]['year'])\n    f_t = int(sub_df.iloc[i]['FirstID'])\n    s_t = int(sub_df.iloc[i]['SecondID'])\n    testing.iloc[i]['FirstID'] = f_t\n    testing.iloc[i]['SecondID'] = s_t\n    testing.iloc[i]['f_coach'] = int(d_list[int(season-2003)][str(f_t)])\n    testing.iloc[i]['s_coach'] = int(d_list[int(season-2003)][str(s_t)])\n    testing.iloc[i]['f_rank'] = big_list[season-2003][str(f_t)]\n    testing.iloc[i]['s_rank'] = big_list[season-2003][str(s_t)]\n    for j in list(testing.columns.values)[:19]:\n        k = j[2:]\n        testing.iloc[i][j] = df_list[season-2003].loc[f_t,k]\n    for j in list(testing.columns.values)[19:38]:\n        k = j[2:]\n        testing.iloc[i][j] = df_list[season-2003].loc[s_t,k]    ")


# In[ ]:


testing.tail()


# In[ ]:


X.shape


# In[ ]:


#Deleting some columns which we don't put in the model.
del X['f_Seed_region'],X['s_Seed_region'],X['WTeamID'],X['LTeamID'],X['Season']
del testing['FirstID'],testing['SecondID']


# In[ ]:


X_train = X[:713]
y_train = y[:713]


# In[ ]:


#Checking number of columns is equal
X_train.shape[1] == testing.shape[1]


# In[ ]:


def make_suitable_column(abc):
    yr = int(abc['year'])
    fi = int(abc['FirstID'])
    si = int(abc['SecondID'])
    k = str(yr) + '_' + str(fi) + '_' + str(si)
    return str(k)


# In[ ]:


submission_column = sub_df.apply(make_suitable_column,axis=1)


# In[ ]:


submission_column.name = 'ID'


# In[ ]:


corr = X_train.corr()
# Set up the matplot figure
f,ax = plt.subplots(figsize=(30,25))
#Draw the heatmap using seaborn
sns.heatmap(corr, cmap='inferno', annot=True)


# In[ ]:


from xgboost import XGBClassifier
model_xgb = XGBClassifier()
model_xgb.fit(X_train,y_train)


# In[ ]:


pd.Series(model_xgb.feature_importances_,index=list(X_train.columns.values)).sort_values(ascending=True).plot(kind='barh',figsize=(12,18),title='XGBOOST FEATURE IMPORTANCE')


# In[ ]:




