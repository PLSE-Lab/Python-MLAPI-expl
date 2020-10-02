#!/usr/bin/env python
# coding: utf-8

# In[347]:


import warnings
warnings.simplefilter('ignore', DeprecationWarning)

import pandas as pd
pd.options.mode.chained_assignment = None

df = pd.read_csv('../input/NFL Play by Play 2009-2017 (v4).csv',low_memory=False)
df.info() 


# In[348]:


#get team abbreviations
pd.unique(df.posteam.values.ravel())


# In[349]:


#find Dallas plays with Witten and Romo in them
witten_romo = df[(df["posteam"] == 'DAL')  & (df["desc"].str.contains("J.Witten"))
                 & (df["desc"].str.contains("T.Romo")) & (df['PassAttempt'] == 1)]
witten_romo.head() #see Witten & Romo pass plays


# In[350]:


#find New England plays with Brady and the Gronk
gronk_brady = df[(df["posteam"] == 'NE') & (df["desc"].str.contains("T.Brady")) &
                                            (df["desc"].str.contains("R.Gronkowski")) &
                                            (df['PassAttempt'] == 1)]
gronk_brady.head() #see gronk and brady pass plays


# In[351]:


#find Titans plays with Walker and Mariota
walker_mariota = df[(df["posteam"] == 'TEN') & (df["desc"].str.contains("D.Walker")) & 
               (df["desc"].str.contains("M.Mariota")) & (df['PassAttempt'] == 1)]
walker_mariota.head() #see Walker and Mariota pass plays


# In[352]:


#find New England plays with Brady and the Gronk
gronk_brady = df[(df["posteam"] == 'NE') & (df["desc"].str.contains("T.Brady")) &
                                            (df["desc"].str.contains("R.Gronkowski")) &
                                            (df['PassAttempt'] == 1)]
gronk_brady.head() #see gronk and brady pass plays


# In[353]:


#find Bengals plays with Eifert & Dalton
eifert_dalton = df[(df["posteam"] == 'CIN') & (df["desc"].str.contains("T.Eifert")) & 
               (df["desc"].str.contains("A.Dalton")) & (df['PassAttempt'] == 1)]
eifert_dalton.head() #see Eifert and Dalton pass plays


# In[354]:


#find Carolina plays with Eifert & Dalton
olsen_cam = df[(df["posteam"] == 'CAR') & (df["desc"].str.contains("G.Olsen")) & 
               (df["desc"].str.contains("C.Newton")) & (df['PassAttempt'] == 1)]
olsen_cam.head() #see Eifert and Dalton pass plays


# In[355]:


#find Chargers plays with Gates and Rivers
gates_rivers = df[(df["posteam"] == 'SD') & (df["desc"].str.contains("A.Gates")) & 
               (df["desc"].str.contains("P.Rivers")) & (df['PassAttempt'] == 1)]
gates_rivers.head() #see Gates and Rivers pass plays


# In[356]:


#find Saints plays with Graham and Brees
graham_brees = df[(df["posteam"] == 'NO') & (df["desc"].str.contains("J.Graham")) & 
               (df["desc"].str.contains("D.Brees")) & (df['PassAttempt'] == 1)]
graham_brees.head() #see Gates and Rivers pass plays


# In[357]:


graham_wilson = df[(df["posteam"] == 'SEA') & (df["desc"].str.contains("J.Graham")) & 
               (df["desc"].str.contains("R.Wilson")) & (df['PassAttempt'] == 1)]
graham_wilson.head() 


# In[358]:


bradford_ertz = df[(df["posteam"] == 'PHI') & (df["desc"].str.contains("S.Bradford")) & 
               (df["desc"].str.contains("Z.Ertz")) & (df['PassAttempt'] == 1)]
bradford_ertz.head() 


# In[359]:


ebron_staff = df[(df["posteam"] == 'DET') & (df["desc"].str.contains("M.Stafford")) & 
               (df["desc"].str.contains("E.Ebron")) & (df['PassAttempt'] == 1)]
ebron_staff.head() 


# In[360]:


thomas_bortles = df[(df["posteam"] == 'JAC') & (df["desc"].str.contains("B.Bortles")) & 
               (df["desc"].str.contains("J.Thomas")) & (df['PassAttempt'] == 1)]
thomas_bortles.head() 


# In[361]:


smith_kelce = df[(df["posteam"] == 'KC') & (df["desc"].str.contains("A.Smith")) & 
               (df["desc"].str.contains("T.Kelce")) & (df['PassAttempt'] == 1)]
smith_kelce.head() 


# In[362]:


#combine dataframes into one
witten_romo['duo'] = 'Witten & Romo'
gronk_brady['duo'] = 'Gronkowski & Brady'
olsen_cam['duo'] = 'Olsen & Newton'
eifert_dalton['duo'] = 'Eifter & Dalton'
walker_mariota['duo'] = 'Walker & Mariota'
gates_rivers['duo'] ='Gates & Rivers'
graham_brees['duo'] ='Graham & Brees'
graham_wilson['duo'] ='Graham & Wilson'
bradford_ertz['duo'] = 'Ertz & Bradford'
ebron_staff['duo']= 'Ebron & Stafford'
thomas_bortles['duo'] = 'Thomas & Bortles'
smith_kelce['duo']= 'Kelce & Smith'


frames = [gronk_brady, witten_romo, olsen_cam, eifert_dalton, walker_mariota, gates_rivers,
          graham_brees,graham_wilson,bradford_ertz,ebron_staff,thomas_bortles,smith_kelce]
df2 = pd.concat(frames)
df2.head()


# In[363]:


#how many pass plays for each duo
print (df2["duo"].value_counts())


# In[364]:


#how many plays did the duos have by season
df2['Season'] = df2['Date'].str[:4]
ct = pd.crosstab(df2['duo'],df2['Season'])
ax = sns.heatmap(ct, annot=True, fmt="d",cmap="YlGnBu").set_title('Number of Pass Plays Per Season')


# In[365]:


#how many yards did the duos net on average
import seaborn as sns
sns.set_context("notebook")
flatui = ["#9b59b6", "#2ecc71", "#95a5a6", "#e74c3c", "#34495e", "#3498db"]
sns.set_palette(flatui)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.rcParams['figure.figsize'] = [12.0, 10.0]
sns.boxplot(x="ydsnet", y="duo", data=df2)


# In[366]:


#what percentage of pass play were caught
import numpy as np
df2['pass_complete'] = np.where((df2['PassOutcome']=='Complete'), 1, 0)

comp = pd.crosstab(df2['duo'],df2.pass_complete.astype(bool))
completion_rate = comp.div(comp.sum(1).astype(float),axis=0)*100 
completion_rate.sort_index(axis=1, inplace=True, ascending=False)
completion_rate.plot(kind='barh', stacked=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)


# In[367]:


#lets look at the distribution of plays by net yards and whether it was a scoring play
sns.swarmplot(x="ydsnet", y="duo", hue="sp", data=df2,size=4)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)


# In[368]:


#create a new dataframe with just scoring pass plays from the duos 
#(that aren't interceptions by the defense)
scoring_play = df2[((df2["sp"] == 1) & (df2['InterceptionThrown'] == 0))]
scoring_play.head()


# In[369]:


#how has the most scoring plays?
print (scoring_play["duo"].value_counts())


# In[370]:


#plot the above counts
sns.countplot(y="duo", data=scoring_play, order = scoring_play['duo'].value_counts().index)


# In[371]:


#what percentage of duo pass plays resulted in a score?
score = pd.crosstab(df2['duo'],df2.sp.astype(bool))
score_rate = score.div(score.sum(1).astype(float),axis=0)*100 
score_rate.sort_index(axis=1, inplace=True, ascending=False)
score_rate.plot(kind='barh', stacked=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)


# In[372]:


ct = pd.crosstab(scoring_play['duo'],scoring_play['Season'])
ax = sns.heatmap(ct, annot=True, fmt="d",cmap="YlGnBu").set_title('Number of Scoring Pass Plays Per Season')


# In[373]:


#let's look at conversions to first down
#(ignore scoring plays)
#(in absence of first down obtained field, we assume net yards greater than yards to go = 1st)
df2['conversion'] = np.where((df2['ydsnet']>=df2['ydstogo']) & (df2['sp'] != 1), 1, 0)
conversions = df2[(df2["conversion"] == 1)]
conversions.head()


# In[374]:


#how many first downs did the duos get
print (conversions["duo"].value_counts())


# In[375]:


sns.countplot(y="duo", data=conversions, order = conversions['duo'].value_counts().index)


# In[376]:


#what percentage of non-scoring pass plays resulted in a first down
conv = pd.crosstab(df2['duo'],df2.conversion.astype(bool))
conversion_rate = conv.div(conv.sum(1).astype(float),axis=0)*100 
conversion_rate.sort_index(axis=1, inplace=True, ascending=False)
conversion_rate.plot(kind='barh', stacked=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)


# In[377]:


#how about third-down conversions?
third_downs = df2[(df2['down']==3)]
third_downs['third_conversion'] = np.where((third_downs['ydsnet']>=third_downs['ydstogo']) & (third_downs['sp'] != 1), 1, 0)
third_conversions = third_downs[(third_downs["third_conversion"] == 1)]
third_conversions.head()


# In[378]:


#how many third downs did these duos convert?
print (third_conversions["duo"].value_counts())


# In[379]:


#what percentage of third down conversion attempts resulted in first downs
conv = pd.crosstab(third_downs['duo'],third_downs.third_conversion.astype(bool))
conversion_rate = conv.div(conv.sum(1).astype(float),axis=0)*100 
conversion_rate.sort_index(axis=1, inplace=True, ascending=False)
conversion_rate.plot(kind='barh', stacked=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)


# In[380]:


ct = pd.crosstab(third_conversions['duo'],third_conversions['Season'])
ax = sns.heatmap(ct, annot=True, fmt="d",cmap="YlGnBu").set_title('Number of Third Down Conversions Per Season')


# In[381]:


turnover = df2[((df2["InterceptionThrown"] == 1) | (df2['Fumble'] == 1))]
turnover.head()


# In[382]:


print (turnover["duo"].value_counts())


# In[383]:


df2['turnover'] = np.where(((df2['InterceptionThrown']==1) | (df2['Fumble']==1)), 1, 0)

turn = pd.crosstab(df2['duo'],df2.turnover.astype(bool))
turnover_rate = turn.div(turn.sum(1).astype(float),axis=0)*100 
turnover_rate.sort_index(axis=1, inplace=True, ascending=False)
turnover_rate.plot(kind='barh', stacked=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)


# In[384]:


sns.countplot(y="duo", data=turnover, order = turnover['duo'].value_counts().index)


# In[385]:


ct = pd.crosstab(turnover['duo'],turnover['Season'])
ax = sns.heatmap(ct, annot=True, fmt="d",cmap="YlGnBu").set_title('Number of Turnovers Per Season')


# In[386]:


turnover_rate2 = turn.div(turn.sum(1).astype(float),axis=0)*100 
turnover_rate2.reset_index(level=0, inplace=True)
del turnover_rate2[False]
turnover_rate2.columns.values[1] = 'turnover_rate' 
turnover_rate2['turnover_rank'] = turnover_rate2['turnover_rate'].rank(ascending=True)
turnover_rate2 


# In[387]:


conversion_rate2 = conv.div(turn.sum(1).astype(float),axis=0)*100 
conversion_rate2.reset_index(level=0, inplace=True)
del conversion_rate2[False]
conversion_rate2.columns.values[1] = 'conversion_rate' 
conversion_rate2['conversion_rank'] = conversion_rate2['conversion_rate'].rank(ascending=False)
conversion_rate2 


# In[388]:


scoring_rate2 = score.div(turn.sum(1).astype(float),axis=0)*100 
scoring_rate2.reset_index(level=0, inplace=True)
del scoring_rate2[False]
scoring_rate2.columns.values[1] = 'scoring_rate' 
scoring_rate2['scoring_rank'] = scoring_rate2['scoring_rate'].rank(ascending=False)
scoring_rate2 


# In[389]:


gr = df2.groupby('duo').mean()
df_small = gr[['ydsnet', 'AirYards', 'YardsAfterCatch']]
df_small['ydsnet_rank'] = df_small['ydsnet'].rank(ascending=False)
df_small['AirYards_rank'] = df_small['AirYards'].rank(ascending=False)
df_small['YardsAfterCatch_rank'] = df_small['YardsAfterCatch'].rank(ascending=False)
df_small.reset_index(level=0, inplace=True)
df_small


# In[390]:


from functools import reduce
dfs = [df_small, turnover_rate2, conversion_rate2, scoring_rate2]
df_final = reduce(lambda left,right: pd.merge(left,right,on='duo'), dfs)
df_final['Average Rank'] = (df_final['ydsnet_rank'] + df_final['AirYards_rank'] + df_final['YardsAfterCatch_rank'] + 
                            df_final['turnover_rank'] + df_final['conversion_rank'] + df_final['scoring_rank'])/6
df_final


# In[391]:


df_final[['duo','Average Rank']].sort_values('Average Rank')

