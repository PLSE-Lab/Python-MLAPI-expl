#!/usr/bin/env python
# coding: utf-8

# Exploration of How Social Media Can Predict Winning Metrics Better Than Salary

# In[1]:


import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
color = sns.color_palette()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from ggplot import *


# In[2]:


attendance_valuation_elo_df = pd.read_csv("../input/nba_2017_att_val_elo.csv");attendance_valuation_elo_df.head()


# In[3]:


#attendence,values,and Areas

ax = sns.lmplot(x="AVG", y="VALUE_MILLIONS", data=attendance_valuation_elo_df, hue="CONF", size=12)
ax.set(xlabel="Attendence", ylabel="Team Valuation", title="NBA Players 2016-2017:  Team Valuation, Attendence and Areas")


# In[4]:


#ELO,values,and Areas
p = ggplot(attendance_valuation_elo_df,aes(y="VALUE_MILLIONS", x="ELO", color="CONF")) + geom_point(size=200)
p + ylab("Team Valuation") + xlab("ELO") + ggtitle("NBA Players 2016-2017:  Team Valuation, ELO and Areas")+stat_smooth()


# In[5]:


#ELO,attendence,and Areas
ax = sns.lmplot(x="ELO", y="AVG", data=attendance_valuation_elo_df, hue="CONF", size=12)
ax.set(xlabel='ELO Score', ylabel='Average Attendance Per Game', title="NBA Team AVG Attendance vs ELO Ranking:  2016-2017 Season")


# In[6]:


salary_df = pd.read_csv("../input/nba_2017_salary.csv");salary_df.head()


# In[7]:


pie_df = pd.read_csv("../input/nba_2017_pie.csv");pie_df.head()


# In[8]:


plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");plus_minus_df.head()


# In[9]:


br_stats_df = pd.read_csv("../input/nba_2017_br.csv");br_stats_df.head()


# In[10]:



plus_minus_df.rename(columns={"NAME":"PLAYER", "WINS": "WINS_RPM"}, inplace=True)
players = []
for player in plus_minus_df["PLAYER"]:
    plyr, _ = player.split(",")
    players.append(plyr)

plus_minus_df["PLAYER"] = players
plus_minus_df.head()


# In[11]:



nba_players_df = br_stats_df.copy()
nba_players_df.rename(columns={'Player': 'PLAYER','Pos':'POSITION', 'Tm': "TEAM", 'Age': 'AGE', "PS/G": "POINTS"}, inplace=True)
nba_players_df.drop(["G", "GS", "TEAM"], inplace=True, axis=1)
nba_players_df = nba_players_df.merge(plus_minus_df, how="inner", on="PLAYER")
nba_players_df.head()


# In[12]:



pie_df_subset = pie_df[["PLAYER", "PIE", "PACE", "W"]].copy()
nba_players_df = nba_players_df.merge(pie_df_subset, how="inner", on="PLAYER")
nba_players_df.head()


# In[13]:


salary_df.rename(columns={'NAME': 'PLAYER'}, inplace=True)
salary_df["SALARY_MILLIONS"] = round(salary_df["SALARY"]/1000000, 2)
salary_df.drop(["POSITION","TEAM", "SALARY"], inplace=True, axis=1)
salary_df.head()


# In[14]:


diff = list(set(nba_players_df["PLAYER"].values.tolist()) - set(salary_df["PLAYER"].values.tolist()))


# In[15]:


len(diff)


# In[16]:



nba_players_with_salary_df = nba_players_df.merge(salary_df); nba_players_with_salary_df.head()


# In[17]:



plt.subplots(figsize=(30,15))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY)")
corr = nba_players_with_salary_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns,annot =True)


# In[ ]:





# In[18]:


sns.lmplot(x="SALARY_MILLIONS", y="WINS_RPM", data=nba_players_with_salary_df)


# In[19]:


results = smf.ols('W ~POINTS', data=nba_players_with_salary_df).fit()


# In[20]:


print(results.summary())


# In[21]:


results = smf.ols('W ~WINS_RPM', data=nba_players_with_salary_df).fit()


# In[22]:


print(results.summary())


# In[23]:


results = smf.ols('SALARY_MILLIONS ~POINTS', data=nba_players_with_salary_df).fit()


# In[24]:


print(results.summary())


# In[25]:


results = smf.ols('SALARY_MILLIONS ~WINS_RPM', data=nba_players_with_salary_df).fit()


# In[26]:


print(results.summary())


# In[ ]:





# In[27]:



p = ggplot(nba_players_with_salary_df,aes(x="POINTS", y="WINS_RPM", color="SALARY_MILLIONS")) + geom_point(size=200)
p + xlab("POINTS/GAME") + ylab("WINS/RPM") + ggtitle("NBA Players 2016-2017:  POINTS/GAME, WINS REAL PLUS MINUS and SALARY")


# In[28]:


wiki_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv");wiki_df.head()
wiki_df.info()


# In[29]:


wiki_df.rename(columns={'names': 'PLAYER', "pageviews": "PAGEVIEWS"}, inplace=True)


# In[30]:


median_wiki_df = wiki_df.groupby("PLAYER").median()


# In[31]:



median_wiki_df_small = median_wiki_df[["PAGEVIEWS"]]


# In[32]:


median_wiki_df_small = median_wiki_df_small.reset_index()


# In[33]:


nba_players_with_salary_wiki_df = nba_players_with_salary_df.merge(median_wiki_df_small)
nba_players_with_salary_wiki_df.head()
nba_players_with_salary_wiki_df.info()


# In[34]:


twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");twitter_df.head()


# In[35]:


nba_players_with_salary_wiki_twitter_df = nba_players_with_salary_wiki_df.merge(twitter_df)


# In[36]:


nba_players_with_salary_wiki_twitter_df.head()
nba_players_with_salary_wiki_twitter_df.info()


# In[37]:



plt.subplots(figsize=(30,15))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY & TWITTER & WIKIPEDIA)")
corr = nba_players_with_salary_wiki_twitter_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,annot=True)


# In[38]:


#Determimne whether social media influence increases as salary increases and as wins_rpm increases.
#Also determine whether higher salary means higher wins_rpm or higher social media influence means higher wins_rpm.
#color represents social influence, darker means higher.
sns.lmplot( x="SALARY_MILLIONS", y='WINS_RPM', data=nba_players_with_salary_wiki_twitter_df, fit_reg=False, hue='PAGEVIEWS', legend=False, palette="Blues")
sns.lmplot( x="SALARY_MILLIONS", y='WINS_RPM', data=nba_players_with_salary_wiki_twitter_df, fit_reg=False, hue='TWITTER_FAVORITE_COUNT', legend=False, palette="Blues")


# In[39]:


#salary distribution per team
plt.subplots(figsize=(15,15))
df = nba_players_with_salary_wiki_twitter_df
sns.boxplot( x=df["SALARY_MILLIONS"], y=df["TEAM"] )


# In[40]:


#pageviews distribution per team
plt.subplots(figsize=(15,15))
df = nba_players_with_salary_wiki_twitter_df
sns.boxplot( x=df["PAGEVIEWS"], y=df['TEAM'] )


# In[41]:


#k-means for players
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
 
#Only cluster on these values
nba_players_with_salary_wiki_twitter_df_new = nba_players_with_salary_wiki_twitter_df.dropna(axis=0,how="any")
numerical_df = nba_players_with_salary_wiki_twitter_df_new.loc[:,["SALARY_MILLIONS", 'WINS_RPM', 'PAGEVIEWS', 'TWITTER_FAVORITE_COUNT','TWITTER_RETWEET_COUNT']]
 
#Scale to between 0 and 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit(numerical_df))
print("-----------------------------------")
print(scaler.transform(numerical_df))
 


# In[42]:


#Add back to DataFrame
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3)
kmeans = k_means.fit(scaler.transform(numerical_df))
nba_players_with_salary_wiki_twitter_df_new['cluster'] = kmeans.labels_
nba_players_cluster = nba_players_with_salary_wiki_twitter_df_new.loc[:,['Rk','PLAYER',"SALARY_MILLIONS", 'WINS_RPM', 'PAGEVIEWS', 'TWITTER_FAVORITE_COUNT','TWITTER_RETWEET_COUNT','cluster']]
nba_players_cluster.head()


# In[43]:


for i in [0,1,2]:
    nba_players_cluster[nba_players_cluster['cluster'] == i].head()


# In[44]:


#since we include 3 columns related to social media, the result above shows high priority to social influence.
#let's just use one metric to represent social power.
#Only cluster on these values
nba_players_with_salary_wiki_twitter_df_new = nba_players_with_salary_wiki_twitter_df.dropna(axis=0,how="any")
numerical_df = nba_players_with_salary_wiki_twitter_df_new.loc[:,["SALARY_MILLIONS", 'WINS_RPM', 'PAGEVIEWS']]
 
#Scale to between 0 and 1

scaler = MinMaxScaler()
scaler.fit(numerical_df)
k_means = KMeans(n_clusters=3)
kmeans = k_means.fit(scaler.transform(numerical_df))
nba_players_with_salary_wiki_twitter_df_new['cluster'] = kmeans.labels_
nba_players_cluster = nba_players_with_salary_wiki_twitter_df_new.loc[:,['Rk','PLAYER',"SALARY_MILLIONS", 'WINS_RPM', 'PAGEVIEWS','cluster']]
for i in [0,1,2]:
    nba_players_cluster[nba_players_cluster['cluster'] == i].head()

