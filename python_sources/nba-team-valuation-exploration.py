#!/usr/bin/env python
# coding: utf-8

# 

# In[44]:



import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[45]:


attendance_df = pd.read_csv("../input/nba_2017_attendance.csv");attendance_df.head()


# In[46]:


endorsement_df = pd.read_csv("../input/nba_2017_endorsements.csv");endorsement_df.head()


# In[47]:


valuations_df = pd.read_csv("../input/nba_2017_team_valuations.csv");valuations_df.head()


# In[49]:


pie_df = pd.read_csv("../input/nba_2017_pie.csv");pie_df.head()


# In[50]:


plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");plus_minus_df.head()


# In[51]:


br_stats_df = pd.read_csv("../input/nba_2017_br.csv");br_stats_df.head()


# In[52]:


elo_df = pd.read_csv("../input/nba_2017_elo.csv");elo_df.head()


# In[53]:


attendance_valuation_df = attendance_df.merge(valuations_df, how="inner", on="TEAM")


# In[54]:


attendance_valuation_df.head()


# In[85]:


nba_2017_twitter_players_df = pd.read_csv("../input/nba_2017_twitter_players.csv");nba_2017_twitter_players_df.head(10)


# In[90]:


salary_df = pd.read_csv("../input/nba_2017_salary.csv");salary_df.head()


# In[91]:


#switch the name to player in salary
salary_df.rename(columns={'NAME': 'PLAYER'}, inplace=True); salary_df.head()


# In[92]:


salary_twitter_df = nba_2017_twitter_players_df.merge(salary_df, how="inner", on="PLAYER"); salary_twitter_df.head()


# EDA

# In[99]:


sns.kdeplot(salary_twitter_df["TWITTER_FAVORITE_COUNT"], color="mediumpurple", shade=True)
plt.show()


# In[93]:


sns.lmplot(x="TWITTER_FAVORITE_COUNT", y="SALARY", data=salary_twitter_df)
plt.show()


# In[95]:


plt.figure(figsize=(8,15))
sns.boxplot(x="SALARY", y="TEAM",data=salary_twitter_df, orient="h")
plt.show()


# In[96]:


plt.figure(figsize=(8,15))
sns.boxplot(x="TWITTER_FAVORITE_COUNT", y="TEAM",data=salary_twitter_df, orient="h")
plt.show()


# In[98]:


corr = salary_twitter_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[94]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"));sns.pairplot(salary_twitter_df, hue="TEAM")


# In[57]:


#ME
#joint distribution
with sns.axes_style('white'):
    sns.jointplot("PCT", "VALUE_MILLIONS", data=attendance_valuation_df, kind='hex')


# In[58]:



with sns.axes_style('white'):
    sns.jointplot("AVG", "VALUE_MILLIONS", data=attendance_valuation_df, kind='hex')


# In[59]:


corr = attendance_valuation_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[60]:


valuations = attendance_valuation_df.pivot("TEAM", "AVG", "VALUE_MILLIONS")


# In[61]:


plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Team AVG Attendance vs Valuation in Millions:  2016-2017 Season")
sns.heatmap(valuations,linewidths=.5, annot=True, fmt='g')


# In[62]:


results = smf.ols('VALUE_MILLIONS ~AVG', data=attendance_valuation_df).fit()


# In[63]:


print(results.summary())


# In[64]:


sns.residplot(y="VALUE_MILLIONS", x="AVG", data=attendance_valuation_df)


# In[65]:


attendance_valuation_elo_df = attendance_valuation_df.merge(elo_df, how="inner", on="TEAM")


# In[66]:


attendance_valuation_elo_df.head()


# In[67]:


corr_elo = attendance_valuation_elo_df.corr()
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Team Correlation Heatmap:  2016-2017 Season (ELO, AVG Attendance, VALUATION IN MILLIONS)")
sns.heatmap(corr_elo, 
            xticklabels=corr_elo.columns.values,
            yticklabels=corr_elo.columns.values)


# In[68]:


corr_elo


# In[69]:



ax = sns.lmplot(x="ELO", y="AVG", data=attendance_valuation_elo_df, hue="CONF", size=12)
ax.set(xlabel='ELO Score', ylabel='Average Attendence Per Game', title="NBA Team AVG Attendance vs ELO Ranking:  2016-2017 Season")


# In[70]:


attendance_valuation_elo_df.groupby("CONF")["ELO"].median()


# In[71]:


attendance_valuation_elo_df.groupby("CONF")["AVG"].median()


# In[72]:


results = smf.ols('AVG ~ELO', data=attendance_valuation_elo_df).fit()


# In[73]:


print(results.summary())


# In[74]:


from sklearn.cluster import KMeans


# In[75]:


k_means = KMeans(n_clusters=3)


# In[76]:


cluster_source = attendance_valuation_elo_df.loc[:,["AVG", "ELO", "VALUE_MILLIONS"]]


# In[77]:


kmeans = k_means.fit(cluster_source)


# In[78]:


attendance_valuation_elo_df['cluster'] = kmeans.labels_


# In[79]:


ax = sns.lmplot(x="ELO", y="AVG", data=attendance_valuation_elo_df,hue="cluster", size=12, fit_reg=False)
ax.set(xlabel='ELO Score', ylabel='Average Attendence Per Game', title="NBA Team AVG Attendance vs ELO Ranking Clustered on ELO, AVG, VALUE_MILLIONS:  2016-2017 Season")


# In[80]:


kmeans.__dict__


# In[81]:


kmeans.cluster_centers_


# In[82]:


cluster_1 = attendance_valuation_elo_df["cluster"] == 1


# In[83]:


attendance_valuation_elo_df[cluster_1]


# In[ ]:




