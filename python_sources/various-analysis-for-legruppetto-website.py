#!/usr/bin/env python
# coding: utf-8

# **Quick analysis on men top divisions teams**
# 
# This first notebook will be focused on teams from men first and second division. CTM are excluded because one year is missing, and because they are a very heterogeneous block, from full professionnal teams to full amateurs  teams. WTT and CPT are all full-professionnal teams, and run the major races of the season.

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

df = pd.read_csv('/kaggle/input/uci-pro-road-cycling-dataset/UCIRiders0519_2x.csv', delimiter=',')


# In[ ]:


df_pcteams = df[df["Category"] == "PCT"]
df_wtteams = df[(df["Category"] == "WTT") | (df["Category"] == "PRO")]
df_pro = df_pcteams.append(df_wtteams)
df_pro.head(10)


# **Number of riders and teams**

# In[ ]:


all_by_year = {}
wtt_by_year = {}
pct_by_year = {}
for y in set(df_pro['Year']):
    all_by_year[y] = df_pro[df_pro['Year'] == y]
    wtt_by_year[y] = df_wtteams[df_pro['Year'] == y]
    pct_by_year[y] = df_pcteams[df_pro['Year'] == y]
for y in set(df_pro['Year']):
    print(y)
    print(str(len(all_by_year[y]))+"  <-  "+str(len(wtt_by_year[y]))+"  &  "+str(len(pct_by_year[y])))
    print('----------------------------------------')


# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure()
fig.suptitle("Number of racers in top categories by year")
x = sorted(set(df_pro['Year']))
nb_by_year_all = {}
nb_by_year_wt = {}
nb_by_year_cp = {}
for year in x:
    nb_by_year_all[year] = len(all_by_year[year])
    nb_by_year_wt[year] = len(wtt_by_year[year])
    nb_by_year_cp[year] = len(pct_by_year[year])
fig, ax = plt.subplots()
ax.plot(nb_by_year_all.keys(), nb_by_year_all.values(), label="All riders")
ax.plot(nb_by_year_wt.keys(), nb_by_year_wt.values(), label="WT riders")
ax.plot(nb_by_year_cp.keys(), nb_by_year_cp.values(), label="PC riders")
plt.legend()


# We can notice variations each year in the number of riders from the two top divisions, but it remains in a range between 900 and 1100 riders. Number of WT riders is almost constant (let's assume slightly decreasing) while there are stronger variations in the number of riders from PC teams. Thus, the overall riders curve shapes like the PC riders curve, they are strongly correlated.

# In[ ]:


nb_all_team_by_year = {}
nb_wt_team_by_year = {}
nb_pc_team_by_year = {}
for year in x:
    nb_all_team_by_year[year] = len(set(all_by_year[year]['Team Code']))
    nb_wt_team_by_year[year] = len(set(wtt_by_year[year]['Team Code']))
    nb_pc_team_by_year[year] = len(set(pct_by_year[year]['Team Code']))
fig2 = plt.figure()
fig2.suptitle("Number of teams in top categories by year")
fig2, ax2 = plt.subplots()
ax2.plot(nb_all_team_by_year.keys(), nb_all_team_by_year.values(), label="All teams")
ax2.plot(nb_wt_team_by_year.keys(), nb_wt_team_by_year.values(), label="WT teams")
ax2.plot(nb_pc_team_by_year.keys(), nb_pc_team_by_year.values(), label="PC teams")
plt.legend()


# 2008-2014 seemed to be a bad period for PC teams, with their number decreasing. It is on a growing tendancy since then.

# **Internationalisation**

# In[ ]:


countries_by_year = {}
for year in x:
    countries_by_year[year] = set(all_by_year[year]['Country'])
riders_by_country_by_year = {}
for year in x:
    riders_by_country_by_year[year] = all_by_year[year]['Country'].value_counts()
    print(year)
    print(riders_by_country_by_year[year].head(5))
    print("**************")


# The most represented countries are almost always the same: Italy, Spain, France, Belgium, Germany and Netherlands.
# Let's look how they evolved one from another.

# In[ ]:


fra = {}
bel = {}
ita = {}
esp = {}
ger = {}
ned = {}
for year in x:
    fra[year] = riders_by_country_by_year[year]['FRA']
    bel[year] = riders_by_country_by_year[year]['BEL']
    ita[year] = riders_by_country_by_year[year]['ITA']
    esp[year] = riders_by_country_by_year[year]['ESP']
    ger[year] = riders_by_country_by_year[year]['GER']
    ned[year] = riders_by_country_by_year[year]['NED']
fig3 = plt.figure()
fig3.suptitle("Number of riders from major countries")
fig3, ax3 = plt.subplots()
ax3.plot(fra.keys(), fra.values(), label="France")
ax3.plot(bel.keys(), bel.values(), label="Belgium")
ax3.plot(ita.keys(), ita.values(), label="Italy")
ax3.plot(esp.keys(), esp.values(), label="Spain")
ax3.plot(ger.keys(), ger.values(), label="Germany")
ax3.plot(ned.keys(), ned.values(), label="Netherlands")
plt.legend()


# In 2005, Italy and Spain were the most represented countries. But the number of their riders has strongly decreased since then. Meanwhile, numbers of riders from France and Belgium,who were quite fare from the lead kept constant. Thus, they now share the lead with Italy, while Spain is distanced. Behind, Netherlands, with its growing tendancy, took the 5th rank to Germany, whose tendancy is slightly decreasing.

# In[ ]:


continents_by_year = {}
for year in x:
    continents_by_year[year] = set(all_by_year[year]['Continent'])
riders_by_continent_by_year = {}
for year in x:
    riders_by_continent_by_year[year] = all_by_year[year]['Continent'].value_counts()
    print(year)
    print(riders_by_continent_by_year[year])
    print("**************")


# We can easily see that Europe provides really much more riders than any other continent in the two top divisions teams. So, we will let Europe apart and focus on these other continents.

# In[ ]:


ame = {}
oce = {}
asi = {}
afr = {}
for year in x:
    ame[year] = riders_by_continent_by_year[year]['AME']
    oce[year] = riders_by_continent_by_year[year]['OCE']
    asi[year] = riders_by_continent_by_year[year]['ASI']
    afr[year] = riders_by_continent_by_year[year]['AFR']
fig4 = plt.figure()
fig4.suptitle("Number of extra-european riders by continent")
fig4, ax4 = plt.subplots()
ax4.plot(ame.keys(), ame.values(), label="Americas")
ax4.plot(oce.keys(), oce.values(), label="Oceania")
ax4.plot(asi.keys(), asi.values(), label="Asia")
ax4.plot(afr.keys(), afr.values(), label="Africa")
plt.legend()


# A hierarchy seems to appear within the presence of riders from various continents in the top divisions teams. Americas, then Oceania, and the others below. Presence of riders from Americas and Oceania is on a growing tendancy, whereas Asia is in stagnation. Africa had a peek in 2013, when MTN-Qhubeka (today Dimension Data) become a PC team, and then a WT team.
