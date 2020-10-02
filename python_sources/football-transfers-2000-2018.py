#!/usr/bin/env python
# coding: utf-8

# ## Intro

# In this kernel we will go through football transfer market and create a high level overview of what's happenning there. Some numbers will be relatively well known some might surprise. Anyway, it is goign to be interesting :)
# 
# As usual, we will start with importing python libraries to work with data and explore a little our dataset.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[ ]:


data = pd.read_csv('../input/top250-00-19.csv')


# In[ ]:


data.head()


# ## Find Top 5 Leagues by turnover in selling players

# In[ ]:


league_from = data.groupby(['League_from'])['Transfer_fee'].sum()
top5sell_league = league_from.sort_values(ascending=False).head(5)
top5sell_league = top5sell_league/1000000
top5sell_league.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(18,6))
ax.bar(top5sell_league.index, top5sell_league.values, color='orange')
ax.set_ylabel("$ millions", color='navy')
ax.set_yticklabels(labels=[i for i in range(0,8000, 1000)], color='navy')
ax.set_xticklabels(labels=top5sell_league.index, color='navy')


# ## Find Top 5 Leagues by turnover in buying players

# In[ ]:


league_to = data.groupby(['League_to'])['Transfer_fee'].sum()
top5buy_league = league_to.sort_values(ascending=False).head(5)
top5buy_league = top5buy_league/1000000
top5buy_league.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(18,6))
ax.bar(top5buy_league.index, top5buy_league.values, color='navy')
ax.set_ylabel("$ millions", color='black')
ax.set_yticklabels(labels=[i for i in range(0,16000, 2000)], color='black')
ax.set_xticklabels(labels=top5buy_league.index, color='black')


# ## Profits?

# In[ ]:


diff_league = top5sell_league - top5buy_league
diff_league = diff_league.sort_values(ascending=False)
diff_league.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(18,6))
ax.bar(diff_league.index, diff_league.values)
ax.set_ylabel("$ millions")


# It is not a new information that EPL has the biggest amount of money involved and that clubs can permit themselves to buy a lot as they gain money from other sources. What is really interesting is that clubs from French League1 in general make money on transfers. I actually thought that all of the numbers will be with the minus sign. 

# ## Summary

# In[ ]:


league_summary = pd.concat([top5sell_league, top5buy_league], axis=1)
league_summary = league_summary.assign(diff=diff_league)
new_columns = league_summary.columns.values
new_columns[[0, 1]] = ['sell', 'buy']
league_summary.columns = new_columns
league_summary.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(18,6))

sales = league_summary["sell"]
buys = league_summary["buy"]
x = league_summary.index
width=0.4
N = len(x)
loc = np.arange(N)

ax.bar(loc, sales, width, bottom=0, label="Sell")
ax.bar(loc+width, buys, width, bottom=0, label="Buy")

ax.set_title("Buys and sales by major leagues")
ax.legend()
ax.set_xticks(loc + width / 2)
ax.set_xticklabels(x)
ax.set_ylabel("$ millions")
ax.autoscale_view()


# ## Analyzing clubs
# ## Sellers

# In[ ]:


club_from_sum = data.groupby(['Team_from'])['Transfer_fee'].sum()
club_from_count = data.groupby(['Team_from'])['Transfer_fee'].count()
club_from_mean_price = (club_from_sum/1000000) / club_from_count


# In[ ]:


plt.figure(figsize=(18,6))
sellers_mean = club_from_mean_price.sort_values(ascending=False)[:20]
g = sns.barplot(sellers_mean.index, sellers_mean.values, palette="Greens_r")
g.set_title("Mean price of sold player per club")
g.set(ylabel="$ millions", xlabel="Team selling a player")
plt.xticks(rotation=90)


# ## Buyers

# In[ ]:


club_to_sum = data.groupby(['Team_to'])['Transfer_fee'].sum()
club_to_count = data.groupby(['Team_to'])['Transfer_fee'].count()
club_to_mean_price = (club_to_sum/1000000) / club_to_count


# In[ ]:


plt.figure(figsize=(18,6))
buy_mean = club_to_mean_price.sort_values(ascending=False)[:20]
g = sns.barplot(buy_mean.index, buy_mean.values, palette=sns.cubehelix_palette(20))
g.set_title("Mean price of bought player per club")
g.set(ylabel="$ millions", xlabel="Team buying a player")
plt.xticks(rotation=90)


# ## Profits?

# In[ ]:


diff_club = club_from_sum - club_to_sum
diff_club = diff_club.sort_values(ascending=False)
diff_club = diff_club.dropna()


# In[ ]:


diff_club = diff_club/1000000
diff_club.head(15)
# in millions


# In[ ]:


fig, ax = plt.subplots(figsize=(18,6))
make_money = diff_club.sort_values(ascending=False)[:10]
ax.bar(make_money.index, make_money.values, color="orange")
ax.set_title("Clubs that make money on transfer market")
ax.set_ylabel("$ millions")
ax.set_xticklabels(make_money.index, rotation=90)
# ax.autoscale_view()


# In[ ]:


diff_club.tail(15)
# in millions


# In[ ]:


fig, ax = plt.subplots(figsize=(18,6))
lose_money = diff_club.sort_values(ascending=True)[:10]
ax.bar(lose_money.index, lose_money.values, color="black")
ax.set_title("Clubs that lose money on transfer market")
ax.set_ylabel("$ millions")
ax.set_xticklabels(lose_money.index, rotation=90)
ax.autoscale_view()


# ## Summary Top and Bottom 15 for clubs
# ## Total sum of sales

# In[ ]:


club_from_sum = club_from_sum.sort_values(ascending=False)
club_from_sum = club_from_sum/1000000
club_from_sum.head(15)


# In[ ]:


plt.figure(figsize=(20,6))
g = sns.barplot(club_from_sum.head(15).index, club_from_sum.head(15).values, palette=sns.color_palette("hls", 15))
g.set_title("Top sales clubs")
g.set(ylabel="$, millions", xlabel="Team selling")


# In[ ]:


club_from_sum.tail(15)


# In[ ]:


plt.figure(figsize=(20,6))
g = sns.barplot(club_from_sum.tail(15).index, club_from_sum.tail(15).values, palette=sns.color_palette("Wistia_r", 15))
g.set_title("Bottom sales clubs")
g.set(ylabel="$, millions", xlabel="Team selling")


# ## Mean price of a sale

# In[ ]:


club_from_mean_price = club_from_mean_price.sort_values(ascending=False)
club_from_mean_price.head(15)
# in millions


# In[ ]:


plt.figure(figsize=(18,6))
g = sns.barplot(club_from_mean_price.head(15).index, club_from_mean_price.head(15).values, palette=sns.color_palette("YlGnBu_r", 15))
g.set_title("Mean price of a sale, Top")
g.set(ylabel="$, millions", xlabel="Team selling")


# In[ ]:


club_from_mean_price.tail(15)
# I should've put some minimum borderline of let's say 10 men moved


# In[ ]:


plt.figure(figsize=(18,6))
g = sns.barplot(club_from_mean_price.tail(15).index, club_from_mean_price.tail(15).values, palette=sns.color_palette("YlGnBu", 15))
g.set_title("Mean price of a sale, Bottom")
g.set(ylabel="$, millions", xlabel="Team selling")
plt.xticks(rotation=45)


# ## Total sum of buys

# In[ ]:


club_to_sum = club_to_sum.sort_values(ascending=False)
club_to_sum = club_to_sum/1000000
club_to_sum.head(15)


# In[ ]:


plt.figure(figsize=(18,6))
g = sns.barplot(club_to_sum.head(15).index, club_to_sum.head(15).values, palette=sns.color_palette("OrRd_r", 15))
g.set_title("Total historical spend on players, Top")
g.set(ylabel="$, millions", xlabel="Team buing")
plt.xticks(rotation=45)


# In[ ]:


club_to_sum.tail(15)


# In[ ]:


plt.figure(figsize=(18,6))
g = sns.barplot(club_to_sum.tail(15).index, club_to_sum.tail(15).values, palette=sns.color_palette("OrRd", 15))
g.set_title("Total historical spend on players, Top")
g.set(ylabel="$, millions", xlabel="Team buing")
plt.xticks(rotation=45)


# ## Mean price of a buy

# In[ ]:


club_to_mean_price = club_to_mean_price.sort_values(ascending=False)
club_to_mean_price.head(15)


# In[ ]:


plt.figure(figsize=(18,6))
g = sns.barplot(club_to_mean_price.head(15).index, club_to_mean_price.head(15).values, palette=sns.color_palette("magma", 15))
g.set_title("Mean price of a bought player, Top")
g.set(ylabel="$, millions", xlabel="Team buing")
plt.xticks(rotation=45)


# In[ ]:


club_to_mean_price.tail(15)


# In[ ]:


plt.figure(figsize=(18,6))
g = sns.barplot(club_to_mean_price.tail(15).index, club_to_mean_price.tail(15).values, palette=sns.color_palette("magma_r", 15))
g.set_title("Mean price of a bought player, Top")
g.set(ylabel="$, millions", xlabel="Team buing")
plt.xticks(rotation=45)


# ## Conclusions

# This is a basic analysis, but it already answers few interesting questions.
# 1. The richest league - English Premier League (obvious)
# 2. Clubs of which Top-5 league make money on transfer market? - Ligue1
# 3. The biggest sellers in the market, top5: Monaco, FC Porto, Real Madrid, Chelsea, Liverpool
# 4. The biggest buyers in the market, top5: Chelsea, Man City, Real Madrid, FC Barcelona, Man Utd
# 5. Clubs that sell expensive players on average: Athletic Bilbao, RB Leipzig, Monaco, FC Augsburg, FC Barcelona
# 6. Clubs that buy expensive players on average: SIPG, FC Barcelona, CC Yatai, Man Utd, Real Madrid
# 7. Clubs that made money during last 18 years on the transfer market: FC Porto, Benfica, Udinese Calcio, River Plate, Parma
# 8. Clubs that lost money (a lot of money actually): Man City, Chelsea, Man Utd, FC Barcelona, Paris SG
# 
# I've being looking at this summary for 10 minutes or so trying to make some smart conclusions and what did I understand? All the answers are logical if you know at least something about football. Big clubs with a lot of money buy best players and trying to win all the possible tropheys. Clubs out of top 5 leagues just trying to make money and portugeses do that the best. Also it is not that much about the money, but about the level of the league - best players go to better leagues and this is clear and completely logical.
# Although, I think I could find more interesting stuff in dynamics, by analyzing the market divided by periods. For example, we can see that English clubs spent a lot of money on transfers since 2000, but last 10 years Spanish clubs totally dominate Europe. Does it correlate with increased expenses on transfers? This could be a problem to resolve in the next kernel.
