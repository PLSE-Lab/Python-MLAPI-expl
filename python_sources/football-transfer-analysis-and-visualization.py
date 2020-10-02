#!/usr/bin/env python
# coding: utf-8

# # Football Data Analysis & Visualization
# 
# <p>This is my first try at Data Analysis and Visualization. 
# Along with the analysis we will also visualize the data using different graphs.
# The questions which we will look into are:</p>
# 
# <p>Pandas are used for analysis of Data and for visualization, matplotlib and seaborn is used. Different graphs like barplot, scatterplot, lineplot, jointplot, regplot are used to visualize the data.</p>
# 
# Reference Link:
# 1. <a href= "https://www.kaggle.com/slehkyi/football-transfers-2000-2018">Kernel Referred 1</a>
# 2. <a href= "https://www.kaggle.com/dhandepp/transfer-analysis-and-prediction-beginner">Kernel Referred 2</a>
# 
# 
# 

# <p> We analyze:</p>
# 
# <b>A. Analyzing Leagues</b>
# 1. Top 5 Leagues by turnover in selling player
# 2. Top 5 Leagues by turnover in buying player
# 3. Profits of League
# 4. Summary
# 
# <b>B. Analyzing Clubs</b>
# 1. Top Sellers
# 2. Top Buyers
# 3. Profitable Clubs
# 4. Summary
# 
# <b>C. Positions</b>
# 1. Different Positions Bought
# 2. Individual Position and top teams buying that position
# 3. Top Buying Team for each position
# 4. Top Selling Team for each position
# 5. Summary of Top Buying and Top Selling Position for each position
# 
# <b>D. Teams and Transfer</b>
# 1. Highest Bought Player for each Team
# 2. Highest Sold Player for each Team
# 3. Summary
# 
# <b>E. Age Analysis w.r.t Transfer Fee</b>
# 1. Number of players for each age
# 2. Age and Tranfer relation represented by:
# i. Scatterplot
# ii. Jointplot
# iii. Regplot
# iv. Lineplot
# 
# <b>F. Seasons and Transfer Fee</b>
# 1. Total Tranfer Fee per Season
# 2. Total Tranfer Fee per Season for Top Leagues
# 3. Total Position Brought each Season in Premier League
# 
# <b>G. Arsenal Analysis (Because its my favourite club)</b>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[ ]:


df = pd.read_csv('../input/top-250-football-transfers-from-2000-to-2018/top250-00-19.csv')
df.head()


# #### One Player Named "Marzouq Al-Otaibi" has his age 0, so we replace that age with proper age of 25; the age at which he transferred 

# In[ ]:


df = df.replace({0:25})


# #### Replacing Positions with a more general position
# F: Forward
# M: Midfield
# D: Defender
# G: Goalkeeper

# In[ ]:


position_map = {'Right Winger': 'F','Centre-Forward':'F','Left Winger':'F','Centre-Back':'D','Central Midfield':
               'M','Attacking Midfield': 'M', 'Defensive Midfield': 'M','Second Striker': 'F', 'Goalkeeper': 'G',
               'Right-Back':'D','Left Midfield': 'M', 'Left-Back':'D','Right Midfield':'M','Forward':'F','Sweeper':'M',
               'Defender':'D','Midfielder':'M'}


# In[ ]:


df['New_position'] = pd.Series(df.Position.map(position_map), index = df.index)


# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<style type="text/css">\ntable.dataframe td, table.dataframe th {\n    border: 1px  black solid !important;\n  color: black !important;\n}\n</style>')


# ## Analyzing League

# ### Top 5 Leagues by turnover in Selling Player

# In[ ]:


league_from = df.groupby(['League_from'])['Transfer_fee'].sum()
top5sell_league = league_from.sort_values(ascending = False).head(5)
top5sell_league = top5sell_league / 1000000 
top5sell_league.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(10,4))
ax.bar(top5sell_league.index, top5sell_league.values, color='green')
ax.set_ylabel("$ millions", color='navy')
ax.set_xlabel("Top 5 Selling Leagues", color='navy')
plt.show


# ### Top 5 Leagues by turnover in Buying Player

# In[ ]:


league_to = df.groupby(['League_to'])['Transfer_fee'].sum()
top5buy_league = league_to.sort_values(ascending = False).head(5)
top5buy_league = top5buy_league/1000000
top5buy_league.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(10,4))
ax.bar(top5buy_league.index, top5buy_league.values, color='orange')
ax.set_ylabel("$ millions", color='navy')
ax.set_xlabel("Top 5 Buying Leagues", color='navy')
plt.show


# ### Summary 

# In[ ]:


diff_league = top5sell_league - top5buy_league  
diff_league = diff_league.sort_values(ascending = False)
diff_league.head()


# In[ ]:


league_summary = pd.concat([top5sell_league,top5buy_league],axis = 1)
league_summary['Diff'] = diff_league
new_columns = league_summary.columns.values
new_columns[[0,1]]=['sell','buy']
league_summary.columns = new_columns
league_summary.fillna(0)
league_summary.head(5)


# In[ ]:


a = league_summary.reset_index()
a.plot(x='index',y=['sell','buy'],kind='barh')
plt.title('Summary')
plt.ylabel('Top 5 Leagues')
plt.xlabel('$ in million')
plt.show()


# We observe that Premier League Teams certaining spends more money in buying then selling as compared to all leagues.
# Also Ligue 1 is the only among the top 5 who earned more from selling.

# ## Analyzing Clubs 

# ### Top Sellers 
# Total Money gained divided by Total Players sold

# In[ ]:


sellers = df.groupby(['Team_from'])['Transfer_fee'].sum()
total_sold = df.groupby(['Team_from'])['Transfer_fee'].count()
biggest_sellers = (sellers/1000000) / total_sold 
biggest_sellers = biggest_sellers.sort_values(ascending  = False).head()
biggest_sellers


# ### Top Buyers
# Total Money spent divided by Total Players bought

# In[ ]:


buyers = df.groupby(['Team_to'])['Transfer_fee'].sum()
total_sold = df.groupby(['Team_to'])['Transfer_fee'].count()
biggest_buyers = (buyers/1000000) / total_sold 
biggest_buyers = biggest_buyers.sort_values(ascending  = False).head()
biggest_buyers


# #### Top Sellers and Top Buyers graphically

# In[ ]:


plt.figure(figsize=(18,6))

plt.subplot(2,2,1)
plt.barh(biggest_sellers.index, biggest_sellers.values,color='grey')
plt.xlabel("Count of money gained by selling in $")
plt.ylabel("Teams name")

plt.subplot(2,2,2)
plt.barh(biggest_buyers.index, biggest_buyers.values)
plt.xlabel("Count of money spent by buying in $")
plt.ylabel("Teams name")

plt.show


# ### Summary

# In[ ]:


profit_club = sellers - buyers  
profit_club = profit_club.sort_values(ascending = False)
profit_club = profit_club.dropna()
profit_club = profit_club/1000000
top5profit = profit_club.head()
top5loss = profit_club.tail()
print(top5profit)
print()
print(top5loss)


# In[ ]:


plt.figure(figsize=(18,6))
plt.title("Top 5 and Bottom 5")

plt.subplot(2,2,1)
sns.barplot(top5profit.index,top5profit.values,palette = "Greens_r")
plt.xlabel("Top 5 Clubs")
plt.ylabel("$ in million")

plt.subplot(2,2,2)
sns.barplot(top5loss.index,top5loss.values,palette = "Blues")
plt.xlabel("Bottom 5 clubs")
plt.ylabel("$ in million")

plt.show()


# We can observe that from the top 5, 2 teams belong to "Portuguese league" and from bottom 5, 3 clubs belong in "Premier League"  

# ## Positions

# ### Different Position Bought 

# In[ ]:


each_position = df.Position.value_counts()
print(each_position)
fig, ax = plt.subplots(figsize=(35,15))
ax.bar(each_position.index, each_position.values)
ax.set_ylabel("Number of Players bought")


# No doubt that the demand from centre forward position is much greater than any position.

# Values by New Position

# In[ ]:


each_position_g = df.New_position.value_counts()
print(each_position_g)
fig, ax = plt.subplots(figsize=(15,6))
sns.barplot(each_position_g.index, each_position_g.values)
ax.set_ylabel("Number of Players bought")
ax.set_xlabel("Positions")


# ### Individual Position and Top Teams buying that position

# * Here we look into a Attacking Midfield

# In[ ]:


def Position_bought(pos):
    positions_bought = df.groupby(['Position'])['Team_to'].value_counts() 
    position = positions_bought.loc[pos]
    position_top5 = position.head(5)
    print(position_top5)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(position_top5.index, position_top5.values,color='Black')
    ax.set_xlabel("Teams")
    return ax.set_ylabel("Number of Players bought")

Position_bought('Attacking Midfield')


# Values by New Position

# In[ ]:


def Position_bought(pos):
    positions_bought = df.groupby(['New_position'])['Team_to'].value_counts() 
    position = positions_bought.loc[pos]
    position_top5 = position.head(5)
    print(position_top5)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(position_top5.index, position_top5.values)
    ax.set_xlabel("Teams")
    return ax.set_ylabel("Number of Players bought")

Position_bought('M') #Finding Midfields Bought


# ### Top Buying Team for each position

# In[ ]:


all_position_buy = df.groupby(['Position'])
position_team_buy = df.groupby(['Position'])['Team_to'].value_counts()
pos_list_buy=[]
team_list_buy=[]
count_list_buy=[]

for pos,name in all_position_buy:
    top_team_name = position_team_buy[pos].index[0]
    count = position_team_buy[pos][0]
    team_list_buy.append(top_team_name)
    count_list_buy.append(count)
    pos_list_buy.append(pos)
    
pos_team_buy = pd.DataFrame({'Position':pos_list_buy,'Team':team_list_buy,'Players Bought':count_list_buy})
pos_team_buy = pos_team_buy.set_index('Position')
pos_team_buy.sort_values(by='Players Bought',ascending = False,inplace = True)
pos_team_buy


# In[ ]:


fig, ax = plt.subplots(figsize=(30,10))
ax.bar(x = pos_team_buy.index,height = pos_team_buy['Players Bought'])
ax.set_ylabel("Number of Players bought")
ax.set_xticklabels(labels=(pos_team_buy.index +'\n'+ pos_team_buy['Team']), color='navy')
get_ipython().run_line_magic('pinfo', 'ser_xticklables')


# ### Top Selling Team for each position

# In[ ]:


#tells us which player position is sold by which team the most time 
all_position_sell = df.groupby(['Position'])
position_team_sell = df.groupby(['Position'])['Team_from'].value_counts()
pos_list_sell=[]
team_list_sell=[]
count_list_sell=[]

for pos,name in all_position_sell:
    top_team_name = position_team_sell[pos].index[0]
    count = position_team_sell[pos][0]
    team_list_sell.append(top_team_name)
    count_list_sell.append(count)
    pos_list_sell.append(pos)
    
pos_team_sell = pd.DataFrame({'Position':pos_list_sell,'Team':team_list_sell,'Players Sold':count_list_sell})
pos_team_sell = pos_team_sell.set_index('Position')
pos_team_sell.sort_values(by='Players Sold',ascending = False,inplace = True)
pos_team_sell


# In[ ]:


fig, ax = plt.subplots(figsize=(30,10))
ax.bar(x = pos_team_sell.index,height = pos_team_sell['Players Sold'])
ax.set_ylabel("Number of Players sold")
ax.set_xticklabels(labels=pos_team_sell.index +'\n'+ pos_team_sell['Team'], color='navy')
get_ipython().run_line_magic('pinfo', 'ser_xticklables')


# ### Summary 

# In[ ]:


positions_transfer_summary = pd.merge(pos_team_buy,pos_team_sell,on = "Position")
positions_transfer_summary = positions_transfer_summary.rename(columns={'Team_x': 'Team Bought','Team_y':'Team Sold'})
positions_transfer_summary


# ## Teams and  Transfer 

# ###  Highest Bought Player for each team

# In[ ]:


buy_filter = df.filter(['Team_to','Name','Transfer_fee','New_position'])
buy_filter = buy_filter.sort_values(['Team_to','Transfer_fee'],ascending = [True,False]).groupby(['Team_to']).first()
top1buy = buy_filter.sort_values('Transfer_fee',ascending = False)
top1buy = top1buy.reset_index()
top1buy = top1buy.rename(columns = {'Team_to':'Team'})
top1buy = top1buy.set_index('Team')
top1buy.head(10)


# ###  Highest Sold Player for each team

# In[ ]:


sell_filter = df.filter(['Team_from','Name','Transfer_fee','New_position'])
sell_filter = sell_filter.sort_values(['Team_from','Transfer_fee'],ascending = [True,False]).groupby(['Team_from']).first()
#top1sell = sell_filter.first()
top1sell = sell_filter.sort_values('Transfer_fee',ascending = False)
top1sell = top1sell.reset_index()
top1sell = top1sell.rename(columns = {'Team_from':'Team'})
top1sell = top1sell.set_index('Team')
top1sell.head(10)
#top1sell.head(10)
#type(top1sell)


# ### Summary 

# In[ ]:


transfer_summary = pd.merge(top1buy,top1sell,on='Team')
transfer_summary = transfer_summary.rename(columns={'Name_x': 'Player Bought','Name_y':'Player Sold', 
                                                    'Transfer_fee_x': 'Bought For','Transfer_fee_y':'Sold For'})
transfer_summary.head(5)


# Prices for Forward Position is certaininly greater and most costly for the teams to buy

# In[ ]:


a = transfer_summary.reset_index().head(10)
a.plot(x='Team',y=['Bought For','Sold For'],kind='barh')
plt.title('Summary for Cost of top Players Bought v/s top Players Sold')
plt.ylabel('Teams')
plt.xlabel('$ in million')
plt.show()


# ## Age Analysis w.r.t Transfer Fee 

# ### Number of Players for Each Age  

# In[ ]:


plt.figure(figsize=(15,7))
sns.countplot(x='Age',data=df)


# ### Age and Transfer Fee with: 

# #### i. ScatterPlot 

# In[ ]:


sns.scatterplot(x = df.Age, y = (df.Transfer_fee/1000000))


# #### ScatterPlot with Marginal Distribution of Age and Transfer Fee
# 
# Also Position wise separation

# In[ ]:


# Plot the data
plt.figure(figsize=(20,10))
plt.subplot(2,2,2)
sns.scatterplot(x = df.Age, y = (df.Transfer_fee/1000000),hue=df.New_position)
plt.title("Joint Distribution of Age and Transfer Fee")

# Plot the Marginal X Distribution
plt.subplot(2,2,4)
plt.hist(x = df.Age, bins = 15,edgecolor='black', linewidth=1.2)
plt.title("Marginal Distribution of Age")


# Plot the Marginal Y Distribution
plt.subplot(2,2,1)
plt.hist(x = (df.Transfer_fee/1000000),orientation = "horizontal",bins = 50,edgecolor='black', linewidth=1)
plt.title("Marginal Distribution of Transfer Fee")

# Show the plots
plt.show()


# #### ii. JointPlot 

# In[ ]:


sns.jointplot(x= (df.Transfer_fee/1000000),  y=df.Age)


# #### iii. regplot

# In[ ]:


sns.regplot(x=df.Age, y=(df.Transfer_fee/1000000), data=df, fit_reg=False, scatter_kws={"alpha": 0.2})


# #### iv. Lineplot

# In[ ]:


agetransfer = df.groupby('Age')['Transfer_fee'].agg('mean')
agetransfer.plot()


# ## Season and Tranfer Fee 

# ### Total tranfer fee per season

# In[ ]:


season_wise = df.groupby('Season')['Transfer_fee'].agg('sum').reset_index()
plt.figure(figsize=(20,10))
sns.lineplot(x='Season', y='Transfer_fee', data = season_wise)


# ### Total Tranfer Fee per Season for Top Leagues

# In[ ]:


season_leagues = df.groupby(['League_to','Season'])['Transfer_fee'].agg('sum').reset_index()
select_leagues = ['Premier League','Serie A','LaLiga','Ligue 1','1.Bundesliga']
season_top5league = season_leagues.loc[season_leagues['League_to'].isin(select_leagues)]
plt.figure(figsize=(20,10))
#sns.lineplot(x='Season', y=(aa.Transfer_fee/10000000), data = aa,color='black')
ax = sns.lineplot(x='Season', y=(season_top5league.Transfer_fee/10000000),hue='League_to', data = season_top5league)
ax.set(xlabel="Top 5 Leagues Spending over 18 years", ylabel = "Transfer Fee")
plt.show()


# ### Total Position Brought each Season in Premier League

# In[ ]:


season_league = df.groupby(['League_to','Season','New_position']).agg(
    {'New_position': 'count'}).rename(columns={'New_position':'C'}).reset_index()
pl_league = season_league.loc[season_league['League_to']=='Premier League']

plt.figure(figsize=(20,10))
ax = sns.barplot(x='Season', y='C',hue='New_position',data = pl_league)
ax.set(xlabel="Premier League Spending over 18 years", ylabel = "Transfer Fee")
plt.show()


# ## Arsenal Analysis
# 
# 1. Top Players Bought & Top Players Sold
# 2. Top Position Bought & Top Position Sold
# 3. Teams/Leagues to which players are sold & from which players are bought
# 4. Season wise buys and sell

# ### Separating out Arsenal Data 

# In[ ]:


arsenal = df.loc[(df['Team_to'] == 'Arsenal') | (df['Team_from'] == 'Arsenal') ]
arsenal.head()


# ### Separating out Players bought by Arsenal and Players sold by Arsenal 

# In[ ]:


arsenal_buy = arsenal.loc[arsenal['Team_to']=='Arsenal']
arsenal_sell = arsenal.loc[arsenal['Team_from']=='Arsenal']


# #### Top 5 Arsenal Buys 

# In[ ]:


top5buy_a = arsenal_buy[['Name','Position','Transfer_fee','Team_from']]
top5buy_a.sort_values(by='Transfer_fee',ascending = False).head()


# #### Top 5 Arsenal Players Sold 

# In[ ]:


top5sell_a = arsenal_sell[['Name','Position','Transfer_fee','Team_to']]
top5sell_a.sort_values(by='Transfer_fee',ascending = False).head()


# #### League and Team from which most players are bought

# In[ ]:


leaguebought_a = arsenal_buy.groupby('League_from')['Position'].agg('count')
leaguebought_a.sort_values(ascending = False).head()


# In[ ]:


teambought_a = arsenal_buy.groupby('Team_from')['Position'].agg('count')
teambought_a.sort_values(ascending = False).head()


# #### League and Team to which most players are sold

# In[ ]:


leaguesold_a = arsenal_sell.groupby('League_to')['Position'].agg('count')
leaguesold_a.sort_values(ascending = False).head()


# In[ ]:


teamsold_a = arsenal_sell.groupby('Team_to')['Position'].agg('count')
teamsold_a.sort_values(ascending = False).head()


# #### Season wise buy

# In[ ]:


season_wise_buy = arsenal_buy.groupby('Season')['Name'].agg('count')
a = season_wise_buy.reset_index()

season_wise_transfer = arsenal_buy.groupby('Season')['Transfer_fee'].agg('sum')
b = season_wise_transfer.reset_index()

c = pd.merge(a,b,on='Season')
c = c.rename(columns = {'Name':'Players_bought','Transfer_fee':'Total_Spent'})

plt.figure(figsize=(20,10))
plt.title("Total Money Spent per Season with number of players bought")
g = sns.barplot("Season",(c.Total_Spent/1000000),data = c)
for index, row in c.iterrows():
    g.text(row.name,row.Players_bought, round(row.Players_bought,2), color='black', ha="center")
plt.show()


# In[ ]:


season_wise_sell = arsenal_sell.groupby('Season')['Name'].agg('count')
x = season_wise_sell.reset_index()

season_wise_transfer_sell = arsenal_sell.groupby('Season')['Transfer_fee'].agg('sum')
y = season_wise_transfer_sell.reset_index()

z = pd.merge(x,y,on='Season')
z = z.rename(columns = {'Name':'Players_sold','Transfer_fee':'Total_gained'})

plt.figure(figsize=(20,10))
plt.title("Total Money Gained per Season with number of players sold")
h = sns.barplot("Season",(z.Total_gained/1000000),data = z)
for index, row in z.iterrows():
    h.text(row.name,row.Players_sold, round(row.Players_sold,2), color='black', ha="center")
plt.show()


# In[ ]:


final_merge = pd.merge(c,z,on="Season")
final_merge['Profit(+)/Loss(-)'] = final_merge['Total_gained'] - final_merge['Total_Spent']
final_merge

