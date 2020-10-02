#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# Here I will try to explore fantasy basketball data and create player pricing algorithms using linear regression and random forest. Dataset contains:
# 
# players.json
# -  File format: JSON
# - File contains metadata, fantasy scores and prices for players in Euroleague
# - JSON Tree is provided at the end
# 
# players_metadata.csv
# - File format: csv
# - File contains position and nationality for each player

# ####   import necessary modules

# In[ ]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# ####  read json file

# In[ ]:


data=pd.read_json("../input/players.json")
data.head()


# #### Lets look a bit more in detail to the stats column

# In[ ]:


stats=data.stats.apply(pd.Series)
stats['player_id']=data.id
stats.head(3)


# ####   Describe&visualize player costs, avg_points, selections :

# In[ ]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"  #to display more than one output
data.cost.describe().astype(int)
stats.avg_points.describe() 
stats.selections.describe()
df_2=pd.DataFrame()
df_2['cost']=data.cost
df_2['avg_points']=stats.avg_points
df_2['selections']=stats.selections
sns.set()
sns.pairplot(df_2)


# -  avg_points seems to correlated with player cost

# ####  Let's read csv file

# In[ ]:


player_df=pd.read_csv("../input/players_metadata.csv",delimiter="|")
player_df.head(3)


# ####  to match names, first upper first and last names in data and combine them:

# In[ ]:


data['Player_Name']=data.last_name.str.upper()+", "+data.first_name.str.upper()
data['avg_points']=stats.avg_points
data.head(3)


# ####  now I can merge 

# In[ ]:


merged_df=pd.merge(data,player_df,how='inner',left_on='Player_Name',right_on='Player_Name')
merged_df.head(3)


# #### most common 3 nationalities among players and boxplots for avg_points for each type of position for the top-3 nationalities are:

# In[ ]:


merged_df.groupby('nationality').size().sort_values(ascending=False).head(3)
top3=merged_df[merged_df.nationality.isin(['United States of America','Serbia','Spain'])]
sns.boxplot(x='nationality',y='avg_points',hue='position',data=top3)


# #### Now I want to extract scores, price change and percentage price change for each player-round pair. I will also create columns for previous round score and price: 

# In[ ]:


player_list=[]
round_list=[]
price_list=[]
price_change_list=[]
percantage_change_list=[]
score_list=[]
prevscore_list=[]
prevprice_list=[]
for i in range(len(stats.prices)):
    player=stats.player_id[i]
    for key,value in stats.prices[i].items():
        round=int(key)
        try:
            price=stats.prices[i][str(round)]
            price_change=stats.prices[i][str(round+1)]-stats.prices[i][str(round)]
            percentage_change=(stats.prices[i][str(round+1)]-stats.prices[i][str(round)])/stats.prices[i][str(round)]
        except KeyError:
            price=0
            price_change=0
            percentage_change=0;
        player_list.append(player)
        round_list.append(round)
        price_list.append(price)
        price_change_list.append(price_change)
        percantage_change_list.append(percentage_change)
        try:
            score=stats.scores[i][str(round)]
        except KeyError:
            score=0
        score_list.append(score)
        try:
            prevscore=stats.scores[i][str(round-1)]
            prevprice=stats.prices[i][str(round-1)]
        except KeyError:
            prevscore=0
            prevprice=0
        prevscore_list.append(prevscore)
        prevprice_list.append(prevprice)
        
change=pd.DataFrame({'player_id':player_list,'rounds':round_list,'score':score_list,'price':price_list,'price_change':price_change_list,'percentage_change':percantage_change_list,'prevscore':prevscore_list,'prevprice':prevprice_list})
change[0:7]
 


# #### Let's visualize first 5 players price changes

# In[ ]:


sns.relplot(x="rounds", y="price", 
            hue="player_id", 
            facet_kws=dict(sharex=False),
            kind="line", legend="full", data=change[change.rounds<=30][change.player_id<=5])
plt.show()


# #### I want to predict player price. I take rounds 2-25 as train set and remaining rounds as test set. I will train model with player_id, rounds, prevscore and prevprice

# In[ ]:


train_df=change[change.rounds>1][change.rounds<=25].drop(['price','price_change','percentage_change','score'],axis=1)
target=change[change.rounds>1][change.rounds<=25].price
test_df=change[change.rounds>25][change.rounds<34].drop(['price','price_change','percentage_change','score'],axis=1)
test_target=change[change.rounds>25][change.rounds<34].price
train_df.head()


# #### linear regression gives:

# In[ ]:


import statsmodels.formula.api as sm
result = sm.ols(formula='target ~ rounds+ prevscore + prevprice+ C(player_id)', data=train_df).fit()
predict=result.predict(test_df)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(test_target, predict))


# #### Random Forest has better accuracy:

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=500, random_state=1)
model.fit(train_df, target)
preds = model.predict(test_df)
print(mean_absolute_error(test_target, preds))

