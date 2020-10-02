#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')

#data processing

import numpy as np
import pandas as pd

#data visualization

import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[7]:


pokemon = pd.read_csv('../input/pokemon.csv')
combats = pd.read_csv('../input/combats.csv')


# In[8]:


display(pokemon.head())
display(pokemon.describe())
display(pokemon.shape)
pokemon.info()


# In[9]:


display(combats.head())
display(combats.describe())
display(combats.info())


# In[10]:


pokemon.loc[pokemon['Name'].isnull()==True]


# In[11]:


pokemon.loc[pokemon['Name']=='Abra']


# In[12]:


pokemon.loc[62 , 'Name']="Primeape"


# In[13]:


pokemon.loc[62]


# In[14]:


pokemon.columns
combats.columns


# In[15]:


pokemon['Total'] = pokemon['HP'] + pokemon['Attack'] + pokemon['Defense'] + pokemon['Sp. Atk'] + pokemon['Sp. Def'] + pokemon['Speed']


# In[16]:


cols = ['first' , '']


# In[17]:


no_total_dict = dict(zip(pokemon['#'],pokemon['Total']))
cols = ['First_pokemon' , 'Second_pokemon' , 'Winner']


# In[18]:


combat_dif = combats[cols].replace(no_total_dict)


# In[19]:


display(combat_dif.head())


# In[20]:


combats['Losers'] = combats.apply(lambda x: x['First_pokemon'] if x['First_pokemon'] !=x['Winner'] else x['Second_pokemon'],axis=1)


# In[21]:


combats.head()


# In[22]:


combats['is_first_win'] = combats['First_pokemon'] == combats['Winner']
combats['diff_stat'] = combat_dif['First_pokemon']-combat_dif['Second_pokemon']


# In[23]:


display(combats.head())


# In[24]:


#Merging the DataBase
nfirsts = combats['First_pokemon'].value_counts()
nseconds = combats['Second_pokemon'].value_counts()
nfights = nfirsts + nseconds


# In[ ]:





# In[25]:


fight_df = pd.DataFrame({
    'nfights':nfights,
    'nwins':combats['Winner'].value_counts()
},columns = ['nfights','nwins'])


# In[26]:


fight_df['Win_ratio'] = fight_df['nwins']/fight_df['nfights']


# In[27]:


fight_df = fight_df.sort_values(by='Win_ratio')
display(fight_df.tail())


# In[28]:


fight_df.info()
display(fight_df.loc[fight_df['Win_ratio'].isnull()])


# In[29]:


fight_df.loc[231,['nfights','nwins','Win_ratio']] = 0


# In[30]:


fight_df['#']= fight_df.index
pokemon_fight_df = pokemon.copy()
win_ratio_dict = dict(zip(fight_df['#'],fight_df['Win_ratio']))
pokemon_fight_df['Win_ratio'] = pokemon_fight_df['#'].replace(win_ratio_dict)
display(pokemon_fight_df.head())


# In[31]:


pokemon_fight_df.info()


# In[32]:


non_fight_pokemon_df = pokemon_fight_df.loc[(pokemon_fight_df['Win_ratio']>1) | pokemon_fight_df['Win_ratio'].isnull()]


# In[33]:


display(non_fight_pokemon_df)


# In[34]:


# Data of pokemon that have fight and those haveing fight_ratio <1
have_fight_df = pokemon_fight_df.loc[pokemon_fight_df['Win_ratio']<=1]
#Visualising th data 
sns.lmplot(x = 'Total' , y= 'Win_ratio' , data = have_fight_df)


# In[35]:


# as the data is in straight line so it can be predicted linearly using logistic regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[36]:


lr.fit(have_fight_df['Total'].values.reshape(-1,1),have_fight_df['Win_ratio'].values.reshape(-1,1))


# In[37]:


non_fight_pokemon_df['Win_ratio'] = lr.predict(non_fight_pokemon_df['Total'].values.reshape(-1,1)) 


# In[38]:


display(non_fight_pokemon_df)


# In[39]:


frames = [have_fight_df , non_fight_pokemon_df]
database_pokemon = pd.concat(frames)
database_pokemon.info()
database_pokemon.columns


# In[40]:


train_df = combats.copy()
hp_dict = dict(zip(database_pokemon['#'],database_pokemon['HP']))
attack_dict = dict(zip(database_pokemon['#'],database_pokemon['Attack']))
defense_dict = dict(zip(database_pokemon['#'],database_pokemon['Defense']))
sp_atk_dict = dict(zip(database_pokemon['#'],database_pokemon['Sp. Atk']))
sp_def_dict = dict(zip(database_pokemon['#'],database_pokemon['Sp. Def']))
speed_dict = dict(zip(database_pokemon['#'],database_pokemon['Speed']))
total_dict = dict(zip(database_pokemon['#'],database_pokemon['Total']))
win_ratio_dict = dict(zip(database_pokemon['#'],database_pokemon['Win_ratio']))


# In[41]:


train_df['first_hp'] =  train_df['First_pokemon'].replace(hp_dict)
train_df['first_attack'] =  train_df['First_pokemon'].replace(attack_dict)
train_df['first_defense'] =  train_df['First_pokemon'].replace(defense_dict)
train_df['first_sp_atk'] =  train_df['First_pokemon'].replace(sp_atk_dict)
train_df['first_sp_def'] =  train_df['First_pokemon'].replace(sp_def_dict)
train_df['first_speed'] =  train_df['First_pokemon'].replace(speed_dict)
train_df['first_total'] =  train_df['First_pokemon'].replace(total_dict)
train_df['first_win_ratio'] =  train_df['First_pokemon'].replace(win_ratio_dict)
train_df['second_hp'] =  train_df['Second_pokemon'].replace(hp_dict)
train_df['second_attack'] =  train_df['Second_pokemon'].replace(attack_dict)
train_df['second_defense'] =  train_df['Second_pokemon'].replace(defense_dict)
train_df['second_sp_atk'] =  train_df['Second_pokemon'].replace(sp_atk_dict)
train_df['second_sp_def'] =  train_df['Second_pokemon'].replace(sp_def_dict)
train_df['second_speed'] =  train_df['Second_pokemon'].replace(speed_dict)
train_df['second_total'] =  train_df['Second_pokemon'].replace(total_dict)
train_df['second_win_ratio'] =  train_df['Second_pokemon'].replace(win_ratio_dict)
train_df.info()


# In[42]:


train_df['is_first_win'] = train_df.apply(lambda x: 1 if x['is_first_win']==True else 0, axis=1) 


# In[43]:


display(train_df.head())


# In[44]:


no_need_columns = ['First_pokemon', 'Second_pokemon', 'Winner', 'Losers']
train_df = train_df.drop(no_need_columns, axis=1)


# In[45]:


y = train_df['is_first_win']
no_need_columns = ['first_win_ratio','second_win_ratio','diff_stat','is_first_win']
# no_need_columns = ['diff_stat','is_first_win']
x =  train_df.drop(no_need_columns , axis=1)


# In[46]:


from sklearn.model_selection import train_test_split


# In[47]:


xtrain , xtest , ytrain , ytest = train_test_split(x , y , train_size = 0.8 , random_state= 15) 


# In[48]:


display(xtrain.info())


# In[49]:


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[ ]:





# In[50]:


lda = LinearDiscriminantAnalysis()
lda.fit(xtrain,ytrain)
round(lda.score(xtest,ytest)*100,2)


# In[51]:


ridge = Ridge()
ridge.fit(xtrain,ytrain)
round(ridge.score(xtest,ytest)*100 , 2)


# In[52]:


lasso = Lasso()
lasso.fit(xtrain,ytrain)
round(lasso.score(xtest,ytest)*100,3)


# In[53]:


rfc = RandomForestClassifier()
rfc.fit(xtrain,ytrain)
round(rfc.score(xtest , ytest)*100 , 2)


# In[54]:


dtc = DecisionTreeClassifier()
dtc.fit(xtrain,ytrain)
round(dtc.score(xtest , ytest)*100 , 2)


# In[55]:


lr = LogisticRegression()
lr.fit(xtrain,ytrain)
round(lr.score(xtest , ytest)*100 , 2)
# acc['Logistic Regression']


# In[56]:


per = Perceptron()
per.fit(xtrain,ytrain)
round(per.score(xtest , ytest)*100 , 2)


# In[57]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain,ytrain)
round(knn.score(xtest , ytest)*100 , 2)


# In[58]:


gnb = GaussianNB()
gnb.fit(xtrain,ytrain)
round(gnb.score(xtest,ytest)*100,3)


# In[60]:


test = pd.read_csv('../input/tests.csv')
test.info()


# In[61]:


test['first_hp'] =  test['First_pokemon'].replace(hp_dict)
test['first_attack'] =  test['First_pokemon'].replace(attack_dict)
test['first_defense'] =  test['First_pokemon'].replace(defense_dict)
test['first_sp_atk'] =  test['First_pokemon'].replace(sp_atk_dict)
test['first_sp_def'] =  test['First_pokemon'].replace(sp_def_dict)
test['first_speed'] =  test['First_pokemon'].replace(speed_dict)
test['first_total'] =  test['First_pokemon'].replace(total_dict)
test['second_hp'] =  test['Second_pokemon'].replace(hp_dict)
test['second_attack'] =  test['Second_pokemon'].replace(attack_dict)
test['second_defense'] =  test['Second_pokemon'].replace(defense_dict)
test['second_sp_atk'] =  test['Second_pokemon'].replace(sp_atk_dict)
test['second_sp_def'] =  test['Second_pokemon'].replace(sp_def_dict)
test['second_speed'] =  test['Second_pokemon'].replace(speed_dict)
test['second_total'] =  test['Second_pokemon'].replace(total_dict)


# In[62]:


display(test.info())


# In[63]:


no_need_columns = ['First_pokemon', 'Second_pokemon',]
test = test.drop(no_need_columns, axis=1)


# In[64]:


y = rfc.predict(test)


# In[65]:


output = pd.DataFrame({'Winner': y})


# In[68]:


output.to_csv('../Result.csv',index=False)


# In[ ]:




