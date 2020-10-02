#!/usr/bin/env python
# coding: utf-8

# # Gotta Analize'Em all!
# ![](https://geekandsundry.com/wp-content/uploads/2017/06/rsz_landscape-1456483171-pokemon2.jpg)

# In[15]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import os
import itertools
print(os.listdir("../input/pokemon-challenge/"))
plt.style.use('ggplot')


# # Reading data and some basic descriptive statistics

# In[2]:


combats = pd.read_csv('../input/pokemon-challenge/combats.csv')
display(combats.head())


# ## Feature extraction: number of victories for each pokemon

# In[3]:


wins_by_pokemon = combats.groupby('Winner')[['First_pokemon']].count().rename(columns={'First_pokemon': 'num_wins'})
wins_by_pokemon.head()


# # Pokemon dataset
# - Use *#* (pokemon id) as index 
# - Attach *num_wins* extracted above
# - More EDA after

# In[4]:


pokemon = pd.read_csv('../input/pokemon-challenge/pokemon.csv')
pokemon.set_index('#', inplace=True)
display(pokemon.describe())
pokemon['num_wins'] = wins_by_pokemon.num_wins
pokemon['Generation'] = pokemon['Generation'].astype('category')
pokemon.head()


# In[5]:


pokemon.head()


# # Correlation between pokemon stats and num_wins

# In[8]:


numdf = pokemon.select_dtypes(exclude=['object', 'category', 'bool'])
plt.figure(figsize=(15,10))
sns.heatmap(numdf.corr(), cbar=True, annot=True, square=True, fmt='.2f')
plt.show()


# We can note that speed has a good correlation with num_wins

# # Scatter with regression line on each pokemon stat in relation to *num_wins*

# In[28]:


y_col = 'num_wins'
X_cols = list(set(numdf.columns) - {y_col})
fig, axes = plt.subplots(ncols=2, nrows=(len(X_cols)//2 + len(X_cols)%2),figsize=(15,15))
for X_col, ax in zip(X_cols, itertools.chain(*axes)):
    sns.regplot(x=X_col, y=y_col, data=pokemon, ax=ax)
plt.show()


# From this plot we can cleary see what pearson correlation matrix says before: we can cleary see that *Speed* stat at difference of others have a very evident crescent direction: when the speed increase also very probably the number of victories increase. <br> This is also the meaning of the pearson coefficient 0.91 between speed and *num_wins* calculated before

# # Imputing null values on pokemon 

# In[29]:


pokemon.isnull().sum()


# We can't say because there is that null value on *Name* but we can hypotize some reason for the null values in the other columns:
# - _Type 2_ contains 386 null values because many pokemon have only one type
# - *num_wins* contains null values from the pokemon that never win :(
# 
# Now let's examine the null value in _Name_
# 

# In[30]:


pokemon[pokemon.Name.isnull()]


# In[31]:


pokemon.iloc[60:70]


# From this second table we cleary recognize who is the missing name, it's Primeape!

# In[32]:


pokemon.loc[63, 'Name'] = 'Primeape'


# In[33]:


pokemon.loc[63]


# We solve also the num_wins null values: it is caused by pokemon that never wins!

# In[34]:


pokemon['num_wins'] = pokemon.num_wins.fillna(0)


# # Feature Extraction

# ## Battle wins by Type 1
# - We group by the *Type 1* column and we count the number of wins by each primary type
# - Another useful information is the number of victories for that type divided by the number of pokemon that have that type

# In[35]:


def count_wins_by(col):
    grouped = pokemon.groupby(col)[['num_wins']].agg(['sum', 'count'])
    grouped.columns = [' '.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.rename(columns={'num_wins sum': 'num_wins', 'num_wins count': 'num_pokemon'})
    grouped['normalized_num_wins'] = grouped['num_wins'] / grouped['num_pokemon']
    return grouped


# In[36]:


victorytype1 = count_wins_by('Type 1')
ax1 = victorytype1.reset_index().plot(x='Type 1', y='num_wins', kind='bar', title='Victory by pokemon main type', legend=False)
ax1.set_ylabel('#wins')
ax2 = victorytype1.reset_index().plot(x='Type 1', y='normalized_num_wins', kind='bar', title='Victory normalized with num of pokemon\n of the same type by pokemon main type', legend=False)
ax2.set_ylabel('#wins/#pokemon')
plt.show()


# ## Battle wins by either two types
# - Let's extract a new column that embed the two types of pokemon (if the second is present)
# - Because there are many combinations of types we consider only the most 20 most winning combination of classes

# In[37]:


pokemon['Type'] = pokemon['Type 1'] + pokemon['Type 2'].fillna('')
pokemon.head()


# In[38]:


victorytypes = count_wins_by('Type')
victorytypes = victorytypes.sort_values(by=['num_wins', 'normalized_num_wins'], ascending=False)
ax1 = victorytypes.iloc[:20, :].reset_index().plot(x='Type', y='num_wins', kind='bar', title='Victory by pokemon main type', legend=False)
ax1.set_ylabel('#wins')
ax2 = victorytypes.iloc[:20, :].reset_index().plot(x='Type', y='normalized_num_wins', kind='bar', title='Victory normalized with num of pokemon\n of the same type by pokemon main type', legend=False)
ax2.set_ylabel('#wins/#pokemon')
plt.show()


# > ## Join the features previuosly extracted in pokemon dataset

# In[39]:


pokemon = pokemon.join(victorytypes, on='Type', rsuffix='_Type')
pokemon = pokemon.join(victorytype1, on='Type 1', rsuffix='_Type1')
pokemon = pokemon.rename(columns={'normalized_num_wins': 'normalized_num_wins_Type', 'num_pokemon': 'num_pokemon_Type'})
pokemon.head()


# ## Evolutions
# - Load another dataset that contains the id of the previous evolution
# - We extract the feature that count the number of successive evolution of a pokemon
# - We extract the attribute *has_mega_evolution* for each pokemon because a pokemon that can megaevolves during battle have a great stat boots
# - We extract the attribute *is_mega* because a Mega pokemon has very high stats and powerful ability

# In[40]:


pokemon2 = pd.read_csv('../input/pokemon-evolutions/pokemon_species.csv')
pokemon2.set_index('id', inplace=True)
display(pokemon2.describe())
display(pokemon2.head())


# ### Count the number of possible evolutions
# - First we sort in a decreasing order the id of the pokemon which is evolved to simplify the calculus
# - We create a copy of pokemon dataset indexed by pokemon name called *pokemon_iname* 
# -  We do the same in *pokemon2* dataset and we map the column *evolves_from_species_id* to pokemon name
# - This is necessary because *pokemon* dataset has a strong difference in the id column: it contains also the mega pokemon: for example the pokemon #4 should be Charmender but it is Mega Venosaur in the *pokemon* dataset 

# In[41]:


evolved = pokemon2.sort_values(by='evolves_from_species_id', ascending=False).dropna(subset=['evolves_from_species_id'])
evolved.evolves_from_species_id.astype(dtype='int', inplace=True)
evolved.head()


# In[42]:


def id_to_name(id_pokemon):
    result =  pokemon2.loc[int(id_pokemon), 'identifier']
    assert isinstance(result, str), print(type(result), result)
    return result.capitalize().strip()


# In[43]:


if 'num_evolutions' in pokemon.columns:
    del pokemon['num_evolutions']
pokemon_iname = pokemon.set_index('Name')
pokemon_iname['num_evolutions'] = 0
evolved['identifier'] = evolved.identifier.str.capitalize().str.strip()
evolved_iname = evolved.set_index('identifier')
evolved_iname['evolves_from_species_id'] = evolved_iname['evolves_from_species_id'].transform(id_to_name)
for evolved_pokemon in evolved_iname.itertuples():
    if evolved_pokemon.evolves_from_species_id in pokemon_iname.index and evolved_pokemon.Index in pokemon_iname.index:
        pokemon_iname.loc[evolved_pokemon.evolves_from_species_id, 'num_evolutions']+= pokemon_iname.loc[evolved_pokemon.Index, 'num_evolutions'] + 1
pokemon = pokemon.join(pokemon_iname[['num_evolutions']], on='Name')
del pokemon_iname, evolved_iname


# In[44]:


pokemon.head()


# In[72]:


is_mega = pokemon.Name.str.startswith('Mega ')
has_megaevol_names = pokemon.loc[is_mega, 'Name'].transform(lambda name: name.split()[1])
print(has_megaevol_names[:5])
pokemon['is_mega'] = is_mega
pokemon['has_mega_evolution'] = pokemon['Name'].isin(has_megaevol_names)
pokemon.head(10).T


# # Some plots on evolutions

# Let's start with some basic descriptive statistics:

# In[67]:


pokemon[['num_evolutions']].describe()


# Who is the pokemon with 8 evolutions?

# In[66]:


pokemon.query('num_evolutions==8')


# It's Evee! (obliviously)<br>
# Now let's see how *num_evolutions* impact of victories

# In[62]:


pokemon.boxplot(by='num_evolutions', column='num_wins', figsize=(10, 10))


# Obliviously a Pokemon with 0 remaining evolutions has in mean more victories because probabilly is a pokemon on the last evolution, a legendary pokemon, a mega pokemon but also simply a weak pokemon with no evolutions (for example dunsprace). All this cases cause a large variance on pokemons with 0 evolutions but in mean a better chance to win

# In[79]:


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14,9))
pokemon.boxplot(by='Legendary', column='num_wins', figsize=(5,5), ax=axs[0])
pokemon.boxplot(by='is_mega', column='num_wins', figsize=(5,5), ax=axs[1])
pokemon.boxplot(by='has_mega_evolution', column='num_wins', figsize=(5,5), ax=axs[2])


# The first boxplot is oblivios and also the second but the third present an interesting similarity with the second plot. But we can't say with certainty that the pokemon have a megastone during battle and can megaevolve, maybe the more wins of pokemon that can megaevolve is by the power of the non-megaevolved pokemon (for example Charizard, Mewtwo, Lucario, ecc.)

# ## Other features:
# - *has_second_type*

# In[80]:


pokemon['has_second_type'] = ~pokemon['Type 2'].isna()
pokemon.head()


# # Feature transformation for the model
# - Remove unuseful columns for battle winner prediction
# - Trasformation from string to integers for categorical columns

# In[81]:


print(pokemon.columns)
display(pokemon.head())


# In[82]:


pokemon.drop(columns=['Name'], inplace=True)
catcols = ['Type 1', 'Type 2', 'Type' ]
pokemon[catcols] = pokemon[catcols].transform(lambda catcol: catcol.astype('category').cat.codes, axis=1)


# In[83]:


pokemon.head().T


# # Data normalization
# We map each column in a Normal distribution with 0 mean and 1 variance (N(0,1)). This is obtained by diving each value by the column mean divided by standard deviation

# In[84]:


numvars = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'num_wins', 'num_wins_Type', 'num_pokemon_Type', 'normalized_num_wins_Type', 'num_wins_Type1', 'num_pokemon_Type1', 'normalized_num_wins_Type1']
pokemon[numvars] = pokemon[numvars].transform(lambda col: (col - col.mean()) / (col.std()))


# # Correlation between old and new variables

# In[87]:


plt.figure(figsize=(15,10))
sns.heatmap(pokemon[numvars].corr(), cbar=True, annot=True, square=True, fmt='.2f')


# The only relevant correlation is between *num_wins_Type(1)* and *num_pokemon_Type(1)* so we remove *num_wins_Type(1)* because is more dependant from the dataset and there is also the column*normalized_num_wins_Type(1)* on the dataset

# In[ ]:


pokemon.drop(columns=['num_wins_Type', 'num_wins_Type1'], inplace=True)


# ### Convert bool columns to int

# In[ ]:


bool_cols = pokemon.select_dtypes('bool').columns
pokemon[bool_cols] = pokemon[bool_cols].transform(lambda col: col.astype('int'))


# In[ ]:


pokemon.head()


# # Data saving

# In[ ]:


pokemon.to_csv('pokemon_mikedev_preprocessed.csv')
combats.to_csv('combats.csv')
get_ipython().system('cp ../input/pokemon-challenge/tests.csv ./')


# # Preparing the dataset for predict the winner of a match
# 1. Tranform the Winner column to binary variable

# In[ ]:


# Check that every id of Winner is on the column First_pokemon or in Second_pokemon
assert ((combats.Winner == combats.First_pokemon) | (combats.Winner == combats.Second_pokemon)).all()
combats['Winner'] = (combats.Winner == combats.Second_pokemon).astype('int')
combats.head()


# - Join the Pokemon information on this table. <br>
# Because this operation involves only the X variables we do all-in-one by concatenating the train and the test set

# In[ ]:


pokemon.head()


# In[ ]:


test = pd.read_csv('../input/pokemon-challenge/tests.csv')
display(test.head())


# In[ ]:


train_test = pd.concat([combats, test], join='inner')
train_len = len(combats)


# In[ ]:


train_test =     train_test        .merge(pokemon, left_on='First_pokemon', right_index=True, how='left')        .merge(pokemon, left_on='Second_pokemon', right_index=True, suffixes=('_first_pokemon', '_second_pokemon'), how='left')


# # Feature extraction from the battle dataset
# - ~~A type is super effective against another and the resistance~~  Not very informative because a pokemon can use moves of different type
# -  Stats difference

# In[ ]:


statcols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
for statcol in statcols:
    train_test[statcol + '_diff'] = train_test[statcol+'_first_pokemon'] - train_test[statcol+'_second_pokemon']


# ### Split into train and test

# In[ ]:


y = combats.Winner.values
combats = train_test.iloc[:train_len, :]
test = train_test.iloc[train_len:, :]
combats['Winner'] = y
display(combats.head())
display(test.head())
del train_test


# ## Save Battle dataset ready for prediction

# In[ ]:


combats.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

