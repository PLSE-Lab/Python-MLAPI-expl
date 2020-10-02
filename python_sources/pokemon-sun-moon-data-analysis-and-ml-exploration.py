#!/usr/bin/env python
# coding: utf-8

# # Pokemon Sun/Moon Data Analysis and Machine Learning Exploration (WIP)
# 1. Introduction, Questions to Answer
# 2. Data Import and Cleaning
# 3. Feature Engineering
# 4. Exploratory Data Analysis
# 5. Machine Learning Models

# # 1. Introduction
# 
# In this ongoing open exploration of Pokemon data, I'm mainly interested in Pokemon stats, types, and the supposed "power creep" from Generation from Generation. When talking about a Pokemon's stats, I'm referring to the fact that each Pokemon species has a base numerical value for its Hit Points (HP), Attack, Defense, Special Attack, Special Defense, and Speed. These stats are used as a part of the in-game calculations that determine damage done and damage recieved in Pokemon battles. Generally, the higher the values, the more powerful the stat. As a shorthand, I will refer to these values together in the following format: HP/Attack/Defense/Special Attack/Special Defense/Speed. I will also look at the sum of these stats, called the Base Stat Total (BST).
# 
# Some questions to answer:
# * How much has the distribution of BST among Pokemon changed among non-legendary, fully evolved pokemon from Gen 1 to Gen 7?
# * How are pokemon stats distributed and has it changed over time? Some Pokemon, like Celebi, have stats that are flat across the board  (100/100/100/100/100/100), while some, like Kartana, have widely varying stats (59/181/131/59/31/109).
# * Which types have had the most additions in each generation from 2 - 7?

# # 2. Data Import and Cleaning

# In[1]:


# Import analysis and visualization packages
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Import data
pokemon = pd.read_csv('../input/pokemon.csv')
movesets = pd.read_csv('../input/movesets.csv')


# In[3]:


pokemon.info()


# There are 1061 entries, but the Sun/Moon Pokedex only contains 802 Pokemon. Inspecting the data shows that entries 802 onward contain information about different Pokemon formes. 

# In[4]:


pokemon.iloc[802:808]


# Some of these formes have the same base stat distribution as the original Pokemon. See Pikachu, for example. 

# In[5]:


pokemon[pokemon['species']=='Pikachu'][['ndex','species','forme','hp','attack','defense','spattack','spdefense','speed','total']]


# I'm interested in stat distribution in this analysis and I don't want to overcount Pokemon formes if they don't add any new information, so I will clean the data by removing duplicates where the species, ndex, and base stats are the same, keeping only the first instance.

# In[6]:


# Data cleaning
pokemon.drop_duplicates(subset=['species','ndex','hp','attack','defense','spattack','spdefense','speed','total'], keep='first', inplace=True)


# In[7]:


# Testing cleanup
print('Testing duplicate forme removal...')
# There should be 1 Pikachu with forme 'Pikachu'
print(pokemon[pokemon['species']=='Pikachu']['forme'] == 'Pikachu')
print(pokemon[pokemon['species']=='Pikachu'].shape[0] == 1)
# There should be 2 Raichu, regular and Alolan forme
print(pokemon[pokemon['species']=='Raichu'].shape[0] == 2)
# There should be 4 Deoxys
print(pokemon[pokemon['species']=='Deoxys'].shape[0] == 4)
# There should be 2 Rotom
print(pokemon[pokemon['species']=='Rotom'].shape[0] == 2)


# In[8]:


pokemon[pokemon['species']=='Pikachu'][['ndex','species','forme','hp','attack','defense','spattack','spdefense','speed','total']]


# In[9]:


#pokemon.iloc[880:][['id','species','hp','attack','defense','spattack','spdefense','speed','total','forme']]


# # 3. Feature Engineering

# The data already contains a feature called 'pre-evolution' that states a Pokemon's pre-evolved form. From this feature we can engineer a new feature called 'is_fully_evolved' that is 1 if the Pokemon has no evolutions (aside from Mega-evolutions) and 0 otherwise. 
# 
# The feature called 'forme' is prefixed with '(Mega' if the Pokemon's form is a Mega evolution. Mega evolutions have different stats than their base forms but we will want to distinguish Megas and sometimes exclude them, so we use the 'forme' feature to engineer an 'is_mega' feature.
# 
# We will engineer another useful but simple feature called 'is_forme' that just lets us know if the Pokemon's species name is the same as the name in the 'forme' column. If not, the Pokemon is some special version of its basic form. We have already discarded formes that have the same base stat distribution as their basic form. 
# 
# Again using the 'pre-evolution' feature, we'll engineer a feature called 'is_first_stage' that is 1 if the Pokemon has no pre-evolutions and 0 otherwise.

# In[10]:


n_pokemon = pokemon.shape[0]
is_fully_evolved = np.zeros(n_pokemon)
is_mega = np.zeros(n_pokemon)
is_forme = np.zeros(n_pokemon)

for i, species in enumerate(pokemon['species']):
    # Check if pokemon name is found in the pre-evolution list.
    # If it is not, then it must be fully evolved
    if pokemon[pokemon['pre-evolution'].isin([species])].shape[0] == 0:
        is_fully_evolved[i] = 1
        
    if len(pokemon['forme'].iloc[i].split()) > 1:
        if pokemon['forme'].iloc[i].split()[1] == '(Mega':
            is_mega[i] = 1
        
    if pokemon['species'].iloc[i] != pokemon['forme'].iloc[i]:
        is_forme[i] = 1


# In[11]:


pokemon['is_fully_evolved'] = is_fully_evolved


# In[12]:


pokemon['is_first_stage'] = pokemon['pre-evolution'].isnull()
pokemon['is_mega'] = is_mega
pokemon['is_forme'] = is_forme


# The weight is in the string format: 'X lbs'. We'll change this to a numeric value in lbs.
# 
# The height is in the string format: 'X'Y"'. We'll change this to a numeric value in feet.

# In[13]:


pokemon['weight'] = pokemon['weight'].apply(lambda x: float(x.split()[0]))


# In[14]:


def height_to_numeric(height):
    height = height.split("'")
    feet = float(height[0])
    inches = float(height[1][:2])
    return feet + (inches/12)
    


# In[15]:


pokemon['height'] = pokemon['height'].apply(height_to_numeric)


# We'll next create a feature called 'generation' that divides the data up by which generation the Pokemon is from. This is easily done with pokedex numbers.
# In addition, any Pokemon with a forme with the prefix '(Alola' must be from Generation 7 and any Mega pokemon must be from Generation 6.

# In[16]:


generation_limits = [151, 251, 386, 493, 649, 721, 807]
def generation(ndex):
    if 1 <= ndex <= 151:
        return 1
    elif 152 <= ndex <= 251:
        return 2
    elif 252 <= ndex <= 386:
        return 3
    elif 387 <= ndex <= 493:
        return 4
    elif 494 <= ndex <= 649:
        return 5
    elif 650 <= ndex <= 721:
        return 6
    elif 722 <= ndex <= 807:
        return 7


# In[17]:


pokemon['generation'] = pokemon['ndex'].apply(generation)


# In[18]:


for i in range(n_pokemon):
    if len(pokemon['forme'].iloc[i].split()) > 1:
        if pokemon['forme'].iloc[i].split()[1] == '(Alola':
            pokemon['generation'].iloc[i] = 7
        elif pokemon['forme'].iloc[i].split()[1] == '(Mega':
            pokemon['generation'].iloc[i] = 6


# In[19]:


pokemon[['ndex','species','forme','is_fully_evolved','is_first_stage','is_mega','is_forme','generation']].iloc[800:820]


# We'll bring in the movesets dataset to determine how many moves a Pokemon can learn in total. This number will be a new feature called 'moveset_size'.

# In[20]:


movesets.shape
moveset_size = np.zeros(n_pokemon)
for i in range(n_pokemon):
    current_forme = pokemon.iloc[i]['forme']
    # The set of 'formes' in the movesets dataframe sometimes only has the species name (ie 'Burmy')
    # rather than the full forme name (ie 'Burmy (Plant Cloak)'). So we need to check if this is the case
    # and split the forme name and take just its species name. 
    if movesets[movesets['forme'].isin([current_forme])]['forme'].shape[0] == 0:
        current_forme = current_forme.split()[0]
    if movesets[movesets['forme'].isin([current_forme])]['forme'].shape[0] != 0:
        current_set = movesets[movesets['forme']==current_forme]
        moveset_size[i] = current_set.dropna(axis=1).shape[1] - 3

pokemon['moveset_size'] = moveset_size


# In[21]:


pokemon[(pokemon['moveset_size']>=120)][['forme','total','moveset_size']]


# # 4. Exploratory Data Analysis (WIP)

# In[22]:


mean_fe_stats_bygen = pokemon[pokemon['is_fully_evolved']==1].groupby(by='generation').mean()


# In[23]:


sns.set_style('darkgrid')
sns.lmplot(x='generation', y='total', data=pokemon[(pokemon['is_fully_evolved']==1) & pokemon['is_mega']==0], aspect=1.3)
plt.title('Fully evolved Pokemon (excluding Megas)')
plt.ylabel('Base Stat Total')
yt = plt.yticks(range(100,850,50))


# In[24]:


fig = plt.figure(figsize=(10,6))
sns.violinplot(x='generation', y='total', data=pokemon[(pokemon['is_fully_evolved']==1) & pokemon['is_mega']==0],
              palette='Pastel1')
plt.title('Fully evolved Pokemon (excluding megas)')
yl = plt.ylabel('Base Stat Total')


# In[25]:


fig = plt.figure(figsize=(10,6))
sns.boxplot(x='generation', y='total', data=pokemon[(pokemon['is_fully_evolved']==1) & pokemon['is_mega']==0],
              palette='Pastel1', whis=0.5)
plt.title('Fully evolved Pokemon (excluding megas)')
yl = plt.ylabel('Base Stat Total')


# In[26]:


fig = plt.figure(figsize=(10,6))
bins = np.arange(160,800,20)
pokemon[(pokemon['is_first_stage']==1) & (pokemon['is_fully_evolved']==0)]['total'].plot.hist(bins=bins, color='grey', edgecolor='black', linewidth=1.2, alpha=0.5, normed=True, label='First stage')
pokemon[(pokemon['is_fully_evolved']==0) & (pokemon['is_first_stage']==0)]['total'].plot.hist(bins=bins, color='orange', edgecolor='black', linewidth=1.2, alpha=0.5, normed=True, label='Middle stage')
pokemon[(pokemon['is_fully_evolved']==1) & (pokemon['is_first_stage']==0) & (pokemon['is_mega']==0)]['total'].plot.hist(bins=bins, color='red', edgecolor='black', linewidth=1.2, alpha=0.5, normed=True, label='Fully evolved')
pokemon[(pokemon['is_mega']==1)]['total'].plot.hist(bins=bins, color='blue', edgecolor='black', linewidth=1.2, alpha=0.5, normed=True, label='Mega')
plt.legend()
plt.xlabel('Base Stat Total')


# In[27]:


strong_pokemon = pokemon[pokemon['total'] > 440]


# In[28]:


fig = plt.figure(figsize=(10,6))
sns.boxplot(x='generation', y='total', data=strong_pokemon,
           palette='Pastel1', whis=1)
plt.title('Pokemon with BST exceeding 440')
plt.ylabel('Base Stat Total')


# In[29]:


def stat_distribution(hp,attack,defense,spattack,spdefense,speed):
    stat_max = max([hp,attack,defense,spattack,spdefense,speed])
    stat_min = min([hp,attack,defense,spattack,spdefense,speed])
    stat_range = stat_max - stat_min
    return stat_max/stat_min

stat_distr = np.zeros(n_pokemon)
for i in range(n_pokemon):
    stat_distr[i] = stat_distribution(pokemon.hp.iloc[i], pokemon.attack.iloc[i], pokemon.defense.iloc[i],
                                     pokemon.spattack.iloc[i], pokemon.spdefense.iloc[i], pokemon.speed.iloc[i])


# In[30]:


pokemon['stat_distribution'] = stat_distr


# In[31]:


fig = plt.figure(figsize=(10,6))
sns.boxplot(x='generation', y='stat_distribution', data=pokemon[(pokemon['stat_distribution']<20) & (pokemon['total']>450)],
           palette='Pastel1')
plt.title('Pokemon with BST > 450')
yl = plt.ylabel('Maximum Stat / Minimum Stat')
xl = plt.xlabel('Generation')
yt = plt.yticks(range(1,11,1))


# In[32]:


pokemon[(pokemon['stat_distribution']>=7) & (pokemon['is_fully_evolved']==1)][['species','hp','attack','defense','spattack','spdefense','speed']]


# In[33]:


fig = plt.figure(figsize=(10,6))
sns.lmplot(x='ndex', y='stat_distribution', data=pokemon[(pokemon['stat_distribution']<20) & (pokemon['total']>450)])


# In[34]:


fig = plt.figure(figsize=(9,6))
typecounts = pokemon.groupby(['type1']).count()['total'] + pokemon.groupby(['type2']).count()['total']
cmap = plt.cm.get_cmap('tab20b')
typecounts.sort_values(axis=0, ascending=False).plot(kind='bar', color=cmap(np.linspace(0,1,len(typecounts+1))))
yl = plt.ylabel('count')


# In[35]:


typecounts1 = pokemon.groupby(['type1']).count()['total']
typecounts2 = pokemon.groupby(['type2']).count()['total']
sortind = np.argsort(typecounts1)[::-1]
fig, ax = plt.subplots(figsize=(10,6))
index = np.arange(len(typecounts))
bar_width = 0.35
rects1 = ax.bar(index, typecounts1[sortind], bar_width, label='Type 1')
rects2 = ax.bar(index + bar_width, typecounts2[sortind], bar_width, label='Type 2')
xt = plt.xticks(range(len(typecounts)), typecounts.index, rotation=90)
lg = plt.legend()
yl = plt.ylabel('count')


# In[36]:


fig = plt.figure(figsize=(9,6))
pokemon[pokemon['is_fully_evolved']==1].groupby('type1').mean()['total'].sort_values(axis=0, ascending=False).plot(kind='bar', color=cmap(np.linspace(0,1,len(typecounts+1))))
yl = plt.ylabel('Average BST')


# In[37]:


fig = plt.figure(figsize=(9,6))
pokemon[pokemon['is_fully_evolved']==1].groupby('type2').mean()['total'].sort_values(axis=0, ascending=False).plot(kind='bar', color=cmap(np.linspace(0,1,len(typecounts+1))))
yl = plt.ylabel('Average BST')


# In[38]:


fig = plt.figure(figsize=(10,6))
sns.lmplot(x='total',y='moveset_size',data=pokemon[(pokemon['is_fully_evolved']==1)],hue='generation', 
           fit_reg=False, aspect=1.5, size=8, palette='plasma')
yl = plt.ylabel("Moveset Size", fontsize=16)
xl = plt.xlabel('Base Stat Total', fontsize=16)
t = plt.title('Fully evolved Pokemon', fontsize=16)


# In[39]:


fig = plt.figure()
sns.lmplot(x='height',y='weight',data=pokemon)
yl = plt.ylabel("Weight (lbs)", fontsize=16)
xl = plt.xlabel('Height (feet)', fontsize=16)
t = plt.title('Pokemon sizes with linear fit', fontsize=16)


# In[40]:


pokemon[(pokemon['height']>25)][['forme','height','weight']]


# In[41]:


fig = plt.figure(figsize=(10,4))
bins = list(range(0,200,2))#+list(range(50,200,10))
(pokemon['weight']/pokemon['height']).plot.hist(bins=bins, linewidth=1.2, edgecolor='black')
xl = plt.xlabel('Weight / Height (lbs per feet)')
t = plt.title('Pseudo density of Pokemon (excl Cosmoem)')


# In[42]:


pokemon[(pokemon['weight']/pokemon['height']) <1.5][['forme','height','weight']]


# In[43]:


pokemon[(pokemon['weight']>661) & (pokemon['weight']<881)].sort_values(by='weight', ascending=False)[['forme','weight','height']]


# # 5. Machine Learning Analysis (WIP)

# **A.** Use Principal Component Analysis to visualize the variance within the numeric data.

# In[44]:


pokemon.columns


# In[45]:


pokemonML = pokemon[['hp','attack','defense','spattack','spdefense','speed','total','weight','height','generation','moveset_size']]


# In[46]:


X = pokemonML
y = pokemon['is_fully_evolved']


# In[47]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
scaled_X = scaler.transform(X)


# In[48]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled_X)


# In[49]:


X_pca = pca.transform(scaled_X)
X_pca.shape


# In[50]:


plt.figure(figsize=(10,6))
nfeHandle = plt.scatter(X_pca[(y.values==0),0], X_pca[(y.values==0),1], color='green')
feHandle = plt.scatter(X_pca[(y.values==1),0], X_pca[(y.values==1),1], color='purple')
plt.xlabel('First PC')
plt.ylabel('Second PC')
plt.legend(('Not Fully Evolved','Fully Evolved'))


# In[51]:


print(pca.explained_variance_ratio_)


# The first PC explains 41% of the variance, and the second PC explains a further 13%. From the scatterplot above, we can see that the two PCs are able to roughly separate the groups into fully evolved and not fully evolved.

# In[52]:


first_bool = (pokemon['is_first_stage']==1) & (pokemon['is_fully_evolved']==0)
middle_bool = (pokemon['is_fully_evolved']==0) & (pokemon['is_first_stage']==0)
final_bool = (pokemon['is_fully_evolved']==1) & (pokemon['is_first_stage']==0) & (pokemon['is_mega']==0)
mega_bool = (pokemon['is_mega']==1)

plt.figure(figsize=(12,8))
nfeHandle = plt.scatter(X_pca[(first_bool),0], X_pca[(first_bool),1], color='blue')
middleHandle = plt.scatter(X_pca[(middle_bool),0], X_pca[(middle_bool),1], color='orange')
feHandle = plt.scatter(X_pca[(final_bool),0], X_pca[(final_bool),1], color='purple')
megaHandle = plt.scatter(X_pca[(mega_bool),0], X_pca[(mega_bool),1], color='green')
plt.xlabel('First PC')
plt.ylabel('Second PC')
plt.legend(('First Stage','Middle Stage','Fully Evolved','Mega'))


# In[ ]:


# Visualize the contributions of each feature to the first 2 PCs
pca_components = pd.DataFrame(pca.components_, columns=X.columns)
plt.figure(figsize=(12,6))
sns.heatmap(pca_components,cmap='PiYG')
plt.yticks((0.5,1.5), ('PC 1', 'PC 2'))


# * near-zero contribution from the "generation" feature for both PCs 
# * "total" feature (Base Stat Total) is the strongest contributor for the first PC, while all others (except "generation") are roughly equal contributors

# In[ ]:


#plt.figure(figsize=(8,6))
#sns.lmplot(x='weight', y='total', data=pokemon)
sns.pairplot(data=X.drop('generation', axis=1))


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.drop('generation', axis=1), y, test_size=0.30, random_state=101)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators=100, random_state=60)
rfc.fit(X_train, y_train)


# In[ ]:


predictions = rfc.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




