#!/usr/bin/env python
# coding: utf-8

# # My Clash Royale battles' EDA
# 
# ## Queries
# 
# - Trophies over battles
# - Difference of trophies between me and my opponents
#   - Trophies difference over battles
#   - Number of battles I had more trophies than my opponent
#   - Number of battles I had less trophies than my opponent
#   - Number of battles me and the opponent had the same trophies
#   - Correlation between difference of trophies and crows won in each battle
#   - Battle's result count based on diff of trophies
# - Difference between my deck mean level and op's deck mean level
#   - Cards mean lvl diff over time 
#   - Cards mean lvl diff by result
# - Opponents' elixir
#   - Median deck elixir cost by arena
#   - Opponents' elixir based on my battle result
# - Exploring arenas
#   - Number of battles played in each arena
# - Stats by arena
#   - Total battles in arena
#   - Zooming in victories and defeats
#     - Mine and opponents' card types (troops, buildings, spells, commons, rares and epics)
#     - Opponents' cards median level
#     - Top 20 opponents' cards

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import seaborn as sns
import numpy as np
import operator
import matplotlib.pyplot as plt
import re


# In[ ]:


df = pd.read_csv('../input/8V280L8VQ-clash-royale-da.csv', sep=',')


# In[ ]:


# Reversing dataframe order

df = df.sort_index(axis=0, ascending=False)
fixed_index = list(df.index)
fixed_index.sort()
df = df.set_index([fixed_index])

# Cleaning data
df['my_result'] = df['my_result'].apply(lambda x: x.strip())


# In[ ]:


def arena(trophies):
    if trophies < 400:
        return '1'
    elif trophies < 800:
        return '2'
    elif trophies < 1100:
        return '3'
    elif trophies < 1400:
        return '4'
    elif trophies < 1700:
        return '5'
    elif trophies < 2000:
        return '6'
    elif trophies < 2300:
        return '7'
    elif trophies < 2600:
        return '8'
    elif trophies < 3000:
        return '9'
    elif trophies < 3400:
        return '10'
    elif trophies < 3800:
        return '11'
    else: 
        return '12'


# In[ ]:


df['arena'] = df['my_trophies'].apply(lambda row: arena(row))


# In[ ]:


pd.set_option('display.max_columns', None)  
df.describe()


# ## Trophies over time (battles)

# In[ ]:


(df['my_trophies'].rolling(window = 1)
     .mean()
     .plot(figsize=(20, 10)))


# ## Difference of trophies between me and my opponents

# In[ ]:


df_diff_troph = df[['my_trophies', 'opponent_trophies']]

df_diff_troph = df_diff_troph.assign(troph_diff = df.my_trophies - df.opponent_trophies)


# In[ ]:


df_diff_troph['troph_diff'].describe()


# ### Trophies difference over time (battles, window = 5)

# In[ ]:


(df_diff_troph['troph_diff'].rolling(window = 5)
     .mean()
     .plot(figsize=(20, 10)))


# ### Trophies difference over time (battles, window = 20)

# In[ ]:


(df_diff_troph['troph_diff'].rolling(window = 20)
     .mean()
     .plot(figsize=(20, 10)))


# ### Number of battles I had more trophies than my opponent

# In[ ]:


df_diff_troph[df_diff_troph.troph_diff > 0]['troph_diff'].describe()


# ### Number of battles I had less trophies than my opponent

# In[ ]:


df_diff_troph[df_diff_troph.troph_diff < 0]['troph_diff'].describe()


# ### Number of battles me and the opponent had the same trophies

# In[ ]:


df_diff_troph[df_diff_troph.troph_diff == 0]['troph_diff'].describe()


# ### Correlation between difference of trophies and crows won in each battle

# In[ ]:


df['troph_diff'] = df_diff_troph['troph_diff']

corr = df[['troph_diff', 'points']].corr()

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)

corr


# ### Battles' result count based on diff of trophies

# #### When I have more or the same number of trophies than my opponent

# In[ ]:


df[['troph_diff', 'my_result']][df.troph_diff >= 0].groupby(['my_result']).count()


# #### When I have less trophies than my opponent

# In[ ]:


df[['troph_diff', 'my_result']][df.troph_diff < 0].groupby(['my_result']).count()


# ### Finding more linear correlations

# In[ ]:


def corrs(df, columns_list, plot_size=(50, 50), font_scale = 1.5):
    if columns_list == 0:
        corrs = df.corr().round(2)        
    else:
        corrs = df[columns_list].corr().round(2)
        
    mask = np.zeros_like(corrs)
    mask[np.triu_indices_from(mask, k = 0)] = True        

    sns.set(rc={'figure.figsize': plot_size})

    sns.set(font_scale=font_scale)

    sns.heatmap(corrs, xticklabels=corrs.columns.values, yticklabels=corrs.columns.values, 
                annot=True, mask=mask)
    
    plt.show()        


# In[ ]:


corrs(df, ['points', 'my_deck_elixir', 'op_deck_elixir', 'my_troops', 'op_troops', 
           'my_buildings', 'op_buildings', 'my_spells', 'op_spells', 
           'my_commons', 'op_commons', 'my_rares', 'op_rares', 'my_epics', 'op_epics'], 
      font_scale = 3)


# ## Difference between my deck mean level and op's deck mean level

# In[ ]:


df_aux = df.filter(regex=('my_[A-Z]|op_[A-Z]'))

lvl_diff = []

for row in df_aux.itertuples(index=False):  
    my_lvls = []
    op_lvls = []
    
    for col in row._fields:        
        if not col.startswith('_'):
            if getattr(row, col) > 0:
                if re.match('my_[A-Z]', col):
                    my_lvls.append(getattr(row, col))
                else:
                    op_lvls.append(getattr(row, col))
    
    lvl_diff.append(np.mean(my_lvls) - np.mean(op_lvls))
    
df['cards_lvl_mean_diff'] = lvl_diff


# In[ ]:


df['cards_lvl_mean_diff'].describe()


# In[ ]:


corrs(df, ['points', 'cards_lvl_mean_diff'], font_scale = 3)


# ### Cards mean lvl diff over time (battles, window = 5)

# In[ ]:


(df['cards_lvl_mean_diff'].rolling(window = 5)
     .mean()
     .plot(figsize=(20, 10)))


# ### Cards mean lvl diff over time (battles, window = 20)

# In[ ]:


(df['cards_lvl_mean_diff'].rolling(window = 20)
     .mean()
     .plot(figsize=(20, 10)))


# ### Cards mean lvl diff by battle result

# In[ ]:


sns.set(rc={'figure.figsize':(15, 10)})
sns.boxplot(x=df['my_result'], y=df['cards_lvl_mean_diff'])


# ## Opponents' elixir

# In[ ]:


df['op_deck_elixir'].describe()


# 1. ### Opponents elixir over time (battles, window = 5)

# In[ ]:


(df['op_deck_elixir'].rolling(window = 5)
     .mean()
     .plot(figsize=(20, 10)))


# ### Opponents elixir over time (battles, window = 20)

# In[ ]:


(df['op_deck_elixir'].rolling(window = 20)
     .mean()
     .plot(figsize=(20, 10)))


# ### Median deck elixir cost by arena

# In[ ]:


df[['op_deck_elixir', 'arena']].groupby(['arena']).median()


# ### Opponents' elixir based on my battle result

# In[ ]:


df[df.my_result == 'Victory']['op_deck_elixir'].describe()


# In[ ]:


df[df.my_result == 'Defeat']['op_deck_elixir'].describe()


# In[ ]:


sns.set(rc={'figure.figsize':(15, 10)})
sns.boxplot(x=df['my_result'], y=df['op_deck_elixir'])


# ## Exploring arenas

# #### Number of battles played in each arena

# In[ ]:


df.groupby(['arena'])['points'].count()


# In[ ]:


sns.set(rc={'figure.figsize':(15, 10)})
sns.countplot(x = df['arena'])


# #### Stats by arena

# In[ ]:


def op_cards_median_level(battles_by_result):
    return np.mean(list(filter(lambda x: x != 0, [y for x in battles_by_result.filter(regex=('op_[A-Z]')).values.tolist() for y in x] )) )


def top_op_cards(n, battles_by_result):
    aux_cards_dict = {}

    aux_filtered_df = battles_by_result.filter(regex=('op_[A-Z]'))
    
    for row in aux_filtered_df.itertuples(index=False):
        for col in row._fields:        
            if not col.startswith('_'):
                if getattr(row, col) > 0:
                    if col in aux_cards_dict:
                        aux_cards_dict[col] = aux_cards_dict[col] + 1
                    else:
                        aux_cards_dict[col] = 1

    return pd.DataFrame(data=sorted(aux_cards_dict.items(), key=operator.itemgetter(1), reverse=True), columns=['Card', 'Times_Used']).head(n)    


# In[ ]:


for i in reversed(range(1, 10)):
    print('~~~ Arena %d stats ~~~' % i)
    
    battles_in_arena = df[df.arena == str(i)]
    print('Total battles in arena %d: %d' % (i, battles_in_arena.shape[0]))
    print('-------')
    print(battles_in_arena.groupby(['my_result'])['arena'].count())
    
    sns.set(rc={'figure.figsize':(10, 10)})
    sns.countplot(x=battles_in_arena['my_result'], order=['Victory', 'Defeat', 'Draw'])
    plt.show()    
    
    print('--- Victories ---')
    
    battles_by_vic = battles_in_arena[battles_in_arena.my_result == 'Victory']
    
    print('Cards types count (median)')
    
    cards_types1_vic = (battles_by_vic[['my_troops', 'my_buildings', 'my_spells', 
                                       'op_troops', 'op_buildings', 'op_spells']]
                       .describe())
    
    print(cards_types1_vic.loc[['50%']])
    
    print('----')
    
    cards_types2_vic = (battles_by_vic[['my_commons', 'my_rares', 'my_epics', 
                                        'op_commons', 'op_rares', 'op_epics']]
                       .describe())    
    
    print(cards_types2_vic.loc[['50%']])
    
    
    print('\nOpponents\' cards median level')
    
    print( op_cards_median_level(battles_by_vic) )    
    
    print('\nTop 20 opponents\' cards')

    print(top_op_cards(20, battles_by_vic))
    
    
    print('\n--- Defeats ---')
    
    battles_by_def = battles_in_arena[battles_in_arena.my_result == 'Defeat']
    
    print('Cards types count (median)')
    
    cards_types1_def = (battles_by_def[['my_troops', 'my_buildings', 'my_spells', 
                                       'op_troops', 'op_buildings', 'op_spells']]
                       .describe())    
    
    print(cards_types1_def.loc[['50%']])        
    
    print('----')
    
    cards_types2_def = (battles_by_def[['my_commons', 'my_rares', 'my_epics', 
                                        'op_commons', 'op_rares', 'op_epics']]
                       .describe())       
    
    print(cards_types2_def.loc[['50%']])
    
    
    print('\nOpponents\' cards median level')
    
    print( op_cards_median_level(battles_by_def) )    
    
    print('\nTop 20 opponents\' cards')
    
    print(top_op_cards(20, battles_by_def))
    
    print('----')
    
    sns.set(rc={'figure.figsize':(10, 10)})
    sns.barplot(x='val', y='type', data=pd.DataFrame(data={'type': cards_types1_vic.columns,
                                                             'val': (cards_types1_vic.loc[['50%']]
                                                                     .values.tolist()[0])
                                                          })).set_title('Cards types 1 (Victory)')   
    plt.show()

    sns.set(rc={'figure.figsize':(10, 10)})
    sns.barplot(x='val', y='type', data=pd.DataFrame(data={'type': cards_types1_def.columns,
                                                             'val': (cards_types1_def.loc[['50%']]
                                                                     .values.tolist()[0])
                                                          })).set_title('Cards types 1 (Defeat)')   
    plt.show()

    sns.set(rc={'figure.figsize':(10, 10)})
    sns.barplot(x='val', y='type', data=pd.DataFrame(data={'type': cards_types2_vic.columns,
                                                             'val': (cards_types2_vic.loc[['50%']]
                                                                     .values.tolist()[0])
                                                          })).set_title('Cards types 2 (Victory)')   
    plt.show()    

    sns.set(rc={'figure.figsize':(10, 10)})
    sns.barplot(x='val', y='type', data=pd.DataFrame(data={'type': cards_types2_def.columns,
                                                             'val': (cards_types2_def.loc[['50%']]
                                                                     .values.tolist()[0])
                                                          })).set_title('Cards types 2 (Defeat)')   
    plt.show()  


    
    
    print('------------------------------------------------\n')


# In[ ]:




