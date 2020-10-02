#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sp
import sklearn

from matplotlib import pyplot as plt
from scipy.stats import norm, skew
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neural_network
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import datetime
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import confusion_matrix
pd.set_option('display.max_columns', 500)
import warnings
warnings.filterwarnings('ignore')

Examining the dataset
# In[ ]:


dataset = pd.read_csv('../input/Pokemon.csv')
dataset.head()

Alright, the concept of Mega Evolutions seems to be interesting:
There are two ways to flag the mega evolution pokemon
(a) Check if Mega is in the name (Bad idea - Meganium also has Mega in the name)
(b) Filter out cases where the Pokemon number is the same and then do the first check
# In[ ]:


duplicate_nums = pd.DataFrame(dataset['#'].value_counts())
duplicate_nums = duplicate_nums[duplicate_nums['#'] > 1]
duplicate_nums.reset_index(level=0, inplace=True)
duplicate_entries = dataset[dataset['#'].isin(duplicate_nums['index'])]
mega_evolution = duplicate_entries[duplicate_entries['Name'].str.contains('Mega')]
mega_evolution

Create a mega flag in the main dataset
# In[ ]:


dataset = pd.merge(dataset, mega_evolution[['Name']], on=['Name'], how='left', indicator='Mega_Flag')
dataset['Mega_Flag'] = np.where(dataset.Mega_Flag == 'both', 1, 0)
dataset.head()

Alright, time for some Exploratory Data Analysis on this dataQuestion 1: Which is the most/least popular Pokemon type?
# In[ ]:


dataset.rename(columns={'Type 1': 'Type_1', 'Type 2': 'Type_2'}, inplace=True)
q1a = pd.value_counts(dataset.Type_1).to_frame().reset_index()
q1b = pd.value_counts(dataset.Type_2).to_frame().reset_index()
q1a.rename(columns={'Type_1': 'Type'}, inplace=True)
q1b.rename(columns={'Type_2': 'Type'}, inplace=True)
q1 = q1a.append(q1b)


# In[ ]:


#The value counts function doesn't account for NaNs
q1.fillna(0)


# In[ ]:


q1 = q1.groupby(['index']).sum()


# In[ ]:


q1.reset_index(level=0, inplace=True)
q1 = q1.sort_values(by=['Type'],ascending = False)


# In[ ]:


ax = sns.catplot(x="Type",y="index", data= q1)
ax.set(xlabel='Number of Pokemon', ylabel='Pokemon Type')
plt.show()

Question 2: Single Type/Dual Type Pokemon Count? 
# In[ ]:


dataset['Dual_Type']  = np.where(dataset.Type_2.isnull(), "No", "Yes")
dataset.head(10)


# In[ ]:


q2 = pd.value_counts(dataset.Dual_Type).to_frame().reset_index()
q2.set_index('index')


# In[ ]:


q2.plot(kind = "pie", y="Dual_Type", autopct='%1.1f%%',labels=q2['index'])

Question 3: Based on Generation, Legendary, Megas, which type is more popular?
# In[ ]:


q3a = dataset[['Type_1','Generation','Legendary','Mega_Flag']]
q3b = dataset[['Type_2','Generation','Legendary','Mega_Flag']]
q3a.rename(columns={'Type_1': 'Type'}, inplace=True)
q3b.rename(columns={'Type_2': 'Type'}, inplace=True)
q3 = q3a.append(q3b)
q3.head()


# In[ ]:


q3 = q3.groupby(['Type','Generation']).count()[['Legendary']].reset_index()
q3.head()


# In[ ]:


g = sns.FacetGrid(q3, col="Generation",height=4, aspect=4, col_wrap=1)
g = g.map(plt.bar,'Type','Legendary',color=['lightgreen', 'black', 'darkslateblue', 'yellow', 'pink'
                                            ,'brown','red','mediumpurple','indigo','limegreen','khaki',
                                           'lightcyan','lightgrey','purple','deeppink','darkgoldenrod',
                                           'lightslategrey','dodgerblue'])
g.add_legend()
plt.show()

Analysis:

Generation 1 has a lot of Poison Pokemon which is not in abundance in the remaining generations, this could be attributed to a lot of dual typings, Pokemon like the Bulbasaur line, The Paras line, The Bellsprout line, The Oddish line all have a Grass/Poison dual typing. Even a Ghost pokemon like the Gastly line has the dual typing with Poison 

The Dark and Steel type Pokemon have only been introduced from the second generation onwards, hence the low number of these Pokemon are present in Generation 1.. In case you were wondering why there are still those types of Pokemon in Generation 1, Pokemon like the Magnemite line were Pure Electric converted into Electric/Steel in the later generations
# In[ ]:


q3a = dataset[['Type_1','Generation','Legendary','Mega_Flag']]
q3b = dataset[['Type_2','Generation','Legendary','Mega_Flag']]
q3a.rename(columns={'Type_1': 'Type'}, inplace=True)
q3b.rename(columns={'Type_2': 'Type'}, inplace=True)
q3 = q3a.append(q3b)
q3.head()


# In[ ]:


q3['Legendary'].value_counts()


# In[ ]:


q3 = q3.loc[q3.Legendary]


# In[ ]:


q3 = q3.groupby(['Type']).count()[['Legendary']].reset_index()
q3.head()


# In[ ]:


q3 = q3.sort_values(by=['Legendary'],ascending = False)
ax = sns.catplot(x="Legendary",y="Type", data= q3)
ax.set(xlabel='Number of Legendary Pokemon', ylabel='Pokemon Type')
plt.show()


# In[ ]:


q3a = dataset[['Type_1','Generation','Legendary','Mega_Flag']]
q3b = dataset[['Type_2','Generation','Legendary','Mega_Flag']]
q3a.rename(columns={'Type_1': 'Type'}, inplace=True)
q3b.rename(columns={'Type_2': 'Type'}, inplace=True)
q3 = q3a.append(q3b)
q3.head()


# In[ ]:


q3 = pd.DataFrame(q3.loc[q3['Mega_Flag'] == 1])
q3.head()


# In[ ]:


q3 = q3.groupby(['Type']).count()[['Legendary']].reset_index()
q3.head()


# In[ ]:


q3 = q3.sort_values(by=['Legendary'],ascending = False)
ax = sns.catplot(x="Legendary",y="Type", data= q3)
ax.set(xlabel='Number of Pokemon that can Mega Evolve', ylabel='Pokemon Type')
q3.head()

Analysis:

When it comes to Legendaries and Mega Evolutions, the Psychic and Dragon type Pokemon are more abundant while they are further down in the popularity when those filters are not appliedQuestion 4: Stat Distributions

Without looking at the data I am guessing that Rock, Steel and Ground types will be bulky and slow which means they might have good defenses and slow speed. Another hypothesis is Electric Pokemon will be fast and will be good at special attack. One more thing I can think of is that Fire and Water Pokemon will have the cliched moves like Flamethrower and Surf so they will be special attackers too as it makes sense in the Pokemon Games. Let's analyze if these hypothesis is correct
# In[ ]:


q4raw_1 = dataset[['Type_1','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']]
q4raw_2 = dataset[['Type_2','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']]
q4raw_1.rename(columns={'Type_1': 'Type'}, inplace=True)
q4raw_2.rename(columns={'Type_2': 'Type'}, inplace=True)
q4raw = q4raw_1.append(q4raw_2)
q4raw.rename(columns={'Sp. Atk': 'Special_Attack'}, inplace=True)
q4raw.rename(columns={'Sp. Def': 'Special_Defense'}, inplace=True)
q4raw.head()


# In[ ]:


q4raw = q4raw[q4raw.Type.notnull()]
q4raw.shape


# In[ ]:


q4melt = pd.melt(q4raw, id_vars = ['Type'], value_vars = ['HP', 'Attack','Defense','Special_Attack','Special_Defense','Speed'])
q4melt.head()


# In[ ]:


q4melt.rename(columns={'variable': 'Stat_Type'}, inplace=True)
q4melt.rename(columns={'value': 'Stat_Value'}, inplace=True)
sns.set()
g = sns.catplot(x = "Stat_Type", y = "Stat_Value", col = "Type", data = q4melt, kind = "violin", col_wrap = 1, aspect = 3)
plt.grid(True)

On the speed front, the hypothesis seems to be correct as the rock and steel pokemon have median speed below 50. Also, the electric pokemon have a median speed close to 100

In terms of bulk, the hypothesis is also correct, in fact, the defense of steel pokemon even rises to 250+ (courtesy of Mega Steelix)

The initial hypothesis is a bit off in terms of the water and fire pokemon's special attack because although fire pokemon have a higher median special attack than the rest of the special pokemon, the same cannot be said for water pokemon. Take Kingler for example, he can learn surf but that doesn't mean his special attack is high. There are also too many water pokemon to make this generalization.