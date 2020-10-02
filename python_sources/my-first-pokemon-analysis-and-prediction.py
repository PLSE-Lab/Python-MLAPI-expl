#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
init_notebook_mode(connected=True)
sns.set_style('whitegrid')


# In[ ]:


pokemon = pd.read_csv('../input/pokemon/Pokemon.csv')


# In[ ]:


pokemon.head()


# In[ ]:


pokemon.info()


# Type 2 is having null data. If you are a Pokemon fan, you know that not every pokemon have 2 types, so it's normal

# In[ ]:


pokemon['Legendary'].value_counts()


# There are 65 Legendary Pokemon from Gen 1 to Gen 6.
# OMG, I thought the number is only about 20. The last pokemon game I play was gen 3. I have no Idea about Gen 4 5 6

# In[ ]:


pokemon.groupby(['Generation','Legendary']).count()['Name']


# Okay, Gen 3 has the most legendary pokemon, and in gen 4 5 6, total legend pokemon dramatically increase (Why the heck did you do that Nintendo?)

# In[ ]:


pokemon[pokemon['Legendary']]['Type 2'].isnull().value_counts()


# Legendary Pokemons often have 2 Types (yess, they are Legends)

# Let transform the data of Legendary column to number so that we can visualize it.

# In[ ]:


Legend = pd.get_dummies(pokemon['Legendary'],drop_first=True)


# In[ ]:


pokemon.drop(['#','Legendary'],axis=1,inplace=True)


# In[ ]:


pokemon = pd.concat([pokemon,Legend],axis=1)
pokemon.rename({True:'Legend'},axis=1, inplace=True)


# In[ ]:


pokemon.head()


# First, use pairplot to view the correlation

# In[ ]:


g=sns.pairplot(pokemon[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Legend']],hue='Legend')


# In[ ]:


corr = pokemon[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']].corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(12,6))
sns.heatmap(corr,mask=mask,cmap='magma',annot=True)


# * Special Defence is related to Defense and Special Attack!! 
# * Attack and Deffence also have correlation

# Let see top 10 strongest pokemon

# In[ ]:


plt.figure(figsize=(12,6))
sns.barplot(y='Name',x='Total', 
            data=pokemon.iloc[pokemon['Total'].sort_values(ascending=False)[:10].index], 
            palette="rainbow")


# Wow, Mewtwo is really strong, maybe the strongest Pkm so far. And you can clearly see that with Megaevolution, Nintendo boosted the total stats of Pokemon

# Top 10 Pokemon have biggest Special Attack

# In[ ]:


df = pokemon.iloc[pokemon['Sp. Atk'].sort_values(ascending=False)[:10].index]
fig = px.bar(df, x='Sp. Atk', y='Name', orientation='h',color='Legend',
            category_orders={'Name': df['Name'].tolist()})
fig.show()


# Mewtwo again. Most of strong pkms are Legendary

# Top 10 Pokemon have biggest Attack

# In[ ]:


df = pokemon.iloc[pokemon['Attack'].sort_values(ascending=False)[:10].index]
fig = px.bar(df, x='Attack', y='Name', orientation='h',color='Generation',
            category_orders={'Name': df['Name'].tolist()})
fig.show()


# Generation 3 have outstanding pokemons the most. Imbalance generation  

# OK, Let go to compare 2 pokemons based on Attack Defense and HP Speed.

# In[ ]:


def pokeplot(pokemon1,pokemon2,stat1,stat2):
    f = sns.FacetGrid(pokemon[(pokemon['Generation'].apply(lambda x: x in [1,2,3,4,5,6]))], hue='Legend', size=8)        .map(plt.scatter, stat1, stat2, alpha=0.5)        .add_legend()
    plt.subplots_adjust(top=0.9)
    f.fig.suptitle('{} vs. {}'.format(stat1, stat2))
    f.ax.set_xlim(0,)
    f.ax.set_ylim(0,)
    
    pkm1 = pokemon[pokemon['Name'] == pokemon1]
    pkm2 = pokemon[pokemon['Name'] == pokemon2]
    
    plt.scatter(pkm1[stat1],pkm1[stat2],s=100,marker='x', c='#d400ff')
    plt.text(pkm1[stat1]+3,pkm1[stat2]-6, pokemon1, 
             fontsize=13, weight='bold', color='#d400ff')
    
    plt.scatter(pkm2[stat1],pkm2[stat2],s=100,marker='x', c='#ff0000')
    plt.text(pkm2[stat1]+3,pkm2[stat2]-6, pokemon2, 
             fontsize=13, weight='bold', color='#ff0000')


# In[ ]:


pokeplot('Mewtwo','Rayquaza','Attack','Defense')


# In[ ]:


pokeplot('Mewtwo','Rayquaza','Speed','HP')


# # Predict whether a pokemon is legendary base on stats (Basic ML)
# 

# 1. Go to check the data and clean it

# First, we need to convert Type to number, and if Type= null make it to 0

# In[ ]:


def type_numbering(string) : 
    if string == 'Normal' :
        return 1
    elif string== 'Fire' :
        return 2
    elif string == 'Fighting' :
        return 3
    elif string == 'Water' :
        return 4
    elif string == 'Flying' :
        return 5
    elif string == 'Grass' :
        return 6
    elif string == 'Poison' :
        return 7
    elif string == 'Electric' :
        return 8
    elif string == 'Ground' :
        return 9
    elif string == 'Psychic' :
        return 10
    elif string == 'Rock' :
        return 11
    elif string == 'Ice' :
        return 12
    elif string == 'Bug' :
        return 13
    elif string == 'Dragon' :
        return 14
    elif string == 'Ghost' :
        return 15
    elif string == 'Dark' :
        return 16
    elif string == 'Steel' :
        return 17
    elif string == 'Fairy' :
        return 18
    else :
        return 0


# In[ ]:


pokemon['Type 1'] = pokemon['Type 1'].apply(type_numbering)
pokemon['Type 2'] = pokemon['Type 2'].apply(type_numbering)


# In[ ]:


pokemon.drop('Name',axis=1,inplace=True)


# 2. Create data for training and testing by using Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X= pokemon.drop('Legend',axis=1)
y= pokemon['Legend']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=90)


# 3. Using model

# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmode = LogisticRegression()


# In[ ]:


logmode.fit(X_train,y_train)


# In[ ]:


y_pred = logmode.predict(X_test)


# In[ ]:


cm = pd.DataFrame(confusion_matrix(y_test,y_pred), 
                  index = ['Non-Legendary', 'Legendary'], columns = ['Non-Legendary', 'Legendary'])
cm.index.name = 'Actual'
cm.columns.name = 'Predict'
print(cm)


# In[ ]:


print(classification_report(y_test,y_pred))


# **Decision Tree**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dtc = DecisionTreeClassifier(max_depth=5)


# In[ ]:


dtc.fit(X_train,y_train)


# In[ ]:


y_pred2 = dtc.predict(X_test)


# In[ ]:


cm = pd.DataFrame(confusion_matrix(y_test,y_pred2), 
                  index = ['Non-Legendary', 'Legendary'], columns = ['Non-Legendary', 'Legendary'])
cm.index.name = 'Actual'
cm.columns.name = 'Predict'
print(cm)


# In[ ]:


print(classification_report(y_test,y_pred2))


# **Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc= RandomForestClassifier(max_depth=5)


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


y_pred3 = rfc.predict(X_test)


# In[ ]:


cm = pd.DataFrame(confusion_matrix(y_test,y_pred3), 
                  index = ['Non-Legendary', 'Legendary'], columns = ['Non-Legendary', 'Legendary'])
cm.index.name = 'Actual'
cm.columns.name = 'Predict'
print(cm)


# In[ ]:


print(classification_report(y_test,y_pred3))


# **Support Vecter Machines**

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svc= SVC()


# In[ ]:


from sklearn.model_selection import GridSearchCV
params = {'C':[0.1, 1, 10, 100, 1000],'gamma':[1, 0.1, .01, .001, .0001]}
grid = GridSearchCV(estimator=svc, param_grid=params,refit=True,verbose=2)
grid.fit(X_train, y_train)


# In[ ]:


print(grid.best_score_)
print(grid.best_params_)


# In[ ]:


pred_g = grid.predict(X_test)


# In[ ]:


cm = pd.DataFrame(confusion_matrix(y_test,pred_g), 
                  index = ['Non-Legendary', 'Legendary'], columns = ['Non-Legendary', 'Legendary'])
cm.index.name = 'Actual'
cm.columns.name = 'Predict'
print(cm)


# In[ ]:


print(classification_report(y_test,pred_g))


# Thank you so much
