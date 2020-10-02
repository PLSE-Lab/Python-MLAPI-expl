#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# In[ ]:


pokemon = pd.read_csv('../input/pokemon/Pokemon.csv')
pokemon.head()


# In[ ]:


pokemon = pokemon.drop('#', axis =1)


# In[ ]:


pokemon.isnull().sum()


# In[ ]:


pokemon.shape


# In[ ]:


pokemon.info()


# Exploratory Data Analysis

# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(pokemon['Type 1'], hue = 'Legendary', data = pokemon)#type1 is Each pokemon has a type, this determines weakness/resistance to attacks


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(pokemon['Type 2'], hue = 'Legendary', data = pokemon) 


# In[ ]:


plt.figure(figsize=(15,20))
plt.subplot(4,3,1)
sns.barplot(x = 'Legendary', y ='Total', data = pokemon)# Total is sum of all stats that come after this, a general guide to how strong a pokemon is
plt.subplot(4,3,2)
sns.barplot(x = 'Legendary', y ='HP', data = pokemon)#HP is hit points, or health, defines how much damage a pokemon can withstand before fainting
plt.subplot(4,3,3)
sns.barplot(x = 'Legendary', y ='Attack', data = pokemon) # the base modifier for normal attacks 
plt.subplot(4,3,4)
sns.barplot(x = 'Legendary', y ='Defense', data = pokemon)# the base damage resistance against normal attacks
plt.subplot(4,3,5)
sns.barplot(x = 'Legendary', y ='Sp. Atk', data = pokemon)
plt.subplot(4,3,6)
sns.barplot(x = 'Legendary', y ='Sp. Def', data = pokemon)
plt.subplot(4,3,7)
sns.barplot(x = 'Legendary', y ='Speed', data = pokemon)


# In[ ]:


sns.countplot(x = 'Generation', hue = 'Legendary', data = pokemon)


# In[ ]:


plt.figure(figsize=(15,20))
plt.subplot(4,3,1)
sns.distplot(pokemon['Total'])# Total is sum of all stats that come after this, a general guide to how strong a pokemon is
plt.subplot(4,3,2)
sns.distplot(pokemon['HP'])#HP is hit points, or health, defines how much damage a pokemon can withstand before fainting
plt.subplot(4,3,3)
sns.distplot(pokemon['Attack']) # the base modifier for normal attacks 
plt.subplot(4,3,4)
sns.distplot(pokemon['Defense'])# the base damage resistance against normal attacks
plt.subplot(4,3,5)
sns.distplot(pokemon['Sp. Atk'])
plt.subplot(4,3,6)
sns.distplot(pokemon['Sp. Def'])
plt.subplot(4,3,7)
sns.distplot(pokemon['Speed'])


# In[ ]:


plt.figure(figsize=(15,20))
plt.subplot(4,3,1)
sns.boxplot(pokemon['Total'])# Total is sum of all stats that come after this, a general guide to how strong a pokemon is
plt.subplot(4,3,2)
sns.boxplot(pokemon['HP'])#HP is hit points, or health, defines how much damage a pokemon can withstand before fainting
plt.subplot(4,3,3)
sns.boxplot(pokemon['Attack']) # the base modifier for normal attacks 
plt.subplot(4,3,4)
sns.boxplot(pokemon['Defense'])# the base damage resistance against normal attacks
plt.subplot(4,3,5)
sns.boxplot(pokemon['Sp. Atk'])
plt.subplot(4,3,6)
sns.boxplot(pokemon['Sp. Def'])
plt.subplot(4,3,7)
sns.boxplot(pokemon['Speed'])


# In[ ]:


plt.figure(figsize=(15,10)) #manage the size of the plot
sns.heatmap(pokemon.corr(),annot=True, square = True) 
plt.show()


# In[ ]:


pokemon.columns


# In[ ]:


pokemon = pokemon.drop(columns=['Name', 'Type 2'], axis = 1)


# Outlier Detection

# In[ ]:


from scipy import stats
z=np.abs(stats.zscore(pokemon['HP']))
threshold=3
print(np.where(z>3))


# In[ ]:


from scipy import stats
z=np.abs(stats.zscore(pokemon['Attack']))
threshold=3
print(np.where(z>3))


# In[ ]:


from scipy import stats
z=np.abs(stats.zscore(pokemon['Defense']))
threshold=3
print(np.where(z>3))


# In[ ]:


from scipy import stats
z=np.abs(stats.zscore(pokemon['Sp. Atk']))
threshold=3
print(np.where(z>3))


# In[ ]:


from scipy import stats
z=np.abs(stats.zscore(pokemon['Sp. Def']))
threshold=3
print(np.where(z>3))


# In[ ]:


df1=pokemon[(z< 3)]
print(df1)


# In[ ]:


df1.shape


# In[ ]:


df1['Type 1']= df1['Type 1'].astype('category')
df1['Type 1']= df1['Type 1'].cat.codes

df1['Legendary']= df1['Legendary'].astype('category')
df1['Legendary']= df1['Legendary'].cat.codes


# In[ ]:


df1.head()


# In[ ]:


X = df1.drop('Legendary', axis =1)
y = df1['Legendary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state = 100)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


def model_train(classifier, X_train, y_train, X_test, y_test):
  model =classifier.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  print(classification_report(y_test, y_pred))
  print('Accuracy of model is ', accuracy_score(y_test, y_pred))


# In[ ]:


model_train(LogisticRegression(),X_train, y_train, X_test, y_test )


# In[ ]:


model_train(RandomForestClassifier(),X_train, y_train, X_test, y_test )


# In[ ]:


model_train(GradientBoostingClassifier(),X_train, y_train, X_test, y_test )


# In[ ]:


model_train(BaggingClassifier(),X_train, y_train, X_test, y_test )


# In[ ]:


model_train(xgb.XGBClassifier(),X_train, y_train, X_test, y_test )


# Improvements can be done using hyper parameter tuning
