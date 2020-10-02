#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import KFold,train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from statistics import mode


# In[ ]:


Pokemon=pd.read_csv("/kaggle/input/pokemon/Pokemon.csv")


# In[ ]:


Pokemon.columns


# In[ ]:


Pokemon.shape


# In[ ]:


Pokemon.head()


# In[ ]:


D=pd.DataFrame({"Dtype":Pokemon.dtypes,"Null":Pokemon.isnull().sum(),"Percentage NUlls":Pokemon.isnull().sum()/len(Pokemon)*100,"Uniques":Pokemon.nunique()})
D


# the "#" doesnt tell us anything

# In[ ]:


Pokemon=Pokemon.drop("#",axis=1)


# ![alt text](https://images.squarespace-cdn.com/content/v1/51d3a8f6e4b085686832d41d/1379381612026-4Q7SD886Q6JDDMPBGKA6/ke17ZwdGBToddI8pDm48kF6me4WIKcQyiJGDHKOTMXZZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZamWLI2zvYWH8K3-s_4yszcp2ryTI0HqTOaaUohrI8PIEYrKsYlfql8PAsl_09JHAxJlGpVb3NHgTRYRkIuvMzQ/Pok%C3%87mon+Gotta+Catch+%27Em+All_Logo_EN_800px_150dpi.png)

# # UNIVARIATE ANALYSIS

# In[ ]:


Pokemon['Type 1'].value_counts(normalize=True)


# In[ ]:


Pokemon['Type 1'].value_counts().plot(kind="bar",color="green")


# 
# 
# *  14 percent of pokemon having 1st type water , it is the most common  type 1 of Pokemon
# *   Flying Type pokemons are the rarest pokemon first Type
# for a pokemon
# 
# 

# In[ ]:


Pokemon['Type 2'].value_counts(normalize=True)


# In[ ]:


Pokemon['Type 2'].value_counts().plot(kind="bar",color="green")


# 
# 
# *  23 % of pokemon having 2nd type as Flying , it is the most common  type 1 of Pokemon
# *  Bug Type pokemons are the rarest pokemon 2nd Type
# for a pokemon
# 
# 
# *   Suprisingly Flying type which is the rarest 1st type but is the most common 2nd Type 
# 
# 
# 
#  

# In[ ]:


sns.distplot(Pokemon['Attack'])


# Most pokemons have attack between 70 and 100,and they are normally Distributed

# In[ ]:


sns.distplot(Pokemon['Defense'])


# In[ ]:


Pokemon['Generation'].value_counts().plot(kind="bar")


# Most of the pokemons are generation 1 Pokemons
# Generation 6 are the least

# In[ ]:


Pokemon['Legendary'].value_counts(normalize=True)


# In[ ]:


Pokemon['Legendary'].value_counts()


# There are only 65 Legendary Pokemons amoung 800 Pokemons

# In[ ]:


Pokemon['Legendary'].value_counts().plot(kind="bar")


# Only 8 % of the Pokemons are LEGENDARY, remaining are normal Pokemons

# In[ ]:


Pokemon[Pokemon['Legendary']==True]["Name"].head()


# In[ ]:


w=WordCloud()
legend=" "
for i in Pokemon[Pokemon['Legendary']==True]["Name"]:
  legend=legend+" "+i
plt.figure(figsize=(15,7))
plt.grid(False)
plt.imshow(w.generate(legend),interpolation='bilinear',)


# ![alt text](https://i.pinimg.com/originals/ef/04/cc/ef04cc59edfd2cbdfb52999c79018b34.jpg)

# In[ ]:


plt.figure(figsize=(15,7))
sns.distplot(Pokemon['HP'])


# In[ ]:


Pokemon['HP'].describe()


# Mean Health of all pokemons is 69,  most of the pokemons have health between 65 to 80

# # BIVARIATE ANALYSIS

# In[ ]:


sns.catplot(x="Legendary",y="Type 1",data=Pokemon,kind="bar",legend=True)


# In[ ]:


Ct=pd.crosstab(Pokemon['Type 1'],Pokemon['Legendary'])
Ct.div(Ct.sum(1),axis=0).plot.bar(stacked=True)


# Most of the Legendary TYpe Pokemons have first Type Flying
# 
# 
# 

# After Flying type Dragon Type are legendary
# 

# In[ ]:


corr=Pokemon.corr()
corr=corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool))
plt.figure(figsize=(15,8))
sns.heatmap(corr,annot=True,cmap="YlGnBu")


# By this heat Map we can find the continious variables which really effect our categorical variable LEGENDARY

# ![alt text](https://i.pinimg.com/originals/72/a3/a2/72a3a255b546145656e651ffd981045f.jpg)

# *The Above image cointains all the pokemons*

# In[ ]:


def fill_type2(x):
  return Pokemon[Pokemon["Type 1"]==x["Type 1"]]["Type 2"].mode()[0]


# In[ ]:


Pokemon['Type 2']=Pokemon.apply(lambda x:fill_type2 if pd.isnull(x["Type 2"]) else x["Type 2"],axis=1)


# #  Model Building 

# # Classify if the pokemon is legendary or not

# In[ ]:


new_pokemon=pd.get_dummies(Pokemon.drop("Legendary",axis=1))


# In[ ]:


Maxmin=MinMaxScaler()


# In[ ]:


Pokemon_scaled=Maxmin.fit_transform(new_pokemon)


# In[ ]:


new_pokemon_scaled=pd.DataFrame(Pokemon_scaled,columns=new_pokemon.columns)


# In[ ]:


x=new_pokemon_scaled
y=Pokemon['Legendary'].replace({True:1,False:0})


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,train_size=0.9,shuffle=True,random_state=92)


# In[ ]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# As our target is not equally distributed we cannot use accuracy,we are going to use another metric called as f1 score

# To avoid over fitting we are using Kfold to make our model perfectly fit our data

# In[ ]:


def model_build(model,x_train,y_train,x_test,y_test):
   k=KFold(shuffle=True,random_state=94,n_splits=4)
   a=1
   for i,j  in k.split(x_train,y_train):
               x_trainn,x_val=  x_train.iloc[i],x_train.iloc[j]
               y_trainn,y_val=  y_train.iloc[i],y_train.iloc[j]

               model.fit(x_trainn,y_trainn)
               train_scores=model.predict(x_trainn)
               test_scores=model.predict(x_val)
               val_scores=model.predict(x_test)
               print("{}.Train f1 score is {} and test f1_score is {} validation score {}".format(a,f1_score(train_scores,y_trainn),f1_score(y_val,test_scores),f1_score(y_test,val_scores)))
           

               a=a+1
   test=model.predict(x_test)
   trains=model.predict(x_train)
   return model,test,trains


# In[ ]:


X=RandomForestClassifier(max_features=0.5,random_state=88,n_estimators=500,max_depth=100)
X,test_predictions,train_predictions=model_build(X,x_train,y_train,x_test,y_test)


# In[ ]:


L=KNeighborsClassifier(n_neighbors=1)
L,l_test_score,l_train_scores=model_build(L,x_train,y_train,x_test,y_test)


# In[ ]:


A=AdaBoostClassifier(n_estimators=400,learning_rate=0.7)
A,a_test_score,a_train_scores=model_build(A,x_train,y_train,x_test,y_test)


# In[ ]:


accuracy=pd.DataFrame({"test":[f1_score(a_test_score,y_test),f1_score(test_predictions,y_test),f1_score(l_test_score,y_test)],"train":[f1_score(a_train_scores,y_train),f1_score(train_predictions,y_train),f1_score(l_train_scores,y_train)]},index=["Ada","random","Neighbors"])
accuracy.plot(kind="bar")


# Adaboost is the best of them all

# In[ ]:


x=new_pokemon_scaled[ranks['columns'][:500]]
y=Pokemon['Legendary'].replace({True:1,False:0})


# After getting the new features x,y ,now we must go to train_test_split and run again

# In[ ]:


final=[]
for i in range(len(l_test_score)):
      final.append(mode([test_predictions[i],l_test_score[i],a_test_score[i]]))


# In[ ]:


f1_score(final,y_test)


# Final F1 Score after after ensembling 3 models is 90 percent, so we can predict if our pokemon is legendary with 90 percent confidence 
