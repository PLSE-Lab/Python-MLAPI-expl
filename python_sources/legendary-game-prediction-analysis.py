#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data=pd.read_csv('/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')


# In[ ]:


data.head()


# In[ ]:


data.info()


# **Let's Check whether we are having missing values or not !**

# In[ ]:


sns.heatmap(data.isnull(),cmap='Blues')


# Well that's great we are having no missing values :)

# **As our data is given in a well mannered form so let's split it into Red vs Blue , Removing the game id and storing the result in a new DataFrame**

# In[ ]:


result=pd.DataFrame({'BlueWin':data['blueWins']})
result.head()


# In[ ]:


data.drop(['gameId'],inplace=True,axis=1)


# In[ ]:


data.head(1)


# In[ ]:


data.head(1)


# In[ ]:


blue_features=[]
red_features=[]
for col in list(data):
    if(col[0]=='r'):
        red_features.append(col)
    if(col[0]=='b'):
        blue_features.append(col)


# In[ ]:


blue_features


# In[ ]:


blue=data[blue_features]
red_features.append("blueWins")
red=data[red_features]


# In[ ]:


blue.head()


# In[ ]:


red.head()


# **We are now ready to take off towards visualization of our data :)**

# **Well in the below 2 graphs blue one corresponds to BLUE team win and red one corresponds to RED team win :)**

# In[ ]:


g=sns.PairGrid(data=red,hue='blueWins',palette='Set1')
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend();


# In[ ]:


g=sns.PairGrid(data=blue,hue='blueWins',palette='Set1')
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend();


# Let's check out how many times which team won most!

# In[ ]:


Red_win=len(data[data['blueWins']==0])
Blue_win=len(data['blueWins'])-len(data[data['blueWins']==1])
print(Blue_win)
print(Red_win)


# Well it's a tie betweem them :)

# In[ ]:


g = sns.countplot(x=data['blueWins'])


# **Now let's move towards predicting our model**

# As we are having both the data of both the teams so many of you will think we will predict using one of the team data and it will gonna to give correct result but it will goona to give **Baised** result so we will predict using **Principal Component Analysis** to get more accurate result

# Now we will se the corelation between **blue Wins and Red team other datasets** and will select the most negatively correlated so to train our model for the most accurate **0** value results

# In[ ]:


plt.figure(figsize=(16,5))
sns.heatmap(red.corr(),annot=True,cmap='Reds')


# In the above correlation is negative between blueWins and rest data because it is showing 1 for the **Blue win** and 0 for **Red win**

# Removing the data which doesn't giving any strong corelation

# In[ ]:


red.drop(['redWardsPlaced','redWardsDestroyed','redFirstBlood','redHeralds','redTowersDestroyed','redTotalJungleMinionsKilled','blueWins'],axis=1,inplace=True)


# In[ ]:


red.head(1)


# Now we will se the corelation between **blue Wins and Blue team other datasets** and will select the most positively correlated so to train our model for the most accurate **1** value results

# In[ ]:


plt.figure(figsize=(18,5))
sns.heatmap(blue.corr(),annot=True,cmap='Blues')


# Removing the least positively corelated items from it 

# In[ ]:


blue.drop(['blueTotalJungleMinionsKilled','blueWardsPlaced','blueWardsDestroyed','blueFirstBlood','blueHeralds','blueTowersDestroyed'],axis=1,inplace=True)


# In[ ]:


blue.head(1)


# In[ ]:


final_data=pd.concat([red,blue],axis=1)


# In[ ]:


final_data.head(2)


# **Now let's work on our Model fitting and prediction**

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# 1. **Now let's see the result if we take all of our data and use it for modeling and prediction**

# In[ ]:


x=data.drop('blueWins',axis=1)
y=data['blueWins']


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=9)


# In[ ]:


final_LR=LogisticRegression()


# In[ ]:


final_LR.fit(x_train,y_train)


# In[ ]:


result_LR=final_LR.predict(x_test)
print(accuracy_score(result_LR,y_test))


# Well on doing nothing and puting our data as it is we get an accuracy of **72%** :)

# 2. **Now let's see the result of our modified data :)**

# In[ ]:


x_mod=final_data.drop('blueWins',axis=1)
y_mod=final_data['blueWins']


# In[ ]:


x_mod_train,x_mod_test,y_mod_train,y_mod_test=train_test_split(x_mod,y_mod,test_size=0.3,random_state=9)


# In[ ]:


mod_LR=LogisticRegression()


# In[ ]:


mod_LR.fit(x_mod_train,y_mod_train)


# In[ ]:


Mod_result=mod_LR.predict(x_mod_test)


# In[ ]:


accuracy_score(Mod_result,y_mod_test)


# **Well we get an accuracy of 73% just 1 more percent than the rwa data well well well the game is savage :(**

# **Principal Component Analysis (PCA)**

# 1. **Applying on the raw data**

# In[ ]:


x=data.drop('blueWins',axis=1)
y=data['blueWins']


# Standardising our data so as to get our result much better :)

# In[ ]:


x=preprocessing.StandardScaler().fit_transform(x)


# In[ ]:


pca=PCA(n_components=2)


# In[ ]:


components=pca.fit_transform(x)


# Here in the component's we have transformed our data of many column into the 2 columns which predict our data best :)

# In[ ]:


plt.figure(figsize=(10,8))
plt.scatter(components[:,0],components[:,1],c=y,cmap='plasma')
plt.xlabel('First Principal Comp')
plt.ylabel('Second Principal COmp')


# **As you can see that we are having a different type of data which contains outlier's also in a large quantity that's why our prediction model is not giving much high accuracy**

# 2. **Modified Data**

# In[ ]:


x=final_data.drop('blueWins',axis=1)
y=data['blueWins']


# In[ ]:


x=preprocessing.StandardScaler().fit_transform(x)


# In[ ]:


pca=PCA(n_components=3)


# In[ ]:


components=pca.fit(x)


# In[ ]:


transfrom=components.transform(x)


# In[ ]:


plt.figure(figsize=(10,8))
plt.scatter(transfrom[:,0],transfrom[:,1],c=y,cmap='plasma')
plt.xlabel('First Principal Comp')
plt.ylabel('Second Principal COmp')


# **Well in this case the story remains the same :(**

# In[ ]:


components.components_


# In[ ]:


final_data.drop('blueWins',axis=1,inplace=True)


# In[ ]:


q=pd.DataFrame(components.components_,columns=final_data.columns)
q


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(q,cmap='plasma')


# Look above for the **PCA** visualization

# **I HOPE YOU WILL LIKE THIS NOTEBOOK AND IT WILL ANSWER MANY OF YOUR QUESTION'S !!!**

# In[ ]:




