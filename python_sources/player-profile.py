#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans


# In[ ]:


path='../input/data.csv'
dl=pd.read_csv(path)
dl.head()


# In[ ]:


dl.columns


# In[ ]:


dl.shape


# In[ ]:


dl.describe()


# In[ ]:


dl.nunique()


# In[ ]:


dl.dtypes


# In[ ]:


dl.isnull().sum()


# In[ ]:


dl['Finishing'].fillna(dl['Finishing'].mean(), inplace = True)
dl['ShotPower'].fillna(dl['ShotPower'].mean(), inplace = True)
dl['HeadingAccuracy'].fillna(dl['HeadingAccuracy'].mean(), inplace = True)
dl['Volleys'].fillna(dl['Volleys'].mean(), inplace = True)
dl['Curve'].fillna(dl['Curve'].mean(), inplace = True)


# In[ ]:


dl['Strength'].fillna(dl['Strength'].mean(), inplace = True)
dl['BallControl'].fillna(dl['BallControl'].mean(), inplace = True)
dl['Aggression'].fillna(dl['Aggression'].mean(), inplace = True)
dl['Interceptions'].fillna(dl['Interceptions'].mean(), inplace = True)
dl['Marking'].fillna(dl['Marking'].mean(), inplace = True)
dl['StandingTackle'].fillna(dl['StandingTackle'].mean(), inplace = True)


# In[ ]:


plt.rcParams['figure.figsize']=(20,20)
hm=sns.heatmap(dl[['Age', 'Overall', 'Potential','Acceleration', 'Aggression', 'Agility', 'Balance', 'BallControl','Body Type','Composure', 'Crossing','Dribbling', 'FKAccuracy',
                   'Finishing','HeadingAccuracy', 'Interceptions','Jumping', 'LongPassing', 'LongShots','Marking', 'Penalties', 'Positioning','ShortPassing', 'ShotPower',
                   'Skill Moves', 'SlidingTackle','SprintSpeed', 'Stamina', 'StandingTackle', 'Strength', 'Vision','Volleys']].corr(), annot = True)


# In[ ]:


dl['Offensive']=dl[['Finishing','ShotPower','HeadingAccuracy','Volleys','Curve']].mean(axis=1)
dl['Middle']=dl[['Crossing','ShortPassing','Dribbling','LongPassing','Aggression','Interceptions']].mean(axis=1)
dl['Defensive']=dl[['BallControl','Strength','Aggression','Interceptions','Marking','StandingTackle','SlidingTackle']].mean(axis=1)


# In[ ]:


Info=dl[['Age', 'Overall', 'Potential','Offensive','Middle','Defensive','Position']]
Info.head()


# In[ ]:


n = 0 
for x in ['Offensive','Middle','Defensive']:
    for y in ['Offensive','Middle','Defensive']:
        n += 1
        plt.subplot(3 , 3 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        sns.regplot(x = x , y = y , data = dl)
        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )
plt.show()


# In[ ]:


n = 0 
for cols in ['Offensive','Middle','Defensive']:
    n += 1 
    plt.subplot(1 , 3 , n)
    sns.violinplot(x = cols , y = 'Position' , data = dl )
plt.show()


# In[ ]:


for pos in ['ST','CB']:
    plt.scatter(x = 'Offensive' , y = 'Defensive' , data = dl[dl['Position'] == pos] ,label = pos)
plt.xlabel('off'), plt.ylabel('def') 
plt.legend()
plt.show()


# In[ ]:



X=dl.loc[dl['Position'].isin(['ST', 'CB']),['Offensive','Defensive']]   
X.head()


# In[ ]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)


# In[ ]:


plt.scatter(X['Offensive'], X['Defensive'], c=y_kmeans, s=200, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

