#!/usr/bin/env python
# coding: utf-8

# Hereby I will extract some info from the wide Fifa 2017 Dataset.
# I will select 31 features, X, and the player Rating as a target, y.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plotting

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


players = pd.read_csv('../input/FullData.csv',parse_dates=True)
players['Height'] = players.Height.apply(lambda x: x.replace(' cm', '')).astype('int')
players['Weight'] = players.Weight.apply(lambda x: x.replace(' kg', '')).astype('int')
players = players.dropna(axis=1)


# In[ ]:


y = players.Rating.values


# In[ ]:


plt.rcParams["patch.force_edgecolor"] = True
plt.hist(players.Rating,bins=40)
plt.xlabel('Rating')
plt.ylabel('Counts')
plt.title('Histogram of Football player Rating')
plt.show()


# In[ ]:


selected_features = ['Height', 'Weight','Ball_Control', 'Dribbling', 'Marking', 'Sliding_Tackle',
       'Standing_Tackle', 'Aggression', 'Reactions', 'Attacking_Position',
       'Interceptions', 'Vision', 'Composure', 'Crossing', 'Short_Pass',
       'Long_Pass', 'Acceleration', 'Speed', 'Stamina', 'Strength', 'Balance',
       'Agility', 'Jumping', 'Heading', 'Shot_Power', 'Finishing',
       'Long_Shots', 'Curve', 'Freekick_Accuracy', 'Penalties', 'Volleys']


# In[ ]:


players = pd.concat([players[selected_features],players.Rating],axis=1)


# Now I will convert Rating column in some numerical labels. I can first use 3 categories of player, i.e. poor, intermediate and star.

# In[ ]:


players['Rating_label'] = 0
players['Rating_label'][players.Rating > 85] = 2
players['Rating_label'][(players.Rating > 65) & (players.Rating <= 85)] = 1
players['Rating_label'][(players.Rating <= 65)] = 0


# In[ ]:


from sklearn.decomposition import PCA

model = PCA()
model.fit(X)
transformed = model.transform(X)

xs, ys = transformed[:,0], transformed[:,1]
plt.scatter(xs, ys, c=players.Rating_label.values)
plt.title('Principal Component Analysis features')
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.show()


# In[ ]:




