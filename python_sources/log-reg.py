#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

data = pd.read_csv("../input/character-predictions.csv")




# In[ ]:


cult = {
    'Summer Islands': ['summer islands', 'summer islander', 'summer isles'],
    'Ghiscari': ['ghiscari', 'ghiscaricari',  'ghis'],
    'Asshai': ["asshai'i", 'asshai'],
    'Lysene': ['lysene', 'lyseni'],
    'Andal': ['andal', 'andals'],
    'Braavosi': ['braavosi', 'braavos'],
    'Dornish': ['dornishmen', 'dorne', 'dornish'],
    'Myrish': ['myr', 'myrish', 'myrmen'],
    'Westermen': ['westermen', 'westerman', 'westerlands'],
    'Westerosi': ['westeros', 'westerosi'],
    'Stormlander': ['stormlands', 'stormlander'],
    'Norvoshi': ['norvos', 'norvoshi'],
    'Northmen': ['the north', 'northmen'],
    'Free Folk': ['wildling', 'first men', 'free folk'],
    'Qartheen': ['qartheen', 'qarth'],
    'Reach': ['the reach', 'reach', 'reachmen'],
}

def get_cult(value):
    value = value.lower()
    v = [k for (k, v) in cult.items() if value in v]
    return v[0] if len(v) > 0 else value.title()

data.loc[:, "culture"] = [get_cult(x) for x in data.culture.fillna("")]

#-------- culture induction -------


# In[ ]:


datacopy = data.copy()


data.drop(["name", "alive", "pred", "plod", "isAlive", "dateOfBirth", "DateoFdeath"], 1, inplace = True)


data.loc[:, "title"] = pd.factorize(data.title)[0]
data.loc[:, "culture"] = pd.factorize(data.culture)[0]
data.loc[:, "mother"] = pd.factorize(data.mother)[0]
data.loc[:, "father"] = pd.factorize(data.father)[0]
data.loc[:, "heir"] = pd.factorize(data.heir)[0]
data.loc[:, "house"] = pd.factorize(data.house)[0]
data.loc[:, "spouse"] = pd.factorize(data.spouse)[0]

data.fillna(value = -1, inplace = True)
''' $$ The code below usually works as a sample equilibrium. However in this case,
 this equilibirium actually decrease our accuracy, all because the original 
prediction data was released without any sample balancing. $$

data = data[data.actual == 0].sample(350, random_state = 62).append(data[data.actual == 1].sample(350, random_state = 62)).copy(deep = True).astype(np.float64)

'''
Y = data.actual.values

Odata = data.copy(deep=True)

data.drop(["actual"], 1, inplace = True)

# sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
# fig=plt.gcf()
# fig.set_size_inches(30,20)
# plt.show()



from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(data, Y)

# plt.scatter(data.loc[:, "house"],Y)
# plt.xlabel("Data")
# plt.ylabel("Prediction")

for index,row in datacopy.iterrows():
    if(Y[index] == 1):
        survived = (datacopy.loc[:, "name"])
# print(data.columns)
print(survived)
survived.to_csv("../survived.csv")
print('LogisticRegression Accuracy: ',LR.score(data, Y))



# In[ ]:




