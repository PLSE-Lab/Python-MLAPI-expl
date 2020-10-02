#!/usr/bin/env python
# coding: utf-8

# The National Hockey League is a professional ice hockey league with 30 teams based in
# Canada and the United States. The NHL is the greatest ice hockey league in the world and is
# divided into 2 conferences: Eastern and Western, that have 16 and 14 teams respectively and are
# divided in 2 equally sized divisions. The goal of each team is to win the Stanley Cup. To qualify
# for the right to challenge for the Stanley Cup a team must make the playoffs. After 82 regular
# season games the top 3 teams of each division qualify along with the top 2 teams in the
# conference that did not qualify for a divisional spot. Teams are ranked based on the number of
# points they have obtained, which are 2 for a win and 1 for an overtime loss.
# 
# How effectively can we predict if a team is going to make the playoffs given team stats excluding points.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# I decided to look into different models for predicting how effectively I could use team statistics that weren't  points. The ideal method was to use Random Forest (15 trees) to classify weather a team was a playoff team with the input set [Goal Differential, Penalty Minutes,  Power Play %, Penalty Kill %]  and the output was whether or not the team made the playoffs.
# 
# Only data after 1963 was usable since special team statistics were not tracked before 1963

# In[ ]:


df = pd.read_csv('../input/Teams.csv')
df = df[df['year'] >= 1963]

dataSet = pd.concat([df['year'],
                     df['GF'] - df['GA'],
                     df['PIM'],
                     df['PPG'] / df['PPC'],
                     1 - (df['PKG'] / df['PKC'])],
                    axis=1, keys=['Year', 'Goal Diffrential', 'PIM', 'PPP', 'PKP'])
dataSet['Playoff'] = [pd.isnull(x) for x in df['playoff']]
dataSet = dataSet[pd.notnull(dataSet['PPP'])]
print(dataSet.head())


# After setting up the data set break the set into a training and a testing set.
# 
# The NHL has changed throughout the ages, with goalies stopping more shots and players constantly pushing the boundaries on their physical capabilities.
# 
# Since this data set ends in 2011 I am going to use the prediction accuracy to find the start of the most recent era of hockey.

# In[ ]:


year = []
accuracy = []
for start in range(1963, 2011):
    count = 0
    total = 0
    while total < 100:
        data = dataSet[dataSet['Year'] >= start]
        data['is_train'] = np.random.uniform(0, 1, len(data)) <= .9    
        train, test = data[data['is_train'] == True], data[data['is_train'] == False]
        clf = RandomForestClassifier(n_jobs=2, n_estimators=15)
        features = data.columns[1:5]
        clf.fit(train[features], train['Playoff'])
    
        for i, j in zip(clf.predict(test[features]), test['Playoff']):
            total += 1
            if i == j:
                count += 1
    ratio = count / total
    year.append(start)
    accuracy.append(ratio)
plt.plot(year, accuracy)
plt.xlabel('Year')
plt.axis([1963, 2010, .7, 1])
plt.ylabel('Classifier Accuracy')
plt.show()
    


# **Conclusion** 
# 
# As we can see, by using a random forest we can effectively classify if a team will make the NHL playoffs given their [Goal Differential, Penalty Minutes,  Power Play %, Penalty Kill %] . 
# 
# The reason for the data from the 1990's having higher accuracy than the 1980's is due to the fact that 9 franchises were added in the 90's while 16 teams continued to make the playoffs each year, this meant that the requirements to make the playoffs was different between the eras.
# 
# Thanks for reading my first kernel, I would really appreciate any feedback 

# In[ ]:


contender = []
for year in range (1970, 2012):
    df_year = df[df['year'] == year]
    d = df_year.sort_values('Pts', ascending=False)
    length = len(d)
    if length == 0:
        continue
    contender_points = d.iloc[length//3, 13] # take the top 30%
    
    for i, r in df_year.iterrows():
        if r['Pts'] >= contender_points:
            contender.append(True)
        else:
            contender.append(False)
data = dataSet[dataSet['Year'] >= 1970]
print(len(data), len(contender))
data['Cont'] = contender
print(data.head())

    
    

