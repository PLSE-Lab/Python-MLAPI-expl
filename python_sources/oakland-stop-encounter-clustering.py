#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import re

sns.set()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <h1> Welcome to my stop encounters kernel! </h1>
# <h2>This kernel will go over some data mining concepts to understand this data better</h2>
# 
# Table of Contents:<br>
# **1. [One-Hot-Encoding](#Encoding)** <br>
# **2. [Visualization](#Visualization)** <br>
# **3. [Clustering](#Clustering)** <br>
# **4. [Exploring classification](#Classification)**<br>
# 
# Let's start by looking at the data. 

# In[ ]:


stops = pd.read_csv('../input/stop-data-april-2013-to-sept.-2015.csv')
stops.sample(10)


# <h3>For now we will be using the following fields fields</h3>
# * ReasonForEncounter
# * ResultOfEncounter
# * ResultOfSearch
# * SDRace
# * SearchConducted
# * Sex
# * TypeOfSearch<br>
# 
# Let us look at the null values of the data

# In[ ]:


sns.heatmap(stops.isnull(), cbar=False, cmap=sns.color_palette("GnBu_d"))


# It seems that we only have missing values when a Search wasn't conducted. Then there would be no TypeOfSearch or ResultOfSearch. From the given data, we can see that there's examples where the columns we are interested in have multiple comma seperated features, such as `ResultOfSearch: 'Other Evidence,Narcotics,Firearms` 

# <a id="Encoding"></a> 
# # **1. One-Hot-Encoding:**
# 
# Let's start by removing all the trailing commas in the dataframe

# In[ ]:


def strip_comma(text):
    if pd.isna(text):
        return np.nan
    else:
        return text[:-1] if text[-1] is ',' else text


# In[ ]:


stops['ReasonForEncounter'] = stops['ReasonForEncounter'].apply(strip_comma)
stops['ResultOfEncounter'] = stops['ResultOfEncounter'].apply(strip_comma)
stops['ResultOfSearch'] = stops['ResultOfSearch'].apply(strip_comma)
stops['SDRace'] = stops['SDRace'].apply(strip_comma)
stops['SearchConducted'] = stops['SearchConducted'].apply(strip_comma)
stops['TypeOfSearch'] = stops['TypeOfSearch'].apply(strip_comma)
stops['Sex'] = stops['Sex'].apply(strip_comma)


# `ReasonForEncounter` looks ready to be one-hot-encoded, but let's just drop the one row which has `Other-Consensual` as it's reason

# In[ ]:


print('Before:')
print(stops['ReasonForEncounter'].value_counts())
stops = stops[~(stops['ReasonForEncounter'] == 'Other-Consensual')]
print()
print('After:')
print(stops['ReasonForEncounter'].value_counts())


# We can see that there's two values `FI Report,FI Report ` and `Report Taken-No Action,Report Taken-No Action ` which seem to be typos

# In[ ]:


print('Before:')
print(stops['ResultOfEncounter'].value_counts())
stops['ResultOfEncounter'].replace('Report Taken-No Action,Report Taken-No Action', 'Report Taken-No Action', inplace=True)
stops['ResultOfEncounter'].replace('FI Report,FI Report', 'FI Report', inplace=True)
print()
print('After:')
print(stops['ResultOfEncounter'].value_counts())


# In[ ]:


print('Before:')
print(stops['ResultOfSearch'].value_counts())


# This will require a more complicated means of One Hot Encoding because each value in the current dataframe may need multiple values to be hot.

# In[ ]:


stops['ResultOfSearch'] = stops['ResultOfSearch'].str.replace('&',',')
stops['ResultOfSearch'] = stops['ResultOfSearch'].str.replace(' , ',',')


# In[ ]:


stops = pd.concat([stops, stops['ResultOfSearch'].str.get_dummies(sep=',')], axis=1, sort=False); 


# In[ ]:


print('Before:')
print(stops['TypeOfSearch'].value_counts())
print()
print('After:')
stops = stops[~(stops['TypeOfSearch'] == 'ContactDate')]
stops['TypeOfSearch'].value_counts()


# In[ ]:


race = stops['SDRace']
encounters = pd.get_dummies(stops, columns=['ReasonForEncounter','ResultOfEncounter', 'SDRace', 'SearchConducted', 'Sex', 'TypeOfSearch'])
encounters.drop(columns=['ContactDate','ContactTime', 'Location 1','ResultOfSearch'], inplace=True)
encounters.head()


# <h3>We've successfully one-hot-encoded this column!<h3>
# <a id="Visualization"></a> 
# # **2. Visualization:**
# Let's generate a correlation matrix

# In[ ]:


def plot_corr_seaborn(df_corr,threshold):
    mask = np.zeros_like(df_corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 15)
        fig.dpi = 100
        ax = sns.heatmap(df_corr, mask=mask, vmax=1, square=True)


# Let's see the highest correlation values for each attribute.

# In[ ]:


pd.Series.to_frame(encounters.corr().replace(1,0).max()).T


# In[ ]:


plot_corr_seaborn(encounters.corr(), 1)


# In[ ]:


felony_arrests = {'Male': len(encounters[(encounters['Sex_M'] == 1) & (encounters['ResultOfEncounter_Felony Arrest'] == 1)])/len(encounters[encounters['Sex_M'] == 1]), 
                  'Female': len(encounters[(encounters['Sex_F'] == 1) & (encounters['ResultOfEncounter_Felony Arrest'] == 1)])/len(encounters[encounters['Sex_F'] == 1])}
plt.bar(felony_arrests.keys(), felony_arrests.values())

plt.ylabel('Percentage of all interactions')
plt.xlabel('Genders')
plt.title('Felony Arrest frequency')


# In[ ]:


felony_arrests = {'Male': len(encounters[(encounters['Sex_M'] == 1) & (encounters['ResultOfEncounter_Misdemeanor Arrest'] == 1)])/len(encounters[encounters['Sex_M'] == 1]), 
                  'Female': len(encounters[(encounters['Sex_F'] == 1) & (encounters['ResultOfEncounter_Misdemeanor Arrest'] == 1)])/len(encounters[encounters['Sex_F'] == 1])}
plt.bar(felony_arrests.keys(), felony_arrests.values())
plt.title('Felony Misdemeanor frequency')
plt.ylabel('Percentage of all interactions')
plt.xlabel('Genders')


# It seems that women are much more likely to get charged with Felony Misdemeanor than Felony Arrest, 
# whereas men are more likely to be charged with Felony Arrest than Felony Misdemeanor
# 
# <a id="Clustering"></a> 
# # **3. Clustering:**
# Let's perform k-means clustering with doubling k-values to see which value of k will be the optimal cluster numbers. This will be when adding more clusters will not raise the model score by much.

# In[ ]:


# clustering for k = 2 to 32 by doubling values.
from sklearn.cluster import KMeans
max_k = 7
ks = [2**k for k in range(1,max_k+1)]
scores = []

for k in ks:
    model = KMeans(n_clusters=k, n_jobs=-1, init='k-means++')
    model.fit_predict(encounters.values)
    scores.append(-model.score(encounters.values))


# In[ ]:


fig = plt.figure(figsize = (8,8))
plt.plot(ks, scores)
plt.title('cost-clusters tradeoff')
plt.ylabel('total intra-cluster distance')
plt.xlabel('k')
plt.show()


# Cluster size of 8 seems to be optimal, given that we are happier off maintining a higher score with less amount of clusters. k=16 may as well be good choice too. it is usually not a good sign if there's no clear sign of an elbow point, but for this data k=8 or k=16 are arguably the best sizes to pick here.

# In[ ]:


kmeans = KMeans(n_clusters=8, n_init=50, n_jobs=-1, init='k-means++', random_state=0)
kmeans.fit_predict(encounters.values);


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(encounters)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

y_kmeans = kmeans.predict(encounters)


# In[ ]:


fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 10)
ax.set_ylabel('Principal Component 2', fontsize = 10)
ax.set_title('2 component PCA', fontsize = 20)
ax.scatter(principalDf['principal component 1'], principalDf['principal component 2'], c= y_kmeans)
plt.show()


# In[ ]:


encounters['Cluster'] = y_kmeans


# In[ ]:


encounters[encounters['Cluster'] == 0].describe().loc[['mean']]


# In[ ]:


encounters[encounters['Cluster'] == 1].describe().loc[['mean']]


# In[ ]:


encounters[encounters['Cluster'] == 2].describe().loc[['mean']]


# In[ ]:


encounters[encounters['Cluster'] == 3].describe().loc[['mean']]


# In[ ]:


encounters[encounters['Cluster'] == 4].describe().loc[['mean']]


# In[ ]:


encounters[encounters['Cluster'] == 5].describe().loc[['mean']]


# In[ ]:


encounters[encounters['Cluster'] == 6].describe().loc[['mean']]


# In[ ]:


encounters[encounters['Cluster'] == 7].describe().loc[['mean']]


# In[ ]:


assert len(encounters) == len(race)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split


# In[ ]:


x_cols = [ 'ReasonForEncounter_Consensual Encounter', 'ReasonForEncounter_Probable Cause', 'ReasonForEncounter_Probation/Parole', 'ReasonForEncounter_Reasonable Suspicion',
          'ReasonForEncounter_Traffic Violation', 'SearchConducted_No', 'SearchConducted_Yes', 'Sex_F', 'Sex_M',
          'ResultOfEncounter_Citation','ResultOfEncounter_FI Report', 'ResultOfEncounter_Felony Arrest', 'ResultOfEncounter_Misdemeanor Arrest', 
          'ResultOfEncounter_Report Taken-No Action', 'ResultOfEncounter_Warning']
output = pd.Categorical(race).codes
print(output)
print(dict( enumerate(pd.Categorical(race).categories) ))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(encounters[x_cols].values, output, test_size=0.3)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[ ]:


model = Sequential([
    Dense(15),
    Activation('relu'),
    Dense(11),
    Activation('relu'),
    Dense(7),
    Activation('softmax')
])


# In[ ]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15)
model.evaluate(X_test, y_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=7)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


# This hints that it's not possible to classify stop consequences based on logistics

# In[ ]:




