#!/usr/bin/env python
# coding: utf-8

# # The EEG data is for the events of recording eye or eyes (not sure) in open or closed positions

# A simple anaylsis of the data is perform on the data. <br>
#  <a href='#Data_Precentage'>Data split percentage</a><br>
#  <a href='#mean_sd'>Means and Standard Deviations</a><br>
#  <a href='#corr_hist_norms'>Correlations, Histograms and Normalizations</a>

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

import os
print(os.listdir("../input"))

inputData = pd.read_csv(r"../input/eeg_clean.csv");


# In[ ]:


print(inputData.dtypes)
print(inputData.columns)
print("Data shape:",inputData.shape)
print(inputData.head())
print(inputData.describe())
print(inputData.info())
# Check for any nulls
print(inputData.isnull().sum())


# # Precentage of data that is split into open and closed eyes 
# <a id='Data_Precentage'></a>

# In[ ]:


plt.figure( figsize=(10,10))
inputData['eye'].value_counts().plot.pie(autopct="%1.1f%%")
plt.title("Data division on eye state (open/closed)",fontsize=10)
plt.show()


# # Lets find mean and standard deviation
# <a id='mean_sd'></a>

# In[ ]:


length  = len(inputData.columns)-1
colors  = ['xkcd:cloudy blue', 'xkcd:dark pastel green', 'xkcd:dust', 'xkcd:electric lime', 'xkcd:fresh green', 'xkcd:light eggplant', 'xkcd:nasty green', 'xkcd:really light blue', 'xkcd:tea', 'xkcd:warm purple', 'xkcd:yellowish tan', 'xkcd:cement', 'xkcd:dark grass green', 'xkcd:dusty teal'] 

print ("***************************************")
print ("DISTIBUTION OF VARIABLES IN DATA SET")
print ("***************************************")
dropped = inputData.drop(["eye"],axis=1)
plt.figure(figsize=(15,30))
# Leavout the last column of Outcome
for i,j,k in itertools.zip_longest(dropped.columns[:],range(length),colors):
    plt.subplot(length/2,length/4,j+1)
    sns.distplot(dropped[i],color=k)
    plt.title(i)
    plt.subplots_adjust(hspace = .3)
    plt.axvline(dropped[i].mean(),color = "k",linestyle="dashed",label="MEAN")
    plt.axvline(dropped[i].std(),color = "b",linestyle="dotted",label="STANDARD DEVIATION")
    plt.legend(loc="upper right")
plt.show()    


# # Correlations, histograms, pair plots - With tests on normalizations
# <a id='corr_hist_norms'></a>

# In[ ]:


fig = plt.figure(figsize = (10,10))
ax = fig.gca()
sns.heatmap(inputData.corr(), annot=True, fmt=".2f")
plt.title("Correlation",fontsize=5)
plt.show()


# In[ ]:


# Lets take a few samples and plot. Taking whole data plot is too much a wait
eyeOpen = inputData.loc[inputData["eye"]=="Open"].sample(frac=0.01)
eyeClose = inputData.loc[inputData["eye"]=="Closed"].sample(frac = 0.01)
v = pd.concat([eyeOpen,eyeClose])


sns.pairplot(data=v,hue="eye")
plt.title("Skewness",fontsize =10)
plt.show()

fig = plt.figure(figsize = (10,10))
ax = fig.gca()
sns.scatterplot(x="O1", y="O2", hue="eye",data=v)
plt.title("O1 vs O2 on eye",fontsize =10)
plt.show()

fig = plt.figure(figsize=(10,10))
ax = fig.gca()
v.hist(ax=ax)
plt.xlabel('Probes',fontsize=10)
plt.ylabel('Counts',fontsize=10)
ax.tick_params(labelsize=10)
plt.title('Recording probes count',fontsize=10)
plt.grid()
plt.ioff()
plt.show()


# # Row-wise unit norm normalization 

# In[ ]:


# Lets normalize the data via l2 unit norm (row-wise) to see what happens
print ("******************************************")
print ("ROWISE UNIT NORM NORMALIZATION OF DATA AND")
print ("******************************************")
from sklearn import preprocessing
v2 = v.copy()
v3 = v2.loc[:, v.columns != 'eye']
v3 = preprocessing.normalize(v3, norm='l2',axis =1)
v2.loc[:, v2.columns != 'eye'] = v3;

sns.pairplot(data=v,hue="eye")
plt.title("Skewness",fontsize =10)
plt.show()

fig = plt.figure(figsize = (10,10))
ax = fig.gca()
sns.scatterplot(x="O1", y="O2", hue="eye",data=v2)
plt.title("O1 vs O2 on eye",fontsize =10)
plt.show()

fig = plt.figure(figsize=(10,10))
ax = fig.gca()
v2.hist(ax=ax)
plt.xlabel('Probes',fontsize=10)
plt.ylabel('Counts',fontsize=10)
ax.tick_params(labelsize=10)
plt.title('Recording probes count',fontsize=10)
plt.grid()
plt.ioff()
plt.show()


# In[ ]:


# Row-wise MinMax normalization


# In[ ]:


print ("****************************")
print ("MINMAX NORMALIZATION OF DATA")
print ("****************************")
v2 = v.copy()
v3 = v2.loc[:, v.columns != 'eye']
v3 = v3.apply(lambda x:(x.astype(float) - min(x))/(max(x)-min(x)), axis = 1)
v2.loc[:, v.columns != 'eye'] = v3;

sns.pairplot(data=v,hue="eye")
plt.title("Skewness",fontsize =10)
plt.show()

fig = plt.figure(figsize = (10,10))
ax = fig.gca()
sns.scatterplot(x="O1", y="O2", hue="eye",data=v2)
plt.title("O1 vs O2 on eye",fontsize =10)
plt.show()

fig = plt.figure(figsize=(10,10))
ax = fig.gca()
v2.hist(ax=ax)
plt.xlabel('Probes',fontsize=10)
plt.ylabel('Counts',fontsize=10)
ax.tick_params(labelsize=10)
plt.title('Recording probes count',fontsize=10)
plt.grid()
plt.ioff()
plt.show()


# In[ ]:




