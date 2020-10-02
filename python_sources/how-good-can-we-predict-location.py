#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/indoor-positioning/beacon_readings.csv")
data['Time'] = data['Time'].map(lambda l: l.split(' ')[2])
data['Hours'] = data['Time'].map(lambda l: l.split(':')[0])
data['Minutes'] = data['Time'].map(lambda l: l.split(':')[1])
data['Seconds'] = data['Time'].map(lambda l: l.split(':')[2])
data['Time'] = (data['Minutes'].values.astype('float')*60+                data['Seconds'].values.astype('float')) -                 data['Minutes'].values.astype('float')[0]*60
data['Time'] = data['Time'].map(lambda l:int(round(l)))
data['Time'] = data['Time']-data['Time'].values[0]
data.drop(['Hours','Minutes','Seconds','Date'],axis=1,inplace=True)
data['Distance A'] = data['Distance A'].replace(to_replace=0, method='ffill')
data['Distance B'] = data['Distance B'].replace(to_replace=0, method='ffill')
data['Distance C'] = data['Distance C'].replace(to_replace=0, method='ffill')
data = data.set_index(['Time'])


# In[ ]:


from sklearn.cluster import KMeans

position_data = data.values[:,3:]
km = KMeans(5)
km.fit_transform(position_data)

f = plt.figure()
plt.title('Locations')
plt.scatter(position_data[:,0]+np.random.randn(position_data.shape[0]),
            position_data[:,1]+np.random.randn(position_data.shape[0]),
            c=km.labels_,cmap="viridis_r")
cbar = plt.colorbar()
cbar.set_ticks(range(5))
cbar.set_ticklabels(range(5))

data['Labels'] = km.labels_
data.drop('Position X',axis=1,inplace=True)
data.drop('Position Y',axis=1,inplace=True)


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier


classifiers_names = ["Nearest Neighbors", "Random Forest","Extremely Randomized"]


classifiers = [
    KNeighborsClassifier(5),
    RandomForestClassifier(33),
    ExtraTreesClassifier(33)]

X = data.values[:,:3]
y = data.values[:,3]
skf = [(trIdx,tsIdx) for (trIdx,tsIdx) in StratifiedKFold(3).split(X,y)]

mean_scores = np.zeros_like(classifiers)
std_scores  = np.zeros_like(classifiers)
for (idx,clf) in enumerate(classifiers):
    this_clf_score = []
    for (train_index, test_index) in skf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]       
        clf.fit(X_train, y_train)
        this_clf_score.append(clf.score(X_test, y_test))
    mean_scores[idx] = np.mean(np.array(this_clf_score))
    std_scores[idx]  = np.std(np.array(this_clf_score))  

print(mean_scores)
plt.figure(figsize=(13,5))
plt.bar(np.arange(len(classifiers)),mean_scores,width=0.4,yerr=std_scores,alpha=0.4)
plt.xticks(np.arange(len(classifiers)),classifiers_names,fontsize=13)
plt.ylabel("Mean Accuracy",fontsize=15)


# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(3)
X_pca = pca.fit_transform(X)

# to be visualized in seaborn
pca_df = pd.DataFrame(X_pca)
pca_df.columns = ['PCA Dimension 1','PCA Dimension 2','PCA Dimension 3']
pca_df['Labels'] = y

f = plt.figure(figsize=(13,13))
sns.pairplot(data,hue='Labels')
sns.pairplot(pca_df,hue='Labels')


# In[ ]:


from sklearn.model_selection import GridSearchCV,StratifiedKFold

from sklearn.svm import SVC

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)

mean_scores = np.zeros((int(max(y))+1,))
std_scores = np.zeros_like(mean_scores)

for idx in np.unique(y):
    y_ova = np.zeros_like(y)
    y_ova[y==idx] = 1
    temp_score = []
    for (train_index, test_index) in skf:
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y_ova[train_index], y_ova[test_index]       
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(3))
        grid.fit(X_train, y_train)
        clf = SVC(kernel='rbf',C=grid.best_params_['C'],gamma=grid.best_params_['gamma'])
        clf.fit(X_train, y_train)
        temp_score.append(clf.score(X_test, y_test))
    mean_scores[idx] = np.mean(np.array(temp_score))
    std_scores[idx] = np.std(np.array(temp_score)) 
    
print(mean_scores)
plt.figure(figsize=(13,5))
plt.title("One-vs-All SVM")
plt.bar(np.arange(len(mean_scores)),mean_scores,width=0.4,yerr=std_scores,alpha=0.4)
#plt.xticks(np.arange((int(max(y))+1),classifiers_names,fontsize=13)
plt.ylabel("Mean Accuracy",fontsize=15)  
plt.xlabel("Locations")


# In[ ]:




