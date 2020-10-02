# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
plt.rcParams['figure.figsize'] = (16,9)
plt.ion()

#### Helper Function

def show_plot(C,X):
    colors = ['r','g','b','y','c','r']
    for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        plt.scatter(points[:,0], points[:,1], s=20, c=colors[i], alpha=0.2)
        plt.scatter(C[i,0], C[i,1], marker='D', s=200, c='black', edgecolor='white')
        
    plt.draw()
    plt.pause(5)

def get_random_centroids(X,k):
    C_x = np.random.randint(0, np.max(X)-20, size=k)
    C_y = np.random.randint(0, np.max(X)-20, size=k)
    
    C= np.array(list(zip(C_x,C_y)))
    return C,C_x,C_y

#Euclidean Distance Calculator

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

################


df = pd.read_csv("/kaggle/input/dataset.csv")
print(df.head())

#get values f1, f2 and plotting it

f1= df["V1"]
f2= df["V2"]

X = list(zip(f1,f2))
print(X)


#Init K random centroids
k = 3

C,C_x,C_y = get_random_centroids(X,k)

plt.scatter(f1,f2,c='black', s=7)
plt.scatter(C_x,C_y, marker='*', s=100, c='r')
plt.show()

C_old = np.zeros(C.shape)
clusters = np.zeros(len(X))

#Kmeans
change_in_c = dist(C,C_old,None)

while change_in_c !=0:
    #part 1: Cluster Assignment. Assign points to closest cluster
    for i in range(len(X)):
        distances_x_centroids = dist(X[i],C)
        #we need to return index of the element which is minimum
        clusters[i] = np.argmin(distances_x_centroids)
    
    C_old = deepcopy(C)
    
    #part 2: cluster movement: Find the new centroids
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
        
    show_plot(C,X)
    
    change_in_c = dist(C, C_old, None)

plt.show()




