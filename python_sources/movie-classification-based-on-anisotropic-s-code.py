# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 19:32:46 2017

@author: Ke Wang
"""

#Movie Classification

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA # Principal Component Analysis module
from sklearn.cluster import KMeans # KMeans clustering 
import matplotlib.pyplot as plt # Python defacto plotting library
import seaborn as sns # More snazzy plotting library
from mpl_toolkits.mplot3d import Axes3D
import random
def search(num,label_color):
    word = label_color[num]
    n = len(label_color)
    new = list([])
    for i in range(n):
        if label_color[i] == word:
            new.append(i)
    n2 = len(new)
    Takei = np.arange(n2)
    np.random.shuffle(Takei)
    new1 = np.array(new)
    result = new1[Takei[0:5]]
    return result

def pca_svd(X0, n_components):
    X_mean = np.mean(X0, axis = 0)
    X1 = X0 - X_mean
    u_1,s_1,v_1 = np.linalg.svd(X1.T)
    X_trans = np.matrix(u_1)[:,0:n_components]
    X_under = np.matrix(X1) * X_trans
    return X_trans,X_under

movie = pd.read_csv(r'E:\tmdb_5000_movies.csv')
str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in movie.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
# Get to the numeric columns by inversion            
num_list = movie.columns.difference(str_list)      
movie_num = movie[num_list]
#del movie # Get rid of movie df as we won't need it now
movie_num.head()
movie_num = movie_num.fillna(value=0, axis=1)

# Extract the values in movie_num
movie_data = np.matrix(movie_num)
movie_mean = np.mean(movie_data,axis = 0)
movie_data1 = (movie_data - movie_mean)/np.sqrt(np.var(movie_data,axis = 0))


n_components = 3

M_trans,M_under = pca_svd(movie_data1,n_components) 
M_under = M_under/np.sqrt(np.var(M_under,axis = 0))
fig = plt.figure(figsize = (9,7))
ax = Axes3D(fig)
ax.scatter(np.array(M_under[:,0]),np.array(M_under[:,1]),np.array(M_under[:,2]), c='goldenrod')
plt.figure()
plt.scatter(np.array(M_under[:,0]),np.array(M_under[:,1]), c='goldenrod')


# Set a 3 KMeans clustering
kmeans = KMeans(n_clusters=5)
# Compute cluster centers and predict cluster indices
X_clustered = kmeans.fit_predict(M_under)

# Define our own color map
LABEL_COLOR_MAP = {0 : 'r',1 : 'g',2 : 'b',3:'k',4:'y'}
label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

# Plot the scatter digram
fig2 = plt.figure(figsize = (7,7))
ax1 = Axes3D(fig2)
ax1.scatter(np.array(M_under[:,0]),np.array(M_under[:,1]),np.array(M_under[:,2]), c= label_color) 

plt.figure()
plt.scatter(np.array(M_under[:,0]),np.array(M_under[:,1]),c= label_color)
# Any results you write to the current directory are saved as output.