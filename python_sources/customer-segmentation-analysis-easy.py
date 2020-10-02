# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plot some graphs, scatter plot in here 
import seaborn as sns # to beautify matplots
from sklearn.cluster import KMeans # for clustering data using KMeans modeling 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
 
# Any results you write to the current directory are saved as output.
df = pd.read_csv(r'../input/Mall_Customers.csv') # Reading csv file from the environment
print(df.head()) #printing 1st 5 rows of dataframe
print(df.describe()) # printing basic information about data like mean, median etc
print(df.info()) # printing information about metadata like dtypes, not null entries

sns.set() 
plt.style.use('ggplot')
# Performing basic Exploratory Data Analysis on important coloumns like Annual Income and Spending score
pd.plotting.scatter_matrix(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], alpha = 0.6, figsize = [12,10], diagonal = 'hist', marker = 'o', color = '#66b3ff', hist_kwds={'color':['burlywood']})
plt.savefig('Graphical EDA.png')
plt.clf()
# Analysing Gender performance and ploting a pie plot
sizes = [df.Gender[df.Gender == 'Female'].count(),df.Gender[df.Gender == 'Male'].count()]
ratio = (sizes[0]/sizes[1])
print('For each male {} female(s) shop '.format(round(ratio,2)))
plt.figure(figsize = (5,5))
plt.pie(sizes, labels = ['Female','Male'], colors = ['#c2c2f0','#ffb3e6'], radius = 0.75, autopct = '%1.0f%%')
plt.savefig('Gender Ratio.png')
plt.clf()
#Modeling and evaluating cluster between Age and Spending Score
ds1 = df[['Age', 'Spending Score (1-100)']]
Ks = range(1,10)
inertia = []
for k in Ks:
    model = KMeans(n_clusters = k, random_state = 86)
    model.fit(ds1)
    inertia.append(model.inertia_)

# Ploting Inertia Plot to find best number of clusters
plt.plot(Ks, inertia, 'o')
plt.plot(Ks, inertia, '-', alpha = 0.45)
plt.ylabel('Inertia')
plt.xlabel('No. of Clusters')
plt.title('Interia plot for Age vs. Spending Score')
plt.savefig('Inertia Plot 1.png')
plt.clf()
# Number of clusters should be 4
model1 = KMeans(n_clusters = 4, random_state = 86)
model1.fit(ds1)
model1_pred = model1.predict(ds1)

labels1 = ['Usual Shoppers','Young Shopaholic','Old Shoppers','Young Shoppers'] 
color1 = ['Blue','Green','Yellow','Violet']
d1=dict(zip(np.sort(np.unique(model1_pred)),labels1))
dc1=dict(zip(np.sort(np.unique(model1_pred)),color1))

# Converting dataframe into a numpy array
sc_x1 = np.array(ds1.iloc[:, 0])
sc_y1 = np.array(ds1.iloc[:, 1])
# Increasing the default figure size
plt.figure(figsize = (9,9))
for g in np.unique(model1_pred):
    ix = np.where(model1_pred == g)
    plt.scatter(sc_x1[ix], sc_y1[ix], c=dc1[g], s=50, cmap='viridis', label = d1[g])

# Adding centroids to the previous plot
plt.scatter(model1.cluster_centers_[:,0], model1.cluster_centers_[:,1], s=100, c='black', label = 'Centroid')
plt.ylabel('Spending Score (1-100)')
plt.xlabel('Age')
plt.title('Age vs. Spending Score')
plt.legend(loc='best')
plt.savefig('Clustered Scatter plot 1.png')
plt.clf()
#Modeling and evaluating cluster between Age and Spending Score
ds2 = df[['Annual Income (k$)', 'Spending Score (1-100)']]
Ks = range(1,10)
inertia = []
for k in Ks:
    model = KMeans(n_clusters = k, random_state = 86)
    model.fit(ds2)
    inertia.append(model.inertia_)

# Ploting Inertia Plot to find best number of clusters again
plt.plot(Ks, inertia, 'o')
plt.plot(Ks, inertia, '-', alpha = 0.45)
plt.ylabel('Inertia')
plt.xlabel('No. of Clusters')
plt.title('Interia plot for Annual Income vs. Spending Score')
plt.savefig('Inertia Plot 2.png')
plt.clf()
# Number of clusters should be 5
model2 = KMeans(n_clusters = 5, random_state = 86)
model2.fit(ds2)
model2.labels_
model2_pred = model2.predict(ds2)

labels2 = [ 'Cheapstake', 'Squanderer','Economical', 'General', 'In Debt']
color2 = ['Pink','Blue','Orange','Violet','Yellow']
d2=dict(zip(np.sort(np.unique(model2_pred)),labels2))
dc2=dict(zip(np.sort(np.unique(model2_pred)),color2))

# Converting dataframe into a numpy array again
sc_x2 = np.array(ds2.iloc[:, 0])
sc_y2 = np.array(ds2.iloc[:, 1])
plt.figure(figsize = (11,9))

for g in np.unique(model2_pred):
    ix = np.where(model2_pred == g)
    plt.scatter(sc_x2[ix], sc_y2[ix], c=dc2[g], s=50, cmap = 'plasma', label = d2[g])
plt.scatter(model2.cluster_centers_[:,0], model2.cluster_centers_[:,1], s=100, c='black', label = 'Centroid')
plt.ylabel('Spending Score (1-100)')
plt.xlabel('Annual Income (k$)')
plt.title('Annual Income vs. Spending Score')
plt.legend(loc='best')
plt.savefig('Clustered Scatter plot 2.png')
plt.clf()