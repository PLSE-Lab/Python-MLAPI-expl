#!/usr/bin/env python
# coding: utf-8

# Hello everyone. At this notebook, we will examine the data and then apply some clustering technique. I am using to practice and I will be really glad if I receive tips fom you guys. If in someway, I am helping anyone to improve his knowledge, please let me know :).
# 
# So, firstly, we're gonna download the data and have a look at its columns

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

data = pd.read_csv(os.path.join("../input", "Mall_Customers.csv")) #here we are dowloading the data. 
data.head(10)


# As you guys can see above, the data has 5 columns, where:
# 
# there is a customer ID to have a reference about a customer
# His gender, divided in Male and Female.
# His age
# His annual income
# And his spending score.
# Next step is to see the range for all the variables and see if there is outliers

# In[ ]:


data.info()

AgePlot = data['Age'].value_counts().sort_values()
GenderPlot = data['Gender'].value_counts().sort_values()
IncomePlot = data['Annual Income (k$)'].value_counts().sort_values()
ScorePlot = data['Spending Score (1-100)'].value_counts().sort_values()
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

axs[0, 0].bar(AgePlot.index, AgePlot)
axs[0, 0].title.set_text('Age Plot')
axs[1, 0].bar(GenderPlot.index, GenderPlot)
axs[1, 0].title.set_text('Gender Plot')
axs[0, 1].bar(IncomePlot.index, IncomePlot)
axs[0, 1].title.set_text('Income Plot')
axs[1, 1].bar(ScorePlot.index, ScorePlot)
axs[1, 1].title.set_text('Score Plot')


# As you can see above, there is some outliers in the income plot, however it won't be necessary a issue. In order to improve visualization, in the income and score plots, it can be used ranges of values to reduce the number of features in the x axis.
# 
# The next step is to get a deeper understand of the data. Let's now plot how the data is distributed using different colors from male and female, the axis Y as the income plot and the axis X as the age.

# In[ ]:


colors = { 'Male' : 'blue', 'Female': 'red'}
plt.scatter(data['Age'], data['Annual Income (k$)'], c = data['Gender'].apply(lambda x: colors[x]))
plt.show()


# As detailed above, we can see that, there aren't some strong differences about Gender and their anual income. When look at age, we can see that the people who receives more money, are the ones between 27 and 50 (by the graphic we conclude that young or retired people at this case does not have more than 100k annual income).
# 
# the next step then, it's to analyse each isolated variable (Age and anual income, with the Spending Score

# In[ ]:


fig, axs1 = plt.subplots(1, 2, figsize=(8,8))
colors = { 'Male' : 'blue', 'Female': 'red'}
axs1[0].scatter(data['Age'], data['Spending Score (1-100)'], c = data['Gender'].apply(lambda x: colors[x]))
axs1[0].title.set_text('Spending Score by Age')
axs1[1].scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c = data['Gender'].apply(lambda x: colors[x]))
axs1[1].title.set_text('Spending Score by Annual Income')


# By the graphics above, we can see some interestingclusters. For example, looking to age. we can see a correlation about who has a great spending score and a young age. Looking to annual income, we can see that people who earns around 50k, has a similar spending score.
# 
# and once again, gender doesn't seems to have a huge impact. 
# 
# And now is time to try some clusters, from the two graphs above (forgeting about the age. We are going to compare:
# 
# * k means
# * Hierarchical Cluster (Ward)
# * Hierarchical Cluster (Complete)
# * DBS
# 
# All the parameters will be available for practicing. Remember that:
# 
# * **K means** : Number of clusters
# * **Hierarchical Clusters**: Number of clusters
# * **DBS** : Epsilon and distance.

# In[ ]:


from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

numberAge = 8
numberIncome = 9
epsilonAge = 5
epsilonIncome = 8
min_samples = 3

fig, axs1 = plt.subplots(4, 2, figsize=(20,20))
colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'pink', 'orange', 'black', 'brown', 'purple', 'deepskyblue','salmon','khaki']
linkage = ['ward', 'complete']

Xage = data[['Age','Spending Score (1-100)']].values
kmeansAge = KMeans(n_clusters=numberAge, random_state=0)
ykmeansAge = kmeansAge.fit_predict(Xage)

wardAge = AgglomerativeClustering(n_clusters=numberAge, linkage = linkage[0])
ywardAge = wardAge.fit_predict(Xage)

completeAge = AgglomerativeClustering(n_clusters=numberAge, linkage = linkage[1])
ycompleteAge = completeAge.fit_predict(Xage)

DbsAge = DBSCAN(eps=epsilonAge, min_samples=min_samples)
yDbsAge = DbsAge.fit_predict(Xage)

for i in range (numberAge):
    axs1[0,0].scatter(Xage[ykmeansAge == i, 0], Xage[ykmeansAge == i, 1], s = 100, c = colors[i], label = 'Cluster' + str(i))
    axs1[1,0].scatter(Xage[ywardAge == i, 0], Xage[ywardAge == i, 1], s = 100, c = colors[i], label = 'Cluster' + str(i))
    axs1[2,0].scatter(Xage[ycompleteAge == i, 0], Xage[ycompleteAge == i, 1], s = 100, c = colors[i], label = 'Cluster' + str(i))

for i in range(yDbsAge.max()):
    axs1[3,0].scatter(Xage[yDbsAge == i, 0], Xage[yDbsAge == i, 1], s = 100, c = colors[i], label = 'Cluster' + str(i))
axs1[3,0].scatter(Xage[yDbsAge == -1, 0], Xage[yDbsAge == -1, 1], s = 100, c = 'grey', label = 'No Cluster')

    
axs1[0,0].scatter(kmeansAge.cluster_centers_[:, 0], kmeansAge.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
axs1[0,0].title.set_text('Cluster kMeans with Age')
axs1[1,0].title.set_text('Ward Hierarchical with Age')
axs1[2,0].title.set_text('Complete Hierarchical with Age')
axs1[3,0].title.set_text('DBS with Age')


Xincome = data[['Annual Income (k$)','Spending Score (1-100)']].values
kmeansIncome = KMeans(n_clusters=numberIncome, random_state=0)
ykmeansIncome = kmeansIncome.fit_predict(Xincome)

wardIncome = AgglomerativeClustering(n_clusters=numberIncome, linkage = linkage[0])
ywardIncome = wardIncome.fit_predict(Xincome)

completeIncome = AgglomerativeClustering(n_clusters=numberIncome, linkage = linkage[1])
ycompleteIncome = completeIncome.fit_predict(Xincome)

DbsIncome = DBSCAN(eps=epsilonIncome, min_samples=min_samples)
yDbsIncome = DbsIncome.fit_predict(Xincome)

for i in range (numberIncome):
    axs1[0,1].scatter(Xincome[ykmeansIncome == i, 0], Xincome[ykmeansIncome == i, 1], s = 100, c = colors[i], label = 'Cluster' + str(i))
    axs1[1,1].scatter(Xincome[ywardIncome == i, 0], Xage[ywardIncome == i, 1], s = 100, c = colors[i], label = 'Cluster' + str(i))
    axs1[2,1].scatter(Xincome[ycompleteIncome == i, 0], Xage[ycompleteIncome == i, 1], s = 100, c = colors[i], label = 'Cluster' + str(i))

for i in range(yDbsIncome.max()):
    axs1[3,1].scatter(Xincome[yDbsIncome == i, 0], Xincome[yDbsIncome == i, 1], s = 100, c = colors[i], label = 'Cluster' + str(i))
axs1[3,1].scatter(Xincome[yDbsIncome == -1, 0], Xage[yDbsIncome == -1, 1], s = 100, c = 'grey', label = 'No Cluster')
    
    
    
axs1[0,1].scatter(kmeansIncome.cluster_centers_[:, 0], kmeansIncome.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
axs1[0,1].title.set_text('Cluster kMeans with Annual income')
axs1[1,1].title.set_text('Ward Hierarchical with Annual income')
axs1[2,1].title.set_text('Complete Hierarchical with Annual income')
axs1[3,1].title.set_text('DBS with Annual Income')
 

for i in range(4):
    for j in range(2):
        axs1[i,j].legend()

    




# In[ ]:




