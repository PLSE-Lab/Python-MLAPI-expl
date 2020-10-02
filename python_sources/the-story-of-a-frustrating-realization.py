#!/usr/bin/env python
# coding: utf-8

# ### Hello! My name is Yvan David Hernandez and I'm a student (undergraduate) participating in this challenge in the context of a graduate class. As I'm in finals, the lack of sleep, the stress and the obvious fact that this is at a level where I doubt my input will count a lot, you can assume I'll go for the composition component. 
# 
# #### I hope you enjoy the story

# In[ ]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score


# It all started a rainy night in Bogota (Colombia). The cold, wet wind blows on the top floor of one downtown building. The big night moths go around one of the last windows with light. Behind the window inside the room is Code. Code is a group of data scientists who I won't give neither names nor gendres. We shall call them 1, 2, 3 and 4. They started to work on the survey results that Kaggle realized in 2018. 

# In[ ]:


data = pd.read_csv('../input/multipleChoiceResponses.csv')


# In[ ]:


data.head()


# After visualizing the data frame, 3 and 4 expressed their excitment while looking at the head. <br>
# "We will have to change a lot about this data frame if we want to be able to go deep inside" said 2. <br> 
# While everybody was waiting for a dumb comment from 1, 4 started to think what should be taken out to be able to work. <br>
# "Let's start" he said, "by taking away the time of completion, and let's work only on the multiple choice answers. For sure there's a lot in there."

# In[ ]:


data = data.drop(data.index[0])
data = data.drop(data.columns[0], axis=1)
data.head()


# "By looking at the data we have to take in count that there is going to be open text answers" said 1, at everyones surprises. <br>
# "Ok", said 4, "let's take them down with a loop". <br>
# "We could do it through the heading of the columns" said 3, "like this!"

# In[ ]:


titulos =data.columns.values


# In[ ]:


for i in range(len(titulos)):
    string = titulos[i]
    l = string[-1]
    if (l!='1' and l!='2' and l!='3' and l!='4' and l!='5' and l!='6' and l!='7' and l!='8' and l!='9' and l!='0'):
        data = data.drop(string, axis=1)
        continue
    #elif(data.loc[:,string].isna().any() == True):
    #    print(string)
    #    data = data.drop(string,axis=1)
    else:
        dummies = pd.get_dummies(data[string], prefix=string)
        data = pd.concat([data, dummies], axis=1)
        data = data.drop(string, axis=1)


# "Well done!!" they all cheered
# ![Cheersurl](https://media.giphy.com/media/BQAk13taTaKYw/giphy.gif)
# "This for sure worked" said 4. "Let's check it out!"

# In[ ]:


data.head()


# "Nice work 3!" said 4, "Now we can work it by doing a PCA" <br>
# "But we should try to clusted it no?" said 1 <br>
# "If you want, go for it! What do you think 2?" answered 4.<br>
# "I'll work with 1 into the clustering" responded 2.

# ### a) PCA

# 3 and 4 started working but didn't do much until 3 mentioned Wiltmore's book called 'Introduction to Scientific and Techincal Computing' (2017) and more specifically it's tenth chapter. After reading it they started to doing it right.

# In[ ]:


features = np.shape(data)[1]
#Matriz de covarianza
cov_matrix = np.array(data.cov())
#Autovalores y autovectores
val, vec = np.linalg.eig(cov_matrix)


# In[ ]:


#Ordena vec y val de mayor a menor
vec = vec[:,val.argsort()[::-1]]
val = val[val.argsort()[::-1]]


# In[ ]:


#Vectores, nuevo sistema coordenado
vec1 = vec[:,0]
vec2 = -vec[:,1]


# In[ ]:


#Varianza
var = np.sum(data.var())


# In[ ]:


#Varianza por componente
def var_comp(vec,data,comp):
    return np.sum(data.dot(vec[:,comp])**2)/len(data)


# In[ ]:


var_comps = []
for i in range(features):
    var_comps.append(var_comp(vec,data,i)/var)


# In[ ]:


fig = plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
x = np.arange(features)
plt.plot(x,var_comps)
plt.xlabel('Principal Component')
plt.ylabel('Prop. Variance Explained')
plt.scatter(x,var_comps,facecolors="none",edgecolors='blue')
plt.title('Principal Component Analysis')

plt.subplot(1,3,2)
x = np.arange(features)
plt.plot(x,var_comps)
plt.xlim(0,5)
plt.xlabel('Principal Component')
plt.ylabel('Prop. Variance Explained')
plt.scatter(x,var_comps,facecolors="none",edgecolors='blue')
plt.title('Zoom on the PCA')

plt.subplot(1,3,3)
cum_var = np.cumsum(np.array(var_comps))
plt.plot(x,cum_var)
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Prop. Variance Explained')
plt.scatter(x,cum_var,facecolors="none",edgecolors='blue')
plt.title('Cumulative Variance')


# After working so hard, 3 and 4 were disapointed at their results.![DisapointingUrl](https://media.giphy.com/media/14iZJ2agTAAb6w/giphy.gif) They didn't got a clear view of the components that the data was made of, and couldn't get any conclusion, except that the optimal K would be probably 2. They did a cluster of it.

# In[ ]:


cluster3 = KMeans(n_clusters=2, random_state=10)
cluster_labels3 = cluster3.fit_predict(data)
data['clusters'] = cluster_labels3


# In[ ]:


aplot=data[data.clusters==0].sum()[0:-2]
result = aplot.sort_values(ascending=False)
ma1= result[0:10]
print('We can construct the first cluster that assembels the folowing features')
print(ma1)
aplot1=data[data.clusters==1].sum()[0:-2]
result = aplot1.sort_values(ascending=False)
ma2= result[0:10]
print('We can construct visualize the 2nd cluster that assembels the folowing features')
print(ma2)


# Even if it wasn't much, 3 and 4 could say with confidence that from the people that answered the survey, (a fair trait of the computer scientists of the world) are mostly males that have a high education (mostly master's degree) and that they like and use Python with scikit-learn and Matplotlib. Nevertheless, when they turned to tell 1 and 2 about it, they found them play rock paper scissors lizard spock. In that very second, the classic joke came to 4's mind but didn't want to help the joke mood. ![xkcd](https://i.redd.it/5cjdqxcg07k11.png)

# "Stop it" said 4 after the 3rd round of RPSLS. <br>
# "But why?" <br>
# "Because you aren't doing a thing! You're playing while we are working. Not cool" <br>
# "But we are working! It's running, but taking sooooo long!" said 2 "Check it out!" <br>

# ## b) Clustering
# 
# For the realization of the clusters, 1 and 2 also read the chapter 10 in the same book. This with the objective of finding the best k to cluster on.

# In[ ]:


#The following code takes a very huge while to run. I higly descourage to run it! To make it more clear even, I'll comment it!

#range_n_clusters = np.arange(80,100)
#score = []
#for n_clusters in range_n_clusters:
#    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
#    cluster_labels = clusterer.fit_predict(data)
#    silhouette_avg = silhouette_score(data, cluster_labels)
#    score.append(silhouette_avg)
#    print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)
#    
#plt.plot(range_n_clusters,score)
#plt.xlabel('K')
#plt.ylabel('silhouette score')

#The output, or partial output (didn't had the time to finish it) would be 

#('For n_clusters =', 80, 'The average silhouette_score is :', -0.057214161693202946)
#('For n_clusters =', 81, 'The average silhouette_score is :', -0.055780258021949893)
#('For n_clusters =', 82, 'The average silhouette_score is :', -0.057544768067351072)
#('For n_clusters =', 83, 'The average silhouette_score is :', -0.058861850121202298)
#('For n_clusters =', 84, 'The average silhouette_score is :', -0.05930968617258206)
#('For n_clusters =', 85, 'The average silhouette_score is :', -0.057119735705425945)
#('For n_clusters =', 86, 'The average silhouette_score is :', -0.057890150542789474)
#('For n_clusters =', 87, 'The average silhouette_score is :', -0.059931949263046097)
#('For n_clusters =', 88, 'The average silhouette_score is :', -0.057302577437669824)
#('For n_clusters =', 89, 'The average silhouette_score is :', -0.055365256535527901)
#('For n_clusters =', 90, 'The average silhouette_score is :', -0.055882421109439827)
#('For n_clusters =', 91, 'The average silhouette_score is :', -0.057461071270653689)
#('For n_clusters =', 92, 'The average silhouette_score is :', -0.056341328445298294)
#('For n_clusters =', 93, 'The average silhouette_score is :', -0.06012728198878943)
#('For n_clusters =', 94, 'The average silhouette_score is :', -0.059186392435306767)
#('For n_clusters =', 95, 'The average silhouette_score is :', -0.057227079284224901)
#('For n_clusters =', 96, 'The average silhouette_score is :', -0.058374450053829408)
#('For n_clusters =', 97, 'The average silhouette_score is :', -0.05236556541252551)
#('For n_clusters =', 98, 'The average silhouette_score is :', -0.056208036133553449)


# "Check it out!" said 1, "This is the average silhouette score for diferent numbers of clusters. The one with the highest will be the optimal number of clusters to work our data set. 
# 

# Oh no! 1 and 3 couldn't finish running the full silhouette scores because the power went down. Nonetheless they realize that even if they are all negatives, the $k=97$ clusters has the highest score. So they tried it!!

# In[ ]:


cluster2 = KMeans(n_clusters=97, random_state=10)
cluster_labels2 = cluster2.fit_predict(data)
data['clusters'] = cluster_labels2


# In[ ]:


aplot=data[data.clusters==0].sum()[0:-2]
result = aplot.sort_values(ascending=False)
ma1= result[0:10]
print('We can construct the first cluster that assembels the folowing features')
print(ma1)

aplot1=data[data.clusters==1].sum()[0:-2]
result = aplot1.sort_values(ascending=False)
ma2= result[0:10]
print('We can construct the cluster 2 that assembels the folowing features')
print(ma2)

aplot4=data[data.clusters==4].sum()[0:-2]
result4 = aplot4.sort_values(ascending=False)
ma4= result4[0:10]
print('We can construct the cluster 5 that assembels the folowing features')
print(ma4)

aplot5=data[data.clusters==5].sum()[0:-2]
result5 = aplot5.sort_values(ascending=False)
ma5= result5[0:10]
print('We can construct the cluster 6 that assembels the folowing features')
print(ma5)

aplot25=data[data.clusters==35].sum()[0:-2]
result25 = aplot25.sort_values(ascending=False)
ma25= result25[0:10]
print('We can construct the cluster 36 that assembels the folowing features')
print(ma25)

aplot65=data[data.clusters==65].sum()[0:-2]
result65 = aplot65.sort_values(ascending=False)
ma65= result65[0:10]
print('We can construct the cluster 66 that assembels the folowing features')
print(ma65)

aplot85=data[data.clusters==85].sum()[0:-2]
result85 = aplot85.sort_values(ascending=False)
ma85= result85[0:10]
print('We can construct the cluster 86 that assembels the folowing features')
print(ma85)


# "From all of this info we can actually draw some conclusions on the comunity" said 3

# ## Conclusions

# Code drew the folowing conclusions from the diferent procedures they did upon the data. <br>
# The clustering in 2, made clear that there were two groups in the survey. The academics with bachelors, masters degree's in computer science that have been in data science for a small time (Question 8). And another larger group where these features are not dominant. Also, the male gender is very percistent and it's fair to asume a precense of female gender in both groups as there isn't a aglomeration of them. <br> The second group agrees in the priorities of education in ML (Question 41) as in the academia there isn't a clear component on the subject (they don't agree in those priorities).
# This only shows what everybody already knows, there are data scientists in and out of academia, they are most of them big time fans of Python. Nonetheless, it's outstanding and ridiculous the poor proportion of females in both of these domains. 

# From the small clusters, there can be numerous conclusions. Here we will state a few of what can be deduced of them. The small clusters aglomerates people that represent the link between R and data visualization to do buisness decisions, but they are a small portion of the survey. We acknowledge the big presence of India, China and USA as they form diferent clusters. All of these clusters reflect the lack of female presence. 

# The final conclusion of this work is that even if the evolution and the dinamic of data science is clear, is terribly frustrating to see the poor porcentage of women and how even the comunity doesn't acknowledge (there is not one question about it). 

# In[ ]:




