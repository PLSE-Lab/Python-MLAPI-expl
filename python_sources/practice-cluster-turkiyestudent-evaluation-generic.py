#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Attribute Information: instr: Instructor's identifier; values taken from {1,2,3} class: Course code (descriptor); values taken from {1-13} repeat: Number of times the student is taking this course; values taken from {0,1,2,3,...} attendance: Code of the level of attendance; values from {0, 1, 2, 3, 4} difficulty: Level of difficulty of the course as perceived by the student; values taken from {1,2,3,4,5} Q1: The semester course content, teaching method and evaluation system were provided at the start. Q2: The course aims and objectives were clearly stated at the beginning of the period. Q3: The course was worth the amount of credit assigned to it. Q4: The course was taught according to the syllabus announced on the first day of class. Q5: The class discussions, homework assignments, applications and studies were satisfactory. Q6: The textbook and other courses resources were sufficient and up to date. Q7: The course allowed field work, applications, laboratory, discussion and other studies. Q8: The quizzes, assignments, projects and exams contributed to helping the learning. Q9: I greatly enjoyed the class and was eager to actively participate during the lectures. Q10: My initial expectations about the course were met at the end of the period or year. Q11: The course was relevant and beneficial to my professional development. Q12: The course helped me look at life and the world with a new perspective. Q13: The Instructor's knowledge was relevant and up to date. Q14: The Instructor came prepared for classes. Q15: The Instructor taught in accordance with the announced lesson plan. Q16: The Instructor was committed to the course and was understandable. Q17: The Instructor arrived on time for classes. Q18: The Instructor has a smooth and easy to follow delivery/speech. Q19: The Instructor made effective use of class hours. Q20: The Instructor explained the course and was eager to be helpful to students. Q21: The Instructor demonstrated a positive approach to students. Q22: The Instructor was open and respectful of the views of students about the course. Q23: The Instructor encouraged participation in the course. Q24: The Instructor gave relevant homework assignments/projects, and helped/guided students. Q25: The Instructor responded to questions about the course inside and outside of the course. Q26: The Instructor's evaluation system (midterm and final questions, projects, assignments, etc.) effectively measured the course objectives. Q27: The Instructor provided solutions to exams and discussed them with students. Q28: The Instructor treated all students in a right and objective manner. Q1-Q28 are all Likert-type, meaning that the values are taken from {1,2,3,4,5}
# 

# In[ ]:


import pandas as pd


# In[ ]:


dataset = pd.read_csv('/kaggle/input/uci-turkiye-student-evaluation-data-set/turkiye-student-evaluation_generic.csv')


# In[ ]:


dataset.head()


# In[ ]:


dataset.shape


# In[ ]:


dataset.info()


# In[ ]:


dataset.isnull().any()


# In[ ]:


dataset.describe()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.countplot(dataset['class'])
#distribution of class more of 3 and 13 class


# In[ ]:


plt.figure(figsize = (15,10))
sns.countplot(dataset['class'],hue = dataset['nb.repeat'])
#The value of repeat for the majority is 1


# In[ ]:


plt.figure(figsize = (15,10))
sns.countplot(x='class', hue='difficulty', data=dataset)


# In[ ]:


sns.countplot(x='difficulty', hue='nb.repeat', data=dataset)


# In[ ]:


plt.figure(figsize=(20, 20))
sns.boxplot(data=dataset.iloc[:,5:31 ])
#Most of the questions related to the instructor has higher range with less spread


# In[ ]:


data=dataset.iloc[:,5:]
data


# In[ ]:


#Before performing PCA it is important to perform standard scaling did not do here as all the data was in same scale
from sklearn.decomposition import PCA 

pca = PCA(n_components = None) 
data = pca.fit_transform(data) 
explained_variance = pca.explained_variance_ratio_


# In[ ]:


explained_variance


# In[ ]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 10), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#The number of cluster by elbow method seems to be 2


# In[ ]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++')
y_kmeans = kmeans.fit_predict(data)

# Visualising the clusters
plt.scatter(data[y_kmeans == 0, 0], data[y_kmeans == 0, 1], s = 100, c = 'yellow', label = 'Cluster 1')
plt.scatter(data[y_kmeans == 1, 0], data[y_kmeans == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(data[y_kmeans == 2, 0], data[y_kmeans == 2, 1], s = 100, c = 'red', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'blue', label = 'Centroids')
plt.title('Clusters of students')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()


# In[ ]:


df_cluster = pd.DataFrame(y_kmeans,columns = ['Cluster'])
dataset = dataset.join(df_cluster)


# In[ ]:


dataset.Cluster.value_counts().plot(kind = 'bar')


# In[ ]:


import collections
collections.Counter(y_kmeans)


# In[ ]:


#Now ploting the dendogram for hirearchial clustering
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(data, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('questions')
plt.ylabel('Euclidean distances')
plt.show()


# In[ ]:


#From the dendogram we can see the appropriate cluster is 2 as the largest vertical distance in the dendogram
#passes through the lines is 2
# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(data)
X = data
# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'yellow', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'red', label = 'Cluster 2')
plt.title('Clusters of STUDENTS')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()


# In[ ]:


dataset = dataset.join(pd.DataFrame(y_hc,columns = ['CLuster2']))

