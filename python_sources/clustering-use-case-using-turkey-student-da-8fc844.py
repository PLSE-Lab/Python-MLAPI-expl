#!/usr/bin/env python
# coding: utf-8

# # Clustering Use case

# # In the first part, the kernel tries to define the column names and feature.
# # There are 5 columns about instrutor and class details. instructor and class features are categorical other ones are numeric features.
# # 
# # The next 28 columns consist of evaluation of students for each course and its instructor. The values varies from 1 to 5. Higher score means higher approval or satisfaction rate. The evaluation values are numeric since they have a level of order according to each other.
Attribute Information:

instr: Instructor's identifier; values taken from {1,2,3}
class: Course code (descriptor); values taken from {1-13} 
repeat: Number of times the student is taking this course; values taken from {0,1,2,3,...} 
attendance: Code of the level of attendance; values from {0, 1, 2, 3, 4} 
difficulty: Level of difficulty of the course as perceived by the student; values taken from {1,2,3,4,5} 
Q1: The semester course content, teaching method and evaluation system were provided at the start. 
Q2: The course aims and objectives were clearly stated at the beginning of the period. 
Q3: The course was worth the amount of credit assigned to it. 
Q4: The course was taught according to the syllabus announced on the first day of class. 
Q5:	The class discussions, homework assignments, applications and studies were satisfactory. 
Q6: The textbook and other courses resources were sufficient and up to date.	
Q7: The course allowed field work, applications, laboratory, discussion and other studies. 
Q8: The quizzes, assignments, projects and exams contributed to helping the learning.	
Q9: I greatly enjoyed the class and was eager to actively participate during the lectures. 
Q10: My initial expectations about the course were met at the end of the period or year. 
Q11: The course was relevant and beneficial to my professional development. 
Q12: The course helped me look at life and the world with a new perspective. 
Q13: The Instructor's knowledge was relevant and up to date. 
Q14: The Instructor came prepared for classes. 
Q15: The Instructor taught in accordance with the announced lesson plan. 
Q16: The Instructor was committed to the course and was understandable. 
Q17: The Instructor arrived on time for classes. 
Q18: The Instructor has a smooth and easy to follow delivery/speech. 
Q19: The Instructor made effective use of class hours. 
Q20: The Instructor explained the course and was eager to be helpful to students. 
Q21: The Instructor demonstrated a positive approach to students. 
Q22: The Instructor was open and respectful of the views of students about the course. 
Q23: The Instructor encouraged participation in the course. 
Q24: The Instructor gave relevant homework assignments/projects, and helped/guided students. 
Q25: The Instructor responded to questions about the course inside and outside of the course. 
Q26: The Instructor's evaluation system (midterm and final questions, projects, assignments, etc.) effectively measured the course objectives. 
Q27: The Instructor provided solutions to exams and discussed them with students. 
Q28: The Instructor treated all students in a right and objective manner.

Q1-Q28 are all Likert-type, meaning that the values are taken from {1,2,3,4,5}

# # #The kernel imports many libraries. 
# # Pandas for data manupulation, numpy for numeric calculations, pyplot and seaborn for graphs 

# In[ ]:


#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# # In this part the kernel reads the data with pandas library's function of read_csv

# In[ ]:


dataset = pd.read_csv('../input/turkiye-student-evaluation_generic.csv')


# # The kernel wants to show the beginning part of dataset. Head function is used for it.

# In[ ]:


dataset.head()


# # To understand the statistical values of each column, it uses describe function. 

# In[ ]:


dataset.describe()


# # The kernel wants to show how many of the students had made evaluation for each class, it uses below function. It shows that a class is evaluated more than others.

# In[ ]:


plt.figure(figsize=(20, 6))
sns.countplot(x='class', data=dataset)


# # we can see the distribution of evaluations for each question. Some questions have higher evaluation rate according to others. 

# In[ ]:


plt.figure(figsize=(20, 20))
sns.boxplot(data=dataset.iloc[:,5:31 ]);


# 

# # In this part kernel wants to understand students response for classes. It tries to calculate mean value for each question. Firstly it defines some related variables which are null. Then it creates a new data frame for these variable.
# 
# # By using for it creates a loop. It takes 13 courses and adds the total scores for each course. They are the first 12 questions. So there is a sum number for each course. The sum is total scores of students for the class.
# 
# # The number of class are not the same. So its meaningful to use mean value.

# In[ ]:


# Calculate mean for each question response for all the classes.
questionmeans = []
classlist = []
questions = []
totalplotdata = pd.DataFrame(list(zip(classlist,questions,questionmeans))
                      ,columns=['class','questions', 'mean'])
for class_num in range(1,13):
    class_data = dataset[(dataset["class"]==class_num)]
    
    questionmeans = []
    classlist = []
    questions = []
    
    for num in range(1,13):
        questions.append(num)
    #Class related questions are from Q1 to Q12
    for col in range(5,17):
        questionmeans.append(class_data.iloc[:,col].mean())
    classlist += 12 * [class_num] 
    print(classlist)
    plotdata = pd.DataFrame(list(zip(classlist,questions,questionmeans))
                      ,columns=['class','questions', 'mean'])
    totalplotdata = totalplotdata.append(plotdata, ignore_index=True)
    


# # After having the total number of classes, kernel wants to take mean of each. 

# In[ ]:


plt.figure(figsize=(20, 10))
sns.pointplot(x="questions", y="mean", data=totalplotdata, hue="class")


# # In the graph, we can see that some classes have higher mean value scores than others. There is 

# # A similar approach can be used to see the satisfaction or approval of instructors according to student evaluation. The questions from 13 to 28 are instructor related questions.
# 
# # There are totaly 3 different instructors. The kernel use for loop to calculate the sum of total scores for each instructor. Then it calculated the mean value. Because the number of evaluations are not equal for each instructor. So its better to use mean value.
# 

# In[ ]:


# Calculate mean for each question response for all the classes.
questionmeans = []
inslist = []
questions = []
totalplotdata = pd.DataFrame(list(zip(inslist,questions,questionmeans))
                      ,columns=['ins','questions', 'mean'])
for ins_num in range(1,4):
    ins_data = dataset[(dataset["instr"]==ins_num)]
    questionmeans = []
    inslist = []
    questions = []
    
    for num in range(13,29):
        questions.append(num)
    
    for col in range(17,33):
        questionmeans.append(ins_data.iloc[:,col].mean())
    inslist += 16 * [ins_num] 
    plotdata = pd.DataFrame(list(zip(inslist,questions,questionmeans))
                      ,columns=['ins','questions', 'mean'])
    totalplotdata = totalplotdata.append(plotdata, ignore_index=True)


# # According to instructors we can see that 3rd instructor has less approval rate.

# In[ ]:


plt.figure(figsize=(20, 5))
sns.pointplot(x="questions", y="mean", data=totalplotdata, hue="ins")


# # To go deeper, the kernel wants to understand which class of instructor 3 have lowest approval rate. It defines new null variables to calculate the total scores for each class that Instructor 3 have.

# In[ ]:


# Calculate mean for each question response for all the classes for Instructor 3
dataset_inst3 = dataset[(dataset["instr"]==3)]
class_array_for_inst3 = dataset_inst3["class"].unique().tolist()
questionmeans = []
classlist = []
questions = []
totalplotdata = pd.DataFrame(list(zip(classlist,questions,questionmeans))
                      ,columns=['class','questions', 'mean'])
for class_num in class_array_for_inst3:
    class_data = dataset_inst3[(dataset_inst3["class"]==class_num)]
    
    questionmeans = []
    classlist = []
    questions = []
    
    for num in range(1,13):
        questions.append(num)
    
    for col in range(5,17):
        questionmeans.append(class_data.iloc[:,col].mean())
    classlist += 12 * [class_num]
    
    plotdata = pd.DataFrame(list(zip(classlist,questions,questionmeans))
                      ,columns=['class','questions', 'mean'])
    totalplotdata = totalplotdata.append(plotdata, ignore_index=True)


# # The kernel use matplotlib to show the graph. Its better to use mean value again to have a fair comparison.

# In[ ]:


plt.figure(figsize=(20, 8))
sns.pointplot(x="questions", y="mean", data=totalplotdata, hue="class")


# #  Cluster the students based on the questionaire data

# # With iloc function, the kernel creates a new dataframe for question evaluatins only. Its from 6th column to 33rd column.

# In[ ]:


dataset_questions = dataset.iloc[:,5:33]


# # The head function show the first 5 rows of new dataframe

# In[ ]:


dataset_questions.head()


# # PCA is used here to decrease number of features. Because there are 28 evaluation features. Its too many to use all. So the kernel wants to decrease the number of features for a better approach.

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
dataset_questions_pca = pca.fit_transform(dataset_questions)


# # From scikit learn library, the kernel use kmeans method. It can be one of the best solutions to understand the students according to the evaluation metrics. 
# # It uses 7 cluster and try to find "within cluster sum of squares". The optimum number is tried to avhieved.
# # By appliying plot function the elbow method is shown.

# In[ ]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 7):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(dataset_questions_pca)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 7), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# # 3 or 4 clusters can be best. The kernel accepts 3 clusters as best option.
# # Fit_predict function is used to divide the data into 3 clusters.

# In[ ]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++')
y_kmeans = kmeans.fit_predict(dataset_questions_pca)


# # In this part, kernel wants to show the best graph for visualization. It uses scatter plot to show each cluster. 
# # For centroids the values are tried to be located in best location in graph by scatter plot.  On x axis, we see PCA1 values and on y axis the kernel show PCA2 values. 

# In[ ]:


# Visualising the clusters
plt.scatter(dataset_questions_pca[y_kmeans == 0, 0], dataset_questions_pca[y_kmeans == 0, 1], s = 100, c = 'yellow', label = 'Cluster 1')
plt.scatter(dataset_questions_pca[y_kmeans == 1, 0], dataset_questions_pca[y_kmeans == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(dataset_questions_pca[y_kmeans == 2, 0], dataset_questions_pca[y_kmeans == 2, 1], s = 100, c = 'red', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'blue', label = 'Centroids')
plt.title('Clusters of students')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()


# # The visual form of values are shown as in the graph.

# # Checking number of students in each cluster by counter function

# In[ ]:


import collections
collections.Counter(y_kmeans)


# # The second method for clustering can be a dendogram. The kernel use scipy for clustering.
# # Ward method is used to cluster the students.

# In[ ]:


# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(dataset_questions_pca, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('questions')
plt.ylabel('Euclidean distances')
plt.show()


# # Hierarchical Clustering is used to fit the dataset. From sklearn, agglomerative clustering is used. 2 clusters can be applied in this method.

# In[ ]:


# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(dataset_questions_pca)
X = dataset_questions_pca
# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'yellow', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'red', label = 'Cluster 2')
plt.title('Clusters of STUDENTS')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()


# # The kernel checks the number of students for each cluster. 

# In[ ]:


import collections
collections.Counter(y_hc)


# # The negative  cluster (on the left in graph) are similar to kmeans. But 2 and 3rd clusters are accpeted as one cluster in hierarchical clustering.

# If we compare the clusters of Kmeans and Hierarchical process, we can see cluster with red ( Negative is matching approximately)

# Please upvote if you like the kernel

# In[ ]:




