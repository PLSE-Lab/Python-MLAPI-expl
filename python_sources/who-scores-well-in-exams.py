#!/usr/bin/env python
# coding: utf-8

# # __Who scores well in exams?__
# **Kacper Kubara**, **_8th of December 2018_**  
#   
#   **Content:**
#   1. [Introduction](#1)
#   1. [Exploring the data](#2)  
#       1. [General distribiution of exam scores](#3)
#       1. [Exam scores based on the gender](#4)
#       1. [Exam scores based on the previous education](#5)
#       1. [Exam scores based on the lunch type](#6)
#       1. [Exam scores based on the test preparation course](#6)
#   1. [Clustering the data with K-Means](#7)
#       1. [Data Preprocessing](#8)
#       1. [K-Means](#9)
#       1. [Visualisation](#10)
#   1. [Summary & Conclusions](#11)  

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display


#   ## Introduction
#   Using the [Students Performance in Exams Dataset](https://www.kaggle.com/spscientist/students-performance-in-exams) we will try to understand what affects the exam scores. The data is a bit limited, but with a good visualisation it will be fun to spot any relations. After exploring the data, I will apply K-Means algorithm to see if there any clusters in the dataset based on the exam scores. It is my very first Kernel but I will try my best to provide you with some interesting observations from the dataset. For the full code please visit my [Github profile](https://github.com/KacperKubara/Students_Exam_Performance). 
#   
#   ## Exploring the data  
#   Let's import the libraries and display first few rows of the dataset to see how the data looks like!  

# In[ ]:


dataset = pd.read_csv("../input/StudentsPerformance.csv")
dataset.head()


# ### General distribiution of exam scores
# There are 5 features which might affect the scores of each exam. First thing to analyse would be to see how the scores are distributed within each exam (Maths, Reading, Writing). We will plot histograms to see if there any differences in the scores' distribution

# In[ ]:


# Create figures and axes
fig0, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize = [12.8, 9.6])
# Plot the histograms
sns.distplot(dataset["math score"], kde = False, label = "Maths", ax = ax0, color = 'b')
ax0.set_title("Math") 
ax0.set_xlabel("") # Remove xlabel
sns.distplot(dataset["reading score"], kde = False, label = "Reading", ax = ax1, color = 'g')
ax1.set_title("Reading")
ax1.set_xlabel("")
sns.distplot(dataset["writing score"], kde = False, label = "Writing", ax = ax2, color = 'y')
ax2.set_title("Writing")
ax2.set_xlabel("")


# The scores are distribiuted in the Gaussian manner. It is hard to draw any conclusion from the graphs above: they all look very similar and we don't have enough data for the plots to  look more smoothly. Though, the Math graph seems to be slightly more shifted to the left when comparing it to other graphs, but it is almost unnoticable. Let's immerse ourselves further into the dataset to explore more interesting correlations!

# ### Exam scores based on the gender  
# It will be interesting to see how the exam scores differ when we divide the exam results based on gender.

# In[ ]:


# Create figures and axes
fig1, ((ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(3, 2, figsize = [12.8, 9.6])

# Plot the histograms for exam scores distribiution based on gender
dataset_male = dataset[dataset["gender"] == "male"]
dataset_female = dataset[dataset["gender"] == "female"]

# Plot the exam distributions
sns.distplot(dataset_male["math score"], kde = False, label = "Maths", ax = ax3, color = 'b')
ax3.set_xlabel("Math_male")
sns.distplot(dataset_female["math score"], kde = False, label = "Maths", ax = ax4, color = 'b')
ax4.set_xlabel("Math_female")
sns.distplot(dataset_male["reading score"], kde = False, label = "Reading", ax = ax5, color = 'g')
ax5.set_xlabel("Reading_male")
sns.distplot(dataset["reading score"], kde = False, label = "Reading", ax = ax6, color = 'g')
ax6.set_xlabel("Reading_female")
sns.distplot(dataset_male["writing score"], kde = False, label = "Writing", ax = ax7, color = 'y')
ax7.set_xlabel("Writing_male")
sns.distplot(dataset_female["writing score"], kde = False, label = "writing", ax = ax8, color = 'y')
ax8.set_xlabel("Writing_female")


# And let's see the mean of each exam score

# In[ ]:


# Visualise the mean score based on gender
male_mean = dataset_male[["math score", "reading score", "writing score"]].mean()
female_mean = dataset_female[["math score", "reading score", "writing score"]].mean()
mean_scores_by_gender = pd.concat([male_mean, female_mean], axis = 1, names = ["test", "lol"])
mean_scores_by_gender.columns = ["Male Mean", "Female Mean"] 
display(mean_scores_by_gender)


# Something interesting here! It seems that in this particular dataset women are better in reading and writing, whereas men scores better in maths! It is an interesting observation, in fact you can find a lot of information in the Internet exploring this topic. It also seems that women have more similar results to each other (smaller variance), whereas men's results tend to differ more . 

# ### Exam scores based on the previous education  
# Next question is if the education level of parents can affect the exam scores. Personally, I think that it does have a statistical impact, but let's verify that! 

# In[ ]:


# Display the labels for the education
display(dataset["parental level of education"].unique())


# We should change the labels of "some high school" to "high school" and "some college" to "college".

# In[ ]:


dataset["parental level of education"] = dataset["parental level of education"].map(lambda x: "high school" if x == "some high school" else x)
dataset["parental level of education"] = dataset["parental level of education"].map(lambda x: "college" if x == "some college" else x)
education_level_list = dataset["parental level of education"].unique()
display(education_level_list)


# Much better now! It is time to plot some bar graphs to see the scores' distribution.

# In[ ]:


# Initialise the figure and df_mean to store mean values
df_mean = pd.Series()
fig2 , ax = plt.subplots(3, 1, figsize = [12.8, 15], sharex= True)

# Create neat table for mean values
for i, education_level in enumerate(education_level_list):
    mean = dataset[dataset["parental level of education"] == education_level].mean()
    mean = mean.rename(education_level)
    df_mean = pd.concat([df_mean, mean], axis = 1, sort = False)

df_mean = df_mean.drop(df_mean.columns[0], axis = 1)

# Plot the exam score based on parental education
ax[0] = sns.barplot(x = "parental level of education", y = "math score", 
                    data = dataset, estimator = np.mean, ax = ax[0])
ax[1] = sns.barplot(x = "parental level of education", y = "reading score", 
                    data = dataset, estimator = np.mean, ax = ax[1])
ax[2] = sns.barplot(x = "parental level of education", y = "writing score", 
                    data = dataset, estimator = np.mean, ax = ax[2])
for axes in ax:
    axes.set_xlabel("")


# Hmm... It is hard to spot any differences by looking on bar graphs. But we still can display the mean values as a table or a heatmap.

# In[ ]:


# Display the mean table
display(df_mean)

# Display a heatmap with the numeric values in each cell
fig4, ax9 = plt.subplots(figsize=(12.8, 6))
sns.heatmap(df_mean,linewidths=.1, ax=ax9)


# Indeed, it seems that a lower parental level of education has a negative impact on the exam scores. Children of parents whose the highest education level was college or high school have noticeably lower exam scores than their peers. Similarly, parents with master's or bachelor's degree have children who scores much better in the exams.

# ### Exam scores based on the lunch type  
# It might be amusing to think that type of lunch students have is correlated to their exam scores. On the other hand, we can see from the dataset that there are two types of lunch: **standard** and **free/reduced**. So it depends on the parents' financial situation rather than on the type of the dish. There might be some correlation be here, so let's try to visualise the problem.
# 

# In[ ]:


# Results based on the lunch type
dataset_lunch = dataset[["lunch", "math score", "reading score", "writing score"]].copy()
dataset_lunch = dataset_lunch.groupby(by = ["lunch"]).mean()
# Display the table and the heatmap
display(dataset_lunch)
fig5, ax10 = plt.subplots(figsize=(12.8, 6))
sns.heatmap(dataset_lunch,linewidths=.1, ax=ax10)


# Surprisingly enough, there is a huge disproportion between students who have a **free/reduced** lunch when compared to those having **standard** lunch. There might be a sociological reason for that, if you have an idea what is the reason let me know in comments! 

# ### Exam scores based on the test preparation course  
# The last thing I want to explore in this dataset is to determine how the completion of the test preparation course affects the exam scores. There are only two categorical variables: **none** and **completed**. This time, following [Occam's razor principle](https://en.wikipedia.org/wiki/Occam%27s_razor), I will display only the simple table of the means.

# In[ ]:


dataset_preparation = dataset[["test preparation course", "math score", "reading score", "writing score"]].copy()
dataset_preparation = dataset_preparation.groupby(by = ["test preparation course"]).mean()
display(dataset_preparation)


# Not surprisingly, students who didn't undertake the test preparation course scores **5-10%** less than their peers. 

# ## Clustering the data with K-Means
# My idea was to cluster the dataset to see whether there any similarities between certain groups of students. For example, if there is a particular group of students who scores well in maths but poorly in reading and writing. To do that, I had to preprocess the data and use K-Means algorithm for clustering. I thought about using [PCA](http://sebastianraschka.com/Articles/2014_pca_step_by_step.html), but since I will use only exam scores for the clusters I can easily put the data points on the 3-D scatter plot. 

# ### Preprocessing
# There few obstacles to remove before using K-Means algorithm, namely:
# * Numerical scores has to become categorical features
# * Data has to be labelled and one hot encoded
# Since the exam scores are integer values **0-100%**, I had to assign labels for certain value range. For example **0-35%** score becomes **very poor**, **35-55%** becomes **poor**, etc. Once it was done, data was labelled and one hot encoded using modules from sklearn.

# In[ ]:


def score_labels(x):
    if x<35:
        return "very low"
    if x>=35 and x<55:
        return "low"
    if x>=55 and x<65:
        return "average"
    if x>=65 and x<75:
        return "good"
    if x>=75 and x<85:
        return "high"
    if x>=85 and x<=100:
        return "very high"    
# Read the data
"""
Create classes for the exam scores
0-35%    - very low
35-55%   - low
55-65%   - average
65%-75%  - good
75-85%   - high
85%-100% - very high
"""
# Make an average score from 3 exams and label them as above
average_score = dataset.iloc[:,-3:]
x_num =  dataset.iloc[:,-3:]
average_score = average_score.applymap(score_labels)
x = average_score
x_copy = x.copy()

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
hot_enc_x   = OneHotEncoder()
label_enc_x = LabelEncoder()

x = x.apply(label_enc_x.fit_transform)
x = hot_enc_x.fit_transform(x).toarray()
display(x[:,:5])


# ### K-Means
# Once the data has been preprocessed, we can apply the K-Means algorithm. I used the [Elbow Method](https://en.wikipedia.org/wiki/Elbow_method_(clustering)) to find the optimum number of clusters(it is 5). For the clarity, I won't put that part of the code here, but you can refer to my [Github](https://github.com/KacperKubara) page for the full code.

# In[ ]:


# Fitting K-Means to the dataset
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5, init = 'k-means++')
y_kmeans = kmeans.fit_predict(x)
x_num["cluster"] = y_kmeans


# ### Visualisation 
# Now, we can plot the clusters on the 3D plot!

# In[ ]:


# Visualising the clusters
from mpl_toolkits.mplot3d import axes3d
fig6 = plt.figure(figsize = (12.8, 9))
ax11 = fig6.add_subplot(111, projection='3d')
ax11.scatter((x_num[x_num.cluster == 0])["math score"].values, (x_num[x_num.cluster == 0])["reading score"].values, (x_num[x_num.cluster == 0])["writing score"].values, s = 100, c = 'red', label = 'Cluster 1')
ax11.scatter((x_num[x_num.cluster == 1])["math score"].values, (x_num[x_num.cluster == 1])["reading score"].values, (x_num[x_num.cluster == 1])["writing score"].values, s = 100, c = 'blue', label = 'Cluster 2')
ax11.scatter((x_num[x_num.cluster == 2])["math score"].values, (x_num[x_num.cluster == 2])["reading score"].values, (x_num[x_num.cluster == 2])["writing score"].values, s = 100, c = 'green', label = 'Cluster 3')
ax11.scatter((x_num[x_num.cluster == 3])["math score"].values, (x_num[x_num.cluster == 3])["reading score"].values, (x_num[x_num.cluster == 3])["writing score"].values, s = 100, c = 'cyan', label = 'Cluster 4')
ax11.scatter((x_num[x_num.cluster == 4])["math score"].values, (x_num[x_num.cluster == 4])["reading score"].values, (x_num[x_num.cluster == 4])["writing score"].values, s = 100, c = 'magenta', label = 'Cluster 5')
ax11.set_title('Clusters of Students')
ax11.set_xlabel('Math')
ax11.set_ylabel('Reading')
ax11.set_zlabel('Writing')
ax11.legend()


# To be fair, the scores seem to be linearly related to each other, i.e. when someone scores high in maths he is likely to score well in other exams. The are no discrete clusters in the dataset when presented the graph above.

# ## Summary & Conclusions  
#  We started analysis from exploring how the features affect the exam scores in the dataset. We could see some obvious relations such as men tend to score better in maths whereas women tend to do well in reading an writing, or students who didn't do the test preparation course scores **5-10%** worse than students who completed the course. There were also some non-obivious relations (the most interesting ones!), such as the fact that education level of the parents affects the exam scores of their children, or lunch type (effectively related to the financial situtation of family) also have an impact on exam scores.  
#   To score well in exams, the social background of the student does have a statistical impact on their academic performance (at least in this dataset). Though, we can observe that students who put more time in the preparation (by doing test preparation course) have much better scores, and one can exceel in the exams just by putting an effort to prepare well!
#    I hope you enjoyed reading my Kernel. If you do have any questions or you have spotted a typo somewhere, please let me know :)
