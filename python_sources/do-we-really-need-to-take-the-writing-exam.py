#!/usr/bin/env python
# coding: utf-8

# ![exams.png](attachment:exams.png)
# 
# This notebook takes a look into the factors that affect the students performance in the exam and also answer a few problem statements like,
# *Is it possible for a student to take just the read and the math exams and not the write exam separately?*
# 
# The answer to this is determined by using Classification, Regression and Clustering!
# 
# By creating buckets for grades and assigning grades to each student based on the average of the score a **Classification Model ** can be built.
# 
# A **Regression Model** can be used to predict the writing score just be checking the math and the reading code.
# 
# This theory of trying to relate the math and read score to the writing score is supported by **Clustering** and checking for the clusters that show behaviour of points when read and write score are taken as features.

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")


# In[ ]:


#number of rows
len(df)


# In[ ]:


#data types
df.dtypes


# In[ ]:


#unique values
df['gender'].unique()


# In[ ]:


df['race/ethnicity'].unique()


# In[ ]:


df['parental level of education'].unique()


# In[ ]:


df['lunch'].unique()


# In[ ]:


df['test preparation course'].unique()


# In[ ]:


#check for null values
df.isnull().sum()


# In[ ]:


#renaming columns for easier use
df = df.rename(columns = {"race/ethnicity":"Group", "parental level of education":"Parent_Education", 
                     "test preparation course":"Preparation", "math score":"Math_Score",
                     "reading score":"Reading_Score", "writing score":"Writing_Score"})


# In[ ]:


#check the  columns
df.head()


# In[ ]:


#create a new column for average score
average_score = []
for i in range (0,1000):
    average_score.append((df['Math_Score'][i] + df['Reading_Score'][i] + df['Writing_Score'][i])/3)

#adding column to dataset
df['Average_Score'] = average_score


# In[ ]:


#creating function to assign grades
def grade(x):
    if (x >= 90 and x <=100): gr = 'A'
    elif (x >= 80 and x<90): gr = 'B'
    elif (x>=70 and x<80): gr = 'C'
    elif (x>=60 and x<70): gr = 'D'
    elif (x>=50 and x<60): gr = 'E'
    else:  gr = 'Fail'
    return gr

#to create a new list containing all the grades
grades = []
for i in range (0,1000):
    gr = grade(df['Average_Score'][i])
    grades.append(gr)
    
    
#adding to dataframe
df['Grades'] = grades


# In[ ]:


#check the columns
df.head()


# In[ ]:


#Gender Distribution
sns.countplot(df['gender'], palette = 'dark')


# In[ ]:


#Group Distribution
sns.countplot(df['Group'], palette = 'dark')


# In[ ]:


#Lunch distribution
sns.countplot(df['lunch'], palette = 'dark')


# In[ ]:


#Preparation count
sns.countplot(df['Preparation'], palette = 'dark')


# In[ ]:


#distribution of grades
plt.figure(figsize = (8,8))
sns.countplot(x = df['Grades'])


# In[ ]:


#Histogram for Math 
sns.distplot(df['Math_Score'], color = 'red')


# In[ ]:


#Histogram for Reading
sns.distplot(df['Reading_Score'], color = 'blue')


# In[ ]:


#Histogram for Writing
sns.distplot(df['Writing_Score'], color = 'green')


# In[ ]:


#histogram for the average score 
sns.distplot(df['Average_Score'], color = 'purple')


# In[ ]:


#Distribution of parent education in each group
plt.figure(figsize = (10,10))
sns.countplot(df['Group'], hue = df['Parent_Education'])


# In[ ]:


#Average score per group wrt gender
plt.figure (figsize = (8,8))
sns.barplot(x = df['Group'], y = df['Average_Score'], hue = df['gender'])


# In[ ]:


#Grades based on preparation
plt.figure(figsize = (7,7))
sns.barplot(x = df['Grades'], y = df['Average_Score'], hue = df['Preparation'])


# # Classification
# 
# Can we determine the grade of the student by knowing just the features?

# In[ ]:


#Label Encode
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
df.gender = le.fit_transform(df.gender)
df.lunch = le.fit_transform(df.lunch)
df.Group = le.fit_transform(df.Group)
df.Preparation = le.fit_transform(df.Preparation)
df.Parent_Education = le.fit_transform(df.Parent_Education)


# In[ ]:


#Regression Plot
plt.figure(figsize = (10,10))
corr = df.corr() 
ax = sns.heatmap( corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True) 
ax.set_xticklabels( ax.get_xticklabels(), rotation=45, horizontalalignment='right' );


# In[ ]:


#Now let us try to classify based on the features apart from math, reading and writing score
X = df.iloc[:,0:8].values
y = df.iloc[:,-1].values
y = le.fit_transform(y)


# In[ ]:


#scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)    


# In[ ]:


#split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ***Logistic Regression***

# In[ ]:


#logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# In[ ]:


#predict value 
y_pred = classifier.predict(X_test)


# In[ ]:


#classification report 
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# ***Random Forest Classification***

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[ ]:


#predict value 
y_pred = classifier.predict(X_test)


# In[ ]:


#Accuracy score
accuracy_score(y_test, y_pred)


# In this classification, it is easier to classify the student into grades by using simple Logistic Regression.

# # **Regression**

# *Does the student need a writing test? Can it be predicted by using just the Math and Reading score?*

# ***Multivariate Linear Regression***

# In[ ]:


#Multivariate linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[ ]:


#predict the value
y_pred = regressor.predict(X_test)


# In[ ]:


#accuracy
from sklearn.metrics import r2_score, mean_squared_error
r2_score(y_test, y_pred)


# In[ ]:


mean_squared_error(y_test, y_pred)


# ***Random Forest Regression***

# In[ ]:


#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)


# In[ ]:


#predict the value
y_pred = regressor.predict(X_test)


# In[ ]:


#accuracy
from sklearn.metrics import r2_score, mean_squared_error
r2_score(y_test, y_pred)


# In[ ]:


mean_squared_error(y_test, y_pred)


# With a high r2_score and very low Mean Squared Error we can use the Random Forest Regression model to predict the writing score by just using the math and read score.

# In[ ]:


#regression plot
sns.regplot(x = y_pred, y = y_test)


# # **Clustering**

# Can a pattern be noticed by clustering the reading and the writing scores?

# ***KMeans Clustering***

# In[ ]:


#Using the K Means method
X = df.iloc[:,[6,7]].values


# In[ ]:


#Elbow plot method to determine the number of clusters
from sklearn.cluster import KMeans
wcss = []

for i in range (1,11):
    clusterer = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10)
    clusterer.fit(X)
    wcss.append(clusterer.inertia_)


# In[ ]:


plt.plot(range(1,11), wcss)
plt.title("Elbow Method")


# Choosing 5 as the best number of clusters.

# In[ ]:


kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10)
kmeans.fit(X)


# In[ ]:


ykmeans = kmeans.fit_predict(X)


# In[ ]:


#visualize the clusters
plt.scatter(X[ykmeans == 0,0], X[ykmeans == 0,1], s = 100, c = 'red', label = 'C1')
plt.scatter(X[ykmeans == 1,0], X[ykmeans == 1,1], s = 100, c = 'blue', label = 'C2')
plt.scatter(X[ykmeans == 2,0], X[ykmeans == 2,1], s = 100, c = 'green', label = 'C3')
plt.scatter(X[ykmeans == 3,0], X[ykmeans == 3,1], s = 100, c = 'maroon', label = 'C4')
plt.scatter(X[ykmeans == 4,0], X[ykmeans == 4,1], s = 100, c = 'black', label = 'C5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'centroids')
plt.title("Reading vs Writing")
plt.xlabel("Reading")
plt.ylabel("Writing")
plt.legend()
plt.show()


# As can be seen from the plot, students with higher reading score have a higher writing score. This along with the regression data can be thus used to assign grades to the student based on the features by using Classification.

# Please do upvote if you like my approach, support is highly appreciated.
# Thank You!
