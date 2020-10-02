#!/usr/bin/env python
# coding: utf-8

# **HELLO EVERYONE!!!**
# 
# I use a very basic approach for analysing the data of iris dataset using EDA(Exploratory Data Analysis).First i visualize data with the help of **Matplotlib** and **Seaborn** libraries and then apply following ML algorithms for predictions:
# 
# 1. SVM
# 2. KNN
# 3. Decision Tree Regression
# 
# I also refer this notebook https://www.kaggle.com/ash316/ml-from-scratch-with-iris/notebook for better understanding of iris dataset.
# 
# If you find this notebook to be useful then **Please Upvote**.
# 
# 
# 
# 

# In[ ]:


#importing basic libraries:

import pandas as pd    #for data preprocessing
import numpy as np     #for linear algebra
import matplotlib.pyplot as plt   #for data visualisation
import seaborn as sns      #for data visualisation


# In[ ]:


df = pd.read_csv('../input/iris/Iris.csv')        #load the dataset

df.head()      #extracting the head of dataset


# In[ ]:


df.info()           #for checking if there are null values or inconsistent datatype in dataset.
                    #as we can see there are no NA values so there is no need of preprocessing.


# In[ ]:


df.drop('Id',axis = 1,inplace = True)        #Drop the first column as it doesn't have any information.
df.head(3)                                   #extracting the 3 values from dataset.


# In[ ]:


df['Species'].value_counts()         #checking the number of class in target(species column).
                                     #As we can see, there are 50 values of each type of species class.


# # **DATA VISUALISATION:- **

# In[ ]:


sns.FacetGrid(df, hue='Species',height= 5).map(plt.scatter, 'SepalLengthCm', 'SepalWidthCm')
plt.legend()
plt.show()


# The above graph shows distribution of different species on the basis of sepal length and sepal width.It can be seen that setosa can be classify on the basis of these two features but other two classes are overlap with each other.Lets try with other remaining features.

# In[ ]:


sns.FacetGrid(df, hue='Species',height= 5).map(plt.scatter, 'PetalLengthCm', 'PetalWidthCm')
plt.legend()
plt.grid()
plt.show()


# Petal length and petal width gives us a more clean plot than using sepal features.This means that petal features can help us in acheiving more correct predictions.

# # Pair Plots:-
# 
# The pair plots gives us relationship between all the features with each other.With the help of these plots we can easily visualise which feature gives us more better predictions. 

# In[ ]:


sns.pairplot(df, hue = 'Species',height = 3)
plt.show()


# **Observations:**
# 
# By analysing all the plots above, we can see that petal features are being more useful for predictions.Also petal length is fairly good feature than petal width.

# In[ ]:


sns.heatmap(df.corr(),cmap='ocean_r',annot=True)
plt.show()


# Above graph shows that petal length and width have a better correlation between them.

# In[ ]:


sns.violinplot(data = df ,x ='Species',y = 'PetalLengthCm', height= 15)
plt.grid()
plt.show()


# The violin plot shows the probablity density of each class.It also consist box plots at the centre of the curves.The white dot in the middle of the box plot shows 50th percentile of each class.The thinner the curve ,the lower is the probablity density.

# # BUILDING THE MODEL:-
# 
# I use the following ML algorithms for prediction:
# 
# 1. SVM(Support Vector Machine)
# 2. KNN(K-Nearest Neighbour)
# 3. Decision Tree Regression
# 
# According to above visualisation, i use only petals features as they tends to give better result,
# 

# In[ ]:


#importing the libraries
from sklearn.model_selection import train_test_split            #to split the training and testing data
from sklearn.svm import SVC                                     #for support vector machine algorithm
from sklearn.neighbors import KNeighborsClassifier              #for KNN algorithm
from sklearn.tree import DecisionTreeClassifier                 #for DecisionTree algorithm
from sklearn.metrics import confusion_matrix, accuracy_score    #for calculating accuracy


# In[ ]:


x_train = df.iloc[:,2:4].values         #taking petal length and petal width features values

y_train = df.iloc[:,4].values           #taking the target values


# In[ ]:


training_x, test_x, training_y, test_y = train_test_split(x_train, y_train, test_size = .25, random_state =0)
# splitting the dataset into training data and testing data(25%)


# # Support Vector Machine(SVM):-

# In[ ]:


model_1 = SVC()
model_1.fit(x_train,y_train)

prediction_1 = model_1.predict(test_x)

print(confusion_matrix(test_y, prediction_1))
print('\n')
print('Accuracy score:',accuracy_score(test_y,prediction_1))


# # K-Nearest Neighbour(KNN):-

# In[ ]:


model_2 = KNeighborsClassifier(n_neighbors=5)
model_2.fit(x_train,y_train)

prediction_2 = model_2.predict(test_x)

print(confusion_matrix(test_y, prediction_2))
print('\n')
print('Accuracy score:',accuracy_score(test_y,prediction_2))


# # Decision Tree Regression:-

# In[ ]:


model_3 = DecisionTreeClassifier()
model_3.fit(x_train,y_train)

prediction_3 = model_3.predict(test_x)

print(confusion_matrix(test_y, prediction_3))
print('\n')
print('Accuracy score:',accuracy_score(test_y,prediction_3))


# **Observations:**
# 
# Petals features gives more accurate predictions than sepals as shown by heatmap correlations and by applying above algorithms we verify it.
# 
# 
# 
# Guys this is my first notebook on kaggle.I try to make this notebook as simple as i can. I am also a learner and new to this feild. If you find this notebook to be useful then **Please Upvote**.
# 
# Thank you.
# 
