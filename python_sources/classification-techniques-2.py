#!/usr/bin/env python
# coding: utf-8

# # Classification and Regression Tree (CART)
# 
# # KNN 

# In[ ]:


from IPython.display import Image
Image("KNN-Algorithm-using-Python-1.png")


#  - KNN is a simple yet powerful classification algorithm.
#  - It requires no training for making predictions, which is typically one of the most difficult parts of a machine learning algorithm.
#  - The KNN algorithm have been widely used to find document similarity and pattern recognition

# ## KNN's Idea:- 

# - The intuition behind the KNN algorithm is one of the simplest of all the supervised machine learning algorithms. 
# - It simply calculates the distance of a new data point to all other training data points.
# - The distance can be of any type e.g Euclidean or Manhattan etc.
# - It then selects the K-nearest data points, where K can be any integer. 
# - Finally it assigns the data point to the class to which the majority of the K data points belong.

#   - KNN: Look at the K closest labeled data points
#   - Classification method.
#   - First we need to train our data. Train = fit
#   - fit(): fits the data, train the data.
#   - predict(): predicts the data 
#   - x: features
#   - y: target variables(normal, abnormal)
#   - n_neighbors: K. In this example it is 3. it means that Look at the 3 closest labeled data points

# In[ ]:


from IPython.display import Image
Image("class_prediction.jpg")


# ## SUPERVISED LEARNING
#     Supervised learning: It uses data that has labels. Example, there are orthopedic patients data that have labels normal and abnormal.
#     There are features(predictor variable) and target variable. Features are like pelvic radius or sacral slope
#     Target variables are labels normal and abnormal
#     Aim is that as given features(input) predict whether target variable(output) is normal or abnormal
#     Classification: target variable consists of categories like normal or abnormal
#     Regression: target variable is continious like stock market
#    
# 

# ## Problem Statement :- Data for classifying patients based on two classes
# 
# ## Biomechanical features of orthopedic patients
# 
# ## Columns:-
#     pelvic_incidence
#     pelvic_tilt numeric
#     lumbar_lordosis_angle
#     sacral_slope
#     pelvic_radius
#     degree_spondylolisthesis
#     class

# ## About Data :-
#     The data have been organized in two different but related classification tasks.
# 
#     column_3C_weka.csv (file with three class labels)
# 
#     The first task consists in classifying patients as belonging to one out of three categories: Normal (100 patients), Disk Hernia (60 patients) or Spondylolisthesis (150 patients).
#     
#     column_2C_weka.csv (file with two class labels)
# 
#     For the second task, the categories Disk Hernia and Spondylolisthesis were merged into a single category labelled as 'abnormal'. Thus, the second task consists in classifying patients as belonging to one out of two categories: Normal (100 patients) or Abnormal (210 patients).

# ## Field Descriptions:
# 
#     Each patient is represented in the data set by six biomechanical attributes derived from the shape and orientation of the pelvis and lumbar spine (each one is a column):
# 
#     pelvic incidence
#     pelvic tilt
#     lumbar lordosis angle
#     sacral slope
#     pelvic radius
#     grade of spondylolisthesis

# ## Step 1:- Imports

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# read csv (comma separated value) into data
data = pd.read_csv('column_2C_weka.csv')
print(plt.style.available) # look at available plot styles
plt.style.use('_classic_test')


# ## STEP 2:-EXPLORATORY DATA ANALYSIS (EDA)

# In[ ]:


data.head()


# - Features are pelvic_incidence, pelvic_tilt numeric, lumbar_lordosis_angle, sacral_slope, pelvic_radius and degree_spondylolisthesis
# - Target variable  is class

# #### ROWS AND COLUMNS

# In[ ]:


data.shape


# In[ ]:


print('No of columns in the dataset:',data.columns.size)
print("Name of Columns:\n",data.columns.values)


# #### To check about Numerics/Categorical Columns

# In[ ]:


data.info()


#  - length: 310 (range index)
#  - Features are float
#  - Target variables are object that is like string

# #### Summary of the Data

# In[ ]:


data.describe()


# #### Pairplot on the data

# In[ ]:


sns.pairplot(data,hue="class",palette="Set2")
plt.show()


# ##### pd.plotting.scatter_matrix:
# 
#     green: normal and red: abnormal
#     c: color
#     figsize: figure size
#     diagonal: histogram of each features
#     alpha: opacity
#     s: size of marker
#     marker: marker type

# #### Scatter Matrix on the Data to find the relations among Features

# In[ ]:


color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 200,
                                       marker = '*',
                                       edgecolor= "black")
plt.show()


# ## Okay, scatter matrix there are relations between each feature but how many normal(green) and abnormal(red) classes are there.

# #### Class Distributuon

# In[ ]:


data.loc[:,'class'].value_counts()


# In[ ]:


sns.countplot(x="class", data=data)


# #### KNN Invoking

# In[ ]:


# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']
knn.fit(x,y)
prediction = knn.predict(x)
print('Prediction: {}'.format(prediction))


# ## Measuring model performance:
#     Accuracy which is fraction of correct predictions is commonly used metric.

# ##  Split our data train and test sets.
# 
#     train: use train set by fitting
#     test: make prediction on test set.
#     With train and test sets, fitted data and tested data are completely different
#     train_test_split(x,y,test_size = 0.3,random_state = 1)
#     x: features
#     y: target variables (normal,abnormal)
#     test_size: percentage of test size. Example test_size = 0.3, test size = 30% and train size = 70%
#     random_state: sets a seed. If this seed is same number, train_test_split() produce exact same split at each time
#     fit(x_train,y_train): fit on train sets
#     score(x_test,y_test)): predict and give accuracy on test sets

# In[ ]:


# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
knn = KNeighborsClassifier(n_neighbors = 3)
x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
#print('Prediction: {}'.format(prediction))
print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test)) # accuracy


# ## Model complexity:
# 
#     K has general name. It is called a hyperparameter. For now just know K is hyperparameter and we need to choose it that gives best performance.
#     Literature says if k is small, model is complex model can lead to overfit. It means that model memorizes the train sets and cannot predict test set with good accuracy.
#     If k is big, model that is less complex model can lead to underfit.
#     At below, I range K value from 1 to 25(exclude) and find accuracy for each K value. As you can see in plot, when K is 1 it memozize train sets and cannot give good accuracy on test set (overfit). Also if K is 18, model is lead to underfit. Again accuracy is not enough. However look at when K is 18(best performance), accuracy has highest value almost 88%.

# In[ ]:


# Model complexity
neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))

# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('Value VS Accuracy',fontsize=20)
plt.xlabel('Number of Neighbors',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.xticks(neig)
plt.grid()
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))


# #### TUNING AND HYPERPARAMETERS

# #### CASE 1:-

# In[ ]:


# grid search cross validation with 1 hyperparameter
from sklearn.model_selection import GridSearchCV
grid = {'n_neighbors': np.arange(1,25)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV
knn_cv.fit(x,y)# Fit

# Print hyperparameter
print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 
print("Best score: {}".format(knn_cv.best_score_))


# #### CASE 2:-

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
grid = {'n_neighbors': np.arange(1,25)}
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV
knn_cv.fit(x_train, y_train)

print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(knn_cv.score(x_train, y_train)))
print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(knn_cv.score(x_test, y_test)))
print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 
print("Best score: {}".format(knn_cv.best_score_))


# #### CASE 3:-

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=12, p=6, metric='minkowski')
knn.fit(x_train, y_train)

print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(knn.score(x_train, y_train)))
print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(knn.score(x_test, y_test)))


# #### CASE 4:-

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
train_score_knn=[]
test_score_knn=[]
for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=12, p=i, metric='minkowski')
    knn.fit(x_train, y_train)
    train_score_knn.append(knn.score(x_train, y_train))
    test_score_knn.append(knn.score(x_test, y_test))
    print('For p value=',i)
    print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(knn.score(x_train, y_train)))
    print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(knn.score(x_test, y_test)))
    print()


# ## Pros and Cons of KNN
# 
# 
# ## Pros
# - It is extremely easy to implement
# - It is lazy learning algorithm and therefore requires no training prior to making real time predictions. This makes the KNN algorithm much faster than other algorithms that require training e.g SVM, linear regression, etc.
# - Since the algorithm requires no training before making predictions, new data can be added seamlessly.
# - There are only two parameters required to implement KNN i.e. the value of K and the distance function (e.g. Euclidean or Manhattan etc.)
# 
# ## Cons
# - The KNN algorithm doesn't work well with high dimensional data because with large number of dimensions, it becomes difficult for the algorithm to calculate distance in each dimension.
# - The KNN algorithm has a high prediction cost for large datasets. This is because in large datasets the cost of calculating distance between new point and each existing point becomes higher.
# - The KNN algorithm doesn't work well with categorical features since it is difficult to find the distance between dimensions with categorical features.

# In[ ]:




