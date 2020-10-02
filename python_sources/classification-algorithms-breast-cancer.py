#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this notebook, I will apply classification algorithms with scikit-learn library. Firstly, EDA(Exploratory Data Analysis) will be applied to dataset. Then, different algorithms will classify dataset.
# 
# [0. EDA(Exploratory Data Analysis)](#1)
# 
# [1. K-Nearest Neighbor Algorithm](#2)
# 
# [2. Support Vector Machine(SVM)](#3)
# 
# [3. Naive-Bayes Classification](#4)
# 
# [4. Decision Tree Classification](#5)
# 
# [5. Random Forest Classification](#6)
# 
# [6.  Conclusion](#7)
# 
# [7.  References](#8)
# 

# <a id="1"></a> <br>
# ## EDA (Exploratory Data Analysis)

# In[ ]:


# Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv('../input/breastCancer.csv')
data.head()


# In[ ]:


# Clear the noisy attributes
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.tail()


# In[ ]:


M = data[data.diagnosis=='M']
B = data[data.diagnosis=='B']


# In[ ]:


plt.scatter(M.radius_mean,M.texture_mean,color='red',label='Malignant',alpha=0.3)
plt.scatter(B.radius_mean,B.texture_mean,color='green',label='Benign',alpha=0.3)
plt.xlabel('Malignant')
plt.ylabel('Benign')
plt.legend()
plt.show()


# In[ ]:


# Change M and B values to 0 and 1
# Prepare x and y values for KNN algorithm
data.diagnosis= [1 if each=="M" else 0 for each in data.diagnosis]
y=data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)


# In[ ]:


# Normalization
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[ ]:


# Train-Test Split for Learning
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# <a id="2"></a><br>
# ## KNN(K-Nearest Neighbor) Classification
# 
# ![KNN](https://www.kdnuggets.com/wp-content/uploads/knn2.jpg)

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3) #k=3
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("{} nn score: {}".format(3,knn.score(x_test,y_test)))


# In[ ]:


# Hyperparameter Tuning
score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()


# <a id="3"></a><br>
# ## SUPPORT VECTOR MACHINE(SVM)

# ![](http://www.saedsayad.com/images/SVM_2.png)
# 
# I will not explain how SVM algorithm works. However, I found a great page for understand SVM algorithm. 
# So you can look detailed information about SVM in there : [Support Vector Machine](http://www.saedsayad.com/support_vector_machine.htm)
# 
# **Note : Our x,y, train-test split(x_data,y_data) values are prepared from the previous algorithm**

# In[ ]:


from sklearn.svm import SVC
svm = SVC(random_state=1) # Return the same value every time
svm.fit(x_train,y_train)

# test
print("primy accuracy of SVM algorithm : ",svm.score(x_test,y_test))


# <a id="4"></a><br>
# ## Naive-Bayes Classification

# Here is the another probabilstic approach for machine learning. You can look the Bayes Theorem from this page.
# ![Naive-Bayes](http://www.saedsayad.com/images/Bayes_rule.png)
# [Naive-Bayes Classification](http://www.saedsayad.com/naive_bayesian.htm)
# 
# **Note : Our x,y, train-test split(x_data,y_data) values prepared from the previous algorithm**

# In[ ]:


# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

#test
print("Accuracy of Naive-Bayes Algorithm",nb.score(x_test,y_test))


# <a id="5"></a><br>
# ## Decision Tree Classification

# <img src="https://tr.akinator.com/bundles/elokencesite/images/akinator.png?v95" alt="drawing" height="300" width="300"/>

# Have you ever played Akinator Game? It is a great example of decision trees. If you haven't yet, let me explain. The goal of the game is predict a famous character, based of bunch of questions which asked to us. After the each answer, questions are getting more relevant to famous person which is in our mind. Finally, akinator shows his predict. Mostly, he justifies about the prediction.
# 
# Decision tree algorithm works behind the akinator. There are splits which decides to person to based on the answers. For example if we are looking for a person which is blonde then it redirects the future predictions to "blonde" people. "Blonde" input is given by the user as an answer for a question.
# 
# If you want to play the game, the link is below.
# 
# https://en.akinator.com/

# ![](https://annalyzin.files.wordpress.com/2016/07/decision-trees-titanic-tutorial.png?w=616&h=342)
# Source : https://annalyzin.files.wordpress.com/2016/07/decision-trees-titanic-tutorial.png?w=616&h=342

# You can see the splits. It can be more than two splits. In this example, we have a binary tree. The bottom nodes known as "leaf". These are our predictions. The top node is "root" node. In a nutshell, let's start the code.
# 
# * EDA and Normalization have already done in the previous sections. 
# * Train-Test split is obtained in the previous sections. But I want to change percentage of test split to 15%
# 
# Let's start at the algorithm.

# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=42)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
# Accuracy
print("Accuracy of Decision Tree Algorithm",dt.score(x_test,y_test))


# <a id="6"></a><br>
# # Random Forest Classification

# * The random forest is a "ensemble learning" algorithm. So, it includes more than one classification algorithm. The "forest" name comes from to our trees. In a nutshell, random forest classification is includes 'n' trees in itself. 
# 
# * While we make predictions, different results(classes) can occur. In order to reduce our prediction to a singe class, we will use "Majority Voting". It is simple, which count of class is bigger than to other it will be final result.
# 
# * Estimator = Number of trees for our random forest classification model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,random_state=1) # Number of tree = 100
rf.fit(x_train,y_train)
print("Accuracy of Random Forest Algorithm",rf.score(x_test,y_test))


# We can see that, accuracy is increased. However, we can't be sure about the number of estimators. In order to choose best number of estimators:

# In[ ]:


accuracy_list=[]
for i in range(1,11,1):
    rf = RandomForestClassifier(n_estimators=i,random_state=1) # Number of tree = 100
    rf.fit(x_train,y_train)
    accuracy_list.append(rf.score(x_test,y_test))
    #print("Accuracy of Random Forest Algorithm for {} trees: {}".format(i,rf.score(x_test,y_test)))
plt.plot(range(1,11),accuracy_list)
plt.xlabel("Number of estimators")
plt.ylabel("Accuracy")
plt.show()


# We can see that, optimum number of estimators is 4. There is no need to 100 trees. Hence, we can obtain same result with less trees.

# <a id="7"></a><br>
# ## Conclusion
# 
# * You have seen which algorithm is better. Of course **none of them**. All algorithms have trade-offs. You should pick your algorithm for your scenario. 
# 
# * Do not forget tuning **hyperparameters** for the better results.
# 

# <a id="8"></a><br>
# ## References
# 
# https://www.kaggle.com/kanncaa1/machine-learning-tutorial-for-beginners
# 
# http://www.saedsayad.com/
# 
# 
