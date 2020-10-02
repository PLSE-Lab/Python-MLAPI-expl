#!/usr/bin/env python
# coding: utf-8

# # Cihan Yatbaz
# ###  03 / 11 / 2018
# 
# 
# 
# 1.  [Introduction:](#0)
# 2. [Exploratory Data Analysis (EDA) :](#1)
# 3. [K-NEAREST NEIGHBORS ( KNN ) :](#2)
#     1. [KNN  :](#3)
#     2. [Model Complexity  :](#4)  
# 4. [REGRESSION :](#5)
# 5. [Support Vector Machine ( SVM) :](#6)
# 6. [Naive Bayes :](#7)
# 7. [Decission Tree :](#8)
# 8. [Random Forest :](#9)
# 9. [Confusion Matrix :](#10)
# 10. [CONCLUSION :](#11)
# 

# <a id="0"></a> <br>
# # 1) Introduction
# 
# We will be working on this kernel Biomechanical features of orthopedic patients data. This kernel will do some inspections with KNN. Firstly we will examine dataset. Let's start 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings
#ignore warnings
warnings.filterwarnings("ignore")
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf-8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# read csv(comma separated value) into data
data = pd.read_csv("../input/column_2C_weka.csv")


# In[ ]:


data.columns  #Columns in our data 


# We have Abnormal and Normal values in our data and let's see them now

# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


#Now let's look at the data of our data
data.info()


# In[ ]:


data.describe()


# <a id="1"></a> <br>
# # 2) Exploratory Data Analysis (EDA)

# ### Scatter Matrix Plot

# In[ ]:


colors = ['cyan' if i == 'Normal' else 'orange' for i in data.loc[:,'class']]
pd.plotting.scatter_matrix(data.loc[:, data.columns !='class'],
                                       c = colors,
                                       figsize =[15,15],
                                       diagonal ='hist',    # histogram of each features
                                       alpha = 0.5,
                                       s = 200,
                                       marker = '*',
                                       edgecolor="black"
                          )
plt.savefig('graph4.png')
plt.show()


# ### Count plot
# Now let's see how many of our Normal and Abnormal values exist

# In[ ]:


sns.countplot(x="class", data=data)
data.loc[:,'class'].value_counts()


# ### Which value belongs to the selected point?

# In[ ]:


# We list them separately according to 'Abnormal' and 'Normal' properties
N = data[data['class']=="Normal"]
A = data[data['class']=="Abnormal"]
print("NORMAL")
N.info()
print()
print("ABNORMAL")
A.info()


# In[ ]:


# Scatter Plot
plt.figure(figsize=[12,8])
plt.scatter(N.pelvic_radius, N.pelvic_incidence, color="cyan", label="Normal")
plt.scatter(A.pelvic_radius, A.pelvic_incidence, color="orange", label="Abnormal")
plt.xlabel("radius_mean")
plt.ylabel("pelvic_incidence")
plt.legend() # To show labels
plt.savefig('graph3.png')
plt.show()


# ### In the figure above, how can we predict whether a point we want to find is Normal or Abnormal? Let's learn together.
# * First we set a point.
# * Then we choose the value 3 closest to this point.
# * If the Abnormal number is higher in these values, it is Abnormal in your new value. If the number of Normals is high, our new value becomes normal.
# 

# <a id="2"></a> <br>
# # 3) K-NEAREST NEIGHBORS ( KNN )

# <a id="3"></a> <br>
# ## A) KNN

# In[ ]:


data['class'] = [1 if each=='Normal' else 0 for each in data['class']]
data_class = data['class']   # This is what we do for convenience
y = data_class.values
x_d = data.drop(["class"], axis=1)   # We will use other features except Class


# * Now we are doing normalization. Because if some of our columns have very high values, they will suppress other columns and do not show much.
# * Formulel : (x- min(x)) / (max(x) - min(x))

# In[ ]:


# Normalization
x = (x_d - np.min(x_d)) / (np.max(x_d) - np.min(x_d))


# * Now we reserve 70% of the values as 'train' and 30% as 'test'.

# In[ ]:


#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# Let's create our KNN model
# * n_neighbours = k  ----->   We determine the k value and try to increase the accuracy of the result

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=22)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)


# In[ ]:


print(" {} nn score : {}".format(22, knn.score(x_test, y_test)))


# ### Let's try to find the best k value with loop now

# In[ ]:


score_list = []
for each in range(1,30):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
#Plot
plt.figure(figsize=[13,8])
plt.plot(range(1,30),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.savefig('graph2.png')
plt.show()


# ### If k value has a value of 22 or 26, our test results will give the best result.

# <a id="4"></a> <br>
# ## B) Model Complexity
# * We can use Model Complexity to find the best value.  This way we can easily find the best result by comparing the Value and Accuracy.

# In[ ]:


# Model complexity
rand = np.arange(1,30)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(rand):
    # k from 1 to 30(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # fit with knn
    knn.fit(x_train, y_train)
    train_accuracy.append(knn.score(x_train, y_train))           # train accuracy
    test_accuracy.append(knn.score(x_test, y_test))              # test accuracy

# Plot
plt.figure(figsize=[13,8])
plt.plot(rand, test_accuracy, label='Testing Accuracy' , color='red')
plt.plot(rand, train_accuracy, label='Training Accuracy', color='black')
plt.legend()
plt.title(' K Value vs Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('K Values')
plt.xticks(rand)
plt.savefig('graph.png')
plt.show()

print("Best accuracy is {} with K = {}".format(np.max(test_accuracy), 1+test_accuracy.index(np.max(test_accuracy))))


# <a id="5"></a> <br>
# # 4) REGRESSION 
# * Let's create data1 that includes pelvic_incidence that is feature and sacral_slope that is target variable

# In[ ]:


data1 = A  # We have previously defined A to Class Abnormal.
x = np.array(data1.loc[:, 'pelvic_incidence']).reshape(-1,1)
y = np.array(data1.loc[:, 'sacral_slope']).reshape(-1,1)

# Scatter
plt.figure(figsize=[10,10])
plt.scatter(x=x, y=y)
plt.xlabel("pelvic incidence")
plt.ylabel("sacral slope")
plt.show()


# In[ ]:


# LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score    
#Regression
reg = LinearRegression()
# Fit
reg.fit(x,y)

# Prediction: 
pred_space = np.linspace(min(x), max(x)).reshape(-1,1)
y_head = reg.predict(pred_space)

# r2 score with LinearRegression
print('R^2 score: ',reg.score(x, y))
# r2 score with metrics
print('R^2 score metrics: ', r2_score(y, reg.predict(x)))

# Plot regression line and scatter
plt.subplots(figsize=(12,10))
plt.plot(pred_space, y_head, color='red', linewidth=3)
plt.scatter(x=x,y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.savefig('graph6.png')
plt.show()


# <a id="6"></a> <br>
# # 5) Support Vector Machine ( SVM)
# *  Specifies the line that will pass between 2 values in the table and tries to keep margin highest

# In[ ]:


from sklearn.svm import SVC
svm = SVC(random_state = 1)
svm.fit(x_train, y_train)

#test
print("Accuracy of SVM Score : ", svm.score(x_test, y_test))


# <a id="7"></a> <br>
# # 6) Naive Bayes
# * Determines the probability that the point in the selected circle can be A or B.

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
#test
print("Accuracy of Naive Score : ", nb.score(x_test, y_test))


# <a id="8"></a> <br>
# # 7) Decission Tree
# * It allows us to differentiate between different classes with Splits

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
#test
print("Accuracy of Decision Tree Score : ", dtc.score(x_test, y_test))


# <a id="9"></a> <br>
# # 8) Random Forest
# * We have multiple decision trees in a random forest and Random forest giving us the most correct results by doing the transaction with decision tree while we process.
# * As a result, it improves the accuracy and reliability of the model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100, random_state=1)
#n_estimators =100 -> Determines how many trees we have
rfc =rfc.fit(x_train, y_train)
#test
print("Random Forest Score", rfc.score(x_test, y_test))


# <a id="10"></a> <br>
# # 9) Confusion Matrix
# * Confusion Matrix : With the result, it shows how many mistakes we have from A and B.
# * Confusion Matrix enables us to visualize these results

# In[ ]:


from sklearn.metrics import confusion_matrix
y_pred = rfc.predict(x_test)
y_true = y_test

cm = confusion_matrix(y_true, y_pred)

# confusion matrix visualization

f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cm, annot = True, linewidths = 0.5, linecolor = 'green', fmt=".0f", ax=ax)
plt.savefig('graph7.png')
plt.show()


# #### 0 = Abnormal values       &      1 = Normal values
# * In our test, there are 60 true, 6 false results for Abnormal.
# * In our test, there are 20 true, 7 false results for Normal.

# <a id="11"></a> <br>
# > # CONCLUSION                                                                                                                                                      
# Thank you for your votes and comments                                                                                                                                              
# <br>**If you have any suggest, May you write for me, I will be happy to hear it.**
