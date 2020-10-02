#!/usr/bin/env python
# coding: utf-8

# # Prediction Of Male - Female According To The Notes
# 1. [Data Editing](#data_editing)
# 2. [KNN Algorithm](#knn)
#     - [Train Test Split](#knn_train_test_split)
#     - [KNN Model](#knn_model)
#     - [Find K Value](#knn_find_k)
#     - [Accuracy](#knn_accuracy)
# 3. [Support Vector Machine (SVM)](#svm)
# 4. [Naive Bayes Classification](#nbc)

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Data Editing <a id="data_editing"></a>

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")
data


# > **Here I will make the forecast based on the grades they received. Therefore, I do not need any data other than notes. That's why I delete those columns.**

# In[ ]:


data = data.drop(["race/ethnicity","parental level of education","lunch","test preparation course"],axis=1) # axis = 1 => for columns
data


# In[ ]:


# If there is a line with a null value, let's delete it.
data = data.dropna()


# In[ ]:


data.info()


# In[ ]:


data.describe().T


# > **Since I am guessing, I equate girls in the gender column to 0 and boys to 1.**

# In[ ]:


# male = 0 | female = 1
print("gender: ", data.gender.value_counts())
data["gender"] = [ 0 if each == "male" else 1 for each in data.gender]
data


# In[ ]:


plt.scatter(data[data["gender"] == 1]["writing score"],data[data["gender"] == 1]["reading score"],color="blue",label="famale",alpha= 0.3)
plt.scatter(data[data["gender"] == 0]["writing score"],data[data["gender"] == 0]["reading score"],color="red",label="male",alpha= 0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()


# In[ ]:


df = data.copy()


# # KNN Algorithm <a id="knn"></a>
# > * In pattern recognition, the k-nearest neighbors algorithm is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. <br>
# > * You can read [this article](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761) for more information.
# 

# In[ ]:


from IPython.display import Image
Image(url="https://i.ibb.co/KL2vG7W/knn2.jpg")


# In[ ]:


y = data.gender.values
x_data = data.drop(["gender"],axis=1)
x_data


# In[ ]:


x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
x


# ## Train Test Split <a id="knn_train_test_split"></a>

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)

print("x_train",x_train.shape)
print("x_test",x_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)


# ## KNN MODEL <a id="knn_model"></a>

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

prediction = knn.predict(x_test)


# In[ ]:


print("{} nn score: {} ".format(5,knn.score(x_test,y_test)*100))


# ## Find K Value <a id="knn_find_k"></a>
# > We can follow a method like the one below to find the appropriate n_neighbors value.

# In[ ]:


score_list = []

for each in range(20,40):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(20,40),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()


# As can be seen, the max point is the value 30. So the most optimal value is 30.

# ## KNN Algorithm Accuracy <a id="knn_accuracy"></a>

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)

print("{} KNN Score: {} ".format(30,knn.score(x_test,y_test)*100))


# # Support Vector Machine (SVM) <a id="svm"></a>
# > * A support vector machine (SVM) is a supervised machine learning model that uses classification algorithms for two-group classification problems. After giving an SVM model sets of labeled training data for each category, they're able to categorize new text. So you're working on a text classification problem.

# In[ ]:


data = df
data


# In[ ]:


y = data.gender.values
x_data = data.drop(["gender"],axis=1)

# normalization
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)


# In[ ]:


from sklearn.svm import SVC

svm = SVC(random_state=42)
svm.fit(x_train,y_train)

print("Accuracy of SVM Algo: ", svm.score(x_test,y_test)*100)


# # Naive Bayes Classification <a id="nbc"></a>

# In[ ]:


Image(url="https://i.ibb.co/YpP7JY1/1-39-U1-Ln3t-Sd-Fqsf-Qy6ndx-OA.png")


# In[ ]:


data = df
data


# In[ ]:


y = data.gender.values
x_data = data.drop(["gender"],axis=1)

# normalization
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)


# In[ ]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train,y_train)
print("Accuracy of Naive Bayes Algo:",nb.score(x_test,y_test)*100)

