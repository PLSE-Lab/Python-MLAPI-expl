#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# The Iris dataset contains measurements for 150 iris flowers from three different species.
# 
# The three classes in the Iris dataset:
# 
#     Iris-setosa (n=50)
#     Iris-versicolor (n=50)
#     Iris-virginica (n=50)
# 
# The four features of the Iris dataset:
# 
#     sepal length in cm
#     sepal width in cm
#     petal length in cm
#     petal width in cm

# #### I will try to apply KNN and SVM on the dataset and try to improve their performances

# In[ ]:


#Importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style = 'whitegrid'
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


iris = pd.read_csv("../input/Iris.csv",index_col=0)


# #### Let's check the head of the Dataset

# In[ ]:


iris.head()


# #### Some exploratory data analysis

# In[ ]:


sns.pairplot(iris,hue='Species',palette='dark', markers='o')
plt.show()


# #### From the pairplot we can identify that Setosa is the most seperable from other species
# 
# ### Creating a kde plot of sepal_length versus sepal width for Setosa species

# In[ ]:


setosa = iris[iris['Species']=='Iris-setosa']
plt.figure(figsize=(8,5))
sns.kdeplot( setosa['SepalWidthCm'], setosa['SepalLengthCm'],cmap="inferno", shade=True, shade_lowest=False)
plt.show()


# In[ ]:


sns.violinplot(y='Species', x='SepalLengthCm', data=iris, inner='quartile')
plt.show()
sns.violinplot(y='Species', x='SepalWidthCm', data=iris, inner='quartile')
plt.show()
sns.violinplot(y='Species', x='PetalLengthCm', data=iris, inner='quartile')
plt.show()
sns.violinplot(y='Species', x='PetalWidthCm', data=iris, inner='quartile')
plt.show()


# ### Standardizing the variables for applying KNN

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(iris.drop('Species',axis=1))


# In[ ]:


scaled_features = scaler.transform(iris.drop('Species',axis=1))


# In[ ]:


df = pd.DataFrame(scaled_features,columns=iris.columns[:-1])
df.head()


# #### Train test split

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features,iris['Species'],test_size=0.3, random_state=55)


# ### Applying KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

# Starting with k=1
knn = KNeighborsClassifier(n_neighbors=1)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


pred = knn.predict(X_test)


# ### Predictions and Evaluations

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))
print("\n")
print(classification_report(y_test,pred))


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
accuracy = accuracy_score(y_test,pred)
print("KNN Accuracy: %.2f%%" % (accuracy * 100.0))


# #### The accuracy is not bad. Trying to explore whether a better K value can be chosen or not

# ### Choosing a k value

# In[ ]:


error_rate = []

for i in range(1,50):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(15,8))
plt.plot(range(1,50),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='violet', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()


# #### As we can see here in this case, higher the K value after 25, the error rate is getting higher.
# #### k = 1  was a good choice, still we will try to apply k = 5 to see the results.  We are not applying k=20 or k=22 as that will overfit the model 

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=5')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# #### As we can see from the confusion matrix, though we have got one point, still two points are hard to get. Trying to get those 2 points the model can be overfitted.

# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
accuracy = accuracy_score(y_test,pred)
print("KNN Accuracy: %.2f%%" % (accuracy * 100.0))


# ### Applying SVM

# In[ ]:


iris.head()


# In[ ]:


# Train-test split for SVM, as we don't want the scaled features to put in our SVM model

X_svm = iris.drop('Species',axis=1)
y_svm = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X_svm, y_svm, test_size=0.3,random_state=60)


# #### Trainng the SVM model

# In[ ]:


from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train,y_train)


# ### Model Evaluation

# In[ ]:


predictions = svc_model.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
accuracy = accuracy_score(y_test,predictions)
print("SVM Accuracy: %.2f%%" % (accuracy * 100.0))


# #### The model is pretty good, but still we couldn't classify those two points that couldn't be classified with the KNN model.
# #### But we can try to tune the parameters to get a even better result (just for practice)
# #### To tune the hyperparameters, we will use the GridSearch method.

# In[ ]:


from sklearn.model_selection import GridSearchCV


# #### Creating a dictionary of parameters to feed Gridsearch. Gridsearch will choose the best parameters for SVM

# In[ ]:


param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [1,0.1,0.01,0.001,0.0001]} 


# In[ ]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)


# #### We can inspect the best parameters by some of Gridsearch functions

# In[ ]:


grid.best_params_


# In[ ]:


grid.best_estimator_


# ### Prediction and Evaluation of SVM

# In[ ]:


grid_predictions = grid.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,grid_predictions))
print("\n")
print(classification_report(y_test,grid_predictions))


# ### Here from the confusion matrix we can see that, we managed to classify one of those points using GridSearch that we couldn't  classify in the previous stages.

# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
accuracy = accuracy_score(y_test,grid_predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

