#!/usr/bin/env python
# coding: utf-8

# ## Applying classification on Iris dataset

# *Best way to learn is to visualize the stuff. Keeping this in mind, I have created a python flask webapp which deals with iris dataset and predict the species based on attributes chosen by User*
# 
# More info: https://github.com/vjcalling/Species-Predictor
# 
# 

# In[ ]:


from IPython.display import Image
Image("../input/species-predictor/main_app.png")


# In[ ]:


import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv("../input/iris-dataset/iris.csv")


# In[ ]:


df.head(2)


# In[ ]:


df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# In[ ]:


sns.pairplot(df); #relationship b/w attr


# In[ ]:


array = df.values
X = array[:,0:4]    #1st 4 cols are training attributes
Y = array[:,4]      #5th col is the class (species name in our case)


# In[ ]:


validation_size = 0.20
seed = 41
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()


# In[ ]:


logit.fit(X_train,Y_train)


# In[ ]:


logit.predict(X_test)


# In[ ]:


mysample = np.array([4.5,3.2,1.2,0.5])
ex1 = mysample.reshape(1,-1)
logit.predict(ex1)


# In[ ]:


ex2 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)
logit.predict(ex2)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[ ]:


knn = KNeighborsClassifier()
dtree = DecisionTreeClassifier()
svm = SVC()


# In[ ]:


knn.fit(X_train, Y_train)
print("accuracy :" , knn.score(X_test,Y_test))


# In[ ]:


dtree.fit(X_train, Y_train)
print("accuracy :" , dtree.score(X_test,Y_test))


# In[ ]:


svm.fit(X_train, Y_train)
print("accuracy :" , svm.score(X_test,Y_test))


# In[ ]:




