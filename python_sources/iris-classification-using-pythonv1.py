#!/usr/bin/env python
# coding: utf-8

# ##Decision Tree Classifier For IRIS Dataset##

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

iris = pd.read_csv("../input/Iris.csv") 
iris = iris.drop('Id', 1)


# In[ ]:


iris.head()


# In[ ]:


print(iris.shape)


# In[ ]:


iris["Species"].value_counts()


# In[ ]:


names = iris['Species'].unique()#set(iris['Species'])

x,y = iris['PetalLengthCm'],  iris['PetalWidthCm']
#plt.figure(1)
#plt.subplot(221)
for name in names:
    cond = iris['Species'] == name
    plt.plot(x[cond], y[cond], linestyle='none', marker='o', label=name)

plt.legend(numpoints=1, loc='lower right')
plt.title("The Iris Data Set", fontsize=15)
plt.xlabel('Petal Length (cm)', fontsize=12)
plt.ylabel('Petal Width (cm)', fontsize=12)
plt.show()


# In[ ]:


from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
X = iris[[0,1,2,3]]
Y = iris[[4]]
print(X[1:5])
print(Y.head())  


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
#print(len(X_train))
#print(len(X_test))
#print(X_test)


# In[ ]:


tree = DecisionTreeClassifier()#criterion='entropy'
                               #, max_depth = 3
                               #, random_state = 0)
tree.fit(X_train,Y_train)
Y_pred_tree = tree.predict(X_test)

print("Accuracy: %.2f" % accuracy_score(Y_test,Y_pred_tree))


# <img src="http://scikit-learn.org/stable/_images/iris.svg">

# In[ ]:


x1=pd.DataFrame([{'SepalLengthCm': 4.9, 'SepalWidthCm': 3.0, 'PetalLengthCm': 1.4, 'PetalWidthCm': 0.2}])
test_pred = tree.predict(x1)
print(test_pred)


# In[ ]:





# In[ ]:




