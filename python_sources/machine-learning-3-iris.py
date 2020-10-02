#!/usr/bin/env python
# coding: utf-8

# Using different algorithm like DecisionTreeClassifier, KNeighborsClassifier, Logistic Regression, Random Forest, etc. to solve the iris data set

# In[1]:


from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()


# In[2]:


X =iris.data
Y = iris.target


# In[9]:


X


# In[10]:


Y


# In[3]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)


# In[11]:


print("X_train")
print(X_train)


# In[12]:


print("X_test")
print(X_test)


# In[13]:


print("Y_train")
print(Y_train)


# In[14]:


print("Y_test")
print(Y_test)


# ## DecisionTreeClassifier

# In[4]:


from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train,Y_train)


# In[16]:


prediction = my_classifier.predict(X_test)

print("prediction")
print(prediction)
print("-----------------")
print("accuracy_score")
print( accuracy_score(Y_test,prediction))


# ## KNeighborsClassifier

# In[6]:


from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()
my_classifier.fit(X_train,Y_train)


# In[17]:


prediction = my_classifier.predict(X_test)

print("prediction")
print(prediction)
print("-----------------")
print("accuracy_score")
print( accuracy_score(Y_test,prediction))


# ## Gaussian Naive Bayes

# In[18]:



from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
prediction = gaussian.predict(X_test)

print("prediction")
print(prediction)
print("-----------------")
print("accuracy_score")
print( accuracy_score(Y_test,prediction))


# ## Logistic Regression

# In[19]:



from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
prediction = logreg.predict(X_test)

print("prediction")
print(prediction)
print("-----------------")
print("accuracy_score")
print( accuracy_score(Y_test,prediction))


# ## Support Vector Machines

# In[20]:



from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, Y_train)
prediction = svc.predict(X_test)

print("prediction")
print(prediction)
print("-----------------")
print("accuracy_score")
print( accuracy_score(Y_test,prediction))


# ## Linear SVC

# In[21]:


# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
prediction = linear_svc.predict(X_test)

print("prediction")
print(prediction)
print("-----------------")
print("accuracy_score")
print( accuracy_score(Y_test,prediction))


# ## Random Forest

# In[23]:



from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(X_train, Y_train)
prediction = randomforest.predict(X_test)

print("prediction")
print(prediction)
print("-----------------")
print("accuracy_score")
print( accuracy_score(Y_test,prediction))


# In[ ]:





# In[ ]:




