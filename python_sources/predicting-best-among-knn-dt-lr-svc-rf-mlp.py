#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train = pd.read_csv('../input/train.csv', header=None)
trainLabel = pd.read_csv('../input/trainLabels.csv', header=None)
test = pd.read_csv('../input/test.csv', header=None)
print(plt.style.available)
plt.style.use('ggplot')


# In[ ]:


X,y = train,np.ravel(trainLabel)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
a = knn.score(X_test, y_test)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=1)
tree.fit(X_train, y_train)
tree.predict(X_test)
b = tree.score(X_test,y_test)


# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
model.predict(X_test)
c = model.score(X_test,y_test)


# In[ ]:


from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
svc.predict(X_test)
d = svc.score(X_test,y_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
Random_forest = RandomForestClassifier()
Random_forest.fit(X_train, y_train)
Random_forest.predict(X_test)
e = Random_forest.score(X_test, y_test)


# In[ ]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(15, 15))
mlp.fit(X_train, y_train)
mlp.predict(X_test)
f = mlp.score(X_test,y_test)


# In[ ]:


dict = {}


# In[ ]:


scores = [a,b,c,d,e,f]
labels = ['knn','DT','LR','SVC','RF','MLP']
for i in range(len(scores)):
    dict[labels[i]] = scores[i]
print(dict)


# In[ ]:


lists = sorted(dict.items()) 
x, y = zip(*lists) 
plt.plot(x, y,marker='.')
plt.show()


# In[ ]:


test.head()


# In[ ]:


a = svc.predict(test)


# In[ ]:


submission = pd.DataFrame(svc.predict(test))
submission.columns = ['Solution']
submission['Id'] = np.arange(1,submission.shape[0]+1)
submission = submission[['Id', 'Solution']]


# In[ ]:


submission.to_csv('submission_with_scaling.csv', index=False)

