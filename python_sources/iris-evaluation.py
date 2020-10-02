#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import os
os.listdir("../input")


# In[ ]:


iris=pd.read_csv("../input/Iris.csv")


# In[ ]:


iris.head()


# In[ ]:


iris.info()


# In[ ]:


iris.tail()


# In[ ]:


iris1=iris.drop("Id", axis=1)


# In[ ]:


iris1.columns= ['sepal_length', 'sepal_width' , 'petal_length', 'petal_width','species']


# In[ ]:


iris1.head()


# In[ ]:


iris1['Species']=iris['Species'].astype('category')
iris1.dtypes


# In[ ]:


print(iris1.Species.unique())


# In[ ]:


print(iris1['Species'].value_counts())
iris2=pd.DataFrame(iris['Species'].value_counts())
iris2


# In[ ]:


iris1.shape


# In[ ]:


iris1.describe()


# In[ ]:


iris1.size


# In[ ]:


iris1.isnull().sum()


# In[ ]:


iris1.min()


# In[ ]:


iris1.max()


# In[ ]:


iris1.median()


# In[ ]:


iris1['Species'].value_counts().plot(kind="bar");


# In[ ]:


sns.set(style="whitegrid", palette="GnBu_d", rc={'figure.figsize':(11.7,8.27)})

title="Compare the Distributions of Sepal Length"

sns.boxplot(x="species", y="sepal_length", data=iris1)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()


# In[ ]:


title="Compare the Distributions of Sepal Width"

sns.boxplot(x="species", y="sepal_width", data=iris1)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()


# In[ ]:


title="Compare the Distributions of Petal Length"

sns.boxplot(x="species", y="petal_length", data=iris1)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()


# In[ ]:


title="Compare the Distributions of Petal width"

sns.boxplot(x="species", y="petal_width", data=iris1)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()


# In[ ]:


sns.countplot(x='petal_length', data = iris1)


# In[ ]:


sns.countplot(x='petal_width', data = iris1)


# In[ ]:


plt.figure(figsize=(7,4)) 
sns.heatmap(iris1.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())
plt.show()


# In[ ]:


corr = iris1.corr()
corr


# In[ ]:


# import correlation matrix to see parametrs which best correlate each other
# According to the correlation matrix results Petal LengthCm and
#PetalWidthCm have positive correlation which is proved by the scatter plot discussed above

import seaborn as sns
import pandas as pd
corr = iris1.corr()
plt.figure(figsize=(10,8)) 
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           cmap='viridis', annot=True)
plt.show()


# 

# In[ ]:


# Modify the graph above by assigning each species an individual color.
sns.FacetGrid(iris1, hue="Species", size=5)    .map(plt.scatter, "sepal_length", "sepal_width")    .add_legend()
plt.show()


# In[ ]:


X = iris.drop(['Id', 'Species'], axis=1)
y = iris['Species']
# print(X.head())
print(X.shape)
# print(y.head())
print(y.shape)


# In[ ]:


k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    scores.append(metrics.accuracy_score(y, y_pred))
    
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X, y)
y_pred = logreg.predict(X)
print(metrics.accuracy_score(y, y_pred))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[ ]:


Model = GaussianNB()
Model.fit(X_train, y_train)

y_pred = Model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[ ]:



Model = DecisionTreeClassifier()
Model.fit(X_train, y_train)
y_pred = Model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[ ]:


Model=RandomForestClassifier(max_depth=2)
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[ ]:




