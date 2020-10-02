#!/usr/bin/env python
# coding: utf-8

# **<font size="4">Welcome to the EDA and Prediction world!</font>**

# <font size="3">Importing Libraries</font>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# <font size="3">Reading Data</font>

# In[ ]:


data = pd.read_csv('../input/Iris.csv')


# In[ ]:


data.head()


# **As there is no need for 'ID' column of the dataset, we will delete it!**

# In[ ]:


del data['Id']


# In[ ]:


data.head()


# <font size="3">Statistical Analysis of Data</font>

# In[ ]:


data.describe()


# In[ ]:


data.info()


# **<font size="2">Total number of Null Values</font>**

# In[ ]:


data.isnull().sum()


# **<font size="2">Replacing the Categorical Variable with numbers!</font>**

# In[ ]:


replace_map = {'Species': {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-verginica': 3}}
labels = data['Species'].astype('category').cat.categories.tolist()
replace_map_comp = {'Species' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
data_new = data.copy()
# Copy of original data, so that it must not get affected by modification
data_new.replace(replace_map_comp, inplace=True)


# In[ ]:


print(replace_map_comp)
# This is the encoding of categorical variable


# **<font size="2">New Data, modified one...</font>**

# In[ ]:


data_new.head()


# **<font size="2">Shape(rows,columns)</font>**

# In[ ]:


data_new.shape


# **<font size="3">Scatter Plots</font>**

# In[ ]:


plt.scatter(data_new['Species'],data_new['SepalLengthCm'])
plt.xlabel('Species')
plt.ylabel('SepalLengthCm')
plt.title('Scatter Plot')
plt.show


# In[ ]:


plt.scatter(data_new['Species'],data_new['SepalWidthCm'])
plt.xlabel('Species')
plt.ylabel('SepalWidthCm')
plt.title('Scatter Plot')
plt.show


# In[ ]:


plt.scatter(data_new['Species'],data_new['PetalLengthCm'])
plt.xlabel('Species')
plt.ylabel('PetalLengthCm')
plt.title('Scatter Plot')
plt.show


# In[ ]:


plt.scatter(data_new['Species'],data_new['PetalWidthCm'])
plt.xlabel('Species')
plt.ylabel('PetalWidthCm')
plt.title('Scatter Plot')
plt.show


# **<font size="3">Heatmap--> Correlation between features</font>**

# In[ ]:


f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(data_new.corr(),annot=True,cmap='magma')
# all variables are highly correlated to species, whether +ve or -ve.


# **<font size="3">Pairplot</font>**

# In[ ]:


sns.pairplot(data_new, hue="Species", size=3)
# sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")
# Used for comparing two variables at a time
# each will have a plot against others as well as itself


# **<font size="3">Facetgrid</font>**

# In[ ]:


sns.FacetGrid(data, hue="Species", size=6).map(sns.kdeplot, "PetalLengthCm").add_legend()


# In[ ]:


sns.FacetGrid(data, hue="Species", size=6).map(sns.kdeplot, "PetalWidthCm").add_legend()


# In[ ]:


sns.FacetGrid(data, hue="Species", size=6).map(sns.kdeplot, "SepalWidthCm").add_legend()


# In[ ]:


sns.FacetGrid(data, hue="Species", size=6).map(sns.kdeplot, "SepalLengthCm").add_legend()


# **<font size="3">Machine Learning</font>**

# In[ ]:


X = data_new.iloc[:,:4].values
y = data_new.iloc[:,-1].values
# data into Input and Output features


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=100)
# Splitting into Train and Test sets


# **<font size="3">Support Vector Machines</font>**

# In[ ]:


from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train,y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# **<font size="3">Evaluation of Model</font>**

# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print('Confusion Matrix: ',confusion_matrix(y_test,y_pred))
print()
print('Accuracy Score: ',accuracy_score(y_test,y_pred))
print()
print(classification_report(y_test,y_pred))


# **<font size="3">Gridsearch CV</font>**

# <font size="2">Shuffling of data</font>

# In[ ]:


from sklearn.utils import shuffle
X,y = shuffle(X,y,random_state=0)


# <font size="2">Different parameters of SVM tested</font>

# In[ ]:


param_grid = [{'C': [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']},
              {'C': [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000], 'kernel': ['linear']}]
# param_grid = {'C': [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000], 'gamma': ['auto'], 'kernel': ['rbf']}


# In[ ]:


from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(estimator=SVC(), param_grid=param_grid, scoring='accuracy',cv=10, verbose=1, refit=True,n_jobs=-1)
grid_search = grid.fit(X, y)


# <font size="3">Best parameter set</font>

# In[ ]:


grid_search.best_params_


# In[ ]:


grid_search.best_score_


# **<font size="5">Completed! Thanks.</font>**

# <font size="3">Please upvote the Kernel if it helped you.</font>
