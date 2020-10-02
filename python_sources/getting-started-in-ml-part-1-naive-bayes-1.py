#!/usr/bin/env python
# coding: utf-8

# # Classification using Naive bayes on iris dataset

# ## 1. Importing necessary packages

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# ## 2. Importing the dataset

# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


iris = pd.read_csv("../input/Iris.csv")
print("Type: "+str(type(iris)))
print("Names of column fields: \n"+str(iris.columns))
print("\n Shape of dataset: "+str(iris.shape))
print("Head of dataset: ")
print(iris.head())


# In[ ]:


print("\n Brief description of data: ")
print(iris.describe())


# ## 3.Preprocessing the data

# In[ ]:


# Shuffling data
from sklearn.utils import shuffle
iris = shuffle(iris)


# In[ ]:


# Seperating target and features
X = iris.iloc[:,1:5].values
y = iris.iloc[:,5].values


# In[ ]:


# Scaling and normalization
#X_scale = (X-X.mean(0))/X.std(0)
X_scale = X


# In[ ]:


# Encoding categorical data
labelencoder_y = LabelEncoder()
y_encode = labelencoder_y.fit_transform(y)

print("Name of classes: "+str(labelencoder_y.classes_))
print("Classes: "+str(np.unique(y_encode)))


# ## 4. Visualizing the data

# In[ ]:


print(y.reshape(150,1).shape)
print(X.shape)


# In[ ]:


# Using pairplot available in the seaborn package to visualize data
data = np.hstack((X,y.reshape(150,1)))
columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','Species']
sns.pairplot(pd.DataFrame(data= data, columns= columns),hue="Species")


# In[ ]:


# from above plot we can easily arrive to following conclusions
# 1. from the plot of SepalWidthCm vs PetalWidthCm we can see a clear seperation between Setosa and other #species based on PetalWidth
# 2. from the plot of SepalWidthCm vs SepalLengthCm there is ALMOST  boundary between versicolor and verginica and a clear boundary between setosa and other two species based on the first obeservation we can with 100 certainity and accuracy predict whether specie is setosa or not, based on second obervation we can with high accuracy predict the specie.


# ## 5. Splitting the data into test/train

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scale, y_encode,test_size=0.3)


# ## 6. Classification algorithm - Naive bayes

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)


# ## 7. Prediction

# In[ ]:


y_pred = clf.predict(X_test)


# ## 8. Accuracy

# In[ ]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# ## 9. Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test, y_pred),
                                colorbar=True,
                                show_absolute=False,
                                show_normed=True)
plt.xticks(np.arange(0,3,1),labelencoder_y.classes_,rotation=90,size=15)
plt.yticks(np.arange(0,3,1),labelencoder_y.classes_,size=15)
plt.xlabel('Predicted label',color='red',fontsize=20)
plt.ylabel('True label',color='red',fontsize=20)
plt.show()


# ## 10. Importance of each feature

# In[ ]:


print(list(zip(iris.columns[1:5].values, abs(clf.coef_.sum(axis=0)))))
plt.barh(iris.columns[1:5].values, abs(clf.coef_.sum(axis=0)))

