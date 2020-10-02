#!/usr/bin/env python
# coding: utf-8

# # Exploring the dataset
# The first step to undertand what is the Iris dataset and what is possible to build with is explore its data.

# ## Importing the libraries

# In[ ]:


from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ## Loading the dataset

# In[ ]:


# Loading iris dataset and transforming it on a Data Frame
iris_ds = datasets.load_iris()
iris = pd.DataFrame(data=np.c_[iris_ds['data'], iris_ds['target']], columns= iris_ds['feature_names'] + ['target'])
# Printing the classes of flowers
print(iris_ds['target_names'])
# Printing the first 5 rows
iris.head()


# It is possible to see that this dataset has four characteristics (sepal length, sepal width, petal length, and petal width) and a class (target). The classification goes on 0 to 2, representing setosas, versicolors, and virginicas, respectively.

# ## Ploting characteristics
# Attempting to better understand some patterns from Iris dataset, it is possible to plot its characteristics related to the flower classifications.

# In[ ]:


fig, axs = plt.subplots(4, 1, figsize=(15, 10))
plt.subplots_adjust(hspace=0.5)
fig.suptitle('Iris characteristics', fontsize=20)
sns.lineplot(x='sepal length (cm)', y='target', data=iris, ax=axs[0])
sns.lineplot(x='sepal width (cm)', y='target', data=iris, ax=axs[1])
sns.lineplot(x='petal length (cm)', y='target', data=iris, ax=axs[2])
sns.lineplot(x='petal width (cm)', y='target', data=iris, ax=axs[3])
plt.show()


# It is possible to see that each characteristic has some behavior related to flower the classifications. For example, as the petal width increases, the classification changes too.

# # Building classifying model

# In[ ]:


features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
X = iris[features]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=10)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
model


# ## Testing model

# In[ ]:


y_predicted = model.predict(X_test)

predicted_data = X_test.copy()
predicted_data['target'] = y_predicted

fig, axs = plt.subplots(4, 1, figsize=(15, 10))
plt.subplots_adjust(hspace=0.5)
fig.suptitle('Iris characteristics (predictions)', fontsize=20)
sns.lineplot(x='sepal length (cm)', y='target', data=predicted_data, ax=axs[0])
sns.lineplot(x='sepal width (cm)', y='target', data=predicted_data, ax=axs[1])
sns.lineplot(x='petal length (cm)', y='target', data=predicted_data, ax=axs[2])
sns.lineplot(x='petal width (cm)', y='target', data=predicted_data, ax=axs[3])
plt.show()


# It is possible to see that the behavior of each characteristic related to the flower classifications has some similarity to the charts plotted before, with all the data. Therefore, the classification model has some accuracy.

# # Evaluating the model
# However, just get conclusions based on charts is not the best choice, thus let's evaluate the model accuracy with the code below:

# In[ ]:


acc = accuracy_score(y_test, y_predicted)
print('Accuracy:', acc, 'that is, %.2f'%(acc*100),'%')

dmetric = classification_report(y_test, y_predicted)
print('Evaluation report\n', dmetric)


# # Conclusions
# The obtained results have 97.78% of accuracy and, therefore, represent successfully the patterns and behaviors of the Iris dataset and is capable to classify flowers in this context.
