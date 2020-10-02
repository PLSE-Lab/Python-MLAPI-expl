#!/usr/bin/env python
# coding: utf-8

# # KNN classification of Iris Species dataset

# In[ ]:


##import all the libraries and framework
import numpy as np
import pandas as pd


# In[ ]:


## loading a dataset

iris_data = pd.read_csv('../input/Iris.csv')


# In[ ]:


iris_data.head(5)      ##explore head of data


# In[ ]:


iris_data.shape


# #### we can say from above that dataset contains 150 rows and 6 columns
# 
# now describe dataset

# In[ ]:


iris_data.describe()


# #### finding number of instances from each class

# In[ ]:


iris_data.groupby('Species').size()


# In[ ]:


features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']


# In[ ]:


X = iris_data[features].values


# In[ ]:


y= iris_data['Species'].values


# In[ ]:


y


# #### since knn classifier does not accepts string as labels we need to encode them

# In[ ]:


from sklearn.preprocessing import LabelEncoder ##import sklearn LabelEncoder


# In[ ]:


la_en = LabelEncoder()


# In[ ]:


y = la_en.fit_transform(y)  ### encoding string values to numbers


# In[ ]:


print(set(y.tolist()))  ### now we can see that we have only three labels or classes


# In[ ]:


from sklearn.cross_validation import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## Now we have split our data into train and test we need to find a optimal value for k
# ### Let's visualize data

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure()
sns.pairplot(iris_data.drop("Id", axis=1), hue = "Species", size=5, markers=["o", "s", "D"])
plt.show()


# ### Time for prediction

# In[ ]:


### import sklearn libraries


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score


# In[ ]:


knn_classifier = KNeighborsClassifier(n_neighbors=3)


# In[ ]:


knn_classifier.fit(X_train, y_train)


# In[ ]:


y_pred = knn_classifier.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


accuracy = accuracy_score(y_test,y_pred)*100
print('accuracy = ' , accuracy ,'%' )


# In[ ]:




