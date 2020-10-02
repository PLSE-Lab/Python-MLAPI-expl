#!/usr/bin/env python
# coding: utf-8

# # KNN classification of Iris dataset

# In[1]:


##import all the libraries and framework
import numpy as np
import pandas as pd


# In[2]:


## loading a dataset

iris_data = pd.read_csv('../input/Iris.csv')


# In[3]:


iris_data.head(5)      ##explore head of data


# In[4]:


iris_data.shape


# #### we can say from above that dataset contains 150 rows and 6 columns
# 
# now describe dataset

# In[5]:


iris_data.describe()


# #### finding number of instances from each class

# In[6]:


iris_data.groupby('Species').size()


# In[7]:


features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']


# In[8]:


X = iris_data[features].values


# In[9]:


y= iris_data['Species'].values


# In[10]:


y


# #### since knn classifier does not accepts string as labels we need to encode them

# In[11]:


from sklearn.preprocessing import LabelEncoder ##import sklearn LabelEncoder


# In[12]:


la_en = LabelEncoder()


# In[13]:


y = la_en.fit_transform(y)  ### encoding string values to numbers


# In[14]:


print(set(y.tolist()))  ### now we can see that we have only three labels or classes


# In[15]:


from sklearn.cross_validation import train_test_split


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## Now we have split our data into train and test we need to find a optimal value for k
# ### Let's visualize data

# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


plt.figure()
sns.pairplot(iris_data.drop("Id", axis=1), hue = "Species", size=5, markers=["o", "s", "D"])
plt.show()


# ### Time for prediction

# In[19]:


### import sklearn libraries


# In[20]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score


# In[21]:


knn_classifier = KNeighborsClassifier(n_neighbors=3)


# In[22]:


knn_classifier.fit(X_train, y_train)


# In[23]:


y_pred = knn_classifier.predict(X_test)


# In[127]:


y_pred


# In[130]:


accuracy = accuracy_score(y_test,y_pred)*100
print('accuracy = ' , accuracy ,'%' )


# In[ ]:




