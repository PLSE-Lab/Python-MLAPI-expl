#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.io as pio
#pio.renderers


# In[ ]:


# load dataset
iris = pd.read_csv('/kaggle/input/iris/Iris.csv')


# ## EDA

# In[ ]:


iris.head()


# In[ ]:


iris.describe()


# In[ ]:


iris.info()


# In[ ]:


iris.drop('Id',inplace=True,axis=1)


# In[ ]:


iris.columns = map(str.lower, iris.columns)
iris.columns


# In[ ]:


# By removing the hue you can plot histograms for univariate distributions
plt.style.use('ggplot')
sns.pairplot(iris,hue='species',palette='colorblind')


# ## 3D Scatter plot

# In[ ]:


from mpl_toolkits.mplot3d.axes3d import Axes3D


# In[ ]:


X= iris[iris['species']=='Iris-setosa']['sepallengthcm']
Y= iris[iris['species']=='Iris-setosa']['sepalwidthcm']
Z= iris[iris['species']=='Iris-setosa']['sepalwidthcm']


X2= iris[iris['species']=='Iris-virginica']['sepallengthcm']
Y2= iris[iris['species']=='Iris-virginica']['sepalwidthcm']
Z2= iris[iris['species']=='Iris-virginica']['sepalwidthcm']

X3= iris[iris['species']=='Iris-versicolor']['sepallengthcm']
Y3= iris[iris['species']=='Iris-versicolor']['sepalwidthcm']
Z3= iris[iris['species']=='Iris-versicolor']['sepalwidthcm']


# In[ ]:


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X, Y, Z,c='b',label='Setosa')
ax.scatter(X2, Y2, Z2,c='r',label='Virginica')
ax.scatter(X3, Y3, Z3,c='g',label='Versicolor')
plt.legend()


# ## Interactive 3D Scatter plot

# In[ ]:


import plotly.express as px
df = px.data.iris()
fig = px.scatter_3d(iris, x='sepallengthcm', y='sepalwidthcm', z='petalwidthcm',
              color='species')
# to see interactive plot, just remove 'renderer="svg"'
fig.show(renderer="kaggle")


# ## Support Vector Machine

# In[ ]:


from sklearn.model_selection import train_test_split
X = iris.drop('species',axis=1)
y= iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)


# ## Evaluate model

# In[ ]:


prediction = model.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(prediction,y_test))
print(classification_report(prediction,y_test))


# ## PCA (dimension reduction)

# In[ ]:


# Scale Data
from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
scale.fit(iris.drop('species',axis=1))
scaled_data = scale.transform(iris.drop('species',axis=1))


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=3)


# In[ ]:


pca.fit(scaled_data)


# In[ ]:


transformed_pca = pca.transform(scaled_data)


# In[ ]:


transformed_pca.shape


# In[ ]:


transformed_iris = pd.DataFrame(transformed_pca,columns=['component1', 'component2', 'component3'])
transformed_iris['species'] = iris['species']
transformed_iris.head()


# In[ ]:


fig = px.scatter_3d(iris, x=transformed_iris['component1'], y=transformed_iris['component2'], 
                    z=transformed_iris['component3'],color='species')
# to see interactive plot, just remove 'renderer="svg"'
fig.show(renderer="kaggle")


# In[ ]:


#Train model after PCA


# In[ ]:


transformed_iris.head()


# In[ ]:


X2 = transformed_iris.drop('species',axis=1)
y2= transformed_iris['species']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3)


# In[ ]:


from sklearn.svm import SVC
model2 = SVC()
model2.fit(X_train2,y_train2)


# ## Evaluate model

# In[ ]:


prediction2 = model2.predict(X_test2)


# In[ ]:


print(confusion_matrix(prediction2,y_test2))
print(classification_report(prediction2,y_test2))

