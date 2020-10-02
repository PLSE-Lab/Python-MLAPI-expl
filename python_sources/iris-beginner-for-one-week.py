#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


file='../input/Iris.csv'
iris=pd.read_csv(file)


# In[ ]:


iris.shape


# In[ ]:


iris.info()


# In[ ]:


iris.boxplot(column='SepalLengthCm',by='Species')
plt.show()


# In[ ]:


iris.boxplot(column='SepalWidthCm',by='Species')
plt.show()


# In[ ]:


iris.boxplot(column='PetalLengthCm',by='Species')
plt.show()


# In[ ]:


iris.boxplot(column='PetalWidthCm',by='Species')
plt.show()


# In[ ]:


print(iris.head())


# In[ ]:


irissetosa=iris[iris['Species']=='Iris-setosa']
irisvirginica=iris[iris['Species']=='Iris-virginica']
irisversicolor=iris[iris['Species']=='Iris-versicolor']


# In[ ]:


plt.scatter(irisvirginica['SepalLengthCm'],irisvirginica['SepalWidthCm'])
plt.scatter(irissetosa['SepalLengthCm'],irissetosa['SepalWidthCm'],c='r')
plt.scatter(irisversicolor['SepalLengthCm'],irisversicolor['SepalWidthCm'],c='g')
plt.ylabel("Spepal width Cm")
plt.xlabel("Spepal length Cm")
plt.legend()
plt.show()


# In[ ]:


plt.scatter(irisvirginica['PetalLengthCm'],irisvirginica['PetalWidthCm'])
plt.scatter(irissetosa['PetalLengthCm'],irissetosa['PetalWidthCm'],c='r')
plt.scatter(irisversicolor['PetalLengthCm'],irisversicolor['PetalWidthCm'],c='g')
plt.ylabel("Petal width Cm")
plt.xlabel("Petal length Cm")
plt.legend(['virginica','setosa','versicolor'])
plt.show()

