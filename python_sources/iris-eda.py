#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
iris_data=pd.read_csv("../input/Iris.csv")
iris=iris_data.drop("Id",axis=1)


# In[ ]:


print(iris.head())


# In[ ]:


print(iris.tail())


# In[ ]:


print(iris.columns)


# In[ ]:


print(iris["Species"].value_counts())


# In[ ]:


print(iris["Species"].unique())


# In[ ]:


print("About Data : \n")
print("Iris Dataset has total 150 Observations\n")
print("And 4 features :")
for i in range(4):
    print(iris.columns[i])
print("\nAnd one class label 'Species' which has total 3 classes :\n")
print(iris["Species"].unique())
print("\nEach class contains 50 Observations\n")
print("Objective :")
print("Our objective is to classify iris plants into three species in this classic dataset")


# In[ ]:


print("Descriptive Statistics of Iris_Setosa Flower Data :\n")
print(iris[iris["Species"]=="Iris-setosa"].describe())


# In[ ]:


print("Descriptive Statistics of Iris_Versicolor Flower Data :\n")
print(iris[iris["Species"]=="Iris-versicolor"].describe())


# In[ ]:


print("Descriptive Statistics of Iris_Virginica Flower Data :\n")
print(iris[iris["Species"]=="Iris-virginica"].describe())


# In[ ]:


iris_setosa=iris[iris["Species"]=="Iris-setosa"]
iris_versicolor=iris[iris["Species"]=="Iris-versicolor"]
iris_virginica=iris[iris["Species"]=="Iris-virginica"]


# In[ ]:


# Univariate Analysis of Iris Data


# In[ ]:


#  Iris Setosa Data Analysis


# In[ ]:


# PDF Distribution of Sepal Length of Different Flowers
sns.distplot(iris_setosa["SepalLengthCm"],label="Setosa")
sns.distplot(iris_versicolor["SepalLengthCm"],label="Vesicolor")
sns.distplot(iris_virginica["SepalLengthCm"],label="Virginica")
plt.legend()
plt.show()


# In[ ]:


# CDF Distribution of Sepal Length of Different Flowers
sns.set_style("whitegrid")
x=iris_setosa["SepalLengthCm"].sort_values()
y=iris_versicolor["SepalLengthCm"].sort_values()
z=iris_virginica["SepalLengthCm"].sort_values()
plt.plot(x,np.cumsum(x)/np.sum(x),label="Setosa")
plt.plot(y,np.cumsum(y)/np.sum(y),label="Versicolor")
plt.plot(z,np.cumsum(z)/np.sum(z),label="Virginica")
plt.legend()
plt.show()


# In[ ]:


sns.boxplot(y=iris["SepalLengthCm"],x="Species",data=iris)
plt.show()


# In[ ]:


sns.violinplot(y=iris["SepalLengthCm"],x="Species",data=iris)
plt.show()


# In[ ]:


# PDF Distribution of Sepal Width of Different Flowers
sns.distplot(iris_setosa["SepalWidthCm"],label="Setosa")
sns.distplot(iris_versicolor["SepalWidthCm"],label="Vesicolor")
sns.distplot(iris_virginica["SepalWidthCm"],label="Virginica")
plt.legend()
plt.show()


# In[ ]:


# CDF Distribution of Sepal Width of Different Flowers
sns.set_style("whitegrid")
x=iris_setosa["SepalWidthCm"].sort_values()
y=iris_versicolor["SepalWidthCm"].sort_values()
z=iris_virginica["SepalWidthCm"].sort_values()
plt.plot(x,np.cumsum(x)/np.sum(x),label="Setosa")
plt.plot(y,np.cumsum(y)/np.sum(y),label="Versicolor")
plt.plot(z,np.cumsum(z)/np.sum(z),label="Virginica")
plt.legend()
plt.show()


# In[ ]:


sns.boxplot(y=iris["SepalWidthCm"],x="Species",data=iris)
plt.show()


# In[ ]:


sns.violinplot(y=iris["SepalWidthCm"],x="Species",data=iris)
plt.show()


# In[ ]:


# PDF Distribution of Petal Length of Different Flowers
sns.distplot(iris_setosa["PetalLengthCm"],label="Setosa")
sns.distplot(iris_versicolor["PetalLengthCm"],label="Vesicolor")
sns.distplot(iris_virginica["PetalLengthCm"],label="Virginica")
plt.legend()
plt.show()


# In[ ]:


# CDF Distribution of Petal Length of Different Flowers
sns.set_style("whitegrid")
x=iris_setosa["PetalLengthCm"].sort_values()
y=iris_versicolor["PetalLengthCm"].sort_values()
z=iris_virginica["PetalLengthCm"].sort_values()
plt.plot(x,np.cumsum(x)/np.sum(x),label="Setosa")
plt.plot(y,np.cumsum(y)/np.sum(y),label="Versicolor")
plt.plot(z,np.cumsum(z)/np.sum(z),label="Virginica")
plt.legend()
plt.show()


# In[ ]:


sns.boxplot(y=iris["PetalLengthCm"],x="Species",data=iris)
plt.show()


# In[ ]:


sns.violinplot(y=iris["PetalLengthCm"],x="Species",data=iris)
plt.show()


# In[ ]:


# PDF Distribution of Petal Width of Different Flowers
sns.distplot(iris_setosa["PetalWidthCm"],label="Setosa")
sns.distplot(iris_versicolor["PetalWidthCm"],label="Vesicolor")
sns.distplot(iris_virginica["PetalWidthCm"],label="Virginica")
plt.legend()
plt.show()


# In[ ]:


# CDF Distribution of Petal Width of Different Flowers
sns.set_style("whitegrid")
x=iris_setosa["PetalWidthCm"].sort_values()
y=iris_versicolor["PetalWidthCm"].sort_values()
z=iris_virginica["PetalWidthCm"].sort_values()
plt.plot(x,np.cumsum(x)/np.sum(x),label="Setosa")
plt.plot(y,np.cumsum(y)/np.sum(y),label="Versicolor")
plt.plot(z,np.cumsum(z)/np.sum(z),label="Virginica")
plt.legend()
plt.show()


# In[ ]:


sns.boxplot(y=iris["PetalWidthCm"],x="Species",data=iris)
plt.show()


# In[ ]:


sns.violinplot(y=iris["PetalWidthCm"],x="Species",data=iris)
plt.show()


# In[ ]:


# Bi-variate Analysis


# In[ ]:


sns.pairplot(data=iris,hue="Species")
plt.show()


# In[ ]:


sns.scatterplot(x="SepalLengthCm",y='SepalWidthCm',hue='Species',data=iris)
plt.show()


# In[ ]:


sns.scatterplot(x="SepalLengthCm",y='PetalLengthCm',hue='Species',data=iris)
plt.show()


# In[ ]:


sns.scatterplot(x="SepalLengthCm",y='PetalWidthCm',hue='Species',data=iris)
plt.show()


# In[ ]:


sns.scatterplot(x="SepalWidthCm",y='PetalLengthCm',hue='Species',data=iris)
plt.show()


# In[ ]:


sns.scatterplot(x="SepalWidthCm",y='PetalWidthCm',hue='Species',data=iris)
plt.show()


# In[ ]:


sns.scatterplot(x="PetalLengthCm",y='PetalWidthCm',hue='Species',data=iris)
plt.show()


# In[ ]:


print("Observations :")
print("Iris setosa can be differentiated clearly from Versicolor and Virginica")
print("Petal Length and Petal width are two important features by which we can classify our data")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



