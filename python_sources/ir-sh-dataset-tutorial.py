#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns

print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv("../input/Iris.csv")


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.columns


# In[ ]:


clear_data = data.drop(["Id"],axis=1)
print(clear_data.columns)


# In[ ]:


data.Species.unique()


# In[ ]:


setosa = data[data.Species == "Iris-setosa"]
versicolor = data[data.Species == "Iris-versicolor"]
virginica = data[data.Species == "Iris-virginica"]


# In[ ]:


data.corr()


# In[ ]:


f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(),annot= True,linewidths= 0.5,fmt= "0.1f",ax=ax)


# In[ ]:


plt.plot(setosa.Id, setosa.SepalWidthCm, color="cyan", label ="setosa",linestyle="-")
plt.plot(versicolor.Id, versicolor.SepalWidthCm, color="orange", label ="versicolor")
plt.plot(virginica.Id, virginica.SepalWidthCm, color="pink", label ="virginica")
plt.xlabel("Id")
plt.ylabel("SepalWidthCm")
plt.title("SepalWidthCm")
plt.legend()
plt.show()

plt.plot(setosa.Id, setosa.SepalLengthCm, color="cyan", label ="setosa",linestyle="-")
plt.plot(versicolor.Id, versicolor.SepalLengthCm, color="orange", label ="versicolor")
plt.plot(virginica.Id, virginica.SepalLengthCm, color="pink", label ="virginica")
plt.xlabel("Id")
plt.ylabel("SepalLengthCm")
plt.title("SepalLengthCm")
plt.legend()
plt.show()

plt.plot(setosa.Id, setosa.PetalLengthCm, color="cyan", label ="setosa",linestyle="-")
plt.plot(versicolor.Id, versicolor.PetalLengthCm, color="orange", label ="versicolor")
plt.plot(virginica.Id, virginica.PetalLengthCm, color="pink", label ="virginica")
plt.xlabel("Id")
plt.ylabel("PetalLengthCm")
plt.title("PetalLengthCm")
plt.legend()
plt.show()

plt.plot(setosa.Id, setosa.PetalWidthCm, color="cyan", label ="setosa",linestyle="-")
plt.plot(versicolor.Id, versicolor.PetalWidthCm, color="orange", label ="versicolor")
plt.plot(virginica.Id, virginica.PetalWidthCm, color="pink", label ="virginica")
plt.xlabel("Id")
plt.ylabel("PetalWidthCm")
plt.title("PetalWidthCm")
plt.legend()
plt.show()


# In[ ]:


plt.scatter(setosa.PetalLengthCm,setosa.PetalWidthCm, color="red",label = "setosa")
plt.scatter(versicolor.PetalLengthCm,versicolor.PetalWidthCm, color="green",label = "versicolor")
plt.scatter(virginica.PetalLengthCm,virginica.PetalWidthCm, color="blue",label = "virginica")
plt.legend
plt.xlabel("PetalLengthCm")
plt.ylabel("PetalWidthCm")
plt.title("Petal")
plt.show()

plt.scatter(setosa.SepalWidthCm,setosa.SepalLengthCm, color="red",label = "setosa")
plt.scatter(versicolor.SepalWidthCm,versicolor.SepalLengthCm, color="green",label = "versicolor")
plt.scatter(virginica.SepalWidthCm,virginica.SepalLengthCm, color="blue",label = "virginica")
plt.legend
plt.xlabel("SepalWidthCm")
plt.ylabel("SepalLengthCm")
plt.title("Sepal")
plt.show()


# In[ ]:


plt.hist(setosa.PetalLengthCm, bins=15) # bins are number of bars
plt.xlabel("PetalLengthCm")
plt.ylabel("Frequency of PetalLengthCm ")
plt.title("Histogram 1")
plt.show()

plt.hist(versicolor.SepalWidthCm, bins=40)
plt.xlabel("SepalWidthCm")
plt.ylabel("Frequency of SepalWidthCm ")
plt.title("Histogram 2")
plt.show()

plt.hist(virginica.SepalLengthCm, bins=100)
plt.xlabel("SepalLengthCm")
plt.ylabel("Frequency of SepalLengthCm ")
plt.title("Histogram 3")
plt.show()


# In[ ]:


clear_data.plot(grid = True, alpha = 0.9, subplots = True)
plt.show()


# In[ ]:


plt.subplot(2,1,1)
plt.plot(setosa.Id, setosa.PetalLengthCm, color="red", label ="setosa")
plt.ylabel("setosa - PetalLengthCm")

plt.subplot(2,1,2)
plt.plot(versicolor.Id, versicolor.PetalLengthCm, color="green", label ="versicolor")
plt.ylabel("versicolor - PetalLengthCm")
plt.show()


# In[ ]:




