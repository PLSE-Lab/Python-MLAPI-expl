#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/Iris.csv")


# In[ ]:


data.info()


#  Data table shows us that there are 150 entries and 6 columns which four of them are float, one of them integer and one of them object(string) in this data set.

# In[ ]:


data.describe()


# In[ ]:


ax1 = sns.boxplot(x="Species", y="SepalLengthCm", data=data)


# Iris-setosa can be said to have normal distribution in case of SepalLengthCm. On the other hand, Iris-virginica and Iris-versicolor are skewed to the right. Also, Iris virginica has an outlier.

# In[ ]:


ax = sns.boxplot(x="Species", y="SepalWidthCm", data=data)


# Iris-setosa and Iris-virginica can be said to have normal distribution in case of SepalWidthCm. On the other hand, Iris-versicolor is skewed to the left. Also, Iris virginica has outliers.

# In[ ]:


ax = sns.boxplot(x="Species", y="PetalLengthCm", data=data)


# All of them are skewed to the left. Also, Iris-setosa and versicolor have outliers.

# In[ ]:


ax = sns.boxplot(x="Species", y="PetalWidthCm", data=data)


# Iris-setosa has outliers. The other are skewed to the right.

# In[ ]:


plt.scatter(data.SepalLengthCm,data.PetalLengthCm, color='red',alpha=0.5,label='Length')
plt.scatter(data.SepalWidthCm,data.PetalWidthCm, color='blue',alpha=0.5,label='Width')

plt.legend()
plt.xlabel('Sepal')
plt.ylabel('Petal')
plt.title('Scatter Plot')
plt.show()


# In[ ]:


data.corr()


#  It is a correlation matrix which shows the relations between features. The range of this matrix is between -1 and 1. When numbers get closer to -1, it means that there is a negative correlation(while one of them rises, other decreases) between features, on the other hand, when numbers get closer to 1, it means that there is a positive correlation(while one of them rises, other increases) between features. Around 0(zero) means that there is no relation and they dont affect each other.
#   From this table, it is understood that between Id and SepalWidthCm, there is a negative correlation and also between PetalLengthCm and SepalWidthCm, there is a negative correlation. We can say that when SepalWidthCm incereases, PetalLengthCm decreases.
#    Below, we will see the table of correlation matrix.

# In[ ]:



f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(), annot = True, linewidth=0.5, fmt = '.2f', ax=ax)

plt.show()


# In[ ]:


data.columns


# There exist 6 indexes.

# In[ ]:


data.Species.unique()


# In the species detail, we can see that there are 3 objects.

# In[ ]:


data.Species.describe()


# In[ ]:


setosa = data[data.Species == 'Iris-setosa']
versicolor = data[data.Species == 'Iris-versicolor']
virginica = data[data.Species == 'Iris-virginica']


# In[ ]:


setosa.describe()


# In[ ]:


versicolor.describe()


# In[ ]:


virginica.describe()


# In[ ]:


plt.scatter(setosa.PetalLengthCm,setosa.PetalWidthCm, color ="red", label="setosa")
plt.scatter(versicolor.PetalLengthCm,versicolor.PetalWidthCm, color ="blue", label="versicolor")
plt.scatter(virginica.PetalLengthCm,virginica.PetalWidthCm, color ="black", label="virginica")
plt.legend(loc='lower right')
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.title('Scatter Plot')
plt.show()


# Setosa spreads out narrower than the other in case of PetalLengthCm and PetalWidthCm. 

# In[ ]:


plt.scatter(setosa.SepalLengthCm,setosa.SepalWidthCm, color ="red", label="setosa")
plt.scatter(versicolor.SepalLengthCm,versicolor.SepalWidthCm, color ="blue", label="versicolor")
plt.scatter(virginica.SepalLengthCm,virginica.SepalWidthCm, color ="black", label="virginica")
plt.legend(loc='lower right')
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')
plt.title('Scatter Plot')
plt.show()


# Also, with this table, we can say that setosa is more different type than the others because of distribution.

# In[ ]:


plt.hist(setosa.SepalLengthCm, color ="yellow", alpha=0.5, label="setosa")
plt.hist(versicolor.SepalLengthCm, color ="red",alpha=0.5, label="versicolor")
plt.hist(virginica.SepalLengthCm, color ="black",alpha=0.5, label="virginica")
plt.legend(loc='best')
plt.xlabel('SepalLengthCm')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()


# In[ ]:


plt.hist(setosa.PetalLengthCm, color ="yellow", alpha=0.5, label="setosa")
plt.hist(versicolor.PetalLengthCm, color ="red",alpha=0.5, label="versicolor")
plt.hist(virginica.PetalLengthCm, color ="black",alpha=0.5, label="virginica")
plt.legend(loc='best')
plt.xlabel('PetalLengthCm')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

