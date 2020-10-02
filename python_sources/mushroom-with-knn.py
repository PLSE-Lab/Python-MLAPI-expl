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
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/mushrooms.csv")


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


# missing data
data.isnull().sum()


# In[ ]:


sns.heatmap(data.isnull())
plt.title("Missing Data",fontsize=(13),color="blue")
plt.show()


# In[ ]:


data["class"].value_counts()


# In[ ]:


sns.countplot(data["class"])
plt.title("CLASS",color = "red")
plt.show()


# In[ ]:


# data summary 
data.describe()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
data = data.apply(lb.fit_transform)


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


sns.heatmap(data.describe()[1:].transpose(),annot=True,linewidths=.5,linecolor="white",cmap="coolwarm")
plt.title("Data Summary",fontsize=(15),color="red")
plt.show()


# In[ ]:


p =len(data[data['class'] == 1])
e = len(data[data['class']== 0])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'p','e'
sizes = [p,e]
colors = ['skyblue', 'yellowgreen']
explode = (0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# In[ ]:


data.habitat.unique()


# In[ ]:


sns.countplot(data.habitat)
plt.show()


# In[ ]:


plt.figure(figsize=(8,6))

# Data to plot
labels = 'Habitat Type:0','Habitat Type:1','Habitat Type:2','Habitat Type:3','Habitat Type:4','Habitat Type:5',"Habitat Type:6"
sizes = [len(data[data['habitat'] == 0]),len(data[data['habitat'] == 1]),
         len(data[data['habitat'] == 2]),
         len(data[data['habitat'] == 3]),
         len(data[data["habitat"] == 4]),
         len(data[data["habitat"] == 5]),
         len(data[data["habitat"] == 6])]
         
colors = ['skyblue', 'yellowgreen','orange','gold',"lightcoral","lightskyblue","gold"]
explode = (0,0,0,0,0,0,0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=180)
 
plt.axis('equal')
plt.show()


# In[ ]:


# countplot with cap-surface
sns.countplot(data["cap-surface"])
plt.show()


# In[ ]:


plt.figure(figsize=(8,5))

labels="Cap-Surface Type:0","Cap-Surface Type:1","Cap-Surface Type:2","Cap-Surface Type:3"
sizes =[len(data[data["cap-surface"]  == 0]),
        len(data[data["cap-surface"] == 1]),
        len(data[data["cap-surface"] == 2]),
        len(data[data["cap-surface"] == 3])]
colors = ['skyblue', 'yellowgreen','orange','gold',"lightcoral","lightskyblue","gold"]
explode = (0,0,0,0)

#plot
plt.pie(sizes,explode = explode , labels = labels ,colors = colors,autopct="%1.1f%%",
        shadow=True,startangle=180)
plt.axis("equal")
plt.show()


# In[ ]:


sns.countplot(data["stalk-root"])
plt.show()


# In[ ]:


plt.figure(figsize=(10,8))

labels="Stalk_Root Type:0","Stalk_Root Type:1","Stalk_Root Type:2","Stalk_Root Type:3","Stalk_Root Type:4"
sizes = [len(data[data["stalk-root"] == 0]),
         len(data[data["stalk-root"] == 1]),
         len(data[data["stalk-root"] == 2]),
         len(data[data["stalk-root"] == 3]),
         len(data[data["stalk-root"] == 4])]

colors = ['skyblue', 'yellowgreen','orange','gold']
explode = (0,0,0,0,0)

plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct="%1.1f%%",
        shadow=True,startangle=180)
plt.axis("equal")
plt.show()


# In[ ]:


#displot for cap-shape
sns.distplot(data["cap-shape"],kde=False,bins=30,color="violet")
plt.show()


# In[ ]:


#displot for cap-surface
sns.distplot(data["cap-surface"],kde=False,bins=30,color="violet")
plt.show()


# In[ ]:


#displot for odor 
sns.distplot(data["odor"],kde=False,bins=30,color="violet")
plt.show()


# In[ ]:


sns.distplot(data["cap-color"],kde=False,bins=30,color="violet")
plt.show()


# In[ ]:


sns.kdeplot(data["class"],shade=True, color="r")
plt.show()


# In[ ]:


plt.figure(figsize=(12,12))
sns.countplot(x="cap-surface",data = data ,hue = "class", palette="GnBu")
plt.show()


# In[ ]:


x = data.drop(["class"],axis=1)
y = data["class"].values


# In[ ]:


# train test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


# In[ ]:


from sklearn.preprocessing import StandardScaler
st = StandardScaler()
x_train = st.fit_transform(x_train)
x_test = st.fit_transform(x_test)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)


# In[ ]:


from sklearn.model_selection import cross_val_score
accuries = cross_val_score(estimator=knn,X=x_train, y = y_train,cv=10)
print("avarge accuracy :" ,np.mean(accuries))
print("avarge std :",np.std(accuries))


# In[ ]:


knn.fit(x_train,y_train)
print("knn score :",knn.score(x_test,y_test))


# In[ ]:


predictions = knn.predict(x_test)


# In[ ]:


plt.scatter(y_test, predictions)
plt.xlabel("Values",color = "red")
plt.ylabel("Pred",color = "blue")
plt.show()

