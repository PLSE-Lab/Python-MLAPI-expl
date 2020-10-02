#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing the header files

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#reading the dataset

data = pd.read_csv('/kaggle/input/iris/Iris.csv')


# In[ ]:


data.head()


# In[ ]:


#renaming the columns ... just to be comfortable while working (totally optional)
data.columns=['Id','SepalLength(cm)','SepalWidth(cm)','PetalLength(cm)','PetalWidth(cm)','Class']

#dropping the Id column
data.drop('Id',axis=1,inplace=True)


# In[ ]:


data.head()


# # Exploratory Data Analysis

# In[ ]:


#checking the null values

data.isnull().sum()


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


sns.set_style('whitegrid')
sns.pairplot(data,hue='Class')


# One can see from the above drawn plots that PetalWidth(cm) and PetalLength(cm) are strongly correlated. Similarly, SepalLength(cm) and PetalLength(cm) are also correlated.To show this we can separately plot them below.

# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(15,7.5))

sns.scatterplot(x='PetalWidth(cm)',y='PetalLength(cm)',hue='Class',data=data,ax=ax[0])
sns.scatterplot(x='SepalLength(cm)',y='PetalLength(cm)',hue='Class',data=data,ax=ax[1])

fig.show()


# # Defining and fitting the model

# In[ ]:


#scaling the dataset (can be avoided here since the dimensions of the columns are same i.e cm)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data.drop('Class',axis=1))


# In[ ]:


scaled_features=scaler.transform(data.drop('Class',axis=1))


# In[ ]:


df_scaled = pd.DataFrame(scaled_features,columns=data.columns[:-1])
df_scaled.head()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(df_scaled,data['Class'],test_size=0.2,shuffle=True,random_state=101)


# In[ ]:


accuracy_rate = []
for i in range(1,25):
  knn = KNeighborsClassifier(n_neighbors=i)
  score = cross_val_score(knn,df_scaled,data['Class'],cv=10)
  accuracy_rate.append(score.mean())


# In[ ]:


error_rate = []
for i in range(1,25):
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(x_train,y_train)
  pred = knn.predict(x_test)
  error_rate.append(np.mean(pred != y_test))


# In[ ]:


#plotting Error_Rate vs K and Accuracy_Rate vs K

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.plot(range(1,25),error_rate,color='orange',linestyle='solid',marker='o',markerfacecolor='black',markersize=10)
plt.title('Error Rate vs K')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.subplot(1,2,2)
plt.plot(range(1,25),accuracy_rate,color='yellow',linestyle='solid',marker='o',markerfacecolor='green',markersize=10)
plt.title('Accuracy Rate vs K')
plt.xlabel('K')
plt.ylabel('Accuracy Rate')


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train,y_train)


# In[ ]:


prediction = knn.predict(x_test)
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))


# In[ ]:


print(accuracy_score(prediction,y_test))

