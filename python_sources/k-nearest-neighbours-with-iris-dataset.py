#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **We are going to work with the well-known supervised machine learning algorithm called k-NN or k-Nearest Neighbors. For this exercise, we will use the Iris data set for classification. The attribute Species of the data set will be the variable that we want to predict.**

# # Import all the required libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# # Import Iris dataset

# In[ ]:


df = pd.read_csv('../input/iris/Iris.csv',index_col=0)


# In[ ]:


df.head()


# In[ ]:


df.describe


# In[ ]:


df.shape


# In[ ]:


x = df.drop(['Species'],axis=1)


# In[ ]:


y = df['Species']


# In[ ]:


df.isnull().sum()


# # Some Explenatory Data Analysis with Iris

# In[ ]:


fig = df[df.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
df[df.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
df[df.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# The above graph shows relationship between the sepal length and width. Now we will check relationship between the petal length and width.

# In[ ]:


fig = df[df.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
df[df.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)
df[df.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=df)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=df)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=df)


# As we can see that the Petal Features are giving a better cluster division compared to the Sepal features. This is an indication that the Petals can help in better and accurate Predictions over the Sepal. 

# # Now let us see how are the length and width are distributed

# In[ ]:


df.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()


# In[ ]:


sns.set(style="ticks", color_codes=True)
g = sns.pairplot(df,hue = "Species", size=3, markers=["o", "s", "D"])


# **Split the dataset into training and testing dataset. The testing dataset is generally smaller than training one as it will help in training the model better.**

# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4,random_state = 5)


# # Train the model with K nearest neighbour algorithm

# **To find the best value of K,we apply the loop over range(1,50) and append the accuracy score of each K into the list accu.**

# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accu = []

for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors= i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    w = accuracy_score(y_test,pred_i)
    accu.append(w)


# **Plot the line graph of k = 1 to 50 and accu. to find the K with greatest accuracy**

# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,50),accu,marker='o',markersize=10,markerfacecolor ='red')


# **Accuracy is gratest at k=9 so,predict the Species wih n_neighbours = 9**

# In[ ]:


knn = KNeighborsClassifier(n_neighbors= 9)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)


# In[ ]:


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# **At k = 9 accuracy is 1.00 so,we cannot take k=9 because of overfitting.
# let's take k=3**

# In[ ]:


knn = KNeighborsClassifier(n_neighbors= 3)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# **Calculate the accuracy of our model**

# In[ ]:


accuracy = accuracy_score(y_test,y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 3)) + ' %.')


# **Here we get the best value of k for our Model i.e. k=3.**
