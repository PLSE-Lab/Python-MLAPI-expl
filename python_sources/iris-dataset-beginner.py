#!/usr/bin/env python
# coding: utf-8

# ![](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png)

#  Hello everyone! This is my first project into the world of Data Scientists, so it's not something too complicated.
# 
# # 1.Prepare the problem
#  **a) Loading the libraries**

# In[ ]:


import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')


# **b) Loading the dataset **

# In[ ]:


iris=pd.read_csv('../input/iris/Iris.csv')


# # 2. Prepare Data 
# **a) Cleaning Data **

# In[ ]:


iris.columns


# In[ ]:


iris.shape


# In[ ]:


iris.info()

# We don't have null values in our data, all seems fine


# I want to rename the columns for an easier work:

# In[ ]:


iris.rename(columns={'SepalLengthCm':'sepal_length','SepalWidthCm':'sepal_width',
                     'PetalLengthCm':'petal_length','PetalWidthCm':'petal_width'},inplace=True)


# In[ ]:


iris.head()


#  # 3.Explore Data
# **  a) Descriptive statistics**

# In[ ]:


iris.describe()  


# In[ ]:


# We can see the descriptive statistics grouped by species
iris.groupby('Species').describe()


# **b) Data visualizations**
# 

# In[ ]:


sns.heatmap(iris.drop('Id',axis=1).corr(),annot=True)


# In[ ]:


sns.pairplot(iris.drop('Id',axis=1),hue='Species',palette='bright')


#  From these graphics we can see that the highest correlation is between petal_length and petal_width, which are POSITIVE correlated, meaning that an increasing in petal's length
# will determine an increasing in petal's width. Also , a high correlation is between petal_length - sepal_length, followed by petal_width - sepal_length. A negative correlation, meaning that  an increasing will determine a decreasing, or inverse, is present between petal_length - sepal_width and petal_width - sepal_width. Also, Iris-virginica has a large petal's width & length, compared with the others 2 which are similar regarding petal's measurements. 
#     

# In[ ]:


# For visualizing the summary statistics for each species, we use boxplot 
species=iris.Species.unique()
k=1
plt.figure(figsize=(21,8))
for i in species:
    plt.subplot(1,3,k)
    k=k+1
    sns.boxplot(data=iris.drop('Id',axis=1)[iris.Species==i],width=0.5,fliersize=5)
    plt.title(str('Boxplot'+i.upper()))


# # Supervised learning 
# ****K-Nearest Neighbors****
# 
# Through this model we want to label a new set of measurements of petals and sepals. Because we want to see if these observations are setosa, virginica or versicolor (3 categories), this model is called *Classification* . Also, it's called *Supervised* learning because we know before the prediction which labels will be applied to the unlabeled data.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X=iris.drop(['Species','Id'],axis=1) # This is also called the independent value (training data)
y=iris.Species # This is also called the dependent value (the target), the value obtained through the independent values


# **Preparing the model and testing the accuracy of it **

# In[ ]:


def prediction(k,train,target):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(train,target) 
    new=pd.DataFrame([[4.5,2.3,3.1,2],[6,2.1,3,4.8]]) # I added 2 random observations 
    new_obsv=knn.predict(new)
    X_train, X_test, y_train, y_test= train_test_split(train,target,test_size=0.3,random_state=21,stratify=target)
    knn.fit(X_train,y_train)
    knn.predict(X_test)
    print('This observations belong to :', new_obsv,'with an accuracy of:',knn.score(X_test,y_test))
    


# In[ ]:


prediction(6,X,y)
prediction(9,X,y)

# The accuracy for these 2 are different, so a plot will be useful for visualizing the influence of the neighbors 


# In[ ]:


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3,random_state=21,stratify=y)
neighbors=np.arange(1,100)
accuracy_list=np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    knn.predict(X_test)
    accuracy_list[i]=knn.score(X_test,y_test)


# In[ ]:


from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure
output_notebook()

p1=figure(plot_height=500,plot_width=900,title='The influence of neighbors numbers on accuracy',
          x_axis_label='Number of neighbors',y_axis_label='Accuracy of KNN model')
p1.line(x=neighbors,y=accuracy_list)
p1.circle(x=neighbors,y=accuracy_list)
show(p1)


# We can observe how the accuracy is dropping when the number of neighbors is increasing

# In[ ]:


neigh = [i for i, x in enumerate(accuracy_list) if x == max(accuracy_list)]
accuracy_list=list(accuracy_list)


# In[ ]:


for i in neigh:
    print('The maxim accuracy is :',max(accuracy_list),' number of neighbours:',i+1)


# So these are the perfect numbers of neighbours for KNN.

# **So, that's it. Feel free to ask questions or correct me if I mistaken anything. I will really apreciate it! Thank you.**
