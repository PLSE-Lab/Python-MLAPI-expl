#!/usr/bin/env python
# coding: utf-8

# # Visualize 64 dimension image using PCA
# Visualization gives more insight into the dataset. But it becomes impossible to visualize the multidimensional data. 
# 
# PCA comes in handy in this situation to visualize data with multi-dimensions. 
# 
# Below I have shown how we can use PCA on a 8*8 image and convert it to 2 dimensions and then visualize it easily.  

# ## Importing Libraries

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# ## Loading digit dataset from sklearn
# Sklearn provides many toy dataset and these dataset can be used directly without the need of downloading from external website. In this kernel dataset used consist of 64 dimention images. It has total 1797 images and these belonging to 10 classes (0 to 9).

# In[ ]:


from sklearn import datasets
digits = datasets.load_digits() 


# ## Data Analysis
# Visualize some data points

# In[ ]:



plt.gray() 
plt.matshow(digits.images[0]) # first digit
plt.show()

plt.matshow(digits.images[1]) #second digit
plt.show()


# .data is used for attributes of dataset i.e. 64 pixel values
# 
# .target is used for digit class

# In[ ]:


x=digits.data 
y=digits.target

print(x.shape)
print(x)
print()
print(y.shape)
print(y)


# In[ ]:


x=pd.DataFrame(x) #converting in to dataframe
y=pd.DataFrame(y,columns=['target']) 


# In[ ]:


#combining x and y
df=pd.DataFrame(x)
df['target']=y
df.head()


# ## Pre-processing
# Preprocessing involves subtracting the mean from each column.

# In[ ]:


x=x.sub(x.mean(axis=0), axis=1)
x=x.values


# ## Using PCA for 2 principal components
# PCA forms the basis of multivariate data analysis based on projection methods. The most important use of PCA is to represent a multivariate data table as smaller set of variables. 
# 
# PCA finds lines, planes and hyper-planes in the K-dimensional space that approximate the data as well as possible in the least squares sense. A line or plane that is the least squares approximation of a set of data points makes the variance of the coordinates on the line or plane as large as possible.
# 
# In below code two principal components are calculated.

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2) #Selecting 2 component PCA
principalComponents = pca.fit_transform(x) #Fit the model with X and apply the dimensionality reduction on X.
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, y], axis = 1) #Concatinating principal components with class(y)


# # Plot even digits of dataset

# In[ ]:


#plot even digits
d1=finalDf

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = [0,2,4,6,8]
colors = ['r', 'g', 'b','y','black']

for target, color in zip(targets,colors):
    indicesToKeep = (d1['target'] == target)
    ax.scatter(d1.loc[indicesToKeep, 'principal component 1']
               , d1.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# Even digits belongd to class 0,2,4,6,8. Class 0,6 and 4 form one cluster while class 2 and 8 form another cluster. 
# 

# # Plot first five digits of dataset

# In[ ]:


#plot first five digits
d1=finalDf.head(5)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = [0,1,2,3,4]
colors = ['r', 'g', 'b','y','black']

for target, color in zip(targets,colors):
    indicesToKeep = (d1['target'] == target)
    ax.scatter(d1.loc[indicesToKeep, 'principal component 1']
               , d1.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.savefig("6a_without_lib")


#  First five digits belong to class 0,1,2,3,4 respectively. All of them are clearly separated from each other

# # Plot last five digits of dataset

# In[ ]:


#plot last five digits
d1=finalDf.tail(5)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = [9,0,8,9,8]
colors = ['r', 'g', 'b','r','b']

for target, color in zip(targets,colors):
    indicesToKeep = (d1['target'] == target)
    ax.scatter(d1.loc[indicesToKeep, 'principal component 1']
               , d1.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# Last five digits belong to class 9,0,8,9 and 8 respectively. Clearly, we can see 3 clusters. Each belonging to class 0,8 and 9.

# # Plot odd digits of dataset

# In[ ]:


#plot odd digits
d1=finalDf

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = [1,3,5,7,9]
colors = ['r', 'g', 'b','y','black']

for target, color in zip(targets,colors):
    indicesToKeep = (d1['target'] == target)
    ax.scatter(d1.loc[indicesToKeep, 'principal component 1']
               , d1.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# Class belonging to 1,3,5,7,9. None of the class is clearly separated. Predicting class will cause confusion.

# # Plot digits belonging to class 1,2,6 and 8.

# In[ ]:


#plot class 1,2,6,8
d1=finalDf

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = [1,2,6,8]
colors = ['r', 'g', 'b','y']

for target, color in zip(targets,colors):
    indicesToKeep = (d1['target'] == target)
    ax.scatter(d1.loc[indicesToKeep, 'principal component 1']
               , d1.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# Plot digit belonging to class 1,2,6 and 8. Class 6 has a considerable amount of separation from others. While class 1,2 and 8 may cause confusion.

# ## Conclusion
# PCA can be used to visulize high dimentional dataset. And relation between data can be found easily.

# Upvote my work if you like it.
