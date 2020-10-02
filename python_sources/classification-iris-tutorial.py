#!/usr/bin/env python
# coding: utf-8

# Our **goal** is to predict **something** from **past data**. Here **something** is "Which class does a particular plant observation belong to?". We can predict **multiple observations** as well. We **train the model using past data** and **test the model on new data**. We can also split available data into train, validation and test data, and then perform training, validating and testing of the model.
# 
# What we are trying to figure out is "**formula for success**". for e.g. we know that a+b=c. Here we know the formula, we have past data and we can achieve 100% accurate result. So what we do is pass past data and algorithms to the process, and in return we get the prediction right. Sometimes we get **errors** in our predictions, so we *keep on correcting till we reach our potential in minimizing error*. Most likely, we will never be 100% correct, but we have to **increase our probability of success by reducing errors**. Each algorithm has its own assumptions, nuances and performance. This is something very complicated for layman or to be precise who doesnt have in depth maths and statistics background. So I will cover that later once you are more comfortable with the basics or fundamentals. We need to learn to crawl first before we can walk.

# This is the first step. This is your **toolbox** for machine learning. Without it, you will be struggling. You don't want to reinvent the wheel. So you import all the packages, modules and classes needed to perform machine learning on your data.

# In[ ]:


import pandas as pd
import sklearn.datasets as skl_datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as sklms
import sklearn.linear_model as skllm
from sklearn.neighbors import KNeighborsClassifier


# Now, you your process is ready to begin. You start with** bringing in your raw material**. Sometimes it will be very dirty and sometimes it will be as **tidy** as like Sklearn datasets. These are already cleaned up and ready to feed to your process. I will talk about how to make data from dirty to tidy in another article. Till then lets KISS (keep it simple)

# In[ ]:


ds_iris = skl_datasets.load_iris()


# Once you have imported all the raw material i.e. your raw data to the procss, you would like to understand the background of it. It will tell you which **features or columns or attributes or independent variables** are present to predict the **target or outcome or dependent variable**. It will also tell you how many **observations or rows or instances** are present in the dataset. Sometimes you will get data that doesnt have a header or top row and sometimes it may have summary columns or rows. You need to ensure that your final output is that is like a table where all columns except last one are features and last column represents target variable, and all rows are observations. *From now onwards, I will only say features and target.*

# In[3]:


print(ds_iris.DESCR)


# Now you can look at your data at a high level. Feature names, Target Name, Features data, Target data

# In[ ]:


X = ds_iris.data
features = ds_iris.feature_names
y = ds_iris.target
target = ds_iris.target_names


# We start with looking at feature data. What is the matrix or table size?  and few initial rows. It is stored in form of Numpy array where 150 rows and 4 columns are present. So it is a 150x4 matrix.

# In[5]:


print(X.shape)
X[:5,:]


# Next we look at is feature names. As these names have text that is little complicated we can simplify them by converting them to simple names. Lets keep them simple. Naming convention we could follow is all lowercase and hyphen between each word

# In[6]:


features


# In[ ]:


features = ['sepal-length','sepal-width','petal-length','petal-width']


# In[8]:


features


# Finally we look at target variable. Its size and top 5 values.

# In[9]:


print(y.shape)
y[:5]


# Now we look at target data. It clearly tells you that you have 3 classes. Each observation belongs to one of three classes available in past target data. All new data will be classified belonging to one of 3 classes. It also tells you about the data type of the target variable.

# In[10]:


target


# So what we are trying to work out is that we look at past data (features and target), train the model, and then use this model to predict on new available data in future

# So we have X, y and features, target name. We will leverage this available to data to build our model and carry out rest of machine learning process to reach our end goal i.e. prediction on new data

# From here on, we explore our data both in numbers and graphs. Both are important to get a feel of your data

# But before we explore, we need to collect all different pieces of information and bundle them up in a dataframe, more so from a convenience perspective. Pandas DataFrame is like your table in a database or a spreadsheet tab or table.

# In[ ]:


# we convert numpy array to pandas dataframe
df_X = pd.DataFrame(X)
# we reshape the target variable to make it same number of rows as in X and 
# 1 column
y = y.reshape(-1,1)
# we convert y also to a dataframe
df_y = pd.DataFrame(y)
# finally we merge X and y together. axis=1 means concatenate column wise
df = pd.concat([df_X,df_y], axis=1)


# In[ ]:


# now we update feature and target names to the merged dataframe
features.append("class")
df.columns = features


# In[13]:


# Finally, we look at the head of our monster :-)
df.head()


# In[14]:


# Looks complete
# Did you notice first column contains integers. This is index of your table
# or dataframe
df.index


# Here we begin exploring our data, both numerically and graphically

# In[15]:


df.describe()
# Looking at data, 
# I remember the song from Michael Jackson "The way you make me feel"
# You get a feeler for your data
# Each plant has a sepal and a petal
# Both have length and width
# We use these characteristics to classify each plant
# count - tells me that there no missing values or null observations in
# each column
# Looking at mean of Sepal length tells you that 
# it is the longest among all features
# You look at spread now and find petal length to have maximum variance
# you also get to know max and min of each feature
# also percentiles for each column


# It is important to classify your data and divide them into multiple target classes

# In[16]:


pd.pivot_table(data=df,index='class', aggfunc=[np.mean, np.median])


# In[17]:


pd.pivot_table(data=df,index='class', aggfunc=[len, np.std])


# In[18]:


# Min, Max, Mean and Standard Deviation gives you a good feel about your data
# This is where ART comes into the picture. You can make your machine learning
# model more beautiful by look at all features and imagining what other
# features can be derived from available ones
df.groupby('class').agg([min, max, np.mean, np.std]).round(2)


# In[19]:


# we also look at data types of all features and target
df.info()
# data types looks ok. in case you dont feel ok, you need to convert them to 
# appropriate types like int, float, String, date time, etc.
# this will be covered in later articles


# Pictures are easy to grasp than just plain numbers. Lets explore plots or graphs now

# In[20]:


# first plot we look at is box plot for all the numerical features
sns.boxplot(df[['sepal-length','sepal-width','petal-length','petal-width']])
plt.show()


# Looks much better, right? Plots are always better than just plain numbers

# In[21]:


# time of finding correlation between features and target, also within features
# why within features is a topic for later articles
sns.pairplot(df, hue='class')
plt.show()


# Just pair plot can help you classify observations into multiple classes. But what you are looking forward to do is to teach computer and not yourself. So back to numbers :-)

# There are many more distribution plot classes available, but for later articles.

# Time for teaching. Lets train our model.

# In[ ]:


# Prepare X and y
array = df.values
X = array[:,0:4]
y = array[:,4]
# Split into training and test dataset. 0.3 means 70% is training observations
validation_size = 0.30
# This is the random seed. So that when you run this code, we are on same 
# wavelength to discuss the model and results
seed = 7
# We are using Kfold cross validation to split the dataset and perform training
# and testing of the model. I will cover this later.
no_of_splits = 10
# This is your algorithm and you instantiate and pass it along with data
model = skllm.LogisticRegression()

# Used for splitting the dataset. Will be expanded in later article
kfold = sklms.KFold(n_splits=no_of_splits, random_state=seed)

# Here goes everything into the frying pan
X_train, X_test, Y_train, Y_test = sklms.train_test_split(
    X, y, test_size=validation_size, random_state=seed)

# and here comes the output or prediction. in this case we measure accuracy
cv_results = sklms.cross_val_score(model, X_train, Y_train,                              cv=kfold, scoring='accuracy')


# In[23]:


msg = "%s: %f (%f)" % ("Logistic regression:", cv_results.mean(), cv_results.std())
print(msg)


# Are we done? Not yet. We want to look at our predictions also.

# In[24]:


# Here we use another algorithm to predict
knn = KNeighborsClassifier()
# we train our model
knn.fit(X_train, Y_train)
# predict on our test observations
predictions = knn.predict(X_test)
# and here are our predictions
print(predictions)


# So next time we can use this model to predict on new data. We just need input features to predict. This is our magic formula for success.
