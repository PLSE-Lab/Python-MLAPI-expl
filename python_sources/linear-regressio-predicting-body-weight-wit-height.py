#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Python Modules

# In[ ]:


import matplotlib.pyplot as plt


# ### Importing the data

# In[ ]:


df=pd.read_csv('../input/weight-height/weight-height.csv')
df.head()


# ### Shape

# In[ ]:


df.shape


# So we have the height and weight of 10000 individuals

# ### Getting data Information

# In[ ]:


df.info()


# We have one object and two rows with float values in our dataset.

# ### Describing the data

# In[ ]:


df.describe()


# We can see that mean height is 169 cm and weight is 106 pounds 

# ### Getting count of Gender in the Data

# In[ ]:


df['Gender'].value_counts()


# So we have equal distribution of Male and Female in the datset

# ### Scatter Plot with Weight as a function of Height

# In[ ]:


df.plot(kind='scatter',x='Height',y='Weight');


# We can see that there is almost a Linear relation between Height and Weight.As Hieght increase weight increases.This is quite obvious as bone weight would be more in taller people.

# ### Male and Female Separation on Scatter Plot

# In[ ]:


males=df[df['Gender']=='Male']
females=df[df['Gender']=='Female']


# In[ ]:


fig,ax = plt.subplots()
males.plot(kind='scatter',x='Height',y='Weight',
          ax=ax,color='blue',alpha=0.3,
          title='Male and Female Populations')
females.plot(kind='scatter',x='Height',y='Weight',
          ax=ax,color='red',alpha=0.3,
          title='Male and Female Populations');


# So we can see that there is a clear seperation between Male and Female.Women generally have lower height than Men so their weighs alos proportionally lower.

# In[ ]:


df['Genddercolor'] = df['Gender'].map({'Male':'blue','Female':'red'})


# In[ ]:


df.plot(kind='scatter',x='Height',y='Weight',c=df['Genddercolor'],alpha=0.3,title='Male & Female Population');


# So we managed to plot the same graph with different technique using map command.

# In[ ]:


fig,ax = plt.subplots()
ax.plot(males['Height'],males['Weight'],'ob',females['Height'],females['Weight'],'or',alpha=0.3)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Male & Female Populations');


# Here we used simple matplotlib techinique to plot he same graph.

# ### Histogram

# In[ ]:


males['Height'].plot(kind='hist',bins=50,range=(50,80),alpha=0.3,color='blue')
females['Height'].plot(kind='hist',bins=50,range=(50,80),alpha=0.3,color='red')
plt.title('Height distribution')
plt.legend(['Males','Females'])
plt.xlabel('Height in')
plt.axvline(males['Height'].mean(),color='blue',linewidth=2)
plt.axvline(females['Height'].mean(),color='red',linewidth=2);


# As expected men are taller than Females.The red and blue lines how the mean of Female and Male height.

# ### Cumulative Distribution

# In[ ]:


males['Height'].plot(kind='hist',bins=200,range=(50,80),alpha=0.3,color='blue',cumulative=True,normed=True)
females['Height'].plot(kind='hist',bins=200,range=(50,80),alpha=0.3,color='red',cumulative=True,normed=True)

plt.title('Height Distribution')
plt.legend(['Males','Females'])
plt.xlabel('Height (in)')

plt.axhline(0.8)
plt.axhline(0.5)
plt.axhline(0.2);


# ### Box plot

# In[ ]:


dfpvt=df.pivot(columns='Gender',values='Weight')
dfpvt.head(2)


# In[ ]:


dfpvt.plot(kind='box');
plt.title('Weight Box Plot')
plt.ylabel('Weight (lb)')


# So we can see the box plot shws the spread of weight for male and female.

# ### Weight Prediction

# In[ ]:


X=df['Height'].values[:,None]
X.shape


# In[ ]:


y=df.iloc[:,2].values
y.shape


# ### Splitting the test train data

# In[ ]:


from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# ### Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)


# ### Predicting the Weight

# In[ ]:


y_test=lm.predict(X_test)
print(y_test)


# ### Plotting the given data against the predicted data

# In[ ]:


plt.scatter(X,y,color='b')
plt.plot(X_test,y_test,color='black',linewidth=3)
plt.xlabel('Height in inches')
plt.ylabel('Weigth in Pounds')
plt.show()


# The blue dots are the actual weight and the black line indicates the linear model prediction.

# ### Model Performance

# In[ ]:


y_train_pred=lm.predict(X_train).ravel()
y_test_pred=lm.predict(X_test).ravel()


# In[ ]:


from sklearn.metrics import mean_squared_error as mse,r2_score


# In[ ]:


print("The Mean Squared Error on Train set is:\t{:0.1f}".format(mse(y_train,y_train_pred)))
print("The Mean Squared Error on Test set is:\t{:0.1f}".format(mse(y_test,y_test_pred)))


# The mean squared error value for a good model should have low value.

# In[ ]:


print("The R2 score on the Train set is:\t{:0.1f}".format(r2_score(y_train,y_train_pred)))
print("The R2 score on the Test set is:\t{:0.1f}".format(r2_score(y_test,y_test_pred)))


# The R2 Square error for a good model should be close to 1.
