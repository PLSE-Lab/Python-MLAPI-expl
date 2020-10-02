#!/usr/bin/env python
# coding: utf-8

# # In FIFA19, prediction of players overall score
# 
# FIFA is a football based game developed by game giants EA Sports every year. So, it was no surprise that EA Sports launched FIFA 19 earlier this year. FIFA 19 contains a rich dataset with plenty of attributes covering all aspects of a real-life footballer in an attempt to immitate him as much as possible in the virtual world. This rich dataset provides a huge oppurtunity for us, data scientists or data analysts (you choose the term !) to analyze and come up with visualizations and patterns. In this paper, I will try to learn analyzing models performance. In the end, I will end up with a model with tuned alpha parameter of regulizer. 
# 
# The data can be found via the link below:
# https://www.kaggle.com/karangadiya/fifa19
# 

# In[ ]:


#Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
# file path
import os
print(os.listdir("../input"))


# In[ ]:


path='../input/data.csv'
df= pd.read_csv(path)


# In[ ]:


#Let's see the features of the dataset
df.head()
df.columns


# In[ ]:


#since there are a lot of unnecessary features for prediction of 'Overall score' I dropped them out . 
df=df.drop(['Unnamed: 0','ID', 'Position', 'Name', 'Photo', 'Nationality','Potential', 'Flag', 'Club', 'Club Logo', 'Value', 'Wage', 'Special',
       'Preferred Foot', 'International Reputation', 'Weak Foot',
       'Skill Moves', 'Work Rate', 'Body Type', 'Real Face',
       'Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until', 'Release Clause','LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB','Height', 'Weight'] , 1)


# In[ ]:


#To make sure the left features are appropriate
df.columns


# In[ ]:


df.describe()


# In[ ]:


len(df)


# In[ ]:


df=df.dropna()
#There is a lot of data that is missing, therefore we drop those lines of data
df.isnull().sum()


# In[ ]:


len(df)


# In[ ]:


f , axes = plt.subplots(figsize = (10,5))
ax = sns.countplot('Age', data = df)
plt.ylabel('Number of players')


# In[ ]:


f , axes = plt.subplots(figsize = (20,5))
ax = sns.countplot('Overall', data = df)
plt.ylabel('Number of players')
plt.xlabel('Overall Score')


# # Splitting the data
# The main task here to use all features to determine the overall score of the player.

# In[ ]:


#First of all I will split the data into three sets: train, validation and test sets (60:20:20)
train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])


# In[ ]:


train.head()


# In[ ]:


validate.head()


# In[ ]:


test.head()


# In[ ]:


train.describe()


# In[ ]:


#The training values of y should contain only Overall score and x values should be without Overall score.
x_train=np.array(train.drop(['Overall',], axis=1))
y_train=np.array(train['Overall'])

x_validate=np.array(validate.drop(['Overall',], axis=1))
y_validate=np.array(validate['Overall'])

x_test=np.array(test.drop(['Overall',], axis=1))
y_test=np.array(test['Overall'])


# In[ ]:


len(x_train), len(x_validate) , len(x_test)


# # Model assessment:

# In[ ]:


#I created a function that runs a model and gives the MSE for the selected test set and alpha
def model_run(x1, y1, x2, y2, alpha):
    model = linear_model.Ridge(alpha)
    model.fit(x1, y1)
    predictions=model.predict(x2)
    
    differences=[(x-y)**2 for (x,y) in zip(predictions, y2)]
    MSE=sum(differences)/len(differences)
    
    return MSE


# In[ ]:


#This creates a set of values of alpha
list=np.logspace(-3,5,100)


# In[ ]:


#Let's create a loop to run a model for the values of alpha with training data set
alpha=[]
yaxis=[]
for item in list:
    alpha.append((item))
    yaxis.append(model_run(x_train, y_train, x_train, y_train, item))
a=yaxis


# In[ ]:


#Let's create a loop to run a model for the values of alpha with validation data set
alpha=[]
yaxis=[]
for item in list:
    alpha.append((item))
    yaxis.append(model_run(x_train, y_train, x_validate, y_validate, item))
b=yaxis


# In[ ]:


#Let's create a loop to run a model for the values of alpha with testing data set
alpha=[]
yaxis=[]
for item in list:
    alpha.append((item))
    yaxis.append(model_run(x_train, y_train, x_test, y_test, item))
c=yaxis


# In[ ]:


#To illustrate the results of comparison, I will try to plot the values of MSE with respect to values of alpha.
#In addition, the graph shows the minimum results on each validation and testing data.
#The red line shows the optimum value of alpha among both sets. 
plt.plot(alpha, a, label='Training')
plt.xscale('log')
plt.plot(alpha, b, label='Validation')
plt.xscale('log')
plt.plot(alpha, c, label='Testing')
plt.xscale('log')

plt.xlabel('alpha')
plt.ylabel('MSE')
plt.plot(alpha[b.index(min(b))], min(b), 'ro')
plt.plot(alpha[c.index(min(c))], min(c), 'ro')

sum=[]
for i in range(100):
    sum.append(b[i]+c[i])
plt.axvline(x=alpha[sum.index(min(sum))], label='Minimum of sum of training and testing sets', color='r')

plt.legend()


# In[ ]:


# Let's determine the minimum point of sum of validation and testing sets:
best_alpha=alpha[sum.index(min(sum))]
print('The tuned parameter of regulizer is = ' +str(best_alpha))
# For our case the value of alpha is optimum to this model. 


# In[ ]:


#Let's see the MSE and R2 for validation set:
MSE=b[sum.index(min(sum))]
print('MSE = '+ str(MSE))

FVU=MSE/np.var(y_validate)
R2=1-FVU
print('R2 = '+ str(R2))


# In[ ]:


#Let's see the MSE and R2 for test set:
MSE=c[sum.index(min(sum))]
print('MSE = '+ str(MSE))

FVU=MSE/np.var(y_test)
R2=1-FVU
print('R2 = '+ str(R2))


# # Results:
# As a resut, the alpha value for the regulizer is 8902.15 while the MSE and R2 is 6.8 and 0.85. The value of R2 shows that the model at some level performs well on given data with selected features.
