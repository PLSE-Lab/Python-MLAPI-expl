#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.tree import export_graphviz
from sklearn.metrics import recall_score
import os
print(os.listdir("../input"))
sns.set()


# In[ ]:


df=pd.read_csv("../input/HR_comma_sep.csv")


# In[ ]:


df.head(5)


# In[ ]:


df.info()


# Dataset contains 14999 rows and 10 columns, each row has the details of an employee.  
# 
# 2 variables are categorical, remaining columns are of int and float
# 
# ## Checking for any missing values

# In[ ]:


display(df.isnull().any())


# In[ ]:


df.Department.unique()


# In[ ]:


df.salary.unique()


# In[ ]:




fig,ax = plt.subplots(2,3, figsize=(10,10))               # 'ax' has references to all the four axes
sns.distplot(df['satisfaction_level'], ax = ax[0,0]) 
sns.distplot(df['last_evaluation'], ax = ax[0,1]) 
sns.distplot(df['number_project'], ax = ax[0,2]) 
sns.distplot(df['average_montly_hours'], ax = ax[1,0]) 
sns.distplot(df['time_spend_company'], ax = ax[1,1]) 
sns.distplot(df['promotion_last_5years'], ax = ax[1,2])
 
plt.show()


# ## Employess count 

# In[ ]:


fig = plt.figure(figsize=(15,7))
sns.countplot(x='left',data=df)
plt.show()


# ## Employees in each Department

# In[ ]:


fig = plt.figure(figsize=(15,7))
sns.countplot(x='Department',data=df)
plt.show()


# Sales Department has got more employees, next comes technical and Support departments.
# 
# ## Which Department employess left the company most

# In[ ]:


fig = plt.figure(figsize=(15,7))
sns.barplot(x='Department',y='left',data=df)
plt.show()


# hr Department employees has left the company most, next was accounting, technical, sales and support so on.

# In[ ]:


fig = plt.figure(figsize=(15,7))
sns.boxplot(x='left',y='salary',data=df)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(15,7))
sns.countplot(x='salary',data=df)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(15,7))
sns.boxplot(x="left", y= "satisfaction_level", data=df)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(15,7))
sns.boxplot(x="left", y= "number_project", data=df)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(15,7))
sns.violinplot(x="left", y= "last_evaluation", data=df)
plt.show()


# ## Data Preprocessing
# Convert the salary column to categorical

# In[ ]:


df.salary=df.salary.astype('category')
df.salary=df.salary.cat.reorder_categories(['low', 'medium', 'high'])
df.salary = df.salary.cat.codes


# In[ ]:


# Get dummies and save them inside a new DataFrame
departments = pd.get_dummies(df.Department)
# Take a quick look to the first 5 rows of the new DataFrame called departments
print(departments.head(5))


# In[ ]:


departments = departments.drop("accounting", axis=1)
df = df.drop("Department", axis=1)
df = df.join(departments)
df.head(5)


# ## Percentage of Employee Churn

# In[ ]:


n_employees = len(df)

# Print the number of employees who left/stayed
print(df.left.value_counts())

# Print the percentage of employees who left/stayed
print(df.left.value_counts()/n_employees*100)


# 11,428 employees stayed, which accounts for about 76% of the total employee count. Similarly, 3,571 employees left, which accounts for about 24% of them

# ## Correlation Matrix

# In[ ]:


fig = plt.figure(figsize=(15,7))
cor_mat=df.corr()
sns.heatmap(cor_mat)
plt.show()


# ###  Seperating target and features
# lets seperate the dependent variable(target) and the independent variables(predictors) seperately

# In[ ]:


target=df.left
features=df.drop('left',axis=1)


# ### Splitting the dataset
# will split both target and features into train and test sets with 75%/25% ratio, respectively

# In[ ]:


target_train, target_test, features_train, features_test = train_test_split(target,features,test_size=0.25,random_state=42)


# ## Deceision tree 

# In[ ]:


model = DecisionTreeClassifier(random_state=42)
model.fit(features_train, target_train)
model.score(features_train,target_train)*100


# In[ ]:


#model.fit(features_test,target_test)
model.score(features_test,target_test)*100


# In[ ]:


from sklearn import tree
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
import re
export_graphviz(model,"tree.dot")

check_call(['dot','-Tpng','tree.dot','-o','tree.png'])

# Annotating chart with PIL
img = Image.open("tree.png")
draw = ImageDraw.Draw(img)
img.save('sample-out.png')
PImage("sample-out.png", height=2000, width=1900)


# ### Purning the tree
# As we saw above the accuracy is 100% on training and test set, model is overfitting,
# So fisrt check the option purne the tree, by setting the maximum depth

# In[ ]:


model_depth_5 = DecisionTreeClassifier(max_depth=5, random_state=42)

# Fit the model
model_depth_5.fit(features_train,target_train)

# Print the accuracy of the prediction for the training set
print(model_depth_5.score(features_train,target_train)*100)

# Print the accuracy of the prediction for the test set
print(model_depth_5.score(features_test,target_test)*100)


# Second option to overfitting is limiting the sample size in a leaf(node)

# In[ ]:


model_sample_100 = DecisionTreeClassifier(min_samples_leaf=100, random_state=42)

# Fit the model
model_sample_100.fit(features_train,target_train)

# Print the accuracy of the prediction (in percentage points) for the training set
print(model_sample_100.score(features_train,target_train)*100)

# Print the accuracy of the prediction (in percentage points) for the test set
print(model_sample_100.score(features_test,target_test)*100)


# Evaluating the model
# 

# In[ ]:


#Predict whether employees will churn using the test set
prediction = model.predict(features_test)

# Calculate precision score by comparing target_test with the prediction
precision_score(target_test, prediction)


# In[ ]:


# Calculate recall score by comparing target_test with the prediction
recall_score(target_test, prediction)

