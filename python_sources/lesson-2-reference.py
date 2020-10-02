#!/usr/bin/env python
# coding: utf-8

# # Applying AdaBoost to Our Heart Disease Dataset

# ### Quick Python tutorial:

# In[1]:


"""
Declare variable
"""
a = 50


"""
printing variables
    - Note if you just enter the variable name
    at the last line of a single cell, it will
    automatically get printed on a Jupyter notebook
"""
print("Hello world!")
print(a)


"""
For Loop
"""
for i in range(5):
    print("hello!")
    
    
"""
import packages from third party source
    - must install previously from shell (not necessary for Kaggle environment)
        - $ pip install numpy on a terminal window if you are using Unix-based operating system (ie Mac, Linux)
"""
# pip install numpy
import numpy



# # AdaBoost and Basic Exploratory Data Analysis

# In[2]:


# Import required packages

# numpy for basic computations
import numpy as np

# Pandas to manage the data. 
# * Most datatsets are in the form of a Pandas dataframe or an SQL table
import pandas as pd

# Helps with visualization
import seaborn as sns

# mAtH iS nEedEd
import math

# Matplotlib and plotly to create graphs
import matplotlib.pyplot as plt
import plotly.tools as tls


# In[3]:


# Get our dataset
df = pd.read_csv('../input/heart.csv')


# In[4]:


# Show some basic information
df.info()


# In[5]:


# Show the top 5 elements of the dataframe
df.head()


# In[6]:


# Show the bottom 5 elements of the dataframe
df.tail()


# In[7]:


# Count how many instances of heart-disease vs none that we have
df.target.value_counts()


# ## What each column name stands for:
# 
# ###### age: age in years
# ###### sex(1 = male; 0 = female)
# ###### cp : chest pain type ( 0-2 states different types of angina and  3 means no chest pain )
# ###### trestbps : resting blood pressure (in mm Hg on admission to the hospital)
# ###### chol : serum cholestoral in mg/dl
# ###### fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# ###### restec : gresting electrocardiographic results
# ###### thalach : maximum heart rate achieved
# ###### exang : exercise induced angina (1 = yes; 0 = no)
# ###### oldpeak : ST depression induced by exercise relative to rest
# ###### slope : the slope of the peak exercise ST segment
# ###### ca : number of major vessels (0-3) colored by flourosopy
# ###### thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# ###### target: 1 or 0 (Heart disease or not)

# In[8]:


# Check null, important since it may cause issues with our ML model
# Perfect, no null data!
df.isnull().sum()


# In[9]:


# Generate a blank figure with MatplotLib
fig = plt.figure(figsize=(15,15))

# Configure the axes
ax = fig.gca()

# Fill the figure with histogram plots for every column in our dataframe
df.hist(ax = ax)

# Show
plt.show()


# In[10]:


# Basic statistical analysis
df.describe()


# ### Let's look at some visualizations with Seaborn!
# Seaborn was built on top of Matplotlib
# Has many improvements over Matplotlib, however Matplotlib and Plotly is still much more popular
# 

# In[11]:


# Visualize with Seaborn
# sns is the name of our import
sns.FacetGrid(df, hue="target", height=5)    .map(plt.scatter, "age", "thalach")    .add_legend()
plt.show()


# In[12]:


# Visualize with Seaborn
# sns is the name of our import
sns.FacetGrid(df, hue="target", height=5)    .map(plt.scatter, "age", "oldpeak")    .add_legend()
plt.show()


# In[13]:


# Visualize with Seaborn
# sns is the name of our import
sns.FacetGrid(df, hue="target", height=3)    .map(plt.scatter, "thal", "ca")    .add_legend()
plt.show()


# In[14]:


# Visualize with Seaborn
# sns is the name of our import
sns.FacetGrid(df, hue="target", height=5)    .map(plt.scatter, "oldpeak", "thalach")    .add_legend()
plt.show()


# In[15]:


# Visualize with Seaborn
# sns is the name of our import
sns.FacetGrid(df, hue="target", height=3)    .map(plt.scatter, "restecg", "ca")    .add_legend()
plt.show()


# In[16]:


# Import all of our machine learning modules

# This import will allow us to very easily split our data into a testing and training set
from sklearn.model_selection import train_test_split

# Import our AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

# Improt confusion_matrix to log the amount of correctly and incorrectly classified entries in depth
from sklearn.metrics import confusion_matrix

# Easy way to get our model's accuracy
from sklearn.metrics import accuracy_score


# In[18]:


# For 'cp' column, it records chest pain type. Number 3 mean no chest pain, number 0-2 means different tyoe of angina. 
# To simplify it, group the number 0-2 together as disease positive, number 3 as disease negative
number=[0,1,2]
for col in df.itertuples():
    if col.cp in number:
        df['cp'].replace(to_replace=col.cp, value=1, inplace=True)


# ## Testing the accuracy when the top 8 features are used for fitting

# In[19]:


df_top8 = df.loc[:,['cp','oldpeak','thal','ca','thalach','age','chol','trestbps','exang']]


# In[20]:


# Get our x and y from our dataframe
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Split data into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(df_top8,y,
                                                 test_size=0.25,
                                                 random_state=0)

# Initialize our AdaBoost classifier
# Last week we passed in a DecisionTreeClassifier() to show that AdaBoost is based off of decision trees
# However, this isn't necessary as by default AdaBoost uses DecisionTreeClassifier() if no arguments are sent in
# Note, you can also pass in different classifiers!
clf = AdaBoostClassifier()

# Fit our classifier on training data
clf.fit(x_train,y_train)

# Output predictions for each of our entries in our testing dataset
prediction = clf.predict(x_test)

# Use the accuracy_score function to get our accuracy
accuracy = accuracy_score(prediction,y_test)

# Confusion matrix is the amount of misclassfied in each group
# The diagonal is the correctly classified elements
# When target = 0, 26 are correctly classified and 6 elements are misclassfied
# When target = 1, 37 are correctly classified and 6 elements are misclassified
cm = confusion_matrix(prediction,y_test)


# prfs = precision_recall_fscore_support(prediction,y_test)
print('AdaBoost Accuracy: ',accuracy)
print('\n')
print('Confusion Matrix: \n',cm)
print('\n')


# 
# ### Accuracy with different test sizes

# In[25]:


testSize = [0.5,0.4,0.3,0.25,0.2,0.15,0.1]

acc = []
for i in testSize:
    x_train,x_test,y_train,y_test = train_test_split(x_std,
                                                     y,
                                                     test_size=i)
    clf = AdaBoostClassifier()
    clf.fit(x_train,y_train)
    prediction=clf.predict(x_test)
    acc.append(accuracy_score(prediction,y_test))

models_dataframe=pd.DataFrame(acc,index=testSize)   
models_dataframe


# ### Larger training sizes create overfitting issues, lower creates underfitting issues, 70-75% is usually a sweet spot!

# In[ ]:





# In[ ]:




