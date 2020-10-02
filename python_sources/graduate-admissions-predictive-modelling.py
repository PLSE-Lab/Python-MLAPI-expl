#!/usr/bin/env python
# coding: utf-8

# <h1>Graduate Admissions</h1>
# <b>  We will be doing predictive modelling for Graduate Admissions dataset. Predicting what are the chances of a student of getting   admission into a Graduate school based on the given features in the dataset.</b><br>
# <b>The skills covering are as follows:<b>
#            1. Data Cleaning
#            2. Exploratory Data Analysis
#            3. Data Visualisation
#            4. Machine Learning

# <h3>Importing necessary Python libraries</h3>

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualisation
import matplotlib.pyplot as plt #data visualisation
import warnings  #Filter warnings
warnings.filterwarnings("ignore")

#Reference for dataset files
import os
print(os.listdir("../input/"))


# So we have two file in the input repository which are <strong>Admission_Predict.csv</strong> and <strong>Admission_Predict_Vert1.1.csv</strong>
# 
# <ul><li><strong>Admission_Predict.csv</strong> is the main dataset that we are going to clean, explore and perform some visualisation on, so to prepare it for training our Machine Learning model.</li>
#     <li><strong>Admission_Predict_Vert1.1.csv</strong> is the testing data that we are going to use for testing the model's performance, how good our model generalize.</li></ul>

# In[ ]:


#Reading the dataset, creating a DataFrame from Admission_Predict.csv
df = pd.read_csv('../input/Admission_Predict.csv')
df.head()


# As we can we have various features in our dataset like Serial No., GRE Score, TOEFL Score, University Rating, and more. The last column is our target, and the rest are the features we are going to use for our prediction.<br>
# <b>But do you think a Serial No. can have any effect on our target value ?</b><br>
# There is no significance of this column in our data, so we will clean our data, removing Serial No. column.

# In[ ]:


#Removing Serial No. column from our DataFrame
df.drop(columns=['Serial No.'],inplace=True)
df.head()


# Now we are good to go, let's see the variety of datatypes in our data

# In[ ]:


df.info(),
print("Shape",df.shape) #shape of dataset => (400,8)


# In[ ]:


df.describe() #statistical inferences


# From above information, we can understand that we have no missing values in our data.<br>
# <b>describe()</b> provides statistical information about our data such as mean, standard deviation and more.

# In[ ]:


df.isnull().sum() #To check missing values without using describe()
#columns depict no missing count


# In[ ]:


#Data Visualisation
#Comparing every feature with the target variable
plt.figure(figsize=(17,7))
sns.scatterplot(df['CGPA'],df['Chance of Admit '],hue=df['Chance of Admit '])


# It can be seen in the above visualisation, the more you have the GRE Score the chances are higher of a student of getting into Grad School.

# In[ ]:


plt.figure(figsize=(17,7))
sns.scatterplot(df['GRE Score'],df['Chance of Admit '],hue=df['Chance of Admit '])


# CGPA plays a very important role, there higher chances if the your CGPA is high, the chance of admission is pretty solid.

# In[ ]:


plt.figure(figsize=(17,7))
sns.scatterplot(df['TOEFL Score'],df['Chance of Admit '],hue=df['Chance of Admit '])


# In[ ]:


plt.figure(figsize=(17,7))
sns.barplot(df['University Rating'],df['Chance of Admit '])


# In[ ]:


plt.figure(figsize=(17,7))
sns.barplot(df['SOP'],df['Chance of Admit '])


# In[ ]:


plt.figure(figsize=(17,7))
sns.barplot(df['LOR '],df['Chance of Admit '])


# In[ ]:


#Finding out how features are correlated to each other in our data
corr = df.corr()
sns.heatmap(corr,annot=True,cmap='Blues') #creates a heatmap that tells the correlation among all the features


# Features like CGPA, GRE Score, and TOEFL Score plays a very dominant role in determining chance of  admit.

# In[ ]:


#Comparing these potential features among each others
plt.figure(figsize=(17,7))
sns.scatterplot(df['GRE Score'],df['TOEFL Score'],hue=df['University Rating'],palette=['red','blue','green','yellow','purple'])


# In[ ]:


plt.figure(figsize=(17,7))
sns.scatterplot(df['GRE Score'],df['CGPA'],hue=df['University Rating'],palette=['red','blue','green','yellow','purple'])


# In[ ]:


plt.figure(figsize=(17,7))
sns.scatterplot(df['CGPA'],df['TOEFL Score'],hue=df['University Rating'],palette=['red','blue','green','yellow','purple'])


# <h1>Building Machine Learning Model</h1>

# I am going to prepare data for testing, as I told you that we are going to use <b>Admission_Predict_Ver1.1.csv</b>. But if you will look at the data, it contains the first 400 rows same as we have in our training data, so we will retrieve the last 100 rows from this data.

# In[ ]:


#Reading Test dataset
testdf = pd.read_csv('../input/Admission_Predict_Ver1.1.csv',skiprows=400) #skipping first 400 rows
test_org = testdf.copy() #reserve original test DataFrame
testdf.columns=['Serial No.','GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research','Chance of Admit']
testdf.drop(testdf[['Serial No.','Chance of Admit']],axis=1,inplace=True) #drop Serial No., and Chance of Admit
testdf.head()


# In[ ]:


#Preparing data for our model
x = df.iloc[:,:-1].values
y = df.iloc[:,7].values
testX = testdf.iloc[:,:].values
testY = test_org.iloc[:,8].values


# In[ ]:


#Importing libraries
from sklearn.preprocessing import StandardScaler #Feature Scaling
scaler = StandardScaler()
x = scaler.fit_transform(x)
testX = scaler.fit_transform(testX)


# <h2>Linear Regression</h2>

# In[ ]:


#Importing Linear Regression model from sklearn
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)  #Fitting training data


# In[ ]:


#Prediction on test data
predict = regressor.predict(testX)
regressor.score(x,y)


# In[ ]:


#Checking mean squared error
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
mse = mean_squared_error(predict,testY)
mae = mean_absolute_error(predict,testY)
msle = mean_squared_log_error(predict,testY)
print("mean_squared_error : %f\nmean_absolute_error : %f\nmean_squared_log_error : %f"%(mse,mae,msle))


# The Linear Regression model shows an accuracy of about 80.35% on training data which is quite good. After testing we got cost values:
# <ul><li>mean_squared_error : 0.002002</li>
#     <li>mean_absolute_error : 0.034729</li>
#     <li>mean_squared_log_error : 0.000758</li></ul>

# <h2>Decision Tree</h2>

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
treeRegressor = DecisionTreeRegressor(criterion='mse',random_state=0,max_depth=6)
treeRegressor.fit(x,y)


# In[ ]:


predict = treeRegressor.predict(testX)
treeRegressor.score(x,y)


# In[ ]:


mse = mean_squared_error(predict,testY)
mae = mean_absolute_error(predict,testY)
msle = mean_squared_log_error(predict,testY)
print("mean_squared_error : %f\nmean_absolute_error : %f\nmean_squared_log_error : %f"%(mse,mae,msle))


# The Decision Regression model shows an accuracy of about 88.9% with maximum depth of tree as 6 on training data which is quite good. After testing we got cost values:
# <ul><li>mean_squared_error : 0.004924</li>
#     <li>mean_absolute_error : 0.051753</li>
#     <li>mean_squared_log_error : 0.001858</li></ul>

# <h2>Random Forest</h2>

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=110,criterion='mse',max_depth=6,random_state=0)
rf.fit(x,y)


# In[ ]:


predict = rf.predict(testX)
rf.score(x,y)


# In[ ]:


mse = mean_squared_error(predict,testY)
mae = mean_absolute_error(predict,testY)
msle = mean_squared_log_error(predict,testY)
print("mean_squared_error : %f\nmean_absolute_error : %f\nmean_squared_log_error : %f"%(mse,mae,msle))


# As you can see the Random Forest Regressor outperform the Linear Regression and Decision Tree with better score :
# After testing we got cost values:
# <ul><li>mean_squared_error : 0.002548</li>
#     <li>mean_absolute_error : 0.035929</li>
#     <li>mean_squared_log_error : 0.000975</li></ul>

# In[ ]:




