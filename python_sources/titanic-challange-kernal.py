#!/usr/bin/env python
# coding: utf-8

# <font color="#f4425f"> <h1> Titanic: Machine Learning from Disaster Challenge</h1></font>
# ****
# [Click Here For Homepage Of Challange](https://www.kaggle.com/c/titanic/)
# ****
# 

# # Importing Libraries
# ****
# Let's start by importing all the required libraries
# 

# In[43]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display #Display DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))


# # Creating and analysing Dataset
# ****
# Here we will create two DataFrames 
# - ***df_train***
# - ***df_test***
# 
# After creating the DataFrames,<br>
# Lets start by getting some information about it

# In[44]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


# In[45]:


display(df_train.head())
display(df_test.head())


# In[46]:


df_train.info()


# In[47]:


df_train.describe()


# # Cleaning Dataset
# ****
# As we can see in the **df_train.info()** that column **Cabin** has the lowest values and highest **NaN**.<br>
# So first check the total **NaN** values in column **Cabin** of the dataframe.
# Also we will find the Percentage of NaN values in the entier Column.
# 
# > NOTE: We need to clean both of the dataset **Training and Testing**.

# In[48]:


cabin_nan = df_train["Cabin"].isna().sum()
print("Number of NaN in Cabin : "+str(cabin_nan))
print("Percentage : "+str(cabin_nan/len(df_train["Cabin"])*100)+"%")


# As we can see that **NaN** is *>=70%* of the total values in the column.
# <br>This shows that column **Cabin** is not a good feature for our model. So lets remove it. 
# 

# In[49]:


# Training Data
df_train.drop(columns="Cabin",inplace=True)
df_train.info()


# In[50]:


# Testing Data
df_test.drop(columns="Cabin",inplace=True)
df_test.info()


# As we can see in the info that column **Age** and **Embarked** in **Training Dataset** and **Fare**, **Age** and **Embarked** in **Testing Dataset** also contain some **NaN** values.<br>
# But these **NaN** Values are very less in numbers so instead of removing it we replace it.
# For different columns we are replacing it with different values:
# - Column **Age** with *Median*.
# - Column **Embarked** with *"Item with maximum frequency"*.
# - Column **Fare** with *Mean*.

# In[51]:


df_train.Age.replace(np.nan, df_train.Age.median(), inplace=True)
df_train.info()


# In[52]:


df_test.Age.replace(np.nan, df_test.Age.median(), inplace=True)
df_test.Fare.replace(np.nan, df_test.Fare.mean(), inplace=True)
df_test.info()


# In[53]:


df_train.Embarked.replace(np.nan, df_train.Embarked.value_counts().idxmax(), inplace=True)
df_train.info()


# As we know that Names and Ticket numbers are not very good features for clssification because these are very distinct and it is very hard to find patterns among them, this could result in an very complex and less accurate model.
# <br>Therfore here we are removing the columns **Name** and **Ticket** from the Dataset.

# In[54]:


df_train.drop(columns=["Name", "Ticket"], inplace=True)
df_train.head()


# In[55]:


df_test.drop(columns=["Name", "Ticket"], inplace=True)
df_test.head()


# ## One-Hot Encoding
# ****
# In this section we will convert the following columns into **ONE-HOT ENCODING**:
# - **Pclass**.
# - **Embarked**
# - **Sex**
# <br><br>
# After conversion we merge them into our Dataframe **df_train**.
# <br>In last we will drop the columns which are converted into One-Hot Encoding.

# In[56]:


#Pclass Training set
pclass_one_hot = pd.get_dummies(df_train['Pclass'])
df_train = df_train.join(pclass_one_hot)
df_train.head()


# In[57]:


#Pclass Test set
pclass_one_hot_t = pd.get_dummies(df_test['Pclass'])
df_test = df_test.join(pclass_one_hot)


# In[58]:


#Embarked Training set
embarked_one_hot = pd.get_dummies(df_train.Embarked)
df_train = df_train.join(embarked_one_hot)
df_train.head()


# In[59]:


#Embarked Test set
embarked_one_hot_t = pd.get_dummies(df_test.Embarked)
df_test = df_test.join(embarked_one_hot_t)


# In[60]:


#Sex Training set
sex_one_hot = pd.get_dummies(df_train.Sex)
df_train = df_train.join(sex_one_hot)
df_train.head()


# In[61]:


#Sex Test set
sex_one_hot_t = pd.get_dummies(df_test.Sex)
df_test = df_test.join(sex_one_hot_t)


# In[62]:


# Removing "Pclass","Sex","Embarked" from Training set
df_train.drop(columns=["Pclass","Sex","Embarked"], inplace=True)
df_train.head()


# In[63]:


# Removing "Pclass","Sex","Embarked" from Testing set
df_test.drop(columns=["Pclass","Sex","Embarked"], inplace=True)
df_test.head()


# # Normalization
# ****
# In this section we will normalize the dataset.
# <br><br>As we can see in the dataset that **Age** and **Fare** have high values than other feature, this can *Overfit* our Model. So to come across this we need to normalize these columns.
# <br><br>
# For normalization we will be using the **MinMaxScaler()** function from *sklearn.preprocessing*.

# In[64]:


# Normalization in Training set
x = df_train.values
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
norm_df=pd.DataFrame(x_scaled)
df_train.Age = norm_df[2]
df_train.Fare = norm_df[5]
df_train.head()


# In[65]:


# Normalization in Testing set
x_t = df_test.values
min_max_scaler = MinMaxScaler()
x_scaled_t = min_max_scaler.fit_transform(x_t)
norm_df_t=pd.DataFrame(x_scaled_t)
df_test.Age = norm_df_t[1]
df_test.Fare = norm_df_t[4]
df_test.head()


# # Prepaing and Fitting Data
# ****
# In this sectioin we will follow the steps given below to prepare the data for *Training and Testing*:<br>
# 1. Separate the **Features and Lables** from the dataframe.
# 2. Splitting the data into **training** and **testing** data using the **train_test_split()** method of *sklearn.model_selection*.
# 3. Creating the classifiers and fitting the values. For this we are using two different classifiers : 
#     - **XGB Classifier**
#     - **Random Forest Classifier**
#    <br>after that we select the classifier with higher accuracy for the final prediction.
#    
# 4. After fitting the model we will test our model and check the accuracy.
# 

# In[66]:


y= df_train.Survived.values
X= df_train.drop(columns=["Survived","PassengerId"]).values


# In[67]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 


# In[68]:


clf = xgb.XGBClassifier(max_depth=8, learning_rate=0.2, n_estimators=100)
clf.fit(X_train, y_train)


# In[69]:


print('Accuracy on test set: {:.2f}'.format(accuracy_score(y_test, clf.predict(X_test))))
print('Accuracy on Train set: {:.2f}'.format(accuracy_score(y_train, clf.predict(X_train))))


# In[70]:


clf_rf = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=0)
clf_rf.fit(X_train, y_train)


# In[71]:


print('Accuracy on test set: {:.2f}'.format(accuracy_score(y_test, clf_rf.predict(X_test))))
print('Accuracy on Train set: {:.2f}'.format(accuracy_score(y_train, clf_rf.predict(X_train))))


# # Prediction and Submission
# ****
#    > As we can see that **Random forest Classifier** has the higher accuracy we are using it for final prediction.
# 
# In last we will predict the lables for the **unknown Test set** and Submit it.
# To do so we have to follow the given steps:
# - Store the pridiction in a variable **predicted_lables**.
# - The pridiction will be in the form of a *Numpy array*, therefore we will convert it into a Dataframe with column name "**Survived**".
# - Then we will create a dataframe **df_submission** and slice it for two columns **"PassengerId"** and **"Survived"**
# - In last we will convert our Dataframe into a **CSV** file using **df_submission.to_csv()** method.
# 
# > This **CSV** file generated has to be submitted manually.

# In[72]:


X_pre = df_test.drop(columns="PassengerId").values
predicted_lables = clf_rf.predict(X_pre)
df_temp = pd.DataFrame(predicted_lables)
df_temp.columns = ["Survived"]
df_submission = df_test.join(df_temp)
df_submission = df_submission[["PassengerId", "Survived"]]
df_submission.head()


# In[73]:


df_submission.to_csv("submission.csv", index=False)


# ### Author : Gauransh Kumar
