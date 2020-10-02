#!/usr/bin/env python
# coding: utf-8

# In[157]:


import sqlite3
import pandas as pd 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt 


# In[6]:


cnx = sqlite3.connect('../input/database.sqlite')
data = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)
data.shape


# In[4]:


data.head()


# Let's get a brief overview of what this data contains. 

# In[16]:


data.describe()


# Let's look at all the columns present in the dataset

# In[12]:


data.columns


# ## Data Cleaning 

# In[26]:


# Let's look for null values 
data.isnull().sum()
#data[data.isnull().any(axis = 1)]


# Let's drop the null values 

# In[29]:


data = data.dropna()
data.shape


# <p style="font-family: Arial; font-size:1.75em;color:blue; font-style:bold"><br>
# 
# Declare the Columns You Want to Use as Features
# <br><br></p>
# 

# 

# In this we can make a dictionay of coorelations of all the features on overall ratings. Let's do that. This is being done to select our features. 

# In[82]:


features =list(data.columns[~(data.columns.str.contains('id'))])
features
# We also do not need overall_rating as that is our target and date 
features.remove('date')
features.remove('overall_rating')
features
data[features].head()


# We can see above that there are more non numeric values in this data. Remove those. 

# In[84]:


remove = ['preferred_foot', 'attacking_work_rate', 'defensive_work_rate']
for i in remove:
    features.remove(i)
features


# In[92]:


# Let's loop through all the features 

for feature in features:
    co = data['overall_rating'].corr(data[feature])
    dict[feature] = co
dict


# Let's also plot these values on line plot to get an idea of the dependence of overall rating on r

# ### Correlation Plot

# In[117]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
x_values = []
y_values = []
for value in dict:
    x_values.append(value)
    y_values.append(dict[value])

# Plotting the values using matplotlib.pyplot
plt.xlabel('Features in the data')
plt.ylabel('Correlation Coefficient with Overall Rating')
plt.title('Correlation of Overall Rating with different features')
plt.yticks([0, 1])

#Adjusting the size of the image 
from matplotlib.pyplot import figure
figure(num = None, figsize = (30, 6), dpi=80, facecolor='w', edgecolor='k')

plt.plot(x_values, y_values)
plt.show()


# In[129]:


get_ipython().run_line_magic('matplotlib', 'inline')

#Plotting a subplot
fig, axis = plt.subplots(figsize = (40, 8))
# Grid lines, Xticks, Xlabel, Ylabel

axis.yaxis.grid(True)
axis.set_title('Overall Player Rating',fontsize=10)
axis.set_xlabel('Player Features',fontsize=10)   
axis.set_ylabel('Correlation Values',fontsize=10)
axis.set_yticks([0,1])
axis.set_yticklabels(['0', '1'])

# # We can also use this to set figure size 
# f.set_figheight(15)
# f.set_figwidth(15)

axis.plot(x_values, y_values)
plt.show()


# <p style="font-family: Arial; font-size:1.75em;color:blue; font-style:bold"><br>
# 
# Specify the Prediction Target
# <br><br></p>
# 

# In[131]:


# Let's also specifiy the target 
target = ['overall_rating']


# In[132]:


# Obtain the X and y values for regression analysis
X = data[features]
y = data[target]


# In[139]:


# Let us look at a typical row from our features:
X.head()
# X.iloc[2]


# Let us also display our target values: 

# In[146]:


y.head() 


# <p style="font-family: Arial; font-size:1.75em;color:blue; font-style:bold"><br>
# 
# Split the Dataset into Training and Test Datasets
# <br><br></p>
# 

# In[173]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 324)


# Feel free to go through the dataset and explore it as much as you can 

# In[185]:


X_train.describe()


# In[186]:


X_test.describe()


# <p style="font-family: Arial; font-size:1.75em;color:blue; font-style:bold"><br>
# 
# (1) Linear Regression: Fit a model to the training set
# <br><br></p>
# 
# 

# In[187]:


regressor = LinearRegression()
regressor.fit(X_train, y_train)


# <p style="font-family: Arial; font-size:1.75em;color:blue; font-style:bold"><br>
# 
# Perform Prediction using Linear Regression Model
# <br><br></p>
# 

# In[188]:


y_prediction = regressor.predict(X_test)
y_prediction


# In[189]:


#Let's explore the predictions. 
y_prediction.mean()


# <p style="font-family: Arial; font-size:1.75em;color:blue; font-style:bold"><br>
# 
# What is the mean of the expected target value in test set ?
# <br><br></p>

# In[190]:


y_test.describe().transpose()


# <p style="font-family: Arial; font-size:1.75em;color:blue; font-style:bold"><br>
# 
# Evaluate Linear Regression Accuracy using Root Mean Square Error
# 
# <br><br></p>
# 

# In[191]:


RMSE = sqrt(mean_squared_error(y_true = y_test, y_pred = y_prediction))


# In[192]:


print(RMSE)


# That is a pretty low error value and that is very good. Let's also find the accuracy and r2 score. 

# In[193]:


from sklearn.metrics import r2_score, accuracy_score


# In[194]:


print(r2_score(y_test, y_prediction))


# In[195]:


#print(accuracy_score(y_test, y_prediction))


# ## Error
# We get an error for the above value. To find out why, [click here](http://https://www.kaggle.com/questions-and-answers/92771****)

# <p style="font-family: Arial; font-size:1.75em;color:blue; font-style:bold"><br>
# 
# (2) Decision Tree Regressor: Fit a new regression model to the training set
# <br><br></p>
# 

# In[199]:


decision_regressor = DecisionTreeRegressor(max_depth = 50)
decision_regressor.fit(X_train, y_train)


# <p style="font-family: Arial; font-size:1.75em;color:blue; font-style:bold"><br>
# 
# Perform Prediction using Decision Tree Regressor
# <br><br></p>

# In[200]:


y_prediction = decision_regressor.predict(X_test)
y_prediction.mean()


# Let's find out the error value

# In[201]:


RMSE = sqrt(mean_squared_error(y_test, y_prediction))
print(RMSE)


# We got a surprisingly low error value for this data set compared to Linear Regressor even though the mean of linear regressor was closer to the actual one. We can reduce the error value by increasing the depth of the Decision Tree. We can also increase the accuracy by increasing the size of the training set ( set testsize = 0.1 )

# In[ ]:




