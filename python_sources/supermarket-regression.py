#!/usr/bin/env python
# coding: utf-8

# Start by importing all the libraries i will be using

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Makes graph display in notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# Import my data

# In[5]:


supermarket_data = pd.read_csv('../input/train.csv')


# In[6]:


supermarket_data.head()


# In[7]:


#Displaty the number of rows and columns
supermarket_data.shape


# In[8]:


supermarket_data.describe()


# In[9]:


supermarket_data.dtypes


# In[10]:


#Remove Id columns
cols_2_remove = ['Product_Identifier', 'Supermarket_Identifier', 'Product_Supermarket_Identifier']

new_data = supermarket_data.drop(cols_2_remove, axis=1)


# In[11]:


new_data.head()


# In[12]:


cat_cols = ['Product_Fat_Content','Product_Type',
            'Supermarket _Size', 'Supermarket_Location_Type',
           'Supermarket_Type' ]

num_cols = ['Product_Weight', 'Product_Shelf_Visibility',
            'Product_Price', 'Supermarket_Opening_Year', 'Product_Supermarket_Sales']


# In[13]:


for col in cat_cols:
    print('Value Count for', col)
    print(new_data[col].value_counts())
    print("---------------------------")


# ## DATA VISUALIZATION
# ### BAR PLOT

# In[14]:


counts = new_data['Supermarket_Type'].value_counts() # find the counts for each unique category
counts


# In[15]:


colors = ['green', 'red', 'blue', 'yellow', 'purple']

for i,col in enumerate(cat_cols):
    fig = plt.figure(figsize=(6,6)) # define plot area
    ax = fig.gca() # define axis  
    
    counts = new_data[col].value_counts() # find the counts for each unique category
    counts.plot.bar(ax = ax, color = colors[i]) # Use the plot.bar method on the counts data frame
    ax.set_title('Bar plot for ' + col)


# # Scatter plot for Numerical Features

# In[16]:


new_data.head(3)


# In[17]:


for col in num_cols:
    fig = plt.figure(figsize=(6,6)) # define plot area
    ax = fig.gca() # define axis  

    new_data.plot.scatter(x = col, y = 'Product_Supermarket_Sales', ax = ax)


# In[18]:


for col in cat_cols:
    sns.set_style("whitegrid")
    sns.boxplot(col, 'Product_Supermarket_Sales', data=new_data)
    plt.xlabel(col) # Set text for the x axis
    plt.ylabel('Product Supermarket Sales')# Set text for y axis
    plt.show()
  


# In[ ]:





# ## FEATURE ENGINEERING
# 
# Transform categorical features into numerical features

# In[19]:


#save the target value
y_target = new_data['Product_Supermarket_Sales']
new_data.drop(['Product_Supermarket_Sales'], axis=1, inplace=True)


# In[20]:


new_data.head(2)


# __Option 1: You can use the pandas get_dummies function when working smaller categories__

# In[21]:


# dummy_data = pd.get_dummies(new_data)
# dummy_data.head()


# In[22]:


from sklearn.preprocessing import LabelEncoder


# In[23]:


for cat in cat_cols:
    lb = LabelEncoder()
    lb.fit(list(new_data[cat].values))
    new_data[cat] = lb.transform(list(new_data[cat].values))


# In[24]:


new_data.head()


# ## Fill in Missing Values

# In[25]:


new_data.isnull().sum()


# In[26]:


mean_pw = np.mean(new_data['Product_Weight'])


# In[27]:


new_data['Product_Weight'].fillna(mean_pw, inplace=True)


# In[28]:


new_data.isnull().sum()


# In[29]:


new_data.head()


# ## PERFORM NORMALIZATION AND SCALING

# In[30]:


from sklearn.preprocessing import StandardScaler


# In[31]:


scaler = StandardScaler()
scaler.fit(new_data)

scaled_data = scaler.transform(new_data)


# In[34]:


# Split our data into train and test set
from sklearn.model_selection import train_test_split


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(scaled_data, y_target, test_size = 0.3)


# In[39]:


print("Shape of train data", X_train.shape)
print("Shape of train target ", y_train.shape)
print("Shape of test data", X_test.shape)
print("Shape of test target", y_test.shape)


# In[59]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 
import xgboost as xgb
from sklearn.metrics import mean_absolute_error


# In[55]:


# Using Linear Model
lm = LinearRegression()
lm.fit(X_train, y_train)

#Prediction
predictions_lm = lm.predict(X_test)

#Calculate error
lm_error = mean_absolute_error(y_test, predictions_lm)
print("Mean Absolute Error for Linear model is", lm_error)


# In[58]:


# Using Linear Model
rand_model = RandomForestRegressor(n_estimators=400, max_depth=6)
rand_model.fit(X_train, y_train)

#Prediction
predictions_rf = rand_model.predict(X_test)

#Calculate error
rf_error = mean_absolute_error(y_test, predictions_rf)
print("Mean Absolute Error for Random Forest model is", rf_error)


# In[64]:


# Using ensemble technique
xgb_model = xgb.XGBRegressor(max_depth=4, n_estimators=500, learning_rate=0.1)

xgb_model.fit(X_train, y_train)

#Prediction
predictions_xgb = xgb_model.predict(X_test)

#Calculate error
xgb_error = mean_absolute_error(y_test, predictions_xgb)
print("Mean Absolute Error for XGB model is", xgb_error)


# In[ ]:




