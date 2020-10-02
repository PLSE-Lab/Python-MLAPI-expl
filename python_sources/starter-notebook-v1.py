#!/usr/bin/env python
# coding: utf-8

# ### 1. Import the Necessary Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Model
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### 2. Load the Datasets

# In[ ]:


# read in all data
train = pd.read_csv('/kaggle/input/dsn-ai-futa-challenge/Train.csv')
test = pd.read_csv('/kaggle/input/dsn-ai-futa-challenge/Test.csv')
submission = pd.read_csv('/kaggle/input/dsn-ai-futa-challenge/Sample_Submission.csv')


# In[ ]:


#print out the shape of both Train and Test data
print('Train shape',train.shape)
print('Test shape',test.shape)


# In[ ]:


#Print the head of train
train.head()


# In[ ]:


#Print the head of test
test.head()


# The goal of this competition is to predict <b>Product_Supermarket_Sales</b>. If you look closely, you will realize that it is not provided in the test data.

# ### 3. Exploratory Data Analysis (EDA)
# 
# Product_Supermarket_Sales is the target variable we are trying to predict, so lets explore it.

# In[ ]:


#Total number of entries that are missing in each column
train.isna().sum()


# In[ ]:


plt.figure(figsize=(10,6)) # Set the size of the plot
sns.heatmap(train.corr(),annot=True) # Correlation Heatmap


# At a glance we can see that the highest correlated feature to Product_Supermaket_Sales is Product_Price followed by Supermaket_Opening_Year. And that makes sense because the more a shop sells  expensive goods the higher their total sales get.
# Another observation is that it seems the year of opening also has some correlation with product sales. Lets plot some one to one plot to see if this is a negative or positive trend .
# 

# In[ ]:


#scatterplot of all features
cat_col = ['Product_Fat_Content','Product_Type','Supermarket_Location_Type','Supermarket_Type']#get categorical features of train data

for columns in cat_col: 
    sns.set()
    cols = ['Product_Identifier', 'Supermarket_Identifier',
           'Product_Fat_Content', 'Product_Shelf_Visibility', 'Product_Type',
           'Product_Price', 'Supermarket_Opening_Year',
           'Supermarket_Location_Type', 'Supermarket_Type',
           'Product_Supermarket_Sales']
    plt.figure()
    sns.pairplot(train[cols], size = 3.0, hue=columns)
    plt.show()


# From the plot above, we can confirm that an increase in price of product really makes Total sales increase. Also there seems to be a very little trend in the Supermarket  opening year and the total sales, otherwise no other feature really correlates with Total sales

# **Dealing with missing values**
# 
# *Taking care of missing entries in the data set*
# There are different strategies to fill missing value in the data.You can check online to know more.<br>
# In this notebook I will use mode and mean strategy to fill both categorical features and numerical features respectively.

# In[ ]:


# the columns contain missing value are 1.Product_Weight(Numerical) 2. Supermarket _Size (Categorical)
train['Product_Weight'].fillna(train['Product_Weight'].mean(),inplace=True)
train['Supermarket _Size'].fillna(train['Supermarket _Size'].mode()[0],inplace=True)

# We will have to use the same strategy for out test data
test['Product_Weight'].fillna(test['Product_Weight'].mean(),inplace=True)
test['Supermarket _Size'].fillna(test['Supermarket _Size'].mode()[0],inplace=True)


# In[ ]:


#Now we have no missing values in both train and test data
train.isna().sum()


# In[ ]:


test.isna().sum()


# ### 4. Features Engineering
# Simple rule: Same transformation or generation must be done to both train and test data

# In[ ]:


# Concatenate train and test sets for easy feature engineering.
# You can as well apply the transformations separately on the train and test data intead of concatenating them.
ntrain = train.shape[0]
ntest = test.shape[0]

#get target variable
y = train['Product_Supermarket_Sales']

all_data = pd.concat((train,test)).reset_index(drop=True)

#drop target variable
all_data.drop(['Product_Supermarket_Sales'], axis=1, inplace=True)

print("Total data size is : {}".format(all_data.shape))


# In[ ]:


# Let's Create the squarred root of Product_Price
all_data['Product_Price_sqrt'] = np.sqrt(all_data['Product_Price'])

#Create some cross features
all_data['cross_Price_weight'] = all_data['Product_Price'] * all_data['Product_Weight']


# **Encoding some categorical features for easy usability by Machine Learning Algorithms**

# In[ ]:


all_data.columns


# In[ ]:


one_hot_cols = ['Supermarket_Type','Supermarket _Size','Product_Type','Supermarket_Location_Type']

label_cols = ['Product_Identifier','Supermarket_Identifier','Product_Fat_Content']


# Applying One hot encoding to one_hot_cols

# In[ ]:


all_data = pd.get_dummies(all_data,prefix_sep="_",columns=one_hot_cols)


# Applying Label encoding to label_cols

# In[ ]:


for col in label_cols:
    all_data[col] = all_data[col].factorize()[0]


# In[ ]:


# We are going to drop Product_Supermarket_Identifier' since it's just an ID and we don't need it.
all_data.drop('Product_Supermarket_Identifier',axis=1,inplace=True)


# Now that we are done with feature engineering, Let's split our data back to train and test

# In[ ]:


#Lets get the new train and test set
train = all_data[:ntrain]
test = all_data[ntrain:]

print('Train size: ' + str(train.shape))
print('Test size: ' + str(test.shape))


# In[ ]:


train.head()


# In[ ]:


test.head()


# You can do more feature engineering to improve your scores. You can also consider scalling too if you are using linear models.

# ### 5. Modelling

# In[ ]:


# Spllitting train data into training and validation set. We are using just 20%(0.2) for validation 
X_train,X_test,y_train,y_test = train_test_split(train,y,test_size=0.2,random_state=42)


# In[ ]:


# Define the model
lr = LinearRegression()


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


y_hat = lr.predict(X_test)


# **Check Validation Score and Training Score**
# 
# We are not expecting large difference in the values.

# In[ ]:


print('Validation scores', np.sqrt(mean_squared_error(y_test, y_hat)))

print('Training scores', np.sqrt(mean_squared_error(y_train, lr.predict(X_train))))


# Seems We got a pretty good model...<br>
# Also we did not overfit because our mean_squared_error is lower on Validation data compare to train data

# Now Let's get our prediction for submission

# In[ ]:


test_pred = lr.predict(test);test_pred


# ### 6. Submission File

# In[ ]:


submission.head()


# In[ ]:


submission['Product_Supermarket_Sales'] = test_pred


# In[ ]:


submission.to_csv('first_submission.csv',index=False)
#If you submit this you should at least find a better position on the LeaderBoard


# **Improvement Tips.....**
# 
# 1. Generate more features
# 2. Use other Cross-Validation Techniques
# 3. Try Tree based models
# 
# if you have any question, kindly drop it in the comment section below.

# **Don't Forget to give this Notebook an upvote if you found its content helpful.**<br>
# ***@Christomesh***
