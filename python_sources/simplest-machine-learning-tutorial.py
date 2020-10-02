#!/usr/bin/env python
# coding: utf-8

# Read Data using a library named pandas

# In[ ]:


import pandas as pd
mydata_path = '../input/simple.csv'
mydata = pd.read_csv(mydata_path)


# Understand It

# In[ ]:


mydata.head()


# Choose feature to use and output to predict

# In[ ]:


#We will use a,b,c as features
features=['a','b','c']
X = mydata[features]
# just have a look and make sure everything is the way you want
# because training may take some time ...
X.head()


# In[ ]:


#to predict y
y = mydata.y


# Import Library and create model, we use a simple algorith called Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
my_model = DecisionTreeRegressor()


# Now Train the model

# In[ ]:


#X contains the fetaures that will be used to predict y
my_model.fit(X, y)


# In[ ]:


X.describe()


# Make some test cases

# In[ ]:


#We create 5 test conditions
case_one =[1,1,1] 
case_two =[1,0,1] 
case_three =[0,0,0] 
case_four =[1,0,0] 
case_five =[0,1,0] 
y_test=[case_one,case_two,case_three,case_four,case_five]


# Now we predict the output of the model for the above five cases.

# In[ ]:


print(my_model.predict(y_test))


# This was easy, for the model ! Could you observe the pattern in the data ? Can you predict it yourself case_six [0,0,1]?
