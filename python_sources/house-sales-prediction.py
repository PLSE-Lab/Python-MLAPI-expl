#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv('../input/kc_house_data.csv')


# In[ ]:


data.shape


# In[ ]:


data.describe()


# In[ ]:


data['date'] = pd.to_datetime(data['date'])


# In[ ]:


data.head()


# # Counting houses having particular bedrooms

# In[ ]:


data['bedrooms'].value_counts()


# # Let's plot a bar graph for the same for better understanding

# In[ ]:


data['bedrooms'].value_counts().plot(kind='bar')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Count')


# In[ ]:


# Houses having 3 and 4 bedrooms are sold most


# # Now let us find that which factors are affecting the price of the house.

# # 1. Square Feet

# In[ ]:


plt.scatter(data.sqft_living,data.price)
plt.title("Price vs Square Feet")
plt.xlabel("Square Feet")
plt.ylabel("Price")


# # 2. Location of the house i.e. zipcode

# In[ ]:


plt.scatter(data.zipcode, data.price)
plt.xlabel("Price")
plt.ylabel("Zip")


# # Now let's see if is there any relationship between bedroom and bathroom

# In[ ]:


data.groupby(['bedrooms', 'bathrooms']).size()


# In[ ]:


plt.scatter(data.bedrooms,data.bathrooms)
plt.title("Bedrooms vs Bathrooms")
plt.xlabel("Bedrooms")
plt.ylabel("Bathrooms")


# In[ ]:


#Hence we concluded that bathrooms is diretly proportional to bedrooms and bedrooms to size of the house


# # Now let's create a model for prediction based on linear regression

# # Then we will test it's accuracy

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


reg = LinearRegression()


# In[ ]:


#As we want to predict Price of the house so we will set labels as the Price column


# In[ ]:


labels = data['price']


# In[ ]:


conv_dates = [1 if values == 2014 else 0 for values in data.date]
data['date'] = conv_dates
train1 = data.drop(['id','price'],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(train1, labels,test_size = 0.2,random_state = 2)


# In[ ]:


reg.fit(x_train,y_train)


# # Testing Accuracy

# In[ ]:


reg.score(x_test,y_test)


# # The model that i just developed above has the accuracy of 71%
