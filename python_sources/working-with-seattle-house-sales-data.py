#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# # Load house sales data

# In[ ]:


sales = pd.read_csv('../input/house_sales.csv')


# # Inspect loaded data

# In[ ]:


sales.head()


# # Draw scatter plot of price vs sqft

# In[ ]:


import matplotlib.pyplot as plt
plt.plot(sales['sqft_living'], sales['price'], 'bo')


# # Split data into training and test set

# In[ ]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(sales, test_size=0.2, random_state=0)


# In[ ]:


train.head()


# # Create a simple regression model of sqft to price

# In[ ]:


from sklearn import linear_model
sqft_regr = linear_model.LinearRegression()
sqft_model = sqft_regr.fit(train.sqft_living.values.reshape(-1, 1), train.price.values.reshape(-1, 1))


# # Investigate model details

# In[ ]:


sqft_model.coef_


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mean_squared_error(test['price'], sqft_model.predict(test['sqft_living'].values.reshape(-1, 1)))


# In[ ]:


r2_score(test['price'], sqft_model.predict(test['sqft_living'].values.reshape(-1, 1)))


# # Plot outputs

# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(train['sqft_living'], train['price'],  color='black')
plt.plot(train['sqft_living'], sqft_model.predict(train['sqft_living'].values.reshape(-1, 1)), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()


# # Explore other  features in the data

# In[ ]:


my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']


# # Build a regression model with these features

# In[ ]:


my_features_regr = linear_model.LinearRegression()
my_features_model = my_features_regr.fit(train[my_features], train.price.values.reshape(-1, 1))


# # Investigate model details

# In[ ]:


my_features_model.coef_


# In[ ]:


mean_squared_error(test['price'], my_features_model.predict(test[my_features]))


# In[ ]:


r2_score(test['price'], my_features_model.predict(test[my_features]))


# # Apply learnt models to predict prices for 3 houses

# In[ ]:


house1 = sales[sales.id.isin(['5309101200'])]


# In[ ]:


house1


# In[ ]:


print (house1['price'])


# In[ ]:


sqft_model.predict(house1.sqft_living.values.reshape(1, -1))


# In[ ]:


my_features_model.predict(house1[my_features])


# # Prediction for 2nd fancier house

# In[ ]:


house2 = sales[sales.id.isin(['1925069082'])]


# In[ ]:


house2


# In[ ]:


house2['price']


# In[ ]:


sqft_model.predict(house2.sqft_living.values.reshape(1, -1))


# In[ ]:


my_features_model.predict(house2[my_features])

