#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


car = pd.read_csv("../input/usa-cers-dataset/USA_cars_datasets.csv")


# In[ ]:


car.head()


# # Exploratory Data Analysis
# 
# # Drop : unnamed :0 Column 

# In[ ]:


car.drop("Unnamed: 0",axis=1)


# # Checking if there is any null value available

# In[ ]:


car.isna().sum()


# # Maximum number of car brand available for auction : - Ford : 1200 (approx) in number

# In[ ]:


car["brand"].value_counts().plot()


# # Maximum price of car : model is : Mercedes benz

# In[ ]:


maxprice=car.groupby("brand")
maxprice=maxprice["model","price"].max()
maxprice=maxprice.sort_values(by="price",ascending=False)
maxprice.plot(kind="bar")


# # Max. number of car model present for auction is of : Door ( Model)

# In[ ]:


carmodel = car["model"].value_counts().plot()


# # Max no. of car having vehicle registration number of year : 2019

# In[ ]:


car["year"].value_counts().plot(kind="bar")


# # Car with highest auction price listed is of brand : Nissan   

# In[ ]:


carmodel=car.groupby("year")
carmodel=carmodel["brand","price"].max()
carmodel=carmodel.sort_values(by="year",ascending=False)
carmodel


# # Max. number of car listed with title status of  : Clean vehicle

# In[ ]:


car["title_status"].value_counts().plot(kind="bar")


# # Best mileage car brand is : peterbilt with model: Truck

# In[ ]:


maxmileage=car.groupby("brand")
maxmileage=maxmileage["model","mileage"].max()
maxmileage=maxmileage.sort_values(by="mileage",ascending=False)
maxmileage


# In[ ]:


car["mileage"].mean()


# # Max. listed car color is White

# In[ ]:


car["color"].value_counts().plot()


# # From canada and ontario state, it is dodge brand car listed with max price.
# # From USA and Wyoming state, it is toyota brand car listed with max price.

# In[ ]:


car_state=car.groupby("country")
car_state=car_state["state","brand","year","price"].max()
car_state


# # Max number of car are from USA

# In[ ]:


car["country"].value_counts().plot(kind="bar")


# # Max number of car  are from the state : ontario ( Canada ) and wyoming ( USA) 

# In[ ]:


country=car.groupby("country")
country=country["state"].max()
country


# In[ ]:


car.columns


# In[ ]:


car.head()


# # Droping few variable which has least impact on price

# In[ ]:


car=car.drop(["Unnamed: 0","year","vin","lot","condition"],axis=1)


# In[ ]:


car.columns


# # Seprating target and independent varibale

# In[ ]:


x=car.drop("price",axis=1)
y=car["price"]


# # Applying label encoding on all the independent variable
# 

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le = LabelEncoder()


# In[ ]:


x = x.apply(le.fit_transform)


# In[ ]:


x.head()


# # spliting the data in to test and train

# In[ ]:



from sklearn.model_selection import train_test_split,GridSearchCV
x_train,x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state=42)


# In[ ]:


# Applying Stochastic gradient descent regressor techinique 


# In[ ]:


from sklearn.linear_model import SGDRegressor


# In[ ]:


lin_model=SGDRegressor()


# In[ ]:


param_grid = {
    'alpha': 10.0 ** -np.arange(1, 7),
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'learning_rate': ['constant', 'optimal', 'invscaling'],
}


# In[ ]:


sgd = GridSearchCV(lin_model, param_grid)


# In[ ]:


import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)


# In[ ]:


sgd.fit(x_train, y_train)


# In[ ]:


y_pred=sgd.predict(x_test)


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


r2=-(r2_score(y_test,y_pred))


# # R2 is very low using SGD technique Hence will explore another technique to improve R2

# In[ ]:


r2


# # Now will apply Gradient Boosting Regressor technique

# In[ ]:


from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
model = ensemble.GradientBoostingRegressor()
model.fit(x_train, y_train)


# In[ ]:


y_pred2=model.predict(x_test)


# In[ ]:


r2=r2_score(y_test,y_pred2)


# # R2 has improved to 0.63 by using GradientBoostingRegressor

# In[ ]:


r2


# # will apply random forest regressor technique

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(x_train, y_train)


# In[ ]:


y_pred3=forest_reg.predict(x_test)


# In[ ]:


r2=r2_score(y_test,y_pred3)


# # R2 incresed to 0.64 applying RandomForestRegressor

# In[ ]:


r2


# In[ ]:


# At last will apply simple linear regression technique 


# In[ ]:


#creat linear regression
from sklearn.linear_model import LinearRegression
lin_model=LinearRegression()
lin_model.fit(x_train,y_train)


# In[ ]:


y_pred4=lin_model.predict(x_test)


# In[ ]:


r2=r2_score(y_test,y_pred4)


# In[ ]:



r2


# # Will forecast predicted price of all the car avaiable for sale by using forest regressor technique with highest R2 value that is 0.64

# In[ ]:


y_predection=forest_reg.predict(x)


# In[ ]:


y_predection


# In[ ]:


new_price = pd.DataFrame(y_predection)
new_price


# In[ ]:


car["predict_price"]=new_price


# # now you can check and compare actual and predicted price below

# In[ ]:


car.head()

