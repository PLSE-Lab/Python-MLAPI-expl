#!/usr/bin/env python
# coding: utf-8

# # Impute Missing Values using mean, mode, median and KNN

# In[ ]:


import pandas as pd
import numpy as np
cars = pd.read_csv('../input/imports-85.data.txt') # imported the dataset
cars.head() # shows the first five rows which helps to undertsand how data is presented


# In[ ]:


columnnames=['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 
           'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio',
            'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
# we have updated the column names from the link given by you and it also given us ? indicates missing value

cars = pd.read_csv('../input/imports-85.data.txt',names=columnnames)

cars=cars.replace("?",np.nan)
cars.isnull().sum()

# The below are the missing values in the data set


# In[ ]:


cars["num-of-doors"].value_counts() # just wanted to check how many times four accounts and two accounts


# In[ ]:


convert = {"num-of-doors": {"four": 4, "two": 2}}
cars.replace(convert, inplace=True) # here we are converting the categorical one to numerical
cars.head()


# In[ ]:


to_drop = ["symboling", "make", "fuel-type", "aspiration", "body-style", "drive-wheels", "engine-location", "engine-type", "num-of-cylinders", "fuel-system", "engine-size"]

cars = cars.drop(to_drop, axis=1)

# here we can drop columns which are not continous for effective visulization using BAR chart in below


# In[ ]:


cars = cars.astype("float")
cars.isnull().sum()


# In[ ]:


cars.describe()


# In[ ]:


import missingno as msno
import matplotlib.pyplot as plt
msno.bar(cars, figsize=(10, 5), fontsize=10, color='cyan')


# The above results shows the missing values.
# 
# __Dropping the entire attributes or corresponding rows depends on the what is our goal.__
# As the data is continouse, we can use mean,median, mode, regression,multiple imputation. 
# 
# __Normalized losses:__ Total 41 missing values. we cant ignore entire tuples which will lead to deletion 41 rows. This will lead almost deletion 20% of entire data.I beleive K Nearest neighbour will suits best if incase the goal is to predict normalized losses.Therefore imputation is always better than dropping the rows.
# 
# __no of doors:__ Here the data given here is categorical, but we can transform to numerical. or we replace the catgorical data based on the mode.
# 
# __Bore, stroke,horse power, peak rpm, and price:__ we can transform enire numeric data with mean or using K nearest neighbour. However as peak rpm and horsepower have only 2 missing values in each, we can take minute risk of removing entire tuble which wont be having major effect on the model.
# 
# __Imputing__
# 
# __No.of doors:__ We can delete the enire rows where missing value occurs or best way is to replace the data with mode. You can see the working above i.e. we converted the two, four to 2, 4. (categorical to numerical for better view and for better access)
# ((please find the code above cells) 
# 
# __Bore and Stroke__ Both the attributes consists of 4 missing values in each and mean or median imputations works better as they are more distributed equally (Approxmate analysis). (please find the code below cells) 
# 
# __Horse power and peak rpm:__ Both the attributes consists of only 2 missing values in each and we can drop the tuples. however it is not the better choice, as descriptive stats seems ok, we can perform mean for imputing values ( dont see outliers, so no need choose median)(please find the code below cells) 
# 
# __Price:__ from the descriptive stats we realised that more data points are aligned towards the prices above 75%, Mean and mediam imputations are not good for price attribute. In addition their is a lot of variation between min and max price. I believe KNN or regression is best suitable to impute missing values for price attribute. we will use KNN to impute values.      (please find the code below cells) 

# In[ ]:


cars["bore"]=cars["bore"].fillna(cars["bore"].mean())
cars["stroke"]=cars["stroke"].fillna(cars["stroke"].mean())
cars["num-of-doors"]=cars["num-of-doors"].fillna(cars["num-of-doors"].median())
cars["horsepower"]=cars["horsepower"].fillna(cars["horsepower"].mean())
cars["peak-rpm"]=cars["peak-rpm"].fillna(cars["peak-rpm"].mean())
cars.isnull().sum()


# In[ ]:


from fancyimpute import KNN    
X_filled_knn = KNN(k=3).complete(cars[['horsepower', 'peak-rpm', 'price']])


# In[ ]:


X_filled_knn = KNN(k=3).complete(cars[['horsepower', 'peak-rpm', 'price']])


# In[ ]:


X_filled_knn = pd.DataFrame(X_filled_knn, columns = ['horsepower', 'peak-rpm', 'price'])


# In[ ]:


cars['price'] = np.round(X_filled_knn['price'], 0)
cars.isnull().sum()

