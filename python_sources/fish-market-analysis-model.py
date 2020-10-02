#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


# In[ ]:





# In[ ]:


dataset = pd.read_csv('../input/fish-market/Fish.csv')


# In[ ]:


dataset


# In[ ]:


dataset.rename(columns={'Length1' :'Body Height', 'Length2' : 'Total Length', 'Length3' : 'Diagonal Length'} , inplace= True)


# In[ ]:


dataset.head(10)


# In[ ]:


dataset.tail(10)


#                                       Investigating the Dataset

#                         Check if the Dataset contains an empty value or missing value and also it datatype

# In[ ]:


for empty in dataset.columns:
    
    print('Am Column {} with {} missing values and {} datatype'.format(empty,dataset[empty].isnull().sum(),
                                                                       dataset[empty].dtype))


# In[ ]:


dataset.describe()


# In[ ]:


dataset.Species.value_counts()


# In[ ]:





# Based on the value_counts above it seems whitefish is least represented here, let visualize it with a histogram

#                                  Histogram for the distribution of the 7-Species

# In[ ]:




plt.hist(dataset.Species.value_counts())


# In[ ]:





# In[ ]:





#                                    Visualiziung Independent variable as a 1D data 

# In[ ]:





# In[ ]:



dataset.hist(bins = 20, grid = False, xlabelsize= 10, ylabelsize= 10, linewidth = 3.0)
plt.tight_layout(rect=(0, 0, 1.5, 1.5))  


#                           Checking the correlation between the variables 

# In[ ]:





# In[ ]:


corr = dataset.corr()
corr


# In[ ]:


f, ax = plt.subplots(figsize=(10,6))
sbn.heatmap(corr, annot=True, ax = ax, linewidths=1.9, fmt='1.2f')
f.subplots_adjust(top = 1)


#                The above heatmap show the how the six variable correlate with the weight of the fishies

#                                      *Diving into Data cleaning and modeling* 

# In[ ]:


# dividing the dataset into X and Y category
features = ['Species','Body Height', 'Total Length', 'Diagonal Length', 'Height', 'Width']
x = dataset[features]
y = dataset['Weight']
#y = dataset.Weight   another way of getting the column value of weight


#   Since we have no missing values, we go straight into checking for linearity of the datasets and encoding the categorical data 

# ### it is adviceable to validate that certain assumptions are met in terms of the dataset, we will pick two independent variables and check there linearity with the dependent variable

# 

# In[ ]:


plt.scatter(x['Height'],y, color = 'red')
plt.title('Height & Weight')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.grid(True)


# In[ ]:





# In[ ]:


plt.scatter(x['Width'],y , color = ['blue'])
plt.title('Width $ Weight')
plt.xlabel('Width')
plt.ylabel('Weight')
plt.grid(True)


# As we can see there exist a linear relationship between these variable in each case.
# 
# 1. in the case of the height, the fishes gain more weight as they grow in height, but the weight gain isnt the same in all species, some has a sharp weight rise as they gain little gain, while some are gradual.
# 
# 2. whiles in the case of the width of the fishes there seems to be a uniform weight gain rate between the species.

# we will be transforming the categorical data with the help of columntransformer

# In[ ]:


encode = preprocessing.LabelEncoder()
x['Species']= encode.fit_transform(x['Species'])

transformer = ColumnTransformer([('transfor', OneHotEncoder(), ['Species'] )],  remainder = 'passthrough')
x=  np.array(transformer.fit_transform(x), dtype = np.float)


# In[ ]:


x.shape


# In[ ]:


x = pd.DataFrame(x)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


# In[ ]:





#          As a precaution we are going to drop one of the dummy variable to avoid the dummy variable trap

# In[ ]:


x = x.drop(columns = [0])


# In[ ]:



x.columns=['Perkki' ,'Perch','Piki' ,'Roach', 'Smelt', 'Whitefish', 'Body Height',  'Total Length',  'Diagonal Length', 
           'Height', 'Width']


# In[ ]:





# In[ ]:


x.shape


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
ran_reg = RandomForestRegressor(random_state= 0)
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
ran_reg.fit(X_train,Y_train)


# In[ ]:





# In[ ]:


y_pred = lin_reg.predict(X_test)
y_ran_pred = ran_reg.predict(X_test)
y_pred_error = mean_absolute_error(Y_test , y_pred)
y_ran_error = mean_absolute_error(Y_test, y_ran_pred)
y_pred_r2_score = r2_score(Y_test, y_pred)
y_ran_r2_score = r2_score(Y_test, y_ran_pred)


# In[ ]:


print('Result of using a Linear Regression Model\n',
      y_pred,'\n..............................................\n')
print('Result of using a Random Forest Regression Model\n',y_ran_pred)


# Based on the above result Did a better job than the Linear regrssion model, there is more room for improvement so i think, i will implement a features elimination to see if things can improve..

# 

# 

# In[ ]:



print(' mean_absolute_error for the linear regression', y_pred_error)
print(' mean_absolute_error for the linear regression', y_ran_error)
print(' R square for the linear regression', y_pred_r2_score)
print(' R square for the linear regression', y_ran_r2_score)


# This is my first commit, just started learning some 4days ago, and this is my first real world data too, all corrections are accepted thanks alot...

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




