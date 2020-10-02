#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# - Pandas will be used to work with dataframes.
# - NumPy is used to work with arrays
# - Pyplot class from Matplotlib is used for plotting different 
#   types of plots to gain better insights

# # **Data Preprocessing**

# In[ ]:


# importing the 'insurance' dataset
df = pd.read_csv("../input/insurance-premium-prediction/insurance.csv")


# In[ ]:


# printing the shape - rows & columns
df.shape


# In[ ]:


# printing first 5 rows of dataset
df.head()


# In[ ]:


# printing data types of all columns
df.dtypes


# - Dataframe has total 7 columns .
# - 3 of them are categorical - 'sex','smoker','region' .
# - Rest 4 are numerical in nature. 'age' and 'children' have integer values,
#   'bmi' and 'expenses' have floating point values

# In[ ]:


# printing all unique values in categorical variables
a = df['sex'].unique()
b = df['children'].unique()
c = df['smoker'].unique()
d = df['region'].unique()
print(a,'\n',b,'\n',c,'\n',d)


# In[ ]:


# checking for duplicate rows present
df.duplicated().sum()


# In[ ]:


# removing duplicate rows
df = df.drop_duplicates()


# In[ ]:


# checking for null values present
df.isnull().sum()


# - No Null values are present in any column of dataset

# In[ ]:


# printing data types of all columns
df.dtypes


# In[ ]:


df.describe(include='all')


# - Total count decreased to 1337 after dropping a duplicate row

# # **Data Visualization**

# In[ ]:


# plotting frequecy distribution for different variables
hist = df.hist(figsize = (15,15),color='#EC7063')


# - More than 200 people are around 20 years of age.
# - A large no. of people have bmi value between 25-35.
# - Most people have single children only.
# - Expenses value mostly lies below 15000.

# In[ ]:


# frequecny distribution of each value present in 'region' variable
df.region.value_counts().plot(kind="bar",color='#58D68D')


# - Value counts for different regions is almost equal

# In[ ]:


# pie-chart to plot frequency of smokers and non-smokers
df.smoker.value_counts().plot(kind="pie")


# - A very large no. of people are non-smokers

# In[ ]:


# mean expenses for smokers and non-smokers
df.groupby('smoker').expenses.agg(["mean"])


# - Being a smoker costs much more insurance premium . 

# In[ ]:


# mean expenses both male & female
df.groupby('sex').expenses.agg(["mean"])


# - Not much difference between premiums of male and female

# In[ ]:


# find corelation between numerical variables
df.corr()


# - Variables 'age' and 'bmi' have a strong corelation with expenses

# In[ ]:


# bar graphs to show trends in 'expenses' variable w.r.t other variables present
a = ['age','children','bmi','sex','smoker','region']
for i in a:
    x = df[i]
    y = df['expenses']
    plt.bar(x,y,color='#A569BD')
    plt.xlabel(i)
    plt.ylabel('expenses')
    plt.show()


# 
# - Increase in age causes a slight increase in expenses.
# - Expenses decreases as no. of children increases.
# - As BMI value for a person increases, Expenses also get increase.
# - Sex of people doesn't really affect the expenses
# - Smokers pay way more than non-smokers.
# - No observed trend in expenses with region.

# In[ ]:


# splitting dependent variable from independent variables
x = df.drop(columns=['expenses'])
y = df['expenses']
x.head()


# In[ ]:


# One Hot Encoding all the categorical variables 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[( 'encoder', OneHotEncoder() , [1,4,5] )], remainder='passthrough')
x = np.array(ct.fit_transform(x))


# # Splitting the Dataset

# In[ ]:


# splitting the dataset into train & test part (with 25% as test)
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.25 , random_state=1)


# In[ ]:


# Feature Scaling the variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)


# # Training the model

# In[ ]:


# training the model with multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[ ]:


print("Model intercept",regressor.intercept_,"Model co-efficent",regressor.coef_)


# # Making predictions

# In[ ]:


# making predictions on test dataset
y_pred = regressor.predict(x_test)


# # Model Evaluation

# In[ ]:


# printing 'Root Mean Squared Value' for train and test part of the dataset separately
from sklearn import metrics
print("RMSE")
print(np.sqrt(metrics.mean_squared_error(y_test,regressor.predict(x_test))))

