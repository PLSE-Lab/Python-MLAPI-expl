#!/usr/bin/env python
# coding: utf-8

# Import libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


automobile = pd.read_csv('/kaggle/input/automobile-dataset/Automobile_data.csv')
automobile.head(2)


# In[ ]:


automobile.dtypes


# In[ ]:


automobile.describe()


# In[ ]:


automobile.info()


# In[ ]:


automobile.isnull().any().sum()


# In[ ]:


automobile.columns


# In[ ]:


obj_col = automobile.select_dtypes('object').columns
obj_col


# Dataset does not contain any null value but it contains '?' values.  
# So, 1st to find the columns containing '?'

# In[ ]:


col_list = []
for col in automobile.columns:
    i = automobile[col][automobile[col] == '?'].count()
    if i > 0:
        col_list.append(col)
        print(col, i)
col_list


# col_list contains all the columns that contain '?' values.   
# Now It has to be replaced that values with the mean of their respective columns

# In[ ]:


automobile[col_list].dtypes


# In[ ]:


automobile[col_list].head(4)


# All of these columns contains numeric values except 'num-of-doors'.  
# so convert them into numeric 

# In[ ]:


null_list = ['normalized-losses','bore','stroke', 'horsepower','peak-rpm','price']
for col in null_list:
    automobile[col] = pd.to_numeric(automobile[col], errors = 'coerce')


# In[ ]:


automobile.isnull().sum().sort_values(ascending = False)


# filling all the null values in the columns of null-list with their mean

# In[ ]:


for col in null_list:
    automobile[col] = automobile[col].fillna(automobile[col].mean())


# In[ ]:


automobile.isnull().any().sum()


# Now only column 'num-of-doors' contain the '?' symbol.  
# there are two possible ways:
# * to remove the two rows 
# * to fill them with the mode value

# In[ ]:


automobile[automobile['num-of-doors'] == '?']


# In[ ]:


automobile.drop(index = [27,63], inplace = True)


# In[ ]:


automobile[automobile['num-of-doors'] == '?']


# # EDA  
# Univariate analysis

# In[ ]:


automobile.make.value_counts().head(10).plot(kind = 'bar', figsize = (8,2))
plt.title('Number of Vehicles by make')
plt.ylabel('Number of vehicles')
plt.xlabel('Make')


# Insurance risk rating Histogram
# 

# In[ ]:


automobile.symboling.hist(bins = 6, color = 'g')
plt.title('Insurance risk rating of vehicles')
plt.ylabel('Number of vehicles')
plt.xlabel('Risk rating')


# Normalized - losses histogram

# In[ ]:


automobile['normalized-losses'].hist(bins = 5, color = 'orange')
plt.title('Normalized losses of vehicles')
plt.ylabel('Number of vehicles')
plt.xlabel('Normalized losses')


# Fuel type bar chart

# In[ ]:


automobile.head(3)


# In[ ]:


automobile['fuel-type'].value_counts().plot(kind = 'bar', color = 'purple')


# In[ ]:


automobile['aspiration'].value_counts().plot.pie(figsize = (5,5), autopct = '%.2f')


# In[ ]:


automobile.horsepower[np.abs(automobile.horsepower-automobile.horsepower.mean())<=(3*automobile.horsepower.std())].hist(bins=5,color='red')
plt.title('Horse power histogram')
plt.ylabel('Number of vehicles')
plt.xlabel('Horse power')


# Curb weight histogram

# In[ ]:


automobile['curb-weight'].hist(bins=5, color = 'brown')
plt.title('curb weight histogram')
plt.ylabel('Number of vehicles')
plt.xlabel('Curb weight')


# Drive Wheels bar chart

# In[ ]:


automobile['drive-wheels'].value_counts().plot(kind ='bar', color = 'grey')
plt.title('Drive Wheels diagram')
plt.ylabel('Number of vehicles')
plt.xlabel('Drive wheels')


# Number of doors bar chart

# In[ ]:


automobile['num-of-doors'].value_counts().plot(kind = 'bar', color = 'purple')
plt.title('Number of doors frequency diagram')
plt.ylabel('Number of vehicles')
plt.xlabel('Number of doors')


# **Findings**  
# We have taken some key features of the automobile dataset for this analysis and below are our findings.
# 
# 1. Toyota is the make of the car which has most number of vehicles with more than 40% than the 2nd highest Nissan
# 2. Most preferred fuel type for the customer is standard vs trubo having more than 80% of the choice
# 3. For drive wheels, front wheel drive has most number of cars followed by rear wheel and four wheel. There are very less number of cars for four wheel drive.
# 4. Curb weight of the cars are distributed between 1500 and 4000 approximately
# 5. Symboling or the insurance risk rating have the ratings between -3 and 3 however for our dataset it starts from -2. There are more cars in the range of 0 and 1.
# 6. Normalized losses which is the average loss payment per insured vehicle year is has more number of cars in the range between 65 and 150.

# ## Correlation Analysis

# In[ ]:


sns.set_context('notebook', font_scale =1.0, rc = {'line.linewidth': 2.5})
plt.figure(figsize = (13, 7))
a = sns.heatmap(automobile.corr(), annot = True, fmt = '.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation = 90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation = 30)


# Findings: There are some good inferences we can take it from the correlation heat map.
# 
# 1. Price is more correlated with engine size and curb weight of the car
# 2. Curb weight is mostly correlated with engine size, length, width and wheel based which is expected as these adds up the weight of the car
# 3. Wheel base is highly correlated with length and width of the car
# 4. Symboling and normalized car are correlated than the other fields

# ## Bivariate Analysis

# In[ ]:


plt.rcParams['figure.figsize'] = (23,10)
ax = sns.boxplot(x='make', y='price', data = automobile)


# Boxplot of Price and make  
# Findings: Below are our findings on the make and price of the car  
# * The most expensive car is manufacture by Mercedes benz and the least expensive is Chevrolet  
# * The premium cars costing more than 20000 are BMW, Jaquar, Mercedes benz and Porsche  
# * Less expensive cars costing less than 10000 are Chevrolet, Dodge, Honda, Mitsubishi, Plymoth and Subaru  
# * Rest of the cars are in the midrange between 10000 and 20000 which has the highest number of cars

# In[ ]:


g = sns.lmplot('price', 'engine-size', automobile)


# In[ ]:


g = sns.lmplot('normalized-losses','symboling', automobile)


# In[ ]:


plt.scatter(automobile['engine-size'], automobile['peak-rpm'])
plt.xlabel('Engine size')
plt.ylabel('peak RPM')


# In[ ]:


g = sns.lmplot('city-mpg', 'curb-weight', automobile)


# In[ ]:


g = sns.lmplot('highway-mpg',"curb-weight", automobile,  fit_reg=False)


# In[ ]:


fig = plt.figure(figsize = (6,4))
automobile.groupby('drive-wheels')['city-mpg'].mean().plot(kind='bar', color = 'peru')
plt.title("Drive wheels City MPG")
plt.ylabel('City MPG')
plt.xlabel('Drive wheels')


# In[ ]:


fig = plt.figure(figsize = (6,4))
automobile.groupby('drive-wheels')['highway-mpg'].mean().plot(kind='bar', color = 'peru')
plt.title("Drive wheels Highway MPG")
plt.ylabel('Highway MPG')
plt.xlabel('Drive wheels')


# In[ ]:


plt.rcParams['figure.figsize']=(10,5)
ax = sns.boxplot(x="drive-wheels", y="price", data=automobile)


# In[ ]:


pd.pivot_table(automobile,index=['body-style','num-of-doors'], values='normalized-losses').plot(kind='bar',color='purple')
plt.title("Normalized losses based on body style and no. of doors")
plt.ylabel('Normalized losses')
plt.xlabel('Body style and No. of doors')


# ## Conclusion  
# Analysis of the data set provides
# 
# * How the data set are distributed
# * Correlation between different fields and how they are related
# * Normalized loss of the manufacturer
# * Symboling : Cars are initially assigned a risk factor symbol associated with its price
# * Mileage : Mileage based on City and Highway driving for various make and attributes
# * Price : Factors affecting Price of the Automobile.
# * Importance of drive wheels and curb weight

# In[ ]:




