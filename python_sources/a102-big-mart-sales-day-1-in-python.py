#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Reading the dataset(only for kernel!!)
import pandas as pd
import numpy as np
from collections import Counter
train = pd.read_csv("../input/Train_UWu5bXk.csv")
test = pd.read_csv("../input/Test_u94Q5KV.csv")


# In[ ]:


#ShortCuts for Kaggle you should know:
#Shift+Enter to run the cell

#when inside the bracket of any method:
##press shift+tab once to get a brief view of all the parameters of that method
##press shift+tab twice to get a more detailed view
##press shift+tab 4 times to get a tab of manual page of the method

##Tip: if you modified any dataframe by mistake
##To get to the previous version of dataframe,
##Only deleting the command which modified the dataframe is not enuough
##you will have to rerun the whole kernel.

#Viewing first 5 rows of the training dataset
train.head()


# In[ ]:


#Understanding the internal structure of the  training dataset.
train.info()

## We have 7 object variables, 4 float variables and 1 integer variable.
# We can see that Item_Weight and Item_Outlet has missing values.


# In[ ]:


#Understanding the internal structure of the  test dataset.
test.info()

## We see that there is no Item_Outlet-Sales in the test data. That is the variable that we
## have to predict.


# In[ ]:


train.describe(include='all')


# In[ ]:


train.Outlet_Establishment_Year.describe()


# In[ ]:


train.Outlet_Type.describe()


# In[ ]:


#Missing Values in train
train.isnull().sum()


# In[ ]:


#percentages of various outlet_type
train.Outlet_Type.value_counts(normalize=True)*100


# In[ ]:


#Appending the train and test dataset

## We can't append the datasets unless they have the same number of columns. Therefore, we will
## add another column(Item_outlet_Sales)(the dependent variable), to the test dataset.
test['Item_Outlet_Sales'] = np.nan


# In[ ]:


test.head()


# In[ ]:


## Now, we will append the datasets, and create a new dataset named 'data'
combined = train.append(test)


# In[ ]:


# Checking the dimensions of data
combined.shape


# In[ ]:


## Let's compare to the test and train datasets
train.shape


# In[ ]:


test.shape


# In[ ]:


#Checking the missing(NA) values of the data
## is.null() returns a logical vector which indicates the number of datapoints which are missing
## TRUE indicates missing.
## .sum() will add all the the values to get total missing values (True=1, Fasle=0) 
combined.isnull().sum()


# In[ ]:


## We can see that Item_Weight and Outlet_Size has missing values.
## Item_Outlet_Sales has missing values because it is the dependent variable, and the data points 
## corresponding to the test data have missing values. So we don't have to worry about that.

#Missing imputing values.
#Method-1(Mean method)
## We input the mean of rest of the data in place of the missing values.

#Finding the mean of the variable
## we have to specify to find the mean of the 'non-NA' values
Item_Weight_mean = combined.Item_Weight.mean(skipna=True)


# In[ ]:


Item_Weight_mean


# In[ ]:


## Now, we will imsert this value in place of the missing value
## We will make another dataset to observe.
combined_mean_impute = combined.copy(deep=True)


# In[ ]:


id(combined_mean_impute), id(combined.copy(deep=True))


# In[ ]:


combined_mean_impute.Item_Weight.fillna(Item_Weight_mean, inplace=True)


# In[ ]:


combined_mean_impute.Item_Weight.describe()


# In[ ]:


#Method-2 (Median imputation)
## We input the median of rest of the data in place of the missing values.
Item_Weight_median = combined.Item_Weight.median(skipna=True)


# In[ ]:


Item_Weight_median


# In[ ]:


## Now, we will imsert this value in place of the missing value
## We will make another dataset to observe.
combined_median_impute = combined.copy(deep=True)


# In[ ]:


id(combined_median_impute), id(combined.copy(deep=True))


# In[ ]:


combined_median_impute.Item_Weight.fillna(Item_Weight_median, inplace=True)


# In[ ]:


combined_median_impute.Item_Weight.describe()


# In[ ]:


##Since the only variable with missing values is numerical, we won't be using the k-NN method,
##which is for categorical variables. However, you can have a glance at k-NN implementation
##  https://www.youtube.com/watch?v=u8XvfhBdbMw


# In[ ]:


## First, we need to understand the distriution of Item_Weight. We can understand it better,
## if we can visually see it. Here, we will plot the histogram.
get_ipython().run_line_magic('matplotlib', 'inline')
## inline matplotlib command


# In[ ]:


combined.Item_Weight.plot(kind='hist', color='white', edgecolor='black', figsize=(10,6), title='Histogram of Item_Weight')


# In[ ]:


combined_mean_impute.Item_Weight.plot(kind='hist', color='white', edgecolor='black', figsize=(12,6), title='Histogram of Item_Weight')


# In[ ]:


combined_median_impute.Item_Weight.plot(kind='hist', color='white', edgecolor='black', figsize=(12,6), title='Histogram of Item_Weight')


# In[ ]:


## As we can see, the methods of imputing mean and median for this case 
## would be a bad idea(the data biases too much).
## so generally We will not replace the missing Values by Mean and Median, Instead We will use 
##"Predictive Modelling" to a estimate of the missing values. Search online for more.
## 


## Now, we will do some data cleaning.
## Notice that in Item_Fat_Content, we have LF, low fat and Low Fat, which are same. Also,
## reg and Regular are same. Therefore, add the observations of "LF" and "low fat" to "Low Fat",
## and "reg" to "Regular"

combined_mean_impute.Item_Fat_Content.value_counts()


# In[ ]:


combined.loc[combined.Item_Fat_Content.isin(['LF', 'low fat']),'Item_Fat_Content'] = 'Low Fat'


# In[ ]:


combined.loc[combined.Item_Fat_Content.isin(['reg']),'Item_Fat_Content'] = 'Regular'


# In[ ]:


combined.Item_Fat_Content.value_counts() # Viewing the variable


# In[ ]:


## Notice that Item_Type has factors which are not food items. So, Item_Fat_Content makes no
## sense. Hence we will add a new factor(level): "None", which will correspond to the 
## non-food items in Item_Type.

## Based on Item_Type, for "health and Hygiene", "Household" and "Others",
## we will change the Item_Fat_Content factor to "None".
combined.loc[combined.Item_Type.isin(['Health and Hygiene', 'Household', 'Others']),'Item_Fat_Content'] = 'None'


# In[ ]:


combined.Item_Fat_Content.value_counts() # Viewing the variable


# In[ ]:


## From earlier we know the column Outlet_Size has blank values. We will now procede to replace them.
## We will compare them with other variables, to understand better about the missing variables.
combined.Outlet_Identifier.value_counts()


# In[ ]:


combined.groupby('Outlet_Identifier').Outlet_Size.value_counts(dropna=False)


# In[ ]:


combined.groupby('Outlet_Type').Outlet_Size.value_counts(dropna=False)


# In[ ]:


## As per analysis, we would be better off by fixing category 'Small', on every 'Grocery Store',
## and for the remaining blank values, we will assign category 'Small' again.
combined.loc[combined.Outlet_Identifier.isin(['OUT010','OUT017','OUT045']), 'Outlet_Size'] = 'Small'


# In[ ]:


combined.Outlet_Size.value_counts()


# In[ ]:


## Since we are only concerned with how old the outlet is, and not the establishment year,
## we will substitute Outlet_Establishment_Year with Outlet_Year
combined.Outlet_Establishment_Year.value_counts()


# In[ ]:


combined.Outlet_Establishment_Year = 2013-combined.Outlet_Establishment_Year# Since this is 2013 data


# In[ ]:


combined.Outlet_Establishment_Year.value_counts().sort_index() 


# In[ ]:


## Visualizing Item_MRP
## We will use the density plot for this visulisation
import seaborn as sns

sns.kdeplot(data=combined.Item_MRP, bw=.2)
## It is obvious that we would be better off by converting Item_MRP to Categorical variable


# In[ ]:


combined['MRP_Factor'] = pd.cut(combined.Item_MRP, [0,70,130,201,400], labels=['Low', 'Medium', 'High', 'Very High'])


# In[ ]:


combined.MRP_Factor.value_counts()


# In[ ]:


#Removing the old variable
combined.drop('Item_MRP', axis=1, inplace=True)


# In[ ]:


combined.min()


# In[ ]:


## Notice that Item_Visibility has a minimum value of 0. It seems absurd that an item has 0 
## visibility. Therefore, we will modify that column.
## Here we Group by Item_Identifier, calculate mean for each group(excluding zero values), then we proceed
## to replace the zero values in each group with the group's mean.

## we have to replace 0's by na because, mean() doesnt support exclude '0' parameter 
##but it includes exclude nan parameter which is true by default
combined_mean_impute.loc[combined.Item_Visibility == 0, 'Item_Visibility'] = np.nan


# In[ ]:


#aggregate by Item_Identifier
IV_mean = combined_mean_impute.groupby('Item_Identifier').Item_Visibility.mean()
IV_mean


# In[ ]:


#replace 0 values
for index, row in combined.iterrows():
    if(row.Item_Visibility == 0):
        combined.loc[index, 'Item_Visibility'] = IV_mean[row.Item_Identifier]
        #print(combined.loc[index, 'Item_Visibility'])


# In[ ]:


combined.Item_Visibility.describe()
## see that min value is not zero anymore


# In[ ]:


## If you look at Item_Identifier, i.e. the unique ID of each item, it starts with either 
## FD, DR or NC. If you see the categories, these look like being Food, Drinks
## and Non-Consumables. So, we will create a new broad variable that assigns these three factors
combined['Item_Type_Broad'] = np.nan


# In[ ]:


combined.loc[combined.Item_Identifier.str.contains('DR'), 'Item_Type_Broad'] = 'Drinks'


# In[ ]:


combined.loc[combined.Item_Identifier.str.contains('FD'), 'Item_Type_Broad'] = 'Foods'


# In[ ]:


combined.loc[combined.Item_Identifier.str.contains('NC'), 'Item_Type_Broad'] = 'Non-Consumables'


# In[ ]:


combined.Item_Type_Broad.value_counts()


# In[ ]:


#We will aso create a more brief Item_Type_Broad2 from Item_Type
combined['Item_Type_Broad_2'] = np.nan


# In[ ]:


combined.loc[combined.Item_Type.isin(["Baking Goods","Breads"]), 'Item_Type_Broad_2'] = 'Bakery'


# In[ ]:


combined.loc[combined.Item_Type.isin(["Canned","Frozen Foods", "Dairy"]), 'Item_Type_Broad_2'] = 'Refrigerated'


# In[ ]:


combined.loc[combined.Item_Type.isin(["Meat","Sea Food"]), 'Item_Type_Broad_2'] = 'Non-Veg'


# In[ ]:


combined.loc[combined.Item_Type.isin(["Household","Others", "Health and Hygeine"]), 'Item_Type_Broad_2'] = 'Others'


# In[ ]:


combined.loc[combined.Item_Type.isin(['Hard Drinks']), 'Others'] = 'Alcoholic'


# In[ ]:


combined.loc[combined.Item_Type.isin(["Snack Foods","Soft Drinks", "Snacks"]), 'Item_Type_Broad_2'] = 'Snacks'


# In[ ]:


combined.loc[combined.Item_Type.isin(["Breakfast"]), 'Item_Type_Broad_2'] = 'NOTA'


# In[ ]:


combined.Item_Type_Broad_2.value_counts(dropna=False)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


## We will see if any colums as Outlier's
##applicable only for numberical values i.e. Item_Weight and Outlet_Establishment_Year and Item_Visibilityf
## we will use boxplots
combined.dtypes


# In[ ]:


##lets do it on column of Item_Weight
combined.boxplot("Item_Weight")


# In[ ]:


combined.boxplot("Outlet_Establishment_Year")


# In[ ]:


combined.boxplot("Item_Visibility")
## we can see that Item_Visibility has so many outliers
## Now the problem is we cant replace/remove so many outliers as it will create unwanterd bias


# In[ ]:


combined.Item_Visibility.plot(kind='hist')
## see that the graph is skewed 
## so we cant remove these Outliers which are mostly the right part of the graph
## In data cleaning we replaced some Outliers like where visibility = 0
## by mean of the visibility for that Item_Identifier


# In[ ]:


## if there were any removable outliers, then we would have used the below method
## calculate whiskers say a and b
## a = q1 - 1.5 * (inter quartile range)
##b = q2 + 1.5 * (inter quartile range)
## now we will remove any point which is beyond the whiskers (assuming removable outliers)
## i.e for points > b and points < a
## code will be as follows
## combined.loc[(combined.column_name > b) | (combined.column_name < a), column_name].drop(axis = 0, inplace=True)
## column_name refers to the columns which has theses outliers


# In[ ]:



## so we are done with a little bit of data cleaning 
## and for now we used mean and median to replace missing values, but in general cases we will use
##predictive modelling to replace the missing values
##predictive modelling: Use Machine Learning the features except the one with missing values and predict 
## the values of the feature with missing values.


# In[ ]:




