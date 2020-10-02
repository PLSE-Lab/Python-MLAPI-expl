#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# This is a demonstration of application of Principle Component on the Dataset Big Mart sale. Beofre we apply PCA, we have to do feature engineering, converting all features into numerical data for easier analysis.

# In[ ]:


#Loading data.
train=pd.read_csv("../input/Train.csv")
test=pd.read_csv("../input/Test.csv")


# In[ ]:


print("Rows and Columns in training set:", train.shape)


# In[ ]:


print("Rows and columns in testing set: ", test.shape)


# In[ ]:


#getting familiar with the structure of dataset.
train.columns


# Let's check for missing values in the dataset.

# In[ ]:


for col in train:
    val=train[col].isnull().sum()
    if val>0.0:
        print("Number of missing values in column ",col,":",val)
    #else:
     #   print("Number of values in column ",col,":",85223-val)


# In[ ]:


for col in test:
    val=test[col].isnull().sum()
    if val>0.0:
        print("Number of missing values in column ",col,":",val)
    #else:
     #   print("Number of values in column ",col,":",5681-val)


# For better prediction, we will be filling the missing values. Missing values can be filled by finding it's correlation with other features, sometimes the correlation is so propotional that we directly fill in the missing values. Another solution to this problem could be filling the missing datas with median value or mode value of the feature as would make more sense. 

# Heatmap to predict the correlation of Item weight and Outlet size. Heatmap doesn't work for catatogorical features, which could result in missing correlation between the missing values and the catagorical features, if there would be any relationship. So we first convert the catagorical features into numerical.

# In[ ]:


train.info()


# Here, we find that there are 7 categorical features:
# Item_Identifier, Item_Fat_Content, Item_Type, Outlet_Identifier, Outlet_Size, Outlet_Location_Type, Outlet_Type
# Let's use mapping to convert them into numerical data.

# In[ ]:


print(train["Item_Fat_Content"].value_counts())
print(train["Item_Type"].value_counts())
print(train["Outlet_Identifier"].value_counts())
print(train["Outlet_Size"].value_counts())
print(train["Outlet_Location_Type"].value_counts())
print(train["Outlet_Type"].value_counts())


# In[ ]:


combine=[test, train]
content_mapping = {'Low Fat': 1, 'Regular': 2, 'LF': 3, 'reg': 4, 'low fat': 5}
item_mapping = {'Fruits and Vegetables': 1, 'Snack Foods': 2, 'Household': 3, 'Frozen Foods': 4, 'Dairy': 5, 'Canned':6, 'Baking Goods':7, 'Health and Hygiene':8, 'Soft Drinks':9, 'Meat':10, 'Breads':11, 'Hard Drinks':12, 'Others':13, 'Starchy Foods':14, 'Breakfast':15, 'Seafood':16}
outletIdentifier_mapping ={'OUT027': 27, 'OUT013': 13, 'OUT046': 46, 'OUT035': 35, 'OUT049': 49, 'OUT045':45, 'OUT018':18, 'OUT017':17, 'OUT010':10, 'OUT019':19}
outlet_mapping = {'High': 1, 'Medium': 2, 'Small': 3}
Location_mapping = {'Tier 1': 1, 'Tier 2': 2, 'Tier 3': 3}
Type_mapping = {'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3, 'Grocery Store':4}
for dataset in combine:
    dataset['Item_Fat_Content'] = dataset['Item_Fat_Content'].map(content_mapping)
    dataset['Item_Fat_Content'] = dataset['Item_Fat_Content'].fillna(0)
    dataset['Item_Type'] = dataset['Item_Type'].map(item_mapping)
    dataset['Item_Type'] = dataset['Item_Type'].fillna(0)
    dataset['Outlet_Size'] = dataset['Outlet_Size'].map(outlet_mapping)
    dataset['Outlet_Size'] = dataset['Outlet_Size'].fillna(0) 
    dataset['Outlet_Identifier'] = dataset['Outlet_Identifier'].map(outletIdentifier_mapping)
    dataset['Outlet_Identifier'] = dataset['Outlet_Identifier'].fillna(0)
    dataset['Outlet_Location_Type'] = dataset['Outlet_Location_Type'].map(Location_mapping)
    dataset['Outlet_Location_Type'] = dataset['Outlet_Location_Type'].fillna(0)
    dataset['Outlet_Type'] = dataset['Outlet_Type'].map(Type_mapping)
    dataset['Outlet_Type'] = dataset['Outlet_Type'].fillna(0)

train.head()
test.head()


# Thus, through mapping, we have converted all except Item_Identifier into numeric data. Let's use cat.codes accessor to convert Item_Identifier into numeric data. But cat.codes works only on category datatype, so we'll first convert Item_Identifier into a category type and then use cat.codes to convert the last categorical feature to numeric.

# In[ ]:


train["Item_Identifier"] = train["Item_Identifier"].astype('category')
train.dtypes


# In[ ]:


#converting categorical data to numeric.
train["Item_Identifier"] = train["Item_Identifier"].cat.codes
train.head()


# In[ ]:


test["Item_Identifier"] = test["Item_Identifier"].astype('category')
test.dtypes


# In[ ]:


#converting categorical data to numeric.
test["Item_Identifier"] = test["Item_Identifier"].cat.codes
test.head()


# Thus, we have successfully converted all the catagorical features into numeric. Now it's time to find correlation between various features.

# In[ ]:


#Overall Correlation
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(train.corr(),annot=True)
plt.show()


# From above heatmap, we conclude that Outlet_Size as a positive correlation with Outlet Identifier.

# In[ ]:


pd.crosstab(train['Outlet_Size'], train['Outlet_Identifier'])


# from above,if table, we can safely conclude that, outlet_identifier=10,17,45, outlet_size=0
# if 13 1
# if 18,27,49 2
# if 19,35,46 3
# so, we fill the missing values acordingly

# In[ ]:


for row in train.itertuples(index=True, name='Pandas'):
    if row[9] is None:
        if(row[7]==10 or row[7]==17 or row[7]==45):
            train.loc[row.Index, 'Outlet_Size'] = 0
    else:
        if(row[7]==13):
            train.loc[row.Index, 'Outlet_Size'] =1
        else:
            if(row[7]==18 or row[7]==27 or row[7]==49):
                train.loc[row.Index, 'Outlet_Size'] =2
            else:
                if(row[7]==19 or row[7]==35 or row[7]==46):
                    train.loc[row.Index, 'Outlet_Size'] =3


# In[ ]:


for row in test.itertuples(index=True, name='Pandas'):
    if row[9] is None:
        if(row[7]==10 or row[7]==17 or row[7]==45):
            test.loc[row.Index, 'Outlet_Size'] = 0
    else:
        if(row[7]==13):
            test.loc[row.Index, 'Outlet_Size'] =1
        else:
            if(row[7]==18 or row[7]==27 or row[7]==49):
                test.loc[row.Index, 'Outlet_Size'] =2
            else:
                if(row[7]==19 or row[7]==35 or row[7]==46):
                    test.loc[row.Index, 'Outlet_Size'] =3


# Now, having filled the Outlet_size column, we are left with Item_Weight. From the heatmap, we see that Item_weight don't really have any positive relation with any of other feature, so let's fill it's missing values with the median value of each column.

# In[ ]:


train['Item_Weight'].fillna(train['Item_Weight'].dropna().median(), inplace=True)


# In[ ]:


test['Item_Weight'].fillna(test['Item_Weight'].dropna().median(), inplace=True)


# Now, having dealt with the catagorical values and the missing values, now we are left with scaling the dataset to carry out our PCA algorithm. We have PCA method in python in sklearn to carry it out implicitly, but there we have implemented PCA algorithm from scratch to understand the working of PCA algorithm. It requires som background knowledge of Linear algebra

# In[ ]:


import sklearn.preprocessing as preprocess
X_train = preprocess.scale(train)
X_test = preprocess.scale(test)


# **PCA Algorithm.**

# After having dealt with the missing values and categorical values, it's time to train a model with our data. But before that, let's learn about an useful algorithm, PCA which helps us in data visualization and dimentionality reduction. 

# PCA algorithm aims to provide a mention for better visualization of data, finding correlation between features and in dimensionality reduction by finding out the most significant in describing the full dataset.

# Ours is a 11 dimenional data. OUt of the 12, 11 are features and one is ouyput variable.

# Let's find the mean of each feature which are actually the dimensions of the dataset.

# In[ ]:


mean_Identifier = train["Item_Identifier"].mean()
mean_Weight = train["Item_Weight"].mean()
mean_Fat_Content = train["Item_Fat_Content"].mean()
mean_Visibility = train["Item_Visibility"].mean()
mean_Type = train["Item_Type"].mean()
mean_MRP = train["Item_MRP"].mean()
mean_OIdentifier = train["Outlet_Identifier"].mean()
mean_Year = train["Outlet_Establishment_Year"].mean()
mean_Size = train["Outlet_Size"].mean()
mean_Location = train["Outlet_Location_Type"].mean()
mean_Type = train["Item_Type"].mean()

print(mean_Identifier)
print(mean_Weight)
print(mean_Fat_Content)
print(mean_Visibility)
print(mean_Type)
print(mean_MRP)
print(mean_OIdentifier)
print(mean_Year)
print(mean_Size)
print(mean_Location)
print(mean_Type)


# In[ ]:


mean_vector = np.array([[mean_Identifier,mean_Weight,mean_Fat_Content,mean_Visibility,mean_Type,mean_MRP,mean_OIdentifier,mean_Year,mean_Size,mean_Location,mean_Type]])
print('Mean Vector:\n', mean_vector)
mean_vector.shape


# In[ ]:


X_train=np.array(train.drop("Item_Outlet_Sales", axis=1))


# In[ ]:


scatter_matrix = np.zeros((11,11))
for i in range(X_train.shape[0]):
    scatter_matrix += ((X_train[i,:].reshape(1,11) - mean_vector).T).dot(X_train[i,:].reshape(1,11)- mean_vector)
    #print(i)
print('Scatter Matrix:\n', scatter_matrix)


# In[ ]:


#Computing eigenvectors and corresponding eigenvalues.
# eigenvectors and eigenvalues for the from the scatter matrix
eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

print("Eigen Values:\n",eig_val_sc)
print("Eigen Vectors:\n",eig_vec_sc)


# In[ ]:


for i in range(len(eig_val_sc)):
    eigvec_sc = eig_vec_sc[:,i].reshape(11,1).T
    
    print('Eigenvector {}: \n{}'.format(i+1, eigvec_sc))
    print('Eigenvalue {} from scatter matrix: {}'.format(i+1, eig_val_sc[i]))
    print(40 * '-')


# In[ ]:


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
for i in eig_pairs:
   print(i[0])


# Thus using PCA algorithm, I reduce the dimensionality from 11 to 7.
# We can reject the last 4 dimensions
# The features we drop are Outlet_Establishment_Year, Outlet_Size, Outlet_Location, Outlet_type.
# The features we take into account are Item_Identifier, Item_Weight, ITem_Fat_Content, ITem_Visibilty, Item_Type, Outlet_Identifier, Item_MRP.

# **References** :
# http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
