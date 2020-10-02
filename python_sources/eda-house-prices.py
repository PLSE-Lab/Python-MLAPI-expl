#!/usr/bin/env python
# coding: utf-8

# Hello Kagglers,
# ****
# This is my first Kernel. The main purpose of this Kernel is to only do the Exploratory Data Analysis of the data. Please let me know if you find anything which does not make sense.

# In[ ]:


#Importing all the required libraries
import pandas as pd
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


# Reading the train and test dataset
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


#Printing out the number of samples and features of Training as well as test dataset

print("{} no of features with {} numbers of samples in training".format(train.shape[1],train.shape[0]))
print("{} no of features with {} numbers of samples in testing".format(test.shape[1],test.shape[0]))


# Since, "Id" column will not contribute in model building, I am dropping this column from both (training and test) datasets. We got one less feature for both the datasets.

# In[ ]:


train_id = train['Id']
test_id = test['Id']
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
print("{} no of features with {} numbers of samples in training".format(train.shape[1],train.shape[0]))
print("{} no of features with {} numbers of samples in testing".format(test.shape[1],test.shape[0]))


# Now, lets combine the train and test dataset to make the changes (if any) to the features in both the datasets together. Lets call the combined data set as **full data** now onwards.

# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
full_data = pd.concat((train, test)).reset_index(drop=True)
full_data.drop(['SalePrice'], axis=1, inplace=True)
print("full_data size is : {}".format(full_data.shape))


# full data consists of 2919 samples and 79 features as we would have wanted. All good until now.
# Now, Lets, try to find the missing values in full data. The below code snippet will print out the percentage of the missing values for every features which have missing values.

# In[ ]:


full_data_null = (full_data.isnull().sum() / len(full_data)) * 100
full_data_null = full_data_null.drop(full_data_null[full_data_null == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Data Percentage' :full_data_null})
missing_data


# We can see in the previous dataframe that many variables have large number of null values (a couple have over 90%). Lets see why these have so many null values. We can see from the data description that if Pool Quality is NA, then there is no pool. This makes sense that most of the houses does not have pool. So, we will impute "None" with null values.
# Similarly, there are other features which does not have Misc Features, does not have Alley, does not have Fence, does not have Fire Place , we will impute "None" for all the null values into these features.
# 
# There are some other features where we imputed with either "0" (if continuous) or "None" (if Categorical) for null values where those features were not installed in that home.

# In[ ]:


full_data["PoolQC"] = full_data["PoolQC"].fillna("None")
full_data["MiscFeature"] = full_data["MiscFeature"].fillna("None")
full_data["Alley"] = full_data["Alley"].fillna("None")
full_data["Fence"] = full_data["Fence"].fillna("None")
full_data["FireplaceQu"] = full_data["FireplaceQu"].fillna("None")
full_data["MasVnrType"] = full_data["MasVnrType"].fillna("None")
full_data["MasVnrArea"] = full_data["MasVnrArea"].fillna(0)
full_data['MSSubClass'] = full_data['MSSubClass'].fillna("None")
full_data = full_data.drop(['Utilities'], axis=1)


# In[ ]:


for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    full_data[col] = full_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    full_data[col] = full_data[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    full_data[col] = full_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    full_data[col] = full_data[col].fillna('None')


# For some categorical features we took the mode of that feature in that corresponding Neighbourhood and imputed the Mode values in place of null values. 

# In[ ]:


full_data['MSZoning'] = full_data.groupby("Neighborhood")["MSZoning"].transform(lambda x: x.fillna(x.mode()))
full_data["Functional"] = full_data["Functional"].fillna("Typ")
full_data["LotFrontage"] = full_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
full_data['Electrical'] = full_data.groupby("Neighborhood")["Electrical"].transform(lambda x: x.fillna(x.mode()[0]))
full_data['KitchenQual'] = full_data.groupby("Neighborhood")["KitchenQual"].transform(lambda x: x.fillna(x.mode()[0]))
full_data['Exterior1st'] = full_data.groupby("Neighborhood")["Exterior1st"].transform(lambda x: x.fillna(x.mode()[0]))
full_data['Exterior2nd'] = full_data.groupby("Neighborhood")["Exterior2nd"].transform(lambda x: x.fillna(x.mode()[0]))
full_data['SaleType'] = full_data.groupby("Neighborhood")["SaleType"].transform(lambda x: x.fillna(x.mode()[0]))


# In[ ]:


#Check remaining missing values if any 
full_data_null = (full_data.isnull().sum() / len(full_data)) * 100
missing_data = pd.DataFrame({'Missing Percentage' :full_data_null})
missing_data.head()


# We can see that we have completed imputing the missing values which could result in more accurate model.
# 
# Now, we will try to see the correlation between the features and he target variable (Sale Price) and plot some scatter plots to see if we find any outlier or any interesting pattern.

# In[ ]:


corr = train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
corr = corr.SalePrice
display(corr)


# In[ ]:


train.head()


# The first feature which have the maximum correlation with the Sale Price is OverallQual, As we can see in the scatter plot, that when the OverallQual is increasing, the average Sale Price for each OverallQual is increasing which actually makes sense.

# In[ ]:


plot = plt.subplot()
plot.scatter(x = train['OverallQual'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('OverallQual', fontsize=13)
plt.show()


# Now, the second most important feature is the GrLivArea. We can see in the graph that as the living area increases there is an increase in the SalePrice of the house. But we can also see that there are couple of outliers where GrLivArea>4500 but Sale Price<300000. We may want to remove this outlier for fitting the model.

# In[ ]:


plot = plt.subplot()
plot.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# Removing the outlier we have seen above. And checking the graph again for sanity check.

# In[ ]:


train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)


# In[ ]:


plot = plt.subplot()
plot.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# We can build scatter plots/bar graph for the top N features and see if there some outliers.
# 
# Next, I am plotting the heatmap for correlation between varibales. We may want to remove one of the variables from the pair of variables having large correclation. 

# In[ ]:


corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# As I mentioned before, this was my effort just to do some EDA and post my first Kernel.
# 
# I will be looking forward to post more Kernels with more content as I continue learning.
# 
# Please feel free to comment for any suggestions//improvements.
# Cheers!

# In[ ]:




