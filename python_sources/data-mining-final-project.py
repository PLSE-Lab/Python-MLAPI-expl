#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **This will be your workspace for Kaggle's Machine Learning education track.**
# 
# You will build and continually improve a model to predict housing prices as you work through each tutorial.  Fork this notebook and write your code in it.
# 
# The data from the tutorial, the Melbourne data, is not available in this workspace.  You will need to translate the concepts to work with the data in this notebook, the Iowa data.
# 
# Come to the [Learn Discussion](https://www.kaggle.com/learn-forum) forum for any questions or comments. 
# 
# **To Do List:
# 1. Remove 1stFlrSF
# 2. Garage Cars and Garage Area
# 3. TotRmsAbvGrd and GrLivArea
# 4. **
# 
# # Write Your Code Below
# 
# 

# In[27]:


import pandas as pd

main_file_path = '../input/train.csv'
originalDF = pd.read_csv(main_file_path)
print(originalDF)


# # Data Description

# In[28]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Correlation Matrix
corrmat = originalDF.corr()
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=.8, square=True)


# In[33]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import Imputer
from sklearn import model_selection

def buildAndTestModel():
    
    originalDF = pd.read_csv('../input/train.csv')
    #preprocess(originalDF, originalDF)
    originalDF.dropna(axis = 0, subset = ['SalePrice'], inplace = True)
    y = originalDF.loc[:, "SalePrice"]
    X = originalDF.loc[:, ["LotArea", "YearBuilt", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GarageArea", "YrSold"]]
    
    gbr = GradientBoostingRegressor()
    
    my_imputer = Imputer()
    X = my_imputer.fit_transform(X)
    
    cvScores = model_selection.cross_val_score(gbr, X, y, cv=3,
                                               scoring = 'neg_mean_absolute_error')
    print("Accuracy:", -1 * cvScores.mean())
    
buildAndTestModel()


# # Treating Missing Values

# In[35]:





# # Feature Engineering

# In[37]:


##---------- Create new columns based on existing ones

#replace yearBuilt with Years Old (2018 - the year)
yearBuilt = originalDF.loc[:, "YearBuilt"]
originalDF.loc[:, "YearsOld"] = 2018 - originalDF.loc[:, "YearBuilt"]
originalDF.drop(['YearBuilt'], axis= 1)


#create new column called Yard Size (Lot Area - Ground Floor Size)
lotArea = originalDF.loc[:, "LotArea"]
gFlrSize = originalDF.loc[:, "GrLivArea"]
originalDF.loc[:, "YardSize"] = lotArea - gFlrSize
print(originalDF.loc[:, "YardSize"])



# # Treating Missing Values

# In[40]:


total = originalDF.isnull().sum().sort_values(ascending=False)
percent = (originalDF.isnull().sum()/originalDF.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
## potentially delete PoolQC/MiscFeature/Alley (greater than 90% missing values)

## One Hot Encoding
colNames = ['LotArea', 'Neighborhood','OverallQual','GrLivArea','FullBath','GarageCars','YardSize','YearsOld']
print(originalDF.loc[:, colNames])
trainPredictors = originalDF.loc[:, colNames]
trainDF = pd.get_dummies(trainPredictors)
print(trainDF)

