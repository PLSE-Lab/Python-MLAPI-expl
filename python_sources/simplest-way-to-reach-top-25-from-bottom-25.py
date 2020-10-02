#!/usr/bin/env python
# coding: utf-8

# > ### Everyone is a beginner when they start. 
# 
# Isn't it frustrating when you see your score to be at the bottom 30%, even when you have spent months studying about the subject and days working on the project ?
# 
# Well I know the feeling well. It happens to all of us when we are beginning. But what's most interesting is the everything in this universe is a POWER LAW.
# 
# You ask what is POWER LAW?
# 
# Thats what Peter Theil ( Co-founder of PayPal and Early invester of Facebook and SpaceX ) say about our old exponential curve.
# It takes alot of time in the beginning to reach something and then everything rises exponentially fast after a certain point. We will try to reach after than certain point, in this notebook.
# 
# There are only few little things when you learn (which won't even take alot of time) will help you easily achieve top 25%. 
# 
# To increase the accuracy (or reduce the error score) data prepreprocessing and selecting model and their parameter, plays a big role. Here, we will only focus on data preprocessing part, which will be good enough to give us a big enough jump in score. Later we will train the model with the simplest LinearRegression Model.
# 
# So, are you ready?

# # Data Preprocessing :
# 
# ## There are 4 simple things we will do for our Data-Preprocessing :
# 
# ### Removal :
# - Removal of Features having NaN values more than 15%
# - Removal of similar features who do not provide any extra information. Like GarageArea can also be determine by GarageCars.
# 
# ### Filling the Missing Data :
# - We will fill the null values with their mean/mode.
# - Features with object datatype has to filled with their mode.
# - While Features with float datatype will be filled with thier mean.
# - (optional) Instead of filling null values with their mean, you can also chose a random value from the range of values near their mean. For eg, if the mean of dataset is X, then instead of filling X at every null values, you can fill a random value from the range let say (X-10, X+10). This range is determined by studying the graph. But for simplicity, we won't do that in this notebook. ( Also the % of null values remaining are not that high to make a lasting difference )
# 
# ### Outliers :
# - It is also possible that some data points can deviate significantly from the rest for those features. May by due to error or some special features.
# - This can disturb our learning curve.
# - Not to ignore this. You will see how significantly does this affect our final score, even with just the minor change.
# 
# ### Normalizing our SalePrice:
# - SalePrice, that's what we are after. Our final goal in this project is to predict SalePrice.
# - We will see if our SalePrice is normalize or not.
# - One simple trick to normalize it, if it is not.
# - Again, normalizing the SalePrice will play a significan't role in our final error score.
# 
# Enough Talking, time for action! For Marvel fans, this can be like a quest to get the Infinity Stone in a Complex Universe.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ### Lets see how our universe looks like :

# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


pd.set_option('display.max_columns', None)               # To get all the columns
train.head()


# Ohh...  It looks like this universe has lot of elements. There are so many columns to work with. This at first, can seems overwhelming, but we need not worry, because we have python and pandas. This will make things very simple for us to visualize. You can say we have a Thor Hammer to fight with our enemies.

# # Removal

# ### Removing features with higher null values :
# Having null values is having unknown item our universe. We don't want that. If the unknown items is more than 15%, we will completely remove the entire feature from our universe.
# 
# Lets see which features have null values :

# In[ ]:


def describe_features_with_null_values(df):
    null_count = df.isnull().sum()
    null_values = null_count[null_count > 0]

    null_values = null_values.sort_values(ascending = False)
    
    null_perc = null_values*100/len(df)

    null = pd.DataFrame(null_values, columns = ['Null Count'])
    null['Percentage'] = round(null_perc, 2)

    return null


# In[ ]:


df_null = describe_features_with_null_values(train)
df_null


# You see.. how easy it is to explore our universe with Python.

# We can see 6 features having more than 15% of Null values. We will remove those features.
# 
# For the rest, we will fill them with their mean/mode.

# In[ ]:


higher_null_values_list = list(df_null[df_null['Percentage'] > 15].index)
print("Features having more than 15% Null values are :", higher_null_values_list)


# In[ ]:


train = train.drop(higher_null_values_list, axis = 1)


# ## Removing similar Features :
# - There can be many features, which doen't provide any extra information. So we will remove those.
# - The simplest way to look at that is using correlation and its heatmap.
# 
# ### *Correlation matrix :*
# - Correlation matrix will show how much are the features related to each other. 
# - It assigns a value for every two pairs of features. 
# - Higher values shows higher relation.
# - More related features means, we can obtain the information from either of that feature and remove the other.
# 
# ### *Heatmap :*
# - Visualization of Correlation matrix is simple by using heatmap

# In[ ]:


corrmat = train.corr()        # Finds correlation between all the columns
f, ax = plt.subplots(figsize=(12, 9))             # Increases the figure size to (12, 9)
sns.heatmap(corrmat, vmax = 0.8, square=True);


# - look at the whitist blocks in the above figure. They are more related feature pairs. 

# ### *Looking above, I derived the following most related features :*
# - TotalBsmtSF and 1stFlrSF
# - GarageCars and GarageArea
# - TotRmsAbGrd and GrLivArea
# - GarageYrBlt and YearBuilt
# 
# ### *Now which to keep and which to remove ?*
# - We will keep the feature which are more related to SalePrice.
# - Also we can determine this by our intuition. For eg, TotBsmtSF and 1stFlrSF are equally related. TotBsmtSF represents 'total basement square feet' while 1stFlrSF represents '1st Floor square feet'. As far as my intuition is concerned, I think TotBsmtSF is more likely feature we will consider for buying a house.
# 
# ### *So clearly we will remove :*
# - 1stFlrSF
# - GarageArea
# - TotRmsAbvGrd
# - GarageYrBlt

# In[ ]:


train = train.drop(['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd', 'GarageYrBlt'], 1)


# # Filling the Missing Data
# We will create a function which will automatically fill the missing values with their mean/mode.
# 
# Lets look at again our missing data.

# In[ ]:


df_null = describe_features_with_null_values(train)
df_null


# In[ ]:


def fill_null_values(df):
    
    null_df = describe_features_with_null_values(df)
    
    # Below 2 lines will give us features with object, float/int datatype respectively.
    obj_features = df[null_df.index].dtypes[df[null_df.index].dtypes == object].index
    float_features = df[null_df.index].dtypes[df[null_df.index].dtypes == float].index
        
    for feature in obj_features:
        df[feature] = df[feature].fillna(df[feature].mode().values[0])
    
    for feature in float_features:
        df[feature] = df[feature].fillna(df[feature].mean())
        
    return df


# In[ ]:


train = fill_null_values(train)


# # Outliers :
# Outlier is a rare chace of occurrence within a given data set. It is an observation point that is distint from other observations.
# 
# It is possible that many features have outliers. But what affects our learning curve is the outlier present in the feature which significantly affects the SalePrice.
# 
# We will first find out which features significantly affects our SalePrice. And we will call these our favourite weapons. 
# 
# Now if our favourite weapons are not perfect, we can lose the battle. So we will find out anything that makes it imperfect and then remove that.

# In[ ]:


print("Features which most affects to our SalePrice : \n")
related_cols = corrmat.nlargest(10, 'SalePrice')
print(related_cols['SalePrice'])


# Note : GarageArea and 1stFlrSF is already removed.
# 
# From above we can say that our favourite weapons are : 
# - OverallQual
# - GrLivArea
# - GarageCars
# - TotalBsmtSF
# 
# We will examine our weapons carefully.

# In[ ]:


fig, axes = plt.subplots(2,2, figsize = (8,7))

axes[0][0].scatter(train['OverallQual'], train['SalePrice'])

axes[0][1].scatter(train['GrLivArea'], train['SalePrice'])

axes[1][0].scatter(train['GarageCars'], train['SalePrice'])

axes[1][1].scatter(train['TotalBsmtSF'], train['SalePrice'])

fig.tight_layout()


# ###### From above plots:
# - We can clearly see that SalePrice almost exponentially increases as the feature value increases
# - 2 point in GrLivArea and 1 point in TotalBsmtSF are not following the croud. These are our outliers. It is possible that those house have some drastically different features which gives them lowerprice even if this feature value is good. Lets check if the datapoint in them is the same or not.

# In[ ]:


ind1 = train['TotalBsmtSF'][train['TotalBsmtSF'] > 5000].index.values
ind2 = train['GrLivArea'][train['GrLivArea'] > 4500].index.values

print('Index of datapoint in TotalBsmtSF different from croud :', ind1)
print('Index of datapoint in GrLivArea different from croud :', ind2)


# See... that outlier property is coming from the same datapoint with index 1298. 
# 
# We will remove both the datapoint with index 523 and 1298

# In[ ]:


train = train.drop(ind2)


# Now our weapons are perfect. So lets jump to our next thing.

# # Normality
# - Now its time to check whether our 'SalePrice' shows Normal distribution or not.
# - If they don't show normal distribution, we will apply Normality which will improve our final score.

# In[ ]:


sns.distplot(train['SalePrice']);


# It is clear that our SalePrice is not Normal.
# 
# We can Normalize this by just applying the logarthm to the values. Its that simple trick.

# In[ ]:


# Applying Normality
train['SalePrice'] = np.log(train['SalePrice'])


# In[ ]:


sns.distplot(train['SalePrice']);


# Hooofff... We reached long way. 
# 
# - We remove the unknown and unwanted features from this universe.
# - We filled the missing values
# - We removed the outliers
# - And we also normalized our SalePrice
# 
# Now only one thing is left before we can enter into the battle. And that is an Armor or a shield. 
# 
# We will use Captain America Shield. Means we will convert our object datatype or alphabets to float datatype, so that we can train them.
# 
# So lets get onto that.

# # Converting categorical data to numeric

# In[ ]:


obj_mask = train.dtypes == object
obj_features = list(obj_mask[obj_mask].index)

le = LabelEncoder()
train[obj_features] = train[obj_features].apply(le.fit_transform)


# In[ ]:


train.head()


# ### *Converting Pandas Dataframe to Numpy Array*

# In[ ]:


Y_train = train['SalePrice'].values
X_train = train.drop(['SalePrice'], 1).values


# In[ ]:


print("Shape of X_train :", X_train.shape)
print("Shape of Y_train :", Y_train.shape)


# Bammnn ! Now we also have Captain America's Shield. 
# 
# We are now ready to enter the battle.

# # Model :
# - For the simplicity. We will just use the simple LinearRegression model from sklearn.
# - You can also try out more models and find out which gives you better score.

# In[ ]:


lr = LinearRegression()
lr.fit(X_train, Y_train)

ypred = np.exp(lr.predict(X_train))

# Note : we are applying np.exp(). Thats because our Y_train is Normalized by applying Logarithm

err = round(mean_absolute_error(np.exp(Y_train), ypred)/100000, 5)

print("Error Score for Training Set :", err)


# Our Mean Absolute Error comes out to be 0.14031. 
# 
# Now lets do the same for Test set as well, so that we are ready to upload our final file.

# # Test Set
# ##### Now we will do the same removal and filling the NaN values for Test set.

# In[ ]:


test = test.drop(higher_null_values_list, 1)
test = test.drop(['1stFlrSF', 'GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd'], 1)
test = fill_null_values(test)


# In[ ]:


obj_mask_test = test.dtypes == object
obj_features_test = list(obj_mask_test[obj_mask_test].index)

le = LabelEncoder()
test[obj_features_test] = test[obj_features_test].apply(le.fit_transform)


# In[ ]:


X_test = test.values


# # Test set Predictions

# In[ ]:


test_pred = np.exp(lr.predict(X_test))


# In[ ]:


test_ind = np.arange(1461, 1461 + len(test))
test_series = pd.Series(test_pred, index = test_ind)


# In[ ]:


#test_series.to_csv('predictions.csv')


# ## Boom ! We killed the villian and now we have our Infinity Stone Secured with us.
# 
# I got 0.12365 error score by submitting the above 'prediction.csv' file. You can still do better score by using other models to train.
# 
# This was just the simplest way I found to get a good score. 
# 
# ### *If you find this notebook helpfull, then do give this an UPVOTE. Or share it with others while sharing your own notebook.*
# 
# Peace !

# In[ ]:




