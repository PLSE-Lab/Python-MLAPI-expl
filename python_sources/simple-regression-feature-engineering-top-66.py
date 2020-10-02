#!/usr/bin/env python
# coding: utf-8

# # The purpose of this Kernel is to practice some feature engineering. As I go along, I will try to write out my thoughts as much as possible to show what I am doing. 
# 
# ### My goal here is to cut through the noise of the features available and try to create new, more revealing and meaningful features from what we have available.
# 
# ### The basic idea for this kernel was derived from [Pedro's Notebook](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python) and the excellent data analysis shown in this kernel. 
# 
# ### The feature engineering done here is inspired and in some places copied from [Serigne's Notebook](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard). I thought about changing things to create my own version of it, but hey...why mess with perfection. 
# 
# ### With all the niceties out of the way, let get to work .
# 
# ![Larry David Bernie](https://media.giphy.com/media/3oz8xHY5TPG9CAW0xi/giphy.gif)
# 

# ** Let's start by importing the main data analysis libraries **

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


sample_sub  = pd.read_csv('../input/sample_submission.csv')
train_df  = pd.read_csv('../input/train.csv')
test_df  = pd.read_csv('../input/test.csv')
# sample_sub.head()


# In[ ]:


train_df.head()


# In[ ]:


len(train_df.columns)   


# ### There are 81 columns in our dataset. So that is 81 features that we are dealing with.
# 
#  **We are going to try to remove the variables that are superficial,redundant or don't affect the SalePrice by all that much.**

# In[ ]:


#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train_df.shape))
print("The test data size before dropping Id feature is : {} ".format(test_df.shape))

#Save the 'Id' column
train_ID = train_df['Id']
test_ID = test_df['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train_df.drop("Id", axis = 1, inplace = True)
test_df.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train_df.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test_df.shape))


# ### Let's create a correlation matrix and see how each variable/feature is related to others
# 
# *All the work in the correlation matrix is thanks to [Pedro's Notebook](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)* 

# In[ ]:


corr = train_df.corr()
fig , ax = plt.subplots(figsize = (7,7))
sns.heatmap(corr, vmax = 0.8, square = True)


# ** Taking a quick look at this correlation matrix we can see that some of the strongest correlations are limited to a few variables.
# These correaltions are represented by the faded square inside the large square.Thankfully for us this means that we can focus on a certain number of variables and get a good model accuracy instead of having to work with 81 variables. **

# ## Let's take a look at the 10 variables that are most strongly correlated with Sale Price

# In[ ]:


#### Sales Price Correlation Matrix 

k = 10     # number of variables we are looking for.

cols = corr.nlargest(k,'SalePrice')['SalePrice'].index
# cols
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                 annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# ## It seems that Sale Price of a house is most affected by these variables ( Doesn't include the SalePrice variable itself):
# 
# 1. OverallQual  = Rates the overall material and finish of the house
# 2. GrLivArea    = Above grade (ground) living area square feet
# 3. GarageCars   = Size of Garage in car capacity 
# 4. GarageArea   = Size of garage in square feet
# 5. TotalBsmtSF  = Total square feet of basement area
# 6. 1stFlrSF     = First Floor square feet
# 7. FullBath     = Full Bathrooms Above Grade.
# 8. TotRmsAbvGrd = Total rooms above grade (does not include bathrooms)
# 9. YearBuilt    = Original Construction Date.
# 
# 
# ## Some of these variables are redundant. 
# * GrLivArea and TotRmsAbvGrd seem like twins. Since GrLivArea is the more correlated variable we will keep this.
# * GarageCars is a function of GarageArea since how many cars you can hold in a garage would depend on the area of the garage. 
# * TotalBsmtSF seems to be very similar, if not the same as 1stFlrSF. Since TotalBsmtSF is higher in correaltion to the SalePrice      we will keep this.
# 
# ## Other variables are superficial. 
# * FullBath = really? 
#     
# 
# 

# # TIME TO DO SOME DATA CLEANING 
# 
# 
# ## First let's create a DataFrame containing all the data in one place. That is training and testing data in one place.

# In[ ]:


train_df.shape


# In[ ]:


ntrain = train_df.shape[0]
ntest = test_df.shape[0]
y_train = train_df.SalePrice.values


# In[ ]:


y_train


# ### Let's create a dataframe to contain all the training and testing data. We will drop the SalePrice variable. However, we already have that stored in the y_train variable so not to worry!!

# In[ ]:


all_data = pd.concat((train_df, test_df), sort= False).reset_index(drop = True)
all_data.drop(['SalePrice'], axis=1,inplace=True)
print('The combined data variable size is {}:'.format(all_data.shape))
# all_data.head(5)


# ## Let's replace the missing values first. 
# 
# 

# ### Let's create a DataFrame to keep track of what percentage of values in each column/feature are missing or empty. 

# In[ ]:


all_data_na = (all_data.isna().sum()/len(all_data))*100
all_data_na = all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending = False)
missing_data = pd.DataFrame({'Missing Data ':all_data_na})

# len(all_data_na)   # 34 

missing_data.head(10)


# ### Let's take a look at this graphically too what percentage of the total values in each column are either missing or 0. 

# In[ ]:


fig, ax = plt.subplots(figsize =(10,7))
plt.xticks(rotation='90')
sns.barplot(x = all_data_na.index, y = all_data_na)
plt.xlabel('Features', fontsize = 15)
plt.ylabel('Percentage of missing values', fontsize = 15)
plt.title('Percentage missing by Total values')


# In[ ]:


all_data.shape 


# #### The starting shape of the data is 2919 X 80. We are going to try to eliminate redundant and superficial variables which way we will keep chipping at the columns and hence reducing noise in our data.
# 
# #### We will start by dropping 'PoolQC', 'MiscFeature', 'Alley' and 'Fence' columns from the dataset. All these columns have a higher than 80% empty or missing value rate. **

# In[ ]:


all_data = all_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis =1 )


# 
# ### I have seen some other kernels continue with columns/variables that have a high proportion of missing values. However, I don't feel comfortable moving ahead with columns that have over 80% missing values. At such a point, I feel like we are injecting bias if we replace them with say None or some other values. 
# 
# 

# In[ ]:


all_data.shape   #down to 75 variables now.


# #### Let's drop those 4 columns from the missing_data dataframe as well, just to keep things consistent.

# In[ ]:


missing_data = missing_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis =0)


# In[ ]:


missing_data.head(10)
# len(missing_data)


# # Let's impute some missing values.
# 
# **Most of the work done in the imputing part is inspired, if not outright copied from [Serigne's Notebook](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)**

# > * **FireplaceQu** : Data description says NA means no fireplace.

# In[ ]:


all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna("None")
# all_data['FireplaceQu']


# * **LotFrontage** : Since all the houses surrounding each other would have roughly the same area. We can replace the area by the a
# a**rea of each neighborhood

# In[ ]:


all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
# all_data['LotFrontage']


# ### Dealing with the Garage variables. There are 7 garage variables. 
# 
#  They are : **GarageType, GarageYrBlt, GarageFinish , GarageCars, GarageArea, GarageQual, GarageCond** 
#  
#  **Some of them are redundant:**
#  
# * GarageCars is a function of GarageArea since how many cars a garage can fit depends on the area of the garage. So we will drop the GarageCars variable. 
# * GarageQual and GarageCond are essentially conveying the same information. So I will drop GarageQual. (Flip of a coin)
# 
# *** I don't think that the year a Garage was built has any inherent value to SalePrice, so I will drop it too.**
# 
# **All in all, we will keep these GarageVariables:**
# 
#     GarageType, GarageFinish, GarageArea, GarageCond. 
#     
# **...and drop these GarageVariables:**
#     
#     GarageCars, GarageYrBlt, GarageQual. 
#     
#     
# 
# 

# In[ ]:


all_data = all_data.drop(['GarageCars', 'GarageYrBlt', 'GarageQual'], axis =1)


# In[ ]:


all_data.shape  #The second dimension dropped from 75 to 72 as expected because we dropped 3 variables/features. 


# ## Now, let's take care of the Garage variables.  What will we do?? 
# 
# **GarageType: ** is suposed to represent Garage location such as whether it is attached to the home, or if separate.A missing value here is represented by NA. We will replace this with None. 
# 
# **GarageFinish: ** is suppposed to represent the interior finish of the garage. A missing value here is represented by NA. We will replace this with None.
# 
# **GarageArea: ** is supposed to represent the area of the garage in sq.ft. A missing value here will be replaced by 0. 
# 
# **GarageCond: **  is supposed to represent the condition that Garage is in. Any missing value here is represented by a NA. We will replace that with a None. 
# 
# ## So, we will replace the missing values in **GarageType, GarageFinish, GarageCond** with a None and that in **GarageArea** with a 0.
# 

# In[ ]:


for col in ('GarageType', 'GarageFinish', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')


# In[ ]:


all_data['GarageArea'] = all_data['GarageArea'].fillna(0)


# # NOW, LET'S TAKE A LOOK AT ALL THOSE BASEMENT VARIABLES. 
# 
# ** There are 11 variables describing features related to Basements. **
# 
# Those features are: 
# 
# * **BsmtQual:** Height of the basement. Any missing value is represented by NA. 
# * **BsmtCond:** General condition of the basement. Any missing value is represented by NA.
# * **BsmtExposure:** Walkout or garden level basement walls. Any missing value is represented by NA.
# * **BsmtFinType1:** Quality of basement finished area. Any missing value is represented by NA.
# * **BsmtFinSF1:** Type 1 finished square feet. Any missing value will be replaced by 0. 
# * **BsmtFinType2:** Quality of second finished area (if present). Any missing value is represented by NA.
# * **BsmtFinSF2:** Type 2 finished square feet. Any missing value will be represented by 0. 
# * **BsmtUnfSF:** Unfinished square feet of basement area. Any missing value will be represented by 0. 
# * **TotalBsmtSF:** Total square feet of basement area. Any missing value will be represented by 0. 
# * **BsmtFullBath:** Basement Full Bathrooms. Any missing value will be represented by 0. 
# * **BsmtHalfBath:** Basement Half Bathrooms. Any missing value will be represented by 0. 
# 

# ## What to do: 
# 
# * BsmtQual and BsmtCond are representing physical or condition features for the basement so we will keep them. 
# * BsmtExposure seems like a superficial feature. How many people care about garden level basement walls...whatever that means?!!
# * BsmtFinType1 and BsmtFinType2 represent subjective opinion of an observer of the basement finshed area for type1 and type2. So I will be **dropping these two.**
# * BsmtFinSF1 and BsmtFinSF2 represent the finished square feet of the basement. We can add these two to make a new variable called BsmtFinSF which will be a total of both types of unfinshed basement measurements. 
# * BsmtUnfSF represents unfinished area of the basement. We will be keeping this as well since it might affect the price of the property. We will err on the side of caution and keep it. 
# * TotalBsmtSF would then just be the addition of our new variable BsmtFinSF and BsmtUnfSF(unfinished sqaure area of the basement). We will be keeping this for now. 
# 
# **All in all: **
# 
# * We will drop: BsmtExposure, BsmtFinType1, BsmtFinType2. 
# * We will add BsmtFinSF1 and BsmtFinSF2 to get a new variable called BsmtFinSF:
# * We will keep: 
# 
#     **BsmtQual, BsmtCond(Their empty values will be replaced by None)**
#     
#     **TotalBsmtSF, BsmtFinSF1, BsmtFinSF2,BsmtUnfSF, BsmtFullBath, BsmtHalfBath (Their empty values will be replaced by 0).**

# In[ ]:


all_data = all_data.drop(['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'], axis = 1)


# In[ ]:


all_data.shape  #The second dimension dropped from 72 to 69 as expected. 


# In[ ]:


for col_n in ('BsmtQual', 'BsmtCond'):
    all_data[col_n] = all_data[col].fillna('None')
    
for col_0 in ('TotalBsmtSF','BsmtFinSF1' , 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF'):
    all_data[col_0] = all_data[col_0] .fillna(0)


# ## Next up are Masonry features. The features in question are :
# 
# * MasVnrType = Masonry Veneer Type. The missing values here will be replaced with None.
# * MasVnrArea = Masonry Veneer Area in sq.ft. The missing values here will be replaced with 0. 
# 
# *Courtesy of Serigne*
# 

# In[ ]:


all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)


# ## Next up is MSZoning:
# 
# * which represents the general zoning classification on the sale.
# * 'RL' is by far the most common value. So we can fill in missing values with 'RL'. RL means Residential, low-density.
# 
# *Courtesy of Serigne*
# 

# In[ ]:


all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])


# ## Moving on to utilities
# 
# * For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
# 
# *Courtesy of Serigne*

# In[ ]:


all_data = all_data.drop('Utilities', axis =1 )


# ## Next feature is Functional.
# * This shows home functionality. 
# * Data description means NA means Typical or "Typ.
# 
# *Courtesy of Serigne.*

# In[ ]:


all_data["Functional"] = all_data["Functional"].fillna("Typ")


# ## Next up, we have Exterior1st and Exterior2nd:
# * Exterior 1 represents the exterior covering on the house and Exterior 2 represents if there is more than one covering.
# * Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
# 
# *Courtesy of Serigne* 

# In[ ]:


all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])


# ## Next up is SaleType. 
# 
# * We will just replace this with the most common sale type which is WD-Warranty Deed Conventional. 
# 
# *Courtesy of Serigne*

# In[ ]:


all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


# ## Next up is Electrical. 
# 
# * Represents Electrical System. 
# *  It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
# 
# *Courtesy of Serigne*

# In[ ]:


all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])


# ## KitchenQual is up next...
# 
# * Represents Kitchen Quality.
# * Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.
# 
# > *Courtesy of Serigne* 

# In[ ]:


all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])


# # ...let's see if we missed any??
# 

# In[ ]:


#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# ## Let's now take care of the categorical variables and convert them to numerical values. (Taken from [Serigne's Notebook](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard) since he had already done this and it would speed things up for me

# In[ ]:


#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# #### I changed some of the values in cols from that of Serigne's list since I dropped and kept different values than he did. 

# In[ ]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond',  'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual',  
        'Functional', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))


# ### Remember earlier when we talked about using different basement features to create a total surface area, we are going to do that. 
# 
# **Since there isn't a variable called 2ndFlrSF which implies that all these houses/properties have a basement and the first floor. **
# 
# **So, we will add the Total Basement surface area and the 1st floor surface area to get the total surface area of the house/property.**

# In[ ]:


# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# In[ ]:


# all_data.dtypes

# all_data['MSZoning']


# # We did it!!! We have finished cleaning up the dataset and doing some feature Engineering!
# 
# ![Seth](https://media.giphy.com/media/229OX0vSVNys10AZnS/giphy.gif)

# ### Now Recall the 10 most powerfully correlated variables with SalePrice.
# 
# 1. OverallQual  = Rates the overall material and finish of the house
# 2. GrLivArea    = GrLivArea: Above grade (ground) living area square feet
# 3. GarageCars   = Size of Garage in car capacity 
# 4. GarageArea   = Size of garage in square feet
# 5. TotalBsmtSF  = Total square feet of basement area
# 6. 1stFlrSF     = First Floor square feet
# 7. FullBath     = Full Bathrooms Above Grade.
# 8. TotRmsAbvGrd = Total rooms above grade (does not include bathrooms)
# 9. YearBuilt    = Original Construction Date.
# 
# ** * We have already dropped GarageCars because it was a redundant variable when we have GarageArea. **
# 
# ** * We also dropped YearBuilt because it seems like an extraneous variable (my opinion). **
# 
# ### That leaves us with:
#     1. OverallQual          5. 1stFlrSF
#     2. GrLivArea            6. FullBath
#     3. GarageArea           7. TotRmsAbvGrd
#     4. TotalBsmtSF          
# 
# ### So really we have only 7 variables left that share strong correlation with SalePrice. Everything else is less stronger than the above 7 variables. 
# 
# ### We are going to try to work with and around them to accentuate(if possible) and construct a few more variables that might be better predictors of SalePrice.
# 
# ### To these, seven we will add some of the features which we cleaned up in our last step. The features we will be including are:
# 
# **Hot Features:** These are the features that have the strongest correlation with SalePrice. 
# 
# **Basement based features:** These are stored in the list below called bsmt_feats. 
#     Since area related features are very important to determine house prices, we add one more feature which is the total area of basement, first and second floor areas of each house (*Courtesy Serigne*)
# 
# 

# In[ ]:


# Let's create a list of the features that we are interested in and add sales price to that as well
hot_feats_list = ['GarageArea', 'GrLivArea', 'OverallQual', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd']
bsmt_feats = ['BsmtQual', 'BsmtCond', 'TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath',
              'TotalSF']
misc_feats = ['Functional', 'KitchenQual']
# target_feat= ['SalePrice']


# In[ ]:


# all_data['BsmtCond']


# In[ ]:


all_data.shape


# ### Sooo... 
# 
# * We started from 81 variables and now we are down to 69 variables. Time to put our thinking hats on and really pick the varibles we will be using to train our model. 
# 
# ![](https://media.giphy.com/media/3diE2vbIarCXWXaN15/giphy.gif)
# 
# 

# ## Let's think this thing out aloud. 
# 
# * First thing first, we are keeping the 7 features we discussed earlier which have high correlation with the SalePrice. So they are 
#   **GarageArea,GrLivArea,OverallQual, 1stFlrSF, FullBath, TotRmsAbvGrd**, **TotalBsmtSF**
# 
# * Since **BsmtQual**(height of the basement) and **BsmtCond**(the overall condition of the basement) are covering two separate and distinct features of the basement,  I will leave them in. I have wrestled with dropping **BsmtCond** becuase it is a subjective evaluation. For the first iteration I will keep it. I might try dropping it later to see if the model results improve
# 
# * However, I will be dropping **BsmtFinSF1** and **BsmtFinSf2** as well as **BsmtUnfSF** since their combined value is already captured in the variable **TotalBsmtSF.** 
# 
# * I will be keeping **Functional** which measure the functionality of the house/property and **KitchenQual** which seems like it would have an obvious effect on the price of the house. 
# 
# 
# ## Sooo..I will be eliminating another three features from the dataset. These have to deal with the finished and unfinished basement surfaces. 
# 
# ## This is my first attempt at training this model, so I will be keeping other features in the dataset. Since a  lot of them are not as strongly correlated with SalePrice, I will be erring on the side of caution and keep them rather than not keep them. 
# 
#   
# 

# In[ ]:


# (all_data['LowQualFinSF']!= 0).values.sum()

all_data.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], axis = 1, inplace = True)


# In[ ]:


all_data.shape


# ### OK, so we just came down to 66 variables from out starting point of 81 variables. Not bad!!! 
# 
# ### Let's train some regression models and make some predictions!!!
# 
# ### Before, we do that we need to handle the categorical variables. Luckily for us Pandas has a library to do just that called *get_dummies*. This converts categorical values to dummy variables for us to use in our models
# 
# Once again, Courtesy of [Serigne's extremly well done work!!](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)

# In[ ]:


all_data = pd.get_dummies(all_data)
print(all_data.shape)


# ** Set new training and testing sets **

# In[ ]:


train = all_data[:ntrain]
test = all_data[ntrain:]


# ### Let's import the machine learning libraries we will use. For now I will start small. I will come back and try different models here!!

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rand_for_clf =RandomForestRegressor()
rand_for_clf.fit(train, y_train)


# In[ ]:


pred_price = rand_for_clf.predict(test)
print(pred_price)


# In[ ]:


my_submission = pd.DataFrame({'Id': test_ID, 'SalePrice': pred_price})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




