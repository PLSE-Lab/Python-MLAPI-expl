#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## *About the kernel*
# 
# * Aim of this kernel is to do exploratory data analysis (EDA) of Kaggle competition data - House Prices:Advanced Regression Techniques
# 
# * **I've explained every step in detail without making things complex, to make concepts easy to understand for beginners.**
# 
# * Hope you find this kernel useful.
# 
# * My other kernel : [titanic machine learning from disaster competition](https://www.kaggle.com/cvarun/titanic-survival-a-beginner-s-analysis)

# ## Importing necessary libraries
# 
# As usual, first step is to** import all necessary packages** which we'll be needing in our analysis.

# In[ ]:


#panel data analysis - pandas
import pandas as pd

#numpy for operations related to numpy arrays/series
import numpy as np

#matplotlib & seaborn for visualisations
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
#the above line of code is a magic function, enables us to display plots within our notebooks just below the code.

#maths, stats and stuffs
from scipy import stats


# **UNDERSTANDING THE DATASETS GIVEN**
# 
# * We've been given 3 files, namely "train.csv", "test.csv" & "sample_submission.csv".
# * We will do our analysis on the train data.
# * Based on our analysis, make our predictions for the test data.
# * Generate an output file based on sample_submission data format.

# Reading the train and test datas into separate dataframes, as below.

# In[ ]:


train_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# **INSIGHTS INTO THE DATA**
# 
# * Before begining with any analysis, it's important to have a decent knowledge about the dataset, what type of analysis on which type of data needs to be done, and what our target/objective is.

# In[ ]:


#.columns method gives us name of all the columns in our dataframe
train_df.columns


# In[ ]:


#.dtypes returns a series of all columns and their respective datatype
train_df.dtypes


# In[ ]:


test_df.columns


# In[ ]:


test_df.dtypes


# In[ ]:


#dataframe.info() is like one stop destination for all info on metadata
train_df.info()


# From above, it is clear that 
# * Our train data has numerous columns- 81 in total
# * Range of index is 1460 (this is the max no. of entries, for simplicity we'll take it as the no. of rows)
# * With several columns missing many entries (null values). For eg. looking at Garage Type, it has 1379 non-null entries, meaning 81(1460-1379) null values.
# * Different columns are of different datatypes, here we get a glance of which is numerical and which is categorical feature just by looking at the name of column and its data type.

# In[ ]:


test_df.info()


# **OUR OBJECTIVE**
# 
# * As seen in above output of **test_df.info()**, there is only 1 column missing that is the "SALE PRICE" column
# * Our very obejctive is to predict that "SALE PRICE" column for the test data, based on our analysis on the train data.
# * Hence, SALE PRICE is our target in this analysis.
# * Our analysis majorly includes finding out, which all features (columns) have a relation to/affect the SALE PRICE column, and using those features in our prediction, eliminating rest of the columns in the process.

# ## **BEGIN ANALYSIS....**

# **BASIC Analysis**

# In[ ]:


# by default it displays the first 5 records, any other integer can be specified within the parens as I did, giving 6.
train_df.head(6)


# In[ ]:


test_df.head(6)


# **NUMERICAL TYPE DATA**
# 
# * Simple definition : that data which is expressed as numbers, such data have meaning on measurement, eg : height, weight, age, income, stock price etc. 
# 
# * Numerical data is further split as discrete numerical & continuous numerical.
# 
#     * Discrete numerical data takes on values that can be counted, the values are distinct, separate & the list/range of values it can take may be finite(1,2,3,4) or may go upto infinite(1,2,3,4.....). eg: no of people living in a colony.
#     * Continuous numerical data can not be counted, only measured and can be described using intervals. eg: temperature of a given place, over a period.

# It is a good practice to self identify columns with numerical data, looking at the values, data types. If not pandas provides us with everything.

# In[ ]:


#describe gives us a stats about every NUMERICAL column.
train_df.describe()


# The above is a short statistical summary of various numerical columns from the dataframe. Looking at the SALE PRICE, we get several info on it, such as :
# 
# * mean sale price is 180921.19
# 
# * sale price has a std. deviation of 79442.50. Std. deviation denotes dispersion among values, i.e by how much the values differ from the central tendency.
# 
# * min. sale price is 34900
# 
# * max is sale price is 755000
# 
# 

# **CATEGORICAL DATA**
# 
# * The type of data which can be categorised, or the values of which can be fit into different groups/categoris is called categorical data.
# 
# * eg: gender column, where there are only 2 categories, male-female.

# In[ ]:


#using select_dtypes to select specified data type
train_df.select_dtypes('object').describe()   #this code gives a short statistical summary of categorical data. # object means string type here


# The above summary of categorical vars, gives us several imp info, such as :
# 
# * Take for eg: the LotShape column:
# 
#     * It has 4 unique vales (displayed below)
#     
#     * With "Reg" being the top shape, which occurs 925 times out of 1460

# In[ ]:


# unique () used to get all distinct values from columns
train_df.LotShape.unique()


# In[ ]:


# value_counts() is even better, gives distinct values with their frequencies.
train_df.LotShape.value_counts()


# ## **INSIGHTS USING DATA VISUALISATION**

# Visualising distribution of data is best done using histograms. So what is a histogram ?
# 
# * Histogram forms bins across the range of data, and plots bars which represents the no. of data that fall within each of those bins formed.

# In[ ]:


num_col=train_df.select_dtypes(exclude='object')


# In[ ]:


for i in num_col.columns:
    num_col[i].plot.hist(bins=40,color=('r'))
    plt.xlabel(i)
    plt.show()


# * The above plots are all histograms.
# 
# 
# * Looking at the above plots, it is now even clear which all feature are continuous and which ones discrete.
# 
# 
# * eg: look at histogram plot of YearBuilt, BsmtUnfSF are continuous in range, while OverallQual, BsmtHalfBath are discrete.
# 
# 
# * NOTE : Some of the features though may have numerical values, may come under categorical.
# 

# **Now, finding the relation of features with our target - Sale Price**
# 
# * This involves, plotting scatter plots, between our target(Sale Price) and other predictor features(other than Sale Price).
# 
# * And to analyse whether the said predictor feature has any relation to our target.
# 
# * Before that, there is way to see, if the predictor fearures are related to the target-SALEPRICE. This can be visualised using correlation heatmap.

# In[ ]:


# the .corr() method helps compute correlation of columns with each other, it excludes nulls automatically
corrmap=train_df.corr()


# In[ ]:


# .corr(), returns a correlation matrix, which is displayed below, entries are correlation values
corrmap


# * The above correlation matrix, is really insightful but is a bit lengthy to analyse.
# 
# * But, it can be understood easily using visualisation.
# 
# * We shall now, plot the above corr. matrix using heatmap.

# In[ ]:


# below code is to get those features which have correlation greater than 0.5 with Target-SalePrice

best_corrd=corrmap.index[abs(corrmap['SalePrice'])>0.5] #-ve corr. value means they are correlated but inversely, still we need 'em, hence abs()
print(best_corrd)


# In[ ]:


plt.figure(figsize=(12,12))
sns.heatmap(train_df[best_corrd].corr(),annot=True,cmap='RdYlBu')


# **There is our correlation heatmap.. Now analysing it**
# 
# * As color of the cell shifts towards blue, means it is highly correlated compared to others, to target (also indicated by the value in it)
# 
# * Looking at the map, we see :
# 
#     * *OverallQual*, *TotalBsmtSF*, *GrLivArea*, *GarageArea* are highly correlated to our target-SalePrice.
#     
#     

# In[ ]:


# all in one plot - the seaborn pairplot
col= ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_df[col],height=1.9,aspect=0.99,diag_kind='kde')  # for each facet -> width=aspect*height & height is in inches


# The above pairplot is a informative one :
# 
# * We can analyse whether the target is normally distributed. In our case it is not, SalePrice is skewed.
# 
# 
# * Distribution of predictor variables themselves.
# 
# 
# * We can observe if our target-the SalePrice is linearly related to our predictor variables.
# 
#     * This linear relationship is clearly visible across most variables we plotted.
# 
# 
# * Not only that, we can also observe MULTI-COLLINEARITY, i.e. if the the predictor variables being correlated among themselves.
# 
#     * Eg: plot between ( GrLivArea and TotalBsmtSF ), ( OverallQual and GrLivArea ) likewise.

# **CHECKING RELATIONSHIP OF SAID FEATURES TO TARGET INDIVIDUALLY**

# In[ ]:


f,ax=plt.subplots(4,2,figsize=(15,10))    #method of matplot, allows to plot 2 or more graphs in same figure

#plotting OverallQual with SalePrice.. here both lineplot and barplot of OverallQual are plotted on same graph
sns.lineplot(train_df['OverallQual'],train_df['SalePrice'],ax=ax[0,0])
sns.barplot(train_df['OverallQual'],train_df['SalePrice'],ax=ax[0,0])

#plotting TotalBsmtSF with SalePrice..
sns.scatterplot(train_df['TotalBsmtSF'],train_df['SalePrice'],ax=ax[0,1])

#plotting 1stFlrSF with SalePrice..
sns.scatterplot(train_df['1stFlrSF'],train_df['SalePrice'],marker="*",ax=ax[1,0])    #marker is the shape of the points on scatterplot

#plotting GrLivArea with SalePrice..
sns.scatterplot(train_df['GrLivArea'],train_df['SalePrice'],marker="+",ax=ax[1,1])

#plotting GarageCars with SalePrice..
sns.lineplot(train_df['GarageCars'],train_df['SalePrice'],ax=ax[2,0])
sns.barplot(train_df['GarageCars'],train_df['SalePrice'],ax=ax[2,0])

#plotting GarageArea with SalePrice..
sns.scatterplot(train_df['GarageArea'],train_df['SalePrice'],ax=ax[2,1])

#plotting TotRmsAbvGrd with SalePrice..
sns.barplot(train_df['TotRmsAbvGrd'],train_df['SalePrice'],ax=ax[3,0])

#plotting YearBuilt with SalePrice..
sns.lineplot(train_df['YearBuilt'],train_df['SalePrice'],ax=ax[3,1])

plt.tight_layout()   #this automatically adjusts the placement of the plots in the figure area, without this, the figure labels were overlapping 


# **EXPLANATION OF ABOVE PLOTS**
# 
# * OverallQual has a strong near-linear relation to the SalePrice.
# 
# * The scatter plot of TotalBsmtSF also shows relation to SalePrice, i.e as TotalBsmtSF increases, so does SalePrice.
# 
# * 1stFlrSF, GarageArea, GarageCars & GrLivArea also have a strong correlation to SalePrice.
# 
# * TotRmsAbvGrd also has a linear relation to our target !
# 
# * Looking at YearBuilt-SalePrice plot, though the relation with target is not as strong as the others, still we can see new houses have high sale price compared to older ones (with some exceptions-outliers-explained later).
# 
# ## -------------------------------------
# 
# It is also important to know what the features are, a logical understanding of these features...
# 
# * Looking at GarageCars and GarageArea...
#     
#     * GarageCars is the number of cars that can be fitted into a given GarageArea, so basically they are the same (sum of area under all the cars = area of entire garage)... Hence we won't need both of these features, just one is enough, and we shall **retain GarageCars** since it is strongly correlated to our target than GarageArea.
#     
# * Looking at TotRmsAbvGrd and GrLivArea
#     
#     * Similarly, TotRmsAbvGrd which is Total Rooms above ground, and GrLivArea which is living area above ground, these 2 are also the same, (rooms above ground, living area above ground are ofcourse the same thing !) hence we **retain only GrLivArea**, which has higher correlation to target.
#     
#     
#     

# ## **Outlier Treatment of Highly Correlated Features**
# 
# * Before we proceed, a little bit about outliers.
# 
# 
# * Outliers are the datapoints/observations that lie outside or are far away from the main group of data. (imagine them being the odd one out in the group)
# 
#     * In above plot of SalePrice vs TotalBsmtSF, we see few dots that are away from the main group of dots, these OUTLIERS are beyond 3000 on x-axis, which have to be removed.
#     
#     * Similarly in SalePrice vs GrLivArea, we see few markers away from the main group, they are beyond 4000 on x-axis and need to be removed.
# 
# 
# * Outlier treatment involves dropping those extreme values. 

# In[ ]:


train_df.drop(train_df[train_df.GrLivArea>4000].index, inplace = True)
train_df.drop(train_df[train_df.TotalBsmtSF>3000].index, inplace = True)
train_df.drop(train_df[train_df.GrLivArea>4000].index, inplace = True)
train_df.drop(train_df[train_df.YearBuilt<1900].index, inplace = True)  # we see peak in sale price for few houses pre 1900, hence OUTLIER


# In[ ]:


#post outlier treatment
sns.scatterplot(train_df.GrLivArea,train_df.SalePrice)


# In[ ]:


#post outlier treatment
sns.scatterplot(train_df.TotalBsmtSF,train_df.SalePrice)


# ## **Null Values Treatment**
# 
# * Null values are a hinderance in analysis and prediction, hence needs to be dealt with.
# 
# * Best way is to, replace the nulls with the central tendency or the most representative value of that variable, which may be 
#     * Mean (avg) - for numerical vars.
#     * Median (one at the center) - preferred where data is distorted, has outliers etc. **MEDIAN IS IMMUNE TO OUTLIERS**
#     * Mode (most occuring) - preferred for categorical vars.
#     
# 
# * But, clear knowledge of the data is required before replacing them with central tendency, each of which has its own drawbacks.
# 
# 
# * Since both training and testing data has nulls, instead of treating them separately we'll combine them and treat them after taking out SalePrice out.

# In[ ]:


#extracting our target into separate var
y_train=train_df.SalePrice
train_df.drop(columns=['SalePrice'],inplace=True)


# In[ ]:


train_df.columns==test_df.columns # to show train and test datasets have similar columns, so concat them and treat nulls


# In[ ]:


df_merged = pd.concat([train_df, test_df], axis = 0) #axis=0 to concat along rows ; axis=1 is for columns
df_merged.shape


# In[ ]:


#some vars, though have numerical values but are actually categorical, so convert them
df_merged[['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']] = df_merged[['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']].astype('object')
df_merged.dtypes.value_counts()


# In[ ]:


#columns with missing values
missing_columns = df_merged.columns[df_merged.isnull().any()]
print(missing_columns)
print(len(missing_columns))


# In[ ]:


#to find how many nulls
df_merged[missing_columns].isnull().sum().sort_values(ascending=False)


# * Variables with high number of null values must be dropped.
# 
# 
# * **BUT** , in the description of dataset, it is given that nulls have meaning, eg : PoolQC (decribes pool quality), where NaN in PoolQC means there is NO POOL or 0 pool in the house, which is actually relevant data. Similarly NaN in Fence means there is NO Fence.

# In[ ]:


# impute by "NONE", wherever NaN means absence of that feature in the house

none_imputer = df_merged[['PoolQC','MiscFeature','Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageCond','GarageFinish','GarageQual','BsmtFinType2','BsmtExposure','BsmtQual','BsmtCond','BsmtFinType1','MasVnrType']]
for i in none_imputer.columns:
    df_merged[i].fillna('None', inplace = True)


# In[ ]:


# filling nulls in categorical vars with mode

mode_imputer =  df_merged[['Electrical', 'MSZoning','Utilities','Exterior1st','Exterior2nd','KitchenQual','Functional', 'SaleType']]
for i in mode_imputer.columns:
    df_merged[i].fillna(df_merged[i].mode()[0], inplace = True)  #.mode()[0] because if var. is multimodal, then take the first one


# In[ ]:


# dealing with numericals, filling with median (robust to outliers)

median_imputer = df_merged[['BsmtFullBath','BsmtHalfBath', 'GarageCars', 'MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea','LotFrontage']]
for i in median_imputer.columns:
    df_merged[i].fillna(df_merged[i].median(), inplace = True)


# In[ ]:


print(len(df_merged.columns))
print(df_merged.isnull().any().value_counts())  #checking if any nulls in any columns..


# As seen above, all nulls have been taken care of.

# ## **Normality of Data**
# 
# * Normality is a preferred property in analytics and modelling.
# * There are many reasons for this preference, but to put simply normally distributed data makes our job easier.
# * Normally distributed data is like a bell curve, symmetrical in nature.

# In[ ]:


#checking our target variable
sns.distplot(y_train)
print('SalePrice skew :',stats.skew(y_train))


# * As seen above, our target variable is not normally distributed - it ain't symmetric and also not a proper bell shape as it shows peak (pointy top).
# * If skew between -0.5 and 0.5, it can be taken as near symmetrical data.
# * If skew is in between (-0.5 & -1) or between (0.5 & 1), it is said to be moderately skewed.
# * Where skewness is less than -1 and greater than 1, means highly skewed, such as our target SalePrice which has skew=1.5+
# * However a variable which is not normally distributed or which is SKEWED may be forced to be normal by applying LOG transformation. 

# In[ ]:


# applying log trnsm on saleprice
y_train=np.log1p(y_train)      #remember we imported numpy as np
sns.distplot(y_train)
print('SalePrice skew post transformation:',stats.skew(y_train))


# As seen from above, now our target is forced to follow bell shape by applying log transform also, its skewness is within acceptable range now.

# In[ ]:


#checking skewness of other variables
skewed = pd.DataFrame(data = df_merged.select_dtypes(exclude='object').skew(), columns=['Skew']).sort_values(by='Skew',ascending=False)


# In[ ]:


plt.figure(figsize=(6,13))
sns.barplot(y=skewed.index,x='Skew',data=skewed)


# * As we see lot of vars have skewness more than 1, which means they are highly skewed !
# * Hence we shall transform all such vars. , force them to follow normal distribution by applying log transformation.

# In[ ]:


#filtering numeric vars
df_merged_num = df_merged.select_dtypes(exclude='object')
df_merged_num.head(2)


# In[ ]:


# transforming vars where skew is high
df_trnsfmed=np.log1p(df_merged_num[df_merged_num.skew()[df_merged_num.skew()>0.5].index])

#other vars which have skew<0.5
df_untrnsfmd=df_merged_num[df_merged_num.skew()[df_merged_num.skew()<0.5].index]

#concat them
df_allnums=pd.concat([df_trnsfmed,df_untrnsfmd],axis=1)  #axis=1 coz conact along columns

df_merged_num.update(df_allnums)
df_merged_num.shape


# ## **Encoding of Categorical Variables**
# 
# * We know what categorical vars are.
# * We encode categorical vars, so as to make it understandable to the machine.

# In[ ]:


#filtering only those which are categorical in type
df_merged_cat=df_merged.select_dtypes(exclude=['int64','float64'])
df_merged_cat.head()


# In[ ]:


#encoding the vars using pandas get dummies, get_dummies encodes all cat. vars.
df_dummy_cat=pd.get_dummies(df_merged_cat)


# In[ ]:


#final merging of normalised numerical vars and encoded categorical vars.
df_final_merge=pd.concat([df_merged_num,df_dummy_cat],axis=1)


# In[ ]:


# the above final_merge contains both train & test data(remember we combined both), time to separate them now
df_train_final = df_final_merge.iloc[0:1438, :] # first 1438 rows were train data
df_test_final = df_final_merge.iloc[1438:, :]   #all rows below 1438 were test data
print(df_train_final.shape)
print(df_test_final.shape)
print(y_train.shape)      #our target from train data we separated


# ### Thats all folks !
# 
# 
# * **Now your data is ready to be trained & tested ! Until this step, we performed what is known as pre-processing the data which is a major part of any analysis and prediction.**
# 
# 
# * **The next step would be to import necessary suitable models, fit the training data, evaluate the model and then finally predict the  target i.e SalePrice for the test data.**

# Thank you !
# 
# ~ Varun
