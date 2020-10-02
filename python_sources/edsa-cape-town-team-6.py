#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#House-Prices:-Advanced-Regression-Techniques" data-toc-modified-id="House-Prices:-Advanced-Regression-Techniques-1"><strong>House Prices: Advanced Regression Techniques</strong></a></span><ul class="toc-item"><li><span><a href="#0.-Introduction" data-toc-modified-id="0.-Introduction-1.1"><strong>0. Introduction</strong></a></span><ul class="toc-item"><li><span><a href="#0.1.-Overview" data-toc-modified-id="0.1.-Overview-1.1.1">0.1. <strong>Overview</strong></a></span></li><li><span><a href="#0.2.-Libraries" data-toc-modified-id="0.2.-Libraries-1.1.2"><strong>0.2. Libraries</strong></a></span></li><li><span><a href="#0.3.-Loading-the-Dataset" data-toc-modified-id="0.3.-Loading-the-Dataset-1.1.3"><strong>0.3. Loading the Dataset</strong></a></span></li></ul></li><li><span><a href="#1.-Exploratory-Data-Analysis" data-toc-modified-id="1.-Exploratory-Data-Analysis-1.2"><strong>1. Exploratory Data Analysis</strong></a></span><ul class="toc-item"><li><span><a href="#1.1.-Distribution-summary-statistics" data-toc-modified-id="1.1.-Distribution-summary-statistics-1.2.1">1.1. <strong>Distribution summary statistics</strong></a></span></li><li><span><a href="#1.2.-Correlations" data-toc-modified-id="1.2.-Correlations-1.2.2"><strong>1.2. Correlations</strong></a></span></li><li><span><a href="#1.3.-Outliers" data-toc-modified-id="1.3.-Outliers-1.2.3"><strong>1.3. Outliers</strong></a></span></li><li><span><a href="#1.4.-Missing-Data" data-toc-modified-id="1.4.-Missing-Data-1.2.4"><strong>1.4. Missing Data</strong></a></span></li></ul></li><li><span><a href="#2.-Feature-Engineering-,Data-Cleaning-&amp;-Transforming" data-toc-modified-id="2.-Feature-Engineering-,Data-Cleaning-&amp;-Transforming-1.3"><strong>2. Feature Engineering ,Data Cleaning &amp; Transforming</strong></a></span><ul class="toc-item"><li><span><a href="#2.1.-Imputing-the-remaining-missing-values" data-toc-modified-id="2.1.-Imputing-the-remaining-missing-values-1.3.1">2.1. <strong>Imputing the remaining missing values</strong></a></span></li><li><span><a href="#2.2.-Transforming-categorical-variables" data-toc-modified-id="2.2.-Transforming-categorical-variables-1.3.2">2.2. <strong>Transforming categorical variables</strong></a></span></li><li><span><a href="#2.3.-Label-Encoding-categorical-variables" data-toc-modified-id="2.3.-Label-Encoding-categorical-variables-1.3.3">2.3. <strong>Label Encoding categorical variables</strong></a></span></li><li><span><a href="#2.4.-Feature-Selection" data-toc-modified-id="2.4.-Feature-Selection-1.3.4">2.4. <strong>Feature Selection</strong></a></span></li></ul></li><li><span><a href="#3.-Machine-Leaning-Models" data-toc-modified-id="3.-Machine-Leaning-Models-1.4"><strong>3. Machine Leaning Models</strong></a></span></li><li><span><a href="#4.-Submission" data-toc-modified-id="4.-Submission-1.5"><strong>4. Submission</strong></a></span></li></ul></li></ul></div>

# # **House Prices: Advanced Regression Techniques**

# > ## **0. Introduction**

# >> ### 0.1. **Overview**
# 
# For the regression sprint we have been assigned the Kaggle House Prices challenge. The challenge consists of building a regression model to best predict the house prices of Ames, Iowa. The dataset consist consist of 79 features in total.
# 
# This notebook will start off with performing exploring data analysis. The exploratory data analysis will assist us with the next step which is to perform data cleaning and feature selection.
# 
# Once the data has been preprocessed, we will attempt to fit various regression models which we have been exposed to during this sprint. The first set of regression models will form part of the family of linear regression models. We will finish the modeling section of the notebook by fitting a ensemble notebook model.
# 
# We will then conclude the notebook by discussing the regression model which performed the best in predicting the house prices.

# >> ### **0.2. Libraries**

# In[ ]:


#import some necessary librairies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style='darkgrid', context='notebook', palette='viridis')
sns.despine(top=True,right=True)

from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler,StandardScaler,LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split


# >> ### **0.3. Loading the Dataset**

# In[ ]:


# import the train and test datasets 

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


#  **Inspection of our data**
#  
#  **We wil start off with**
#  - shape : Will show us the number of rows and columns in our dataframe
#  - head() : Will allow us to see the first 5 rows of the dataframe
#  - info  : Will show us the number of columns in dataframe and if we have any missing data 

# In[ ]:


#check the numbers of samples and features for train and test
print("Train data set: %i " % train.shape[0],"samples ",train.shape[1],"features")
print("Test data set: %i " % test.shape[0],"samples ",test.shape[1],"features")


# In[ ]:


# check first 5 rows of train 
train.head()


# In[ ]:


# check first 5 rows of test 
test.head()


# In[ ]:


train.info()


# In[ ]:


test.info()


# **Observation from our data Set**
# 1. Train data set consits of 80 explanatory variables and 1460 rows 
# 2. Test  data set consits of 79 explanatory variables and 1459 rows  
# 3. Our Target variable is the SalePrice which is only present in the train 
# 4. We can see that we have columns that have missing value 
# 5. We have both numeric and catergorical data 
# 6. MSSubClass, YearBuilt, OverallCond and subclass or shown as int64 where by they are catergorical data, they will have to changed to be objects as they assume a linear relationship as int64.

# **Removal of non-variables**
# 
# The id of the data sets are neither a dependent variable (target) or a independent variable (feature). We therefore remove the Id column and store to use later when we want split the combined data set.

# In[ ]:


#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it is unnecessary for the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)


# In[ ]:


#check again the numbers of samples and features
print("train: %i " % train.shape[0],"samples ",train.shape[1],"features")
print("test: %i " % test.shape[0],"samples ",test.shape[1],"features")


# > ## **1. Exploratory Data Analysis**

# **Target Variable**
# 
# The target variable of the dataset is the sales price of the properties. The target variable is the variable we will try to predict and for this reason we will be spending the most time exploring this variable.

# >> ### 1.1. **Distribution summary statistics**
# 
# 
# The pandas describe method is useful for obtaining summary statistics of a variable. The box-and-whisker plot is a graphical depiction of the summary statistics.
# 
# The mean is greater than the median, if this difference is significant it could indicate that the data is skewed postively (skewed to the right). We therefore need to investigate the skewness of the target variable. The box-and-whisker graph shows a few very high observations, which seems like it could be outliers (and which could contribute to the any skewness of the data).
# 
# To further investigate the distribution of the target variable it would be useful to plot a histogram and calculate the kurtosis of the target variable.

# In[ ]:


#seaborn boxplot depciting target variable
fig = sns.boxplot(data=train[['SalePrice']],width = 0.2)
fig.axis(ymax=800000,ymin=0);

print("Median: %f" % train['SalePrice'].median())
train['SalePrice'].describe()


# **Histogram and Probability Plot** 
# 
# In the histogram below the bars represents the actual distribution of the data. The blue line represents a line fitting the curvature of the histogram. Finally the black line represents the curvature of the normal distribution.
# 
# **Probability Plot**
# 
# The probability plot (which is also referred to as the quantile-quantile plot) is a graphical technique used to identify deviations from the normal distribution. The straightline represents the distribution data points from the normal distribution. Data points for a normally distributed data set, should have more obervations plotted in the middle section of the line. A symmetrical distribution (which is a property of the normal distribution) should also have its data points plotted close to the straigthline. Plotted data that deviates from the straight is a visual indicator that the data is not symerically distributed.

# In[ ]:


fig = plt.figure(figsize=(9,4))
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
fig = plt.figure(figsize=(9,4))
res = stats.probplot(train['SalePrice'], plot=plt)

print("kurtosis: %f" % train['SalePrice'].kurtosis())


# **Distribution**
# 
# As  we can see on the histogram the Sales prices are skewed to the right and the Sales Price curvature deviates from the normal distribution. Intuitively this makes sense, since the property prices cannot be less than zero dollars, but the upper bound of property prices are unbounded with only a few properties accounting for sales prices far in excess of mean.
# 
# Kurtosis is a measure of "tailedness". The normal distribution has a kurtosis of 3 units which compares to the Sales Price which has a kurtosis of 6.53 units. Since the kurtosis is greater than three it is deemed to be leptokurtic. Leptokurtic is simply a fancy term to say that the distribution contains more extreme values (outliers) than which is the case of the normal distribution.
# 

# **Target variable transform**
# 
# A common misconception is that the ordinary least square regression method requires the dependent and independent variables to be normally distributed. This however is not true. The OLS regression model and in turn extensions of the model (such as ridge and lasso which is simply the OLS model plus an added penalty term), requires the error terms to be normally distributed.
# 
# OLS based models are calculated by minimizing the sum of the residuals. The residuals are the difference between the predicted values of the calculated formula and the actual dependent variables of your dataset. Since we want the residuals of cheap and and expensive houses to way equally, we decided to log transform the target variable. Another reason why we decided to log transform the target variable is because the predictive power of the team members who did not log transform their target variable were inferior to members who did log transformed their target variable.

# In[ ]:


fig = plt.figure(figsize=(9,4))
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution 
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
fig = plt.figure(figsize=(9, 4))
prob = stats.probplot(train['SalePrice'], plot=plt)

print("kurtosis: %f" % train['SalePrice'].kurtosis())


# **Log transform**
# 
# Now that the target variable has been log transformed the curvature of the target variables distribution mathces more closely that of the normal distribution.
# 

# In[ ]:


#ntrain = train.shape[0]
#ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# >> ### **1.2. Correlations**

# **Correlation Matrix**
# 
# A correlation matrix is a table showing correlation coefficients between each respective variable. A heatmap is a modification of a heatmap where a colour scheme is applied according to respective correlation coefficient in the cell of the matrix. Heatmaps are a visual aid to mannually identify collinear features.
# 
# Collinearity is the case where explainatory features (features which has a strong correlation with the target variable), also have a strong correlation with each other. Including features that are strongly correlated does not improve the accurarcy of regression models. Including collinear variables often causes a superficial improvement in the coefficient of determination (R-squared) and might lead to overfitting.
# 
# We will use the heatmap to identify collinear features and then remove the the feature which has the lowest correlation with the target variable.

# In[ ]:


#Correlation map to see how features are correlated with SalePrice
plt.figure(figsize=[25,12])

corrmat = round(train.corr(),2)
sns.heatmap(corrmat, annot=True,linewidths=0.5,cmap ="YlGnBu" )


# In[ ]:



k=7
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
plt.figure(figsize=(8,8))
sns.set(font_scale=1.25)
sns.heatmap(cm, cbar=True,linewidths=1.5, annot=True, square=True,cmap ="YlGnBu", fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)


# **GarageArea and 1stFlrSF** :The heat map shows that GarageArea and 1stFlrSF features are highly correlated with GarageCars and TotalBsmtSF respectively. Thus the feature with the lowest correlation with SalePrice will be removed, which are features GarageArea and 1stFlrSF.
# 
# **Utilities**: For this feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling therefore we can delete it.

# In[ ]:


#Drop columns 'GarageArea','1stFlrSF','Utilities'
train.drop(['GarageArea','1stFlrSF','Utilities'],axis =1, inplace = True)
test.drop(['GarageArea','1stFlrSF','Utilities'],axis =1, inplace = True)


# ###### **Scatterplot Matrix**
# 
# A scatterplot matrix is a visual aid which depicts the correlations between variables as a scatterplot, as well as the distribution of each selected variable presented on a histogram. For the scatterplot matrix we selected the dependent variable as well as the four variables with highest correlation with Sales Price.
# 
# In the top row of the matrix one can see that all of the selected variables have a positive linear relationship with the Sales Price. Two possible outliers can also be seen for GrLivArea and one outlier for TotalBsmtSF.
# 
# Diagonally across the matrix, one will notice that most of the features roughly approximates a bell shaped distributions.

# In[ ]:


sns.set(style = 'darkgrid', context = 'notebook',palette='viridis')
cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF']
sns.pairplot(train[cols], height =2.5)
plt.show()


# >> ### **1.3. Outliers**
# 
# Linear regression models are rather sensitive to outliers, since it tries to minimize the residuals between predicted and actual observed data. So called outliers have the effect of impacting the fit of the model for only a few observations. To counter this effect, so called outlier datapoints are often removed from the dataset.
# 
# We decided in focussing on the features which has correlation with the target variable that is greater than 0.6. We also decided to only investigate the outliers after having removed collinear features, since we wanted to avoid investigating features which will be dropped.
# 
# 

# In[ ]:


correlTable = pd.DataFrame(train.corr().abs()['SalePrice'].sort_values(ascending=False))
correlTable[correlTable['SalePrice'] > 0.6]


# The above features has the strongest correlation with the target variable. We decided to ignore the "OverallQual" feature, since this feature is categorical in nature.

# **Overall Quality**
# 
# The OverallQual variable is a categorical variable rather than a numerical variable. We therefore decided to plot the OverallQual together with the GrLivArea variable. One can see that the OverallQual variable has a positive linear relationship with the dependent variable. A few outliers are apparent on the scatterplot which we will remove when only comparing GrLivArea against the SalePrice.

# In[ ]:


fig = plt.figure(figsize=(12, 6))
cmap = sns.color_palette("husl", n_colors=10)

sns.scatterplot(x=train['GrLivArea'], y='SalePrice', hue='OverallQual', palette=cmap, data=train)

plt.xlabel('GrLivArea', size=15)
plt.ylabel('SalePrice', size=15)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12) 
    
plt.title('GrLivArea & OverallQual vs SalePrice', size=15, y=1.05)

plt.show()


# **GrLivArea Outliers**
# 
# We decided to deem properties with sized in excess of 4650 units to be outliers and therefore removed them from our dataset.

# In[ ]:


plt.subplots(figsize=(15, 5))

plt.subplot(1, 2, 1)
g = sns.regplot(x=train['GrLivArea'], y=train["SalePrice"], fit_reg=False).set_title("Before")


#Delete outliers
plt.subplot(1, 2, 2) 
train = train.drop(train[(train['GrLivArea']>4650)].index)
g = sns.regplot(x=train['GrLivArea'], y=train['SalePrice'], fit_reg=False).set_title("After")


# In[ ]:


plt.subplots(figsize=(15, 5))

plt.subplot(1, 2, 1)
g = sns.regplot(x=train['BsmtFinSF1'], y=train["SalePrice"], fit_reg=False).set_title("Before")


# **Cars per Garage Outliers**
# 
# We did not find any outliers for the GarageCars feature. 

# In[ ]:


plt.subplots(figsize=(15, 5))

plt.subplot(1, 2, 1)
g = sns.regplot(x=train['GarageCars'], y=train["SalePrice"], fit_reg=False).set_title("Before")


# **Total Basement area**
# 
# We found that removal of GrLivArea outlier features, also removed the outliers for the Total Basement Area features in our dataset.

# In[ ]:


plt.subplots(figsize=(15, 5))

plt.subplot(1, 2, 1)
g = sns.regplot(x=train['TotalBsmtSF'], y=train["SalePrice"], fit_reg=False).set_title("Before")


# **Sale Price**
# 
# We found that out visually that the Sale price >= 13.5 would count as prices in the extreme case and decided to count it as outliers and drop those values.

# In[ ]:


train = train.drop(train[(train['SalePrice']>=13.5)].index)


# >> ### **1.4. Missing Data**

# We will first check what percentage of the data is missing in each of our features in order to know where to perform imputation of the missing data. In order avoid accidentally manipulating a data item from the train data set and not to the test data set or vice versa, we decided to concatenate the two data sets and applying the same manipulations to the combined data set.

# In[ ]:


#Update the target variable dataframe before concatenating the train and test data sets.
y_train = train.SalePrice.values

#Recreate the concatenated All Data variable with the updated train data set
ntrain = train.shape[0]
ntest = test.shape[0]
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

#percentage missing Data 
total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

#show missing values
missing_data[missing_data['Total']>0]


# In[ ]:


#Barplot of missing data
missing_data = missing_data[missing_data['Total']>0]

#Bar charts for missing data
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percent'],palette='plasma')

plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# We also decided to drop the features that have more than 15% of their data missing. Looking at the features that fall under this threshold , we can see that they are not major factors that on avearge people think of when buying, therefore we will not be losing data. These features are also strong candidates for Outliers. 

# In[ ]:


#dealing with missing data
all_data = all_data.drop((missing_data[missing_data['Total'] > 2348]).index,1)
all_data.isnull().sum().max() #just checking that there's no missing data missing...


# In[ ]:


# checking the new percentage of missing Data 
total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(30)


# > ## **2. Feature Engineering ,Data Cleaning & Transforming**

# >> ### 2.1. **Imputing the remaining missing values**

# **none** : For all the features in the for loop we replace NA values with None as the data description says for these features are not there eg:"No basement","No Fence"
# 
# **zero** : missing values are zero for having no basement.
# 
# **mode** : Features that fall under mode, have most frequent values, so it is better to use the mode to fill in the missing data.
# 

# **SaleType** : 'WD' is the most frequent value for this feature. Then we fill in missing values for this feature with the mode ('WD')

# In[ ]:


none = ['Fence','FireplaceQu','GarageType','MasVnrType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MSSubClass']
zero = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','MasVnrArea','GarageYrBlt', 'GarageCars']
mode = ['MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd','SaleType']

#imputing the remaining missing values with most applicable values
for col in none:
    all_data[col] = all_data[col].fillna('None')
for col in zero:
    all_data[col] = all_data[col].fillna(0)
for col in mode:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
    
#Data description says NA means typical 
all_data["Functional"] = all_data["Functional"].fillna("Typ")


# In[ ]:


print("Train:", train.shape)
print("Sales:", y_train.shape)


# Checking if we still have any missing values.

# In[ ]:


#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing percentage' :all_data_na})
print(missing_data.count())


# >> ### 2.2. **Transforming categorical variables**

# **Transforming some numerical variables that are categorical by the description.**

# In[ ]:


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


# >> ### 2.3. **Label Encoding categorical variables**

# **Label Encoding categorical variables that may contain information in their ordering set.**

# In[ ]:



cat_cols = ('FireplaceQu', 'MoSold' ,'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street','CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'ExterCond')

# process columns, apply LabelEncoder to categorical features
for c in cat_cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))


# In[ ]:


# Adding total sqfootage feature ,excluding 1stFlrSF since it was removed due to the correlation 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['2ndFlrSF']


# **Skewed features**
# 
# Since many statistical functions require a distribution to be normal or nearly normal, we will check the skewness of the features. 
# Statistically, two numerical measures of shape , skewness and excess kurtosis ,can be used to test for normality. If skewness is not close to zero, then your data set is not normally distributed.

# In[ ]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(15)


# Thereafter we normalise the ones that fall above 0.75 using Box Cox transformation.
# **Box Cox transformation** helps us eliminate skewness and other distributional features that may affect the analysis process. We use the scipy function boxcox1p which computes the box cox transformation of 1+x.

# In[ ]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])


# **Get dummy variables for all the categorical features.**

# In[ ]:


#Get dummy variables while avoiding the dummy trap
all_data = pd.get_dummies(all_data,drop_first=True)
print(all_data.shape)


# > ### 2.4. **Feature Selection**

# **Splitting data back to the original train and test set**
# 
# Now that we have completed the data manipulation activities we can split the data back into the original train and test data sets.

# In[ ]:


total=all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum()*100/all_data.isnull().count()).sort_values(ascending=False)
#types=train.dtypes
DataMissing=pd.concat([total,percent],axis=1,keys=['total','percentage'])

#show missing values
DataMissing[DataMissing['total']>0]


# In[ ]:


train = all_data[:ntrain]
test = all_data[ntrain:]


print("Train data set: %i " % train.shape[0],"samples ",train.shape[1],"features")
print("Test data set: %i " % test.shape[0],"samples ",test.shape[1],"features\n")

X = train
y = y_train

print("X:", X.shape)
print("y:", y.shape)

print("\nTrain data set: %i " % X.shape[0],"samples ",X.shape[1],"features")
print("Test data set: %i " % test.shape[0],"samples ",test.shape[1],"features\n")


# In[ ]:


X.columns


# > ## **3. Machine Leaning Models**

# ###### **Regression Models**
# 
# For the Regression Sprint we have been exposed to ordinary least squares regression models (OLS regression models) and tree-like models.
# 
# We will start of by analysing our data set by using OLS regression models.
# 
# **Linear Regression Models**
# 
# For multivariate regression are based on minimizing the residuals between the predicted values of the regression models versus the actual observed data.
# Multivariate regression models unfortunately has a weakness for multicollinear independent variables. To counter the OLS weakness for multicollinear features,
# OLS extension models has been developted which has a penalty term.
# 
# **Multivariate OLS regression**
# 
# We will start off by fitting a multivariate regression model since it is the simplest of the regression models which we received training on.

# ###### **Train Test Split**
# 
# Kaggle does not provide us with the actual Sales Price values of the test set. We however still need a test set to test our predicted values against. We will therefore use the provided train data set and split this data set into a train and test set. The dataframes created from the train data set will all be named with leading "x_" and "y_" characters to differentiate it from the original train and test set as provided by Kaggle.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,shuffle=False)


# ###### **Multilinear Regression**

# In[ ]:


#Create regression object
lm = LinearRegression()

#Fit model
lm.fit(X_train, y_train)

#Y intercept of regression model
b = float(lm.intercept_)

#Coefficients of the features
coeff = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])


# ###### **Multivariate Regression Coefficients**
# 
# One of the nice features of linear regression models are that they provide coefficient weights that can be interpreted by the user.

# In[ ]:


# Plot important coefficients
fig = plt.figure(figsize=(8,6))
coefs = pd.Series(lm.coef_, index = X_train.columns)
print("Multivariate OLS Regression picked " + str(sum(coefs != 0)) + " features and eliminated the other " +        str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")

plt.title("Coefficients in the Multivariate OLS")
plt.show()

print("y-intercept:", b)


# ###### **Test Multivariable Regression Model**
# 
# The R-squared value of multivariate regression model on the test set can be interpreted as the model being able to explain up to 90,92% of the variance of the dependent variable based upon the selected independent variables. However it is well known that OLS models inflate the R-squared value when multicollinearity exist between the independent variables. This is one of the reasons why extensions models such as Ridge and Lasso were developted.

# In[ ]:


from sklearn import metrics

predictedTrainPrices = lm.predict(X_train)
trainR2 = metrics.r2_score(y_train, predictedTrainPrices)
predictedTestPrices = lm.predict(X_test)
testR2 = metrics.r2_score(y_test, predictedTestPrices)

print("Trained R-squared: ",trainR2)
print("Test R-squared: ",testR2)


# ###### **Residual Plot**
# 
# A residual plot is a graph that shows the residuals on the vertical axis and the independent variable on the horizontal axis. They are used as a graphical analysis for regression models to detect nonlinearity, outliers and to check if the errors are randomly distributed. For a good regression model we expect that the errors are randomly distributed and the residuals should be randomly scattered around the centerline. Any patterns in the residual plot means that the model is unable to capture some explanatory information.
# The residual plot below shows a fairly random pattern. Quite a few of the residuals deviates far from the centerline, which might indicate some outliers. Lets see whether we can obtain a better fit with some other regression model.

# In[ ]:


# Plot residuals
sns.scatterplot(x=predictedTrainPrices, y=predictedTrainPrices - y_train, palette=cmap, data=train,label = "Training data")
sns.scatterplot(x=predictedTestPrices, y=predictedTestPrices - y_test, palette=cmap, data=test,label = "Test data")
plt.title("Multivariate Regression")
plt.xlabel("Predicted SalePrice values")
plt.ylabel("Residuals")

plt.legend(loc = "best")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()


# ###### **Regularization OLS models for collinearity**
# 
# Ridge and Lasso regressions are examples of OLS extension based models that adds a penalty term to counter the weakness of the ordinary least squares method has for collinear independent variables. Both models requires all features to be regulatrized, since the penalty term is proportional to the size of the feature.
# 
# Regularization is the process whereby constraints are added to the size of coefficients related to each other. Regularization requires all independent variables to be rescaled. Rescaling is a process whereby the range that a variable are measured in is standardised for all variables. We will using z-scores for rescaling of our independent variables. We will see that the due to the rescaling of features the interpretation of the Ridge and Lasso models will be slightly different. Rescaling will be done by making use of the Scikit learn library.

# ###### **Ridge Regression**
# 
# Ridge regression adds a penalty parameter which is applied to the squares of the coefficients. As the size of the coefficients increase, so to will the size of the penalty increase, which in turn leads to the shrinkage of the size of the coefficients.
# 
# Before fitting the Ridge Regression model, we will first need to scale the features. This we will do by using the scale object from the Scikit library.

# In[ ]:


scaler = StandardScaler()


# In[ ]:


X_scaled = pd.DataFrame(scaler.fit_transform(X.values),columns = X.columns)


# ###### **Train Test Split**
# 
# We will need to recreate the train and test data set, since the dependent variables has now been scaled.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,test_size=0.20,shuffle=False)
X_train = pd.DataFrame(X_train)


# In[ ]:


#Create ridge object
ridge = Ridge()

#Fit Ridge model
ridge.fit(X_train, y_train)


#Y intercept of regression model
b = float(lm.intercept_)

#Coefficients of the features
coeff = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])


# ###### **Ridge Regression Coefficients**
# 
# Regularised models are interpreted slightly different that non regularised regression models.
# 
# For non regularised OLS regression models, the y-intercept is the value of the target variable if values of all features are set to zero. Now, since the standardized features are all centered around zero, the intercept should now be interpreted as the value of the target variable if all independent variables have values equal to their mean.
# 
# Coefficients were previously interpreted as the expected change in the target variable given an increase of 1 unit in the independent variable value. Coefficients should now be interpreted as the expected change in target variable given an increase of 1 in the scaled independent variable.

# In[ ]:


type(X_train)


# In[ ]:


# Plot important coefficients
fig = plt.figure(figsize=(8,6))
coefs = pd.Series(ridge.coef_, index = X_train.columns)
print("Ridge Regression picked " + str(sum(coefs != 0)) + " features and eliminated the other " +        str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Ridge Regression")
plt.show()

print("y-intercept:", b)


# ###### **Test Ridge Regression Model**
# 
# The R-squared value has slightly improved compared to what it was for the multivariate OLS regression model.

# In[ ]:


predictedTrainPrices = ridge.predict(X_train)
trainR2 = metrics.r2_score(y_train, predictedTrainPrices)
predictedTestPrices = ridge.predict(X_test)
testR2 = metrics.r2_score(y_test, predictedTestPrices)

print("Trained R-squared: ",trainR2)
print("Test R-squared: ",testR2)


# ###### **Residual Plot**
# 
# The residual plot for the Ridge regression seems almost like an exact copy of the multivariate regression model's residuals plot.

# In[ ]:


# Plot residuals
sns.scatterplot(x=predictedTrainPrices, y=predictedTrainPrices - y_train, palette=cmap, data=train,label = "Training data")
sns.scatterplot(x=predictedTestPrices, y=predictedTestPrices - y_test, palette=cmap, data=test,label = "Test data")
plt.title("Ridge Regression")
plt.xlabel("Predicted SalePrice values")
plt.ylabel("Residuals")

plt.legend(loc = "best")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()


# ###### **Lasso Regression**
# 
# Lasso is able to perform both subset selection as well as shrinkage of coefficients of the features.
# 
# Lasso can be seen as a modification of the ridge regression. Where ridge regression applies the penalty parameter to the sum of the squares of the coefficients, Lasso applies the penalty function to **absolute values** of the coefficients.
# 
# Lasso has one parameter to apply to the penalty term which is known as alpha. The optimise the fit of the Lasso model, requires one to find the optimal value for alpha.
# 
# Since Lasso also requires features to be scaled, we will simply reuse the train and test data set that was created for the Ridge Regression.

# In[ ]:



#Create ridge object
lasso = Lasso(alpha=0.01)

#Fit Lasso model
lasso.fit(X_train, y_train)

intercept = float(lasso.intercept_)

coeff = pd.DataFrame(lasso.coef_, X.columns, columns=['Coefficient'])


# ###### **Lasso Regression Coefficients**
# 
# The coefficients of the Lasso should be interpreted similiar to how the coefficients for Ridge regression are interpreted.

# In[ ]:


# Plot important coefficients
fig = plt.figure(figsize=(8,6))
coefs = pd.Series(lasso.coef_, index = X_train.columns)
print("Ridge Regression picked " + str(sum(coefs != 0)) + " features and eliminated the other " +        str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Ridge Regression")
plt.show()

print("y-intercept:", b)


# We observe that the number of features has significantly reduced compared to prior regression models.

# In[ ]:


predictedTrainPrices = lasso.predict(X_train)
trainR2 = metrics.r2_score(y_train, predictedTrainPrices)
predictedTestPrices = lasso.predict(X_test)
testR2 = metrics.r2_score(y_test, predictedTestPrices)

print("Trained R-squared: ",trainR2)
print("Test R-squared: ",testR2)


# ###### **Residual Plot** 
# 
# The residuals for the Lasso regression model seems to lie slightly closer to the centerline.This indicates a slight improvement from the prior to regression models.

# In[ ]:


# Plot residuals
sns.scatterplot(x=predictedTrainPrices, y=predictedTrainPrices - y_train, palette=cmap, data=train,label = "Training data")
sns.scatterplot(x=predictedTestPrices, y=predictedTestPrices - y_test, palette=cmap, data=test,label = "Test data")
plt.title("Lasso Regression")
plt.xlabel("Predicted SalePrice values")
plt.ylabel("Residuals")

plt.legend(loc = "best")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()


# ###### **Test Lasso Regression Model**
# 
# The R-squared value has not changed much from what it was for Ridge regression, but the numbder of features has dropped by a lot. Note that we have not yet attempted to find the optimal alpha for 
# lasso model.

# ###### **Improving Lasso alpha**
# 
# Let us try to improve the test R-squared by playing a bit with the alpha parameter.

# In[ ]:


#Dataframe which will be used to store resulsts of model fits
LassoResultTable = pd.DataFrame(columns=['alphaVar', 'TrainR', 'TestR'])

for alphaV in np.arange(0.001, 0.010, 0.001):

    lasso = Lasso(alpha=alphaV)
    lasso.fit(X_train, y_train)

    predictedTrainPrices = lasso.predict(X_train)
    trainR2 = metrics.r2_score(y_train, predictedTrainPrices)
    predictedTestPrices = lasso.predict(X_test)
    testR2 = metrics.r2_score(y_test, predictedTestPrices)
    LassoResultTable = LassoResultTable.append(pd.Series([alphaV, round(trainR2,6), round(testR2,6)], index=LassoResultTable.columns ), ignore_index=True)

LassoResultTable


# It seems that for the range we incremented the alpha variable, the highest R-squared value for the test data set is when alpha is equal to 0.004.

# **Ensemble Tree-Like Regression**
# 
# An alternative to ordinary least squares based regression models are Tree-Like regression models. Decision trees is an example of a Tree-Like regression models.
# 
# Decision trees are created by a process referred to as binary recursive partitioning. Binary recursive partitioning splits a data set into sections referred to as partitions or branches. The process of spliting the data into partitions continues until some criteria has been met such as a predefined number of splits or the minimizing of the error function.
# 
# The error function, which decision trees try to minimize, is the sum of the squared deviations from the mean in the two separate partitions. If the value of an error function has been found to be zero, then a terminal node has been found for a section of the decision tree.
# 
# The problem with decision trees are that they tend not to generalize well, which means that they seem to not perform well on data sets which the model was not trained on.
# 
# Random Forests are an ensemble method which can be seen as an extension of decison trees, but which generalizes better to unseen data. In the context of Random Forest, ensemble refers to using multiple  decision trees. Random Forest then predicts target values by deciding on the value for which the majority of the decision trees has predicted as being the target variable value.

# **Random Forests**
# 
# Initializing a Random Forests object requires three parameters to be specified which are:
# - n_estimators 
# - Criteria
# - Random state
# 
# n_estimators refers to the number of trees in each "forest". It is worth mentioning that as the number of trees increases the computational complexity increases. In other words as the number of trees increases resources required from the executing computer increases or may even exceed the capabilities of computer being used. Generally as the number of trees (n_estimators) increases the accucracy initially increases untill it starts to plateau on some level. We decided to create a loop using three variables to test fit of the model.
# 
# Criteria refers to which error function should be minimized. We will be using the mean squared error which is commonly also used in OLS based models.
# 
# Random State is the seed provided to create a random variable on which are used to create the trees of the forest. This variable requires to be specific if
# you want reproduce the same result.

# In[ ]:


#Dataframe which will be used to store resulsts of model fits
RandomForestResultTable = pd.DataFrame(columns=['alphaVar', 'TrainR', 'TestR'])
estimatorsList = [10,50,100]

for var in estimatorsList:
    
    RFregressor = RandomForestRegressor(n_estimators = var,criterion = 'mse', random_state =0)
    #Whether features are regularized or not does not impact Random Forests
    RFregressor.fit(X_train,y_train)
    Trainpredict = RFregressor.predict(X_test)

    predictedTrainPrices = RFregressor.predict(X_train)
    trainR2 = metrics.r2_score(y_train, predictedTrainPrices)
    predictedTestPrices = RFregressor.predict(X_test)
    testR2 = metrics.r2_score(y_test, predictedTestPrices)
    RandomForestResultTable = RandomForestResultTable.append(pd.Series([var, trainR2, testR2], index=RandomForestResultTable.columns ), ignore_index=True)


# In[ ]:


RandomForestResultTable


# ###### **Random Forest Regression Coefficients**
# 
# One of the drawbacks of Tree-Like based regression models are that they do not provide a list of coefficients, which makes it difficult for the model to provide an explanation for its particullar fit.
# 

# ###### **Residual Plot**
# 
# Despite our Random Forest regression models R-squared value being far inferiour to our OLS based models R-squared values, the residuals of the Random Forests seems to lie a lot closer to the centerline. This might indicate that some of our features variables not having a linear relationship with the target variable.

# In[ ]:


# Plot residuals
sns.scatterplot(x=predictedTrainPrices, y=predictedTrainPrices - y_train, palette=cmap, data=train,label = "Training data")
sns.scatterplot(x=predictedTestPrices, y=predictedTestPrices - y_test, palette=cmap, data=test,label = "Test data")
plt.title("Random Forest Regression")
plt.xlabel("Predicted SalePrice values (n_estimators = 100)")
plt.ylabel("Residuals")

plt.legend(loc = "best")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()


# **Random Forest Results**
# 
# Our Random Forest models failed to outperform our linear regression based models. However we also refrained from using exceptional large values for n_estimators variable, because of the limitations of our computers.

# **Model Selection**
# 
# When performing regression analysis one's decision regarding which regression method should be used will partially dependent on the characteristics of the data set and what one desires to achieve.
# 
# **Requires interpretation**
# 
# Often the reason for performing regression analysis is to gain some insight from once data. Linear regression based models lends itself well for gaining insight, since it produces a list coefficients which can be examined (which Tree-Like models does not produce).
# 
# **Characteristics of the data set**
# 
# Certain models makes assumptions of the characteristics of data set. Linear regressin models tends to make more assumptions of the properties of the data set. Tree-Like models tends to make less assumptions of the underlying data set and can therefore be used on a more diverse set of data sets.
# 
# **Try and see approach:**
# 
# If your only concern is to find a model which performs the best on the test set (which is the case with this specific Kaggle competition), then one can simply try different regression methods and choose the model that performs the best.

# **Why we chose Lasso to submit to Kaggle**
# 
# Our criteria for choosing a regression model was to simply use the regression model which best predicted test set's target variable.
# 
# In our case Lasso ended up being the regression model for which we obtained the best score on Kaggle.

# **Kaggle Submission steps**
# 
# 1. Scale test features set, if regularised regression model has been applied.
# 
# 2. Create predicted target variables using scaled test set.
# 
# 3. If target variables has been log (or any transformation method has been applied), then the transformation has to be "reversed". Which in our case requires us to appply the numpy exponent function to our output target variable set.
# 
# 4. Add Id column back to data set and create submission file
# 

# **Scale test features set**

# In[ ]:


test_scaled = scaler.fit_transform(test)


# **Create predicted target variables**

# In[ ]:


test_lasso = lasso.predict(test_scaled)
test_lasso


# **Revese applied transformations (log of data) to predicted variables**

# In[ ]:


predict = np.exp(test_lasso)


# > ## **4. Submission**

# **Add Id column back to data set and create submission file**

# In[ ]:


#Prepare Submission File
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = predict
sub.to_csv('submit.csv',index=False)


# In[ ]:


sub

