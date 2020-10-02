#!/usr/bin/env python
# coding: utf-8

# **HOUSE PRICE PREDICTION USING XGBOOST**

# In[ ]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from sklearn.metrics import classification_report, mean_squared_error, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold,cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from scipy.special import boxcox1p
from xgboost import XGBRegressor


# Load Data

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# Check the numbers of samples and features.

# In[ ]:


print("Training set size:", train.shape)
print("Test Set size: ", test.shape)


# Training set has one extra column than the test set .Check which columns match and which dont. The mismatch is most likely to be because the test set doesnt have the target feature BUT just to be sure...

# In[ ]:


print("Train set columns not  in test set")
print([train_col for train_col in train.columns if train_col not in test.columns ])
print ("Test set columns not in test set")
print ([test_col for test_col in test.columns if test_col not in train.columns])


# Confirmed.....The test set has the SalePrice feature missing. Let take a look at the first few rows of the training and test data sets.

# In[ ]:


#View first 10 rows of data in test and training
print(train.head(10))
print(test.head(10))


# And now a quick statistical summary of the training data...

# In[ ]:


train.describe()


# Save the Id Columns important for submission at the very end.

# In[ ]:


train_ID = train['Id']
test_ID = test['Id']


# Drop The ID Columns from the data sets as they wont add value to our model...

# In[ ]:


train.drop(['Id'], axis = 1, inplace = True)
test.drop(['Id'], axis = 1, inplace = True)


# Lets visualise correlation data . Why is that important ....correlation is a measure of how strongly variables are related.Understandinghow variables are correlated is useful because we can use the value of one variable to predict the value of the other variable. The correlation coefficient ranges from -1.0 to +1.0. The closer it is to +1 or -1, the more closely the two variables are related. Valuea close to 0 mean there is a weak relationship between the variables or none at all. A positive correlation coefficient means that as one variable gets larger the other gets larger and a negative one means that as one gets larger, the other gets smaller. So lets see

# In[ ]:


cormatrix = train.corr()
corplot = plt.subplots(figsize =(15,12))
sns.heatmap(cormatrix, vmin = -1, vmax = 1,cbar = True, square = True, cmap = 'coolwarm')


# Lets take a look at the ten Features Most Related to Sale Price from the correlation matrix we just plotted.

# In[ ]:


k = 10
corr_cols = cormatrix.nlargest(k, 'SalePrice')['SalePrice'].index
cormatrix2 = np.corrcoef(train[corr_cols].values.T)
sns.set(font_scale = 1)
sns.heatmap(cormatrix2,cbar = True, square = True , cmap = 'coolwarm',annot_kws={'size': 10}, annot = True , xticklabels = corr_cols.values, yticklabels = corr_cols.values)


# List most correlated columns

# mostcor = pd.DataFrame(corr_cols)
# mostcor.columns = ['Most Correlated Features']
# mostcor

# The features are described as follows. 
# 
# * OverallQual: Rates the overall material and finish of the house (1 = Very Poor, 10 = Very Excellent)
# * GrLivArea: Above grade (ground) living area square feet
# * GarageCars: Size of garage in car capacity
# * GarageArea: Size of garage in square feet
# * TotalBsmtSF: Total square feet of basement area
# * 1stFlrSF: First Floor square feet
# * FullBath: Full bathrooms above grade
# * TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# * YearBuilt: Original construction date
# * GarageCars and GarageArea are strongly correlated because the more cars that fit into a garage the more area there is in the garage.
# * We shall only use one Garage value in our model which is GarageCars since has a higher correlation value to SalePrice
# * OverallQual Has the highest correlation value to SalePrice because the overall quality of a house determines the selling price.
# * TotRmsAbvGrd is strongly correlated to GrLivArea because the area of living area above ground is related or determined mostly by the number of rooms above ground.Only one of these will be used
# * TotalBsmtSF and 1stFlrSF are strongly correlated because its likely that the size in area of the first floor determines the size of the basement.We will use TotalBsmtSF 

# Can we observe anything interesting between the features most correlated to SalePrice with scatterplots.... A scatter plot matrix could be a great idea for these features . Lets see.

# In[ ]:


#ScatterPlot Matrix of most correlated features
sns.set(style = 'ticks', color_codes= True)
pltmatrix = sns.pairplot(train, vars = ['SalePrice','OverallQual', 'GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt'])


# **OUTLIERS**

# One scatterplot looks interesting. From the scatter plot matrix above the matrix for Sale Price and the Living area GrLivArea there are points that are not following the observable trend because they have a large area but low price.These are outliers .
# **An outlier is an observation that lies an abnormal distance from other values in a random sample from a population or data set**
# Data outliers can mislead the training process because some machine learning models are sensitive to range and distribution of values and this can result in poor results and less accurate models...WOMP WOMP .So they have to be deleted.... 

# In[ ]:


#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)


# Lets analyse the target variable Sale Price by looking at its normal distribution plot. A quick note on normal distribution....The normal distribution is a probability function that describes how the values of a variable are distributed. So why are we interested in it....many algorithms in data science assume that the data follows a  normal distibution and make various calculations with this assumption. So the more the data is close to a normal distribution the more it fits the assumption. The errors your model make should have the same variance meaning the difference between predicted values  and the actual values of the target variable should be constant. This can be ensured by making sure that target variable follows a normal distribution. Lets see how SalePrice is distributed

# In[ ]:


sns.distplot(train['SalePrice'] , fit=norm)


# That does not look normal...
# Sale Price is skewed right  (positively skewed) . Skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean . SalePrice will need to be transformed and make it normal distributed. Let's log transform this variable and see if this variable distribution can get any closer to normal.We use the numpy fuction log1p which  applies log(1+x) to all elements of the column.
# 

# In[ ]:


train["SalePrice"] = np.log1p(train["SalePrice"])


# Lets see if it worked . 

# In[ ]:


sns.distplot(train['SalePrice'] , fit=norm);


# **Features engineering** <br>
# Let's join the train and test data into one dataframe to transform the datasets simultaneously

# First lets save the number of observations in the train and test set for when we have to split the concatenated data set .SalePrice is assigned a variable and then dropped from the joined data sets. 

# In[ ]:


train_num = train.shape[0]
test_num = test.shape[0]
y_train = train.SalePrice.values


# In[ ]:


data = pd.concat((train, test)).reset_index(drop=True)
data.drop(['SalePrice'], axis=1, inplace=True)
print("Concatenated dataframe size is :", (data.shape))


# **MISSING DATA**

# Lets take a look at how much missing data is in the data set . 

# In[ ]:


#sum missing data calculate percentage missing in each column 
missing_df = pd.DataFrame({'Total':train.isnull().sum(), 'Percentage':(train.isnull().sum())/1460*100})
missing_df = missing_df.sort_values(by = 'Total',ascending = False)
missing_df = missing_df.loc[missing_df['Total'] > 0]
print(missing_df)


# Lets see what that looks like with the aid of a bar plot.

# In[ ]:


f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=missing_df.index, y=missing_df['Percentage'].values)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# **PoolQC** has the highest missing values. From the description NA = the house has no pool and because most houses have no pool (99%) we will drop the column
# 
# **MiscFeature** : data description says NA means "no misc feature"
# Since approximately 96% of house dont have a misc feature the coulumn  will be dropped
# 
# Most houses have no **alley access** and so this feature will be dropped as it wont be helpful 
# with predictions. The same logic is applied to the **fence**  feature which has 80% missing values
# 

# In[ ]:


for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence'):
    data.drop([col], axis = 1, inplace = True)


# From correlation matrix of features most related to SalePrice : drop columns mostly correlated to other features.

# In[ ]:


data.drop(['GarageArea','TotRmsAbvGrd', '1stFlrSF'], axis = 1, inplace = True)


# **LotFrontage** : Most houses in the same neighbourhood have the same size lot frontage area so we will fill in missing values by the median LotFrontage of the neighborhood.

# In[ ]:


##Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median())) 


# **GarageType, GarageFinish, GarageQual and GarageCond** : Replacing missing data with None

# In[ ]:


for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    data[col] = data[col].fillna('None')


#  **GarageYrBlt, GarageArea and GarageCars** : Replacing missing data with 0 
# (Since No garage = no cars in such garage.)

# In[ ]:


for col in ('GarageYrBlt', 'GarageCars'):
    data[col] = data[col].fillna(0)    


# **BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath** :
# missing values are likely zero for having no basement
# 

# In[ ]:


for col in ('BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    data[col] = data[col].fillna(0)   


# In[ ]:


for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    data[col] = data[col].fillna('None')


# In[ ]:


data["FireplaceQu"] = data["FireplaceQu"].fillna("None")


# **MasVnrArea and MasVnrType** : NA most likely means no masonry veneer 
# for these houses. We can fill 0 for the area and None for the type.
# 

# In[ ]:


data["MasVnrType"] = data["MasVnrType"].fillna("None")
data["MasVnrArea"] = data["MasVnrArea"].fillna(0)    


#  **MSZoning (The general zoning classification)** :  'RL' is by far  
# the most common value.  So we can fill in missing values with 'RL' 
# 

# In[ ]:


data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])


#  **Utilities** : For this categorical feature all records are "AllPub", except for one "NoSeWa"  
# and 2 NA . Since the house with 'NoSewa' is in the training set, **this feature won't help in predictive modelling**. We can then safely  remove it.

# In[ ]:


data.drop(['Utilities'], axis=1, inplace = True)


# **Functional** : data description says NA means typical

# In[ ]:


data["Functional"] = data["Functional"].fillna("Typ")


# **Electrical** : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.'''
# 

# In[ ]:


data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0], inplace = True)


#  **KitchenQual**: Only one NA value, and same as Electrical, we set 'TA' 
# (which is the most frequent)  for the missing value in KitchenQual.

# In[ ]:


data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])


# **Exterior1st and Exterior2nd** : Again Both Exterior 1 & 2 have 
# only one missing value. We will just substitute in the most common string

# In[ ]:


data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])


# **SaleType** : Fill in again with most frequent which is "WD"

# In[ ]:


data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])


# Lets look at **Numerical** Features in tha data set

# In[ ]:


data.select_dtypes(include=['int64','float64']).columns


# And now the Categorical Features...

# In[ ]:


data.select_dtypes(include=['object']).columns


# Even though MSSubClass, MoSold, YrSold, YearBuilt, YearRemodAdd , and GarageYrBlt have numerical values they are categorical features and need to be transformed.

# In[ ]:


#MSSubClass 
data['MSSubClass'] = data['MSSubClass'].apply(str)


# For MoSold  replace month number with month name. Year and month sold are transformed into categorical features.

# In[ ]:


data['YrSold'] = data['YrSold'].astype(str)
data['MoSold'] = data['MoSold'].astype(str)


# There are ordered categorical features categorical values . These need to be transformed to numerical features with the values changed according to their order . For example ExterQual which is an evaluation of the quality of the material on the exterior <br>
#     Ex	Excellent  <br>
#    Gd	Good<br>
#    TA	Average/Typical<br>
#    Fa	Fair<br>
#    Po	Poor<br>
#    The number of unique values in each column are counted and numerical values are mapped according to the order of the categorical values.

# In[ ]:


#Counting unique values in each of the categorical features with ordered categorical values
for col in ('ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond'):
    print(col)
    print(data[col].value_counts())


# In[ ]:



map1 = {'Po': 1, 'Fa': 2, 'TA': 3,'Gd': 4, 'Ex': 5}

for col in ('ExterCond', 'HeatingQC'):
    data[col] = data[col].map(map1)


# In[ ]:


map2 = {'None': 1, 'Po': 2, 'Fa': 3, 'TA': 4,'Gd': 5, 'Ex': 6}

for col in ('GarageCond', 'GarageQual','FireplaceQu'):
    data[col] = data[col].map(map2)


# In[ ]:


map3 = {'None':1, 'Unf': 2, 'LwQ': 3, 'Rec': 4, 'BLQ': 5,'ALQ': 6, 'GLQ': 7}

for col in ('BsmtFinType2', 'BsmtFinType1'):
    data[col] = data[col].map(map3)


# In[ ]:


map4 = { 'Fa': 1, 'TA': 2,'Gd': 3, 'Ex': 4}

for col in ('ExterQual', 'KitchenQual'):
    data[col] = data[col].map(map4)


# In[ ]:


map5 = {'None': 1, 'Fa': 2, 'TA': 3,'Gd': 4, 'Ex': 5}
data['BsmtQual'] = data['BsmtQual'].map(map5)  


# In[ ]:


map6 = {'None': 1,'Po': 2, 'Fa': 3, 'TA': 4,'Gd': 5,}
data['BsmtCond'] = data['BsmtCond'].map(map6) 


# In[ ]:


map7 = {'None': 1,'No': 2, 'Mn': 3, 'Av': 4,'Gd': 5,}
data['BsmtExposure'] = data['BsmtExposure'].map(map7) 


# The rest of the categorical features need to be transformed to numerical and this is achieved by using dummy variables.

# In[ ]:


cat_cols = ['BldgType', 'CentralAir', 'Condition1', 'Condition2', 'Electrical',
            'Exterior1st','Exterior2nd','Foundation', 'Functional','GarageFinish','GarageType','Heating', 'HouseStyle',
            'LandContour', 'LandSlope', 'LotConfig', 'LotShape', 'MSZoning','MasVnrType', 'Neighborhood', 'PavedDrive', 
            'RoofMatl', 'RoofStyle','SaleCondition', 'SaleType', 'Street','MoSold', 'YrSold',
             'MSSubClass']


# In[ ]:


data = pd.get_dummies(data, columns=cat_cols,prefix=cat_cols,drop_first=True )


# If you log transform the response/target variable, it is required to also log transform feature variables that are skewed. For the numerical features , features with a skew value greater than 0.75 with be log transformed

# In[ ]:


num_feature = ['2ndFlrSF', '3SsnPorch', 'BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF',
               'EnclosedPorch','GrLivArea', 'GarageYrBlt','LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea',
               'MiscVal', 'OpenPorchSF', 'PoolArea', 'ScreenPorch', 'TotalBsmtSF', 'WoodDeckSF','YearBuilt', 'YearRemodAdd']


# In[ ]:


# Check the skew of all numerical features
skewness = data[num_feature].skew()
skewness = skewness[abs(skewness) > 0.75]
skewed_feature = skewness.index


# In[ ]:


for feature in skewed_feature:
    data[feature] = np.log1p(data[feature])


# Lets split the data frame again using the number of rows in the previous train test as index

# In[ ]:


X_train = data[:1458]
X_test = data[1458:]


# Now we define a function using GridSearch that will pick the best values of XGBoost based on the range of values of parameters we will use with the model. Using the optimal values of parameters will help make better predictions as opposed to using default values of parameters.

# In[ ]:


def print_parameters(select_param, select_param_name, parameters):
    grid_search = GridSearchCV(estimator = xgb_model,
                            param_grid = parameters,
                            scoring = 'neg_mean_squared_error',
                            cv = 5,
                            n_jobs = -1)

    grid_result = grid_search.fit(X_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


xgb_model = XGBRegressor()


# Tunning Parameters

# In[ ]:


learning_rate = np.arange(0.01, 0.5, 0.02)
parameters = dict(learning_rate=learning_rate)

print_parameters(learning_rate, 'learning_rate', parameters)


# In[ ]:


n_estimators = range(100, 1000, 100)
parameters = dict(n_estimators=n_estimators)

print_parameters(n_estimators, 'n_estimators', parameters)


# In[ ]:


max_depth = range(0, 5)
parameters = dict(max_depth=max_depth)

print_parameters(max_depth, 'max_depth', parameters)


# In[ ]:


subsample = np.arange(0.2, 1., 0.2)
parameters = dict(subsample=subsample)

print_parameters(subsample, 'subsample', parameters)


# In[ ]:


colsample_bytree = np.arange(0.2, 1.2, 0.2)
parameters = dict(colsample_bytree=colsample_bytree)

print_parameters(colsample_bytree, 'colsample_bytree', parameters)


# In[ ]:


gamma = np.arange(0.001, 0.1, 0.02)
parameters = dict(gamma=gamma)

print_parameters(gamma, 'gamma', parameters)


# In[ ]:


min_child_weight = np.arange(0.5, 2.0, 0.2)
parameters = dict(min_child_weight=min_child_weight)

print_parameters(min_child_weight, 'min_child_weight', parameters)


# Assign the best values or ranges of values to the parameters obtained above to use with model and then use GridSearch to find the best model with the best score ie the lowest mean squared 

# In[ ]:


parameters = {  
                'colsample_bytree':[1],
                'subsample':[0.4,0.6],
                'gamma':[0.041],
                'min_child_weight':[1.1,1.3],
                'max_depth':[3,5],
                'learning_rate':[0.2, 0.25],
                'n_estimators':[400],                                                                    
                'reg_alpha':[0.75],
                'reg_lambda':[0.45],
                'seed':[10]
}


# In[ ]:



grid_search = GridSearchCV(estimator = xgb_model,
                        param_grid = parameters,
                        scoring = 'neg_mean_squared_error',
                        cv = 5,
                        n_jobs = -1)


# In[ ]:


xgb_model = grid_search.fit(X_train, y_train)


# In[ ]:


best_score = grid_search.best_score_


# In[ ]:


best_parameters = grid_search.best_params_


# In[ ]:


accuracies = cross_val_score(estimator=xgb_model, X=X_train, y=y_train, cv=10)


# In[ ]:


accuracies.mean()


# In[ ]:


y_pred = xgb_model.predict(X_test)
y_pred = np.floor(np.expm1(y_pred))


# In[ ]:


submission = pd.concat([test_ID, pd.Series(y_pred)], 
                        axis=1,
                        keys=['Id','SalePrice'])


# In[ ]:


submission.to_csv('sample_submission.csv', index = False)


# In[ ]:


submission

