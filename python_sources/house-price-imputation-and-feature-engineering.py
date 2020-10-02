#!/usr/bin/env python
# coding: utf-8

# <p style="text-align: center;font-size:64px;margin:50px;font-family:Impact, Charcoal, sans-serif;line-height: 140%;;">House Prices: Advanced Regression Techniques</p>

# <img src="https://storage.googleapis.com/kaggle-competitions/kaggle/5407/media/housesbanner.png">

# <a id="menu" style='font-size: 20px;' href='#1'>1. Summary of the problem</a><br>
# <a style='font-size: 20px;' href='#2'>2. Setup</a><br>
# <a style='font-size: 16px;' href='#2.1'>2.1 Load libs</a><br>
# <a style='font-size: 16px;' href='#2.2'>2.2 Custom functions</a><br>
# <a style='font-size: 20px;' href='#3'>3 Target value : SalePrice</a><br>
# <a style='font-size: 16px;' href='#3.1'>3.1 center and un-skew</a><br>
# <a style='font-size: 20px;' href='#4'>4. Data preparation</a><br>
# <a style='font-size: 16px;' href='#4.1'>4.1 Build main data</a><br>
# <a style='font-size: 16px;' href='#4.2'>4.2 Understanding data</a><br>
# <a style='font-size: 16px;' href='#4.3'>4.3 Fill empty values</a><br>
# <a style='font-size: 16px;' href='#4.4'>4.4 Impute empty values</a><br>
# <a style='font-size: 20px;' href='#5'>5 Features engineering</a><br>
# <a style='font-size: 16px;' href='#5.1'>5.1 Categorical and quantitative features</a><br>
# <a style='font-size: 16px;' href='#5.2'>5.2 Environnement features</a><br>
# <a style='font-size: 16px;' href='#5.3'>5.3 Features relative to the Lot</a><br>
# <a style='font-size: 16px;' href='#5.4'>5.4 Features relative to the construction</a><br>
# <a style='font-size: 16px;' href='#5.5'>5.5 Features relative to the composition of the House</a><br>
# 

# # 1. Summary of the problem<a id='1'></a>

# **Goal**
# 
# The job is to predict the sales price for houses given in a dataset. For each Id in the test set, we must predict the value of the SalePrice variable, using the given features. 
# 
# **Metric**
# 
# Submissions are evaluated on ***Root-Mean-Squared-Error (RMSE)*** between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)
# 
# **Submission File Format**
# 
# The file should contain a header and have the following format:
# >Id,SalePrice  
# >1461,169000.1  
# >1462,187724.1233  
# >1463,175221

# # 2. Setup<a id='2'></a>

# ## 2.1 Load libs<a id='2.1'></a>

# In[ ]:


# libs to deal with data
import pandas as pd
import numpy as np
import scipy

# libs to display graphics
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# visual options
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# let's load train and test data
train_data=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_data=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_data['SalePrice']=np.nan
test_data['tmpPrice']=np.nan


# In[ ]:


print("train_data lenght :",len(train_data))
print("test_data lenght :",len(test_data))


# We see that the training dataset is pretty small in comparison with the test dataset !!

# In[ ]:


test_data.head()


# In[ ]:


train_data.head()


# ## 2.2 Custom functions<a id='2.2'></a>

# In[ ]:


# correlation between qualitative features
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


# In[ ]:


# correlation between qualitative and quantitative features
def eta_squared(x,y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT


# In[ ]:


# this function returns the correlation beween two features, by applying the appropriate correlation
# .corr() between quantitative
# cramer between qualitative features
# eta_squared between qualitative and quantitative features
def adaptative_correlation(x,y):
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    # keep only values that are both non na in x and y
    non_na = np.logical_and(~x.isna(),~y.isna())
    x=x[non_na]
    y=y[non_na]
    if (x.dtype.name=='object' or x.dtype.name=='category') and (y.dtype.name=='object' or y.dtype.name=='category'):
        return cramers_v(x,y)
    elif (x.dtype.name=='object' or x.dtype.name=='category'):
        return abs(eta_squared(x,y))
    elif (y.dtype.name=='object' or y.dtype.name=='category'):
        return abs(eta_squared(y,x))
    else:
        return abs(x.corr(y))


# And here is a function that draws a heatmap, regardless of dtype of features 

# In[ ]:


# this function uses the adaptative_correlation function to draw the heatmap between fetures
# regarless of their dtype
def adaptative_heatmap(df):
    b = range(len(df.columns))
    correl=[[adaptative_correlation(df.iloc[:,i],df.iloc[:,j]) if i<j else 1.0 if i==j else 0.0 for i in b] for j in b]
    mask = np.zeros_like(correl, dtype=np.bool)
    mask[np.triu_indices_from(mask,k=1)] = True
    fig, ax = plt.subplots(figsize=(13,10)) 
    sns.heatmap(correl,ax=ax,linewidths=.1,mask=mask, annot=True,cmap='coolwarm')
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.columns)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)


# In[ ]:


# a quick and dirty regressions function that we will use to measure the efficiency 
# of our transformations, if a transformation is benefic it should increase the score of the regression
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score,cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error

imputer = KNNImputer(n_neighbors=3)
n_folds = 5
model = RandomForestRegressor(n_jobs=-1)

def quick_predict(X,y):
    X = pd.get_dummies(X)
    X = pd.DataFrame(imputer.fit_transform(X), columns = X.columns)

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X)
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf))
    print("Quality score: {:.4f} standard deviation ({:.4f})\n".format(rmse.mean(), rmse.std()))


# In[ ]:


quick_predict(train_data.drop(['SalePrice'],axis=1),train_data.SalePrice)


# In[ ]:


# a "mean encoder", transforms a categorical feature by a numerical feature, applying for each
# the mean value of target so that we have a linear correlation between this new feature ant the target value
def mean_encoder(df,columns):
    for col in columns:
        if col not in df.columns:
            print("Error, ",col," not in list of columns")
        else:
            mean_encode = df.groupby(col)['tmpPrice'].mean()
            mean_encode = (mean_encode - mean_encode.min())/(mean_encode.max() - mean_encode.min())
            new_col_name = col + '_enc'
            df.loc[:,new_col_name] = df[col].map(mean_encode)


# # 3 Target value : SalePrice<a id='3'></a>
# ## 3.1 center and un-skew<a id='3.1'></a>

# SalePrice is the target. If we plan to elaborate some linear regression it is better to have a target parameter that has  a normal distribution, centered, normalized, and no skew (symetrical). Let's see if our price hase such a distribution.

# In[ ]:


sns.set(style="whitegrid", color_codes=True)
sns.set(rc={'figure.figsize':(15,8)})
_= sns.distplot(train_data.SalePrice.dropna())


# First of all there are some outliers, very high sales prices 

# In[ ]:


sns.set(style="whitegrid", color_codes=True)
sns.set(rc={'figure.figsize':(15,8)})
_= sns.boxplot(train_data.SalePrice.dropna())


# We will drop prices above 600 000$, they are not significative and may affect our computations

# In[ ]:


len(train_data[train_data.SalePrice>600000])


# In[ ]:


train_data = train_data[train_data.SalePrice<600000]


# now the skew value :

# In[ ]:


scipy.stats.skew(train_data.SalePrice.dropna())


# Data is skewed, the best transformation to apply is given by the boxcox function (see here : https://opendatascience.com/transforming-skewed-data-for-machine-learning/)

# In[ ]:


scipy.stats.boxcox(train_data.SalePrice,lmbda=None)[-1]


# The value is close to zero, so the best transformation is the log :

# In[ ]:


sns.set(style="whitegrid", color_codes=True)
sns.set(rc={'figure.figsize':(15,8)})
_= sns.distplot(np.log1p(train_data.SalePrice.dropna()))


# In[ ]:


train_data['tmpPrice']=np.log1p(train_data.SalePrice)


# In[ ]:


# keep tmpPriceMean so that we can reverse at the end of this work
tmpPriceMean = train_data['tmpPrice'].mean()
train_data['tmpPrice'] = train_data['tmpPrice']


# In[ ]:


sns.set(style="whitegrid", color_codes=True)
sns.set(rc={'figure.figsize':(15,8)})
_= sns.distplot(train_data.tmpPrice.dropna())


# In[ ]:


scipy.stats.skew(train_data.tmpPrice)


# In[ ]:


quick_predict(train_data.drop(['SalePrice','tmpPrice'],axis=1),train_data.tmpPrice)


# # 4. Data preparation
# <a id='4' href="#4.1">next</a>
# ## 4.1 Build main data
# <a id='4.1' href="#4.2">next</a>

# We will merge train and test data, so that we can apply the same transformations to the two data sets and so that we can do some global stats

# In[ ]:


train_data.set_index(train_data.Id,drop=True, inplace=True)
train_data = train_data.drop(['Id'],axis=1)
test_data.set_index(test_data.Id,drop=True, inplace=True)
test_data = test_data.drop(['Id'],axis=1)


# In[ ]:


data=pd.concat([train_data,test_data],keys=['train','test'], join='inner')


# In[ ]:


data.tail()


# In[ ]:


data.head()


# ## 4.2 Understanding data
# <a id='4.2' href="#4.3">next</a>
# 
# We could go ahead holus-bolus and try a lot of things with this dataset, but a data scientist should always try to understand data. So let's think as a real-estate agent.
# 
# If we take a look at the shape of the data

# In[ ]:


print("train data : ",train_data.shape,"test data : ",test_data.shape)


# We see that we have a lot of features (80) but not so much rows of data. So it seems important to understand data, get rid of toxic features, of outliers, and lelect the most important features. To have a better comprehension of the numerous features, we will split them into different famillies :

# **Features relative to environnement**<br>
# MSZoning: Identifies the general zoning classification of the sale.<br>
# Street: Type of road access to property<br>
# Alley: Type of alley access to property<br>
# Neighborhood: Physical locations within Ames city limits<br>
# Condition1: Proximity to various conditions<br>
# Condition2: Proximity to various conditions (if more than one is present)<br>
# <br>
# **Features relative to the Lot**<br>
# LotFrontage: Linear feet of street connected to property<br>
# LotArea: Lot size in square feet<br>
# LotShape: General shape of property<br>
# LandContour: Flatness of the property<br>
# LotConfig: Lot configuration<br>
# LandSlope: Slope of property<br>
# Fence: Fence quality<br>
# ***Fireplace***<br>
# Fireplaces: Number of fireplaces<br>
# FireplaceQu: Fireplace quality<br>
# ***Garage***<br>
# GarageType: Garage location<br>
# GarageYrBlt: Year garage was built<br>
# GarageFinish: Interior finish of the garage<br>
# GarageCars: Size of garage in car capacity<br>
# GarageArea: Size of garage in square feet<br>
# GarageQual: Garage quality<br>
# GarageCond: Garage condition<br>
# ***Pool***<br>
# PoolArea: Pool area in square feet<br>
# PoolQC: Pool quality<br>
# <br>
# **Features relative to the construction**<br>
# MSSubClass: Identifies the type of dwelling involved in the sale.	<br>
# BldgType: Type of dwelling<br>
# HouseStyle: Style of dwelling<br>
# OverallQual: Rates the overall material and finish of the house<br>
# OverallCond: Rates the overall condition of the house<br>
# YearBuilt: Original construction date<br>
# YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)<br>
# RoofStyle: Type of roof<br>
# RoofMatl: Roof material<br>
# Exterior1st: Exterior covering on house<br>
# Exterior2nd: Exterior covering on house (if more than one material)<br>
# MasVnrType: Masonry veneer type<br>
# MasVnrArea: Masonry veneer area in square feet<br>
# ExterQual: Evaluates the quality of the material on the exterior <br>
# ExterCond: Evaluates the present condition of the material on the exterior<br>
# Foundation: Type of foundation<br>
# WoodDeckSF: Wood deck area in square feet<br>
# PavedDrive: Paved driveway<br>
# <br>
# **Features relative to the composition of the house**<br>
# ***basement***<br>
# BsmtQual: Evaluates the height of the basement<br>
# BsmtCond: Evaluates the general condition of the basement<br>
# BsmtExposure: Refers to walkout or garden level walls<br>
# BsmtFinType1: Rating of basement finished area<br>
# BsmtFinSF1: Type 1 finished square feet<br>
# BsmtFinType2: Rating of basement finished area (if multiple types)<br>
# BsmtFinSF2: Type 2 finished square feet<br>
# BsmtUnfSF: Unfinished square feet of basement area<br>
# TotalBsmtSF: Total square feet of basement area<br>
# BsmtFullBath: Basement full bathrooms<br>
# BsmtHalfBath: Basement half bathrooms<br>
# ***Porch***<br>
# OpenPorchSF: Open porch area in square feet<br>
# EnclosedPorch: Enclosed porch area in square feet<br>
# 3SsnPorch: Three season porch area in square feet<br>
# ScreenPorch: Screen porch area in square feet<br>
# <br>
# ***Above basement***<br>
# 1stFlrSF: First Floor square feet<br>
# 2ndFlrSF: Second floor square feet<br>
# LowQualFinSF: Low quality finished square feet (all floors)<br>
# GrLivArea: Above grade (ground) living area square feet<br>
# FullBath: Full bathrooms above grade<br>
# HalfBath: Half baths above grade<br>
# Bedroom: Bedrooms above grade (does NOT include basement bedrooms)<br>
# Kitchen: Kitchens above grade<br>
# KitchenQual: Kitchen quality<br>
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)<br>
# <br>
# **comfort**<br>
# Utilities: Type of utilities available<br>
# Heating: Type of heating<br>
# HeatingQC: Heating quality and condition<br>
# CentralAir: Central air conditioning<br>
# Electrical: Electrical system<br>
# Functional: Home functionality (Assume typical unless deductions are warranted)<br>
# MiscFeature: Miscellaneous feature not covered in other categories<br>
# MiscVal: $Value of miscellaneous feature<br>
# <br>
# **Condition of sale**<br>
# MoSold: Month Sold (MM)<br>
# YrSold: Year Sold (YYYY)<br>
# SaleType: Type of sale<br>
# SaleCondition: Condition of sale<br>

# So we have now famillies of features that are related one to another (but perhaps not correlated).

# ## 4.3 Fill empty values
# <a id='4.3' href="#4.4">next</a>

# In data description, we learn the signification of NA for some data : 
# Alley: NA means 	No alley access<br>
# BsmtQual: NA means	No Basement<br>
# BsmtCond: NA means	No Basement<br>
# BsmtExposure: NA means	No Basement<br>
# BsmtFinType1: NA means	No Basement<br>
# BsmtFinType2: NA means	No Basement<br>
# FireplaceQu: NA means	No Fireplace<br>
# GarageType: NA means	No Garage<br>
# GarageFinish: NA means	No Garage<br>
# GarageQual: NA means	No Garage<br>
# GarageCond: NA means	No Garage<br>
# PoolQC: NA means	No Pool<br>
# Fence: NA means	No Fence<br>
# MiscFeature: NA means	None<br>

# In[ ]:


data[['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
      'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
      'PoolQC','Fence','MiscFeature']]=data[['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
      'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
      'PoolQC','Fence','MiscFeature']].fillna(value="NA")


# ## 4.4 Impute empty values
# <a id='4.4' href="#5">next</a>

# In[ ]:


data.isna().sum()[lambda x: x>0]


# MSZoning : For the 4 NA values, We will apply the most frequent value of MSZoning in the same Neighborhood

# In[ ]:


pd.crosstab(data.MSZoning.fillna('NA'),data.LandContour,dropna=False)


# In[ ]:


data[data.MSZoning.isna()]


# In[ ]:


data[data.Neighborhood=="Mitchel"].MSZoning.value_counts()


# In[ ]:


data[data.Neighborhood=="IDOTRR"].MSZoning.value_counts()


# In[ ]:


data.loc[(data.MSZoning.isna())&(data.Neighborhood=="Mitchel"),"MSZoning"]="RL"
data.loc[(data.MSZoning.isna())&(data.Neighborhood=="IDOTRR"),"MSZoning"]="RM"


# LotFrontage : We have a lot of missing values for lot frontage, but we can estimate it, what is lot frontage : <br><img src="https://upload.wikimedia.org/wikipedia/commons/b/bc/Lot_map.PNG">

# Lot Area (value that we also have in dataset) is lot frontage multiplicated by lot depth. If the Lot is not too irregular we can estimate that lot fontage is proportional to square root of lot area, let's test that hypothesis :

# In[ ]:


data['LotAreaSqr']=np.sqrt(data.LotArea)


# In[ ]:


data.LotShape.value_counts()


# In[ ]:


sns.set(style="whitegrid", color_codes=True)
sns.set(rc={'figure.figsize':(15,8)})
sns.scatterplot(data.LotFrontage, data.LotAreaSqr,hue=data.LotShape, alpha=0.5)
plt.xlabel('Lot Frontage')
plt.ylabel('Square root of lot Area')
plt.show()


# I think this is the best estimation we can get, I will replace missing lot frontage values by the square root of lot area

# In[ ]:


data['LotFrontage']=data.LotFrontage.fillna(data.LotAreaSqr)


# In[ ]:


data = data.drop(['LotAreaSqr'],axis=1)


# **Utilities** : this feature has no "Utility", we have only 1 value different from the 2912 others

# In[ ]:


data = data.drop(['Utilities'],axis=1)


# **Exterior1st**

# In[ ]:


data[data.Exterior1st.isna()]


# In[ ]:


data[(data.YearBuilt>1935)&(data.YearBuilt<1945)&(data.Neighborhood=='Edwards')].Exterior1st.value_counts()


# In[ ]:


data['Exterior1st']=data.Exterior1st.fillna('MetalSd')
data['Exterior2nd']=data.Exterior2nd.fillna('MetalSd')


# **Basement Data**

# In[ ]:


data[data.BsmtFinSF1.isna()]


# In[ ]:


# Fix some errors where house have no basement
data.loc[data.BsmtFinSF1.isna(),'BsmtFinSF2' ]=0.0
data.loc[data.BsmtFinSF1.isna(),'BsmtUnfSF' ]=0.0
data.loc[data.BsmtFinSF1.isna(),'BsmtFullBath' ]=0.0
data.loc[data.BsmtFinSF1.isna(),'TotalBsmtSF' ]=0.0
data.loc[data.BsmtFinSF1.isna(),'BsmtHalfBath' ]=0.0
data.loc[data.BsmtFinSF1.isna(),'BsmtFinSF1' ]=0.0
data.loc[data.BsmtQual=="NA",'BsmtExposure' ]="NA"
data.loc[data.BsmtQual=="NA",'BsmtFullBath' ]=0.0
data.loc[data.BsmtQual=="NA",'BsmtHalfBath' ]=0.0


# **Electrical**

# In[ ]:


data[data.Electrical.isna()]


# In[ ]:


data.Electrical.value_counts()


# In[ ]:


data[['Electrical']]=data[['Electrical']].fillna(value="SBrkr")


# **KitchenQual**
# We will attribute "TA", the most common value for KitchenQual for the houses whose OverallQual is 5

# In[ ]:


data['KitchenQual']=data['KitchenQual'].fillna(value="TA")


# **GarageYrBlt** We have a lot of empty data, they correspond to houses with no garage, we can fill with any value, but the most efficient, I think, is to fill this value with the year the house was built

# In[ ]:


data['GarageYrBlt']=data['GarageYrBlt'].fillna(data.YearBuilt)


# **Other garage features**

# In[ ]:


data[data.GarageCars.isna()]


# In[ ]:


data.loc[data.GarageCars.isna(),data.columns.str.contains('Gar')]


# In[ ]:


sns.boxplot(data[data.GarageType=='Detchd'].GarageCars,data[data.GarageType=='Detchd'].LotFrontage)


# The house where garagecars is missing has a small lotfrontage, it is likely that it has a only 1 car detached garage, let us search the most common garage area for those type of garages :

# In[ ]:


data[(data.GarageType=='Detchd')&(data.GarageCars==1.0)].GarageArea.describe()


# We will fix the garage area to 300

# In[ ]:


data['GarageArea']=data['GarageArea'].fillna(value=300)
data['GarageCars']=data['GarageCars'].fillna(value=1.0)


# **Functional**

# In[ ]:


data.Functional.value_counts()


# In[ ]:


data['Functional']=data['Functional'].fillna(value="Typ")


# **SaleType**

# In[ ]:


data.SaleType.value_counts()


# In[ ]:


data['SaleType']=data['SaleType'].fillna(value="WD")


# **MasVnrType**
# 
# This one is a bit more difficult to impute, there no imediate value to guess, we will have to call a friend : a predicter

# In[ ]:


X = data.drop(['SalePrice', 'tmpPrice','MasVnrType','MasVnrArea'],axis=1)
X = X.reset_index(level=0, drop=True)
X = pd.get_dummies(X)

Y = data[['MasVnrType']]
Y = Y.reset_index(level=0, drop=True)

Y_source = Y.dropna()
X_source = X.loc[Y_source.index,:]
X_target = X.loc[set(X.index) - set(Y_source.index),:]


# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(Y_source.values)


# In[ ]:


Y_source = le.transform(Y_source)


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=300, max_depth=None,min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X_source, Y_source, cv=5)
scores.mean()


# In[ ]:


_=clf.fit(X_source, Y_source)


# In[ ]:


Y_target = pd.DataFrame(le.inverse_transform(clf.predict(X_target))).set_index(X_target.index)


# In[ ]:


Y_target.columns = ['MasVnrType']


# In[ ]:


data.loc['train'].update(Y_target)
data.loc['test'].update(Y_target)


# **MasVnrArea**
# 
# And here we will do the same process to estimate MasVnrArea

# In[ ]:


X = data.drop(['SalePrice', 'tmpPrice','MasVnrArea'],axis=1)
X = X.reset_index(level=0, drop=True)
X = pd.get_dummies(X)

Y = data[['MasVnrArea']]
Y = Y.reset_index(level=0, drop=True)

Y_source = Y.dropna()
X_source = X.loc[Y_source.index,:]

X_target = X.loc[set(X.index) - set(Y_source.index),:]


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(n_estimators=100, min_samples_split=3, random_state=0)
scores = cross_val_score(clf, X_source, Y_source, cv=5, scoring="neg_mean_squared_error",n_jobs=-1)

scores.mean()


# In[ ]:


_=clf.fit(X_source, Y_source)
Y_target = pd.DataFrame(clf.predict(X_target)).set_index(X_target.index)
Y_target.columns = ['MasVnrArea']


# In[ ]:


data.loc['train'].update(Y_target)
data.loc['test'].update(Y_target)


# In[ ]:


data.isna().sum()[lambda x: x>0]


# In[ ]:


quick_predict(data.loc['train'].drop(['SalePrice','tmpPrice'],axis=1),data.loc['train'].tmpPrice)


# # 5 Features engineering
# <a id='5' href="#5.1">next</a>

# ## 5.1 Categorical and quantitative features
# <a id='5.1' href="#5.2">next</a> - <a href="#menu">top</a>

# This is primordial, categorical and numerical columns cannot be handled the same way. For example to compute correlation between two parameters, the algorithm to use is not the same.
# Separating categorical and numerical values is not easy to do, some values are saved as integers (like MSSubClass) but are in reality categorical and should be treated as such. 
# So we have to manually check data types

# In[ ]:


data.head()


# In[ ]:


data.dtypes.value_counts()


# ### Object types

# In[ ]:


print(list(data.select_dtypes(['object']).columns))


# In[ ]:


data[data.select_dtypes(['object']).columns].head()


# ### int64 values

# In[ ]:


print(list(data.select_dtypes(['int64']).columns))


# In[ ]:


pd.set_option('display.max_columns', 100)
data[data.select_dtypes(['int64']).columns].head()


# 'MSSubClass', 'MoSold' are categorical features, in reality Month is a quantitative value (1 to 12) but it's value has no sense in regard to the price, so we will interpret this feature like a qualitative feature

# In[ ]:


data[['MSSubClass', 'MoSold','OverallQual','OverallCond','YrSold']] = data[['MSSubClass', 'MoSold','OverallQual','OverallCond','YrSold']].astype(str)


# ### Float parameters

# In[ ]:


data[data.select_dtypes(['float64']).columns].head()


# All those features are in reality integers :

# In[ ]:


data[['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 
      'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 
      'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 
      'GarageArea']] = data[['LotFrontage', 'MasVnrArea', 'BsmtFinSF1',
                            'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 
                            'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea']].astype(int)


# In[ ]:


quick_predict(data.loc['train'].drop(['SalePrice','tmpPrice'],axis=1),data.loc['train'].tmpPrice)


# ## 5.2 Environnement features
# <a id='5.2' href="#5.3">next</a> - <a href="#menu">top</a><br>
# 
# For the different families of features we will apply the same workflow
# * univariate analysis
#     * distribution, mean, max/min, ....
#     * outliers
#     * Type of distribution (Gaussian, uniform, logarithmic, etc.)
#     * Transform (log ...)
# 
# * multivariate analysis
#     * Study the correlations between attributes.
#     * Identify extra data that would be useful
#     * detect inconsistencies
# 
# Here are features relatives to the environnement
# * MSZoning: Identifies the general zoning classification of the sale.<br>
# * Street: Type of road access to property<br>
# * Alley: Type of alley access to property<br>
# * Neighborhood: Physical locations within Ames city limits<br>
# * Condition1: Proximity to various conditions<br>
# * Condition2: Proximity to various conditions (if more than one is present)<br>

# In[ ]:


env_cols =['MSZoning','Street','Alley','Neighborhood','Condition1','Condition2']


# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 20))

for idx, feat in enumerate(env_cols): 
    ax = axes[int(idx / 2), idx % 2] 
    sns.swarmplot(x=feat, y='tmpPrice', data=data, ax=ax) 
    ax.set_ylabel('')
    ax.set_xlabel(feat)
    ax.set_xticks([])
    ax.set_yticks([])
fig.show();


# In[ ]:


mean_encoder(data,['MSZoning','Street','Alley','Neighborhood'])
#data = data.drop(['MSZoning','Street','Alley','Neighborhood'],axis=1)


# And for condition1 and condition2, they represent in fact the same type of information, it would be more understandable for ML if we had some dummies features summarizing those two columns

# In[ ]:


data['ConditionFeedr'] = (data.Condition2=="Feedr")*1+(data.Condition1=="Feedr")*1
data['ConditionArtery'] = (data.Condition2=="Artery")*1+(data.Condition1=="Artery")*1
data['ConditionPosA'] = (data.Condition2=="PosA")*1+(data.Condition1=="PosA")*1
data['ConditionPosN'] = (data.Condition2=="PosN")*1+(data.Condition1=="PosN")*1
data['ConditionRRAn'] = (data.Condition2=="RRAn")*1+(data.Condition1=="RRAn")*1
data['ConditionRRAe'] = (data.Condition2=="RRAe")*1+(data.Condition1=="RRAe")*1
data['ConditionRRNn'] = (data.Condition2=="RRNn")*1+(data.Condition1=="RRNn")*1
data['ConditionRRNe'] = (data.Condition2=="RRNe")*1+(data.Condition1=="RRNe")*1
data = data.drop(['Condition1','Condition2'],axis=1)


# In[ ]:


quick_predict(data.loc['train'].drop(['SalePrice','tmpPrice'],axis=1),data.loc['train'].tmpPrice)


# ## 5.3 Features relative to the Lot
# <a id='5.3' href="#5.4">next</a> - <a href="#menu">top</a><br>
# 
# LotFrontage: Linear feet of street connected to property   
# LotArea: Lot size in square feet   
# LotShape: General shape of property   
# LandContour: Flatness of the property   
# LotConfig: Lot configuration   
# LandSlope: Slope of property   
# Fence: Fence quality   
# Fireplace   
# Fireplaces: Number of fireplaces   
# FireplaceQu: Fireplace quality   
# Garage   
# GarageType: Garage location   
# GarageYrBlt: Year garage was built   
# GarageFinish: Interior finish of the garage   
# GarageCars: Size of garage in car capacity   
# GarageArea: Size of garage in square feet   
# GarageQual: Garage quality   
# GarageCond: Garage condition   
# Pool   
# PoolArea: Pool area in square feet   
# PoolQC: Pool quality   

# In[ ]:


data[['LotFrontage','LotArea','LotShape','LandContour','LotConfig','LandSlope','Fence','Fireplaces','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PoolArea','PoolQC']].head()


# In[ ]:


lot_cols =['LotFrontage','LotArea','LotShape','LandContour','LotConfig','LandSlope','Fence','Fireplaces','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PoolArea','PoolQC']


# In[ ]:


data['LotShape'].dtype=='object'


# In[ ]:


fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 30))
i=0

for idx, feat in enumerate(lot_cols): 
    if data[feat].dtype=='object':
        ax = axes[int(i / 3), i % 3] 
        sns.swarmplot(x=feat, y='tmpPrice', data=data, ax=ax)
        ax.set_ylabel('')
        ax.set_xlabel(feat)
        ax.set_xticks([])
        ax.set_yticks([])
        i+=1
fig.show();


# We will apply the mean encoder for almost all categorical features, the only one we will treat differently is poolQC

# In[ ]:


mean_encoder(data,['LotShape','LandContour','LotConfig','LandSlope','Fence','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond'])
#data = data.drop(['LotShape','LandContour','LotConfig','LandSlope','Fence','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond'],axis=1)


# **Inconsistencies**

# In[ ]:


data.PoolQC.isna().sum()


# In[ ]:


data['Pool'] = (data.PoolQC!="NA")*1


# In[ ]:


data = data.drop(['PoolQC'],axis=1)


# First let's look if there are some inconsistencies. For pool, if we have no pool whe should have no poll quality and zero for pool Area

# In[ ]:


data[data.Pool==0].PoolArea.value_counts()


# In[ ]:


# there seems to be a pool, and pool quality is NA, we will change to "Gd"
data.loc[(data.Pool==0)&(data.PoolArea>0),'Pool']=1


# Let's take a look now to quantitative features

# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
i=0

for idx, feat in enumerate(lot_cols):
    if feat not in data.columns:continue
    if data[feat].dtype!='object':
        ax = axes[int(i / 3), i % 3] 
        sns.scatterplot(x=feat, y='tmpPrice', data=data, ax=ax)
        ax.set_ylabel('')
        ax.set_xlabel(feat)
        ax.set_xticks([])
        ax.set_yticks([])
        i+=1
fig.show();


# ### LotFrontage: Linear feet of street connected to property

# In[ ]:


sns.set(style="whitegrid", color_codes=True)
sns.set(rc={'figure.figsize':(15,8)})
_= sns.distplot(train_data.LotFrontage.dropna())


# In[ ]:


data[data.LotFrontage>300]


# We will drop those ouliers

# In[ ]:


data = data[data.LotFrontage<300]


# And last transformation, we will try to approach normality

# In[ ]:


scipy.stats.boxcox(data.LotFrontage)[-1]


# In[ ]:


data.tmpPrice.corr(data.LotFrontage)


# In[ ]:


data.tmpPrice.corr(np.power(data.LotFrontage,0.25))


# we will transform Lot Frontage by applying np.Power with a value of 0.35 given by boxcox test

# In[ ]:


data['LotFrontage']=np.power(data.LotFrontage,0.25)


# ### LotArea: Lot size in square feet

# In[ ]:


sns.set(style="whitegrid", color_codes=True)
sns.set(rc={'figure.figsize':(15,8)})
_= sns.distplot(data.LotArea.dropna())


# Lot Area can be kept as it is, we have no NA, I will only delete one big outlier that is over 200000 sqr feet

# In[ ]:


data = data[data.LotArea<100000]


# In[ ]:


data['LotArea'] = np.log1p(data.LotArea)


# **Fireplaces**

# In[ ]:


data.groupby('Fireplaces')['tmpPrice'].mean()


# **Number of car places in garage**

# In[ ]:


data.groupby('GarageCars')['tmpPrice'].mean()


# In[ ]:


data.GarageCars.value_counts()


# We have a very small amount of 4 or 5 places garages, we will replace 4 and 5 by 3

# In[ ]:


data['GarageCars']=data.GarageCars.replace([4,5],3)


# ***Garage year built***   
# First of all we have an outlier

# In[ ]:


data[data.GarageYrBlt>2010]


# In[ ]:


data.loc[data.GarageYrBlt>2010,'GarageYrBlt']=2007


# And we will apply a little transformation so that we have a better linear relation with tmpPrice

# In[ ]:


data['GarageYrBlt']=np.log1p(2011-data.GarageYrBlt)


# In[ ]:


data.groupby('GarageCars')['tmpPrice'].mean()


# And some other improvements that I will not detail

# In[ ]:


data = data.drop(['PoolArea'],axis=1)


# In[ ]:


quick_predict(data.loc['train'].drop(['SalePrice','tmpPrice'],axis=1),data.loc['train'].tmpPrice)


# In[ ]:


data.isna().sum()[lambda x: x>0]


# ## 5.4 Features relative to the construction
# <a id='5.4' href="#5.5">next</a> - <a href="#menu">top</a><br>
# MSSubClass: Identifies the type of dwelling involved in the sale.   
# BldgType: Type of dwelling   
# HouseStyle: Style of dwelling   
# OverallQual: Rates the overall material and finish of the house   
# OverallCond: Rates the overall condition of the house   
# YearBuilt: Original construction date   
# YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)   
# RoofStyle: Type of roof   
# RoofMatl: Roof material   
# Exterior1st: Exterior covering on house   
# Exterior2nd: Exterior covering on house (if more than one material)   
# MasVnrType: Masonry veneer type   
# MasVnrArea: Masonry veneer area in square feet   
# ExterQual: Evaluates the quality of the material on the exterior   
# ExterCond: Evaluates the present condition of the material on the exterior   
# Foundation: Type of foundation   
# WoodDeckSF: Wood deck area in square feet   
# PavedDrive: Paved driveway

# In[ ]:


const_cols = ['MSSubClass','BldgType','HouseStyle','OverallQual','OverallCond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','WoodDeckSF','PavedDrive']


# In[ ]:


adaptative_heatmap(data[const_cols])


# **Categorical features**

# In[ ]:


fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(20, 40))
i=0

for idx, feat in enumerate(const_cols): 
    if data[feat].dtype=='object':
        ax = axes[int(i / 3), i % 3] 
        sns.swarmplot(x=feat, y='tmpPrice', data=data, ax=ax)
        ax.set_ylabel('')
        ax.set_xlabel(feat)
        ax.set_xticks([])
        ax.set_yticks([])
        i+=1
fig.show();


# ### MSSubClass: The building class

# MSSubClass: Identifies the type of dwelling involved in the sale.	
# 
#         20	1-STORY 1946 & NEWER ALL STYLES
#         30	1-STORY 1945 & OLDER
#         40	1-STORY W/FINISHED ATTIC ALL AGES
#         45	1-1/2 STORY - UNFINISHED ALL AGES
#         50	1-1/2 STORY FINISHED ALL AGES
#         60	2-STORY 1946 & NEWER
#         70	2-STORY 1945 & OLDER
#         75	2-1/2 STORY ALL AGES
#         80	SPLIT OR MULTI-LEVEL
#         85	SPLIT FOYER
#         90	DUPLEX - ALL STYLES AND AGES
#        120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
#        150	1-1/2 STORY PUD - ALL AGES
#        160	2-STORY PUD - 1946 & NEWER
#        180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
#        190	2 FAMILY CONVERSION - ALL STYLES AND AGES

# In[ ]:


sns.set(style="whitegrid", color_codes=True)
sns.set(rc={'figure.figsize':(15,8)})
sns.boxplot(data.MSSubClass,data.tmpPrice)


# In[ ]:


pd.crosstab(data.index.get_level_values(0),data.MSSubClass)


# We have a little problem, the 150 value is encountered only in test dataset, we will replace this value with the description that seems the closest : 120

# In[ ]:


data['MSSubClass']=data.MSSubClass.replace('150','120').astype(object)


# ### OverallQual: Overall material and finish quality

# In[ ]:


pd.crosstab(data.index.get_level_values(0),data.OverallQual.fillna('NA'),dropna=False)


# In[ ]:


sns.set(style="whitegrid", color_codes=True)
sns.set(rc={'figure.figsize':(15,8)})
sns.swarmplot(data.OverallQual,data.tmpPrice)


# There is a perfect linear dependency between OverallQual and price (more exacly the log of the price), OverallQual is a qualitative feature but should be transformed to a quantitative value, so that linear dependency is better understood by our algos.

# In[ ]:


adaptative_correlation(data.OverallQual,data.tmpPrice)


# In[ ]:


data['OverallQual'].replace([1, 2], 3, inplace=True)
data['OverallQual'].replace([10], 9, inplace=True)


# We have a better correlation transforming OverallQual to quantitative value, as this feature is ordinal (a higher value represents a higher quality)

# In[ ]:


data['OverallQual'] = data.OverallQual.astype('int')


# **RoofMatl** : this feture is too unbalanced, we will get rid of it

# In[ ]:


data = data.drop(['RoofMatl'],axis=1)


# ### OverallCond: Overall condition rating

# In[ ]:


pd.crosstab(data.index.get_level_values(0),data.OverallCond.fillna('NA'),dropna=False)


# In[ ]:


sns.set(style="whitegrid", color_codes=True)
sns.set(rc={'figure.figsize':(15,8)})
sns.boxplot(data.OverallCond,data.tmpPrice)


# There is a big dependency between OverallCond and price, OverallCond is a qualitative feature, and should remain as qualitative because the relation between price and OverallCond is not linear.

# In[ ]:


data.OverallCond.value_counts()


# In[ ]:


data.OverallCond.dtype


# For the others categorical features there is nothing special to say

# We will apply mean_encoder to the list of categorical features we have reviewed

# In[ ]:


mean_encoder(data,['MSSubClass','BldgType','HouseStyle','OverallCond','RoofStyle','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','PavedDrive'])
#data = data.drop(['MSSubClass','BldgType','HouseStyle','OverallCond','RoofStyle','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','PavedDrive'],axis=1)


# **Numerical features**

# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 20))
i=0

for feat in ['YearBuilt','YearRemodAdd','MasVnrArea','WoodDeckSF']: 
    if data[feat].dtype!='object':
        ax = axes[int(i / 3), i % 3] 
        sns.scatterplot(x=feat, y='tmpPrice', data=data, ax=ax)
        ax.set_ylabel('')
        ax.set_xlabel(feat)
        ax.set_xticks([])
        ax.set_yticks([])
        i+=1
fig.show();


# In[ ]:


data.YearRemodAdd.describe()


# In[ ]:


data[data.YearBuilt>data.YearRemodAdd]


# Remodeling cannot happen before age of the building

# In[ ]:


data.loc[data.YearBuilt>data.YearRemodAdd,'YearRemodAdd']=2002


# In[ ]:


data.YearBuilt.describe()


# In[ ]:


sns.scatterplot(data.YearBuilt,data.tmpPrice)


# In[ ]:


sns.scatterplot(np.power(np.abs(85-(2010 - data.YearBuilt)),2),data.tmpPrice)


# In[ ]:


data['YearBuilt'] = np.power(np.abs(85-(2010 - data.YearBuilt)),2)


# In[ ]:


quick_predict(data.loc['train'].drop(['SalePrice','tmpPrice'],axis=1),data.loc['train'].tmpPrice)


# ## 5.5 Features relative to the composition of the house
# <a id='5.5' href="#5.6">next</a> - <a href="#menu">top</a><br>
# basement    
# BsmtQual: Evaluates the height of the basement    
# BsmtCond: Evaluates the general condition of the basement    
# BsmtExposure: Refers to walkout or garden level walls    
# BsmtFinType1: Rating of basement finished area    
# BsmtFinSF1: Type 1 finished square feet    
# BsmtFinType2: Rating of basement finished area (if multiple types)    
# BsmtFinSF2: Type 2 finished square feet    
# BsmtUnfSF: Unfinished square feet of basement area    
# TotalBsmtSF: Total square feet of basement area    
# BsmtFullBath: Basement full bathrooms    
# BsmtHalfBath: Basement half bathrooms  
#    
# Porch    
# OpenPorchSF: Open porch area in square feet    
# EnclosedPorch: Enclosed porch area in square feet    
# 3SsnPorch: Three season porch area in square feet    
# ScreenPorch: Screen porch area in square feet    
#     
# Above basement    
# 1stFlrSF: First Floor square feet    
# 2ndFlrSF: Second floor square feet    
# LowQualFinSF: Low quality finished square feet (all floors)    
# GrLivArea: Above grade (ground) living area square feet    
# FullBath: Full bathrooms above grade    
# HalfBath: Half baths above grade    
# Bedroom: Bedrooms above grade (does NOT include basement bedrooms)    
# Kitchen: Kitchens above grade    
# KitchenQual: Kitchen quality    
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)    
#     
# comfort    
# Utilities: Type of utilities available    
# Heating: Type of heating    
# HeatingQC: Heating quality and condition    
# CentralAir: Central air conditioning    
# Electrical: Electrical system    
# Functional: Home functionality (Assume typical unless deductions are warranted)    
# MiscFeature: Miscellaneous feature not covered in other categories    
# MiscVal: $Value of miscellaneous feature

# <img src="https://boobiz.fr/house.jpg">

# In[ ]:


basement_cols = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']


# In[ ]:


fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 40))
i=0

for idx, feat in enumerate(basement_cols): 
    ax = axes[int(i / 3), i % 3] 
    if data[feat].dtype=='object':
        sns.swarmplot(x=feat, y='tmpPrice', data=data, ax=ax)
    else:
        sns.scatterplot(x=feat, y='tmpPrice', data=data, ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel(feat)
    ax.set_xticks([])
    ax.set_yticks([])
    i+=1
fig.show();


# Let's verify the consistency of those features. First, the total basement surface should be the sum of others surfaces

# In[ ]:


sns.scatterplot(data.BsmtFinSF1+data.BsmtFinSF2+data.BsmtUnfSF,data.TotalBsmtSF)


# And if basement surface is null, all others features should be zero or NA

# In[ ]:


data.loc[data.TotalBsmtSF==0,['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']].describe()


# BsmtFinSF2 has very low influence

# In[ ]:


adaptative_correlation(data.BsmtFinSF2,data.tmpPrice)


# In[ ]:


data = data.drop(['BsmtFinSF2'],axis=1)


# In[ ]:


pd.crosstab(data.index.get_level_values(0),data.BsmtFullBath)


# In[ ]:


pd.crosstab(data.index.get_level_values(0),data.BsmtHalfBath)


# There are very few houses with 3 fullbath and 2 halfbath (in basement) for the moment we keep those values but keep also in mind that we will have to replace those values

# Mean encoding for categorical features :

# In[ ]:


mean_encoder(data,['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'])
#data = data.drop(['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'],axis=1)


# **Porch**

# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
i=0

for idx, feat in enumerate(['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']): 
    ax = axes[int(i / 2), i % 2] 
    sns.scatterplot(x=feat, y='tmpPrice', data=data, ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel(feat)
    ax.set_xticks([])
    ax.set_yticks([])
    i+=1
fig.show();


# In[ ]:


for idx, feat in enumerate(['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']): 
    print(feat," : ",adaptative_correlation(data[feat],data.tmpPrice))


# In[ ]:





# **Above basement**   
# 1stFlrSF: First Floor square feet   
# 2ndFlrSF: Second floor square feet   
# LowQualFinSF: Low quality finished square feet (all floors)   
# GrLivArea: Above grade (ground) living area square feet   
# FullBath: Full bathrooms above grade   
# HalfBath: Half baths above grade   
# Bedroom: Bedrooms above grade (does NOT include basement bedrooms)   
# Kitchen: Kitchens above grade   
# KitchenQual: Kitchen quality   
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)    

# In[ ]:


floors_cols = ['1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd']


# In[ ]:


fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 40))
i=0

for idx, feat in enumerate(floors_cols): 
    ax = axes[int(i / 3), i % 3] 
    if data[feat].dtype=='object':
        sns.swarmplot(x=feat, y='tmpPrice', data=data, ax=ax)
    else:
        sns.scatterplot(x=feat, y='tmpPrice', data=data, ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel(feat)
    ax.set_xticks([])
    ax.set_yticks([])
    i+=1
fig.show();


# Some features seem to have no influence on price

# In[ ]:


adaptative_correlation(data.LowQualFinSF,data.tmpPrice)


# In[ ]:


data = data.drop(['LowQualFinSF'],axis=1)


# Encoding categorical data

# In[ ]:


mean_encoder(data,['KitchenQual'])
#data = data.drop(['KitchenQual'],axis=1)


# And create new features

# In[ ]:


# Total living area
data['LivingArea'] = data['TotalBsmtSF'] + data['1stFlrSF']+    data['2ndFlrSF']+    data['OpenPorchSF']+    data['3SsnPorch']

data['LivingArea'] = np.power(data['LivingArea'],0.35)


# In[ ]:


# Total Bathrooms
HalfBathfactor = 0.6
data['TotalBath'] = data.BsmtFullBath + HalfBathfactor * data.BsmtHalfBath +                    data.FullBath + HalfBathfactor * data.HalfBath


# **Area by room**   
# approximation of mean area by room above grade, gives an idea of the average size of rooms

# In[ ]:


data['Rooms_Area'] = (data.TotRmsAbvGrd+data.TotalBath) / data.GrLivArea


# **Tranform all areas to have a better skrewed repartition and correlation**

# In[ ]:


for feature in ['GrLivArea','BsmtFinSF1','1stFlrSF','2ndFlrSF','BsmtUnfSF','TotalBsmtSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']:
    data[feature] = np.power(data[feature],0.35)


# ### comfort###
# 
# Utilities: Type of utilities available   
# Heating: Type of heating   
# HeatingQC: Heating quality and condition   
# CentralAir: Central air conditioning   
# Electrical: Electrical system   
# Functional: Home functionality (Assume typical unless deductions are warranted)   
# MiscFeature: Miscellaneous feature not covered in other categories   
# MiscVal: $Value of miscellaneous feature   

# In[ ]:


comfort_cols=['Heating','HeatingQC','CentralAir','Electrical','Functional','MiscFeature','MiscVal']


# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 30))
i=0

for idx, feat in enumerate(comfort_cols): 
    ax = axes[int(i / 3), i % 3] 
    if data[feat].dtype=='object':
        sns.swarmplot(x=feat, y='tmpPrice', data=data, ax=ax)
    else:
        sns.scatterplot(x=feat, y='tmpPrice', data=data, ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel(feat)
    ax.set_xticks([])
    ax.set_yticks([])
    i+=1
fig.show();


# In[ ]:


mean_encoder(data,['Heating','HeatingQC','CentralAir','Electrical','Functional','MiscFeature'])
#data = data.drop(['Heating','HeatingQC','CentralAir','Electrical','Functional','MiscFeature'],axis=1)


# And transform MiscVal, because Miscval is value of the MiscFeature, in $ but as we transformed Prices with log of Prices, we need to transform also Miscval

# In[ ]:


data['MiscVal'] = np.log1p(data.MiscVal)


# ## 5.6 Features relative to the condition of Sale
# <a id='5.6' href="#6">next</a> - <a href="#menu">top</a><br>
# 
# MoSold: Month Sold (MM)   
# YrSold: Year Sold (YYYY)   
# SaleType: Type of sale   
# SaleCondition: Condition of sale   

# In[ ]:


sale_cols = ['MoSold','YrSold','SaleType','SaleCondition']


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 20))
i=0

for idx, feat in enumerate(sale_cols): 
    ax = axes[int(i / 3), i % 3] 
    if data[feat].dtype=='object':
        sns.swarmplot(x=feat, y='tmpPrice', data=data, ax=ax)
    else:
        sns.scatterplot(x=feat, y='tmpPrice', data=data, ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel(feat)
    ax.set_xticks([])
    ax.set_yticks([])
    i+=1
fig.show();


# In[ ]:


mean_encoder(data,['MoSold','YrSold','SaleType','SaleCondition'])
#data = data.drop(['MoSold','YrSold','SaleType','SaleCondition'],axis=1)


# In[ ]:


quick_predict(data.loc['train'].drop(['SalePrice','tmpPrice'],axis=1),data.loc['train'].tmpPrice)


# # 6 Predict SalePrice
# <a id='6' href="#6"></a>

# ## 6.1 Prepare X and y
# <a id='6.1' href="#6.2">next</a> - <a href="#menu">top</a><br>

# In[ ]:


train_lenght = len(data.loc['train'])


# In[ ]:


y = data.tmpPrice.loc['train']
y_target = data.tmpPrice.loc['test']


# In[ ]:


from sklearn.preprocessing import RobustScaler

X = data.drop(['SalePrice', 'tmpPrice'],axis=1)
X = pd.get_dummies(X)
X = RobustScaler().fit_transform(X)

X_target = X[train_lenght:,:]
X = X[:train_lenght,:]


# In[ ]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=0)


# ### Stacked regression

# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score,cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb


# In[ ]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X)
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[ ]:


from sklearn import linear_model

model1 = linear_model.Ridge(alpha=30)

score = rmsle_cv(model1)
print("model score: {:.4f} standard deviation ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


model2 = linear_model.Lasso(alpha=.00075)

score = rmsle_cv(model2)
print("model score: {:.4f} standard deviation ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


model3 = RandomForestRegressor(bootstrap=True, ccp_alpha=0.000001, criterion='mse',
                      max_depth=None, max_features=0.2, max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=1000, n_jobs=-1, oob_score=False,
                      random_state=None, verbose=0, warm_start=False)

score = rmsle_cv(model3)
print("model score: {:.4f} standard deviation ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


model4 =  GradientBoostingRegressor(n_estimators=500, 
                                   learning_rate=0.027, 
                                   max_features=0.7, 
                                   criterion='mse', 
                                   min_samples_leaf=4, 
                                   min_samples_split=4, 
                                   loss='huber',
                                   random_state =42)
score = rmsle_cv(model4)
print("gbr score: {:.4f} standard deviation ({:.4f})\n".format(score.mean(), score.std()))


# **XGBoost**

# In[ ]:


model5 =  xgb.XGBRegressor(learning_rate=0.04,n_estimators=1000,
                                     max_depth=2, min_child_weight=0,
                                     gamma=0, subsample=0.75,
                                     colsample_bytree=0.6,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1.2, seed=27,
                                     reg_alpha=0.0006)


# In[ ]:


score = rmsle_cv(model5)
print("xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


model1.fit(X, y)
model2.fit(X, y)
model3.fit(X, y)
model4.fit(X, y)
model5.fit(X, y)
predictions1 = model1.predict(X_target)
predictions2 = model2.predict(X_target)
predictions3 = model3.predict(X_target)
predictions4 = model4.predict(X_target)
predictions5 = model5.predict(X_target)
predictions = 0.2*predictions1+0.3*predictions2+0.1*predictions3+0.2*predictions4+0.2*predictions5


# In[ ]:


predictions = np.expm1(predictions)


# In[ ]:


submission = pd.DataFrame({'Id': data.loc['test'].index, 'SalePrice': predictions})


# In[ ]:


submission.to_csv("submission.csv", index=False)


# In[ ]:


submission.head()


# In[ ]:





# In[ ]:





# In[ ]:




