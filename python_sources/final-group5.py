#!/usr/bin/env python
# coding: utf-8

# # House Prices - Kaggle Copetitions
# ![image](https://realestateflippingtips.com/wp-content/uploads/2016/12/House-Buying-Tips.jpg)
# __Introduction__:
# ### Competition Description
# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.
# 
# Practice Skills
# Creative feature engineering 
# Advanced regression techniques like random forest and gradient boosting
# Acknowledgments
# The Ames Housing dataset was compiled by Dean De Cock for use in data science education. It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset. 
# 

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Preparing-environment-and-uploading-data" data-toc-modified-id="Preparing-environment-and-uploading-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Preparing environment and uploading data</a></span><ul class="toc-item"><li><span><a href="#Import-Packages" data-toc-modified-id="Import-Packages-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Import Packages</a></span></li><li><span><a href="#Load-Datasets" data-toc-modified-id="Load-Datasets-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Load Datasets</a></span></li></ul></li><li><span><a href="#Exploratory-Data-Analysis-(EDA)" data-toc-modified-id="Exploratory-Data-Analysis-(EDA)-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Exploratory Data Analysis (EDA)</a></span><ul class="toc-item"><li><span><a href="#First-Look-of-our-Data:" data-toc-modified-id="First-Look-of-our-Data:-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>First Look of our Data:</a></span></li><li><span><a href="#Some-Observations-from-the-STR-Details:" data-toc-modified-id="Some-Observations-from-the-STR-Details:-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>What is in theDataframe:</a></span></li></ul>

# ## Import the libraries and datasets 
# ### Import Packages

# In[ ]:


#Numerical and dataframe packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Visualisation packages
import matplotlib.pyplot as plt
import seaborn as sns

#ML and Stats packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy import stats
from scipy.stats import skew, skewtest
from subprocess import check_output

#ignore annoying warning (from sklearn and seaborn)
import warnings
warnings.simplefilter(action = 'ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
def ignore_warn(*args, **kwargs):
    pass

warnings.warn = ignore_warn 


# ### Import The Datasets

# In[ ]:


#Import the datasets
train = pd.read_csv("../input/train.csv", index_col=0)
df_test = pd.read_csv("../input/test.csv", index_col=0)
target_var = train['SalePrice']  #target variable

#Pop the target variable from the training dataset so that we can run transformations on it and look at it as a unique value 
df_train = train.copy()
train_sales = df_train.pop('SalePrice')

#Create new coulumn to distinguish the training and test sets for when we split them again
df_train['training_set'] = True
df_test['training_set'] = False

#Concatenate the testing and training dataset for data preprocessig 
df_full = pd.concat([df_train, df_test])
df_full.head()


# # Exploratory data analysis

# ### First look at the data.
# 
# We looked at the data in a myriad of ways. First was to get an overview of what we are working with a highlight reel.

# In[ ]:


df_full = pd.concat([df_train, df_test], keys=['train', 'test'])
df_full.describe
print(df_full.columns) # check column decorations
print('rows:', df_full.shape[0], ', columns:', df_full.shape[1]) # count rows of total dataset
print('rows in train dataset:', df_train.shape[0])
print('rows in test dataset:', df_test.shape[0])


# Perhaps looking at the dataframes individually could help. So we look at train and then look at the test.

# In[ ]:


train.describe
print(train.columns) # check column decorations
print('rows:', train.shape[0], ', columns:', train.shape[1]) # count rows of total dataset
print('rows in train dataset:', train.shape[0])


# In[ ]:


df_test.describe
print(df_test.columns) # check column decorations
print('rows:', df_test.shape[0], ', columns:', df_test.shape[1]) # count rows of total dataset
print('rows in train dataset:', df_test.shape[0])


# At first glance we can see that in the test set there's one row less than in the train set. Looking at the columns it is the SalePrice. That is to be expected as the SalePrice is what we will be looking for using the prediction. So we know what is in the dataframe. So the next step is to look what is in the dataframe in more detail. That's what we do in the next step.

# ## What is in the dataframe?
# 
# We have to look at the data and perhaps see all the finer parts of the dataframe that we will be making ts regression with. We took a preliminary analysis of the data. We look at the general characteristics of the dataframe. We look at the data types, counts, distinct, count nulls, missing ratio and uniques values of each field.

# In[ ]:


def preliminary_stats(df, pred=None): 
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_values_percent = (df.isnull().sum()/ obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt() 
    print('Data shape:', df.shape)
    
    if pred is None:
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing values_percent', 'uniques', 'skewness', 'kurtosis']
        str = pd.concat([types, counts, distincts, nulls, missing_values_percent, uniques, skewness, kurtosis], axis = 1)

    else:
        corr = df.corr()[pred]
        str = pd.concat([types, counts, distincts, nulls, missing_values_percent, uniques, skewness, kurtosis, corr], axis = 1, sort=False)
        corr_col = 'corr '  + pred
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing_values_percent', 'uniques', 'skewness', 'kurtosis', corr_col ]
    
    str.columns = cols
    dtypes = str.types.value_counts()
    print('___________________________\nData types:\n',str.types.value_counts())
    print('___________________________')
    return str


# In[ ]:


details = preliminary_stats(train, 'SalePrice')
display(details.sort_values(by='corr SalePrice', ascending=False))


# We have to look at the SalePrice and see what is happening with SalePrice as that is the most important aspect of the data as we are trying to create a prediction that will be as accurate as possible. Let's briefly look at the data concerning this. It is important to note that this particular column has no missing values and also 0 nulls. So it is the most complete data set that we could have.

# In[ ]:


print(train_sales.describe())


# Perhaps we need to visualise this data.

# In[ ]:


plt.figure()
plt.subplot(1, 2, 1)
plt.title("Sale Prices Dist")
sns.distplot(train_sales, fit=stats.norm)
plt.subplot(1, 2, 2)
stats.probplot(train_sales, plot=plt)
plt.show()
print("Skewness: %f" % train_sales.skew())
print("Kurtosis: %f" % train_sales.kurt())


# We saw earlier from preliminary assessment that the 'SalePrice' data has a skewness of 1,882876 and is not evenly distributed. The data also has a very high peak and could be better distributed much closer to the line in red. We will need to do a transformation of this data using a log transformation.

# In[ ]:


plt.figure()
plt.subplot(1, 2, 1)
plt.title("Sale Prices Dist")
sns.distplot(np.log(train_sales), fit=stats.norm)
plt.subplot(1, 2, 2)
stats.probplot(np.log(train_sales), plot=plt)
plt.show()
print("Skewness: %f" % np.log(train_sales).skew())
print("Kurtosis: %f" % np.log(train_sales).kurt())


# Much better looking and perhaps we need to keep this log transformation as it better represents more correct alignment to the line in red which is the probability line on the probability graph.

# ### Let's look at the rest of the data.
# 
# Let's look at each data value. Perhaps more importantly we can go according to the correlation to SalesPrice according to the earlier preliminary look. Let's begin with a correlation matrix to get started with this in depth look. 
# 
# We created a heatmap to properly visualise the data.

# In[ ]:


corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()


# Interestingly we see that there is a lot of other correlated values not just to SalesPrice but also to other data values. That is something that we could use later in feature engineering. For now we look at the top 6 features that have a high correlation to SalesPrice.

# In[ ]:


cols = corrmat.nlargest(7, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                 xticklabels=cols.values)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()


# So it seems that these figures are the ones we should be looking to track very closely as they seem to be consistantly coming up. Perhaps a different kind of visualisation could help us see these data features and their individual distributions. 
# 
# ### OverallQual the one with the highest correlation with SalesPrice
# 
# The following data visualisation is looking at how each of the top 6 variables interact with OverallQual and SalesPrice. We might perhaps also get to see some outliers in each of the data types.

# In[ ]:


viz = plt.figure(figsize=(20, 15))
sns.set(font_scale=1.5)

# (Corr= 0.790982) Box plot overallqual/salePrice
viz1 = viz.add_subplot(221); sns.boxplot(x='OverallQual', y='SalePrice', data=train[['SalePrice', 'OverallQual']])

# (Corr= 0.708624) GrLivArea vs SalePrice plot
viz2 = viz.add_subplot(222); 
sns.scatterplot(x = train.GrLivArea, y = train.SalePrice, hue=train.OverallQual, palette= 'Spectral')

# (Corr= 0.680625) GarageCars vs SalePrice plot
viz3 = viz.add_subplot(223); 
sns.scatterplot(x = train.GarageCars, y = train.SalePrice, hue=train.OverallQual, palette= 'Spectral')

# (Corr= 0.650888) GarageArea vs SalePrice plot
viz4 = viz.add_subplot(224); 
sns.scatterplot(x = train.GarageArea, y = train.SalePrice, hue=train.OverallQual, palette= 'Spectral')

viz5 = plt.figure(figsize=(16, 8))
viz6 = viz5.add_subplot(121); 
sns.scatterplot(y = train.SalePrice , x = train.TotalBsmtSF, hue=train.OverallQual, palette= 'YlOrRd')

viz7 = viz5.add_subplot(122); 
sns.scatterplot(y = train.SalePrice, x = train['1stFlrSF'], hue=train.OverallQual, palette= 'YlOrRd')

plt.tight_layout(); plt.show()


# Pretty easy to see the correlations but even more importantly is the visualisation of the outliers. But let's go into each one in detail and perhaps get to see how we can deal with the outliers. Maybe let's move onto GrLivArea.

# ### GRLivArea the next highest correlating variable

# In[ ]:


fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(121)
sns.scatterplot(x = train.GrLivArea, y = train.SalePrice, ax = ax)
plt.show()


# There are two properties there that are clear outliers. The two really big properties with the super low SalesPrice. They have to be adressed. Perhaps in this instance we can get rid of them right off the bat.

# In[ ]:


#Deleting outliers
train = train.drop(train[(train.GrLivArea>4000) & (train.SalePrice<300000)].index)

fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(122)
sns.scatterplot(x =train.GrLivArea, y = train.SalePrice, ax = ax)
plt.show()


# Well that's done. Let'smove onto the next one which is GarageCars. Now in looking at this one we need to consider that the Garage is usually where a carsleeps. In South African townships it is curiously, or maybe not so much, a living space that is given out for rent. A garage, however, is where people put their cars. It follows that you should be able to put your car or cars where there is space enough. The more cars the more parking spaces that are needed for the cars. Now it would be easier to visualise this using a box and whiskers plot if we want to see the bulk of the distribution.

# In[ ]:


fig = plt.figure(figsize=(20,5))
fig1 = fig.add_subplot(131); sns.boxplot(train.GarageCars)
plt.show()


# In[ ]:


df = train[['SalePrice', 'GarageCars']]
fig2 = fig.add_subplot(121); sns.regplot(x='GarageCars', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.GarageCars.corr(df['SalePrice'])))
plt.show()


# One to two cars is the norm. Three is also more common than 4.  And there they are. The outliers. 4 cars can park in a home that costs so little. So the clear outlier in this instance is 4. Not enough for them to justify why it shouldn't becounted as an outlier. We do not have enough data at the moment with just this one varable. Let's look at it in tandem with the next highest correlation feature which is GarageArea.

# In[ ]:


fig = plt.figure(figsize=(20,5))
fig1 = fig.add_subplot(132); sns.boxplot(train.GarageArea)
plt.show()


# In[ ]:


df = train[['SalePrice', 'GarageArea']]
fig2 = fig.add_subplot(121); sns.regplot(x='GarageArea', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.GarageArea.corr(df['SalePrice'])))
plt.show()


# Curiously we see that the data has more than a few outliers. and they are outliers for esimilar reasons. Low SalesPrice and high GarageArea. BUt also it might be better to engineer a new feature called GarageCars as these two variables are so closely linked.  It's not as though people are living in the garage and whatever else goes into the leftover space isn't in there to be stored. So GarageCars might be worth investigating.

# In[ ]:


fig = plt.figure(figsize=(20,5))
fig1 = fig.add_subplot(131); sns.boxplot(train.GarageCars)
fig2 = fig.add_subplot(132); sns.boxplot(train.GarageArea)
fig3 = fig.add_subplot(133); sns.boxplot(train.GarageCars, train.GarageArea)
plt.show()


# In[ ]:


df = train[['SalePrice', 'GarageArea', 'GarageCars']]
df['GarageAreaByCar'] = train.GarageArea/train.GarageCars
df['GarageArea_x_Car'] = train.GarageArea*train.GarageCars

fig = plt.figure(figsize=(20,5))
fig1 = fig.add_subplot(121); sns.regplot(x='GarageAreaByCar', y='SalePrice', data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.GarageAreaByCar.corr(df['SalePrice'])))

fig2 = fig.add_subplot(122); sns.regplot(x='GarageArea_x_Car', y='SalePrice', data=df); plt.legend(['Outliers'])
plt.text(x=-100, y=750000, s='Correlation to SalePrice: {:6.4f}'.format(df.GarageArea_x_Car.corr(df['SalePrice'])))
print('                                                                 Outliers:',(df.GarageArea_x_Car>=3700).sum())
df = df.loc[df.GarageArea_x_Car<3700]
sns.regplot(x='GarageArea_x_Car', y='SalePrice', data=df); plt.title('Correlation with SalePrice less outliers: {:6.4f}'.format(df.GarageArea_x_Car.corr(df['SalePrice'])))
plt.show()
del df


# Next variable to look at is TotalBasementSF. Now, what does this data have as a value? More importantly, what is it's correaltion to SalesPrice? According to our earlier preliminary look we know that it has a correlation of 0.614 to SalesPrice. We also know that it has a complete set of data, around 721 different values out of 1460 and is a measure of the total square feet of the basement. So how does it look?

# In[101]:


df = train[['SalePrice', 'TotalBsmtSF']]
fig2 = fig.add_subplot(121); sns.regplot(x='TotalBsmtSF', y='SalePrice', color="r", data=df)
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.TotalBsmtSF.corr(df['SalePrice'])))
plt.show()


# Doesn't really say much on it's own. So perhaps lets view it where we could see it's greatest impact. Perhaps along with another variable. The next highest variable which is 1stFlrSF.

# In[103]:


#Total Basement Area and 1st Floor
df = train[['SalePrice', 'TotalBsmtSF']]
df['TotalBsmtSFByBms'] = train.TotalBsmtSF/train['1stFlrSF']
df['TotalBsmtSF_x_Bsm'] = train.TotalBsmtSF*train['1stFlrSF']
fig = plt.figure(figsize=(20,5))
fig1 = fig.add_subplot(121); sns.regplot(x='TotalBsmtSFByBms', y='SalePrice', data=df);
plt.title('Correlation with SalePrice: {:6.4f}'.format(df.TotalBsmtSFByBms.corr(df['SalePrice'])))
fig2 = fig.add_subplot(122); sns.regplot(x='TotalBsmtSF_x_Bsm', y='SalePrice', data=df); plt.legend(['Outliers'])
plt.text(x=7e06, y=90000, s='Correlation with SalePrice: {:1.4f}'.format(df.TotalBsmtSF_x_Bsm.corr(df['SalePrice'])))

print('                                                             Outliers:',(df.TotalBsmtSF_x_Bsm>=0.9e07).sum())
df = df.loc[df.TotalBsmtSF_x_Bsm<0.9e07]
sns.regplot(x='TotalBsmtSF_x_Bsm', y='SalePrice', data=df); 
plt.title('Combined Correaltion is even better at ({:1.2f}) after outliers cut!'.format(df.TotalBsmtSF.corr(df.SalePrice)))
plt.text(x=7e06, y=50000, s='Correlation without Outliers: {:1.4f}'.format(df.TotalBsmtSF_x_Bsm.corr(df['SalePrice'])))
plt.show()
del df


# Now we are ready to continue on te path of perfecting the model. we have looked at the values with the highest correlation to SalesPrice and started engineering some features along the way. WIN!!!

# # Dealing with missing values 

# ### Missing values within categorical features

# The first step is to look at the datasets and ditinguish categorical from numerical features as they will require seperate preprocessing steps before the model can be fitted. Importantly, missing values need to be treated differently for categorical (object) and numeric features. Missing values within categorical features should either be encoded or removed altogether if the encoding is not clear.

# #### Determine the percentage of missing values within each categorical feature

# In[ ]:


# Isolate the categorical features from the concatenated dataset.
categoricals = df_full.select_dtypes(include=['object'])

# Determine the number of missing values within each category.
total = categoricals.isnull().sum().sort_values(ascending=False)
# The percentage of missing values. 
percent = round((categoricals.isnull().sum()/categoricals.isnull().count()).sort_values(ascending=False)*100,3)
# Create dataframe of missing value counts as well as their percentage of the whole. 
missing_categorical = pd.concat([total, percent], axis=1, keys=['Total Nulls', 'Percentage (%)'])
missing_categorical.head(30)


# 
# We then visualise the percentage of missing values within categorical features using a bar chart

# In[ ]:


# Create a list of categorical features in order to label each bar on the chart. 
labels=list(categoricals.apply(pd.Series.nunique).index)

# Plot the percentage of missing values for each categorical feature. 
percentge=categoricals.applymap(lambda x: pd.isnull(x)).sum()*100/categoricals.shape[0]
plt.figure(figsize=(15,5))
plt.bar(range(len(percentge)), percentge, align='center', color='r')
plt.xticks(range(len(labels)), labels, rotation=90)
plt.ylabel("Percentage of null values (%)")
plt.ylim([0,100])
plt.xlim([-1, categoricals.shape[1]])
plt.title("Percentage of nulls for each object feature (%)")
plt.show()


# #### Filling missing values within categorical features

# As we see in the bar chart above, many of the categorical features contain very high percentages of missing values. We are particularly intrested in those which contain more than 80% null values. Before we can begin filling these missing values we must firt determine what each actually means. We must separate those missing values which represent a lack of a particular feature, from those which simply mean something other than the possiblities listed. For example, the missing values within the category 'Fence', probably do not mean extremely poor security. It is more likely that the house is enclosed by a boundry wall. And the missing values within the feature 'Electrical', are unlikely to mean that the house has no access to electricity, rather that it is from a source other than the avalible options listed. This is distict from those within features such as 'GarageType' or 'PoolQC', where missing values likely represent the lack of that amenity. The third important consideration is that particular categorical features could be treated as numerical because the categories are represented by integers, such as 'MSSubClass' or dates.  

# In[ ]:


# Plot bar plot for each feature to show the unique categories within each feature and their resepective frequencies.

for category in df_full.dtypes[df_full.dtypes == 'object'].index:
    sns.countplot(y=category, data=df_full)
    plt.title('Frequency of categories within ' + category)
    plt.show()


# In[ ]:


# Print a list of the unique categories within each object feature. 
for category in df_full.dtypes[df_full.dtypes == 'object'].index:
    print("Unique values for {}:".format(category))
    print(missing.loc[:,category].unique())
    print(" ")


# We begin with the third consideration mentioned above, these feature contain integers and if they were left as such, the model would assume that they are numerical features. Therefore we simply convert the data type within these features to strings, preventing the model for mistaking them as numerical. 

# In[ ]:


# Convert numeric categories from integers to strings.
df_full['MSSubClass'] = df_full['MSSubClass'].apply(str)
df_full['YrSold'] = df_full['YrSold'].astype(str)
df_full['MoSold'] = df_full['MoSold'].astype(str)
df_full['YearBuilt'] = df_full['YearBuilt'].astype(str)
df_full['YearRemodAdd'] = df_full['YearRemodAdd'].astype(str) 


# Now we deal with the categories within which the missing values likely do not represent the absence of the amenity. Therefore we fill these values based on our assumptions about what is most suitable given a particular category. The feature 'MSZoning' is an exceptional case because it is dependant on the 'MSSubClass' within which the home is located. For these missing values, we group the zones according to their subclass and then fill them with the most frequent category accordingly. 

# In[ ]:


# Fill missing values within these features with the most likely category they belong to.  
df_full['Functional'] = df_full['Functional'].fillna('Typ') 
df_full['Electrical'] = df_full['Electrical'].fillna("SBrkr") 
df_full['KitchenQual'] = df_full['KitchenQual'].fillna("TA") 


# In[ ]:


# Fill these featues missing values with the MODE as it is the most frequent value, and therfore a reasonable assumption. 
df_full['Exterior1st'] = df_full['Exterior1st'].fillna(df_full['Exterior1st'].mode()[0]) 
df_full['Exterior2nd'] = df_full['Exterior2nd'].fillna(df_full['Exterior2nd'].mode()[0])
df_full['SaleType'] = df_full['SaleType'].fillna(df_full['SaleType'].mode()[0])

# Group 'MSZoning' in order to fill nulls with most frequently observed category within their 'MSSubClass'.  
df_full['MSZoning'] = df_full.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))


# Finally we deal with the missing values which likely represent the lack of the amenity being considered. 

# In[ ]:


# Homes without pools
df_full["PoolQC"] = df_full["PoolQC"].fillna("None")


# Nulls in features relating to a garage are probably due to the lack of one, we fill these values with either 0 or 'None'.  
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    df_full[col] = df_full[col].fillna(0)

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    df_full[col] = df_full[col].fillna('None')

    
# Once again, the missing values within basement categories are likely because the house has no basement.  

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df_full[col] = df_full[col].fillna('None')


# Run through all categorical features to fill any remaining missing values with 'None'. 

# In[ ]:


categorical = []
for i in df_full.columns:
    if df_full[i].dtype == object:
        categorical.append(i)
df_full.update(df_full[categorical].fillna('None'))
print(categorical)


# ### Missing values within numerical features

# Once again we fill the missing values within numerical features based on our assumptions about what is most suitable. Typically, filling these values will zero should be sufficient as they likely represent the lack of information or the lack of the amenity being considered. However, 'LotFrontage', is a special case as it is likely not equal to zero, and could fluctuate drastically depending on the particular'Neighborhood' the home belongs to. We use the median value within each 'Neighborhood' as a fair prediction of the 'LotFrontage'. 

# In[ ]:


# Group homes according to their 'Neighborhood' in order to estimate their 'LotFrontage' with the local median.
df_full['LotFrontage'] = df_full.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# All other missing values within the numeric features are replaced with zero. 
numeric_data_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] # All possible numeric data types to consider.
numerical = []
for i in df_full.columns:
    if df_full[i].dtype in numeric_data_types:
        numerical.append(i)
df_full.update(df_full[numerical].fillna(0))
numerical[1:10]


# ### Feature Engineering

# #### Removing features

# We will continue to clean the dataset by removing any features that are unlikely to make any great contributions to the predictive ability of the model. Features such as whether or not there is a fireplace, pool, or those which have been included in the miscellaneous category probably do not have a huge influence on the sale price of a home. Therefore we remove them. 

# In[ ]:


# We remove any feature that we believe will not influence the prediction of the house price. 
df_full = df_full.drop(['Utilities', 'Street', 'PoolQC','FireplaceQu','MiscFeature'], axis=1)


# #### Adding new features

# Here we add featues which we believe will assist the model in predicting the home's sale price by lumping any features that are related, and therfore likely correlated, into a new feature column. 

# In[ ]:


# Create categorical boolean features based on the presence or absence of particular amenities

df_full['pool?'] = df_full['PoolArea'].apply(lambda x: 'True' if x > 0 else 'False')
df_full['2ndfloor?'] = df_full['2ndFlrSF'].apply(lambda x: 'True' if x > 0 else 'False')
df_full['garage?'] = df_full['GarageArea'].apply(lambda x: 'True' if x > 0 else 'False')
df_full['basement?'] = df_full['TotalBsmtSF'].apply(lambda x: 'True' if x > 0 else 'False')
df_full['fireplace?'] = df_full['Fireplaces'].apply(lambda x: 'True' if x > 0 else 'False')


# In[ ]:


# Combine related features to reduce the number of features the model considers and to remove correlation between predictors. 

df_full['YrBlt_Remod']=df_full['YearBuilt']+df_full['YearRemodAdd']
df_full['Total_area']=df_full['TotalBsmtSF'] + df_full['1stFlrSF'] + df_full['2ndFlrSF']

df_full['Total_finished'] = (df_full['BsmtFinSF1'] + df_full['BsmtFinSF2'] +
                                 df_full['1stFlrSF'] + df_full['2ndFlrSF'])

df_full['Total_Bathrooms'] = (df_full['FullBath'] + (0.5 * df_full['HalfBath']) + # *0.5 for half bathrooms
                               df_full['BsmtFullBath'] + (0.5 * df_full['BsmtHalfBath']))

df_full['Total_porch'] = (df_full['OpenPorchSF'] + df_full['3SsnPorch'] +
                              df_full['EnclosedPorch'] + df_full['ScreenPorch'] +
                              df_full['WoodDeckSF'])

# Remove the features that have been combined in the new features

df_full = df_full.drop(['YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF', '2ndFlrSF','BsmtFinSF1','BsmtFinSF2','1stFlrSF','2ndFlrSF','FullBath','HalfBath','BsmtFullBath','BsmtHalfBath','OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','WoodDeckSF'], axis=1)


# #### Encode categorical features

# In[ ]:





# In[ ]:


# 
for feature in df_full.select_dtypes(include=['object']).columns: # do it for categorical columns only
    replaceValue=df_full.loc[:,feature].value_counts().idxmax()
    df_full[feature]=df_full.loc[:,feature].fillna(replaceValue)
df_full = pd.get_dummies(df_full).reset_index(drop=True)
print(df_full.shape)
print(df_full.info())


# ### Normalising distribution of numeric features

# It is important to normalise the distribution of numeric features to obtain the most predictive model. Before we normalise the data, we must investigate the features distribution to assess their normalily based on a measure of skewness. We begin with the begin by looking at the target features distribution, and then the measures for all possible predictor features. 

# #### Distribution of the target variable 

# In[ ]:


# Visualise the original distribution of the target variable using a histogram with a trend line.
sns.distplot(target_var, bins = 30, )
plt.title('Sale Price before normalisation')

# Print the measure of Skewness and Kurtosis for the target variable before transformation. 
print("Skewness: %f" % target_var.skew())
print("Kurtosis: %f" % target_var.kurt())


# Clearly the target variable is not normally distributed, we see that it is positively skewed. To correct this we log transform the target variable and then visualise the results of this transformation as we before. Normalising positively skewed data, where there is skewness towards large values, is very successfully acheived by log transformation.  

# In[ ]:


# log transform target variable.
target_normal = np.log1p(target_var)

# Visualise normality of the normalised target. 
sns.distplot(target_normal, bins = 30, )
plt.title('Normalised Sale Price')
# Print the measure of Skewness and Kurtosis for the target variable post transformation.
print("Skewness: %f" % target_normal.skew())
print("Kurtosis: %f" % target_normal.kurt())


# We can see that the target variable is far closer to a normal distribution once it has been log transformed. 

# #### Distribution of the numeric predictor features

# Now we look at measures of skewness and kurtosis of all numeric features which we will be using to predict the target variable.  Lognormal distribution is an approximation of a normal distribution, as can be seen from the log transformation of the target variable. It is important to normalise skewed data as it increases the linear relationships between the various predictors and the target. Thereby increasing the predictive ability of the model once fitted. 

# In[ ]:


# Measures of skewness and kurtosis for numeric features in the full dataset prior to transformation. 
for numeric_feature in df_full.select_dtypes(exclude=['object']):
    print(numeric_feature)
    print("Skewness: %f" % df_full[numeric_feature].skew())
    print("Kurtosis: %f" % df_full[numeric_feature].kurt())
    print(" ")


# We plot the measure of skewness for each numerical feature within the training set (as a subset of the full dataset) so that we may visualise the difference made by log transforming the data. Although we are only visualisng the original training set, which has not been preprocessed to the extent that the full dataset has, it gives us insight into the affect this type of data transformation has on the normality of the features. It is far more difficult to observe these differences on the processed dataset, as the encoding of categorical variables  We will however also log transform the full dataset for use in training the model. 

# In[ ]:


# Then we create a bar plot of the measures of skewness for each numeric feature within the training set before transformation.
plt.figure(figsize=(15,5))

numerical_features = df_train.select_dtypes(exclude=['object'])
# Once again we compute skewness for all possible predictors. 
numerical_skewness = numerical_features.apply(lambda x: skew(x)) 
np.abs(numerical_skewness).plot.bar()
plt.title("Skewness calculated on original training set's numeric features")
plt.ylabel("Skewness")

#Can observe skewness at two different thresholds. 

#skewness_feats = skewness_feats[skewness_feats > 0.75]
skewness_feats = skewness_feats[skewness_feats > 1]


# From the graph above we can see that many numeric features within the original training dataset are quite severly skewed. We log transform these features and plot the results to see how the transformation has altered the skewness of each feature. 
# 

# In[ ]:


# Then we create a bar plot of the measures of skewness for each numeric feature within the training set post transformation.
plt.figure(figsize=(15,5))


numerical_features = df_train.select_dtypes(exclude=['object'])
# Once again we compute skewness for all possible predictors once they have been transformed. 
numerical_skewness = numerical_features.apply(lambda x: skew(np.log1p(x))) 
np.abs(numerical_skewness).plot.bar()
plt.title("Skewness calculated on original training set's numeric features once log transformed")
plt.ylabel("Skewness")

#Can observe skewness at two different thresholds. 

#skewness_feats = skewness_feats[skewness_feats > 0.75]
skewness_feats = skewness_feats[skewness_feats > 1]


# Here we see that in general the log transformation reduces the skewness of each of the columns, although unfortunately the training set still includes numerous numeric categories, and other categories which have been removed from the full dataset during preprocessing. 

# Now we will log transform the numeric predictors within the full dataset, which has been previously processed. 

# In[ ]:


# log transform numeric predictors within full dataset
skewness_feats = skewness_feats.index
df_full_norm = df_full.copy()
df_full_norm[skewness_feats] = np.log1p(df_full_norm[skewness_feats])


# Once the numeric features within the full dataset (which has undergone preprocessing) have been transformed we can compare the new measures of skewness and kurtosis to those before the transformation. As we can see, the measures of skewness have been substantially reduced by log transformation, bringing the distribution of these numeric features closer to normal. 

# In[ ]:


# Measures of skewness and kurtosis for numeric features in the full dataset post log transformation. 
for numeric_feature in df_full_norm.select_dtypes(exclude=['object']):
    print(numeric_feature)
    print("Skewness: %f" % df_full_norm[numeric_feature].skew())
    print("Kurtosis: %f" % df_full_norm[numeric_feature].kurt())
    print(" ")


# We have now completed the data preprocessing and can split the dataset back into training and testing sets and begin training the model. 

# ### Splitting data into training and testing sets 

# Before we can train our model we use the column previously created, 'training_set' to separate the full processed dataset back to the original train and test datasets. Then we use scikitlearn's train_test_split to create smaller train and test subsets from the original training set data. This is done because the training set includes the target variable, therefore we are able to test our model based on the actual target values within the subset test dataset. 

# In[ ]:


# Separation of dataset back to original groups based on boolean column we created before concatenating the datasets. 

df_train_norm = df_full_norm[df_full_norm['training_set']==True]
df_test_norm = df_full_norm[df_full_norm['training_set']==False]

# We then drop the columns that we originally created to distinguish them. 

df_train_norm = df_train_norm.drop('training_set', axis=1)
df_test_norm = df_test_norm.drop('training_set', axis=1)


# Now we can create a test and train subset from our normalised training set which we will use to train our data, as well as compare the predictions made on the test subset with the actual target variables within the test subset.

# In[ ]:


# Train and test subsets from original training set. 
X_train, X_test, y_train, y_test = train_test_split(df_train_norm, target_normal, test_size=0.2, random_state=42)


# ### Fitting models and making predictions

# In[ ]:


#Prediction


# In[ ]:


#Random forest goes first


# In[ ]:


rf_skewed = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf_skewed.fit(df_train_skewed, target_skewed)

cv_num=4
scores = cross_val_score(rf_skewed, df_train_skewed, target_skewed, cv=cv_num, scoring='neg_mean_squared_error')
print("cross val scores (skewed dataset):" , np.sqrt((-scores)))
print('average root mean squared log error (skewed dataset) =', np.mean(np.sqrt(-scores)))
preds_skewed = np.expm1(rf_skewed.predict(df_test_skewed))


# In[ ]:


#explore whether the distribution for the training test and the test set is similar. 
#look at its residual plot.


# In[ ]:


plt.hist(preds_skewed)
plt.hist(target_var, alpha = 0.3)
plt.show()
fig, ax = plt.subplots(figsize=(15,15))  
sns.residplot(y_test, y_test-rf_skewed.predict(X_test), lowess=True, color="b")
plt.title("Residual plot for random forest prediction")
plt.show()


# In[ ]:


#Lasso Regression (second attempt)


# In[ ]:


from  sklearn.linear_model import LassoCV
model_lasso = LassoCV(alphas = np.logspace(-4, 10, 2), cv = 4, n_jobs=4, max_iter=1000000000)
model_lasso.fit(df_train_skewed, target_skewed)
#scores = cross_val_score(model_lasso, df_train_skewed, target_skewed, cv=cv_num, scoring='neg_mean_squared_error')
print("cross val scores (lasso):" , np.sqrt(-scores))
print('average root mean squared log error (lasso)=', np.mean(np.sqrt(-scores)))
preds_lasso_skewed = np.expm1(model_lasso.predict(df_test_skewed))
my_submission_nopipe_lasso_skewed = pd.DataFrame({'Id': df_test_skewed.index, 'SalePrice': preds_lasso_skewed})
my_submission_nopipe_lasso_skewed.to_csv('submission_nopipe_lasso_skewed.csv', index=False)


# In[ ]:


###conda install py-xgboost


# In[ ]:


#Blend Models to get final score


# In[ ]:


# xgboost (final attempt)
#based on  https://www.kaggle.com/fiorenza2/journey-to-the-top-10 kernel
#and https://www.kaggle.com/opanichev/ensemble-of-4-models-with-cv-lb-0-11489 .


# In[ ]:


from xgboost.sklearn import XGBRegressor
reoptimize=False
##NB optimisation happening on kaggle
nestim = 1921
if(reoptimize):
    learning_rate_grid = np.linspace(0.05, 0.05, 1)
    max_depth_grid = np.linspace(1, 5, 5, dtype='int')
    n_estimators_grid = np.linspace(1000, 2500, 250, dtype='int')

    xgbgrid={
            #'learning_rate' : learning_rate_grid, 
            #'max_depth' : max_depth_grid,
            'n_estimators' : n_estimators_grid }
    xgb_cv=GridSearchCV(XGBRegressor(colsample_bytree=0.2), xgbgrid, cv=cv_num, refit=True, n_jobs=-1, scoring='neg_mean_squared_log_error')

    xgb_cv.fit(df_train_skewed, target_skewed)
    print(xgb_cv.best_params_)
    nestim = xgb_cv.best_params_['n_estimators']


# In[ ]:


from sklearn.base import BaseEstimator, RegressorMixin
from xgboost.sklearn import XGBRegressor
class CustomEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressors=None):
        self.regressors = regressors

    def fit(self, X, y):
        for regressor in self.regressors:
            regressor.fit(X, y)

    def predict(self, X):
        self.predictions_ = list()
        for regressor in self.regressors:
            self.predictions_.append((regressor.predict(X).ravel()))
        return (np.mean(self.predictions_, axis=0))
    
xgb1 = XGBRegressor(colsample_bytree=0.02,
                 learning_rate=0.05,
                 max_depth=3,
                 n_estimators=nestim
                )

xgb2 = XGBRegressor(colsample_bytree=0.2,
                 learning_rate=0.05,
                 max_depth=3,
                 n_estimators=nestim,
                seed = 1234
                )

xgb3 = XGBRegressor(colsample_bytree=0.2,
                 learning_rate=0.05,
                 max_depth=3,
                 n_estimators=nestim,
                seed = 1337
                )

xgb_ens = CustomEnsembleRegressor([xgb1,xgb2,xgb3])
xgb_ens.fit(df_train_skewed, target_skewed)

scores = cross_val_score(cv=cv_num,estimator=xgb_ens,X = df_train_skewed,y = target_skewed, n_jobs = -1, scoring='neg_mean_squared_error')
print("cross val scores (xgboost ensemble):" , np.sqrt(-scores))
print('average root mean squared log error (xgboost ensemble)=', np.mean(np.sqrt(-scores)))
preds_xgb_skewed = np.expm1(xgb_ens.predict(df_test_skewed))
my_submission_xgb_skewed = pd.DataFrame({'Id': df_test.index, 'SalePrice': preds_xgb_skewed})
my_submission_xgb_skewed.to_csv('submission_xgb_skewed.csv', index=False)


# In[ ]:


df_test.index


# In[ ]:


#FINAL MODEL
#combination of two best 


# In[ ]:


preds_avg=(preds_lasso_skewed+preds_xgb_skewed)/2
my_submission_avg = pd.DataFrame({'Id': df_test.index, 'SalePrice': preds_avg})
my_submission_avg.to_csv('submission_FINAL.csv', index=False)


# In[ ]:





# In[ ]:




