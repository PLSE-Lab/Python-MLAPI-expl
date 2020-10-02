#!/usr/bin/env python
# coding: utf-8

# 

# Import Necessary Packages

# In[ ]:


#invite people for the Kaggle party
import pandas as pd
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
df_train = pd.read_csv('../input/train.csv')


# In[ ]:


#check the decoration
df_train.columns
df_train.head(10)
df_train.describe(include='all')
df_train.describe(include=[np.number])
df_train.describe(include=[np.object])


# # 1. So... What can we expect?
# 
# In order to understand our data, we can look at each variable and try to understand their meaning and relevance to this problem. I know this is time-consuming, but it will give us the flavour of our dataset.
# 
# In order to have some discipline in our analysis, we can create an Excel spreadsheet with the following columns:
# * <b>Variable</b> - Variable name.
# * <b>Type</b> - Identification of the variables' type. There are two possible values for this field: 'numerical' or 'categorical'. By 'numerical' we mean variables for which the values are numbers, and by 'categorical' we mean variables for which the values are categories.
# * <b>Segment</b> - Neighboorhood,Space,quality
# * <b>Expectation</b> - Our expectation about the variable influence in 'SalePrice'. We can use a categorical scale with 'High', 'Medium' and 'Low' as possible values.
# * <b>Conclusion</b> - Our conclusions about the importance of the variable, after we give a quick look at the data. We can keep with the same categorical scale as in 'Expectation'.
# * <b>Comments</b> - Any general comments that occured to us.
# 
# Below list of variables I expected to be important while buying house
# 
# * OverallQual (which is a variable that I don't like because I don't know how it was computed; a funny exercise would be to predict 'OverallQual' using all the other variables available).
# * YearBuilt.
# * TotalBsmtSF.
# * GrLivArea.
# 
# I ended up with two 'building' variables ('OverallQual' and 'YearBuilt') and two 'space' variables ('TotalBsmtSF' and 'GrLivArea'). This might be a little bit unexpected as it goes against the real estate mantra that all that matters is 'location, location and location'. 
# 

# # 2. First things first: analysing 'SalePrice'
# 
# 'SalePrice' is the reason of our quest. 
# 
# 

# In[ ]:


#descriptive statistics summary
df_train['SalePrice'].describe()


# Visualise through histogram

# In[ ]:



#histogram
sns.distplot(df_train['SalePrice']);


# *'Observation below:*
# 
# * *<b>Deviate from the normal distribution.</b>*
# * *<b>Have appreciable positive skewness.</b>*
# * *<b>Show peakedness.</b>*
# 
# *This is getting interesting! 'SalePrice', could you give me your body measures?'*
# Lets check what is skewness and kutosis
# Skewness : Defines where is the symetry leaning towards (so vertical leaning)
# Kutosis: Defines tthe flatness of the curve

# In[ ]:


#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())


# 

# # 'SalePrice', her buddies and her interests

# Relation between GrLivArea and Price

# ### Relationship with numerical variables

# In[ ]:


#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# *Hmmm... It seems that 'SalePrice' and 'GrLivArea' are really old friends, with a <b>linear relationship.</b>*
# 
# *And what about 'TotalBsmtSF'?*

# In[ ]:


#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# 

# 1. ### Relationship with categorical features

# In[ ]:


#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# 

# In[ ]:


var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
#plt.xticks(rotation=90);


# *Although it's not a strong tendency, I'd say that 'SalePrice' is more prone to spend more money in new stuff than in old relics.*
# 
# <b>Note</b>: we don't know if 'SalePrice' is in constant prices. Constant prices try to remove the effect of inflation. If 'SalePrice' is not in constant prices, it should be, so than prices are comparable over the years.

# ### Summary
# 
# Stories aside, we can conclude that:
# 
# * 'GrLivArea' and 'TotalBsmtSF' seem to be linearly related with 'SalePrice'. Both relationships are positive, which means that as one variable increases, the other also increases. In the case of 'TotalBsmtSF', we can see that the slope of the linear relationship is particularly high.
# * 'OverallQual' and 'YearBuilt' also seem to be related with 'SalePrice'. The relationship seems to be stronger in the case of 'OverallQual', where the box plot shows how sales prices increase with the overall quality.
# 
# We just analysed four variables, but there are many other that we should analyse. The trick here seems to be the choice of the right features (feature selection) and not the definition of complex relationships between them (feature engineering).
# 
# That said, let's separate the wheat from the chaff.

# # 3. Keep calm and work smart

# 

# Lets start with:
# * Correlation matrix (heatmap style).
# * 'SalePrice' correlation matrix (zoomed heatmap style).
# * Scatter plots between the most correlated variables (move like Jagger style).

# #### Correlation matrix (heatmap style)

# In[ ]:


#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In my opinion, this heatmap is the best way to get a quick overview of your data and its relationships. (Thank you @seaborn!)
# 
# At first sight, there are two red colored squares that get my attention. The first one refers to the 'TotalBsmtSF' and '1stFlrSF' variables, and the second one refers to the 'Garage*X*' variables. Both cases show how significant the correlation is between these variables. Actually, this correlation is so strong that it can indicate a situation of multicollinearity. If we think about these variables, we can conclude that they give almost the same information so multicollinearity really occurs. Heatmaps are great to detect this kind of situations and in problems dominated by feature selection, like ours, they are an essential tool.
# 
# Another thing that got my attention was the 'SalePrice' correlations. We can see our well-known 'GrLivArea', 'TotalBsmtSF', and 'OverallQual' saying a big 'Hi!', but we can also see many other variables that should be taken into account. That's what we will do next.

# #### 'SalePrice' correlation matrix (zoomed heatmap style)

# In[ ]:


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# According to our crystal ball, these are the variables most correlated with 'SalePrice'. My thoughts on this:
# 
# * 'OverallQual', 'GrLivArea' and 'TotalBsmtSF' are strongly correlated with 'SalePrice'. Check!
# * 'GarageCars' and 'GarageArea' are also some of the most strongly correlated variables. However, Garagecars and Garage area are interrelated. You'll never be able to distinguish them. Therefore, we just need one of these variables in our analysis (we can keep 'GarageCars' since its correlation with 'SalePrice' is higher).
# * 'TotalBsmtSF' and '1stFloor' also intererelated. We can keep 'TotalBsmtSF' just to say that our first guess was right (re-read 'So... What can we expect?').
# * Ah... 'YearBuilt'... It seems that 'YearBuilt' is slightly correlated with 'SalePrice'. Honestly, it scares me to think about 'YearBuilt' because I start feeling that we should do a little bit of time-series analysis to get this right. 
# 
# Let's proceed to the scatter plots.

# #### Scatter plots between 'SalePrice' and correlated variables 

# seaborn again does magic by providing mutiple information

# In[ ]:


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();


# Although we already know some of the main figures, this mega scatter plot gives us a reasonable idea about variables relationships.
# 
# One of the figures we may find interesting is the one between 'TotalBsmtSF' and 'GrLiveArea'. In this figure we can see the dots drawing a linear line, which almost acts like a border. It totally makes sense that the majority of the dots stay below that line. Basement areas can be equal to the above ground living area, but it is not expected a basement area bigger than the above ground living area (unless you're trying to buy a bunker).
# 
# The plot concerning 'SalePrice' and 'YearBuilt' can also make us think. In the bottom of the 'dots cloud', we see what almost appears to be a shy exponential function (be creative). We can also see this same tendency in the upper limit of the 'dots cloud' (be even more creative). Also, notice how the set of dots regarding the last years tend to stay above this limit (I just wanted to say that prices are increasing faster now).
# 
# 

# # 4. Missing data
# 
# Important questions when thinking about missing data:
# 
# * How prevalent is the missing data?
# * Is missing data random or does it have a pattern?
# 
# The answer to these questions is important for practical reasons because missing data can imply a reduction of the sample size. This can prevent us from proceeding with the analysis. Moreover, from a substantive perspective, we need to ensure that the missing data process is not biased and hidding an inconvenient truth.

# In[ ]:


#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# Let's analyse this to understand how to handle the missing data.
# 
# We'll consider that when more than 15% of the data is missing, we should delete the corresponding variable and pretend it never existed. This means that we will not try any trick to fill the missing data in these cases. According to this, there is a set of variables (e.g. 'PoolQC', 'MiscFeature', 'Alley', etc.) that we should delete. The point is: will we miss this data? I don't think so. None of these variables seem to be very important, since most of them are not aspects in which we think about when buying a house (maybe that's the reason why data is missing?). Moreover, looking closer at the variables, we could say that variables like 'PoolQC', 'MiscFeature' and 'FireplaceQu' are strong candidates for outliers, so we'll be happy to delete them.
# 
# In what concerns the remaining cases, we can see that 'Garage*X*' variables have the same number of missing data. I bet missing data refers to the same set of observations (although I will not check it; it's just 5% . Since the most important information regarding garages is expressed by 'GarageCars' and considering that we are just talking about 5% of missing data, I'll delete the mentioned 'Garage*X*' variables. The same logic applies to 'Bsmt*X*' variables.
# 
# Regarding 'MasVnrArea' and 'MasVnrType', we can consider that these variables are not essential. Furthermore, they have a strong correlation with 'YearBuilt' and 'OverallQual' which are already considered. Thus, we will not lose information if we delete 'MasVnrArea' and 'MasVnrType'.
# 
# Finally, we have one missing observation in 'Electrical'. Since it is just one observation, we'll delete this observation and keep the variable.
# 
# In summary, to handle missing data, we'll delete all the variables with missing data, except the variable 'Electrical'. In 'Electrical' we'll just delete the observation with missing data.

# In[ ]:


#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max() #just checking that there's no missing data missing...


# # Out liars!
# 
# To find outliars standarise the data

# ### Univariate analysis

# The primary concern here is to establish a threshold that defines an observation as an outlier. To do so, we'll standardize the data. In this context, data standardization means converting data values to have mean of 0 and a standard deviation of 1.

# In[ ]:


#standardizing data: xi-xmean/standar deviation
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# ### Bivariate analysis

# In[ ]:


#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# What has been revealed:
# 
# * The two values with bigger 'GrLivArea' seem strange and they are not following the crowd. We can speculate why this is happening. Maybe they refer to agricultural area and that could explain the low price. I'm not sure about this but I'm quite confident that these two points are not representative of the typical case. Therefore, we'll define them as outliers and delete them.
# * The two observations in the top of the plot are those 7.something observations that we said we should be careful about. They look like two special cases, however they seem to be following the trend. For that reason, we will keep them.

# In[ ]:


#deleting points
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)


# In[ ]:


#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# We can feel tempted to eliminate some observations (e.g. TotalBsmtSF > 3000) but I suppose it's not worth it. We can live with that, so we'll not do anything.

# # 5. Getting hard core

# ### In the search for normality

# The point here is to test 'SalePrice' in a very lean way. We'll do this paying attention to:
# 
# * <b>Histogram</b> - Kurtosis and skewness.
# * <b>Normal probability plot</b> - Data distribution should closely follow the diagonal that represents the normal distribution.

# In[ ]:


#histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# Ok, 'SalePrice' is not normal. It shows 'peakedness', positive skewness and does not follow the diagonal line.
# 
# But everything's not lost. A simple data transformation can solve the problem. This is one of the awesome things you can learn in statistical books: in case of positive skewness, log transformations usually works well. 

# In[ ]:


#applying log transformation will help to form a bell shape curve
df_train['SalePrice'] = np.log(df_train['SalePrice'])


# In[ ]:


#transformed histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# Done! Let's check what's going on with 'GrLivArea'.

# In[ ]:


#histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)


# Tastes like skewness... *Avada kedavra!*

# In[ ]:


#data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])


# In[ ]:


#transformed histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)


# Next, please...

# In[ ]:


#histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
#fig = plt.figure()
#res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)


# * Something that, in general, presents skewness.
# * A significant number of observations with value zero (houses without basement).
# * A big problem because the value zero doesn't allow us to do log transformations.
# 
# To apply a log transformation here, we'll create a variable that can get the effect of having or not having basement (binary variable). Then, we'll do a log transformation to all the non-zero observations, ignoring those with value zero. This way we can transform data, without losing the effect of having or not basement.
# 
# I'm not sure if this approach is correct. It just seemed right to me. That's what I call 'high risk engineering'.

# In[ ]:


#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1


# In[ ]:


#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])


# In[ ]:


#histogram and normal probability plot
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ##Accuracy score is for classification
from sklearn.metrics import explained_variance_score ## This is for linear regression

df_test=pd.read_csv('../input/test.csv')

#Create X_test
X_test=df_test[['OverallQual','YearBuilt','GrLivArea','TotalBsmtSF','GarageCars']]
X_test.head()

#Checking missing value in test data set
total_missing_values_X_test=X_test.isnull().sum().sort_values(ascending=False)
total_missing_values_X_test

#Checking the missing Garage Cars record
X_test[X_test['GarageCars'].isnull()]

#Updating the missing value to mean value
X_test.at[660,'TotalBsmtSF'] = 1046.12

#Verifying the missing value in TotalBsmtSF
X_test[X_test['TotalBsmtSF'].isnull()]


X_test.at[1116,'GarageCars'] = 2

#Checking missing value in test data set again
total_missing_values_X_test=X_test.isnull().sum().sort_values(ascending=False)
total_missing_values_X_test

X_train=df_train[['OverallQual','YearBuilt','GrLivArea','TotalBsmtSF','GarageCars']]
Y_train=df_train[['SalePrice']]
                  
#Checking missing value in test data set again
total_missing_values_X_train=X_train.isnull().sum().sort_values(ascending=False)
total_missing_values_X_train

regressor=LinearRegression()
regressor.fit(X_train,Y_train)
features=['OverallQual','YearBuilt','GrLivArea','TotalBsmtSF','GarageCars']

Y_pred=regressor.predict(X_train)
Y_pred_test=regressor.predict(X_test[features])
Y_pred_df=pd.DataFrame(Y_pred) 
Y_pred_test_df=pd.DataFrame(Y_pred_test) 
Y_pred_df

explained_variance_score(Y_train,Y_pred_df)
#Y_pred_test_df=pd.DataFrame({'SalesPrice':Y_pred_test[0:]})
Y_pred_test_df.to_csv('sample_submission_pred.csv', index=False)
#Y_pred_test_df=pd.DataFrame({'Id':df_test['Id'],'SalesPrice':Y_pred_test})
#X_test['OverallQual','YearBuilt','GrLivArea','TotalBsmtSF','GarageCars']
#Y_pred_test_df=pd.DataFrame({'Id':df_test['Id'],'Sales':Y_pred_test})












# In[ ]:


## Now with test data predicting values 

