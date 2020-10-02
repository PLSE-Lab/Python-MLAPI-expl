#!/usr/bin/env python
# coding: utf-8

# # Data Science on Housing Prices

# ## Analyzing a Housing dataset
# This Python notebook demonstrates a step-by-step Data Science experiment using tools like Numpy and Pandas to preprocess and analyze the dataset, using statistical techniques, and Sklearn for Machine Learning to predict Housing Prices. 
# 
# <b>Data:</b> Data set contains information from the Ames Assessor's Office used in computing assessed values for individual residential properties sold in Ames, IA from 2006 to 2010. See attached <i>DataDictionary_AmesHousing.txt</i>  for more info. 
# 
# <b>Goal:</b> We want to learn and predict housing prices from information such as Neighborhood, Year Built, Interior and Exterior specifications, overall condition, etc. Having good prediction of housing prices enables both sellers and buyers to make informed decisions when choosing to sell or buy a house.
# 
# <b>Approach:</b> We will start by loding and preprocessing the data using Pandas and Numpy. In-depth statistical analysis, data visualization, and feature engineering will be demonstrated that will help the users to approach similar Data Science problems and a structured way. We will then explore several Machine Learning methods to build regression models and compare prediction performance.
# 
# 

# ## Start by importing the required libraries

# In[ ]:


import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import xgboost as xgb

from scipy.stats import kurtosis, skew # to explore statistics on Sale Price

# Importing plotting libraries
from plotly.offline import init_notebook_mode, iplot, plot 
import plotly.graph_objs as go 
init_notebook_mode(connected=True)

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load and understand the data
# We begin by loading our data, which is stored in tab-separated value (TSV) format. For that, we use the CSV reader from Pandas with tab as separator, which creates a Pandas DataFrame containing the dataset.
# 
# ### Steps for data exploration
# 1. Take a quick peek at the data and data-types.
# 2. Separate numerical features to perform correlation analysis and basic statistics.
# 3. Bucketed pair-plot analysis of highly related features to study their distribution and infer findings.
# 4. Identifying missing values and imputation strategies.
# 5. Studying frequency and sales prices by neighborhoods. 
# 6. Studying sales by year using Plotly's interactive graphs.
# 7. Identifying skew in Sales Price using Moments analysis and exploring outliers using Standard Deviation. 
# 8. Normalizing dependent variable, labelizing categoricals, removing extraneous cols.
# 9. Prepping for Machine Learning.
# 
# ### Machine Learning
# 1. Train baseline linear regressor.
# 2. Demonstrate K-Fold Cross Validation. 
# 3. Determing feature importances using ensemble techniques (eg. XGBoost).
# 4. Feature selection using XGBoost.
# 5. Reiterate training and compare performance metric (RMSE).
# 

# In[ ]:


datafile = "../input/ameshousing/Ames_Housing_Data.tsv"
df=pd.read_csv(datafile, sep='\t')


# In[ ]:


df.head()


# In[ ]:


# Let's have a look at the features
df.info()


# #### So we have 2930 sales records with 82 features (including target - SalePrice).

# #### Let us now define a helper function <i>expandHead()</i> to take a quick peek at the dataset.

# In[ ]:


# Quick peek at the data
def expandHead(x, nrow = 6, ncol = 4):
    # https://stackoverflow.com/a/53873661/1578274
    pd.set_option('display.expand_frame_repr', False)
    seq = np.arange(0, len(x.columns), ncol)
    for i in seq:
        print(x.loc[range(0,nrow), x.columns[range(i,min(i+ncol, len(x.columns)))]])
    pd.set_option('display.expand_frame_repr', True)
    
expandHead(df, 3, 8)


# Per the Data Dictionary, 'The data has 82 columns which include 23 nominal, 23 ordinal, 14 discrete, and 20 continuous variables (and 2 additional observation identifiers)'.
# 
# #### Let's separate numerical features (includes bool).

# In[ ]:


# Exclude nominal and ordinal as well (per data dict)
exc_cols = ['Bedroom AbvGr', 'HalfBath', 'Kitchen AbvGr','Bsmt Full Bath', 'Bsmt Half Bath', 'MS SubClass']
numerical_cols = [c for c in df.columns if df[c].dtype != 'object' and c not in exc_cols]

# expandHead(df.loc[:4, numerical_cols], 4, 8)


# In[ ]:


#Lets start by plotting a heatmap to determine if any variables are correlated
# Correlation Heatmap
# ref: https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57
f, ax = plt.subplots(figsize=(25, 15))
corr = df[numerical_cols].corr()
hm = sns.heatmap(round(corr,2), annot=False, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05) # Set annot=True for pearson labels.
f.subplots_adjust(top=0.93)
t= f.suptitle('Housing Attributes Correlation Heatmap', fontsize=18)


# Quite a few variables are positively correlated with SalePrice, eg. OverallQual, YearBuilt, etc.

# <b>Let's bucket SalePrice</b> into three qualitative categories, based on IQR, and generate a new feature for further analysis. Buckets High, Low, Medium help us identify the distribution of different categories per price ranges.

# In[ ]:


df['N_priceLbl'] = df.SalePrice.apply(lambda p: 
                                    'low' if p < 129500 else
                                   'medium' if p < 213500 else
                                   'high')


# #### Let's pick some correlated features and further visualize the patterns using pair-plot with density.

# In[ ]:


corr_features = ['Overall Qual', 'Year Built', 'Total Bsmt SF', 'Enclosed Porch', 'SalePrice']

# Density and scatter pair plots for highly correlated features
sns.pairplot(df, vars=corr_features, hue='N_priceLbl', diag_kind = 'kde',
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
             height = 4)


# #### Some interesting results can be observed from the density and corr plots: 
# - Most recently built houses fall in higher price bracket.
# - Most houses with enclosed porch fall in lower price.
# - Higher price bracket has larger variance than medium or lower.
# 
# The number of enclosed porches are negatively correlated with year built. It seems that potential housebuyers do not want an enclosed porch and house developers have been building less enclosed porches in recent years. It is also negatively correlated with SalePrice, which makes sense.
# 
# There is some slight negative correlation between OverallCond and SalePrice. There is also strong negative correlation between Yearbuilt and OverallCond. It seems to be that recently built houses tend to been in worse Overall Condition.

# Ignoring the SalePrice variable, it seems *Garage Yr Blt* and *Year Built* are strongly correlated. We'll keep one with lesser missing values.
# *Garage Cars* - *Garage Area* and *TotRms AbvGrd* - *Gr liv Area* are related as well, but let's keep them for now as they represent different values.
# 
# ### Let's plot missing values

# In[ ]:


missing_df = df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')

ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,15))
rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")

# Add value labels
for i, v in enumerate(missing_df.missing_count.values):
    ax.text(v + 10, i, str(v), color='b')
plt.show()


# ### Missing value analysis
# 
# Viewing the data indicates that there are columns which have missing values. The categorical variables with the largest number of missing values are: Alley, FirePlaceQu, PoolQC, Fence, and MiscFeature.
# 
# - Alley: indicates the type of alley access
# - FirePlaceQu: FirePlace Quality
# - PoolQC: Pool Quality
# - Fence: Fence Quality
# - MiscFeature: Miscellaneous features not covered in other categories
# 
# The missing values indicate that majority of the houses do not have alley access, no pool, no fence and no elevator, 2nd garage, shed or tennis court that is covered by the MiscFeature.
# 
# The numeric variables do not have as many missing values but there are still some present. There are 490 values for the LotFrontage, 23 missing values for MasVnrArea and 159 missing values for GarageYrBlt.
# 
# - LotFrontage: Linear feet of street connected to property
# - GarageYrBlt: Year garage was built
# - MasVnrArea: Masonry veener area in square feet

# In[ ]:


# GarageYrBlt has missing values compared to YearBuilt. So let's drop it
df = df.drop('Garage Yr Blt', axis=1)


# There seems to be some inconsistency between GarageArea and other Garage* features..
# that have missing values where area is present.
# Let us explore this.
# 

# In[ ]:


mask = df['Garage Area'] == 0
df.loc[mask, 'Garage Area'].count()


# This confirms our hunch that there is actually no inconsistency, as those missing garages have 0 area.
# 
# ### Imputing missing values
# Let us impute the 1-2 missing values features with corresponding default values.

# In[ ]:


df.loc[df['Bsmt Half Bath'].isnull(), 'Bsmt Half Bath'] = 0.0
df.loc[df['Bsmt Full Bath'].isnull(), 'Bsmt Full Bath'] = 0.0
df.loc[df['Garage Cars'].isnull(), 'Garage Cars'] = 0.0
df.loc[df['BsmtFin SF 1'].isnull(), 'BsmtFin SF 1'] = 0.0
df.loc[df['BsmtFin SF 2'].isnull(), 'BsmtFin SF 2'] = 0.0
df.loc[df['Bsmt Unf SF'].isnull(), 'Bsmt Unf SF'] = 0.0
df.loc[df['Total Bsmt SF'].isnull(), 'Total Bsmt SF'] = 0.0
df.loc[df['Garage Area'].isnull(), 'Garage Area'] = 0.0
df.loc[df['Electrical'].isnull(), 'Electrical'] = 'SBrkr'


# Let's take a quick look at the missing values again

# In[ ]:


missing_df = df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count', ascending=False)
missing_df


# Let's check the proportion of missing values. We will drop more than 95% missing features after checking their importance later.

# In[ ]:


missing_prop = (df.isnull().sum()/df.shape[0]).reset_index()
missing_prop.columns = ['field', 'proportion']
missing_prop = missing_prop.sort_values(by='proportion', ascending=False)
missing_prop.head()


# Sometimes it helps to check for 'constant' variables - we don't need them. 

# In[ ]:


df.loc[df.nunique().values == 1]
# There are no constant variables


# ### Which neighborhoods have most houses?

# In[ ]:


plt.figure(figsize=(14,6))

sns.countplot(x='Neighborhood', data=df, order = df['Neighborhood'].value_counts()[:10].index)
plt.title("Top 10 Most Frequent Neighborhoods", fontsize=20) # Adding Title and seting the size
plt.xlabel("Neighborhood", fontsize=16) # Adding x label and seting the size
plt.ylabel("Sale Counts", fontsize=16) # Adding y label and seting the size
plt.xticks(rotation=45) # Adjust the xticks, rotating the labels

plt.show()


# ### Let's cross SalePrice with Neighborhood to analyze price variance by region

# In[ ]:


plt.figure(figsize=(16,6))
sns.set_style("whitegrid")
g1 = sns.boxenplot(x='Neighborhood', y='SalePrice', 
                   data=df[df['SalePrice'] > 0])
g1.set_title('Neighborhoods by SalePrice', fontsize=20)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_xlabel('Neighborhood', fontsize=18) # Xlabel
g1.set_ylabel('SalePrice', fontsize=18) #Ylabel

plt.show()


# <b>Interesting insight!</b> StoneBr, NridgeHt, and NoRidge seem to have most of the expensive houses. <br />
# BrkSide, OldTown, IDOTRR, MeadowV have the cheapest houses.
# 
# This <b>BoxPlot</b> also helps highlight the spread of the prices, eg. North Ridge seems to have some serious outliers - may be a Mansion?

# ### Interactive graph
# Let's see number of Sales by month and year.

# In[ ]:


# Setting the first trace
trace1 = go.Histogram(x=df["Yr Sold"],
                      name='Year Count')

# Setting the second trace
trace2 = go.Histogram(x=df["Mo Sold"],
                name='Month Count')

data = [trace1, trace2]

# Creating menu options
updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=list([  
             dict(
                 label = 'Years Count',
                 method = 'update',
                 args = [{'visible': [True, False]}, # This trace visible flag
                         {'title': 'Count of Year'}]),
             dict(
                 label = 'Months Count',
                 method = 'update',
                 args = [{'visible': [False, True]},
                         {'title': 'Count of Months'}])
         ]))
])

layout = dict(title='Number of Sales by Year/Month (Select from Dropdown)',
              showlegend=False,
              updatemenus=updatemenus,
#              xaxis = dict(
#                  type="category"
#                      ),
              barmode="group"
             )
fig = dict(data=data, layout=layout)
print("SELECT OPTION BELOW: ")
iplot(fig)


# Interesting. For some reason, the sales tend to be centered around mid year (June); Perhaps the best time to sell your house,
# and Dec/Jan to buy!

# ### How expensive are the houses?

# In[ ]:


df.SalePrice.describe()


# #### Sale prices seems to be centered around the 150k mark.

# In[ ]:


df.SalePrice.plot.hist()


# Sale prices seems to be centered around the 150k mark.
# 
# ### Let us perform the **moment** analysis to understand the spread of data
# Upto abs 0.5 skewness is fairly symmetrical
# 

# In[ ]:


print('Excess kurtosis of normal distribution (should be 0): {}'.format(
    kurtosis(df[df['SalePrice'] > 0]['SalePrice'])))
print( 'Skewness of normal distribution (should be < abs 0.5): {}'.format(
    skew((df[df['SalePrice'] > 0]['SalePrice']))))


# A high kurtosis is a strong indicator of outliers, so is the positive skew.
# 
# ### Let us explore and remove the outliers per the target variable.

# In[ ]:


def explore_outliers(df_num, num_sd = 3, verbose = False): 
    '''
    Set a numerical value and it will calculate the upper, lower and total number of outliers.
    It will print a lot of statistics of the numerical feature that you set on input.
    Adapted from: https://www.kaggle.com/kabure/exploring-the-consumer-patterns-ml-pipeline
    '''
    
    data_mean, data_std = np.mean(df_num), np.std(df_num)

    # Outlier SD
    cut = data_std * num_sd

    # IQR thresholds
    lower, upper = data_mean - cut, data_mean + cut

    # creating an array of lower, higher and total outlier values 
    outliers_lower = [x for x in df_num if x < lower]
    outliers_higher = [x for x in df_num if x > upper]
    outliers_total = [x for x in df_num if x < lower or x > upper]

    # array without outlier values
    outliers_removed = [x for x in df_num if x > lower and x < upper]
    
    print('Identified lower outliers: %d' % len(outliers_lower)) # printing total number of values in lower cut of outliers
    print('Identified upper outliers: %d' % len(outliers_higher)) # printing total number of values in higher cut of outliers
    print('Total outliers: %d' % len(outliers_total)) # printing total number of values outliers of both sides
    print('Non-outlier observations: %d' % len(outliers_removed)) # printing total number of non outlier values
    print("Total percentage of Outliers: ", round((len(outliers_total) / len(outliers_removed) )*100, 4)) # Percentual of outliers in points
    
    if verbose:
        print('\nVerbose: Printing outliers')
        if len(outliers_lower) > 0:
            print(f'Lower outliers: {outliers_lower}')
            
        if len(outliers_higher) > 0:
            print(f'Upper outliers: {outliers_higher}')
    
    return

explore_outliers(df.SalePrice, 5, True)


# We should probably remove these outliers, more than 5 SD away. <br />
# Will leave it for later.

# # Machine Learning
# 
# ## Make predictions, and evaluate results
# Our final step will be to use our fitted model to make predictions on new data. We will use our held-out test set, but you could also use this model to make predictions on completely new data. For example, if we created some features data based on a different State, we could predict House Prices expected in that region!
# 
# We will also evaluate our predictions. Computing evaluation metrics is important for understanding the quality of predictions, as well as for comparing models and tuning parameters.

# In[ ]:


# But first, let's remove the feature created using target variable and split train/test
df.drop('N_priceLbl', axis=1, inplace=True)

# Also, perform logarithmic transformation on target. Why? because we're interested in 
#..relative differences in prices and this normalizes the skew. 
# Read more here: https://stats.stackexchange.com/a/48465/236332
# and here: https://towardsdatascience.com/why-take-the-log-of-a-continuous-target-variable-1ca0069ee935
Y = np.log(df.loc[:, 'SalePrice'] + 1)#.apply(lambda y: )
# Y.fillna(-1)
df.drop('SalePrice', axis=1, inplace=True)

# Also drop 'order' and 'PID' as they're just record identifiers
df.drop(['Order', 'PID'], axis=1, inplace=True)


# Hopefully, that normalized the skew. Let's do a quick sanity check/viz.

# In[ ]:


Y.plot.hist()


# ### Label encode categorical variables
# Most ML algorithms don't fair well with categorical/string data. We must labelize or binarize them before feeding to the models.

# In[ ]:


# Let's Label encode all categorical variables
for c in df.columns:
    df[c]=df[c].fillna(-1) # Imp. for both encoder and regressor. They don't like NaNs.
    if df[c].dtype == 'object':
        le = preprocessing.LabelEncoder()
        df[c] = le.fit_transform(df[c].astype('str')) # https://stackoverflow.com/a/46406995/1578274


# ### Split dataset for training, and testing
# Testing set is also canonically known as the 'held-out' set which is used only to evaluate the trained model at the end. This <b>must never</b> be used during training.

# In[ ]:


# Split into train/test (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size=0.20)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# #### Let's compare some Regression performances.
# Train a baseline regressor

# In[ ]:


# Linear Regression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
reg = linear_model.LinearRegression()
reg.fit(X_train, Y_train)
lr_pred = reg.predict(X_test)


# In[ ]:


mse(Y_test, lr_pred)


# In[ ]:


# Ridge regression using CV
reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3)
reg.fit(X_train, Y_train)
lr_pred = reg.predict(X_test)
mse(Y_test, lr_pred)


# Pretty much the same RMSE.
# 
# ### Let us find feature importances using XGBoost and auto-select best features using SelectPercentile

# In[ ]:


# ref: https://www.kaggle.com/nikunjm88/creating-additional-features/data
xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1,
    'seed' : 0
}

dtrain = xgb.DMatrix(X_train, Y_train, feature_names=X_train.columns.values)
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=150)

# plot the important features
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=30, height=0.8, ax=ax)
plt.show()


# Interestingly, LotFrontage makes it to the top of the list, which was not apparent from the corr matrix before. <br />
# On the contrary, OverallQual lands at position 12.

# #### Let's see if SelectPercentile tells a similar story

# In[ ]:


from sklearn.feature_selection import SelectPercentile, f_classif

X_indices = np.arange(X_train.shape[-1])
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X_train, Y_train)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()


# In[ ]:


top_k = 30
top_cols = df.columns[np.argsort(scores)][-top_k:]
top_scores = np.sort(scores)[-top_k:]

ind = np.arange(top_cols.shape[0])
width = 0.2
fig, ax = plt.subplots(figsize=(12,15))
rects = ax.barh(ind, top_scores, color='darkorange',
        edgecolor='black')
ax.set_yticks(ind)
ax.set_yticklabels(top_cols, rotation='horizontal')
ax.set_xlabel(r'Univariate score ($-Log(p_{value})$)')
ax.set_title("Feature importance using SelectPercentile")

# Add value labels
for i, v in enumerate(top_scores):
    ax.text(v+0.01, i, str(round(v, 4)), color='k')
plt.show()


# Slightly different results than XGBoost. OverallQual made it to the top this time. <br />
# Surprisingly, Street and External Quality are in the top three features!
# 
# But overall, the top 30 features look pretty much the same. <br />
# Let's use these features to test our models again and observe the difference, if any.

# In[ ]:


# Split into train/test
X_train, X_test, Y_train, Y_test = train_test_split(df.loc[:,top_cols], Y, test_size=0.20)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# #### Let's see if using nearby houses (data points) bring any positive improvement in the regression.

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
reg = KNeighborsRegressor(n_neighbors=5)
reg.fit(X_train, Y_train)
lr_pred = reg.predict(X_test)
mse(Y_test, lr_pred)


# Linear Regression comparison

# In[ ]:


reg = linear_model.LinearRegression()
reg.fit(X_train, Y_train)
lr_pred = reg.predict(X_test)
mse(Y_test, lr_pred)


# Seems worse than the baseline. Let's check RidgeCV again

# In[ ]:


reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3)
reg.fit(X_train, Y_train)
lr_pred = reg.predict(X_test)
mse(Y_test, lr_pred)


# Seems RidgeCV with all features is performing better with MSE of 0.0172 so far however, using just 30 features seem to capture most of the variance in data.

# # Improving our model
# We are not done yet! This section describes how to take this notebook and improve the results even more. Try forking this kernel and extending it. See how much you can improve the predictions (or lower RMSE).
# 
# There are several ways we could further improve our model:
# 
# - <b>Expert knowledge:</b> We may not be experts on the Housing industry, but we know a few things we can use: The property hasn't been remodelled if the Year Remodelled is same is Year Built. Regressors does not know that, but we could create a new boolean feature indicating whether or not the house was remodelled.
# - <b>Better tuning:</b> To make this notebook run quickly, we only tried a few hyperparameter settings. To get the most out of our data, we should test more settings. Start by increasing the number of trees in our XGBoost model by setting max_depth=100, or RidgeCV k-fold to 5 or more; it will take longer to train but can be more accurate.
# - <b>Feature engineering:</b> We used the basic set of features given to us, but we could potentially improve them, as indicated in the first point here. For example, we may guess that an area is more or less important depending on whether or not it is closer to the city/downtown using spatial info.
# - <b>Exploring more complex models:</b> While respecting Occam's Razor, we should explore more sophisticated models, eg. Decision Tree Regressors, or even Neural Networks and compare their performances with baselines.
# 
# Good luck!

# ### ---

# ## Thank you for taking the time
# This work is still in progress and refinements will continue. In the meanwhile..
# - Didn't understand something? Ask a question.
# - Something could've been better? Share your feedback.
# - Found the notebook useful? Please share your vote!
# 
# #### Used the Ames Housing Data:
# - Dean De Cock Truman State University Journal of Statistics Education Volume 19, Number 3(2011), www.amstat.org/publications/jse/v19n3/decock.pdf
