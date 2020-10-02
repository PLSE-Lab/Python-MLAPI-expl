#!/usr/bin/env python
# coding: utf-8

# ## LotFrontage Imputation for House Price Competition
# This notebook's purpose is to explore imputation of the missing lot frontage values for the House Price Kaggle competition. _LotFrontage_ is a variable that seems to have high predictive power for the competition, so it is worthwhile to try and fill in the blanks. A few alternatives I came accross are:
# 1. Use [mean LotFrontage of the neighbourhood](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard) 
# 2. Use [mean LotFrontage of entire set](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/discussion/50085)
# 3. Simply [fill all missing values with zero](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/discussion/50085)
# 
# I will benchmark _LotFrontage_ prediction error against these approaches to make sure there  is a noticeable improvement. 

# In[ ]:


# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Special settings for Python notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Ignore FutureWarnings related to internal pandas and np code
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# For the purpose of imputing _LotFrontage_, I'll use both train and test datasets. Once again, the purpose is not to predict target variable _SalePrice_, but rather learn to impute lot frontage. Information leakage is a good topic  for discussion, but let's leave it for another day.

# In[ ]:


# Read in the train and test datasets
raw_train = pd.read_csv('../input/train.csv')
raw_test = pd.read_csv('../input/test.csv')

# Drop SalePrice column from train dataset and merge into one data frame called all_data
raw_train = raw_train.drop('SalePrice', axis=1)
all_data = pd.concat([raw_train, raw_test], ignore_index=True).copy()


# Next, let's sepeare all_data  into train and test dataset for the purpose of imputing _LotFrontage_. There are 486 missing values in the full dataset, or roughly 15% of all records.

# In[ ]:


# Split into known and unknown LotFrontage records
test  = all_data[all_data.LotFrontage.isnull()]
train = all_data[~all_data.LotFrontage.isnull()]
target = train.LotFrontage
print("LotFrontage has {:} missing value, and {:} values avaialble.".format(test.shape[0], train.shape[0]))


# ### Exploration of Relevant Variables
# 
# I'll start with a univariate, density plot and boxplot for _LotFrontage_. Some interesting observations here are: (a) There is a clustering of low _LotFrontage_ values - the peak you see on the left; and (b) There is a fairly long tail of values that are beyond Median + 1.5IQR. In other words there are outliers on both low and high end of _LotFrontage_ value range.

# In[ ]:


fig, ax =plt.subplots(1,2, figsize=(16,4))
sns.distplot(target, ax=ax[0])
sns.boxplot(target, ax=ax[1]);


# Onto multivariate analysis. There are a few key features that I suspect will influence LotFrontage:
# * _LotArea_ - Clearly should have a significant correlation with LotFrontage. This relationship would not be linear, but some 2nd degree polynomial. If all lots were exactly square, then ${LotFrontage} = \sqrt{LotArea}$ and we'd be done :)
# * _LotConfig_ - Corner unit will likely have a larger LotFrontage than Inside lots. Similarly, CulDSac will have a circular shape of its LotFrontage.
# * _LotShape_ - Regular and irregular lots likely have different relationships between LotFrontage and LotArea.
# * _Alley_ - This may indicate, along with other variables, the geometry of the lot (e.g. garage facing the alley, will not influence Lot Frontage)
# * _MSZoning_ - High and low density residentials zones may have different LotFrontage values.
# * _BldgType_ - Townhouse has a much more narrow footprint than a detached house. This should affect LotFrontage quite a bit.
# * _Neighborhood_ - Different parts of the city may have different standard for LotArea and LotFrontage.
# * _Condition1_ & _Condition2_ - Can serve as markers for lot location within the neighbourhood (e.g. lots within same block are similar in size and LotFrontage)
# * _GarageType_ - Garage must have direct access to street, which should influence LotFrontage. Unless of course the garage is facing the alley.
# * _GarageCars_ - Number of cars that fit in the garage will also affect the Lot Frontage, similar to GarageType.
# 
# Before we dive into exploring the relationships between these and the _LotFrontage_ target variable, I'll define a quick and dirty outlier detection function to remove noice from the visualizations.

# In[ ]:


def idOutliers (dat):
    tile25 = dat.describe()[4]
    tile75 = dat.describe()[6]
    iqr = tile75 - tile25 
    out = (dat > tile75+1.5*iqr) | (dat < tile25-1.5*iqr)
    return out


# Let's start by looking at _LotArea_ and its relation to _LotFrontage_. I expect this to be a 2nd degree polynomial, so for plotting I'll take a square root of LotArea. First plot is showing all data, second graph removes outliers.

# In[ ]:


# LotArea vs. LotFrontage
fig, ax =plt.subplots(1,2, figsize=(16,4))
ax[0].set_title('With LotArea outliers')
ax[1].set_title('Without LotArea outliers')
sns.regplot(train.LotArea.apply(np.sqrt), target, ax=ax[0])
ax[0].set(xlabel='sqrt(LotArea)')
sns.regplot(train.LotArea[~idOutliers(train.LotArea)].apply(np.sqrt), target[~idOutliers(train.LotArea)], ax=ax[1])
ax[1].set(xlabel='sqrt(LotArea)');


# Note the tight grouping of very low LotFrontage value in the bottom left. These correspond to what I noticed before in the distribution plot, but now I know that these occur in conjunction with low _LotArea_ as well. Also notice the heteroscedacity of this relationship, meaning that the variance of _LotFrontage_ is not uniform accross the range of _LotArea_. In fact, the larger the overall area of the lot, the more variation there is in _LotFrontage_.

# It may be useful to review the other variables in context of the relationship between _LotFrontage_ and _sqrt(LotArea)_ scatterplot. For example, with _BldgType_ variable I can see that townhouses are all clustered at the low end of both lot area and frontage. Even the end units of townhouses are mostly bound to between 30 and 60 feet of lot frontage. Duplex and single family homes have a much wider range of _LotFrontage_ values, but there seems to be a floor to their lot frontage as well - around 40 feet.

# In[ ]:


train_plot = train[~idOutliers(train.LotArea)].copy()
train_plot['sqrt_LotArea'] = train_plot['LotArea'].apply(np.sqrt)
sns.lmplot(x='sqrt_LotArea', y='LotFrontage', hue='BldgType', aspect=2, fit_reg=False, data=train_plot);


# Looking at the neighbourhood spreads of lot frontage, there are some areas in the city of Ames that have a fairly distinct spread of _LotFrontage_ values. While some neighbourhoods have a wider variety when it comes to lot frontage, the medians largely occupy their own space on the y-axis.

# In[ ]:


# Neighborhood vs. LotFrontage
plt.figure(figsize=(20,5))
sns.boxplot(x=train_plot['Neighborhood'], y=train_plot['LotFrontage'], width=0.7, linewidth=0.8);


# One more relevant observation that I can make is that properties with no access to Alley tend to have a higher median LotFrontage. Gravel alley access seems to be tightly bound, except for outliers, to a short range LotFrontage values. Though these properties (with access to alley) do not make up a large portion of the dataset - only about 180 observations.

# In[ ]:


# Alley vs. LotFrontage
train_plot['Alley'] = train_plot['Alley'].fillna('No Alley')
sns.boxplot(x=train_plot['Alley'], y=train_plot['LotFrontage'], linewidth=0.8);


# One more variable of note is _GarageCars_. The median value of LotFrontage increases with every additional car in the garage, up to 3. Then for 4 and 5 car garages, there is a drop. I suspect that the 4 and 5 car garages are not positioned parallel to the street and therefore do not contribute as much to the lot frontage. There is also very few such observations in the dataset.

# In[ ]:


# GarageCars vs. LotFrontage
sns.boxplot(x=train_plot['GarageCars'], y=train_plot['LotFrontage'], linewidth=0.8);


# ### Model Building and Evaluation
# 
# Now let's actually try to model LotFrontage using the variables I listed above. First, add relevant libraries. I'll be using Support Vector Regressor to build the model and 10 fold validation to benchmark it against existing imputation approaches.

# In[ ]:


from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


# Next I prepare the dataset for training by separating the target variable, _LotFrontage_, from the rest, selecting relevant features, and dummifying categorical variables. 

# In[ ]:


# Pull only the features for training the model. Define target variable
y_lotFrontage = train['LotFrontage']
X_train = train.loc[:,['LotArea', 'LotConfig', 'LotShape', 'Alley', 'MSZoning', 'BldgType', 'Neighborhood', 'Condition1', 'Condition2', 'GarageCars']]

# Dummify categorical variables and normalize the data
X_train = pd.get_dummies(X_train)
X_train = (X_train - X_train.mean())/X_train.std()
X_train = X_train.fillna(0)


# I define an SVR classifier with parameters that have been tuned separately. Finally, the main loop will perform 10-fold validation and calculate the mean absolute error as a comparative value between three imputations. I chose mean absolute error because it's easy to interpret in linear feet of lot frontage.

# In[ ]:


# Classifier with tuned parameteres
clf = svm.SVR(kernel='rbf', C=100, gamma=0.001)

# Set initial scores
acc = 0
acc1 = 0
acc2 = 0

# Defien k-fold object for 10-fold validation
kf = KFold(n_splits=10, shuffle=True, random_state=3) 

# Main evaluator loop over the 10 folds
for trn, tst in kf.split(train):
    
    # Compute benchmark score prediction based on mean neighbourhood LotFrontage
    fold_train_samples = train.iloc[trn]
    fold_test_samples = train.iloc[tst]
    neigh_means = fold_train_samples.groupby('Neighborhood')['LotFrontage'].mean()
    all_mean = fold_train_samples['LotFrontage'].mean()
    y_pred_neigh_means = fold_test_samples.join(neigh_means, on = 'Neighborhood', lsuffix='benchmark')['LotFrontage']
    y_pred_all_mean = [all_mean] * fold_test_samples.shape[0]
    
    # Compute benchmark score prediction based on overall mean LotFrontage
    u1 = ((fold_test_samples['LotFrontage'] - y_pred_neigh_means) ** 2).sum()
    u2 = ((fold_test_samples['LotFrontage'] - y_pred_all_mean) ** 2).sum()
    v = ((fold_test_samples['LotFrontage'] - fold_test_samples['LotFrontage'].mean()) ** 2).sum()
    
    # Perform model fitting 
    clf.fit(X_train.iloc[trn], y_lotFrontage.iloc[trn])
    
    # Record all scores for averaging
    acc = acc + mean_absolute_error(fold_test_samples['LotFrontage'], clf.predict(X_train.iloc[tst]))
    acc1= acc1 + mean_absolute_error(fold_test_samples['LotFrontage'], y_pred_neigh_means)
    acc2 = acc2 + mean_absolute_error(fold_test_samples['LotFrontage'], y_pred_all_mean)

    
print('10-Fold Validation Mean Absolute Error results:')
print('\tSVR: {:.3}'.format(acc/10))
print('\tSingle mean: {:.3}'.format(acc2/10))
print('\tNeighbourhood mean: {:.3}'.format(acc1/10))


# These results show that imputing the _LotFrontage_ using SVR, after parameter tuning, gives an average error of less than 9 feet - a 35% improvement over next best method. Computing the measure of variation shows that the selected variables in the SVR model explain over 50% of the variation contained in _LotFrontage_, that compares to about 25% of the Neighborhood mean approach.

# ### Final Imputation
# 
# I will share the dataset that imputed _LotFrontage_ for use in the community. If you use it, please let me know how it affects your model in comments.

# In[ ]:


# Select columns for final prediction, dummify, and normalize
X_test = test.loc[:,['LotArea', 'LotConfig', 'LotShape', 'Alley', 'MSZoning', 'BldgType', 'Neighborhood', 'Condition1', 'Condition2', 'GarageCars']]
X_test = pd.get_dummies(X_test)
X_test = (X_test - X_test.mean())/X_test.std()
X_test = X_test.fillna(0)


# In[ ]:


# Make sure that dummy columns from training set are replicated in test set
for col in (set(X_train.columns) - set(X_test.columns)):
    X_test[col] = 0

X_test = X_test[X_train.columns]

# Assign predicted LotFrontage value in all_data 
all_data.loc[all_data.LotFrontage.isnull(), 'LotFrontage'] = clf.predict(X_test)

# Output to file
all_data.to_csv('housing_data_with_imputed_LotFrontage.csv')


# This was a nice exercise of value imputation and I hope you find it useful. I think it's a good idea to outsource imputation of values to others when building a winning model. That way a senior Data Scientist can focus on tuning parameters with the dataset that is as complete as possible. 
# 
# ### Appendix

# In[ ]:


# Appendix: Model Tuning
# Gridsearch for best model
#from sklearn.model_selection import GridSearchCV
#Cs = [0.1, 1, 10, 25, 100, 1000]
#gammas = [0.001, 0.01, 0.1]
#parameters = {'kernel':('linear', 'rbf'), 'C':Cs, 'gamma':gammas}
#clf_gd = GridSearchCV(svm.SVR(), parameters, cv=3, verbose=1)
#clf_gd.fit(X_train, y_lotFrontage)

