#!/usr/bin/env python
# coding: utf-8

# ## Load Libraries and Dataset
# Load libraries, dataset and take a look at what we got!

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from pprint import pprint

df = pd.read_csv('/kaggle/input/nyc-property-sales/nyc-rolling-sales.csv')
df = df.drop('Unnamed: 0', 1)
df = df.replace(' -  ', np.nan) # empty data points are not set up properly
df = df.replace(' ', np.nan)
df


# Let's see if there any *rows* which need to be filtered. We'll look at the columns later.

# In[ ]:


print(df[df.duplicated() == True].shape)
print(df[df.isnull().all(axis=1)].shape) 


# 765 duplicate rows; let's get rid of those. Zero rows with all `NaN` entries which is good.

# In[ ]:


df.drop_duplicates(inplace=True)


# ## Inspect dependent variable (`SALE PRICE`)

# In[ ]:


df['SALE PRICE'] = df['SALE PRICE'].fillna(0).astype('int')
df['SALE PRICE'].describe()


# Quite a few 0 entries there. Let's visually inspect price against gross square feet:

# In[ ]:


df['GROSS SQUARE FEET'] = df['GROSS SQUARE FEET'].fillna(0).astype('int')

plt.rcParams['figure.figsize'] = (20, 10)

fig, ax = plt.subplots()
sns.scatterplot(x='GROSS SQUARE FEET', y='SALE PRICE', data=df)
ax.set_xlim([-10**3, 10**4.5])
ax.set_ylim([.5, 10**9])
ax.set_yscale("log")
plt.show()


# We see some unrealistic prices at `10**0`, `10**1` etc. These are probably set arbitrarily and not representative for the true value of the property.

# In[ ]:


df = df[df['SALE PRICE'] != 0]
df = df[df['SALE PRICE'] != 10**0]
df = df[df['SALE PRICE'] != 10**1]
df = df[df['SALE PRICE'] != 10**2]
df = df[df['SALE PRICE'] != 10**3]
df = df[df['SALE PRICE'] != 10**4]


# How are prices distributed?

# In[ ]:


sns.distplot(df['SALE PRICE'])


# There are some crazy outliers, let's get rid of the top x% (and then, to be fair, the bottom y%). We could instead try and set a 'reasonable' minimum/maximum but let's not forget we are looking at prices for both properties such as parking lots and for whole skyscrapers. A skewed distribution is to be expected.
# 
# I define these percentages as hyperparameters to be able to tweak them once the model is in place.

# In[ ]:


HYP = {
    'PRICE_UPPER_Q': .995,
    'PRICE_LOWER_Q': 0.01,
    'N_NEIGHBORS': 50,
    'N_BUILD_CAT': 50,
    'BUILD_LETTER': False
}


# In[ ]:


df = df[df['SALE PRICE'] < df['SALE PRICE'].quantile(HYP['PRICE_UPPER_Q'])]
df = df[df['SALE PRICE'] > df['SALE PRICE'].quantile(HYP['PRICE_LOWER_Q'])]


# In[ ]:


sns.distplot(df['SALE PRICE'])


# Looking better now.

# ## Feature Extraction and Engineering

# I will just browse through all columns and keep, drop or adjust them accordingly.

# In[ ]:


df.columns


# In[ ]:


fig, ax = plt.subplots()
sns.scatterplot(x='GROSS SQUARE FEET', y='SALE PRICE', data=df, hue='BOROUGH')
ax.set_xlim([-10**3, 10**4.5])
ax.set_ylim([10**4, 10**7.5])
ax.set_yscale("log")
plt.show()


# `BOROUGH` is a no-brainer. It has five clean categories and shows nice clustering in the scatterplot. Keeper!

# In[ ]:


df['BOROUGH'] = df['BOROUGH'].astype('category')


# `NEIGHBORHOOD` is trickier. This looks like another clean categorical but this time with 253 unique values. Let's only consider the top `N` most used categories for now but at the same time keep this number adjustable as a hyperparameter.

# In[ ]:


# had to call in the help from a ninja to get this two liner to work:
# https://stackoverflow.com/questions/58494476/pandas-category-keep-only-most-common-ones-and-replace-rest-with-nan/
top = df['NEIGHBORHOOD'].value_counts().head(HYP['N_NEIGHBORS']).index.tolist()
df.loc[~df['NEIGHBORHOOD'].isin(top), 'NEIGHBORHOOD'] = 'OTHER'

df['NEIGHBORHOOD'] = df['NEIGHBORHOOD'].astype('category')


# In[ ]:


df['NEIGHBORHOOD'].value_counts()


# I'm thinking of a similar tactic for `BUILDING CLASS CATEGORY`. Top `N` for now.

# In[ ]:


df['BUILDING CLASS CATEGORY'].value_counts()
top = df['BUILDING CLASS CATEGORY'].value_counts().head(HYP['N_BUILD_CAT']).index.tolist()
df.loc[~df['BUILDING CLASS CATEGORY'].isin(top), 'BUILDING CLASS CATEGORY'] = '00 OTHER'
df['BUILDING CLASS CATEGORY'] = df['BUILDING CLASS CATEGORY'].str.strip().astype('category')


# In[ ]:


df['BUILDING CLASS CATEGORY']


# `BUILDING CLASS AT PRESENT` can be simplified by taking only the letter and not the number. This reduces this categrorical from 140 to 24 unique entries. Judging by these [building code descriptions](https://www1.nyc.gov/assets/finance/jump/hlpbldgcode.html) the different subcategories per letter are much less important for our purpose anyway. For tweaking purposes I did add a boolean switch to the hyperparameters.

# In[ ]:


if HYP['BUILD_LETTER']:
    df['BUILDING CLASS AT PRESENT'] = df['BUILDING CLASS AT PRESENT'].str[0].astype('category')
else:
    df['BUILDING CLASS AT PRESENT'] = df['BUILDING CLASS AT PRESENT'].astype('category')


# `TAX CLASS AT PRESENT` looks to be concise enough to keep as is.

# In[ ]:


df['TAX CLASS AT PRESENT'] = df['TAX CLASS AT PRESENT'].astype('category')


# `BLOCK` and `LOT` seem way too specific for our predictive purposes. We could see if any properties were sold more than once? Going to leave that for now. The same goes for `ADDRESS` and `APARTMENT NUMBER`; too specific. `ZIP CODE` has too many unique values: `182`. If we would simplify it (first two or three numbers for example) we'd be recreating something close to `BOROUGH` is my guess. So not worth it. `EASE-MENT` is (as good as) empty. Drop drop drop.

# In[ ]:


df = df.drop(['BLOCK', 'LOT', 'ADDRESS', 'APARTMENT NUMBER', 'ZIP CODE', 'EASE-MENT'], 1)


# The different `UNIT` features are another reason we see such a skewed distribution in `SALE PRICE`. The biggest property consists of `1844` units! All of these unit features are definitely important. `16313` entries have `0` units though. Not sure what to think of this now---filtering them out.

# In[ ]:


df['RESIDENTIAL UNITS'] = df['RESIDENTIAL UNITS'].astype('int')
df['COMMERCIAL UNITS'] = df['COMMERCIAL UNITS'].astype('int')
df['TOTAL UNITS'] = df['TOTAL UNITS'].astype('int')
df = df[df['TOTAL UNITS'] > 0]
df = df[df['RESIDENTIAL UNITS'] + df['COMMERCIAL UNITS'] == df['TOTAL UNITS']] # obviously these have to be equal


# We already saw that the amount of square feet correlates the strongest with the price so we keep it as a feature. Again of lot of `0` entries here though :-(

# In[ ]:


df['LAND SQUARE FEET'] = df['LAND SQUARE FEET'].fillna(0).astype('int')
df['GROSS SQUARE FEET'] = df['GROSS SQUARE FEET'].fillna(0).astype('int')
df = df[df['LAND SQUARE FEET'] > 0]
df = df[df['GROSS SQUARE FEET'] > 0]


# `YEAR BUILT` has some too low values. Let's get rid of all observations with a year built before first European contact (1524).

# In[ ]:


sns.boxplot(df['YEAR BUILT'])
df = df[df['YEAR BUILT'] > 1524]


# `TAX CLASS AT TIME OF SALE` has no new information compared to `TAX CLASS AT PRESENT`. The same goes for `BUILDING CLASS AT TIME OF SALE`.

# In[ ]:


df = df.drop(['TAX CLASS AT TIME OF SALE', 'BUILDING CLASS AT TIME OF SALE'], 1)


# `SALE DATE` gives lots of options for feature engineering: month of sale, season etc. For now we leave it though.

# In[ ]:


df = df.drop('SALE DATE', 1)


# In[ ]:


df.info()


# Sweet, a concise and clean feature set! Had to drop quite some observations though; 27882 left out of 84548 (33.0%).

# ## Fitting a Linear Regression Model

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.feature_selection/#sequentialfeatureselector
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


# First we have to prepare the categorical features to dummy variables in order to be able to feed them to the model:

# In[ ]:


def one_hot(df, cat):
    one_hots = pd.get_dummies(df[cat], prefix=cat)
    del df[cat]
    return pd.concat([df, one_hots], axis=1)

for cat in df.select_dtypes(include='category').columns:
    df = one_hot(df, cat)

df.columns


# Now before we do anything else we need to split our data into train, test and validation sets. The first is to fit the model. 
# 
# To make sure we are not overfitting the model on the train set, we test it on 'unseen' data from the test set afterwards. Adjusting the hyperparameters of the preprocessing steps can then again overfit the cleaning on the test set. Only when completely done with these adjustments will we do a run on the never seen validation set to get a final performance measure.

# In[ ]:


y = df.pop('SALE PRICE')
X = df

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=55)
X_test, X_valid, y_test, y_valid = train_test_split(
    X_test, y_test, test_size=0.4, random_state=55)

print(X_train.shape, X_test.shape, X_valid.shape)


# In[ ]:


lr = LinearRegression()

sfs = SFS(lr,
          k_features='parsimonious',
          verbose=1,
          scoring='r2',
          cv=5,
          n_jobs=-1)

sfs = sfs.fit(X_train, y_train)


# In[ ]:


pd.DataFrame.from_dict(sfs.get_metric_dict()).T


# `scikit-learn` doesn't offer adjusted $R^2$ out of the box so manually calculating it through $\bar{R}^2 = 1 - (1 - R^2)\frac{n - 1}{n - p - 1}$.

# In[ ]:


X_train_sfs = sfs.transform(X_train)
X_test_sfs = sfs.transform(X_test)

lr.fit(X_train_sfs, y_train)
y_pred = lr.predict(X_test_sfs)

k = len(sfs.k_feature_names_)
n = X_train.shape[0]
r2 = sfs.k_score_
adj_r2 = 1 - (1 - sfs.k_score_) * ((n - 1) / (n - k - 1))

pprint(HYP)
print(f'TRAIN R2: {r2}')
print(f'TRAIN ADJUSTED R2: {adj_r2}')
print(f'TEST R2: {r2_score(y_test, y_pred)}')
print(f'k: {k}')
pprint(list(sfs.k_feature_names_))


# ## Conclusion

# * So first result gives an $R^2$ = .51 for both train and test set with $k$ = 16. Most interesting fact is that there is no square feet nor units features got included! Going back to the cleaning hyperparameters to see if tweaking those have any effect on these results.
# * Increasing the upper and lower price filtering quantiles decreased $R^2$ and increased $k$. Maybe not filter out upper and lower quantiles at all?
# * No! Sweet spot seems to be between 0.01 and 0.995, So filtering out some lowest values and a couple of the highest.
# * Turns out simplifying all the features through the hyperparameters is actually only good for increasing training time.
# * Better results could be obtained with more features. Potential columns could be `SALE DATE` and `ZIPCODE`. Main bottleneck becomes the processing time needed to let SFS find the best subset.
# * Within reasonable time (~3 hours) this kernel was able to search through a feature space of 257 independent variables. This resulted in a model with $R^2 = 0.58$ on the train set and 0.52 on the test set. Validated at $R^2 = 0.56$:

# ## Validation

# In[ ]:


X_valid_sfs = sfs.transform(X_valid)
y_pred_v = lr.predict(X_valid_sfs)

print(f'TRAIN R2:\t {r2}')
print(f'TEST R2:\t {r2_score(y_test, y_pred)}')
print(f'VALIDATION R2:\t {r2_score(y_valid, y_pred_v)}')


# In[ ]:


fig = plot_sfs(sfs.get_metric_dict(), ylabel='R^2')
plt.ylim([0.2, 0.7])
plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.show()


# Because we are runnning SFS with `k_features='parsimonious'` the "smallest feature subset that is within one standard error of the cross-validation performance will be selected". [[docs]](http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#api)
# 
# The above graph clearly shows plateauing around $k > 60$. Best model that SFS returns has indeed $k = 40$.

# ## Bonus: TPOT

# [TPOT](https://github.com/EpistasisLab/tpot) uses genetic programming to optimise a classifier or regressor without too much manual setup. Feature selection, feature construction, model selection, parameter optimisation etc. are all handled by TPOT. We only have to give it our clean dataset!

# In[ ]:


from tpot import TPOTRegressor
from tpot.builtins import StackingEstimator
from sklearn.pipeline import make_pipeline, make_union
from sklearn.ensemble import GradientBoostingRegressor

tpot = TPOTRegressor(generations=10, population_size=50, verbosity=2, scoring='r2', 
                     warm_start=True, n_jobs=-1)


# In[ ]:


# not turning the teapot on again!
#tpot.fit(X_train, y_train)
#print(tpot.score(X_test, y_test))
#tpot.export('tpot_pipeline.py')

# yields:
tpot_model = GradientBoostingRegressor(alpha=0.95, learning_rate=0.1, loss="huber", 
                                       max_depth=8, max_features=0.7, 
                                       min_samples_leaf=2, min_samples_split=19, 
                                       n_estimators=100, subsample=0.6)


# Took this Kaggle kernel 19006.03 seconds on a dedicated CPU thread to come up with the model above.
# 
# Cup of tea anyone?!

# In[ ]:


print(f'TRAIN R2:\t 0.7094491358671142')
tpot_model.fit(X_train, y_train)
tpot_y_pred = tpot_model.predict(X_test)
print(f'TEST R2:\t {r2_score(y_test, tpot_y_pred)}')
tpot_y_pred_v = tpot_model.predict(X_valid)
print(f'VALIDATION R2:\t {r2_score(y_valid, tpot_y_pred_v)}')


# In[ ]:


# results log

#first attempt
{'BUILD_LETTER': True,
 'N_BUILD_CAT': 10,
 'N_NEIGHBORS': 10,
 'PRICE_LOWER_Q': 0.01,
 'PRICE_UPPER_Q': 0.99}
TRAIN R2: 0.509000260340717
TEST R2: 0.5066320072800943
k: 16
['BOROUGH_1',
 'BOROUGH_3',
 'BOROUGH_4',
 'NEIGHBORHOOD_OTHER',
 'NEIGHBORHOOD_UPPER EAST SIDE (59-79)',
 'NEIGHBORHOOD_UPPER EAST SIDE (79-96)',
 'NEIGHBORHOOD_UPPER WEST SIDE (79-96)',
 'BUILDING CLASS CATEGORY_00 OTHER',
 'BUILDING CLASS CATEGORY_03 THREE FAMILY DWELLINGS',
 'BUILDING CLASS CATEGORY_10 COOPS - ELEVATOR APARTMENTS',
 'TAX CLASS AT PRESENT_1',
 'TAX CLASS AT PRESENT_2',
 'TAX CLASS AT PRESENT_2A',
 'BUILDING CLASS AT PRESENT_D',
 'BUILDING CLASS AT PRESENT_F',
 'BUILDING CLASS AT PRESENT_S']

# second
{'BUILD_LETTER': True,
 'N_BUILD_CAT': 10,
 'N_NEIGHBORS': 10,
 'PRICE_LOWER_Q': 0.05,
 'PRICE_UPPER_Q': 0.95}
TRAIN R2: 0.3558789829142702
TEST R2: 0.35023793944264014
k: 22
['BOROUGH_1',
 'BOROUGH_3',
 'BOROUGH_4',
 'BOROUGH_5',
 'NEIGHBORHOOD_BEDFORD STUYVESANT',
 'NEIGHBORHOOD_EAST NEW YORK',
 'NEIGHBORHOOD_JACKSON HEIGHTS',
 'NEIGHBORHOOD_MIDTOWN EAST',
 'NEIGHBORHOOD_OTHER',
 'BUILDING CLASS CATEGORY_01 ONE FAMILY DWELLINGS',
 'BUILDING CLASS CATEGORY_02 TWO FAMILY DWELLINGS',
 'BUILDING CLASS CATEGORY_09 COOPS - WALKUP APARTMENTS',
 'BUILDING CLASS CATEGORY_10 COOPS - ELEVATOR APARTMENTS',
 'TAX CLASS AT PRESENT_1',
 'TAX CLASS AT PRESENT_2',
 'TAX CLASS AT PRESENT_2A',
 'BUILDING CLASS AT PRESENT_A',
 'BUILDING CLASS AT PRESENT_B',
 'BUILDING CLASS AT PRESENT_E',
 'BUILDING CLASS AT PRESENT_I',
 'BUILDING CLASS AT PRESENT_S',
 'BUILDING CLASS AT PRESENT_W']

# third (current)
{'BUILD_LETTER': False,
 'N_BUILD_CAT': 50,
 'N_NEIGHBORS': 50,
 'PRICE_LOWER_Q': 0.01,
 'PRICE_UPPER_Q': 0.995}
TRAIN R2: 0.5814129194095372
TRAIN ADJUSTED R2: 0.5802083522711474
TEST R2: 0.5248503759027321
k: 40
['BOROUGH_1',
 'BOROUGH_3',
 'BOROUGH_4',
 'NEIGHBORHOOD_ASTORIA',
 'NEIGHBORHOOD_CANARSIE',
 'NEIGHBORHOOD_EAST NEW YORK',
 'NEIGHBORHOOD_FLATBUSH-EAST',
 'NEIGHBORHOOD_FLUSHING-NORTH',
 'NEIGHBORHOOD_GRAMERCY',
 'NEIGHBORHOOD_GREENWICH VILLAGE-CENTRAL',
 'NEIGHBORHOOD_GREENWICH VILLAGE-WEST',
 'NEIGHBORHOOD_HARLEM-CENTRAL',
 'NEIGHBORHOOD_MIDTOWN WEST',
 'NEIGHBORHOOD_MURRAY HILL',
 'NEIGHBORHOOD_PARK SLOPE',
 'NEIGHBORHOOD_UPPER EAST SIDE (59-79)',
 'NEIGHBORHOOD_UPPER EAST SIDE (79-96)',
 'NEIGHBORHOOD_UPPER WEST SIDE (79-96)',
 'BUILDING CLASS CATEGORY_01 ONE FAMILY DWELLINGS',
 'BUILDING CLASS CATEGORY_08 RENTALS - ELEVATOR APARTMENTS',
 'BUILDING CLASS CATEGORY_09 COOPS - WALKUP APARTMENTS',
 'BUILDING CLASS CATEGORY_10 COOPS - ELEVATOR APARTMENTS',
 'BUILDING CLASS CATEGORY_26 OTHER HOTELS',
 'BUILDING CLASS CATEGORY_29 COMMERCIAL GARAGES',
 'BUILDING CLASS CATEGORY_37 RELIGIOUS FACILITIES',
 'TAX CLASS AT PRESENT_2',
 'TAX CLASS AT PRESENT_2A',
 'TAX CLASS AT PRESENT_2B',
 'TAX CLASS AT PRESENT_4',
 'BUILDING CLASS AT PRESENT_A3',
 'BUILDING CLASS AT PRESENT_A4',
 'BUILDING CLASS AT PRESENT_A7',
 'BUILDING CLASS AT PRESENT_C5',
 'BUILDING CLASS AT PRESENT_E2',
 'BUILDING CLASS AT PRESENT_K4',
 'BUILDING CLASS AT PRESENT_K9',
 'BUILDING CLASS AT PRESENT_L1',
 'BUILDING CLASS AT PRESENT_O4',
 'BUILDING CLASS AT PRESENT_O8',
 'BUILDING CLASS AT PRESENT_W9']

# maxing out (processing time becomes bottleneck)
{'BUILD_LETTER': False,
 'N_BUILD_CAT': 999,
 'N_NEIGHBORS': 999,
 'PRICE_LOWER_Q': 0.01,
 'PRICE_UPPER_Q': 0.995}
TRAIN R2: 0.610984514559609
TRAIN ADJUSTED R2: 0.6082325101148565
TEST R2: 0.5995258969805612
k: 98
['YEAR BUILT',
 'BOROUGH_1',
 'BOROUGH_3',
 'BOROUGH_4',
 'NEIGHBORHOOD_ARVERNE',
 'NEIGHBORHOOD_ASTORIA',
 'NEIGHBORHOOD_BAYSIDE',
 'NEIGHBORHOOD_BERGEN BEACH',
 'NEIGHBORHOOD_BOERUM HILL',
 'NEIGHBORHOOD_BROOKLYN HEIGHTS',
 'NEIGHBORHOOD_BROWNSVILLE',
 'NEIGHBORHOOD_CAMBRIA HEIGHTS',
 'NEIGHBORHOOD_CANARSIE',
 'NEIGHBORHOOD_CARROLL GARDENS',
 'NEIGHBORHOOD_CLINTON HILL',
 'NEIGHBORHOOD_COBBLE HILL',
 'NEIGHBORHOOD_COBBLE HILL-WEST',
 'NEIGHBORHOOD_CONEY ISLAND',
 'NEIGHBORHOOD_CYPRESS HILLS',
 'NEIGHBORHOOD_DOWNTOWN-FULTON FERRY',
 'NEIGHBORHOOD_DOWNTOWN-FULTON MALL',
 'NEIGHBORHOOD_EAST NEW YORK',
 'NEIGHBORHOOD_EAST RIVER',
 'NEIGHBORHOOD_ELMHURST',
 'NEIGHBORHOOD_FAR ROCKAWAY',
 'NEIGHBORHOOD_FLATBUSH-EAST',
 'NEIGHBORHOOD_FLATBUSH-NORTH',
 'NEIGHBORHOOD_FLATLANDS',
 'NEIGHBORHOOD_FLUSHING-NORTH',
 'NEIGHBORHOOD_FOREST HILLS',
 'NEIGHBORHOOD_FORT GREENE',
 'NEIGHBORHOOD_GERRITSEN BEACH',
 'NEIGHBORHOOD_GOWANUS',
 'NEIGHBORHOOD_GRAMERCY',
 'NEIGHBORHOOD_GRAVESEND',
 'NEIGHBORHOOD_GREENPOINT',
 'NEIGHBORHOOD_GREENWICH VILLAGE-CENTRAL',
 'NEIGHBORHOOD_GREENWICH VILLAGE-WEST',
 'NEIGHBORHOOD_HARLEM-CENTRAL',
 'NEIGHBORHOOD_HARLEM-EAST',
 'NEIGHBORHOOD_HARLEM-UPPER',
 'NEIGHBORHOOD_HOLLIS',
 'NEIGHBORHOOD_LAURELTON',
 'NEIGHBORHOOD_LONG ISLAND CITY',
 'NEIGHBORHOOD_MARINE PARK',
 'NEIGHBORHOOD_MIDTOWN WEST',
 'NEIGHBORHOOD_MILL BASIN',
 'NEIGHBORHOOD_OCEAN HILL',
 'NEIGHBORHOOD_OLD MILL BASIN',
 'NEIGHBORHOOD_OZONE PARK',
 'NEIGHBORHOOD_PARK SLOPE',
 'NEIGHBORHOOD_PARK SLOPE SOUTH',
 'NEIGHBORHOOD_QUEENS VILLAGE',
 'NEIGHBORHOOD_RICHMOND HILL',
 'NEIGHBORHOOD_ROSEDALE',
 'NEIGHBORHOOD_SEAGATE',
 'NEIGHBORHOOD_SHEEPSHEAD BAY',
 'NEIGHBORHOOD_SO. JAMAICA-BAISLEY PARK',
 'NEIGHBORHOOD_SOHO',
 'NEIGHBORHOOD_SOUTH JAMAICA',
 'NEIGHBORHOOD_SOUTH OZONE PARK',
 'NEIGHBORHOOD_SPRING CREEK',
 'NEIGHBORHOOD_SPRINGFIELD GARDENS',
 'NEIGHBORHOOD_ST. ALBANS',
 'NEIGHBORHOOD_SUNNYSIDE',
 'NEIGHBORHOOD_UPPER EAST SIDE (59-79)',
 'NEIGHBORHOOD_UPPER WEST SIDE (79-96)',
 'NEIGHBORHOOD_WASHINGTON HEIGHTS UPPER',
 'NEIGHBORHOOD_WILLIAMSBURG-NORTH',
 'NEIGHBORHOOD_WILLIAMSBURG-SOUTH',
 'BUILDING CLASS CATEGORY_01 ONE FAMILY DWELLINGS',
 'BUILDING CLASS CATEGORY_08 RENTALS - ELEVATOR APARTMENTS',
 'BUILDING CLASS CATEGORY_09 COOPS - WALKUP APARTMENTS',
 'BUILDING CLASS CATEGORY_10 COOPS - ELEVATOR APARTMENTS',
 'BUILDING CLASS CATEGORY_29 COMMERCIAL GARAGES',
 'BUILDING CLASS CATEGORY_32 HOSPITAL AND HEALTH FACILITIES',
 'TAX CLASS AT PRESENT_2',
 'TAX CLASS AT PRESENT_2A',
 'TAX CLASS AT PRESENT_2B',
 'TAX CLASS AT PRESENT_4',
 'BUILDING CLASS AT PRESENT_A1',
 'BUILDING CLASS AT PRESENT_A3',
 'BUILDING CLASS AT PRESENT_A4',
 'BUILDING CLASS AT PRESENT_A7',
 'BUILDING CLASS AT PRESENT_C2',
 'BUILDING CLASS AT PRESENT_C3',
 'BUILDING CLASS AT PRESENT_C5',
 'BUILDING CLASS AT PRESENT_C7',
 'BUILDING CLASS AT PRESENT_E2',
 'BUILDING CLASS AT PRESENT_E9',
 'BUILDING CLASS AT PRESENT_F1',
 'BUILDING CLASS AT PRESENT_H8',
 'BUILDING CLASS AT PRESENT_I5',
 'BUILDING CLASS AT PRESENT_K4',
 'BUILDING CLASS AT PRESENT_K7',
 'BUILDING CLASS AT PRESENT_L1',
 'BUILDING CLASS AT PRESENT_O4',
 'BUILDING CLASS AT PRESENT_P2']

