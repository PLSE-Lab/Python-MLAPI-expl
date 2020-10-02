#!/usr/bin/env python
# coding: utf-8

# ## Why linear regression?
# After trying out the ideas in this Notebook I already know that a linear regression model will not lead to a top score in the competion. The reason I wanted to try it out is that it's a model that is reasonably simple to understand and get an intuitive feel for. I can more or less code the algorithm myself and this brings the benefit that it's a lot easier to understand what is going on and make the right choices when trying to improve the model by cleaning the data, trying out different parameters, et c.
# 
# Just applying Random Forest or Gradient Boosting without knowing how they work or what each parameter of the model does feels kind of scary. It might be reasonably easy to get a good "score" but how can I tell if I reached the right conclusions, how well the model would transfer to a sligthly different scenario or how changes in the raw data would affect the model? I would love to learn more about these models as well but as I've recently gotten started with this I am taking one step at a time.
# 
# I've tried to include all relevant steps of my analysis but have omitted some that would result in a long output. Ok, let's get started with some imports and loading the data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import Ridge, RidgeCV, Lasso, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Load data.
train = pd.read_csv('../input/train.csv', index_col='id', parse_dates=[1], 
                    true_values=['yes'], false_values=['no'])
test = pd.read_csv('../input/test.csv', index_col='id', parse_dates=[1], 
                   true_values=['yes'], false_values=['no'])
print(train.shape)
print(test.shape)


# ## Data cleaning
# As a linear regression model cannot work with categorical variables I have to do some transformations. I will perform the following steps:
# 
# * Merge the test and training data. As I will fill out most of the NaNs with the median value I want to get medians that are representative of the entire data set.
# * Create a numerical scale for **ecology**
# * Create a boolean **is_investment** column based on **product_type**
# * Create a boolean column for each **sub_area**
# * Find NaN values and replace them, for the most part with the median value
# * For **life_sq**, **num_room** and **kitch_sq** we can do better assumptions than the median
# * **buildyear** contains some crazy outliers, clean those up
# 
# Note: I also tried the approach with creating a new column for columns where NaN values could be found, e.g. build_year_NaN set to True for a row where build_year was not available, but that did not yield any better result so I will skip this step here. Maybe it would work better for another type of model?

# In[ ]:


test['price_doc'] = -1
data = pd.concat([train, test])

# Fix ecology column
ecology_map = {'poor': 1, 'satisfactory': 2, 'good': 3, 'excellent': 4, 'no data': np.NaN}
data['ecology'] = data['ecology'].apply(lambda x: ecology_map[x])

# There are 33 NaNs in the product_type column. 
# Set them to is_investment=True as that is the most common value.
data['is_investment'] = data['product_type'].apply(
    lambda x: False if x == 'OwnerOccupier' else True)
del data['product_type']

# Create a categorical value for each sub area
sub_areas = list(data['sub_area'].unique())
for area in sub_areas:
    data[area] = data['sub_area'].apply(lambda x: True if x == area else False)
del data['sub_area']

# Find columns with NaN values...
column_names = data.columns.values.tolist()
NaN_columns = []
for i, col_name in enumerate(column_names):
    s = sum(pd.isnull(data.iloc[:,i]))
    if s > 0:
        NaN_columns.append(i)
# ...and set most of these to the median value
for i in NaN_columns:
    if i in [2, 7, 8]: # life_sq, num_rooms, kitchen_sq
        continue
    else:
        data[column_names[i]]=data[column_names[i]].fillna(data[column_names[i]].median())

# Update NaN values for life_sq, num_room and kitch_sq
life_sq_to_full_sq = float(data['life_sq'].sum()) /     float(data.loc[data['life_sq'] > 0, 'full_sq'].sum())
average_room_size = float(data.loc[data['num_room'] > 0, 'full_sq'].sum()) /     float(data['num_room'].sum())
life_sq_to_kitch_sq = float(data['kitch_sq'].sum()) /     float(data.loc[data['kitch_sq'] > 0, 'full_sq'].sum())
data.loc[data['life_sq'].isnull(), 'life_sq'] =     data.loc[data['life_sq'].isnull(), 'full_sq'] * life_sq_to_full_sq
data.loc[data['num_room'].isnull(), 'num_room'] =     np.round(data.loc[data['num_room'].isnull(), 'full_sq'] / average_room_size)
data.loc[data['kitch_sq'].isnull(), 'kitch_sq'] =     data.loc[data['kitch_sq'].isnull(), 'full_sq'] * life_sq_to_kitch_sq

# Remove outliers from buildyear
median_build_year = data['build_year'].median()
data['build_year'] = data['build_year'].apply(     lambda x: median_build_year if x < 1800 else median_build_year if x > 2017 else x)

# Should output a 0 meaning that there are no NaNs left.
data.isnull().sum().sum()


# There is a lot more that could be done with the data if you want to spend some time with it. I spent a few minutes looking at various outliers, e.g. the first and 99th quantile of the first columns. **max_floor** and **kitch_sq** certainly stands out but I will not spend time on analysing these values further.

# In[ ]:


print("column name\tquantile 1\tquantile 99")
column_quantiles = {}
for c in data.columns.values[2:10]:
    column_quantiles[c] = (data[c].quantile(.0001), data[c].quantile(.9999))
    print(c, "\t", data[c].quantile(.0001), "\t", data[c].quantile(.9999))


# ## Brief data exploration
# It's not my main mission to do an in-depth analysis of the various data points here but one big question is: how can we predict the price of an apartment when the data is collected over several years? Clearly time is a really important factor here - if there is a housing bubble in the middle of the period training a model on the years before the bubble will give poor predictions on what happens after.
# 
# What I (and probably Sberbank as well) am hoping for is that the macro file will contain data that, if fed to the model, will enable us to make predictions "independent" of time. Let's see if there is any ground for that assumption.
# 
# As we have no prices for the test set, July 2015 an onwards, we'll only look at the train set and hope the findings there can be extrapolated.

# In[ ]:


time_group = data.set_index('timestamp').groupby(pd.TimeGrouper(freq='M'))
plt.figure(figsize=(15,8))
(time_group['price_doc'].sum() / time_group['full_sq'].sum()).plot()
plt.title('Price per square meter for Moscow apartements', fontsize=16)
plt.xlabel('Time')
plt.ylabel('Price in RUB (starts at 60,000)')
plt.ylim([60000,180000])
plt.xlim(['2011-11-01', '2015-06-30'])
plt.grid()


# Above is a plot of price per square meter for the apartments in the training set. We start the plot for November 2011 as the sample size before that is quite small (see below). Prices seem to peak in the late spring 2012 and are then followed by a dip to the lowest during autumn of 2012. From there they have kept rising. Let's see how the prices compare to inflation, CPI. To do that we need to import the macro data.
# 
# ### Read the macro data
# The macro data contains some missing fields. A simple approach seems to be to use the 'ffill' option, i.e. if a value is missing in a row, just copy the value from the row above. This will not work if the first row is missing a value so we will first front-fill and the back-fill to cater for that.
# 
# I will also do some reformatting of columns that contain a comma (,) as thousand separator.

# In[ ]:


# Read macro data
# #! pattern found in some columns, treat as NaN
macro = pd.read_csv('../input/macro.csv', na_values='#!', parse_dates=[0])

# Fill in NaN values
macro.fillna(method='ffill', inplace=True)
macro.fillna(method='bfill', inplace=True)

# Remove thousand separator and convert to double
macro_column_names = macro.columns.values.tolist()
for i, col_name in enumerate(macro_column_names):
    if macro.ix[:,i].dtype == object:
        macro.ix[:,i] = macro.ix[:,i].str.replace(',','')
        macro.ix[:,i] = pd.to_numeric(macro.ix[:,i])


# ### Consumer prices compared to CPI
# After having loaded the macro data, let's do this comparison.

# In[ ]:


square_meter_time_series = time_group['price_doc'].sum() / time_group['full_sq'].sum()

fig, ax1 = plt.subplots(figsize=(15,8))
ax1.plot(square_meter_time_series, 'b-')
ax1.set_xlabel('Time')
ax1.set_ylabel('Price in RUB', color='b')
ax1.tick_params('y', colors='b')
ax1.set_ylim([60000, 180000])
ax1.grid()

ax2 = ax1.twinx()
ax2.plot(macro.timestamp, macro.cpi, '-', color='r')
ax2.set_ylabel('CPI', color='r')
ax2.tick_params('y', colors='r')
ax2.set_xlim([pd.to_datetime('2011-11-01'), pd.to_datetime('2015-06-30')])
# Set the scale to make the lines approximately match at the beginning of the period
ax2.set_ylim([110, 600])

fig.tight_layout()
plt.title('Price per square meter compared to Consumer Price Index (CPI)', fontsize=16)
plt.show()


# We can see that the dip in late 2012 was really a dip in real prices but after that the prices seem to have more or less followed the inflation except for the final 4 months of the period where the flattening out actually means a price dip if we look at the inflation. There are a lot of other variables in the macro data. Plotting some of them against the mean square meter price would be interesting but I will leave that for a later endeavour.
# 
# What I would like to investigate next is if there is a difference in the price development for low, medium and high priced apartements. _Note that I will now look at the actual price, not price per square meter._

# In[ ]:


plt.figure(figsize=(15,8))
plt.plot(time_group['price_doc'].quantile(.9), color='b')
plt.plot(time_group['price_doc'].quantile(.5), color='r')
plt.plot(time_group['price_doc'].quantile(.1), color='g')
plt.xlim(['2011-11-01', '2015-06-30'])
plt.title('Price for the apartment at the 10, 50 and 90 percentile', fontsize=16)
plt.ylabel('Price in RUB')
plt.xlabel('Time')
plt.grid()


# The price for the most expensive apartments seem to be a bit more volatile. It is interesting to see that during the dip in late 2012 the price for the cheapest apartements actually went up. Let's look at the sample sizes.

# In[ ]:


plt.figure(figsize=(15,8))
time_group.size().plot(kind='bar')
plt.title('Number of apartements for each month in dataset', fontsize=16)
plt.ylabel('Number of apartements')
plt.xlabel('Time')
plt.grid()


# The median price should be quite reliable but the 10- and 90th percentile respectively will be based on quite small samples for certain months; just above 20 apartements for the months where the total sample is about 200. Still, the price increase for the apartement on the 10th percentile (and drop for the 90th percentile) in the autumn 2012 is probably not just by chance. 
# 
# Not really sure what to make of it though. Did a downturn of the economy increase the demand for cheap apartements as people still needed to buy apartements but did not want to invest too much? You could probably draw a lot of conclusions for this by studying the data in more detail, e.g. "Were the largest apartements just not being sold those months?", "Did investement as opposed to owner-occupier purchase decrease?", et c., et c.
# 
# ## Prediction model
# So now over to the linear regression model.
# 
# The key challenge here will be to use the macro data together with the training data to create a good model. The price of an apartement is probably not only correlated to the macro data on the exact date of contract signing. Also, most of the columns in the macro data contain the same number for every day of each month. It makes sense to me to create one row of macro data per month and add that to the apartement data.
# 
# It could also be a good idea to try to assign different weights to the dates preceding the data of contract signing (and then get information from several preceding months into each apartement training example) but that's beyond what I want to try now.

# In[ ]:


# Create one entry per each year and month, 
# fill with mean value of each column over month
macro['YearMonth'] = macro['timestamp'].map(lambda x: 100*x.year + x.month)
year_month_group = macro.groupby(by='YearMonth')
macro_year_month = year_month_group.mean()

# Create a YearMonth attribute for the apartments as well
data['YearMonth'] = data['timestamp'].map(lambda x: 100*x.year + x.month)

# Now merge the data..
full_data = pd.merge(data, macro_year_month, how='left',                      left_on='YearMonth', right_index=True)
del full_data['timestamp']

# ..and split back into train/test set
last_train_row = train.shape[0]-1
train_proc = full_data.iloc[:last_train_row]
test_proc = full_data.iloc[last_train_row:]

# Move target price data into separate array
train_target_prices = train_proc['price_doc']
del train_proc['price_doc']
del test_proc['price_doc']


# I will now take 70% of the training set values to use for training the model and 30% for cross-validation to get a sense of how well the model performs.

# In[ ]:


# Set a random state for repeatability
random_state = 11

# Create a train/test split for model evaluation
X_train, X_test, y_train, y_test = train_test_split(     train_proc, train_target_prices, test_size=0.3, random_state=random_state)

# Create function for score metric, set any negatives to 0 to avoid math error
def rmsle(y_true, y_pred):
    negative_entries = y_pred[np.argwhere(np.isnan(np.log(y_pred+1)))]
    if (negative_entries):
        print(negative_entries)
        y_pred[y_pred < 0] = 0
    return np.sqrt(mean_squared_error(np.log(y_true+1), np.log(y_pred+1)))


# I will use the above function to evaluate the root mean square logarithmic error of the estimator with my cross-validation set. I tried to feed it (with the make_scorer function) to the model to be used during training but for some reason it didn't work out. This is probably a main drawback as the model will minimize the wrong kind of error but I just don't have any good ideas on how to make it work.
# 
# As taking the logarithm of a negative value will yield a math error we will say that each negative prediction is 0. A good model would, of course, not predict a negative price of an apartement but as I stated already in the beginning, linear regression will not be the key to winning this competition..

# In[ ]:


# RidgeCV
estimator = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 20.0, 100.0), 
                    fit_intercept=True, normalize=True, 
                    scoring='neg_mean_squared_error', cv=None, 
                    gcv_mode=None, store_cv_values=False)
estimator.fit(X_train, y_train)
print("Root mean square logarithmic error:", rmsle(y_test, estimator.predict(X_test)))
print("Best alpha", estimator.alpha_)


# So what does a RMSLE of 0.55 mean? Let's compare the actual and predicted prices for a couple of apartements.

# In[ ]:


print("\tActual price\t\tPredicted price")
for e in enumerate(zip(y_test[100:120], estimator.predict(X_test)[100:120])):
    print(e[0],"{:20,}".format(e[1][0]), "\t{:20,.0f}".format(e[1][1]))


# It clearly does something and some of the values are sort of right but a few are way off. If I were to sell my apartement and the real estate agent gave me an estimate that turned out to be twice or half the final sale price based on the agency's advanced machine learning model I would be less than impressed..
# 
# The way RMSLE works, an error of 1 would mean that on average the model misclassifies the price with a factor of ~2.71 (e). An error of 0.55 means an average misclassification of about 70%.
# 
# Let's see what features had the highest weight.

# In[ ]:


feature_weights = [x for x in zip(X_test.columns.values, estimator.coef_)]
feature_weights.sort(key=lambda x: x[1], reverse=True)
print("Strongest positive features")
for i in range(10):
    print(feature_weights[i])
print("\nStrongest negative features")
for i in range(10):
    print(feature_weights[-i-1])


# I'm not 100% sure how to interpret the fact that 18 of the 20 most important features are areas. On the one hand this might indicate why linear regression won't really work for this type of problem/data. The price of an apartement is heavily influenced by its geographical location. On the other hand, the "area attributes" can only take the values 0 or 1 and (although I have normalized all input variables) it might take a pretty strong coefficient to bump up the importance of these values enough to get the correct effect on the target variable (sale price).
# 
# Maybe a linear regression model would work better if we had a "distance from city center" attribute that would take on linear and not binary values.
# 
# The way a human would estimate the price would certainly be to look at similar apartements in the area so any estimator that could simulate that would probably do better. Maybe that is what you could get a random forest regressor to do (i.e. split on the area attribute high up in the decision trees) but I'll leave that for a future experiment.
# 
# ## Conclusion
# I want to emphasize that I'm quite new at this and I probably made several mistakes along the way. I would be very happy if someone could point these out - that's the best way to learn. Further, I did not put too much rigour and thought into the different steps and perhaps there are ways to make this work better that I just did not see because I didn't spend enough time thinking about it but that's the benefit of not doing this professionally - I can just do enough to learn and then call it a day and move on once I think I have learned enough. To do a real analysis of all this data would for sure take months if not years.
# 
# ### What to do next
# I actually tried out a few other linear models (Lasso and ElasticNet) that do not penalize the square of errors but the absolute errors and they resulted in slightly, but not _that_ much, better results. It might also have been an idea to try out other things with the macro data, e.g. using a longer period of macro data instead of just the current month. However, I have a strong feeling that linear regression is not the way to approach this problem so if I continue working on it I will try out some other models.
# 
# Thank you, if you read all the way here and now at least you know what not to try out for the competition and hopefully you agree with my theory about why!
