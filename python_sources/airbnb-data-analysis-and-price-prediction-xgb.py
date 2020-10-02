#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import json
import math
import xgboost as xgb
import time

from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.gofplots import qqplot


# # 1. Data preprocessing and exploration

# Open data set, cleaning

# In[ ]:


data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.info()


# In[ ]:


data.describe()


# In[ ]:


data[:3]


# ## 1.1 Helper classes

# In[ ]:


class DataHelper:
    '''
    Helper class for plots and data handling functions.
    '''
    
    def __init__(self, data):
        self.data = data
        
    def update_data(self, data):
        self.data = data
        
    def drop_columns(self, column_names):
        '''
        Drop some columns. Must update data with the returned new data set
        '''
        self.data = self.data.drop(column_names, axis = 1)
        return self.data
        
    def get_count_unique_vals(self, column_names):
        '''
        For each of the column names as parameter, count the unique different values
        '''
        counts = {}
        for column_name in column_names:
            counts[column_name] = self.data.groupby(column_name)[column_name].nunique().count()
        return counts
    
    def show_price_distribution(self):
        '''
        Show the price distribution in 3 graphs:
        - price histogram distribution
        - log of price histogram distribution
        - how close the distribution matches a normal distribution (the red line)
        '''
        fig, ax = plt.subplots(1, 3, figsize=(23, 5))
        sns.distplot(self.data['price'], ax = ax[0])
        sns.distplot(self.data['log_price'], ax = ax[1])
        qqplot(self.data['log_price'], line ='s', ax = ax[2])
        ax[2].set_title('Comparison with theoritical quantils of normal distribution')
        
    def show_count_and_distrib_categorical_feature(self, column_name):
        '''
        For a categorical feature (with not too many categories to be rendered), display the count
        and the price distribution for each category.
        '''
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        sns.countplot(self.data[column_name], ax=ax[0])
        ax[0].set_title("Offers count per " + column_name)
        ax[0].set_xlabel(column_name)
        sns.boxplot(x=column_name, y='log_price', data = self.data, ax=ax[1])
        ax[1].set_title("Price per " + column_name)
        ax[1].set_xlabel(column_name)
        ax[1].set_ylabel("Log of price")
        fig.show()
        
    def show_numerical_feature_distribution(self, column_name, bins=30):
        '''
        Show the count distribution for a numerical feature in two plots, one with a log xscale and another
        with normal xscale.
        '''
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].hist(self.data[column_name], bins)
        ax[0].set_title('count of ' + column_name)
        ax[0].set_xlabel(column_name)
        ax[0].set_ylabel("count")
        ax[0].set_yscale('log')
        ax[1].hist(np.log(1 + self.data[column_name]), bins)
        ax[1].set_title('count of ' + column_name)
        ax[1].set_yscale('log')
        ax[1].set_xlabel('Log of ' + column_name)
        ax[1].set_ylabel("count")
        ax[1].set_yscale('log')
        
    def get_pearson_features_correlation(self):
        '''
        Show matrix of pearson features correlation
        '''
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.grid(True)
        ax.set_title("Pearson's correlation")
        corr = self.data.corr()
        sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, ax=ax)
        
    def get_dummies(self):
        '''
        Get dummies for every categorical feature in the data set. Remove the categorical columns.
        Must update data with the returned new data set
        '''
        categorical_columns = list(self.data.select_dtypes(include=['object']).columns)
        for column in categorical_columns:
            self.data = self.data.drop(column, axis = 1).join(pd.get_dummies(self.data[column]))
        return self.data
            
    def standardize_columns(self, column_names):
        '''
        Standardize every columns passed in arg.
        '''
        self.data[column_names] = StandardScaler().fit_transform(self.data[column_names])
        
# Log price will be need in many places later
data['log_price'] = np.log(1 + data['price'])
data_helper = DataHelper(data)


# ## 1.2 Clean useless columns

# In[ ]:


data_helper.get_count_unique_vals(['name', 'host_name', 'host_id', 'neighbourhood'])


# 'name' has any useful information with complex processing (using NLP for instance). There are too many distinct values in 'host_name' and 'host_id' to be usable. We can use the 'neighbourhood' value however.

# In[ ]:


data = data_helper.drop_columns(['host_id', 'id', 'name', 'host_name'])


# ## 1.3 Deal with missing values
# 
# From the data info, last_review and reviews_per_month are missing the same number of values, let's see what we should fill them by.

# In[ ]:


# Count missing values
data[pd.isnull(data['last_review'])]['price'].count()


# In[ ]:


data[pd.isnull(data['last_review'])]['number_of_reviews'].mean()


# Number of reviews are always 0, reviews_per_month missing values should be 0. We will use the earliest date of the dataset to fill last_review as it never was reviewed. This is definitely not ideal as maybe it is just caused by the offer being too recent to have been rated yet, so we could also argue to use the mean of every date instead (or even try to predict it from the other features but this would be very difficult). We will add a new column "never_reviewed" to make the distinction better as it might be an important feature.

# In[ ]:


data['never_reviewed'] = pd.isnull(data['last_review'])
data['reviews_per_month'] = data['reviews_per_month'].fillna(0)

# Remove na and convert to int for usage
data['last_review'] = pd.to_datetime(data['last_review'])
min_date_review = data['last_review'].min()
data['last_review'] = data['last_review'].fillna(min_date_review)
data['last_review'] = data[['last_review']].apply(lambda x: x[0].timestamp(), axis=1)


# In[ ]:


data.info()


# ## 1.4 Price distribution

# In[ ]:


data_helper.show_price_distribution()


# The data seems to follow a normal distribution for the most part but there are some outliers at extreme prices (around <2.6 and >8.5). They might not real announces for Airbnb (we can imagine there is no price at 0 except if they specify the price in another manner in the rental. And some of the very expensive at 10000 might be the maximum price, put for trolling or also because maximum price is reached). Depending of the model used (if the model assumes a normal distribution, such as Naives Bayes), it would be better to remove those data points to get better results. For other models such as Random Forests, we don't assume any distribution so we could keep it. As we might not even want to predict such outliers/trolls and it simplifies to always remove them, we will simply remove them all the time.

# In[ ]:


oldCount = data['price'].count()
data = data[data['log_price'] > 2.6][data['log_price'] < 8.5]
data_helper.update_data(data)
print("Data points lost: ", oldCount - data['price'].count())

#qqplot(np.log(1 + data['price']), line ='s').show()
data_helper.show_price_distribution()


# ## 1.5 Features exploration

# ### 1.5.1 Neighbourhood group

# We focus here on the neighbourhood groups and not neighbourhood as there are 221 different neighbourhoods!

# In[ ]:


data_helper.show_count_and_distrib_categorical_feature('neighbourhood_group')


# ### 1.5.2 Geographic position

# In[ ]:


# Density distribution of the offers geographical position
sns.jointplot(x="longitude", y="latitude", data=data, kind="kde")


# ### 1.5.3 Room type

# In[ ]:


data_helper.show_count_and_distrib_categorical_feature('room_type')


# ### 1.5.4 minimum of nights

# In[ ]:


data_helper.show_numerical_feature_distribution('minimum_nights')


# ### 1.5.5 number_of_reviews

# In[ ]:


data_helper.show_numerical_feature_distribution('number_of_reviews')


# ### 1.5.6 last_review

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.hist(data['last_review'], 100)
ax.set_title('count of last_review')
ax.set_xlabel('last_review (timestamps)')
ax.set_ylabel("count")
ax.set_yscale('log')


# The pike at the beginning is due to the handling of nan, so beside that, nothing particular to observe here.

# ### 1.5.7 reviews_per_month

# In[ ]:


data_helper.show_numerical_feature_distribution('reviews_per_month')


# ### 1.5.8 calculated_host_listings_count

# In[ ]:


data_helper.show_numerical_feature_distribution('calculated_host_listings_count')


# ### 1.5.9 availability_365

# In[ ]:


data_helper.show_numerical_feature_distribution('availability_365')


# Data seems here very skewed around 0. We will try to look a it further.

# In[ ]:


data[data['availability_365'] == 0]['availability_365'].count()


# About a third of the offers are never available, they might not be up to date anymore. Hence there is potentially a big difference between 0 and any other value here. We could try to turn this feature into multiple ones (such as boolean features if available) or not, but we could loose some information this way or simply add some redondant information for when it is exactly 0. It seems the safest option to improve the results without much risk degrading them.

# In[ ]:


data['never_available'] = data['availability_365'] == 0


# ### 1.5.10 Features correlation (Pearson's correlation)

# In[ ]:


data_helper.get_pearson_features_correlation()


# # 2 Predictions

# Format data for predictions. Note that we will always use the log_price as target.

# In[ ]:


# Standardize the data (necessary for Ridge and does not afect xgBoost)
data_helper.standardize_columns(['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'last_review', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365'])

# Get dummies
data = data_helper.get_dummies()


# In[ ]:


# Important to use a seed to have the same base of comparison and data reproducability
seed = 7805

# Train set and test set that will only be used at the end to compare the results at the very end.
X_train, X_test, y_train, y_test = train_test_split(data.drop(['price', 'log_price'], axis=1), data['log_price'], test_size=0.20, random_state=seed)


# ## 2.1 Models classes
# ### 2.1.1 RidgeRegression Model class

# In[ ]:


class RidgeRegressionHelper:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
    def find_best_params(self, seed=None, kfolds=5, alphas=[1]):
        '''
        Perform grid search over the list of alphas
        '''
        kfolds = KFold(n_splits=kfolds, random_state=seed)
        results = []
        for alpha in alphas:
            model = Ridge(alpha = alpha)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=kfolds, scoring='neg_mean_absolute_error')
            results.append([alpha, -scores.mean(), scores.std()])
            
        return pd.DataFrame(results, columns=['alpha', 'mae-mean', 'mae-std'])
    
    def get_performance_score(self, seed=None, kfolds=5, alpha=0.3):
        '''
        Display performance metrics on the training and testing data sets after training the model using the training data and the parameters passed as arguments.
        '''
        # Cross validation
        kfolds = KFold(n_splits=kfolds, random_state=seed)
        model = Ridge(alpha = alpha)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=kfolds, scoring='neg_mean_absolute_error')
        
        # Train and fit to testing set
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        
        results = []
        results.append(['Ridge', -scores.mean(), scores.std(), mean_absolute_error(self.y_test, y_pred), r2_score(self.y_test, y_pred)])
        return pd.DataFrame(results, columns=['Model', 'CV MAE', 'CV MAE std', 'test MAE', 'r2 test score'])
    
    def show_results(self, results):
        '''
        Plot the results MAE relative to the alpha values (use log x scale).
        '''
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(results['alpha'], results['mae-mean'])
        ax.set_xlabel('alpha')
        ax.set_ylabel('MAE')
        ax.set_xscale('log')


# ### 2.1.2 XGBoostRegression Model class

# In[ ]:


class XGBoostRegressionHelper:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.dtrain = xgb.DMatrix(X_train, label=y_train)
        self.dX_test = xgb.DMatrix(X_test)
        self.y_test = y_test
        
    def find_best_params(self, seed=None, kfolds=5, learning_rates=[0.3], max_depths=[6], min_child_weights=[1], num_boost_round=1000):
        '''
        Performs a grid search given the list of parameters.
        
        :param int seed: number used for replicable results
        :param int kfolds: number of folds to use during the cross validation
        :param array of floats learning_rates: different learning rates values to test
        :param array of int max_depths: different max depth values to test
        :param array of int min_child_weights: different mi child weight values to test
        :param int num_boost_round: (same a n_estimators) max number of boosting rounds to do, can stop before
        :returns Dataframe: result set containing mae results and time taken for each combinaton of parameters
        '''
        results = []
        for learning_rate in learning_rates:
            for max_depth in max_depths:
                for min_child_weight in min_child_weights:
                    params = {
                        'max_depth':max_depth,
                        'min_child_weight': min_child_weight,
                        'eta':learning_rate,
                        'objective':'reg:squarederror',
                    }
                    timestamp = time.time()
                    cv_results = xgb.cv(
                        params,
                        self.dtrain,
                        num_boost_round=num_boost_round,
                        seed=seed,
                        nfold=kfolds,
                        metrics={'mae'},
                        early_stopping_rounds=10
                    )
                    totalTime = time.time() - timestamp
                    boostRounds = cv_results['test-mae-mean'].idxmin()
                    results.append([learning_rate, max_depth, min_child_weight, cv_results['test-mae-mean'][boostRounds], cv_results['test-mae-std'][boostRounds], boostRounds, totalTime])
        
        return pd.DataFrame(results, columns=['learning_rate', 'max_depth', 'min_child_weight', 'mae-mean', 'mae-std', 'boostRounds', 'totalTime_seconds'])
    
    def show_results_and_times_relative_to_parameter(self, results, parameter_name):
        '''
        Given a result set from find_best_params, show the mae-mean score and time taken given the best score of each value of one of the parameters.
        
        :param DataFrame results: result set obtained from find_best_params
        :param str parameter_name: name of one of the parameter to optimize in find_best_params
        '''
        best_scores = results.loc[results.groupby(parameter_name)['mae-mean'].idxmin()]
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(best_scores[parameter_name], best_scores['mae-mean'])
        ax[0].set_xlabel(parameter_name)
        ax[0].set_ylabel('MAE')
        ax[1].plot(best_scores[parameter_name], best_scores['totalTime_seconds'])
        ax[1].set_xlabel(parameter_name)
        ax[1].set_ylabel('Total time for CV in seconds')
        
    def get_performance_score(self, seed=None, kfolds=5, learning_rate=0.3, max_depth=6, min_child_weight=1, num_boost_round=1000):
        '''
        Display performance metrics on the training and testing data sets after training the model using the training data and the parameters passed as arguments.
        '''
        params = {
            'max_depth':max_depth,
            'min_child_weight': min_child_weight,
            'eta':learning_rate,
            'objective':'reg:squarederror',
        }
        
        # Cross validation part
        cv_results = xgb.cv(
            params,
            self.dtrain,
            num_boost_round=num_boost_round,
            seed=seed,
            nfold=kfolds,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        boostRounds = cv_results['test-mae-mean'].idxmin()
        
        # Train and fit to testing set
        xgb_reg = xgb.train(
            params,
            self.dtrain,
            num_boost_round=num_boost_round,
        )
        
        y_pred = xgb_reg.predict(self.dX_test)
        
        results = []
        results.append(['XGBoost', cv_results['test-mae-mean'][boostRounds], cv_results['test-mae-std'][boostRounds], mean_absolute_error(self.y_test, y_pred), r2_score(self.y_test, y_pred)])
        return pd.DataFrame(results, columns=['Model', 'CV MAE', 'CV MAE std', 'test MAE', 'r2 test score'])
    
    def get_features_importance(self, learning_rate=0.3, max_depth=6, min_child_weight=1, num_boost_round=50, max_num_features=10):
        '''
        Get the ranking of feature importance as a barh graph.
        '''
        params = {
            'max_depth':max_depth,
            'min_child_weight': min_child_weight,
            'eta':learning_rate,
            'objective':'reg:squarederror',
        }
        
        xgb_reg = xgb.train(
            params,
            self.dtrain,
            num_boost_round=num_boost_round,
        )
        
        xgb.plot_importance(xgb_reg, max_num_features=max_num_features)


# ## 2.2 Baselines

# In[ ]:


# Ridge baseline
ridgeHelper = RidgeRegressionHelper(X_train, y_train, X_test, y_test)
ridgeHelper.get_performance_score(seed=seed)


# In[ ]:


# XGBoost baseline
xgbHelper = XGBoostRegressionHelper(X_train, y_train, X_test, y_test)
xgbHelper.get_performance_score(seed=seed)


# ## 2.3 RidgeRegression hyperparameter tunning

# In[ ]:


# Finding best alpha
results = ridgeHelper.find_best_params(seed = seed, alphas=[0.5, 1, 2, 5, 10, 20])
ridgeHelper.show_results(results)


# We can already see that optimizing alpha will change almost nothing in the MAE results. We can still try to tune alpha slightly more around 1-7.

# In[ ]:


# Finer tuning around 5
finer_results = ridgeHelper.find_best_params(seed = seed, alphas=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
ridgeHelper.show_results(finer_results)


# In[ ]:


best_result = finer_results.loc[finer_results['mae-mean'].idxmin()]
best_alpha = best_result['alpha']
best_result


# In[ ]:


ridgePerformance = ridgeHelper.get_performance_score(seed=seed, alpha=best_alpha)
ridgePerformance


# ## 2.4 XGBoostRegression hyperparameters tunning

# In[ ]:


# Try to find the best parameters for XGBoost
results = xgbHelper.find_best_params(seed = seed, learning_rates=[0.05, 0.1, 0.3], max_depths=[5, 7, 9], min_child_weights=[1, 3, 5])


# In[ ]:


# Let's have a look at how the learning rate influence the results and the time it takes
xgbHelper.show_results_and_times_relative_to_parameter(results, 'learning_rate')


# In[ ]:


# For a better understanding, we can directly look at the results
results


# From these results, we can see that we might be able to get slightly better results by reducing the learning rate further and/or increasing the max_depth. min_child_wieght cannot be further reduced however so we will keep it at 1. However we can also see the execution time growing a lot, so it might not be a worth using a smaller learning rate if the execution time increase too much compared to the improvement.

# In[ ]:


finer_results = xgbHelper.find_best_params(seed = seed, learning_rates=[0.01, 0.025, 0.05], max_depths=[9, 11, 13], min_child_weights=[1])


# In[ ]:


xgbHelper.show_results_and_times_relative_to_parameter(finer_results, 'learning_rate')
finer_results


# In[ ]:


best_result = finer_results.loc[finer_results['mae-mean'].idxmin()]
best_result


# The results are indeed slightly better but as expected the execution time went up 5 folds! Knowing that it would only cost a fifth of that to train the model (we were using 5-fold cross-validation), it would be around 2m30s. This is a very acceptable value in this context but if we were to often update this data set with more values or simply have a bigger dataset, we should pick some higher learning rate with only slightly worse results (like 0.05 with a max_depth of 11).

# In[ ]:


xgbPerformance = xgbHelper.get_performance_score(seed=seed, learning_rate=0.01, max_depth=11, min_child_weight=1, num_boost_round=500)
xgbPerformance


# We can also look at the feature importance, which is often very important to know what part of the analysis to give greater focus and what kind of data is most valuable to collect in the future. However since we separated the categorical features into dummy variables, we would need to do some aggregate over them to get the real total value of this feature.

# In[ ]:


xgbHelper.get_features_importance(learning_rate=0.01, max_depth=11, min_child_weight=1, num_boost_round=500)


# ## 2.5 Discussion of the results

# Firstly we can see that we get much better results with xgBoost regression with r2 score 0.623 vs 0.557 for Rigde regression.
# In general xgb performs rather well on every problem including regression. Besides it runs fast, its computation time scales about linearly with the data set size and can be run using the gpu (which I did not do here as it required more configurations to have it running and for this small project scale was not worth it).
# 
# Looking at the features importance we see that the localization (latitude and longitude) are used the most by some margin along with the various neighbourhood groups, as they are 221 of them and one of them is seen in the top ten ranking, it makes it likely that they sum up to a very large importance together. This is not surprising as one could expect the location of any house or appartment is very determinant of its price in general.
# 
# What is surprising is how the 'availability_365' (number of days when listing is available for booking) is so used, I would not have thought that it could really be used much to predict the price of a renting. One hypothesis to explain that could be that some people are renting they own appartment/one of their room, only when it suits them and then are only ready to do so at a higher price. While people that rent them all year along are using their property for the sole purpose of renting and can accept lower price as the place would not be used otherwise. We can also note that that the 'never_available' column was not so important after all, perhaps because rentings never available are just not up to date and could have a very broad price distribution without much general information.
# 
# We can apply a similar logic for 'minimum_nights', 'last_review', 'number_of_reviews', 'review_per_month' and 'calculated_host_listing_count', for each of them having no minimum nights/many or recent review/many listing on their account would tend to show they are doing more "professional" full time renting with probably more predicable and perhaps lower prices.
# 
# 
# ### Future improvements
# To go further we could try to apply some basic NLP to the listings' names or even see if the names contain some value (for instance duplex/studio/X rooms/penthouse) as it seems a priori a good (but naive) way of telling if a listing will be expensive or not, that would basically be like expending the room-type feature to be much more specific and hold more information. A good sign in that direction is the importance of 'Private room' as it was the smallest group with the most information to get from.
# 
# We could also try keep some information about the host_id for the biggest host listers and put a default value for the ones with few listing to avoid too much of an explosion of features, following my reasoning on why the availability feature is so important.
# 
# There might also be some work that could be done on the host names to get their social background (hence standards for renting) from their name. But I do not know how much information can be acquired this way and it would need some data outside of this data set about it to get some good data. It probably would not help much but might still be useful as a lead.
