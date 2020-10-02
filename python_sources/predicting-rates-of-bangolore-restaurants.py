#!/usr/bin/env python
# coding: utf-8

# The goal of this kernel is to predict the rate of each restaurant. To achieve this, we the clean the data, do feature engineering (count encoding, one-hot encoding, coordinates, distances, # reviews), and test different models (linear, svm and boosting). I will not go in detail of some part of the code, if you are interested please let me know in comments and upvote if find it interesting :). Note also that, I skipped some EDA part that could be find in other kernels.

# # Loading packages

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # Loading data

# In[ ]:


df = pd.read_csv('../input/zomato.csv',low_memory=False)


# # Information on data

# In[ ]:


df.info()


# The data contain 51.717 instances and 17 features from which only one is of type int. One can note, there are missing values in some features (the number of instances is different to 51.717, for example the feature **rate**.
# Let's now check the number of unique values of each feature. In case we have a feature with unique value, it will not be informative for any model.

# In[ ]:


print('Features \t # unique values\n')
for col in list(df):
    print(f'{col}:\t{df[col].nunique()}')


# As can be seen, all the features have non-unique values. However it doesn't mean that there are all informative.
# Note that the number of unique values for features reviews_list and the adress are different from the number of instances in the data. Does it mean that some restaurant are in the same adress receive the same reviews ?

# # Data cleansing

# Let's first look at the top 2 instances of the data.

# In[ ]:


df.head(2)


# ## Renaming some columns

# In[ ]:


df.rename({'approx_cost(for two people)': 'approx_cost_2_people',
           'listed_in(type)':'listed_in_type',
           'listed_in(city)':'listed_in_city'
          }, axis=1, inplace=True)


# ## Converting votes and approx_cost to numeric

# First, let's convert features **votes** and **approx_cost_2_people** to int and drop the features **url** and **phone**, I don't think if they would help any model to accurately predict the rate. Do you have any idea how they could ?  

# In[ ]:


# approx_cost constains some values of format '1,000' wich could not be directly convert to int
# we need to have this format '1000' in order to do the convertion
# We will use the lambda function below to transform '1,000' to '1000' and then to int
replace_coma = lambda x: int(x.replace(',', '')) if type(x) == np.str and x != np.nan else x 
df.votes = df.votes.astype('int')
df['approx_cost_2_people'] = df['approx_cost_2_people'].apply(replace_coma)
df = df.drop(['url', 'phone'], axis=1)


# In[ ]:


df.rate.dtype, df.rate[0]


# In[ ]:


df.rate.unique()


# In[ ]:


(df.rate =='NEW').sum(), (df.rate =='-').sum()


# The current type of our target is **'O'** i.e object. From the unique values, we can note the value **NEW** which is not a rate. We can also not the format of rates are not the same **number/5** and **number /5** (example 4.1/5 and 3.4 /5, note the space between 3.4 and /5). Let's drop instances with rates ='NEW' and '-' which number of instances are respectively 2208 and 69 then we will transform the rate to the format **number**, and type **float**.

# In[ ]:


df = df.loc[df.rate !='NEW']
df = df.loc[df.rate !='-'].reset_index(drop=True)


# In[ ]:


print(f'The new shape of the date is {df.shape}')


# In[ ]:


new_format = lambda x: x.replace('/5', '') if type(x) == np.str else x
df.rate = df.rate.apply(new_format).str.strip().astype('float')
df.rate.head()


# # Converting data to numeric

# Here we will label encode each feature for example 'yes' and 'no' will be converted to 0 and 1.

# In[ ]:


def label_encode(df):
    for col in df.columns[~df.columns.isin(['rate', 'approx_cost_2_people', 'votes'])]:
        df[col] = df[col].factorize()[0]
    return df


# In[ ]:


df_encoded = label_encode(df.copy())
df_encoded.head()


# # Exploratory data analysis (EDA)

# In[ ]:


target = df_encoded.rate.fillna(df_encoded.rate.mean()) # Filling nan values in target by mean


# In[ ]:


plt.figure(figsize=(10,4))
sns.distplot(target)
plt.title('Target distribution')


# From the figure above, we can see that the distribution of the rates is not uniform! We have restaurants we high rates while other have low.

# Correlation between features.

# In[ ]:


corr = df_encoded.corr(method='kendall') # kendall since some of our features are ordinal.
df_encoded = df_encoded.drop(['rate'], axis=1).fillna(-1) #filling nan values by -1


# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True)


# From the heatmap, we can see that some variables are positively correlated and some other negatively. The highest correlation **0.63** is between **name** and **address** and the lowest **-0.44** between **book_table** and **approx_cost_2_people**. Highly correlated features maybe imply redondant information. So we might drop one the them. The correlations that will interest us at this stage are between our target **rate** and the other features. The more a feature is correlated to the target (positively or negatively) the more it could help predicting the target. And from the plot, one can see that the 2 top positively correlated features to the target are **dish_liked**, **vote**. Which means the more a dish is liked and a restaurant receive higher votes the more the rate increase. One can also note from the plot that only one feature **book_table** is negatively correlated to the target. Which means that restaurant in which booking table is needed have low rate.

# # Modeling

# In this part, we will first split the data into training and test sets. Then we will run a random forest regressor to as baseline model to see how it could perform on this data. The performance of our models will be evaluated by computing the root [**mean squared error**](https://en.wikipedia.org/wiki/Mean_squared_error) (RMSE). This metric represent somehow the standard deviation the errors beetwen the true values and the predicted one. The smaller the RSME the better.

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


# Scikit-learn does't have the RSME metric, so let's create it.

# In[ ]:


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ## Spliting data into train and test

# In[ ]:


# Helper function for scaling data to range [0, 1]
# Linear models are very sensitve to outliers, so let's scaled the data.
minmax = lambda x: (x - x.min())/(x.max() - x.min())


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(minmax(df_encoded),target, random_state=2)


# Let's see how linear models (linear regression and svm regressor) and an ensemble one (random forest regressor) will perform.

# In[ ]:


lr = LinearRegression(n_jobs=-1)
svr = SVR()
rf = RandomForestRegressor(random_state=44, n_jobs=-1)
models = [lr, svr, rf]
for model in models:
    model.fit(x_train, y_train)
    pred_model = model.predict(x_test)
    print(f'The RMSE of {model.__class__.__name__} is {rmse(y_test, pred_model)}')


# From the scores above, the random forest regressor outperfomed the linear models. We will keep it as baseline model.

# ## Getting important features

# In[ ]:


def plot_importances(model, cols):
    plt.figure(figsize=(12,6))
    f_imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=True)
    f_imp.plot(kind='barh')


# In[ ]:


plot_importances(rf, list(x_train))


# From the figure above, the most important feature is **votes**, then **cuisines** and **name**. Remember when we on the heatmap votes were also the most correlated to our target! Book_table had the most negatively correlation with the feature, however it seems not to be an important feature for random forest. The third most important feature is **name**, my interpretation of the feature of being important is based on the reputation of the restaurant. Suppose you to go to the restaurant X, and you ask someone if it's a god or bad restaurant. Surely the answer will influence whether you go to X or change for another one. One could then say that our data contain famous restaurants and bad ones.

# ## Plotting true rates VS predicted

# In[ ]:


preds_rf = rf.predict(x_test)
pd.Series(preds_rf).plot(kind='hist', label='predictions')
y_test.reset_index(drop=True).plot(kind='hist', label='true values')
plt.legend()


# ## plotting the first tree

# Here we will plot the first tree of the random forest regressor, with a depth=3.

# In[ ]:


from sklearn.tree import export_graphviz
from IPython.display import Image


# In[ ]:


def convert_dot_to_png(model, max_depth=3, feature_names=list(x_train)):
    export_graphviz(model.estimators_[0], out_file='tree.dot', max_depth=max_depth, feature_names=feature_names, rounded=True)
    get_ipython().system('dot -Tpng tree.dot -o tree.png')


# In[ ]:


convert_dot_to_png(rf)
Image('tree.png')


# ## Tuning hyperparameters

# In this part we will tune some hyperparameters of our baseline model and see if it could imporve the score.

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


# A scoring differ from a metric by indicating whether the metric should be minimize or maximize. As the RMSE metric, Scikit-learn doesn't have RMSE scoring. Let's define it.

# In[ ]:


rmse_scoring = make_scorer(rmse, greater_is_better=False)


# In[ ]:


param_grid = {'n_estimators':[20, 50, 100], 'max_features': [None, 'sqrt', 0.5]}
grid_search = GridSearchCV(estimator= rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring=rmse_scoring)
grid_search.fit(x_train, y_train)
None


# In[ ]:


grid_search_pred = grid_search.predict(x_test)
score_grid_search = rmse(y_test, grid_search_pred)
best_estimator = grid_search.best_estimator_
print(f'The best estimator is:\n {best_estimator} \n and the it\'s score is {score_grid_search}')


# The best results of the grid search on the number of estimators and the max features is for **n_estimators=100** and **max_features=None**, with a score of 0.11 hence an improve of 0.0126.

# ## Could randomness improve our score ?

# Let's run the best model found above with different random seed and see if it could help us improve the score.

# In[ ]:


N_ITER=10
def run_n_iter(estimator, train, target,test, N_ITER=N_ITER):
    pred_n_iter = np.zeros((y_test.shape[0],), dtype='float')
    for i in range(N_ITER):
        estimator.set_params(random_state= i, )
        estimator.fit(train, target)
        pred_n_iter += estimator.predict(test) / N_ITER
    return pred_n_iter

pred_n_iter = run_n_iter(best_estimator, x_train, y_train, x_test)
print(f'The RMSE of {N_ITER} iterations is {rmse(y_test, pred_n_iter)}')


# The randomness give us an imporve of **0.0013**! Could we improve the score significantly ? Let's get seriouse with the data!

# # Feature engineering

# In this part we will add some features to the data and see how they will impact the score.

# ## Count encoding

# Some of our features are categorical, the count encoding is the process of replacing feature values by their frequencies. It is a common feature engineering technic for categorical data.

# In[ ]:


# Defining a helper function for count encoding
def count_encoding(df, cat_cols):
    for col in cat_cols:
        count = df[col].value_counts()
        new_colname = col + '_count'
        df[new_colname] = df[col].map(count)
    return df


# In[ ]:


cat_cols = x_train.columns[~x_train.columns.isin(['votes', 'approx_cost_2_people'])]


# In[ ]:


df_ce = count_encoding(df_encoded.copy(), cat_cols)
x_train_ce, x_test_ce = df_ce.iloc[x_train.index], df_ce.iloc[x_test.index]


# In[ ]:


new_features = [col for col in list(x_train_ce) if 'count' in col]
x_train_ce.loc[:, new_features].head()


# Let's check if the new features will increase the score!

# In[ ]:


pred_n_iter_2 = run_n_iter(best_estimator, x_train_ce, y_train, x_test_ce)
print(f'The RMSE of with engineered features is {rmse(y_test, pred_n_iter_2)}')


# The new features seem to not help the model to accurately predict the target. This means that counting the number of occurence of target values is not informative for the model. 

# ## One-hot encoding

# The one-hot encoding is another common feature engineering technic for categorical data. In this process, the we will have a new feature for each feature value such that we have 1 if the is an occurence of the value and 0 otherwise. For example let say we have the attribute A = \['Yes', 'No'\] then we will have with the one-hot encoding the 2 new features A_Yes = \[1, 0\] and A_No = \[0, 1\].

# In[ ]:


# Defining a helper function for One-hot encoding
# For computational reasons, we will limit the one-hot encoding to 
# features having unique values less or equal 100.
def ohe(df, max_nunique_vals=100, drop_encoded_feature=True):
    for col in list(df):
        if df[col].nunique() <= max_nunique_vals:
            dummies = pd.get_dummies(df[col].astype('category'), prefix=col)
            df = pd.concat([df, dummies], axis=1)
            if drop_encoded_feature:
                df.drop(col, axis=1, inplace=True)
    return df


# In[ ]:


df_ohe = ohe(df_encoded.copy())
x_train_ohe, x_test_ohe = df_ohe.iloc[x_train.index], df_ohe.iloc[x_test.index]


# In[ ]:


x_train_ohe.iloc[:, 8:].head()


# Since the randomness doesn't improve significantly and time consuming for the random forest regressor, we will then run the best estimator with one seed.

# In[ ]:


best_estimator.set_params(random_state=5, n_jobs=-1) # we add n_jobs to speed up the computation.
best_estimator.fit(x_train_ohe, y_train)
pred_n_iter_3 = best_estimator.predict(x_test_ohe)
print(f'The RMSE of with engineered features is {rmse(y_test, pred_n_iter_3)}')


# From the result above, the one-hot encoding decreasedthe score! So it's not a good solution here.

# ## Coordinates data

# Now, let's add coordinates data of restaurants to our original data and see if they will be informative.

# In[ ]:


def get_lat_lon(df):
    # modified code from https://www.kaggle.com/shahules/zomato-complete-eda-and-lstm-model
    locations=pd.DataFrame({"Name":df['location'].unique()})
    locations['Name']=locations['Name'].apply(lambda x: "Bangalore " + str(x))
    lat=[]
    lon=[]
    geolocator=Nominatim(user_agent="app")
    for location in locations['Name']:
        location = geolocator.geocode(location)
        if location is None:
            lat.append(np.nan)
            lon.append(np.nan)
        else:    
            lat.append(location.latitude)
            lon.append(location.longitude)
    locations['lat']=lat
    locations['lon']=lon
    return locations


# In[ ]:


locations = get_lat_lon(df)
# Merging the coordiantes data to the original one
df_coord = df_encoded.copy()
unique_locations = df_coord.location.unique()
df_coord['lat'] = df_coord.location.replace(unique_locations, locations.lat)
df_coord['lon'] = df_coord.location.replace(unique_locations, locations.lon)
df_coord.iloc[:, -5:].head()


# In[ ]:


df_coord = df_coord.fillna(-1)
x_train_coord, x_test_coord = df_coord.iloc[x_train.index], df_coord.iloc[x_test.index]


# In[ ]:


best_estimator.fit(x_train_coord, y_train)
pred_n_iter_4 = best_estimator.predict(x_test_coord)
print(f'The RMSE of with coordinates features is {rmse(y_test, pred_n_iter_4)}')


# The coordiantes features have no influence on score! Let's add the distance from the restaurants to the most place to visit in Bangalor (Lal Bagh) I found on internet. We will use the [**harversine**](https://en.wikipedia.org/wiki/Haversine_formula) distance.

# In[ ]:


# Intalling the package
# If not install in your kaggle docker, uncomment to install it.
# !pip install haversine
from haversine import haversine


# In[ ]:


# default unit of distance is in km
lal_bagh_coordinates= (12.9453, 77.5901)
df_coord['distance_to_lalbagh_km'] = [haversine((lat, lon), lal_bagh_coordinates)
                                    for (lat, lon) in df_coord[['lat','lon']].values]
df_coord.iloc[:, -5:].head() 


# In[ ]:


x_train_dist, x_test_dist = df_coord.iloc[x_train.index,:], df_coord.iloc[x_test.index,:]


# In[ ]:


best_estimator.fit(x_train_dist, y_train)
pred_n_iter_5 = best_estimator.predict(x_test_dist)
print(f'The RMSE of with coordinates features and distance to Lal Bagh is {rmse(y_test, pred_n_iter_5)}')


# Using the distance our score increased a little bit. Let's try now boosting models. For these models, we will use the original data first then plus some engineered features one: count encoded features, coordinates and the distance.

# ## Adding # of reviews to the data

# Let's add the number of reviews to the data. This feature could be interesting as the number of reviews could give an idea on the quality of the restaurant. Restaurants with high reviews could ether mean that it is of good quality so that people would like to recommand it or bad quality to aware people of going there.

# In[ ]:


df_coord['nb_review'] = [len(val) for val in df['reviews_list']]
x_train_dist_coord_review, x_test_dist_coord_review = df_coord.iloc[x_train.index,:], df_coord.iloc[x_test.index,:]
df_coord.iloc[:, -5:].head()


# In[ ]:


best_estimator.fit(x_train_dist_coord_review, y_train)
pred_n_iter_6 = best_estimator.predict(x_test_dist_coord_review)
print(f'The RMSE coordinates features, distance to Lal Bagh and # of reviews is {rmse(y_test, pred_n_iter_6)}')


# Seems that we are adding noise to the data (our score decreased!).

# ## Boosting models

# In[ ]:


import lightgbm as lgb
import catboost as cat
import xgboost as xgb


# In[ ]:


def run_models(models, x_train, y_train, x_test, y_test):
    preds = np.zeros((x_test.shape[0], 3), dtype='float')
    for i, model in enumerate(models):
        model.fit(x_train, y_train)
        tmp = model.predict(x_test)
        print(f'The RMSE of {model.__class__.__name__} is {rmse(y_test, tmp)}')
        preds[:, i] = tmp
    return preds


# In[ ]:


clf_lgb = lgb.LGBMRegressor(random_state=97)
clf_cat = cat.CatBoostRegressor(random_state=2019, verbose=0)
clf_xgb = xgb.XGBRegressor(random_state=500)
models = [clf_lgb, clf_cat, clf_xgb]
preds_models = run_models(models, x_train, y_train, x_test, y_test)


# From the results above, light gradient boosting machine method outperformed Catboost and XGBoost! However the score obtained is lower than the one of random forest. Let's run the models with engineered features.

# In[ ]:


preds_models_2 = run_models(models, x_train_dist, y_train, x_test_dist, y_test)


# The 3 previous scores decreased but still lower thant our baseline model.

# ## Blending

# Blending is a technic of combining the predictions of different estimators. Perfect blending would be with estimators having approximately the same score but different distribution.

#  Let's ckeck if the distribution of our GBM estimators are different. We will use the [**Kolmogorov-Smirnov**](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) statistic (K-test) which is a two-sided test for the null hypothesis that 2 independent samples are drawn from the same continuous distribution. If the K-S statistic is small or the p-value is high, then we cannot reject the hypothesis that the distributions of the two samples are the same.

# In[ ]:


from scipy.stats import ks_2samp


# In[ ]:


GBM_models = ['LGB', 'CatBoost', 'XGBoost']
for i in range(2):
    print('The p-value between {0} and {1} is {2}'.format(GBM_models[0], GBM_models[i+1], ks_2samp(preds_models_2[:, 0],
                                                                                                   preds_models_2[:, i + 1])[1]))


# The p-value of the K-test of the 3 obtained predictions are very low, then we can conclude that the predictions are from the same distribution, hence a blend will not be very effective. Let's now compute the K-test between the predictions of our baseline model and the LGB one (best score of GBM models).

# In[ ]:


print('The p-value between {0} and {1} is {2}'.format('Random forest regressor', GBM_models[0], 
                                                      ks_2samp(preds_rf,preds_models_2[:, 0])[1]))


# The K-test between the random forest and LGBM is lower than those obtained with GBM models. Let's blend the 2 predictions and see the score even it would not be less than the one obtained our baseline model.

# In[ ]:


blend_1 = 0.95*preds_rf + 0.05*preds_models_2[:, 0]
blend_2 = 1.05*preds_rf - 0.05*preds_models_2[:, 0]
print(f'The p-value of the blended predictions (1) between RF and LGBM is {rmse(y_test, blend_1)}')
print(f'The p-value of the blended predictions (2) between RF and LGBM is {rmse(y_test, blend_2)}')


# # Final model

# Up to now we were using only some part of the data (x_train, y_train) for training and other part for testing (x_test, y_test). By doing this, our models are missing information contained on the test set. To overcome this, the common way is doing cross-validation.

# In[ ]:


from sklearn.model_selection import KFold


# In[ ]:


def cv(model, train, target, n_splits=5):
    oof = np.zeros((train.shape[0],), dtype='float')
    kf = KFold(n_splits=n_splits, shuffle=True)
    scores = pd.Series(np.zeros((n_splits,)))
    for i, (tr_idx, te_idx) in enumerate(kf.split(train, target)):
        x_tr, x_te = train.loc[tr_idx], train.loc[te_idx]
        y_tr, y_te = target[tr_idx], target[te_idx]
        model.fit(x_tr, y_tr)
        tmp = model.predict(x_te)
        oof[te_idx] = tmp
        scores[i] = rmse(y_te, tmp)
        print('Fold {} score {}'.format(i, scores[i]))
    return oof, scores


# In[ ]:


# The cross-validation is done on the original data + engineered features (count, coordinates, distance)
oof, scores = cv(best_estimator, df_coord.drop('nb_review', axis=1), target)


# In[ ]:


print('Mean score {}, STD {}'.format(scores.mean(),scores.std()))


# The mean score of the cross-validation is quite similar to the one obtained with our baseline model and the standard deviation is low. Hence, the performance of the random forest regressor on this data set and engineered feature is **stable**.

# # Conclusion

# The goal of this kernel is to predict the rate of the restaurants in Bangalore. We tried different models and different feature engineering. The best model we obtained is the random forest regressor which outperfomed boosting models (big surprise!). A cross-validation showed that the best estimator performance is stable. Please note that, the code and results of this kernel could be improved!. Indeed, other models (like neural nets) could be tested and a deep hyperparameters tuning could be done. Don't hesitate to fork and try new ideas. 

# # Upvote :)

# If you like the work done in this kernel, please upvote!
