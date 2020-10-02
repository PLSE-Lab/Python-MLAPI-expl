#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from scipy.stats import uniform, randint
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split


import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 160)


# In[ ]:


listings = pd.read_csv('../input/listings.csv')
listings.head(2)


# In[ ]:


# listings_data = listings[listings_useful_cols]
listings_data = listings.copy(deep=True)


# # Data Cleaning

# ## Dates

# In[ ]:


# typecastign all dates
listings_data['host_since'] = pd.to_datetime(listings_data['host_since'])
listings_data['first_review'] = pd.to_datetime(listings_data['first_review'])
listings_data['last_review'] = pd.to_datetime(listings_data['last_review'])


# ## Percentages

# In[ ]:


#convert percentages to numeric from range 0 to 1

listings_data['host_response_rate'] = pd.to_numeric(listings_data['host_response_rate'].apply(lambda x: str(x)
                                                                                              .replace('%', '')
                                                                                              .replace('N/A', '')),
                                                    errors='coerce')/100

listings_data['host_acceptance_rate'] = pd.to_numeric(listings_data['host_acceptance_rate'].apply(lambda x: str(x)
                                                                                              .replace('%', '')
                                                                                              .replace('N/A', '')),
                                                    errors='coerce')/100


# ## Currency

# In[ ]:


# convert currency to numeric

listings_data['price'] = pd.to_numeric(listings_data['price'].apply(lambda x: str(x)
                                                                    .replace('$', '')
                                                                    .replace(',', '')),
                                                    errors='coerce')


listings_data['weekly_price'] = pd.to_numeric(listings_data['weekly_price'].apply(lambda x: str(x)
                                                                    .replace('$', '')
                                                                    .replace(',', '')),
                                                    errors='coerce')


listings_data['monthly_price'] = pd.to_numeric(listings_data['monthly_price'].apply(lambda x: str(x)
                                                                    .replace('$', '')
                                                                    .replace(',', '')),
                                                    errors='coerce')

listings_data['security_deposit'] = pd.to_numeric(listings_data['security_deposit'].apply(lambda x: str(x)
                                                                    .replace('$', '')
                                                                    .replace(',', '')),
                                                    errors='coerce')

listings_data['cleaning_fee'] = pd.to_numeric(listings_data['cleaning_fee'].apply(lambda x: str(x)
                                                                    .replace('$', '')
                                                                    .replace(',', '')),
                                                    errors='coerce')

listings_data['extra_people'] = pd.to_numeric(listings_data['extra_people'].apply(lambda x: str(x)
                                                                    .replace('$', '')
                                                                    .replace(',', '')),
                                                    errors='coerce')


# ## Normalizing Review Scores

# In[ ]:


# normalize review score to fit to value between 0 to 1
listings_data['review_scores_rating'] = pd.to_numeric(listings_data['review_scores_rating'], errors='coerce')/100
listings_data['review_scores_accuracy'] = pd.to_numeric(listings_data['review_scores_accuracy'], errors='coerce')/10
listings_data['review_scores_cleanliness'] = pd.to_numeric(listings_data['review_scores_cleanliness'], errors='coerce')/10
listings_data['review_scores_checkin'] = pd.to_numeric(listings_data['review_scores_checkin'], errors='coerce')/10
listings_data['review_scores_communication'] = pd.to_numeric(listings_data['review_scores_communication'], errors='coerce')/10
listings_data['review_scores_location'] = pd.to_numeric(listings_data['review_scores_location'], errors='coerce')/10
listings_data['review_scores_value'] = pd.to_numeric(listings_data['review_scores_value'], errors='coerce')/10


# ## Boolean values Encoding

# In[ ]:


# convert true/false to codes
cleanup_colval = {
    'host_is_superhost' : {'f':0, 't':1},
    'host_identity_verified': {'f':0, 't':1},
    'is_location_exact': {'f':0, 't':1},
    'instant_bookable': {'f':0, 't':1},
    'require_guest_profile_picture': {'f':0, 't':1},
    'require_guest_phone_verification': {'f':0, 't':1},
    'host_has_profile_pic': {'f':0, 't':1}
}

listings_data.replace(cleanup_colval, inplace=True)


# ## Drop some NULL values

# In[ ]:


# host name, is_superhost are non-existent for only 2 rows, so we can remove them 

listings_data = listings_data.dropna(subset=['host_is_superhost'])

# property_type has one missing value, so drop it

listings_data = listings_data.dropna(subset=['property_type'])


# ## Typecasting to numeric

# In[ ]:


listings_data['host_is_superhost'] = listings_data['host_is_superhost'].astype(int).astype('category')
listings_data['host_identity_verified'] = listings_data['host_identity_verified'].astype(int).astype('category')
listings_data['is_location_exact'] = listings_data['is_location_exact'].astype(int).astype('category')
listings_data['instant_bookable'] = listings_data['instant_bookable'].astype(int)
listings_data['require_guest_profile_picture'] = listings_data['require_guest_profile_picture'].astype(int).astype('category')
listings_data['require_guest_phone_verification'] = listings_data['require_guest_phone_verification'].astype(int).astype('category')
listings_data['host_has_profile_pic'] = listings_data['host_has_profile_pic'].astype(int).astype('category')


# ## Encoding a few string features

# In[ ]:


def changeTime(x):
    '''
    change host_response_time columns from string into numerical.
    '''
    if x == 'within an hour':
        x='1'
    elif x == 'within a few hours':
        x='4'
    elif x == 'within a day':
        x='24'
    elif x == 'a few days or more':
        x='48'
    elif x == np.nan:
        np.nan
    else:
        x='96'
        
    return x


def changeStr(x):
    '''
    change back the host_response_time from the numerical into strings
    '''
    if x == 1:
        x='within an hour'
    elif x == 4:
        x='within a few hours'
    elif x == 24:
        x='within a day'
    elif x == 48:
        x= 'a few days or more'
    elif x == 96:
        x= 'No Response'
        
    return x


# In[ ]:


listings_data['host_response_time'] = listings_data['host_response_time'].apply(changeTime).astype(int)


# ## Deriving variables from amenities

# In[ ]:


listings_data['amenities'] = listings_data['amenities'].apply(lambda x: x.replace('{','')
                                                              .replace('}','')
                                                              .replace('"','')
                                                              .replace(' ','_')
                                                              .replace(',',' ')
                                                              .split()
                                                             )


# In[ ]:


# find unique ameneties
amenities_list = []
for val in listings_data['amenities']:
    for item in val:
        item_to_append = 'has_'+item
        if item_to_append not in amenities_list:
            amenities_list.append(item_to_append)
            


# In[ ]:


# add new columns for each item in amenities list
listings_data = listings_data.assign(**dict.fromkeys(amenities_list, np.nan))


# In[ ]:


# Fill in cell value with 1 if the property has a particular amenity, else 0

for index, row in listings_data.iterrows():
    for amenity in amenities_list:
        if amenity[4:] in row['amenities']:
            listings_data.loc[index, amenity] = 1
        else:
            listings_data.loc[index, amenity] = 0


# ## Deriving individual values from host_verifications

# In[ ]:


listings_data['host_verifications'] = listings_data['host_verifications'].apply(lambda x: x.replace('[','')
                                                              .replace(']','')
                                                              .replace("'",'')
                                                              .replace(',','')
                                                              .split()
                                                             )


# In[ ]:


verification_list = []
for val in listings_data['host_verifications']:
    for item in val:
        item_to_append = 'verified_'+item
        if item_to_append not in verification_list:
            verification_list.append(item_to_append)
            
print(verification_list)


# In[ ]:


listings_data = listings_data.assign(**dict.fromkeys(verification_list, np.nan))


# In[ ]:


# Fill in cell value with 1 if the host verification was done, else 0

for index, row in listings_data.iterrows():
    for _ in verification_list:
        if _[9:] in row['host_verifications']:
            listings_data.loc[index, _] = 1
        else:
            listings_data.loc[index, _] = 0


# ## Impute missing values for some features

# In[ ]:


listings_data['security_deposit'] = listings_data['security_deposit'].fillna(0)
listings_data['cleaning_fee'] = listings_data['cleaning_fee'].fillna(0)


# Number of beds is missing for one row. Number of people it can accomodate is not missing for any row.
# Impute it using the `avg number of beds per person * number of persons` this listing can accomodate
# 
# Similarly, impute number of `bathrooms` and `bedrooms`

# In[ ]:


# imputing missing values in beds column
avg_beds_per_person = (listings_data['beds'] / listings_data['accommodates']).mean()
print(avg_beds_per_person)
fillna_val = listings_data['accommodates']*avg_beds_per_person
listings_data['beds'] = listings_data['beds'].fillna(fillna_val).astype(int)
print(listings_data['beds'].isnull().sum())


# In[ ]:


# imputing missing values in bathrooms column
avg_bathrooms_per_person = (listings_data['bathrooms'] / listings_data['accommodates']).mean()
# print(avg_bathrooms_per_person)
fillna_val = listings_data['accommodates']*avg_bathrooms_per_person
listings_data['bathrooms'] = listings_data['bathrooms'].fillna(fillna_val).astype(int)
print(listings_data['bathrooms'].isnull().sum())


# In[ ]:


# imputing missing values in bedrooms column
avg_bedrooms_per_person = (listings_data['bedrooms'] / listings_data['accommodates']).mean()
# print(avg_bedrooms_per_person)
fillna_val = listings_data['accommodates']*avg_bedrooms_per_person
listings_data['bedrooms'] = listings_data['bedrooms'].fillna(fillna_val).astype(int)
print(listings_data['bedrooms'].isnull().sum())


# In[ ]:


listings_data.head(2)


# In[ ]:


# listings_data.to_csv('./listings_cleaned_transformed.csv')


# # Given the attributes of a new listing and it's location, predict the price. The prediction can be used to give a suggestion for price to be set for new listings

# By heuristics, the price of the listing would depend on the the attributes of the house only.
# 
# Features to consider:
# * Amenities offered
# * Features of the house like number of beds, number of people it can accomodate etc
# * Location details (neighbourhood)

# ## Select only relevant features for predicting the price

# In[ ]:


relevant_features = ['neighbourhood_cleansed', 'property_type', 'room_type', 'accommodates', 'bathrooms', 
                    'bedrooms', 'beds', 'bed_type', 'square_feet', 'price', 'guests_included',
                    'instant_bookable', 'cancellation_policy']


# In[ ]:


# add amenities to the list of features
relevant_features.extend(amenities_list)
print(relevant_features)


# In[ ]:


price_predictions_df = listings_data[relevant_features]
price_predictions_df.head(2)


# ## Check for null values

# In[ ]:


for column in price_predictions_df.columns:
    print("% of null values for column ", column, " = ", 100*price_predictions_df[column].isnull().sum()/price_predictions_df.shape[0], '%')


# NULL values in square_feet is 97.5% so drop that column

# ### What is the price distribution ?

# In[ ]:


# histogram of price of listing
plt.figure(figsize=(15,5))
plt.hist(price_predictions_df['price'], bins=20)
plt.xticks(np.arange(0, 1700, step=100))
plt.ylabel('Number of listings')
plt.xlabel('Price, $')
plt.title('Number of listings depending on price')


# plt.savefig('Price distrubution.png')

plt.show()


# As evident, most of the listings have a price tag between $50-400

# In[ ]:


plt.figure(figsize=(15,5))
ax = sns.countplot(x="neighbourhood_cleansed", 
                   data=price_predictions_df,
                  order = price_predictions_df['neighbourhood_cleansed'].value_counts().index)


# too many values to show in neighbourhood

# ### Top Neighbourhoods by count of listings

# In[ ]:


price_predictions_df['neighbourhood_cleansed'].value_counts()[0:15]


# ### What are the types of properties and their count ?

# In[ ]:


plt.figure(figsize=(15,5))
plt.xticks(rotation=30)
ax = sns.countplot(x="property_type", 
                   data=price_predictions_df,
                  order = price_predictions_df['property_type'].value_counts().index)


# * Most common properties are entire house and apartment

# ### What are the types of rooms and their count ?

# In[ ]:


plt.figure(figsize=(15,5))
plt.xticks(rotation=30)
ax = sns.countplot(x="room_type", 
                   data=price_predictions_df,
                  order = price_predictions_df['room_type'].value_counts().index)


# ### How many people in total do the listings accomodate?

# In[ ]:


plt.figure(figsize=(18,5))

plt.xticks(rotation=30)
ax = sns.barplot(x="property_type", y="accommodates", hue="room_type", 
                 data=pd.DataFrame({'accommodates' : price_predictions_df.groupby(['property_type', 'room_type'])['accommodates'].agg('sum')}).reset_index())


# ### How many bathrooms does each property type have?

# In[ ]:


plt.figure(figsize=(18,5))
plt.xticks(rotation=30)
ax = sns.barplot(x="property_type", y="bathrooms", hue="room_type", 
                 data=price_predictions_df)


# ### How does the price compare against the neighbourhood?

# In[ ]:


ax = sns.boxplot(x="room_type", y="price", data=price_predictions_df)


# * The price ranges of an entire apartment have a very high variance

# In[ ]:


price_predictions_df = price_predictions_df.drop(['square_feet'], axis = 1)


# In[ ]:


corr = price_predictions_df[['accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'guests_included']].corr()
# sns.heatmap(uniform_data)

corr.style.background_gradient(cmap='coolwarm')


# High correlation between:
# * number of beds is and number of people a house can accomodate
# * number of bedrroms and number of people a house can accomodate
# 

# ## One-hot encoding of categorical variables

# In[ ]:


price_predictions_df.dtypes


# In[ ]:


price_predictions_df = pd.get_dummies(price_predictions_df, columns = ["neighbourhood_cleansed"], prefix="ngrhd")
price_predictions_df = pd.get_dummies(price_predictions_df, columns = ["property_type"], prefix="property_type")
price_predictions_df = pd.get_dummies(price_predictions_df, columns = ["room_type"], prefix="room_type")
price_predictions_df = pd.get_dummies(price_predictions_df, columns = ["bed_type"], prefix="bed_type")
price_predictions_df = pd.get_dummies(price_predictions_df, columns = ["cancellation_policy"], prefix="cancellation_policy")


# In[ ]:


price_predictions_df.head(2)


# ## Train-test split
# 
# The objective here is to predict the price of a new home and show as a suggested price to a new listing, hence it is a regression problem and the target variable is the price

# In[ ]:


trainX = price_predictions_df.drop(['price'] , axis = 1)
trainy = price_predictions_df['price']
X_train, X_test, y_train, y_test = train_test_split(trainX, trainy, test_size = 0.2, random_state = 42)


# ## XGboost regression with hyperparameter searching

# In[ ]:


xgb_model = xgb.XGBRegressor()

params = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1 
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(100, 150), # default 100
    "subsample": uniform(0.6, 0.4)
}


# In[ ]:


search = RandomizedSearchCV(xgb_model, param_distributions=params, 
                            random_state=42, n_iter=200, 
                            cv=5, verbose=1, n_jobs=-1, return_train_score=True)

search.fit(X_train, y_train, early_stopping_rounds = 10, eval_set=[(X_test, y_test)], eval_metric = 'rmse')


# In[ ]:


def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# In[ ]:


# Top score from all the combinations of parameters produced by randomizedcv
print('Scoring on RMSE')
report_best_scores(search.cv_results_, 1)


# In[ ]:


best_xgbregressor = search.best_estimator_


# In[ ]:


best_xgbregressor


# ### Predicted vs Actual plot 

# In[ ]:


plt.figure(figsize=(16, 6))
ax = sns.regplot(x=y_test, y=best_xgbregressor.predict(X_test), marker="+")
ax.set(xlabel='Actual Price', ylabel='Predicted Price')


# ### XGBoost - Mean square error on Holdout set

# In[ ]:



print(np.sqrt(mean_squared_error(y_test, best_xgbregressor.predict(X_test))))


# ## Top 15 features that influence the price

# In[ ]:


# get feature importances from the model and show top 30
headers = ["name", "score"]
values = sorted(zip(X_train.columns, best_xgbregressor.feature_importances_), key=lambda x: x[1] * -1)
xgb_feature_importances = pd.DataFrame(values, columns = headers)

#plot feature importances for top 15 features
features = xgb_feature_importances['name'][:15]
y_pos = np.arange(len(features))
scores = xgb_feature_importances['score'][:15]
 
plt.figure(figsize=(16,5))
plt.bar(y_pos, scores, align='center', alpha=0.5)
plt.xticks(y_pos, features, rotation='vertical')
plt.ylabel('Score')
plt.xlabel('Features')
plt.title('Feature importances (XGBoost)')

plt.savefig('feature importances XGB.png')
 
plt.show()


# Conclusions from feature importance:
# * 6 out of top 15 important features are amenities, therefore amenities have a high impact on the price of listing
# * Cancellation policy and instant booking feature is not important for deciding the pricing

# ## Neural Networks Regression with Cross Validation

# In[ ]:


kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores = []

for train_index, test_index in kfold.split(X_train.values):   
    X_tr, X_te = X_train.values[train_index], X_train.values[test_index]
    y_tr, y_te = y_train.values[train_index], y_train.values[test_index]
    mlp = MLPRegressor(hidden_layer_sizes=(13,13,13),max_iter=500, random_state =42, activation ='relu')
    mlp.fit(X_tr, y_tr)
    
    y_cvpred = mlp.predict(X_te)
    
    scores.append(mean_squared_error(y_te, y_cvpred))
    


# In[ ]:


print(np.sqrt(scores))


# In[ ]:


plt.figure(figsize=(16, 6))
ax = sns.regplot(x=y_test, y=mlp.predict(X_test), marker="+")
ax.set(xlabel='Actual Price', ylabel='Predicted Price')


# ### Neural Network - Mean square error on Holdout set

# In[ ]:


print(np.sqrt(mean_squared_error(y_test, mlp.predict(X_test))))


# # **XGBoost has given a better performance, so choose XGBoost**
# 
# 

# In[ ]:




