#!/usr/bin/env python
# coding: utf-8

# <center><h2>Taxi fare predictions with deep learning and Tensorflow</h2></center>
# 
# ### In this version of the code i use pandas to load the data and the Tensorflow input pandas function to feed the model.
# 
# #### Notes: 
# * [Link for a Keras version](https://www.kaggle.com/dimitreoliveira/taxi-fare-prediction-with-keras-deep-learning)
# * [Link for a more complete version on Github](https://github.com/dimitreOliveira/NewYorkCityTaxiFare)
# * I'm not using "passenger count" because it something that is not supposed to really matter in this case.
# * I've created two features derived from "hour" (night and late night), according to some research i did it's added an additional value if it's a business day (mon ~ fri) and it's night, also there's another added value if it's dawn (late night).
# * I'm binning latitudes and longitudes to make it easier to work with.
# * Even tough deep learning is robust enough to deal with noisy data, i'm removing outliers (it may save some memory).
# * Currently i'm using both Euclidean and Manhattan distances, it may be a bit redundant, but they have a different meaning and i'm still not sure of witch one is better(if you have some insights about this please let me know)

# ## Dependencies

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# logging in INFO mode let us see the training feedback.
tf.logging.set_verbosity(tf.logging.INFO)


# ## Data clean
# ### Here i'm removing some outliers, and noisy data.
# * Lats and lons that do not belong to New York.
# * Negative fare.
# * Fare greater than 250 (this seems to be noisy data).
# * Rides that begin and end in the same location.

# In[ ]:


def clean(df):
    # Delimiter lats and lons to NY only
    df = df[(-76 <= df['pickup_longitude']) & (df['pickup_longitude'] <= -72)]
    df = df[(-76 <= df['dropoff_longitude']) & (df['dropoff_longitude'] <= -72)]
    df = df[(38 <= df['pickup_latitude']) & (df['pickup_latitude'] <= 42)]
    df = df[(38 <= df['dropoff_latitude']) & (df['dropoff_latitude'] <= 42)]
    # Remove possible outliers
    df = df[(0 < df['fare_amount']) & (df['fare_amount'] <= 250)]
    # Remove inconsistent values
    df = df[(df['dropoff_longitude'] != df['pickup_longitude'])]
    df = df[(df['dropoff_latitude'] != df['pickup_latitude'])]
    
    return df


# ## Feature engineering
# *  Now i'll do some feature engineering and process the data, i'm basically creating 3 kinds of features.
#     *  **Time features**
#         * Year, Month, Day, Hour, Weekday
#         * Night (between 16h and 20h, from monday to friday)
#         * Late night (between 20h and and 6h)
#     * **Coordinate features**
#         * Latitude difference (difference from pickup and dropout latitudes)
#         * Longitude difference (difference from pickup and dropout longitudes)
#     * **Distances features**
#         * Euclidean (Euclidean distance from pickup and dropout)
#         * Manhattan (Manhattan distance from pickup and dropout)
#         * Manhattan distances from pickup location and downtown, JFK, EWR and LGR airports (see if the ride started at one of these locations).
#         * Manhattan distances from dropout location and downtown, JFK, EWR and LGR airports (see if the ride ended at one of these locations).

# In[ ]:


def late_night (row):
    if (row['hour'] <= 6) or (row['hour'] >= 20):
        return 1
    else:
        return 0


def night (row):
    if ((row['hour'] <= 20) and (row['hour'] >= 16)) and (row['weekday'] < 5):
        return 1
    else:
        return 0
    
    
def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    df['pickup_datetime'] =  pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S %Z')
    df['year'] = df['pickup_datetime'].apply(lambda x: x.year)
    df['month'] = df['pickup_datetime'].apply(lambda x: x.month)
    df['day'] = df['pickup_datetime'].apply(lambda x: x.day)
    df['hour'] = df['pickup_datetime'].apply(lambda x: x.hour)
    df['weekday'] = df['pickup_datetime'].apply(lambda x: x.weekday())
    df['pickup_datetime'] =  df['pickup_datetime'].apply(lambda x: str(x))
    df['night'] = df.apply (lambda x: night(x), axis=1)
    df['late_night'] = df.apply (lambda x: late_night(x), axis=1)
    # Drop 'pickup_datetime' as we won't need it anymore
    df = df.drop('pickup_datetime', axis=1)
    
    return df


def add_coordinate_features(df):
    lat1 = df['pickup_latitude']
    lat2 = df['dropoff_latitude']
    lon1 = df['pickup_longitude']
    lon2 = df['dropoff_longitude']
    
    # Add new features
    df['latdiff'] = (lat1 - lat2)
    df['londiff'] = (lon1 - lon2)

    return df


def add_distances_features(df):
    # Add distances from airpot and downtown
    ny = (-74.0063889, 40.7141667)
    jfk = (-73.7822222222, 40.6441666667)
    ewr = (-74.175, 40.69)
    lgr = (-73.87, 40.77)
    
    lat1 = df['pickup_latitude']
    lat2 = df['dropoff_latitude']
    lon1 = df['pickup_longitude']
    lon2 = df['dropoff_longitude']
    
    df['euclidean'] = (df['latdiff'] ** 2 + df['londiff'] ** 2) ** 0.5
    df['manhattan'] = manhattan(lat1, lon1, lat2, lon2)
    
    df['downtown_pickup_distance'] = manhattan(ny[1], ny[0], lat1, lon1)
    df['downtown_dropoff_distance'] = manhattan(ny[1], ny[0], lat2, lon2)
    df['jfk_pickup_distance'] = manhattan(jfk[1], jfk[0], lat1, lon1)
    df['jfk_dropoff_distance'] = manhattan(jfk[1], jfk[0], lat2, lon2)
    df['ewr_pickup_distance'] = manhattan(ewr[1], ewr[0], lat1, lon1)
    df['ewr_dropoff_distance'] = manhattan(ewr[1], ewr[0], lat2, lon2)
    df['lgr_pickup_distance'] = manhattan(lgr[1], lgr[0], lat1, lon1)
    df['lgr_dropoff_distance'] = manhattan(lgr[1], lgr[0], lat2, lon2)
    
    return df


# ## Estimator
# #### Here is where most of the important Tensorflow stuff happens. 
# * I get a list of input columns, theses columns comes from the csv data, so they are supposed to match each other in size.
# * First we need a lists with the buckets for the latitudes and longitudes values, we give the limits and number of buckets we want.
# * Then i'll bucktize (create bins) for the values (the number of buckets is nbuckets, which is a hyperparameter).
# * Tensorflow allows us to cross categorical data, we will do it to create ploc (pickup location) and dloc (dropout location), in practice this will get categorical features and make combinations with another categorical feature, creating all possible combinations with the columns (this probably will create a very sparse column).
# We will create sparse features for "pickup location" (pickup lat with lon), "dropout location" (dropout lat with lon), "pickup and dropout location" (pickup location with dropout location) and "day_hour ride" (weekday with hour), these features are supposed to give the model a better understanding of the data by us giving our insights.
# * The wide columns list will go to the linear model, these are features that may have a linear relation, or are a sparse feature (Tensorflow DNN models do note accept sparse data).
# * The deep columns list will go to the DNN model
# 
# #### On embeddings
# * Here is very important the embedding features, as "pd_pair" and "day_hr" are sparse data, we need a way reduce it's dimension, and we do it by using a embedding, embeddings also helps mapping features to a more "compact form", as we try to find a way to represent a number of features in a lower dimension, e.g. in the "day_hr" feature i have a cross of one-hot encoded features, weekdays (7 features [one for each day]) and hour (24 features [1 for each hour]), so crossing these two would give me an additional 168 features (24 x 7), this would be a lot of input data, it would have a chance to disrupt the model learning, but by using embeddings i can lower it's dimension (168) to a smaller number, in this case i use 10, this way during training my model will also learn a way to map this 168 features to only 10 features.
# * As you can see embeddings are a very powerful tool that Tensorflow give to us, on a very easy to use API.
# * This image may help the understanding: <img src="https://www.tensorflow.org/images/feature_columns/embedding_vs_indicator.jpg" width="450">
# * In this model i use both linear and DNN models, to try to take advantage on both based on the features i have, luckily Tensorflow has that already implemented on it's high level API, so i just have to call tf.estimator.DNNLinearCombinedRegressor passing all the parameters i need.

# In[ ]:


def build_estimator(nbuckets, hidden_units, optimizer, input_columns, run_config=None):
    # Input columns
    (plon, plat, dlon, dlat, year, month, day, hour, weekday, night, late_night, 
     latdiff, londiff, euclidean, manhattan, downtown_pickup_distance, downtown_dropoff_distance, 
     jfk_pickup_distance, jfk_dropoff_distance, ewr_pickup_distance, ewr_dropoff_distance, 
     lgr_pickup_distance, lgr_dropoff_distance) = input_columns

    # Bucketize the lats & lons
    latbuckets = np.linspace(38.0, 42.0, nbuckets).tolist()
    lonbuckets = np.linspace(-76.0, -72.0, nbuckets).tolist()
    b_plat = tf.feature_column.bucketized_column(plat, latbuckets)
    b_dlat = tf.feature_column.bucketized_column(dlat, latbuckets)
    b_plon = tf.feature_column.bucketized_column(plon, lonbuckets)
    b_dlon = tf.feature_column.bucketized_column(dlon, lonbuckets)

    # Feature cross
    ploc = tf.feature_column.crossed_column([b_plat, b_plon], nbuckets ** 2)
    dloc = tf.feature_column.crossed_column([b_dlat, b_dlon], nbuckets ** 2)
    pd_pair = tf.feature_column.crossed_column([ploc, dloc], nbuckets ** 4)
    day_hr = tf.feature_column.crossed_column([weekday, hour], 24 * 7)

    # Wide columns and deep columns
    wide_columns = [
        # Sparse columns
        night, late_night,

        # Anything with a linear relationship
        month, hour, weekday, year
    ]

    deep_columns = [
        # Embedding columns to "group" together
        tf.feature_column.embedding_column(pd_pair, nbuckets),
        tf.feature_column.embedding_column(day_hr, nbuckets),
        tf.feature_column.embedding_column(ploc, nbuckets),
        tf.feature_column.embedding_column(dloc, nbuckets),

        # Numeric columns
        latdiff, londiff,
        euclidean, manhattan,
        downtown_pickup_distance, downtown_dropoff_distance,
        jfk_pickup_distance, jfk_dropoff_distance,
        ewr_pickup_distance, ewr_dropoff_distance,
        lgr_pickup_distance, lgr_dropoff_distance
    ]

    estimator = tf.estimator.DNNLinearCombinedRegressor(
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        dnn_optimizer=optimizer,
        config=run_config)

    return estimator


# ### Tensorflow pandas data input functions
# #### These functions are responsible for feeding data both for model training and prediction.
# * Note: both functions feed data by batch using generators.
# * Shuffle in the test function must be 'False' or else it will shuffle the predicted data.

# In[ ]:


def pandas_train_input_fn(df, label):
    return tf.estimator.inputs.pandas_input_fn(
        x=df,
        y=label,
        batch_size=128,
        num_epochs=100,
        shuffle=True,
        queue_capacity=1000
    )


def pandas_test_input_fn(df):
    return tf.estimator.inputs.pandas_input_fn(
        x=df,
        y=None,
        batch_size=128,
        num_epochs=1,
        shuffle=False,
        queue_capacity=1000
    )


# ### Output function

# In[ ]:


def output_submission(df, prediction_df, id_column, prediction_column, file_name):
    df[prediction_column] = prediction_df['predictions'].apply(lambda x: x[0])
    df[[id_column, prediction_column]].to_csv((file_name), index=False)
    print('Output complete')


# ### Parameters
# * One thing this API of Tensorflow have different from other ml models is that it does not have an epochs number, this API uses number of steps, this is because Tensorflow can be also used for distributed training, so it's easier to distribute training if clusters don't need do synchronize epochs with each other, with steps they simply train x steps and send the results back to the main core.
#     * Each step feeds the model with data equal to the batch size.

# In[ ]:


TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'
SUBMISSION_NAME = 'submission.csv'

# Model parameters
BATCH_SIZE = 512
STEPS = 400000
LEARNING_RATE = 0.001
DATASET_SIZE = 8000000
HIDDEN_UNITS = [256, 128, 64, 32]


# #### Inputting columns to the Tensorflow model
# * This list is basically the columns read from the csv file converted to a Tensorflow feature column type, in this case i use only 2, but there are [many more.](https://www.tensorflow.org/guide/feature_columns)
#     * tf.feature_column.numeric_column, just a normal numeric feature.
#     * tf.feature_column.categorical_column_with_identity, this one is a categorical feature with a number of buckets (it's essentially a one-hot encoded column, but the number of buckets must be >= than the number of possible values, e.g. for week day i need at least 7 buckets).

# In[ ]:


INPUT_COLUMNS = [
    # raw data columns
    tf.feature_column.numeric_column('pickup_longitude'),
    tf.feature_column.numeric_column('pickup_latitude'),
    tf.feature_column.numeric_column('dropoff_longitude'),
    tf.feature_column.numeric_column('dropoff_latitude'),

    # engineered columns
    tf.feature_column.numeric_column('year'),
    tf.feature_column.categorical_column_with_identity('month', num_buckets=13),
    tf.feature_column.categorical_column_with_identity('day', num_buckets=32),
    tf.feature_column.categorical_column_with_identity('hour', num_buckets=24),
    tf.feature_column.categorical_column_with_identity('weekday', num_buckets=7),
    tf.feature_column.categorical_column_with_identity('night', num_buckets=2),
    tf.feature_column.categorical_column_with_identity('late_night', num_buckets=2),
    tf.feature_column.numeric_column('latdiff'),
    tf.feature_column.numeric_column('londiff'),
    tf.feature_column.numeric_column('euclidean'),
    tf.feature_column.numeric_column('manhattan'),
    tf.feature_column.numeric_column('downtown_pickup_distance'),
    tf.feature_column.numeric_column('downtown_dropoff_distance'),
    tf.feature_column.numeric_column('jfk_pickup_distance'),
    tf.feature_column.numeric_column('jfk_dropoff_distance'),
    tf.feature_column.numeric_column('ewr_pickup_distance'),
    tf.feature_column.numeric_column('ewr_dropoff_distance'),
    tf.feature_column.numeric_column('lgr_pickup_distance'),
    tf.feature_column.numeric_column('lgr_dropoff_distance')
]


# ### Load data

# In[ ]:


# Load data in a more compact form
datatypes = {'key': 'str', 
              'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

# Only a fraction of the data
train = pd.read_csv(TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes, usecols=[1,2,3,4,5,6])
test = pd.read_csv(TEST_PATH)


# #### Clean and process data

# In[ ]:


train = clean(train)

train = add_time_features(train)
test = add_time_features(test)

add_coordinate_features(train)
add_coordinate_features(test)

train = add_distances_features(train)
test = add_distances_features(test)

train.head(5)


# #### Split data in train and validation (90% ~ 10%)

# In[ ]:


train_df, validation_df = train_test_split(train, test_size=0.1, random_state=1)


# In[ ]:


# Scale data
# Note: i'm doing this here with sklearn scaler but on the Coursera code the scaling is done with Dataflow and Tensorflow
# Selecting only columns that will be scaled
wanted_columns = ['pickup_longitude', 'pickup_latitude','dropoff_longitude', 
                  'dropoff_latitude','year', 'latdiff', 'londiff', 
                  'euclidean', 'manhattan', 'downtown_pickup_distance', 
                  'downtown_dropoff_distance', 'jfk_pickup_distance', 'jfk_dropoff_distance', 
                  'ewr_pickup_distance', 'ewr_dropoff_distance', 'lgr_pickup_distance', 
                  'lgr_dropoff_distance']

# One-hot encodded features (e.g. weekday) won't be scaled, this is arguable, but in my opinion when you scale one-hot encoded features they may lose it's purpose (true or false).
one_hot_columns = ['month', 'day', 'hour', 'weekday', 'night', 'late_night']

train_df_scaled = train_df[wanted_columns]
validation_df_scaled = validation_df[wanted_columns]
test_scaled = test[wanted_columns]

# Normalize using Min-Max scaling
# Just a quick note: i use the same object to fit and transform all the data sets, because data should be normalized using a single data set(distribution) as parameter.
scaler = preprocessing.MinMaxScaler()
train_df_scaled[wanted_columns] = scaler.fit_transform(train_df_scaled[wanted_columns])
validation_df_scaled[wanted_columns] = scaler.transform(validation_df_scaled[wanted_columns])
test_scaled[wanted_columns] = scaler.transform(test_scaled[wanted_columns])

# Add one-hot encoded features
train_df_scaled[one_hot_columns] = train_df[one_hot_columns]
validation_df_scaled[one_hot_columns] = validation_df[one_hot_columns]
test_scaled[one_hot_columns] = test[one_hot_columns]

train_df_scaled.head(5)


# ### Define model and parameters
# * Currently i'm using Adam optimizer, it seems to be the better default optimizer for DNN.
# * Tensorflow estimator API accepts train and evaluation spec (Specifications), that are classes with some of the information needed for the model training and evaluation, like the input function, labels and many more.

# In[ ]:


# optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.1, l1_regularization_strength=0.001, l2_regularization_strength=0.001)
estimator = build_estimator(16, HIDDEN_UNITS, optimizer, INPUT_COLUMNS)

train_spec = tf.estimator.TrainSpec(input_fn=pandas_train_input_fn(train_df_scaled, train_df['fare_amount']), max_steps=STEPS)
eval_spec = tf.estimator.EvalSpec(input_fn=pandas_train_input_fn(validation_df_scaled, validation_df['fare_amount']), steps=500, throttle_secs=300)


# ### Model parameters

# In[ ]:


print('Dataset size: %s' % DATASET_SIZE)
print('Steps: %s' % STEPS)
print('Learning rate: %s' % LEARNING_RATE)
print('Batch size: %s' % BATCH_SIZE)
print('Input dimension: %s' % train_df_scaled.shape[1])
print('Features used: %s' % train_df.columns)


# ### Train model
# * I'm training using the 'train_and_evaluate' function that allows me to train and evaluate the model at the same time.

# In[ ]:


tf.estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)


# In[ ]:


# Make prediction
prediction = estimator.predict(pandas_test_input_fn(test_scaled))


# In[ ]:


# output prediction
prediction_df = pd.DataFrame(prediction)
output_submission(test, prediction_df, 'key', 'fare_amount', SUBMISSION_NAME)

