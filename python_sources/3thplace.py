import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend
import numpy as np

train_raw=pd.read_pickle("train.pkl")

def clean(df):
    df = df[(-76 <= df['pickup_longitude']) & (df['pickup_longitude'] <= -72)]
    df = df[(-76 <= df['dropoff_longitude']) & (df['dropoff_longitude'] <= -72)]
    df = df[(38 <= df['pickup_latitude']) & (df['pickup_latitude'] <= 42)]
    df = df[(38 <= df['dropoff_latitude']) & (df['dropoff_latitude'] <= 42)]
    df = df[(0 < df['fare_amount']) & (df['fare_amount'] <= 250)]
    df = df[(df['dropoff_longitude'] != df['pickup_longitude'])]
    df = df[(df['dropoff_latitude'] != df['pickup_latitude'])]
    return df

train = clean(train_raw)
train = train.reset_index()

def late_night (df):
    return np.where(((df['hour'] <= 6) | (df['hour'] >= 20)), 1, 0)

def eve(df):
    return np.where(((df['hour'] <= 20) | (df['hour'] >= 16)), 1, 0)

def preprocess_time(train):
    train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
    train['month'] = train['pickup_datetime'].dt.month
    train['year'] = train['pickup_datetime'].dt.year
    train['hour'] = train['pickup_datetime'].dt.hour
    train['weekday'] = train['pickup_datetime'].dt.weekday
    train['night'] = eve(train)
    train['late_night'] = late_night(train)
    return train

def add_coordinate_features(df):
    lat1 = df['pickup_latitude']
    lat2 = df['dropoff_latitude']
    lon1 = df['pickup_longitude']
    lon2 = df['dropoff_longitude']
    df['latdiff'] = abs(lat1 - lat2)
    df['londiff'] = abs(lon1 - lon2)
    return df

def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)

def haversine(lat1,lon1,lat2,lon2):
    R = 6371.0088
    lat1,lon1,lat2,lon2 = map(np.radians, [lat1,lon1,lat2,lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) **2
    c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
    d = R * c
    return round(d,4)


def add_distances_features(df):
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
    df['haverstine'] = haversine(lat1, lon1, lat2, lon2)

    df['downtown_pickup_distance'] = haversine(ny[1], ny[0], lat1, lon1)
    df['downtown_dropoff_distance'] = haversine(ny[1], ny[0], lat2, lon2)
    df['jfk_pickup_distance'] = haversine(jfk[1], jfk[0], lat1, lon1)
    df['jfk_dropoff_distance'] = haversine(jfk[1], jfk[0], lat2, lon2)
    df['ewr_pickup_distance'] = haversine(ewr[1], ewr[0], lat1, lon1)
    df['ewr_dropoff_distance'] = haversine(ewr[1], ewr[0], lat2, lon2)
    df['lgr_pickup_distance'] = haversine(lgr[1], lgr[0], lat1, lon1)
    df['lgr_dropoff_distance'] = haversine(lgr[1], lgr[0], lat2, lon2)
    df['pickup_longitude_binned'] = pd.qcut(df['pickup_longitude'], 16, labels=False)
    df['dropoff_longitude_binned'] = pd.qcut(df['dropoff_longitude'], 16, labels=False)
    df['pickup_latitude_binned'] = pd.qcut(df['pickup_latitude'], 16, labels=False)
    df['dropoff_latitude_binned'] = pd.qcut(df['dropoff_latitude'], 16, labels=False)
    return df

def preprocess(train):
    train = add_coordinate_features(train)
    train = add_distances_features(train)
    train = preprocess_time(train)
    train = train.drop(['key','pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1)
    train = pd.get_dummies(train, columns=['month','weekday'])
    return train

train = preprocess(train)

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras import regularizers
train_df, validation_df = train_test_split(train, test_size=0.01, random_state=1)

train_labels = train_df['fare_amount'].values
validation_labels = validation_df['fare_amount'].values
train_df = train_df.drop(['fare_amount'], axis=1)
validation_df = validation_df.drop(['fare_amount'], axis=1)
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer = 'uniform'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_initializer = 'uniform'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation=tf.nn.relu, kernel_initializer = 'uniform'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32, activation=tf.nn.relu, kernel_initializer = 'uniform'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(16, activation=tf.nn.relu, kernel_initializer = 'uniform'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation=None)
])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.65,
                              patience=2, min_lr=1e-8)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_model3.h5', monitor='val_loss', save_best_only=True)

opt = tf.keras.optimizers.Adam(lr=0.001)

model.compile(optimizer=opt,
              loss='mse',
              metrics=['mse'])

metrics=model.fit(x=train_df_scaled, y=train_labels,validation_data=(validation_df_scaled, validation_labels), epochs=50,batch_size=256,
callbacks=[reduce_lr,checkpoint],shuffle=True)

test_set =pd.read_csv("test.csv")
test = preprocess(test_set).values
test_scaled = scaler.transform(test)

y_test_predictions=model.predict(test_scaled)
submission = pd.DataFrame(
    {'key': test_set.key.values, 'fare_amount': y_test_predictions.squeeze()},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)

print(metrics.history)

#I ATTACHED LOGS OF TRAINING IN INPUT DATA. TRAINING LOOKS VERY PROMISING MSE 10.3 ON TRAINING 11.01 ON VALIDATION

