import pandas as pd
import numpy as np
import sklearn.metrics as sklm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from math import radians, cos, sin, asin, sqrt


########
# Data #
########
df_train = pd.read_csv('mvml/train_MV.csv')

print(df_train.describe())
print(df_train.columns)
print(df_train.dtypes)

# number of passengers
plt.scatter(df_train['passenger_count'], df_train['fare_amount'])
plt.show()

####################
# Data Preparation #
####################

df_train = df_train.dropna(how = 'any', axis = 'rows')

df_train = df_train[df_train['fare_amount'].between(left = 2.5, right = 200)]
df_train = df_train[df_train['passenger_count'].between(left = 1, right = 10)]


# remove outliers
df_train = df_train.drop(df_train[df_train['dropoff_longitude'] > 0].index)
df_train = df_train.drop(df_train[df_train['pickup_longitude'] > 0].index)
df_train = df_train.drop(df_train[df_train['dropoff_latitude'] < 0].index)
df_train = df_train.drop(df_train[df_train['pickup_latitude'] < 0].index)

aux = df_train['dropoff_longitude'].describe()
df_train = df_train.drop(df_train[(df_train['dropoff_longitude'] < aux['mean'] - aux['std']/2) | (df_train['dropoff_longitude'] > aux['mean'] + aux['std']/2)].index)
aux = df_train['pickup_longitude'].describe()
df_train = df_train.drop(df_train[(df_train['pickup_longitude'] < aux['mean'] - aux['std']/2) | (df_train['pickup_longitude'] > aux['mean'] + aux['std']/2)].index)
aux = df_train['dropoff_latitude'].describe()
df_train = df_train.drop(df_train[(df_train['dropoff_latitude'] < aux['mean'] - aux['std']) | (df_train['dropoff_latitude'] > aux['mean'] + aux['std'])].index)
aux = df_train['pickup_latitude'].describe()
df_train = df_train.drop(df_train[(df_train['pickup_latitude'] < aux['mean'] - aux['std']) | (df_train['pickup_latitude'] > aux['mean'] + aux['std'])].index)


def distanceKM(cells):

    lon1, lat1, lon2, lat2 = map(radians, [cells['pickup_longitude'],cells['pickup_latitude'],cells['dropoff_longitude'],cells['dropoff_latitude']])

    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth's radius in km
    return np.round(c * r,3)


def roundDate(df):

    df["distance"] = round(df["distance"]*100,2)

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df['year'] = df["pickup_datetime"].dt.year
    df['month'] = df["pickup_datetime"].dt.month

    df['weekday'] = df["pickup_datetime"].dt.dayofweek
    df['weekend'] = df.apply(lambda row: 1 if row['weekday'] > 4 else 0 , axis=1)


df_train['distance'] = df_train.apply(lambda row: distanceKM(row), axis=1)

# remove trips less then 10 meters
df_train.drop( df_train[ df_train['distance'] < 0.01 ].index, inplace=True )
# remove trips costing more than 110$ and less than 10 km
df_train.drop( df_train[ ( df_train['distance'] < 10 ) & ( df_train['fare_amount'] > 110) ].index, inplace=True)
# remove trips greater than 50 km
df_train.drop(df_train[df_train['distance'] > 50].index,inplace=True)


#df_train['dollarperkm'] = df_train['fare_amount']/df_train['distance']


print((df_train['fare_amount']/df_train['distance']).describe())
plt.scatter(df_train['distance'], df_train['fare_amount']/df_train['distance'])
plt.show()


plt.scatter( df_train['distance'], df_train['fare_amount'], color='blue' )
plt.xlabel('distance [km]')
plt.ylabel('fare_amount [$]')
plt.show()

df_train['distance'] = np.log( df_train['distance'] + 1 )

roundDate(df_train)

selFeatures = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
               'passenger_count', 'distance', 'year','month']


x = np.asarray(df_train[selFeatures])
y = np.array(df_train['fare_amount'])

#ohe = OneHotEncoder()
#tratar = ohe.fit_transform(np.array(df_train['weekday']).reshape(-1,1)).toarray()

#x = np.concatenate([x,tratar], axis=1)
x = StandardScaler().fit_transform(x)

X_train, X_valid, y_train, y_valid = train_test_split(x,y, test_size = 0.3)

knn = KNeighborsRegressor(n_neighbors = 30, algorithm = 'kd_tree', n_jobs=3)
knn.fit(X_train,y_train)


def metrics(train_pred, valid_pred, y_train, y_valid):
    """Calculate metrics:
       Root mean squared error and mean absolute percentage error"""

    # Root mean squared error
    train_rmse = np.sqrt(sklm.mean_squared_error(y_train, train_pred))
    valid_rmse = np.sqrt(sklm.mean_squared_error(y_valid, valid_pred))

    # Calculate absolute percentage error
    train_ape = abs((y_train - train_pred) / y_train)
    valid_ape = abs((y_valid - valid_pred) / y_valid)

    # Account for y values of 0
    train_ape[train_ape == np.inf] = 0
    train_ape[train_ape == -np.inf] = 0
    valid_ape[valid_ape == np.inf] = 0
    valid_ape[valid_ape == -np.inf] = 0

    train_mape = 100 * np.mean(train_ape)
    valid_mape = 100 * np.mean(valid_ape)

    return train_rmse, valid_rmse, train_mape, valid_mape


def evaluate(model, X_train, X_valid, y_train, y_valid):
    """Mean absolute percentage error"""

    # Make predictions
    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)

    # Get metrics
    train_rmse, valid_rmse, train_mape, valid_mape = metrics(train_pred, valid_pred,
                                                             y_train, y_valid)

    print(f'Training:   rmse = {round(train_rmse, 2)} \t mape = {round(train_mape, 2)}')
    print(f'Validation: rmse = {round(valid_rmse, 2)} \t mape = {round(valid_mape, 2)}')


evaluate( knn, X_train, X_valid, y_train, y_valid )

########
# TEST #
########

df_test = pd.read_csv('mvml/test_MV.csv')

df_test['distance'] = df_test.apply(lambda row: distanceKM(row), axis=1)

df_test['distance'] = np.log(df_test['distance']+1)

roundDate(df_test)

x = np.asarray(df_test[selFeatures])

#ohe = OneHotEncoder()
#tratar = ohe.fit_transform(np.array(df_test['weekday']).reshape(-1,1)).toarray()

#x = np.concatenate([x,tratar], axis=1)
x = StandardScaler().fit_transform(x)

teste = knn.predict(x)

submission = pd.DataFrame(
    {'key': df_test.key, 'fare_amount': teste},
    columns = ['key', 'fare_amount'])

submission.to_csv('submission.csv', index = False)
