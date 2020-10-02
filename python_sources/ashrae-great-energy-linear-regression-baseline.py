# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_log_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


#loading dataset
building_metadata_df = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')
weather_test_df = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')
train_df = pd.read_csv('../input/ashrae-energy-prediction/train.csv')
test_df = pd.read_csv('../input/ashrae-energy-prediction/test.csv')
weather_train_df = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')


#joining building and weather dataset with train data
train_df = train_df.merge(building_metadata_df, on = "building_id",how = "left")
train_df = train_df.merge(weather_train_df, on = ["site_id", "timestamp"], how = "left")
del weather_train_df

#joining building and weather dataset with test data
test_df = test_df.merge(building_metadata_df, on = "building_id", how = "left")
test_df = test_df.merge(weather_test_df, on = ["site_id", "timestamp"], how = "left")
del weather_test_df

#function to extract date features from timestamp column
def prepare_date_features(input_df):
    input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
    input_df['year'] = input_df['timestamp'].dt.year
    input_df['quarter_of_year'] = input_df['timestamp'].dt.quarter
    input_df['month_of_year'] = input_df['timestamp'].dt.month
    input_df['day_of_month'] = input_df['timestamp'].dt.day
    input_df["hour_of_day"] = input_df["timestamp"].dt.hour
    return input_df

#extrating date features for train and test dataset
train_df = prepare_date_features(train_df)
test_df = prepare_date_features(test_df)


#function to compute avg value used for missing value imputation 
def average_imputation(input_df, column_name):
    imputation = input_df.groupby(['timestamp'])[column_name].mean()
    input_df.loc[input_df[column_name].isnull(), column_name] = input_df[input_df[column_name].isnull()][[column_name]].apply(lambda x: imputation[input_df['timestamp'][x.index]].values)
    del imputation
    return input_df

#impute null values of train dataset in wind_speed & wind_direction columns with groupby timestamp and corresponding column mean
train_df = average_imputation(train_df, 'wind_speed')
train_df = average_imputation(train_df, 'wind_direction')

#impute null values of test dataset in wind_speed & wind_direction columns with groupby timestamp and corresponding column mean
test_df = average_imputation(test_df, 'wind_speed')
test_df = average_imputation(test_df, 'wind_direction')

#log trasfromation for square_feet column for both train and test dataset
train_df['square_feet'] = np.log(train_df['square_feet'])
test_df['square_feet'] = np.log(test_df['square_feet'])

#encode primary_use using LabelEncoder
le = LabelEncoder()
train_df["primary_use"] = le.fit_transform(train_df["primary_use"])
test_df["primary_use"] = le.fit_transform(test_df["primary_use"])


#drop clumns from train dataset which are having null values for building base line model.
train_df = train_df.drop(['year_built', 
                           'floor_count','air_temperature',
                           'cloud_coverage','dew_temperature','precip_depth_1_hr',
                           'sea_level_pressure'],axis=1)

#drop clumns from test dataset which are having null values for building base line model.
test_df = test_df.drop(['year_built', 
                           'floor_count','air_temperature',
                           'cloud_coverage','dew_temperature','precip_depth_1_hr',
                           'sea_level_pressure'],axis=1)

#Divide the training dataset  into training and testing
X = train_df.drop("meter_reading", axis=1)
y = train_df["meter_reading"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 3)

#training model
lr = LinearRegression()
lr.fit(X_train.drop(['timestamp'],axis=1), y_train)

#prediction on sample test dataset
predictions = lr.predict(X_test.drop(['timestamp'],axis=1))
X_test['prediction'] = predictions
X_test['y_test'] = y_test

#remove negative prediction because we cant score mean_squared_log_error for negative value
X_test = X_test[(X_test['prediction'] >= 0) & (X_test['y_test'] >= 0)]

print("mean_squared_log_error:", np.sqrt(mean_squared_log_error( X_test['y_test'], X_test['prediction'] )))

#get prediction on test datset
test_df_predictions = lr.predict(test_df.drop(['timestamp','row_id'],axis=1))

test_df['meter_reading'] = test_df_predictions

#select only required columns
submission_data = test_df[['row_id','meter_reading']]

#saving submission file
submission_data.to_csv("submission.csv",index=False)

#kaggle competitions submit -c ashrae-energy-prediction -f submission.csv -m "baseline"

