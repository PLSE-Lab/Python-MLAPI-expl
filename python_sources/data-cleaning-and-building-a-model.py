#Please see document here to learn more about the project amd my thought process : https://bit.ly/2VA4TSk

import numpy as np
import pandas as pd

df = pd.read_csv(r"../input/olaresults.txt", sep='\t', header=(0), low_memory=False)

#create trip duration

df['start'] = pd.to_datetime(df['Start_Date'] + ' ' + df['Start_Time'])
df['end'] = pd.to_datetime(df['Stop_Date'] + ' ' + df['Stop_Time'])
df['trip_duration'] = (df.end - df.start) / np.timedelta64(1, 's') / 60

#select only the columns I am going to need:

df = df[['IMEI', 'OLA_COMMENTS', 'trip_duration', 'Smooth_Driving_Score', 'Turning_Score']]


#drop all rows where IMEI is nan - I see we did not record scores for these so they are irrelevant to the project
df = df[pd.notnull(df['IMEI'])]

#creating a separate df with averages for each driver (based on IMEI id), to give more features to work with

#get average score for each imei number
average_of_each = df.groupby('IMEI').agg(np.mean)


#create separate df to feed information into
averages_per_trip = pd.DataFrame(columns= ['Smooth_Driving_Score', 'Turning_Score', 'trip_duration'])

for index, row in df.iterrows():
    trip_imei = row['IMEI']
    averages = average_of_each.loc[trip_imei]
    averages_per_trip = averages_per_trip.append(averages, ignore_index= True)



averages_per_trip.columns=['avg_smooth','avg_turning', 'avg_duration']

df = df.reset_index(drop=True)
averages_per_trip = averages_per_trip.reset_index(drop=True)

#merge dataframes
df[['avg_smooth','avg_turning', 'avg_duration']]=averages_per_trip[['avg_smooth','avg_turning', 'avg_duration']]

#search through OLA_COMMENTS to isolate comments which include 'Unsafe Driving'.

df['unsafe'] = 0
target_pos = 0

for index, row in df.iterrows():
    ola_comments = row['OLA_COMMENTS']
    if type(ola_comments) == float:
        target_pos += 1
        continue
    if ola_comments.find("Unsafe") != -1:
        df.unsafe.iloc[target_pos] = 1
    target_pos += 1

#filtering the df to improve results
df = df[(df['trip_duration'] > 5) & (df['Turning_Score'] < 99) & (df['Turning_Score'] > 0)]

#drop unnecessary columns
df = df.drop(df.columns[[0, 1, 2, -2]], axis=1)


#setting features and labels
labels = np.array(df['unsafe'])
features= df.drop('unsafe', axis = 1)

features = np.array(features)

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(features, labels,
                                                          stratify = labels,
                                                          test_size = 0.3,
                                                          random_state = 8)

#oversampling minority class in training set
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=12, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)

from sklearn.ensemble import RandomForestClassifier

# Create model with 100 trees
clf_rf = RandomForestClassifier(n_estimators=100, random_state=12, class_weight='balanced')
clf_rf.fit(x_train_res, y_train_res)

# Fit on training data
actual = y_val
predictions = clf_rf.predict(x_val)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(actual,predictions))


from sklearn.metrics import classification_report
print(classification_report(actual, predictions))
