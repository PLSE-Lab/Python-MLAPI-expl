import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score
import os
print(os.listdir("../input"))


df=pd.read_csv('../input/metro-bike-share-trip-data.csv')
df.columns
df.info()
df.describe()
sns.heatmap(df.isna())


df['Start Time']=pd.to_datetime(df['Start Time'])
df['End Time']=pd.to_datetime(df['Start Time'])
cols= ['Starting Station Latitude','Starting Station Longitude', 'Ending Station Latitude', 'Ending Station Longitude', 'Trip ID', 'Bike ID', 'Starting Lat-Long', 'Ending Lat-Long']
df.drop(columns = cols, axis=1, inplace = True)

df.dropna(axis=0, inplace = True)
sns.heatmap(df.isna())
df.info()
df.describe(include = 'all')

df.index = df['Start Time']

# sampling of means values of series of DF / groups by Index
df['Duration']=df['Duration']/60
df['Duration'].describe()

fig = plt.figure(figsize=[18, 10])
df_day = df.resample('D').count()
df_day_mean = df.resample('D').mean()
df_day['Duration'].hist()
plt.plot(df_day['Duration'], '-', label='By Days - Count')
plt.plot(df_day_mean['Duration'], '-', label='By Days - Mean')
plt.legend()

df_month = df.resample('M').count()
df_month_mean = df.resample('M').mean()
df_month['Duration'].hist()
plt.plot(df_month['Duration'], '-', label='By month - Count')
plt.plot(df_month_mean['Duration'], '-', label='By Month - Mean')
plt.legend()

# creating time cols
df['hour'] = df['Start Time'].dt.time
df['dayofweek'] = df['Start Time'].dt.dayofweek
df['hour_int'] = df['Start Time'].dt.hour
# stats by all hours/days
df['hour_int'].hist()
df['hour_int'].describe()
df['hour_int'].median()
df['dayofweek'].hist(bins=7)



# starting station stats where start/end stations are similar
df[df['Starting Station ID']==df['Ending Station ID']]['Duration'].mean()
df[df['Starting Station ID']==df['Ending Station ID']]['Duration'].count()
# hour analysis
df[df['Starting Station ID']==df['Ending Station ID']]['hour_int'].hist()
df[df['Starting Station ID']==df['Ending Station ID']]['hour_int'].describe()
# day analysis
df[df['Starting Station ID']==df['Ending Station ID']]['dayofweek'].hist(bins=7)
df[df['Starting Station ID']==df['Ending Station ID']]['dayofweek'].describe()
# 5/6-day stats
df[df['dayofweek']==5]['hour_int'].describe()
df[df['dayofweek']==5]['hour_int'].hist()
df[df['dayofweek']==6]['hour_int'].describe()
df[df['dayofweek']==6]['hour_int'].hist()
# the most popular stations where stations is similar - 3048 station
df[df['Starting Station ID']==df['Ending Station ID']]['Starting Station ID'].value_counts().head()
df[(df['Starting Station ID']==df['Ending Station ID']) & (df['Starting Station ID']==3048)]['hour_int'].describe()
df[(df['Starting Station ID']==df['Ending Station ID']) & (df['Starting Station ID']==3048)]['hour_int'].hist()
df[(df['Starting Station ID']==df['Ending Station ID']) & (df['Starting Station ID']==3048)]['dayofweek'].describe()
df[(df['Starting Station ID']==df['Ending Station ID']) & (df['Starting Station ID']==3048)]['dayofweek'].hist()
df[(df['Starting Station ID']==df['Ending Station ID']) & (df['Starting Station ID']==3048)]['Duration'].describe()



# starting station stats where stations are not similar
df[df['Starting Station ID']!=df['Ending Station ID']]['Duration'].mean()
df[df['Starting Station ID']!=df['Ending Station ID']]['Duration'].count()
# hour analysis
df[df['Starting Station ID']!=df['Ending Station ID']]['hour_int'].hist()
df[df['Starting Station ID']!=df['Ending Station ID']]['hour_int'].describe()
# day analysis
df[df['Starting Station ID']!=df['Ending Station ID']]['dayofweek'].hist(bins=7)
df[df['Starting Station ID']!=df['Ending Station ID']]['dayofweek'].describe()
# 3-day stats
df[df['dayofweek']==3]['hour_int'].describe()
df[df['dayofweek']==3]['hour_int'].hist()
# the most popular stations where stations are not similar - 3048 station
df[df['Starting Station ID']!=df['Ending Station ID']]['Starting Station ID'].value_counts().head()
df[(df['Starting Station ID']!=df['Ending Station ID']) & (df['Starting Station ID']==3030)]['hour_int'].describe()
df[(df['Starting Station ID']!=df['Ending Station ID']) & (df['Starting Station ID']==3030)]['hour_int'].hist()
df[(df['Starting Station ID']!=df['Ending Station ID']) & (df['Starting Station ID']==3030)]['dayofweek'].describe()
df[(df['Starting Station ID']!=df['Ending Station ID']) & (df['Starting Station ID']==3030)]['dayofweek'].hist()
df[(df['Starting Station ID']!=df['Ending Station ID']) & (df['Starting Station ID']==3030)]['Duration'].describe()


# Trip Route/Passholder Type count
df['Trip Route Category'].value_counts().head()
df['Passholder Type'].value_counts().head()


# Random Forest Classification
from sklearn import preprocessing
le1 = preprocessing.LabelEncoder()
le1.fit(df['Passholder Type'])
df['Passholder Type_1'] = le1.transform(df['Passholder Type'])
keys1 = le1.classes_
values1 = le1.transform(le1.classes_)
dictionary1 = dict(zip(keys1, values1))
print(dictionary1)

from sklearn import preprocessing
le2 = preprocessing.LabelEncoder()
le2.fit(df['Trip Route Category'])
df['Trip Route Category_1'] = le2.transform(df['Trip Route Category'])
keys2 = le2.classes_
values2 = le2.transform(le2.classes_)
dictionary2 = dict(zip(keys2, values2))
print(dictionary2)

sns.heatmap(df.corr())

# 
X = df.iloc[:, [0, 9,10,11]].values
y = df.iloc[:, 12].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
sc.get_params()
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))






