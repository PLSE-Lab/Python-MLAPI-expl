#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.facebook.com/codemakerz"><img src="https://scontent.ffjr1-4.fna.fbcdn.net/v/t1.0-9/36189148_736466693143793_2172101683281133568_n.png?_nc_cat=107&_nc_eui2=AeHzxv3SUcQBOfijLP-cEnHkX4z9XQXdeau__2MlErWZ1x07aZ1zx1PzJUDDxL6cpr7oPqYiifggXDptgtP8W5iCoDRjcdILDBYZ5Ig40dqi8Q&_nc_oc=AQmMCNXdzelFB2rdtpk8wN8nC410Wm2yKupYfYS1FxHNejTF0Jhr1G3WIZORKRF3TvFpohMB8Puw29Txxan8CW05&_nc_ht=scontent.ffjr1-4.fna&oh=7b13627e991a4d1b508923041bd7bc22&oe=5D8A7B03" />
# </a>
# Follow Us:
# Facebook: https://www.facebook.com/codemakerz
# 
# <h1>Delhi Weather Classification Using Decision Tree Classification</h1>
# <h3>Help us to increase the accuracy of model. Contact us to post your code.</h3>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/testset.csv")


# In[ ]:


df.head()


# # Problem Set
Given dataset provides the weather data for city Delhi, India. We will try to predict the weather _conds field. Like weather will be smoke, Haze, Clear.

For this we will use Decision Tree Classifier. You can use any other classifier to compare its accuracy.
# # EDA & Data Munging

# In[ ]:


# We can see all the column name has space in there names, lets assign new names with removed space.
df.columns


# In[ ]:


df.columns = map(lambda x: x.strip(), df.columns)


# In[ ]:


df.columns # Space removed


# In[ ]:


# Lets check the usual whether codition. We can see usually delhi's weather is either Haze, Smoke. Not good
# for health. :()
df._conds.value_counts(ascending=False)


# In[ ]:


# Lets plot top 10 weather condition in delhi.
plt.figure(figsize=(15, 10));
df._conds.value_counts().head(10).plot(kind='bar');
plt.title("Top 10 most common weather condition")
plt.plot();
# We can clearly see that haze and smoe are the most commo weather condition in delhi.


# In[ ]:


# Lets see top 10 least condition
plt.figure(figsize=(15, 10));
df._conds.value_counts(ascending=True).head(10).plot(kind="bar");
plt.title("Top 10 least whether condition in delhi");
plt.plot();


# In[ ]:


# common wind direction
df._wdire.value_counts()


# In[ ]:


plt.figure(figsize=(15, 10));
plt.title("Common wind direction in delhi");
df._wdire.value_counts().plot(kind="bar");
plt.plot();


# In[ ]:


# Average temprature
print("average temprature in delhi:", round(df._tempm.mean(axis=0),2))


# In[ ]:


# As we can see there is datetime column, We can extract year from it. Year can ve an important feature
# for us to calculate how temprature is changing according to year
def extract_year(value):
    return (value[0:4])


# In[ ]:


df.head()


# In[ ]:


# function to get month
def extract_month(value):
    return (value[4:6])


# In[ ]:


# Lets check our method
df["year"] = df["datetime_utc"].apply(lambda x:extract_year(x))
df["month"] = df["datetime_utc"].apply(lambda x:extract_month(x))


# In[ ]:


df.head() # So we can see a new column with year added


# In[ ]:


# lets check out data range
print("max, min: ", df.year.max(), ",", df.year.min())


# In[ ]:


# So our given data is from 1996 to 2017. 


# In[ ]:


# Number of records for paticular year
df.year.value_counts()


# In[ ]:


df.groupby("year")._tempm.mean()


# In[ ]:


df_mean = df.groupby("year")._tempm.mean().reset_index().sort_values('_tempm', ascending=True)


# In[ ]:


df_mean.dtypes


# In[ ]:


df_mean.year = df_mean.year.astype("float")


# In[ ]:


df_mean.dtypes


# In[ ]:



df_mean.plot(kind="scatter", x="year", y="_tempm", figsize=(15, 10))

plt.xticks(df_mean.year);
plt.title("Average temprature change");
plt.plot();


# So u can see there was a big change in year 1996-1997. It may be because of many reasons:
# 1. New industries started in the city.
# 2. People started purchasing more vehicles.
# or any other reasons.

# # Missing Values

# In[ ]:


df.isnull().sum()


# We will make copy of original dataset and will take only relevant columns.

# In[ ]:


df.columns


# In[ ]:


df_filtered = df[['datetime_utc', '_conds', '_dewptm', '_fog', '_hail',
       '_hum', '_pressurem', '_rain', '_snow', '_tempm',
       '_thunder', '_tornado', '_vism', '_wdird', '_wdire'
       , '_wspdm', 'year', "month"]]


# In[ ]:


# Lets replace missing values in _dewptm. We can take an avrgae of that year
df_filtered[df_filtered._dewptm.isnull()]


# In[ ]:


# We will try to replace value with average value of that year
for index,row in df_filtered[df_filtered._dewptm.isnull()].iterrows():
    mean_val = df_filtered[df_filtered["year"] == row["year"]]._dewptm.mean()
    df_filtered.at[index, "_dewptm"] = mean_val
    


# In[ ]:


df_filtered[df_filtered._dewptm.isnull()] # We replaced null values fof _dewtmp


# In[ ]:


df_filtered.shape


# In[ ]:


df_filtered.isnull().sum()
# so now we have only relevant columns. Lets handle them one by one.


# In[ ]:


# Handle _hum column.
df_filtered[df_filtered._hum.isnull()]


# In[ ]:


# We will use the same logic o replace as we did before.
# We will try to replace value with average value of that year
for index,row in df_filtered[df_filtered._hum.isnull()].iterrows():
    mean_val = df_filtered[df_filtered["year"] == row["year"]]._hum.mean()
    df_filtered.at[index, "_hum"] = mean_val
    


# In[ ]:


df_filtered[df_filtered._hum.isnull()] # replaced


# In[ ]:


df_filtered.isnull().sum() # Now lets handle _pressurem


# In[ ]:


df_filtered[df_filtered._pressurem.isnull()]


# In[ ]:


df_filtered.head()


# In[ ]:


# if you see pressure column, there are few -9999 values. Which is obviously bad values and it can affect your
# calculations very badly. So we will consider this also missing values. Lets convert them first to the nan
df_filtered._pressurem.replace(-9999.0, np.nan, inplace=True)


# In[ ]:


df_filtered.head() # so now -9999.0 is Nan.Lets again get the number of missing values in _pressurem


# In[ ]:


df_filtered._pressurem.isnull().sum() # So u can see previsously it was 232 and now its 983. 
# We need to check the data for this kin of errors.
# So we will use the same idea as before We will replace missing values with the mean values of _hum column
# for that partcular year.


# In[ ]:



for index,row in df_filtered[df_filtered._pressurem.isnull()].iterrows():
    mean_val = df_filtered[df_filtered["year"] == row["year"]]._pressurem.mean()
    df_filtered.at[index, "_pressurem"] = mean_val
    


# In[ ]:


df_filtered.isnull().sum() # pressurem is also resolved. Lets apply same for other columns. I will make
# it quickly. Process will be the same as above.


# In[ ]:


for index,row in df_filtered[df_filtered._tempm.isnull()].iterrows():
    mean_val = df_filtered[df_filtered["year"] == row["year"]]._tempm.mean()
    df_filtered.at[index, "_tempm"] = mean_val
    


# In[ ]:


for index,row in df_filtered[df_filtered._vism.isnull()].iterrows():
    mean_val = df_filtered[df_filtered["year"] == row["year"]]._vism.mean()
    df_filtered.at[index, "_vism"] = mean_val


# In[ ]:


for index,row in df_filtered[df_filtered._wdird.isnull()].iterrows():
    mean_val = df_filtered[df_filtered["year"] == row["year"]]._wdird.mean()
    df_filtered.at[index, "_wdird"] = mean_val


# In[ ]:


for index,row in df_filtered[df_filtered._wspdm.isnull()].iterrows():
    mean_val = df_filtered[df_filtered["year"] == row["year"]]._wspdm.mean()
    df_filtered.at[index, "_wspdm"] = mean_val


# In[ ]:


df_filtered.isnull().sum()


# In[ ]:


# As we can see _wdire is a categorical feature so we can not apply mean here. We have to get the most frequent
# value of _wdire for a year and then replace missing value with the most frequent value.
for index,row in df_filtered[df_filtered._wdire.isnull()].iterrows():
    most_frequent = df_filtered[df_filtered["year"] == row["year"]]._wdire.value_counts().idxmax()
    df_filtered.at[index, "_wdire"] = most_frequent


# In[ ]:


df_filtered.isnull().sum()


# In[ ]:


# now we can see,  _conds which is again acategorical feature.
# so we will apply again the same strategy as above(_wdire)
for index,row in df_filtered[df_filtered._conds.isnull()].iterrows():
    most_frequent = df_filtered[df_filtered["year"] == row["year"]]._conds.value_counts().idxmax()
    df_filtered.at[index, "_conds"] = most_frequent


# In[ ]:


df_filtered.isnull().sum()


# In[ ]:


## So finally ..... WE HAVE REPLACED ALL THE MISSING VALUES. Phew... Thats a whole big task.


# In[ ]:


df_filtered.year = df_filtered.year.astype("object")
df_filtered.month = df_filtered.month.astype("object")


# In[ ]:


df_filtered.dtypes


# In[ ]:


pd.crosstab(df_filtered.year, [df_filtered.month], values=df_filtered._tempm, aggfunc="mean")


# In[ ]:


# Heatmap for year and average temprature across the month. More red more heat, more blue less heat
plt.figure(figsize=(15, 10));
sns.heatmap(pd.crosstab(df_filtered.year, [df_filtered.month], values=df_filtered._tempm, aggfunc="mean"),
            cmap="coolwarm", annot=True, cbar=True);
plt.title("Average Temprature 1996-2016")
plt.plot();


# Now our dataset doesn;t have any missing values in it. Now we should observe one thing. That our _windre is
# a categorical column and it is also important to predict a whether but the thing is your model does not understand a text value. So we need to encode this categorical column so that we can change it to integer

# In[ ]:


df_filtered._conds.value_counts()


# # Feature & Target Matrix

# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[ ]:


df_filtered.columns


# In[ ]:


feature_columns = ['_wdire', '_dewptm', '_fog', '_hail', '_hum',
       '_pressurem', '_rain', '_snow', '_tempm', '_thunder', '_tornado',
       '_vism', '_wdird', '_wspdm', 'year', 'month', '_conds']


# In[ ]:


# Lets create a new dataset, so that we dont change in our filtered dataset
# We will create dataset in such a way, _wdire(categorical feature in starting position & target variable
# at last which is _conds
df_final = df_filtered[feature_columns]


# In[ ]:


df_final.head()


# In[ ]:


df_final.dtypes


# In[ ]:


df_final._wdire.value_counts()


# In[ ]:


wdire_dummies = pd.get_dummies(df_final["_wdire"])


# In[ ]:


df_final = pd.concat([wdire_dummies, df_final], axis=1)


# In[ ]:


df_final.head()


# In[ ]:


df_final.columns


# In[ ]:


df_final.drop("_wdire", inplace=True, axis=1)


# In[ ]:


df_final.columns


# In[ ]:


X = df_final.iloc[:, 0:-1].values
X.shape


# In[ ]:


y = df_final.iloc[:, -1].values


# In[ ]:


label_encoder = LabelEncoder()


# In[ ]:


y = label_encoder.fit_transform(y)


# In[ ]:


y.shape


# In[ ]:


# SO now our Feature Matrix(X) and target matrix y is ready


# # Train & Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0) # test size =0.25 or 25%


# In[ ]:


print("Shape of X_train", X_train.shape)
print("Shape of X_test", X_test.shape)
print("Shape of y_train", y_train.shape)
print("Shape of y_test", y_test.shape)


# # Create Model

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


clf = DecisionTreeClassifier(criterion="entropy", random_state=0)


# # Train Model

# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


y_pred


# # Accuracy

# In[ ]:


from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, y_pred))


# In[ ]:


# Congrats your model is ready wth 78% of accuracy, Please provide your suggestions to increase the accuracy of the
# model

