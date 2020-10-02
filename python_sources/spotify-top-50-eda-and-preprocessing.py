#!/usr/bin/env python
# coding: utf-8

# # **Preparing the environment and uploading data**

# **Hi, everyone. It's my EDA + Preprocessing for Top 50 Spotify Songs - 2019 dataset. If you have questions or you can help me with my work feel free to share your thoughts in comments. Enjoy the notebook :)**
# ![](https://www.iguides.ru/upload/medialibrary/f69/f69383fcc96af0bff95c8223aa31d18a.jpg)

# **Import packages**

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='dark')

# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_classif


# **Acquire data**

# In[ ]:


data = pd.read_csv("../input/top50spotify2019/top50.csv", encoding='ISO-8859-1')


# # **Exploratory Data Analysis (EDA)**

# **Take a first look**

# In[ ]:


data.head()


# **Drop "Unnamed: 0" column**

# In[ ]:


data.drop('Unnamed: 0', inplace=True, axis=1)


# **Get the shape of our data**

# In[ ]:


data.shape


# **As wee can see, there are strange column names, so we fix that**

# In[ ]:


data.rename(columns={'Track.Name':'Track_Name', 'Artist.Name':'Artist_Name',
                      'Beats.Per.Minute':'Beats_Per_Minute', 'Loudness..dB..':'Loudness',
                      'Valence.':'Valence', 'Length.':'Length', 'Acousticness..':'Acousticness', 'Speechiness.':'Speechiness'}, inplace=True)
data.drop('Track_Name', axis=1, inplace=True)
data.head()


# **Get detailed statistics about data**

# In[ ]:


def detailed_analysis(df, pred=None):
  obs = df.shape[0]
  types = df.dtypes
  counts = df.apply(lambda x: x.count())
  uniques = df.apply(lambda x: [x.unique()])
  nulls = df.apply(lambda x: x.isnull().sum())
  distincts = df.apply(lambda x: x.unique().shape[0])
  missing_ratio = (df.isnull().sum() / obs) * 100
  skewness = df.skew()
  kurtosis = df.kurt()
  print('Data shape:', df.shape)

  if pred is None:
    cols = ['types', 'counts', 'nulls', 'distincts', 'missing ratio', 'uniques', 'skewness', 'kurtosis']
    details = pd.concat([types, counts, nulls, distincts, missing_ratio, uniques, skewness, kurtosis], axis=1)
  else:
    corr = df.corr()[pred]
    details = pd.concat([types, counts, nulls, distincts, missing_ratio, uniques, skewness, kurtosis, corr], axis=1, sort=False)
    corr_col = 'corr ' + pred
    cols = ['types', 'counts', 'nulls', 'distincts', 'missing ratio', 'uniques', 'skewness', 'kurtosis', corr_col]

  details.columns = cols
  dtypes = details.types.value_counts()
  print('____________________________\nData types:\n', dtypes)
  print('____________________________')
  return details


# In[ ]:


details = detailed_analysis(data, 'Energy')
display(details.sort_values(by='corr Energy', ascending=False))


# In[ ]:


data.describe()


# **Calculating the number of songs by genre**

# In[ ]:


data.Genre.value_counts()


# In[ ]:


values = data.Genre.value_counts()
indexes = values.index

fig = plt.figure(figsize=(35, 15))
sns.barplot(indexes, values)

plt.ylabel('Number of values')
plt.xlabel('Genre')


# **Unique artists**

# In[ ]:


data.Artist_Name.value_counts()


# **Liveness distribution**

# In[ ]:


values = data.Liveness.value_counts()
indexes = values.index

fig = plt.figure(figsize=(15, 10))
sns.barplot(indexes, values)

plt.ylabel('Number of values')
plt.xlabel('Liveness')


# **BPM distribution. I use a function to split "Beats_Per_Minute" column by groupes**

# In[ ]:


data.Beats_Per_Minute.describe()


# In[ ]:


def transform(x):
  if x <= 100:
    return '85-100'
  elif x <= 120:
    return '101-120'
  elif x <= 140:
    return '121-140'
  else:
    return '141-190'

groups_of_bpm = data.Beats_Per_Minute.apply(transform)

values = groups_of_bpm.value_counts()
labels = values.index
colors = ['red', 'blue', 'green', 'brown']

fig = plt.figure(figsize=(15, 10))
plt.pie(values, colors=colors, autopct='%1.1f%%', startangle=90, textprops={ 'color': 'w' })

plt.title('BPM distribution', fontsize=15)
plt.legend(labels)
plt.show()


# **Correlation heatmap**

# In[ ]:


correlations = data.corr()

fig = plt.figure(figsize=(12, 10))
sns.heatmap(correlations, annot=True, cmap='GnBu_r', center=1)


# **Relationship between energy and loudness (quite good correlation)**

# In[ ]:


fig = plt.figure(figsize=(15, 10))
sns.regplot(x='Energy', y='Loudness', data=data)


# # **Data preprocessing**

# **Encode categorical features**

# In[ ]:


le = LabelEncoder()

for col in data.columns.values:
  if data[col].dtypes == 'object':
    le.fit(data[col].values)
    data[col] = le.transform(data[col])

data.head()


# **Let's find the best features (according to the heatmap too)**

# In[ ]:


X = data.drop('Loudness', axis=1)
y = data.Loudness

selector = SelectKBest(score_func=f_classif, k=5)
fitted = selector.fit(X, y)
features_scores = pd.DataFrame(fitted.scores_)
features_columns = pd.DataFrame(X.columns)

best_features = pd.concat([features_columns, features_scores], axis=1)
best_features.columns = ['Feature', 'Score']
best_features.sort_values(by='Score', ascending=False, inplace=True)
best_features


# **Split data. Actually we ain't drop columns as we have only 10 (except Artist_Name since there are too many artists and it becomes useless).**

# In[ ]:


X.drop('Artist_Name', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train


# **Scale data**

# In[ ]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

type(X_train), type(X_test)


# # **Model**

# **Apply Linear Regression on our data**

# In[ ]:


regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(regressor.intercept_, regressor.coef_)
print(mean_squared_error(y_test, y_pred))


# **Apply Support Machine Regression**

# In[ ]:


regressor = SVR(C=0.5)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print(mean_squared_error(y_test, y_pred))


# **Apply K-Means Clustering. Actually, I think MSE is excessively high since our data is too small**

# In[ ]:


clustering = KMeans(n_clusters=2)
clustering.fit(X_train, y_train)

y_pred = clustering.predict(X_test)
print(mean_squared_error(y_test, y_pred))


# **SVR performs better than others. I'm going to update this work later. Thanks for reading my notebook :)**
