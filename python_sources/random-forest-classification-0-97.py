import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns 
import os
print(os.listdir("../input"))

df = pd.read_csv('../input/googleplaystore.csv')
df_ = pd.read_csv('../input/googleplaystore.csv')
data = pd.read_csv('../input/googleplaystore_user_reviews.csv')
df.columns

df['Price'].unique()
df['Content Rating'].unique()
df['Type'].unique()

df['Installs']=df['Installs'].apply(lambda x: str(x).replace("+",""))
df['Installs_1']=df['Installs'].apply(lambda x: str(x).replace(",",""))
df['Price']=df['Price'].apply(lambda x: str(x).replace("$",""))

df = df.drop_duplicates(subset=['App'], keep = 'first')
df['Type'].dropna(inplace = True)
sns.heatmap(data=df.isna())
df.drop(columns=['Rating','Size'], inplace = True)
df['Type'].unique()
df.shape

to_drop = ['Last Updated', 'Current Ver']
df.drop(to_drop, axis =1, inplace = True)
df1=df[df.Installs != 'Free']
df2=df1[df1.Price != 'Everyone']

df2['Installs_1']=df2['Installs_1'].astype(int)
df2['Price']=df2['Price'].astype(float)


# transform data (astype('***'))
df2.info()
df2.dtypes
df2.describe()

df2['Reviews'] = pd.to_numeric(df2['Reviews'], errors='coerce')
df2['Price'] = pd.to_numeric(df2['Price'], errors='coerce')

df2.info()
df2.dtypes
df2.describe()

# EDA
df2['Ones'] = 1


sns.heatmap(data=df2.isna())

#general graphs
#general count of installations
plt.figure(figsize=(20,5))
sns.countplot(x=df2['Installs'], palette="hls")
df2[['Installs', 'Ones']].groupby(['Installs']).count().sort_values(by=['Ones']).tail().plot()
df2[['Installs', 'Ones']].groupby(['Installs']).count().sort_values(by=['Ones']).tail()
#types of installations
sns.catplot(x="Installs", hue="Type",data=df2[df2['Type']!= '0'], kind="count", height=5, aspect=3);
sns.catplot(x="Installs", hue="Type",data=df2[df2['Type']!= '0'], col = 'Type',kind="count", height=3, aspect=6);
sns.countplot(x=df2['Type'])
# Android Ver graph
plt.figure(figsize=(40,5))
sns.countplot(x=df2['Android Ver'], palette="hls")
df2[['Android Ver', 'Ones']].groupby(['Android Ver']).count().sort_values(by=['Ones']).tail()
# top values of genres
sns.countplot(x=df2['Genres'], palette="hls")
df2[['Genres', 'Ones']].groupby(['Genres']).count().sort_values(by=['Ones']).tail()
# top values of Content Rating
plt.figure(figsize=(15,5))
sns.countplot(x=df2['Content Rating'], palette="hls")
# top values of Rating
plt.figure(figsize=(15,5))
sns.countplot(x=df_['Rating'], palette="hls")
# TOP count of Installs
df2.groupby(['Installs'])['Ones'].sum().sort_values(ascending=False).head()
# the most popular Apps
plt.figure(figsize=(15,5))
sns.barplot(x=df2.sort_values(by=['Reviews'], ascending=False)['App'][:5], y=df2.sort_values(by=['Reviews'], ascending=False)['Reviews'][:5], palette="hls")
df2.groupby(['App'])['Reviews'].sum().sort_values(ascending=False).head()

# mean ratings of Categoties
pd.set_option('display.max_columns', None)
df_.groupby(['Category'])['Rating'].describe(include = 'all').transpose()
df_.groupby(['Category'])['Rating'].min().sort_values(ascending=False).head()
df_.groupby(['Category'])['Rating'].min().sort_values(ascending=False).tail()

# pred SVM
sns.heatmap(df2.corr())

# Importing the dataset
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score

df2['Type'].dropna(inplace = True)
df.isna().all()
df2['Type'].unique()
df2.info()
df3=df2.dropna()

from sklearn import preprocessing
le1 = preprocessing.LabelEncoder()
le1.fit(df3['Type'])
df3['Type_1'] = le1.transform(df3['Type'])
keys1 = le1.classes_
values1 = le1.transform(le1.classes_)
dictionary1 = dict(zip(keys1, values1))
print(dictionary1)

X = df3.iloc[:, [2, 5,9]].values
y = df3.iloc[:, 11].values

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
classifier = RandomForestClassifier(n_estimators = 1, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
