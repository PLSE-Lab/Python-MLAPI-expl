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

df=pd.read_csv('../input/sharkattack.csv', parse_dates=['Date']) 
df.columns
df.drop(['Unnamed: 0'], axis=1, inplace = True)
df.dropna(inplace=True)
df.index = df.Date

# encoding columns  --- FATAL
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df['Fatal'])
df['Fatal1'] = le.transform(df['Fatal'])
keys = le.classes_
values = le.transform(le.classes_)
dictionary = dict(zip(keys, values))
print(dictionary)

# graphics of time slots
df.drop(['Date'], axis=1, inplace = True)
df_annual = df.resample('A-DEC').sum()
plt.plot(df_annual['Fatal1'], '-', label='Annual')
df_annual1 = df.iloc[:2500, :].resample('A-DEC').sum()
plt.plot(df_annual1['Fatal1'], '-', label='Annual 1987-2019')


# missing values
df.info()
sns.heatmap(data=df.isnull(), cmap = 'Greens', cbar  = False)
df.describe()




# encoding columns - Country
from sklearn import preprocessing
le1 = preprocessing.LabelEncoder()
le1.fit(df['Country'])
df['Country1'] = le1.transform(df['Country'])
keys1 = le1.classes_
values1 = le1.transform(le1.classes_)
dictionary1 = dict(zip(keys1, values1))
print(dictionary1)
# encoding columns - Country Code
from sklearn import preprocessing
le2 = preprocessing.LabelEncoder()
le2.fit(df['Country code'])
df['Country code1'] = le2.transform(df['Country code'])
keys2 = le2.classes_
values2 = le2.transform(le2.classes_)
dictionary2 = dict(zip(keys2, values2))
print(dictionary2)
# encoding columns - Type
from sklearn import preprocessing
le3 = preprocessing.LabelEncoder()
le3.fit(df['Type'])
df['Type1'] = le3.transform(df['Type'])
keys3 = le3.classes_
values3 = le3.transform(le3.classes_)
dictionary3 = dict(zip(keys3, values3))
print(dictionary3)
# encoding columns - Continent
from sklearn import preprocessing
le4 = preprocessing.LabelEncoder()
le4.fit(df['Continent'])
df['Continent1'] = le4.transform(df['Continent'])
keys4 = le4.classes_
values4 = le4.transform(le4.classes_)
dictionary4 = dict(zip(keys4, values4))
print(dictionary4)
# encoding columns - Hemisphere
from sklearn import preprocessing
le5 = preprocessing.LabelEncoder()
le5.fit(df['Hemisphere'])
df['Hemisphere1'] = le5.transform(df['Hemisphere'])
keys5 = le5.classes_
values5 = le5.transform(le5.classes_)
dictionary5 = dict(zip(keys5, values5))
print(dictionary5)
# encoding columns - Activity
from sklearn import preprocessing
le6 = preprocessing.LabelEncoder()
le6.fit(df['Activity'])
df['Activity1'] = le6.transform(df['Activity'])
keys6 = le6.classes_
values6 = le6.transform(le6.classes_)
dictionary6 = dict(zip(keys6, values6))
print(dictionary6)



#Graphics
df['Ones']=1
df[['Ones', 'Country']].groupby(['Country']).count().sort_values(by=['Ones']).tail()
import seaborn as sns
sns.set(rc={'figure.figsize':(15,9.27)})
sns.countplot(x="Country", hue="Fatal", data=df.iloc[:100,:])
df[['Ones', 'Continent']].groupby(['Continent']).count().sort_values(by=['Ones']).tail()

df[['Ones', 'Type']].groupby(['Type']).count().sort_values(by=['Ones']).tail()
sns.countplot(x="Type", hue="Fatal", data=df)
df[['Ones', 'Activity']].groupby(['Activity']).count().sort_values(by=['Ones']).tail()
sns.countplot(x="Activity", hue="Fatal", data=df)

df[['Ones', 'Fatal']].groupby(['Fatal']).count().sort_values(by=['Ones']).tail()
sns.countplot(x="Fatal", hue="Fatal", data=df)

# correlation
sns.heatmap (data=df.corr())




# Logistic Regression
X = df.iloc[:, [9,10,11,12,13]].values
y = df.iloc[:, 7].values
# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


# K-Nearest Neighbors (K-NN)
from sklearn.neighbors import KNeighborsClassifier
classifier1 = KNeighborsClassifier(n_neighbors = 50, metric = 'minkowski', p = 2)
classifier1.fit(X_train, y_train)
# Predicting 
y_pred1 = classifier1.predict(X_test)
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred1))
print(accuracy_score(y_test,y_pred1))


# Kernel SVM 
from sklearn.svm import SVC
classifier2 = SVC(kernel = 'rbf', random_state = 0)
classifier2.fit(X_train, y_train)
# Predicting 
y_pred2 = classifier2.predict(X_test)
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred2))
print(accuracy_score(y_test,y_pred2))


# Naive Bayes
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier3 = GaussianNB()
classifier3.fit(X_train, y_train)
# Predicting
y_pred3 = classifier3.predict(X_test)
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred3)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred3))
print(accuracy_score(y_test,y_pred3))


# Random Forest Classification
# Splitting 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10000, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))