import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from matplotlib.pyplot import hist
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
import os
print(os.listdir("../input"))

df = pd.read_csv('../input/chicago_employees.csv')
df.columns
df['Annual Salary']=df['Annual Salary'].replace('[\$,]',"",regex=True).astype(float)
df['Hourly Rate']=df['Hourly Rate'].replace('[\$,]',"",regex=True).astype(float)

# missing values
df.info()
sns.heatmap(data=df.isnull(), cmap = 'Greens')
df.describe()
df['Annual Salary'].hist()
df['Annual Salary']=df['Annual Salary'].fillna(value=df['Annual Salary'].median())
df['Annual Salary'].hist()
sns.heatmap(data=df.isnull(), cmap = 'Greens')

# frequence
df.drop(['Name'], axis=1, inplace = True)
sns.boxenplot(data=df['Annual Salary'], orient="h")
df['Job Titles'].unique()
# max freq.values
df['Job Titles'].mode().head(10)
df['Job Titles'].value_counts().idxmax()
# TOP % values of Title
df['Ones']=1
df[['Department', 'Job Titles','Ones']].groupby(['Job Titles']).count().sort_values(by=['Ones'])
df[['Department', 'Job Titles','Ones']].groupby(['Job Titles']).count().sort_values(by=['Ones']).
apply(lambda x:100 * x / float(x.sum()))
# TOP % values of Deps
df[['Department', 'Job Titles','Ones']].groupby(['Department']).count().sort_values(by=['Ones'])
df[['Department', 'Job Titles','Ones']].groupby(['Department']).count().sort_values(by=['Ones']).apply(lambda x:100 * x / float(x.sum()))


# TOP salary
#BOX plot of Salary - police
sns.boxplot(x='Annual Salary', y='Job Titles', data=df[df['Job Titles']=='POLICE OFFICER'], orient='h')
sns.boxplot(x='Annual Salary', y='Job Titles', data=df[df['Job Titles']=='POLICE OFFICER (ASSIGNED AS DETECTIVE)'], orient='h')
sns.boxplot(x='Annual Salary', y='Job Titles', data=df[df['Job Titles']=='SERGEANT'], orient='h')
sns.boxplot(x='Annual Salary', y='Department', data=df[df['Department']=='POLICE'], orient='h')
#BOX plot of Salary - fire
sns.boxplot(x='Annual Salary', y='Job Titles', data=df[df['Job Titles']=='FIREFIGHTER'], orient='h')
sns.boxplot(x='Annual Salary', y='Department', data=df[df['Department']=='FIRE'], orient='h')
# sorting TOP salaries
df1=df[['Annual Salary','Job Titles']].groupby(['Job Titles']).sum().sort_values('Annual Salary')
df1.tail()


# encoding columns
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df['Full or Part-Time'])
df['Full or Part-Time'] = le.transform(df['Full or Part-Time'])
keys = le.classes_
values = le.transform(le.classes_)
dictionary = dict(zip(keys, values))
print(dictionary)

le1 = preprocessing.LabelEncoder()
le1.fit(df['Salary or Hourly'])
df['Salary or Hourly'] = le1.transform(df['Salary or Hourly'])
keys1 = le1.classes_
values1 = le1.transform(le1.classes_)
dictionary1 = dict(zip(keys1, values1))
print(dictionary1)

# correlation
sns.heatmap (data=df.corr())

# Random Forest Classification
# Importing the dataset
X = df.iloc[:, [2, 5]].values
y = df.iloc[:, 3].values

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





