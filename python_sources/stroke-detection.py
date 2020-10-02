# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df = pd.read_csv("../input/train_2v.csv")

df.info()

# NULL OR MISSING VALUES 

def findMissingValues(df):
  total = df.isnull().sum().sort_values(ascending=False)
  percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
  missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
  # f, ax = plt.subplots(figsize=(15, 6))
  # plt.xticks(rotation='90')
  # sns.barplot(x=missing_data.index, y=missing_data['Percent'])
  # plt.xlabel('Features', fontsize=15)
  # plt.ylabel('Percent of missing values', fontsize=15)
  # plt.title('Percent missing data by feature', fontsize=15)
  print(missing_data.head())
  # return plt

findMissingValues(df)

# Replace missing values using median 
median = df['bmi'].median()
df['bmi'].fillna(median, inplace=True)

# Drop any rows with null value
df = df.dropna()

findMissingValues(df)

# Change string to numerical values

print("\n\n Change Yes to 1 and No to 0\n")
ever_married_map = {'Yes' : 1, 'No' : 0}
df['ever_married'] = df['ever_married'].map(ever_married_map)

print("\n\n Making gender categorical\n")
df = pd.concat([df,pd.get_dummies(df['gender'], prefix = 'gender')], axis=1)

print("\n\n Making residence type categorical\n")
df = pd.concat([df,pd.get_dummies(df['Residence_type'], prefix = 'residence')], axis=1)

print("\n\n Making work type categorical\n")
df = pd.concat([df,pd.get_dummies(df['work_type'], prefix = 'work')], axis=1)

print("\n\n Change Smoking status values\n")
smoking_map = {'never smoked' : 0, 'formerly smoked' : 1, 'smokes' : 2}
df['smoking_status'] = df['smoking_status'].map(smoking_map)

# Remove redundant columns

print("\n\n Remove ID Column\n")
del df['id']
print("\n\n Remove gender Column\n")
del df['gender']
print("\n\n Remove Residence_type Column\n")
del df['Residence_type']
print("\n\n Remove work_type Column\n")
del df['work_type']

# Rearange Columns

df = df[['age','hypertension','heart_disease','ever_married','avg_glucose_level','bmi','smoking_status','gender_Female',
'gender_Male','gender_Other','residence_Rural','residence_Urban','work_Govt_job','work_Never_worked','work_Private','work_Self-employed','work_children','stroke']]


# Display all columns

pd.set_option('display.max_columns', 20)
print(df.head(5))

df.hist(figsize=(10,11))
#plt.show()

# Preprocess the data to bring values to closer range

dataset_plot = df
dataset_plot[['age','hypertension','heart_disease','ever_married','avg_glucose_level','bmi','smoking_status','gender_Female',
'gender_Male','gender_Other','residence_Rural','residence_Urban','work_Govt_job','work_Never_worked','work_Private','work_Self-employed','work_children']].head(100).plot(style=['o','x','r--','g^'])
plt.legend(bbox_to_anchor=(0.,1.02,1., .102), loc=3,ncol=2, mode="expand", fontsize="x-large", borderaxespad=0.)
#plt.show()

#Preprocessing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
stroke = scaler.fit_transform(dataset_plot[['age','hypertension','heart_disease','ever_married','avg_glucose_level','bmi','smoking_status','gender_Female',
'gender_Male','gender_Other','residence_Rural','residence_Urban','work_Govt_job','work_Never_worked','work_Private','work_Self-employed','work_children']].head(100))
stroke_df = pd.DataFrame(stroke)
stroke_df.columns = ['age','hypertension','heart_disease','ever_married','avg_glucose_level','bmi','smoking_status','gender_Female',
'gender_Male','gender_Other','residence_Rural','residence_Urban','work_Govt_job','work_Never_worked','work_Private','work_Self-employed','work_children']
stroke_df.plot(style=['o','x','r--','g^'])
plt.legend(bbox_to_anchor=(0.,1.02,1., 10.), loc=3,ncol=2, mode="expand", fontsize="x-large", borderaxespad=0.)
#plt.show()

# ML

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = df.drop(['stroke'], axis=1)
y = df['stroke']

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)

#Logistic Regression
logreg = LogisticRegression()
# Train the model using the training sets and check score
logreg.fit(X_train, y_train)
#Predict Output
log_predicted= logreg.predict(X_test)

logreg_score = round(logreg.score(X_train, y_train) * 100, 2)
logreg_score_test = round(logreg.score(X_test, y_test) * 100, 2)
#Equation coefficient and Intercept
print('Logistic Regression Training Score: \n', logreg_score)
print('Logistic Regression Test Score: \n', logreg_score_test)
print('Coefficient: \n', logreg.coef_)
print('Intercept: \n', logreg.intercept_)
print('Accuracy: \n', accuracy_score(y_test,log_predicted))
print('Confusion Matrix: \n', confusion_matrix(y_test,log_predicted))

# Random Forest

random_forest = RandomForestClassifier(n_estimators=10)
random_forest.fit(X_train, y_train)
#Predict Output
rf_predicted = random_forest.predict(X_test)

random_forest_score = round(random_forest.score(X_train, y_train) * 100, 2)
random_forest_score_test = round(random_forest.score(X_test, y_test) * 100, 2)
print('Random Forest Test Score: \n', random_forest_score_test)
print('Accuracy: \n', accuracy_score(y_test,rf_predicted))
print(confusion_matrix(y_test,rf_predicted))
print(classification_report(y_test,rf_predicted))

# Neural Networks

numpyMatrix=np.array(df.values, dtype = np.float64)
X_input = numpyMatrix[:,0:17]
X=X_input
Y = numpyMatrix[:,17]
# create model
model = Sequential()
model.add(Dense(50, input_dim=17, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
tbCallBack = keras.callbacks.TensorBoard(log_dir='./strokeDetection/logs', histogram_freq=0, write_graph=True, write_images=True)

# Fit the model
history = model.fit(X, Y,validation_split=0.33, epochs=100, batch_size=16,callbacks=[tbCallBack])

# evaluate the model
scores = model.evaluate(X, Y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Predict

# new instance where we do not know the answer
from numpy import array
Xnew = array([[37, 0, 0, 0, 98, 29, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]])
ynew = model.predict_classes(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))

Xnew = array([[73, 0, 0, 1, 110, 26, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]])
ynew = model.predict_classes(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
