# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Download Dataset
data = pd.read_csv('/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv')

data.head()

data.isnull().any()

# Divide dataset into features and target
features = data.drop('target', axis = 1)
features.head()

target = data['target']
target.head()

# EDA for dataset
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.pairplot(features)

data.corr()

sns.heatmap(data.corr())

# Scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(features)
df_scaled=scaler.transform(features)
features.columns
feature=pd.DataFrame(df_scaled,columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal'])


# Split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3)

# Using Decision Tree , Random Forest and KNN for classification.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Metrics evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_pred = dtc.predict(X_test)

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

X_Train, X_Test, y_Train, y_Test = train_test_split(feature, target, test_size = 0.3)

knn=KNeighborsClassifier(n_neighbors=20)
knn.fit(X_Train,y_Train)
predictknn=knn.predict(X_Test)


print('DTC', classification_report(y_test,dtc_pred))
print('DTC', confusion_matrix(y_test,dtc_pred))

print('RFC', classification_report(y_test,rfc_pred))
print('RFC', confusion_matrix(y_test,rfc_pred))

print('KNN', classification_report(y_Test,predictknn))
print('KNN', confusion_matrix(y_Test,predictknn))

# Finding the right n_neighbors

error_rate=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i !=y_test))

plt.figure(figsize=(12,8))
plt.plot(range(1,40),error_rate,color='blue',marker='o')

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_Train,y_Train)
predictknn=knn.predict(X_Test)
print('KNN', classification_report(y_Test,predictknn))
print('KNN', confusion_matrix(y_Test,predictknn))

