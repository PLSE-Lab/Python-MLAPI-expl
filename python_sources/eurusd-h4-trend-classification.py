import pandas as pd
import numpy as np 

from sklearn import metrics
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn.cross_validation import train_test_split

from sklearn import linear_model

#########################################################################################
#Load Data
train_data = pd.read_csv("../input/H4_EURUSD.csv")
print("features in train: ", len(train_data.columns))
print(train_data.head())

#########################################################################################
#Data Preprocessing
#numerical & categorical features
numerical = train_data._get_numeric_data().columns
categorical = [item for item in train_data.columns if item not in numerical]
categorical_indexes = [train_data.columns.get_loc(x) for x in categorical]  

#LabelEncoder
from sklearn import preprocessing

for x in categorical:
    le = preprocessing.LabelEncoder()
    le.fit(train_data[x])
    train_data[x] = le.transform(train_data[x])

#encode class variable
le = preprocessing.LabelEncoder()
le.fit(train_data['A487'])
train_data['A487'] = le.transform(train_data['A487'])

#take a look on the encoded data
print(train_data.head())

#########################################################################################
# Feature Importance
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

# fit an Extra Trees model to the data
model = ExtraTreesClassifier(n_estimators = 50)
model.fit(train_data.values[:,:-1], train_data.values[:,-1])

#list feature importance
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
top_features = 10
for f in range(top_features):
    print(f, ':', indices[f], train_data.columns[indices[f]], importances[indices[f]])

#########################################################################################
#split
train, test = train_test_split(train_data, test_size=0.2)
train = train.values
test = test.values

#slice
X_train = train[:,:-1]
X_test = test[:,:-1]
y_train = train[:,-1]
y_test = test[:,-1]

#########################################################################################
#Train
clf = linear_model.LogisticRegression(max_iter=10).fit(X_train, y_train)
clf.score(X_test, y_test)

#########################################################################################
#Evaluate
predictions = clf.predict(X_test)
print ('AUC:', metrics.roc_auc_score(y_test, predictions))
print ('Precision:', metrics.precision_score(y_test, predictions))
print ('Recall:', metrics.recall_score(y_test, predictions))