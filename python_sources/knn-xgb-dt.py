import pandas as pd
import warnings


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
warnings.filterwarnings("ignore")

stars = pd.read_csv('../input/star-dataset/6 class csv.csv')

### Data Cleaning

# Merge Same Colors
stars[stars.columns[5]] = stars[stars.columns[5]].apply(lambda x: 'yellow-white' if x == 'white-yellow' else x)
stars[stars.columns[5]] = stars[stars.columns[5]].apply(lambda x: 'Blue-white' if x == 'Blue white' else x)

# Change to small letters
stars[stars.columns[5]] = stars[stars.columns[5]].apply(lambda x: x.lower())

# Remove Dashes and Spaces
stars[stars.columns[5]] = stars[stars.columns[5]].apply(lambda x: x.replace("-", ""))
stars[stars.columns[5]] = stars[stars.columns[5]].apply(lambda x: x.replace(" ", ""))

### Features Engineering

# Encode the COLOR
from sklearn.preprocessing import LabelEncoder

color_encoder = LabelEncoder()
color_encoder.fit(stars[stars.columns[5]])
stars[stars.columns[5]] = color_encoder.transform(stars[stars.columns[5]])


# Encode the CLASS
class_encoder = LabelEncoder()
class_encoder.fit(stars[stars.columns[6]])
stars[stars.columns[6]] = class_encoder.transform(stars[stars.columns[6]])


# Divide Data into Features and Labels
features = stars.iloc[:, 0:6]
labels = stars[stars.columns[6]]

# SCALE the Features
from sklearn.preprocessing import MinMaxScaler

scale = MinMaxScaler()
scale.fit(features)
features = pd.DataFrame(scale.transform(features))


### Dataset Preparation

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

# Divide Data into Train and Test
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                            random_state=42)
########################################################################################################################
##### KNN
### KNN_NORMAL
print("KNeighborsClassifier")
knn_normal = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn_normal.fit(train_features, train_labels)
knn_normal_acc = accuracy_score(test_labels, knn_normal.predict(test_features))
print("1)KNN_NORMAL:", knn_normal_acc)

### KNN_cross validation
knn_cv = StratifiedKFold(n_splits=5)
knn_fold = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn_cv_scores = cross_val_score(cv=knn_cv, estimator=knn_fold, X=train_features, y=train_labels, scoring="accuracy")
print("2)KNN_cross validation [{:.3f}] +/- {:.3f}".format(knn_cv_scores.mean(), knn_cv_scores.std()))

### KNN_Grid Search
knn_grid = KNeighborsClassifier()
knn_grid_values = {'weights': ['uniform', 'distance', ], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                   'leaf_size': [2, 5, 10, 30, 50]}
knn_grid_acc = GridSearchCV(knn_grid, param_grid=knn_grid_values, scoring='accuracy')
knn_grid_acc.fit(train_features, train_labels)
y_pred_acc = knn_grid_acc.predict(test_features)
print('3)KNN_Grid Search : ', accuracy_score(test_labels, y_pred_acc))

########################################################################################################################
##### XGBClassifier
### XGB NORMAL
xgb_fold = XGBClassifier(n_estimators=1000, n_jobs=-1, random_state=42)
xgb_fold.fit(train_features, train_labels)
xgb_normal_acc = accuracy_score(test_labels, xgb_fold.predict(test_features))
print("4)XGB NORMAL:", xgb_normal_acc)

### XGB cross validation
xgb_cv = StratifiedKFold(n_splits=5)
xgb_fold = XGBClassifier(n_estimators=1000, n_jobs=-1, random_state=42)
xgb_cv_scores = cross_val_score(cv=xgb_cv, estimator=xgb_fold, X=train_features, y=train_labels, scoring="accuracy")
print("5)XGB cross validation {:.3f} +/- {:.3f}".format(xgb_cv_scores.mean(), xgb_cv_scores.std()))

### XGB_Grid Search
xgb_grid = XGBClassifier()
xgb_grid_values = {'booster': ['gbtree'], 'max_depth': [2, 3, 5],
                   'learning_rate': [1, 0.5, 0.1], 'n_estimators': [100, 1000, 5000], 'random_state': [42]}
xgb_grid_acc = GridSearchCV(xgb_grid, param_grid=xgb_grid_values, scoring='accuracy')
xgb_grid_acc.fit(train_features, train_labels)
y_pred_acc = xgb_grid_acc.predict(test_features)
print('6)XGB_Grid Search : ', accuracy_score(test_labels, y_pred_acc))
########################################################################################################################

##### DecisionTreeClassifier
### DT NORMAL
dt_Normal = DecisionTreeClassifier(random_state=42)
dt_Normal.fit(train_features, train_labels)
dt_normal_acc = accuracy_score(test_labels, dt_Normal.predict(test_features))
print("6)DT NORMAL:", dt_normal_acc)

### DT cross validation
dt_cv = StratifiedKFold(n_splits=5)
dt_fold = DecisionTreeClassifier(random_state=42)
dt_cv_scores = cross_val_score(cv=dt_cv, estimator=dt_fold, X=train_features, y=train_labels, scoring="accuracy")
print("5)DT cross validation {:.3f} +/- {:.3f}".format(dt_cv_scores.mean(), dt_cv_scores.std()))

### DT_Grid Search
dt_grid = DecisionTreeClassifier()
dt_grid_values = {'criterion': ['gini', 'entropy', ], 'min_samples_split': [2, 3, 4, 5],
                  'min_samples_leaf': [1, 2, 3, 4, 5],
                  'max_depth': [None, 2, 3, 4, 5],
                  'splitter': ['best'], 'random_state': [42]}
dt_grid_acc = GridSearchCV(dt_grid, param_grid=dt_grid_values, scoring='accuracy')
dt_grid_acc.fit(train_features, train_labels)
y_pred_acc = dt_grid_acc.predict(test_features)
print('7)DT_Grid Search : ', accuracy_score(test_labels, y_pred_acc))
