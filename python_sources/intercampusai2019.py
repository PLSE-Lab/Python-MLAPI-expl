
# Data Science Nigeria InterCampusAI2019 Competition

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O

# Import Data
train_data_path = '../input/intercampusai2019/train.csv'
test_data_path = '../input/intercampusai2019/test.csv'
sub_format_path = '../input/intercampusai2019/sample_submission2.csv'

train_data = pd.read_csv (train_data_path)
test_data = pd.read_csv (test_data_path)
submission = pd.read_csv (sub_format_path)

# Examine the data
train_data.head(3)
train_data.info()
train_data.describe()

test_data.head(3)
test_data.info()
test_data.describe()

# Sorting the Data and some Feature Engineering
def to_years (data, feat_col, new_col_name, present_year=2019):
    '''Function to tranform calendar years to number of years.'''
    data[ new_col_name ] = present_year - data[ feat_col ]
    data = data.drop( feat_col, axis=1 )

# Convert calendar years to number of years...
to_years ( train_data, 'Year_of_birth', 'Age' )
to_years ( train_data, 'Year_of_recruitment', 'Years_of_Work' )


def to_category ( data, feat_list  ):
    '''Funtion to transform object-type variables to categorical variables.'''
    for feature in feat_list:
        obj_feature = data[ feature ]
        data [ feature ] = obj_feature.astype ( 'category' )
        data [ feature ] = data[ feature ].cat.codes

# Transform object_type variables to categorical variables
feat_list = [   
                'EmployeeNo',
                'Division',
                'Qualification',
                'Gender',
                'Channel_of_Recruitment',
                'State_Of_Origin',
                'Foreign_schooled',
                'Marital_Status',
                'Past_Disciplinary_Action',
                'Previous_IntraDepartmental_Movement',
                'No_of_previous_employers',
            ]

to_category ( train_data, feat_list )



def drop_columns ( data, feat_to_drop_list ):
    '''
    Function to discard a list of unwanted data categories.
    '''
    for feature in feat_to_drop_list:
        data.drop ( feature, axis=1 )
# To be used later if the need arises... Possible colunms to drop are: State_of_origin, Marital_Status,


# Building the Model.
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
from mlxtend.classifier import StackingCVClassifier
from sklearn.metrics import accuracy_score
SEED = 0

numeric_transformer = Pipeline(
    steps=[
        ( 'imputer', SimpleImputer(strategy = 'mean') ),
        ( 'scaler', StandardScaler() )
    ]
)


rf_pipe = Pipeline(
    steps = [
        ( 'num', numeric_transformer ),
        ( 'classifier', RandomForestClassifier(
            n_estimators = 1000,
            min_samples_split = 4,
            min_samples_leaf = 1,
            max_features = 'auto',
            max_depth = 40,
            bootstrap = False,
            class_weight=None,
            criterion='gini',
            max_leaf_nodes=None,
            oob_score=False,
            verbose=0, 
            warm_start=False
        ) )
    ]
)


lg_pipe = Pipeline(
    steps = [
        ( 'num', numeric_transformer ),
        ( 'classifier', LGBMClassifier(
                            n_estimators=1000,
                            num_leaves=100,
                            verbose=0,
                            random_state=SEED) )
    ]
)

xgb_pipe = Pipeline(
    steps = [
        ( 'num', numeric_transformer ),
        ( 'xgb', xgb.XGBClassifier(
                    base_score=0.5,
                    booster='gbtree',
                    colsample_bylevel=1,
                    colsample_bynode=1,
                    colsample_bytree=1, 
                    gamma=0, 
                    learning_rate=0.1,
                    max_delta_step=0, 
                    max_depth=3,
                    min_child_weight=0.95, 
                    missing=None,
                    n_estimators=1000,
                    n_jobs=1,
                    nthread=None,
                    objective='binary:logistic', 
                    random_state=SEED,
                    reg_alpha=0, 
                    reg_lambda=1, 
                    scale_pos_weight=1,
                    seed=42, 
                    silent=None, 
                    subsample=1,
                    verbosity=1,
                    eval_metric='error'
                          ))
    ]
)


# Testing the pipelines...
X = train_data.drop( ['Promoted_or_Not'], axis=1 )
y = train_data['Promoted_or_Not']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=SEED )


_ = rf_pipe.fit(X_train, y_train)
y_pred_rf = rf_pipe.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print( f'RF Accuracy = { accuracy_rf }' )

_ = lg_pipe.fit(X_train, y_train)
y_pred_lg = lg_pipe.predict(X_test)
accuracy_lg = accuracy_score(y_test, y_pred_lg)
print( f'LG Accuracy = { accuracy_lg }' )

_ = xgb_pipe.fit(X_train, y_train)
y_pred_xgb = xgb_pipe.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print( f'XGB Accuracy = { accuracy_xgb }' ) # Highest Accuracy


# Try dropping columns...
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure( figsize=(12,10) )
correlation = train_data.corr()
sns.heatmap( correlation, annot=True, cmap=plt.cm.Reds )
plt.show()

correlation_target = abs( correlation[ 'Promoted_or_Not' ] )
relevant_features = correlation_target [ correlation_target > 0.03 ]
print ( relevant_features )

total_set = set(train_data.columns)
relevant_set = set(relevant_features.index)
irrelevant_features = list ( total_set - relevant_set )

drop_columns ( train_data, irrelevant_features )
# Go over testing the pipelines again...


# Create a Stacking Classifier.
stack = StackingCVClassifier(
                    classifiers=[ rf_pipe, lg_pipe, xgb_pipe ],
                    meta_classifier=rf_pipe,
                    cv=5,
                    use_probas=True,
                    use_features_in_secondary=True,
                    verbose=0,
                    random_state=SEED,
                    n_jobs=-1 )

_ = stack.fit(X_train, y_train)

y_pred_stack = stack.predict(X_test)
accuracy_stack = accuracy_score(y_test, y_pred_stack)
print( f'STACK Accuracy = { accuracy_stack }' ) # Same as XGBoost Results...


# Tuning the Model.
# n_estimators = [ int(x) for x in np.linspace(start = 200, stop = 2000, num = 10) ]
# max_features = [ 'auto', 'sqrt' ]
# max_depth = [ int(x) for x in np.linspace(10, 150, num = 15) ]
# max_depth.append ( None )
# min_samples_split = [ 2, 5, 10 ]
# min_samples_leaf = [ 1, 2, 4 ]
# bootstrap = [ True, False ]

# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# rf_random = RandomizedSearchCV ( estimator = RandomForestClassifier(random_state=SEED),
#                                  param_distributions = random_grid,
#                                  n_iter = 100,
#                                  cv = 3,
#                                  verbose=0,
#                                  random_state=SEED,
#                                  n_jobs = -1 )

# _ = rf_random.fit(X_train, y_train)
# y_pred = rf_random.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy) # If better, use its best_params_ in rf model above.

# print(rf_random.best_params_)


# # Create the parameter grid (for GridSearch) based on the results of random search 
# param_grid = {
#     'bootstrap': [False],
#     'max_depth': [40, 45],
#     'max_features': [2, 3],
#     'min_samples_leaf': [1],
#     'min_samples_split': [3, 4],
#     'n_estimators': [800, 1000]
# }
# Create a based model
# rf_default = RandomForestClassifier()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rf_default,
#                            param_grid = param_grid, 
#                            cv = 3,
#                            n_jobs = -1,
#                            verbose = 1)

# _ = grid_search.fit(X_train, y_train)
# y_pred =grid_search.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy) # If better, use its best_params_ in rf model above.

# print(grid_search.best_params_)

# best_grid = grid_search.best_estimator_

# _ = best_grid.fit(X_train, y_train)
# y_pred = best_grid.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy) # If better, use its best_params_ in rf model above.

# Employ the best parameters in the random forest pipeline (rf_pipe) above.



# Prepare submission...
_ = xgb_pipe.fit(X, y)

to_years ( test_data, 'Year_of_birth', 'Age' )
to_years ( test_data, 'Year_of_recruitment', 'Years_of_Work' )
A = test_data [ 'EmployeeNo' ] # Preserve original format.
feat_list = [   'EmployeeNo',
                'Division',
                'Qualification','Gender',
                'Channel_of_Recruitment',
                'State_Of_Origin',
                'Foreign_schooled',
                'Marital_Status',
                'Past_Disciplinary_Action',
                'Previous_IntraDepartmental_Movement',
                'No_of_previous_employers',
            ]
to_category ( test_data, feat_list )
drop_columns ( test_data, irrelevant_features )

test_data['Promoted_or_Not'] = xgb_pipe.predict(test_data)
test_data [ 'EmployeeNo' ] = A

submission = test_data[['EmployeeNo', 'Promoted_or_Not']]
submission.to_csv('submission.csv', index=False)


# DEEP LEARNING APPROACH

# import tensorflow as tf
# from tensorflow import keras

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(20,)),
#     keras.layers.Dense(16, activation=tf.nn.relu),
# 	keras.layers.Dense(16, activation=tf.nn.relu),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid),
# ])

# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# model.fit(X_train, y_train, epochs=50, batch_size=1)

# test_loss, test_acc = model.evaluate(X_test, y_test)
# print('Test accuracy:', test_acc) # No improvement in result...