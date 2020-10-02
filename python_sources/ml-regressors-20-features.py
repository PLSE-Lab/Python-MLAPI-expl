import numpy as np
import pandas as pd 
import warnings
from time import time
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
import xgboost
import catboost
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# imputation function
def imputation_regression (df, package):

    if not df.isnull().values.any():
        print("No missing value cell to impute data!")
        return df
    
    naColumns = df.columns[df.isna().any()].tolist()
    print('   - Columns ' + str(naColumns) + ' will be filled using ' + package.__class__.__name__)
          
    for c in naColumns:
        
        tempDF = pd.DataFrame(df) # or tempDF = df.copy(deep = True)
        naColumnsOtherThanC = [x for x in naColumns if x != c]
        naColumnsStore = tempDF[naColumnsOtherThanC].values
        tempDF[naColumnsOtherThanC] = tempDF[naColumnsOtherThanC].fillna(df.mean()) # or maybe median
        
        train = tempDF[pd.notnull(tempDF[c])]    
        test = tempDF[pd.isnull(tempDF[c])]
        
        indices = train.index.tolist() + test.index.tolist()
        
        X_train = train.loc[:, train.columns != c]
        y_train = train[c]  
        X_test =  test.loc[:, test.columns != c]       
        y_train = y_train.astype(int)
                
        package.fit(X_train, y_train)
        y_pred = package.predict(X_test)
        test[c] = y_pred
        
        filledColumn = pd.concat([train[c], test[c]], ignore_index=True)
        filledColumnDF = pd.DataFrame({'Indices': indices, 'FilledColumn': filledColumn})
        filledColumnDF = filledColumnDF.sort_values('Indices')
        df[c] = filledColumnDF['FilledColumn'].values
        df[naColumnsOtherThanC] = naColumnsStore
        
    return df

# function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# function of grid search
def gridSearch(X, Y, model, param_grid):
    grid_search = GridSearchCV(model, param_grid=param_grid, 
                                        scoring=make_scorer(mae, greater_is_better=False))
    start = time()
    grid_search.fit(X, Y)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)
    return grid_search

# random forest grid paramenters and grid search
def random_forest_regressor(X, Y):
    model = ensemble.RandomForestRegressor()
    param_grid = {'n_estimators': [5, 10, 50, 100],
                  'max_features': [5, 10, 20],
                  'max_depth': [3, 8, None],
                  'min_samples_split': [0.01, 0.05],
                  'min_samples_leaf': [0.01, 0.05],
                  "criterion": ["mse", "mae"]}
    return gridSearch(X, Y, model, param_grid)

# extra tree regressor grid paramenters and grid search
def extra_trees_regressor(X, Y):
    model = ensemble.ExtraTreesRegressor()
    param_grid = {'n_estimators': [5, 10, 50, 100],
                  'max_features': [5, 10, 20],
                  'max_depth': [3, 8, None],
                  'min_samples_split': [0.01, 0.05],
                  'min_samples_leaf': [0.01, 0.05],
                  "criterion": ["mse", "mae"]}
    return gridSearch(X, Y, model, param_grid)
    
# gradient boosting grid paramenters and grid search
def gradient_boosting_regressor(X, Y):
    model = ensemble.GradientBoostingRegressor()
    param_grid = {'loss': ["ls", "lad", "huber"],
                  'learning_rate': [0.01, 0.1, 0.25, 0.5],
                  'n_estimators': [10, 100, 250],
                  'min_samples_split': [0.01, 0.05, 0.1],
                  'min_samples_leaf': [0.01, 0.05],
                  'max_depth' : [3, 8]}
    return gridSearch(X, Y, model, param_grid)

# xgb regressor grid paramenters and grid search
def xgb_regressor(X, Y):
    model = xgboost.XGBRegressor()
    param_grid = {'booster': ["gbtree", "gblinear", "dart"],
                  'learning_rate': [0.01, 0.1, 0.25, 0.5],
                  'n_estimators': [10, 100, 250],
                  'max_depth' : [3, 8],
                  'min_child_weight': [1, 3, 5]}
    gridSearch(X, Y, model, param_grid)

# xgb regressor grid paramenters and grid search
def catboost_regressor(X, Y):
    model = catboost.CatBoostRegressor()
    param_grid = {'learning_rate': [0.01, 0.1, 0.25],
                  'iterations': [500, 1000, 2000],
                  'l2_leaf_reg': [1, 3, 5]}
    gridSearch(X, Y, model, param_grid)
    
# categorize string columns
def categorize_string_values(df, threshold):
    column_list = list(df.columns)
    for col in column_list:
        if df[col].dtypes == object:
            unique_list = list(df[col].unique())
            for unique in unique_list:
                if unique is np.nan:
                    unique_list.remove(unique)
            df[col] = pd.Categorical(df[col], categories=unique_list).codes
            df[col] = df[col].replace(-1,np.nan)
        null_count = df[col].isnull().sum()
        if null_count > threshold:
            df = df.drop([col], axis=1)
    return df
 
def takeSecond(elem):
    return elem[1]   

# get variable importance with random forest
def variable_importance_with_random_forest(X, Y, maxFeature):
    model = ensemble.RandomForestRegressor()
    model.fit(X, Y)
    importance_list = []
    cols = []
    for name, importance in zip(list(train_X_processed.columns), model.feature_importances_):
        importance_list.append([name, importance])
    importance_list.sort(key=takeSecond, reverse=True)
    importance_list = importance_list[:maxFeature]
    for element in importance_list:
        cols.append(element[0])
    return cols

##################################################################################################        
##################################################################################################

# get data    
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape, test.shape)

# target and predictor variables
train_Y = train['SalePrice']
train_X = train.drop(['SalePrice'], axis=1)

# merge them to apply same procedure
df = pd.DataFrame(pd.concat([train_X, test], ignore_index = False))

# categorize string values and drop colums where number of missing values higher than a threshold
df = categorize_string_values(df, df.shape[0] * 0.10)
df = df.drop(['Id'], axis=1)
print(df.shape)

# apply imputation
df = imputation_regression(df, tree.DecisionTreeRegressor())
# print(df.isnull().sum()) 

# get train and test back
train_X_processed = df[:len(train_X.index)]
test_processed = df[len(train.index):]
print(train_X_processed.shape, test_processed.shape)

# select the best features
best_features = variable_importance_with_random_forest(train_X_processed, train_Y, 20)
train_X_processed = train_X_processed[best_features]
test_processed = test_processed[best_features]
print(train_X_processed.shape, test_processed.shape)

##################################################################################################        
##################################################################################################

# models
models = {}
train_error_list = []

# RANDOM FOREST
# run grid search on random forest regressor; will run once to get optimal parameters
# random_forest_regressor(train_X_processed, train_Y)
    # GridSearchCV took 431.78 seconds for 384 candidate parameter settings. 
    # Model with rank: 1 Mean validation score: -20011.464 (std: 843.256) 
    # Parameters: {'criterion': 'mse', 'max_depth': 8, 'max_features': 10, 
    # 'min_samples_leaf': 0.01, 'min_samples_split': 0.01, 'n_estimators': 100} 

# apply the best model
model = ensemble.RandomForestRegressor(n_estimators = 100, criterion = "mse", max_depth = 8, max_features = 10,
                                            min_samples_leaf = 0.01, min_samples_split = 0.01)
model.fit(train_X_processed, train_Y)

# predict   
prediction_train = np.asarray(model.predict(train_X_processed))
prediction_test = np.asarray(model.predict(test_processed))
models[model] = [prediction_train, prediction_test]

# EXTRA TREE REGRESSOR
# extra_trees_regressor(train_X_processed, train_Y)
    # GridSearchCV took 298.23 seconds for 384 candidate parameter settings. 
    # Model with rank: 1 Mean validation score: -20624.097 (std: 578.588) 
    # Parameters: {'criterion': 'mse', 'max_depth': None, 'max_features': 20, 
    # 'min_samples_leaf': 0.01, 'min_samples_split': 0.01, 'n_estimators': 100} 

# apply the best model
model = ensemble.ExtraTreesRegressor(n_estimators = 100, criterion = "mse", max_depth = None, max_features = 20,
                                            min_samples_leaf = 0.01, min_samples_split = 0.01)
model.fit(train_X_processed, train_Y)

# predict   
prediction_train = np.asarray(model.predict(train_X_processed))
prediction_test = np.asarray(model.predict(test_processed))
models[model] = [prediction_train, prediction_test]

# GRADIENT BOOSTING REGRESSOR
# gradient_boosting_regressor(train_X_processed, train_Y)
    # GridSearchCV took 436.12 seconds for 432 candidate parameter settings. 
    # Model with rank: 1 Mean validation score: -18276.227 (std: 619.086) 
    # Parameters: {'learning_rate': 0.1, 'loss': 'huber', 'max_depth': 3, 
    # 'min_samples_leaf': 0.01, 'min_samples_split': 0.1, 'n_estimators': 250} 

# apply the best model
model = ensemble.GradientBoostingRegressor(n_estimators = 250, loss = "huber", learning_rate=0.1,
                                                    max_depth=3, min_samples_leaf=0.01, min_samples_split=0.1)
model.fit(train_X_processed, train_Y)

# predict   
prediction_train = np.asarray(model.predict(train_X_processed))
prediction_test = np.asarray(model.predict(test_processed))
models[model] = [prediction_train, prediction_test]

# XGB REGRESSOR
# xgb_regressor(train_X_processed, train_Y)
    # GridSearchCV took 259.13 seconds for 216 candidate parameter settings. 
    # Model with rank: 1 Mean validation score: -18234.896 (std: 1101.998) 
    # Parameters: {'booster': 'dart', 'learning_rate': 0.1, 'max_depth': 3, 
    # 'min_child_weight': 1, 'n_estimators': 250}

# apply the best model
model = xgboost.XGBRegressor(n_estimators = 250, booster="dart", learning_rate=0.1,
                                                    max_depth=3, min_weight_child=1)
model.fit(train_X_processed, train_Y)

# predict   
prediction_train = np.asarray(model.predict(train_X_processed))
prediction_test = np.asarray(model.predict(test_processed))
models[model] = [prediction_train, prediction_test]

# CATBOOST REGRESSOR
# catboost_regressor(train_X_processed, train_Y)
    # GridSearchCV took 614.48 seconds for 27 candidate parameter settings. 
    # Model with rank: 1 Mean validation score: -19225.251 (std: 886.523) 
    # Parameters: {'iterations': 2000, 'l2_leaf_reg': 1, 'learning_rate': 0.01}

# apply the best model
model = catboost.CatBoostRegressor(learning_rate=0.01, iterations=5000, l2_leaf_reg=1)
model.fit(train_X_processed, train_Y)

# predict   
prediction_train = np.asarray(model.predict(train_X_processed))
prediction_test = np.asarray(model.predict(test_processed))
models[model] = [prediction_train, prediction_test]

# train_error
for name, pred in models.items():
    train_error_list.append([name, np.sqrt(mean_squared_error(np.log(train_Y), np.log(pred[0])))])
    
for error in train_error_list:
    print("The train error of model {} is = {}".format(error[0].__class__.__name__, error[1]))

# stacking
stack_train = pd.DataFrame()
stack_test = pd.DataFrame()
for name, pred in models.items():
    stack_train[name] = pred[0]
    stack_test[name] = pred[1]

model = linear_model.LinearRegression()
model.fit(stack_train, train_Y)
prediction_stack= np.asarray(model.predict(stack_test))

submission = pd.DataFrame({'Id': test.Id, 'SalePrice': prediction_stack})
submission.to_csv("stack.csv", sep=',', encoding='utf-8', index=False)