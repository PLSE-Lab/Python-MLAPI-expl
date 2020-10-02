# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd
from numpy import sort

# Modelling Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.grid_search import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Modelling Helpers
from sklearn.preprocessing import Imputer ,  scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

#xgboost
from xgboost import XGBClassifier , XGBRegressor 

# Configure visualisations
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

def get_data():
    # reading train data
    train = pd.read_csv('../input/train.csv')   
    # reading test data
    test = pd.read_csv('../input/test.csv')
    
    return train,test

def describe_missing_data(df):
    #missing data
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data

def dealing_missing_train_data(df,missing_data):
    #dealing with missing data
    df = df.drop((missing_data[missing_data['Total'] > 1]).index,1)
    df = df.drop(df.loc[df['Electrical'].isnull()].index)
    print("is there has no dealing missing data?",df.isnull().sum().max()) #just checking that there's no missing data missing...
 
    return df 

def delete_Outliers(df):
    #deleting points
#    df = df.sort_values(by = 'GrLivArea', ascending = False)[:2]
#    print(df[:2])
    df = df.drop(df[df['Id'] == 1299].index)
    df = df.drop(df[df['Id'] == 524].index)
    
    return df

def dealing_feature_normality(df,var):
    #applying log transformation
    df[var] = np.log(df[var])
    return df

def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )
    
def Spot_Check(X , y):
    models = []
    models.append(('LR', LinearRegression()))
    models.append(('LASSO', Lasso()))
    models.append(('EN', ElasticNet()))
    models.append(('KNN', KNeighborsRegressor()))
    models.append(('CART', DecisionTreeRegressor()))
    models.append(('SVR', SVR()))
    models.append(('AB', AdaBoostRegressor()))
    models.append(('GBM', GradientBoostingRegressor()))
    models.append(('RF', RandomForestRegressor()))
    models.append(('ET', ExtraTreesRegressor()))
    models.append(('XGBOOST', XGBRegressor()))
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'neg_mean_squared_error'
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=7)
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    
    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    plt.figure(figsize=(15,8))
    ax = plt.subplot()
    plt.boxplot(results)
    ax.set_xlabel(names)
    plt.show()

def Sync_field(train,test):
    # Use only common columns
    columns = []
    for col_a in train.columns.values:
        if col_a in test.columns.values:
            columns.append(col_a)
    train = train[columns]
    test = test[columns]
    
    return train,test

def category_to_num(train,test):
    # Label nominal variables to numbers
    columns = train.columns.values
    nom_numeric_cols = ['MSSubClass']
    dummy_train = []
    dummy_test = []
    for col in columns:
        # Only works for nominal data without a lot of factors
        if train[col].dtype.name == 'object' or col in nom_numeric_cols:
            dummy_train.append(pd.get_dummies(train[col].values.astype(str), col))
            dummy_train[-1].index = train.index
            dummy_test.append(pd.get_dummies(test[col].values.astype(str), col))
            dummy_test[-1].index = test.index
            del train[col]
            del test[col]
    train = pd.concat([train] + dummy_train, axis=1)
    test = pd.concat([test] + dummy_test, axis=1)
     
    return train,test

def other_dealing(df):
    df['HasBsmt'] = pd.Series(len(df['TotalBsmtSF']), index=df.index)
    df['HasBsmt'] = 0 
    df.loc[df['TotalBsmtSF']>0,'HasBsmt'] = 1
    #transform data
    df.loc[df['HasBsmt']==1,'TotalBsmtSF'] = np.log(df['TotalBsmtSF'])
    
    return df

def describe_more( df ):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels
    
def fill_missing_data(train,test,missing_data):
    missing_data = describe_missing_data(test)
    missing_field = (missing_data[missing_data['Total'] >= 1]).index
    for field in missing_field:
        freq_port = train[field].dropna().mode()[0]
        test[field].fillna(freq_port, inplace=True)
   
def quadratic(train,test,feature):
    train[feature+'2'] = train[feature]**2
    test[feature+'2'] = test[feature]**2
        
    return train,test
         
########################################Data Preparetion#######################################    
train,test = get_data()

missing_data = describe_missing_data(train)
train = dealing_missing_train_data(train,missing_data)

train = delete_Outliers(train)
#train = dealing_feature_normality(train,'GrLivArea')
#test = dealing_feature_normality(test,'GrLivArea')
#
#train = other_dealing(train)
#freq_port = train['TotalBsmtSF'].dropna().mode()[0]
#test['TotalBsmtSF'].fillna(freq_port, inplace=True)
#test = other_dealing(test)

targets = train.SalePrice
train.drop('SalePrice', 1, inplace=True)
targets = np.log(targets)

train = train.drop('Id', axis=1)
test_df = test
test = test.drop('Id', axis=1)

train,test = Sync_field(train,test)
#train,test = quadratic(train,test,'OverallQual')
#train,test = quadratic(train,test,'YearBuilt')
#train,test = quadratic(train,test,'YearRemodAdd')
#train,test = quadratic(train,test,'TotalBsmtSF')
#train,test = quadratic(train,test,'2ndFlrSF')
#train,test = quadratic(train,test,'GrLivArea')
#train,test = quadratic(train,test,'Neighborhood')
#train,test = quadratic(train,test,'RoofMatl')

train,test = category_to_num(train,test)
train,test = Sync_field(train,test)


#######################################Spot-Check Algorithms###################################
#plot_variable_importance(train,targets)
#Spot_Check(train,targets)
#search_feature(train,targets)
#train,test = select_feature(train,targets,test,thresh=0.004)
#train,test = data_transform_MinMaxScaler(train,test)
#train,test = data_transform_StandardScaler(train,test)
#train,test = data_transform_Normalizer(train,test)

#df = pd.DataFrame(train)
#plot_correlation_map(df)

#######################################Hyperparameters tuning###################################
# turn run_gs to True if you want to run the gridsearch again.
run_gs = False

if run_gs:
    parameter_grid = {
                 'n_estimators' : [500,600,800,1000],
                 'max_depth' : [3],
                 'learning_rate': [0.1],
                 'subsample': [0.6],
                 'colsample_bytree': [0.2,0.3,0.4,0.5],
                 'colsample_bylevel': [0.6],
                 }
    xgb = XGBRegressor()
    cross_validation = StratifiedKFold(targets, n_folds=5)

    grid_search = GridSearchCV(xgb,
                               scoring='neg_mean_squared_error',
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(train, targets)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best neg_mean_squared_error: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
else: 
    parameters = {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.1, 
                  'subsample': 0.6, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.6}
    
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(train, targets, test_size=test_size, random_state=seed)
    model = XGBRegressor(**parameters)
#    model.fit(X_train, y_train ,verbose=False)
    model.fit(train, targets)

predictions = model.predict(X_test)
print("----------fianl xgboost score:",mean_squared_error(y_test, predictions),"train shape:",train.shape)

################################################################################################
predictions = np.exp(model.predict(test))
print("predictions shape:",predictions.shape)
print(predictions)
submission = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": predictions
    })
submission.to_csv('house_pred.csv', index=False)