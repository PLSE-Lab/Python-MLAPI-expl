#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))
import seaborn as sns
sns.set(style="ticks")
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
from numpy import median
pd.options.display.max_rows = 200
import sklearn.preprocessing as preprocessing
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train_copy = train.copy() 
print(train.info())
print(train.isnull().sum())


# In[ ]:


# https://github.com/PacktPublishing/Python-Feature-Engineering-Cookbook/blob/master/Chapter01/DataPrep_Titanic.ipynb
# http://www.datasciencemadesimple.com/return-first-n-character-from-left-of-column-in-pandas-python/
train['Cabin'].fillna('0').str[:1].value_counts()


# In[ ]:


train['modCabin'] = train['Cabin'].fillna('0').str[:1]
test['modCabin'] = train['Cabin'].fillna('0').str[:1]


# In[ ]:


# import pandas_profiling as pp
# pp.ProfileReport(train)


# In[ ]:


#http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False, dtype=int)
res = vec.fit_transform(train[['modCabin', 'Pclass', 'Age', 'SibSp', 'Parch','Embarked', 'Sex']].fillna('NA').to_dict( orient='records'))
res


# In[ ]:


#https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.04-Feature-Engineering.ipynb
vec.get_feature_names()


# In[ ]:


df = pd.DataFrame(res, columns=vec.get_feature_names())
df.head()


# In[ ]:


#https://towardsdatascience.com/interactive-controls-for-jupyter-notebooks-f5c94829aee6
#https://github.com/WillKoehrsen/Data-Analysis/blob/master/widgets/Widgets-Overview.ipynb
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
@interact
def correlations(column1=list(df.select_dtypes('number').columns), 
                 column2=list(df.select_dtypes('number').columns)):
    print(f"Correlation: {df[column1].corr(df[column2])}")


# In[ ]:


train.head()


# In[ ]:


_=pd.qcut(train.Age,q= [0,.25, .5, .75,1] ).value_counts().plot(kind='bar')


# In[ ]:


_ = sns.distplot(train.Age.dropna())


# In[ ]:


from sklearn import ensemble
from sklearn.model_selection import train_test_split
gbes = ensemble.GradientBoostingClassifier(n_estimators=400,
                                               validation_fraction=0.2,
                                               n_iter_no_change=5, tol=0.01,
                                               random_state=0)
X_train, X_test, y_train, y_test = train_test_split(train.Age, train.Survived, test_size=0.2,
                                                        random_state=0)
gbes.fit(X_train.fillna(0).values.reshape(-1, 1),y_train)
gbes.score(X_test.fillna(0).values.reshape(-1, 1),y_test)


# In[ ]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df, train.Survived, test_size=0.2,
                                                        random_state=0)
gbes.fit(X_train,y_train)
gbes.score(X_test,y_test)


# In[ ]:


print(cross_val_score(gbes, df, train.Survived, cv=3))


# In[ ]:


test_transf = vec.transform(test[['modCabin', 'Pclass', 'Age', 'SibSp', 'Parch','Embarked', 'Sex']].fillna('NA').to_dict( orient='records'))
df_test = pd.DataFrame(test_transf, columns=vec.get_feature_names())
df_test.head()


# In[ ]:


gbespred = gbes.predict(df_test)
out_df = pd.DataFrame({"PassengerId":test["PassengerId"].values})
out_df['Survived'] = gbespred

out_df.to_csv("submission.csv", index=False)
print('out_df.to_csv')
print(out_df.head())
out_df.Survived.value_counts()


# In[ ]:


train.groupby('Survived')['Age','Fare'].describe().T


# In[ ]:


train.groupby(['Survived', 'Embarked'])['Age','Fare'].describe()


# In[ ]:


train[['Age','Fare']].head()


# In[ ]:


#https://wellsr.com/python/python-pandas-density-plot-from-a-dataframe/
pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(train[['Age','Fare']].dropna()),columns=['Age', 'Fare']).plot.kde()


# In[ ]:


preprocessing.KBinsDiscretizer(n_bins=[2,2,2,2], encode='ordinal').fit_transform(train.loc[:,['Age','Fare','SibSp','Parch']].dropna())


# In[ ]:


train.Embarked.value_counts()


# In[ ]:


train.loc[train.Embarked.isnull(),'Embarked']='Q'


# In[ ]:


train['IsCabinNull'] =  train.Cabin.isnull().astype(int)
test['IsCabinNull'] =  test.Cabin.isnull().astype(int)


# In[ ]:


print(train.Age.median())
print(test.Age.median())


# In[ ]:


train.loc[train.Age.isnull(), 'Age'] = 28
test.loc[test.Age.isnull(), 'Age'] = 28


# In[ ]:


print((train.Fare == 0).sum())
print((test.Fare == 0).sum())


# In[ ]:


print(train.Fare.median())
print(test.Fare.median())


# In[ ]:


train.loc[(train.Fare==0), 'Fare'] = train.Fare.median()
test.loc[(test.Fare==0), 'Fare'] = test.Fare.median()
test.loc[(test.Fare.isnull()), 'Fare'] = test.Fare.median()


# In[ ]:


train['age_log'] = np.log(train['Age'])
test['age_log'] = np.log(test['Age'])
train['fare_log'] = np.log(train['Fare'])
test['fare_log'] = np.log(test['Fare'])


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train[['Age']])
train['age_scale'] = scaler.transform(train[['Age']])
test['age_scale'] = scaler.transform(test[['Age']])
scaler = StandardScaler()
scaler.fit(train[['Fare']])
train['fare_scale'] = scaler.transform(train[['Fare']])
test['fare_scale'] = scaler.transform(test[['Fare']])


# In[ ]:


import copy
train_objs_num = len(train)
dataset = pd.concat(objs=[train, test],sort=False)
dataset['ngroup'] = dataset.groupby(['Pclass','Sex']).ngroup()
dataset = dataset['ngroup']
train['ngroup'] = copy.copy(dataset[:train_objs_num])
test['ngroup'] = copy.copy(dataset[train_objs_num:])


# In[ ]:


#https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
train['SibParchSum'] = train['SibSp'] + train['Parch']
test['SibParchSum'] = test['SibSp'] + test['Parch']
train['IsWithFamily'] = (train['SibParchSum']>0).astype(int)
test['IsWithFamily'] = (test['SibParchSum']>0).astype(int)
from sklearn.preprocessing import KBinsDiscretizer
binner = KBinsDiscretizer(encode='ordinal')
binner.fit(train[['Fare']])
train['Fare_bins'] = binner.transform(train[['Fare']])
test['Fare_bins'] = binner.transform(test[['Fare']])
binner = KBinsDiscretizer(encode='ordinal')
binner.fit(train[['Age']])
train['age_bins'] = binner.transform(train[['Age']])
test['age_bins'] = binner.transform(test[['Age']])


# In[ ]:


from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
encoder.fit(train[['Sex']])
train['sex_integer'] = encoder.transform(train[['Sex']])
test['sex_integer'] = encoder.transform(test[['Sex']])
encoder = OrdinalEncoder()
encoder.fit(train[['Embarked']])
train['embarked_integer'] = encoder.transform(train[['Embarked']])
test['embarked_integer'] = encoder.transform(test[['Embarked']])


# In[ ]:


train_after_normalizer = preprocessing.Normalizer().fit_transform(train[['Age', 'Fare','SibSp', 'Parch','sex_integer','embarked_integer','Pclass']])


# In[ ]:


train_after_normalizer = pd.DataFrame(train_after_normalizer, columns = ['Age', 'Fare','SibSp', 'Parch','sex_integer','embarked_integer','Pclass'])


# In[ ]:


print(train.groupby('Pclass')['Fare'].mean())
print(test.groupby('Pclass')['Fare'].mean())
print(test[test.Fare.isnull()])


# In[ ]:


test.loc[test.Fare.isnull(),'Fare'] = 13


# In[ ]:


test.loc[test.PassengerId==1044]


# In[ ]:


#https://www.kaggle.com/diegosch/classifier-evaluation-using-confusion-matrix
_=train.groupby(['Pclass','Sex'])['Age'].plot(kind='hist', legend=True, alpha=0.8)


# In[ ]:


#https://www.kaggle.com/diegosch/classifier-evaluation-using-confusion-matrix
_=train.groupby(['SibSp'])['Age'].plot(kind='hist', legend=True, alpha=0.8)


# In[ ]:


#https://www.kaggle.com/diegosch/classifier-evaluation-using-confusion-matrix
_=train.groupby(['SibSp','Sex'])['Age'].plot(kind='hist', legend=True, alpha=0.8)


# In[ ]:


#https://www.kaggle.com/diegosch/classifier-evaluation-using-confusion-matrix
_=train.groupby(['Parch'])['Age'].plot(kind='hist', legend=True, alpha=0.8)


# In[ ]:


#https://www.kaggle.com/diegosch/classifier-evaluation-using-confusion-matrix
_=train.groupby(['Parch','Sex'])['Age'].plot(kind='hist', legend=True, alpha=0.8)


# In[ ]:


test.groupby(['Pclass','Sex'])['Age'].mean()


# In[ ]:


sns.countplot(x='Survived', hue='Sex', data=train)


# In[ ]:


sns.countplot(x='Survived', hue='Embarked', data=train)


# In[ ]:


sns.countplot(x='Survived', hue='Pclass', data=train)


# In[ ]:


sns.catplot(x='Survived', hue='Sex', col = 'Pclass', kind='count',data=train)


# In[ ]:


#https://www.kaggle.com/singh001avinash/nyc-flight-data-eda
#https://seaborn.pydata.org/generated/seaborn.catplot.html
sns.catplot(x="Sex", y="Fare", hue="Survived", data=train)


# In[ ]:


sns.catplot(x="Survived", y="Age", hue="Sex", data=train, kind='box')


# In[ ]:


sns.countplot(test.Sex)


# In[ ]:


#https://www.kaggle.com/tanetboss/starter-guide-preprocessing-randomforest
train['Survived'].value_counts(sort = False)


# In[ ]:


#https://www.kaggle.com/tanetboss/starter-guide-preprocessing-randomforest
#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(5)


# In[ ]:


total = test.isnull().sum().sort_values(ascending=False)
percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
missing_test_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_test_data.head(5)


# In[ ]:


#https://stackoverflow.com/questions/30503321/finding-count-of-distinct-elements-in-dataframe-in-each-column
train.nunique()


# In[ ]:


y_train = train.Survived


# In[ ]:


train.groupby(train.Cabin.isnull(), as_index=False).size()


# In[ ]:


train[['Pclass','Age','Cabin']].head()


# In[ ]:


train['SibSp'].value_counts(sort=False)


# In[ ]:


train[['SibSp', 'Survived']].groupby('SibSp')['Survived'].sum()


# In[ ]:


(train[['SibSp', 'Survived']].groupby('SibSp')['Survived'].sum()/train['SibSp'].value_counts(sort=False))*100


# In[ ]:


train.Parch.value_counts(sort=False)


# In[ ]:


train[['Parch', 'Survived']].groupby('Parch')['Survived'].sum()


# In[ ]:


(train[['Parch', 'Survived']].groupby('Parch')['Survived'].sum()/train.Parch.value_counts(sort=False))*100


# In[ ]:


(train[['Pclass', 'Survived']].groupby('Pclass')['Survived'].sum()/train.Pclass.value_counts(sort=False))*100


# In[ ]:


#train.groupby('Embarked').corr()


# In[ ]:


#https://www.kaggle.com/kikexclusive/curiosity-didn-t-kill-the-cat-all-in-one
from IPython.display import display, HTML
display(train.describe(include="all").T)


# In[ ]:


#https://stackoverflow.com/questions/27424178/faster-way-to-remove-outliers-by-group-in-large-pandas-dataframe
#https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.mstats.winsorize.html
#https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781785282287/4/ch04lvl1sec54/winsorizing-data
#https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba
from scipy.stats import mstats

#train['Fare'] = mstats.winsorize(train['Fare'], limits = (0.05,0.25))


# In[ ]:


#train['Fare'].describe()


# In[ ]:


display(test.describe(include="all").T)


# In[ ]:


#test['Fare'] = mstats.winsorize(test['Fare'], limits = (0.05,0.25))


# In[ ]:


train.info()


# In[ ]:


train.Embarked.value_counts()


# In[ ]:


#https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue
const_cols = [c for c in train.columns if train[c].nunique(dropna=False)==1 ]
const_cols


# In[ ]:


#https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue
train.groupby('Pclass')['Age'].agg(['size', 'count', 'mean'])


# In[ ]:


train.describe()


# In[ ]:


#https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue
train.groupby('Sex')['Age'].agg(['size', 'count', 'mean'])


# In[ ]:


#https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue
train.groupby('Embarked')['Age'].agg(['size', 'count', 'mean'])


# In[ ]:


#https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

np.random.seed(2019)


numeric_features = ['Age', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['Embarked', 'Sex', 'Pclass']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='lbfgs'))])


Xt_train, Xt_test, yt_train, yt_test = train_test_split(train, y_train,stratify=y_train, test_size=0.3)

clf.fit(Xt_train, yt_train)
print("model score: %.3f" % clf.score(Xt_test, yt_test))
print(cross_val_score(clf, train, y_train, cv=5))


# In[ ]:


#features= ['Age','SibSp','Parch', 'Fare', 'Pclass','Sex','Cabin','Embarked']
features= ['Age', 'Fare','SibSp', 'Parch','Sex','Embarked','Pclass']
#num_features = ['Age','SibSp','Parch', 'Fare']
num_features = ['SibSp','Parch']
num_features_to_scale = ['Age', 'Fare']
cat_features = ['Sex', 'Embarked','Pclass']
train_test_concat = pd.concat([train[features],test[features]])


# In[ ]:


train_test_concat.info()


# In[ ]:


#https://www.kaggle.com/moghazy/eda-for-iris-dataset-with-svm-and-pca
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_test_concat[num_features_to_scale])


# In[ ]:



train_scaled = train[num_features_to_scale].copy()
train_scaled = scaler.transform(train_scaled)
train_scaled = pd.DataFrame(train_scaled,columns=num_features_to_scale)
train_scaled = pd.concat([train_scaled,train[cat_features],train[num_features]],axis=1)
print(train_scaled.info())


# In[ ]:



test_scaled = test[num_features_to_scale].copy()
test_scaled = scaler.transform(test_scaled)
test_scaled = pd.DataFrame(test_scaled,columns=num_features_to_scale)
test_scaled = pd.concat([test_scaled,test[cat_features],test[num_features]],axis=1)
print(test_scaled.info())


# In[ ]:


train_scaled.describe()


# In[ ]:


test_scaled.describe()


# In[ ]:


#https://stackoverflow.com/questions/53254292/pandas-simpleimputer-preserve-datatypes

from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(strategy='most_frequent')
my_imputer.fit(pd.concat([train_scaled,test_scaled]))
data_with_imputed_values = pd.DataFrame(
     my_imputer.transform(train_scaled), columns=train_scaled.columns
).astype(train_scaled.dtypes.to_dict())

test_data_with_imputed_values = pd.DataFrame(
     my_imputer.transform(test_scaled), columns=test_scaled.columns
).astype(test_scaled.dtypes.to_dict())

print(data_with_imputed_values.info())
print(test_data_with_imputed_values.info())


# In[ ]:


#https://markhneedham.com/blog/2017/07/05/pandasscikit-learn-get_dummies-testtrain-sets-valueerror-shapes-not-aligned/
all_data = pd.concat([data_with_imputed_values,test_data_with_imputed_values])
#https://stackoverflow.com/questions/47537823/futurewarning-specifying-categories-or-ordered-in-astype-is-deprecated

for column in all_data.select_dtypes(include=[np.object]).columns:
    data_with_imputed_values[column] = pd.Categorical(data_with_imputed_values[column], categories = all_data[column].unique())
    test_data_with_imputed_values[column] = pd.Categorical(test_data_with_imputed_values[column], categories = all_data[column].unique())


# In[ ]:


#data_with_imputed_values.info()


# In[ ]:


data_with_imputed_values = pd.get_dummies(data_with_imputed_values,drop_first=True)
test_data_with_imputed_values = pd.get_dummies(test_data_with_imputed_values,drop_first=True)


# In[ ]:


data_with_imputed_values.info()


# In[ ]:


test_data_with_imputed_values.info()


# In[ ]:


#https://www.kaggle.com/tanetboss/starter-guide-preprocessing-randomforest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


param_grid  = { 
                'n_estimators' : [500,1200],
               # 'min_samples_split': [2,5,10,15,100],
               # 'min_samples_leaf': [1,2,5,10],
                'max_depth': range(1,5,2),
                'max_features' : ('log2', 'sqrt'),
                'class_weight':[{1: w} for w in [1,1.5]]
              }

GridRF = GridSearchCV(RandomForestClassifier(random_state=15), param_grid)

GridRF.fit(data_with_imputed_values, y_train)
#RF_preds = GridRF.predict_proba(X_test)[:, 1]
#RF_performance = roc_auc_score(Y_test, RF_preds)

print(
    #'DecisionTree: Area under the ROC curve = {}'.format(RF_performance)
     "\nBest parameters \n" + str(GridRF.best_params_))


rf = RandomForestClassifier(random_state=15,**GridRF.best_params_)
rf.fit(data_with_imputed_values, y_train)

Rfclf_fea = pd.DataFrame(rf.feature_importances_)
#print(Rfclf_fea)
Rfclf_fea["Feature"] = list(data_with_imputed_values.columns) 
Rfclf_fea.sort_values(by=0, ascending=False).head(10)


# In[ ]:


cross_val_score(rf, data_with_imputed_values, y_train,cv=5 ) 


# In[ ]:


print(type(data_with_imputed_values))
print(data_with_imputed_values.shape)


# In[ ]:


one = OneHotEncoder(handle_unknown='ignore')
simple = SimpleImputer(strategy='constant', fill_value='missing')
pre = one.fit_transform(simple.fit_transform(train[['Embarked', 'Sex', 'Pclass']]))


# In[ ]:


one.categories_


# In[ ]:


pre.shape


# In[ ]:


data_with_imputed_values.columns


# https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python

# In[ ]:


#https://stackoverflow.com/questions/19071199/pandas-dataframe-drop-columns-whose-name-contains-a-specific-string
def drop_cols_with_str(df,str_v):
    return df[df.columns.drop(list(df.filter(regex=str_v)))]
data_with_imputed_values = drop_cols_with_str(data_with_imputed_values,'Cabin_')  
#data_with_imputed_values = data_with_imputed_values.drop('Sex_female', axis=1)
#data_with_imputed_values = data_with_imputed_values.drop('Embarked_Q', axis=1)
test_data_with_imputed_values = drop_cols_with_str(test_data_with_imputed_values,'Cabin_')  
#test_data_with_imputed_values = test_data_with_imputed_values.drop('Sex_female', axis=1)
#test_data_with_imputed_values = test_data_with_imputed_values.drop('Embarked_Q', axis=1)


# https://www.kaggle.com/tejasrinivas/xgb-baseline-comments-classification

# In[ ]:



from sklearn.model_selection import train_test_split
import xgboost as xgb

#data_with_imputed_values_wo_Fare = data_with_imputed_values.drop('Fare', axis=1)
X_t, X_v, y_t, y_v = train_test_split(data_with_imputed_values,y_train, stratify=y_train, test_size=0.2, random_state=2019)
#X_t, X_v, y_t, y_v = train_test_split(data_with_imputed_values_wo_Fare,y_train, test_size=0.2, random_state=2019)


def runXGB(X_t, X_v, y_t, y_v, feature_names=None, seed_val=2017, num_rounds=200):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['eval_metric'] = 'auc'
    param['min_child_weight'] = 1 #3
    param['subsample'] = 0.5
   # param['colsample_bytree'] = 0.5
    param['seed'] = seed_val
   # param['max_delta_step'] = 8
#     param['objective'] = 'binary:logistic'
#     param['eta'] = 0.1
#     param['max_depth'] = 6
#     param['silent'] = 1
#     param['eval_metric'] = 'auc'
#     param['min_child_weight'] = 1
#     param['subsample'] = 0.5
#     param['colsample_bytree'] = 0.5
#     param['seed'] = seed_val
    
    
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(X_t, label=y_t)


    xgtest = xgb.DMatrix(X_v, label=y_v)
    watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    #model = xgb.train(plst, xgtrain, num_rounds, watchlist)
    return model    

model = runXGB(X_t, X_v, y_t, y_v)


# In[ ]:


preds = model.predict(xgb.DMatrix(data_with_imputed_values))


# In[ ]:


from sklearn.metrics import confusion_matrix
pred_t = model.predict(xgb.DMatrix(data_with_imputed_values))
train_copy['Prediction'] = (pred_t > 0.4).astype(int)
tn, fp, fn, tp = confusion_matrix(train_copy['Survived'], train_copy['Prediction']).ravel()
(tn, fp, fn, tp)


# In[ ]:


xgb.plot_importance(model)


# In[ ]:


# print('Predict the probabilities based on features in the test set')
# pred = model.predict(xgb.DMatrix(test_data_with_imputed_values))
# #pred = clf.predict(test_data_with_imputed_values)
# out_df = pd.DataFrame({"PassengerId":test["PassengerId"].values})
# out_df['Survived'] = (pred>0.5).astype(int)

# out_df.to_csv("submission.csv", index=False)
# print('out_df.to_csv')
# print(out_df.head())


# In[ ]:


def runXGBcv(X, y, feature_names=None, seed_val=2017, num_rounds=100):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.05
    param['max_depth'] = 6
    param['silent'] = 1
    param['eval_metric'] = 'auc'
    param['min_child_weight'] = 3
    param['subsample'] = 0.5
    param['colsample_bytree'] = 0.5
    param['seed'] = seed_val
  #  param['max_delta_step'] = 8
#     param['objective'] = 'binary:logistic'
#     param['eta'] = 0.1
#     param['max_depth'] = 6
#     param['silent'] = 1
#     param['eval_metric'] = 'auc'
#     param['min_child_weight'] = 1
#     param['subsample'] = 0.5
#     param['colsample_bytree'] = 0.5
#     param['seed'] = seed_val
    
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(X, label=y)

    result = xgb.cv(plst, xgtrain, num_rounds, nfold=5,stratified=True, 
           folds=None, metrics=(), obj=None, feval=None, maximize=False, 
           early_stopping_rounds=20, fpreproc=None, as_pandas=True, 
           verbose_eval=None, show_stdv=True, seed=2019, callbacks=None, shuffle=True)
    return result    

result = runXGBcv(data_with_imputed_values, y_train)
#result


# In[ ]:


data_with_imputed_values.describe()


# https://www.kaggle.com/cast42/xg-cv

# In[ ]:


from sklearn.metrics import roc_auc_score
clf = xgb.XGBClassifier(
                max_depth = 7,
                n_estimators=700,
                learning_rate=0.1, 
                nthread=4,
                subsample=1.0,
                colsample_bytree=0.5,
                min_child_weight = 3,
               # max_delta_step = 7,
                seed=1301)

#                 max_depth = 7,
#                 n_estimators=700,
#                 learning_rate=0.1,  
#                 nthread=4,
#                 subsample=0.4, #1.0
#                 colsample_bytree=0.4,
#                 min_child_weight = 5,
#                 seed=1301)
                
                
xgb_param = clf.get_xgb_params()

cvresult = xgb.cv(xgb_param, xgb.DMatrix(data_with_imputed_values,y_train), 
                  num_boost_round=400, nfold=15, metrics=['auc'],
                  early_stopping_rounds=30, 
                  stratified=True,
                  as_pandas = True, show_stdv= True, seed=1301)
print('Best number of trees = {}'.format(cvresult.shape[0]))
clf.set_params(n_estimators=cvresult.shape[0])

clf.fit(data_with_imputed_values, y_train, eval_metric='auc')
print('Overall AUC:', roc_auc_score(y_train, clf.predict_proba(data_with_imputed_values)[:,1]))
# print('Predict the probabilities based on features in the train set')
# pred = clf.predict_proba(data_with_imputed_values, ntree_limit=cvresult.shape[0])

#pred = clf.predict(test_data_with_imputed_values)
#pred = clf.predict(test_data_with_imputed_values)


# In[ ]:


#cvresult


# In[ ]:


clf.feature_importances_


# In[ ]:


xgb.plot_importance(clf)


# In[ ]:


#cv_score = cross_val_score(clf, data_with_imputed_values, y_train,cv=10 ) 


# In[ ]:


#https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/examples/ensemble/plot_comparison_ensemble_classifier.py
from imblearn.ensemble import BalancedRandomForestClassifier
bclf = BalancedRandomForestClassifier(n_estimators=50, random_state=5)
X_t, X_v, y_t, y_v = train_test_split(data_with_imputed_values,y_train, stratify=y_train, test_size=0.1, random_state=0)

bclf.fit(X_t, y_t)
from sklearn.metrics import classification_report,f1_score
print(classification_report(y_v, bclf.predict(X_v)))




# In[ ]:


bclf.fit(data_with_imputed_values, y_train)
print('Predict based on features in the test set')
pred_bclf = bclf.predict(test_data_with_imputed_values)
##pred = clf.predict(test_data_with_imputed_values)
# out_df = pd.DataFrame({"PassengerId":test["PassengerId"].values})
# out_df['Survived'] = pred_bclf

# out_df.to_csv("submission.csv", index=False)
# print('out_df.to_csv')
# print(out_df.head())


# In[ ]:


data_with_imputed_values.columns


# In[ ]:


#https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
from bayes_opt import BayesianOptimization
X_t, X_v, y_t, y_v = train_test_split(data_with_imputed_values,y_train, stratify=y_train, test_size=0.1, random_state=0)

dtrain = xgb.DMatrix(X_t, label=y_t)
dtest = xgb.DMatrix(X_v, label=y_v)
def xgb_evaluate(max_depth, gamma, colsample_bytree):
    params = {'eval_metric': 'auc',
              'max_depth': int(max_depth),
              'subsample': 0.8,
              'eta': 0.1,
              'gamma': gamma,
              'colsample_bytree': colsample_bytree}
    # Used around 1000 boosting rounds in the full model
    cv_result = xgb.cv(params, dtrain, num_boost_round=100, nfold=3)    
    
    # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
    return -1.0 * cv_result['test-auc-mean'].iloc[-1]

xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 7), 
                                             'gamma': (0, 1),
                                             'colsample_bytree': (0.3, 0.9)})
# Use the expected improvement acquisition function to handle negative numbers
# Optimally needs quite a few more initiation points and number of iterations
xgb_bo.maximize(init_points=3, n_iter=5, acq='ei')
#params_xgb = xgb_bo.best_params_


# In[ ]:


params_xgb = xgb_bo.max['params']
params_xgb['max_depth'] = int(params_xgb['max_depth'])
params_xgb


# In[ ]:



#params_xgb['max_depth'] = int(params['max_depth'])
model2 = xgb.train(params_xgb, dtrain, num_boost_round=250)

# Predict on testing and training set
y_test_pred = model2.predict(dtest)>0.5
y_train_pred = model2.predict(dtrain)>0.5

# Report testing and training RMSE
print(classification_report(y_v, y_test_pred))
print(classification_report(y_t, y_train_pred))
model3 = xgb.train(params_xgb, xgb.DMatrix(data_with_imputed_values, label=y_train), num_boost_round=250)
xgbpred = (model3.predict(xgb.DMatrix(test_data_with_imputed_values)) > 0.5).astype(int)
out_df = pd.DataFrame({"PassengerId":test["PassengerId"].values})
out_df['Survived'] = xgbpred

# out_df.to_csv("submission.csv", index=False)
# print('out_df.to_csv')
print(out_df.head())
out_df.Survived.value_counts()


# In[ ]:


#https://www.kaggle.com/nicapotato/simple-catboost
#https://effectiveml.com/using-grid-search-to-optimise-catboost-parameters.html
#https://github.com/catboost/tutorials/blob/master/competition_examples/mlbootcamp_v_tutorial.ipynb
from catboost import CatBoostClassifier
cb_model = CatBoostClassifier(
#                              iterations=700,
#                              learning_rate=0.02,
#                              depth=12,
#                              eval_metric='AUC',
#                              random_seed = 23,
#                              bagging_temperature = 0.2,
#                              od_type='Iter',
#                              #metric_period = 75,
#                              od_wait=100
)
cb_model.fit(X_t, y_t,
             eval_set=(X_v,y_v),
             #cat_features=[2,3,4,5,6,7],
             use_best_model=True,
             )
from sklearn.metrics import classification_report,f1_score
print(classification_report(y_v, cb_model.predict(X_v)))

print(cb_model.get_params())
cat_params = cb_model.get_params()

cb_model = CatBoostClassifier(**cat_params)
cb_model.fit(data_with_imputed_values, y_train,
            # cat_features=[2,3,4,5,6,7],
             
             )



catpred = cb_model.predict(test_data_with_imputed_values)
# out_df = pd.DataFrame({"PassengerId":test["PassengerId"].values})
# out_df['Survived'] = catpred

# out_df.to_csv("submission.csv", index=False)
# print('out_df.to_csv')
# print(out_df.head())
# out_df.Survived.value_counts()


# In[ ]:


#https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_examples.py
# from sklearn.model_selection import GridSearchCV
# search = GridSearchCV(clf,
#                    {'max_depth': [6,7],      #[2,4,5,6,7]
#                     'n_estimators': [100,200,300], 
#                     'learning_rate': [0.08, 0.1],
#                     'min_child_weight':[2,3,4],
#                     'subsample':[0.5,], #[0.1,0.3,0.4,0.5]
#                     'colsample_bytree':[0.4,], #[0.1,0.3,0.4,0.5]
#                     'gamma':[0,0.1,0.2]},
#                    verbose=1,refit=True)
# search.fit(data_with_imputed_values, y_train)
# print(search.best_score_)
# print(search.best_params_)


# In[ ]:


#https://datascience.stackexchange.com/questions/19882/xgboost-how-to-use-feature-importances-with-xgbregressor
clf.get_booster().get_score(importance_type='weight')


# In[ ]:


xgb.plot_importance(clf)


# In[ ]:


#clf.get_booster().get_split_value_histogram('Age',as_pandas=True)


# In[ ]:


from sklearn.metrics import confusion_matrix
pred_t = clf.predict(data_with_imputed_values, ntree_limit=cvresult.shape[0])
train_copy['Prediction'] = pred_t
tn, fp, fn, tp = confusion_matrix(train_copy['Survived'], train_copy['Prediction']).ravel()
(tn, fp, fn, tp)


# In[ ]:


preds_proba = clf.predict_proba(data_with_imputed_values, ntree_limit=cvresult.shape[0])
train_copy['Prediction'] = preds_proba[:,1]


# In[ ]:


preds_proba


# In[ ]:


train_copy.columns


# In[ ]:


train_copy.Pclass.value_counts()


# In[ ]:


sns.swarmplot(x="Pclass", y="Prediction", hue="Survived",
               data=train_copy)
#https://stackoverflow.com/questions/30490740/move-legend-outside-figure-in-seaborn-tsplot
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


sns.swarmplot(x="Embarked", y="Prediction", hue="Survived",
               data=train_copy)
#https://stackoverflow.com/questions/30490740/move-legend-outside-figure-in-seaborn-tsplot
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


#https://seaborn.pydata.org/examples/scatter_bubbles.html

sns.set(style="white")


sns.relplot(x="Age", y="Prediction", hue="Survived", style="Pclass",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=train_copy)


# In[ ]:


g = sns.lmplot(x="Fare", y="Prediction", hue="Survived",
               truncate=True, height=5, data=train_copy)


# In[ ]:


g = sns.lmplot(x="Age", y="Prediction", hue="Survived",
               truncate=True, height=5, data=train_copy)


# In[ ]:


g = sns.catplot(x="Survived", y="Prediction", hue="Pclass",
               kind='box', height=5, data=train_copy)


# In[ ]:


g = sns.catplot(x="Survived", y="Prediction", hue="Sex",
               kind='box', height=5, data=train_copy)


# In[ ]:


g = sns.catplot(x="Survived", y="Prediction", hue="Embarked",
               kind='box', height=5, data=train_copy)


# In[ ]:


sns.relplot(x="Age", y="Prediction", hue="Survived", data=train_copy)


# In[ ]:


sns.relplot(x="Age", y="Prediction", hue="Survived", col="Sex", data=train_copy)


# In[ ]:


#sns.relplot(x="Age", y="Prediction", hue="Embarked",style='Survived', col="Sex", row='Pclass',data=train_copy)
sns.relplot(x="Age", y="Prediction", col="Sex",hue='Survived', row='Pclass',data=train_copy)


# In[ ]:


sns.relplot(x="Fare", y="Prediction", hue="Survived", col="Sex", row='Pclass', data=train_copy)


# In[ ]:


#https://stackoverflow.com/questions/46045750/python-distplot-with-multiple-distributions
target_0 = train_copy.loc[train_copy['Survived'] == 0]
target_1 = train_copy.loc[train_copy['Survived'] == 1]


_= sns.distplot(target_0[['Prediction']], hist=False, rug=True)
_= sns.distplot(target_1[['Prediction']], hist=False, rug=True)


# In[ ]:


#https://seaborn.pydata.org/generated/seaborn.barplot.html
#https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner
ax = sns.catplot(x="Survived", y="Prediction",col='SibSp', data=train_copy, estimator=median,height=4, aspect=.7,kind='bar')


# In[ ]:


((train[['SibSp', 'Survived']].groupby('SibSp')['Survived'].sum()/train['SibSp'].value_counts(sort=False))*100).plot(kind='bar')


# In[ ]:


ax = sns.catplot(x="Survived", y="Prediction",col='Parch', data=train_copy, estimator=median,height=4, aspect=.7,kind='bar')


# In[ ]:


((train[['Parch', 'Survived']].groupby('Parch')['Survived'].sum()/train['Parch'].value_counts(sort=False))*100).plot(kind='bar')


# In[ ]:


temp = train_copy.copy()
temp['Family_members'] = temp.SibSp + temp.Parch
_=sns.catplot(x="Family_members", y="Prediction",kind='box', hue="Survived", col="Sex", row='Pclass', data=temp)


# In[ ]:


_=sns.catplot(x="Embarked", y="Prediction",kind='box', col="Sex",hue='Survived', row='Pclass',data=train_copy)


# In[ ]:


#https://github.com/dmlc/xgboost/issues/1725
#https://machinelearningmastery.com/visualize-gradient-boosting-decision-trees-xgboost-python/
# import matplotlib
# xgb.plot_tree(clf, rankdir='LR')
# fig = matplotlib.pyplot.gcf()
# fig.set_size_inches(150, 200)
# fig.savefig('tree.png')

