#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics, preprocessing, tree
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import xgboost
from sklearn.metrics import roc_auc_score


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # EDA 

# In[ ]:


def missingdata(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    ms= ms[ms["Percent"] > 0]
    f,ax =plt.subplots(figsize=(8,6))
    plt.xticks(rotation='90')
    fig=sns.barplot(ms.index, ms["Percent"],color="green",alpha=0.8)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    return ms


# # Train-Data

# In[ ]:


df1 = pd.read_csv('/kaggle/input/TrainDataset.csv')


# # Test-Data

# In[ ]:


df2 = pd.read_csv('/kaggle/input/TestDataset.csv')


# In[ ]:


df1.head()


# In[ ]:


df2.head()


# In[ ]:


sns.heatmap(df1.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


# In[ ]:


sns.heatmap(df2.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


# In[ ]:


df1.info()


# In[ ]:


df1.isnull().sum()


# In[ ]:


df2.isnull().sum()


# In[ ]:


df1.describe()


# In[ ]:


df1.corr()


# In[ ]:


fig = plt.figure(figsize=(12,10))
sns.heatmap(df1.corr(),annot=True,cmap='Blues')
plt.xticks(rotation = 45)
plt.show()


# In[ ]:


df1.head()


# In[ ]:


def plot_feature(df,col):
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    if df[col].dtype == 'int64':
        df[col].value_counts().sort_index().plot()
    else:
        mean = df.groupby(col)['popularity'].mean()
        df[col] = df[col].astype('category')
        levels = mean.sort_values().index.tolist()
        df[col].cat.reorder_categories(levels,inplace=True)
        df[col].value_counts().plot()
    plt.xticks(rotation=45)
    plt.xlabel(col)
    plt.ylabel('Counts')
    plt.subplot(1,2,2)
    
    if df[col].dtype == 'int64' or col == 'buying_price':
        mean = df.groupby(col)['popularity'].mean()
        std = df.groupby(col)['popularity'].std()
        mean.plot()
        plt.fill_between(range(len(std.index)),mean.values-std.values,mean.values + std.values,                         alpha=0.1)
    else:
        sns.boxplot(x = col,y='popularity',data=df)
    plt.xticks(rotation=45)
    plt.ylabel('popularity')
    plt.show()    


# In[ ]:


plot_feature(df1,'buying_price')


# In[ ]:


plot_feature(df1,'maintainence_cost')


# In[ ]:


plot_feature(df1,'number_of_doors')


# In[ ]:


plot_feature(df1,'number_of_seats')


# In[ ]:


plot_feature(df1,'luggage_boot_size')


# In[ ]:


plot_feature(df1,'safety_rating')


# In[ ]:


plot_feature(df1,'popularity')


# In[ ]:


sns.pairplot(df1,diag_kind='kde')
plt.show()


# In[ ]:


sns.distplot(df1['popularity']);


# In[ ]:


plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.boxplot(df1.popularity)
plt.subplot(1,2,2)
sns.distplot(df1.popularity,bins=20)
plt.show()


# In[ ]:


for col in df1[:-1]:
    plt.figure(figsize=(10,8))
    sns.jointplot(x = df1[col],y = df1["popularity"],kind='reg')
    plt.xlabel(col,fontsize = 15)
    plt.ylabel("popularity",fontsize = 15)
    plt.grid()
    plt.show()


# In[ ]:


rows =2

cols = 3

fig,ax = plt.subplots(nrows=rows,ncols=cols,figsize=(16,5))

col = df1.columns

index = 0

for i in range(rows):
    for j in range(cols):
        sns.distplot(df1[col[index]],ax=ax[i][j])
        index = index + 1
        
plt.tight_layout()


# In[ ]:


col = ['buying_price', 'maintainence_cost', 'number_of_doors','number_of_seats', 'luggage_boot_size', 'safety_rating', 'popularity']


# In[ ]:


col


# In[ ]:


col = ['buying_price', 'maintainence_cost', 'number_of_doors','number_of_seats', 'luggage_boot_size', 'safety_rating', 'popularity']


fig, axis = plt.subplots(3, 3,  figsize=(25, 20))

counter = 0
for items in col:
    value_counts = df1[items].value_counts()
    
    trace_x = counter // 3
    trace_y = counter % 3
    x_pos = np.arange(0, len(value_counts))
    my_colors = 'rgbkymc'
    
    axis[trace_x, trace_y].bar(x_pos, value_counts.values, tick_label = value_counts.index,color=my_colors)
    
    axis[trace_x, trace_y].set_title(items)
    
    for tick in axis[trace_x, trace_y].get_xticklabels():
        tick.set_rotation(90)
    
    counter += 1

plt.tight_layout()
plt.show()


# In[ ]:


fig, axis = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(20, 15))

counter = 0
for items in col:
    
    trace_x = counter // 3
    trace_y = counter % 3
    
    
    axis[trace_x, trace_y].hist(df1[items])
    
    axis[trace_x, trace_y].set_title(items)
    
    counter += 1

plt.tight_layout()
plt.show()


# In[ ]:


def plot_count(x,fig):
    plt.subplot(4,2,fig)
   
    sns.countplot(df1[x],palette=("magma"))
    plt.subplot(4,2,(fig+1))
    
    sns.boxplot(x=df1[x], y=df1.popularity, palette=("magma"))
    
plt.figure(figsize=(15,20))

plot_count('buying_price', 1)
plot_count('maintainence_cost', 3)
plot_count('number_of_doors', 5)
plot_count('number_of_seats', 7)



plt.tight_layout()
plt.show()


# In[ ]:


def plot_count(x,fig):
    plt.subplot(4,2,fig)
   
    sns.countplot(df1[x],palette=("magma"))
    plt.subplot(4,2,(fig+1))
    
    sns.boxplot(x=df1[x], y=df1.popularity, palette=("magma"))
    
plt.figure(figsize=(15,20))

plot_count('luggage_boot_size', 1)
plot_count('safety_rating', 3)





plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(figsize=(25, 6))


plt.subplot(1,2,1)
plt1 = df1.buying_price.value_counts().plot('bar')
plt.title('buying_price')


plt.subplot(1,2,2)
plt1 = df1.maintainence_cost.value_counts().plot('bar')
plt.title('maintainence_cost')

plt.figure(figsize=(25, 6))

plt.subplot(1,2,1)
plt1 = df1.number_of_doors.value_counts().plot('bar')
plt.title('number_of_doors')
plt.show()

plt.subplot(1,2,2)
plt1 = df1.number_of_seats.value_counts().plot('bar')
plt.title('number_of_seats')
plt.show()


plt.figure(figsize=(25, 6))

plt.subplot(1,2,1)
plt1 = df1.luggage_boot_size.value_counts().plot('bar')
plt.title('luggage_boot_size')
plt.show()

plt.subplot(1,2,2)
plt1 = df1.safety_rating.value_counts().plot('bar')
plt.title('safety_rating')
plt.show()









plt.tight_layout()
plt.show()


# In[ ]:


for item in col[:-1]:
    plt.figure(figsize=(10,8))
    sns.violinplot(df1[item],df1["popularity"])
    
    plt.xlabel(item,fontsize=12)
    plt.ylabel("popularity",fontsize=12)
    plt.show()


# In[ ]:


for item in col[:-1]:
    plt.figure(figsize=(10,8))
    sns.boxplot(df1[item],df1["popularity"])
    
    plt.xlabel(item,fontsize=12)
    plt.ylabel("popularity",fontsize=12)
    plt.show()


# In[ ]:


for item in col:
    plt.figure(figsize=(10,8))
    plt.boxplot(df1[item])
    
    plt.xlabel(item,fontsize=12)
    plt.show()


# In[ ]:


sns.pairplot(df1, x_vars=col[:-1], y_vars='popularity', markers="+", size=4)
plt.show()


# In[ ]:


sns.catplot(x="buying_price", y="popularity", hue="maintainence_cost", kind="point", data=df1);


# In[ ]:


for item in df1[:-1]:
    plt.figure(figsize=(10,8))
   
    sns.distplot(df1[item], kde=False, fit=stats.gamma);
    
    plt.xlabel(item,fontsize=12)
    plt.ylabel("popularity",fontsize=12)
    plt.show()


# In[ ]:


df1.describe()


# In[ ]:


q = df1.popularity.describe()
print(q)
IQR    = q['75%'] - q['25%']
Upper  = q['75%'] + 1.5 * IQR
Lower  = q['25%'] - 1.5 * IQR
print("the upper and lower outliers are {} and {}".format(Upper,Lower))


# In[ ]:





# # MODEL BUILDING

# In[ ]:


df1.head()


# In[ ]:


X = df1.drop('popularity',axis=1)


# In[ ]:


y = df1[['popularity']]


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


X.shape


# # Logistic Reg

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# In[ ]:


logreg = LogisticRegression()


# In[ ]:


logreg.fit(X_train,y_train)


# In[ ]:


logreg.intercept_


# In[ ]:


logreg.coef_


# In[ ]:


predlog = logreg.predict(X_test)


# In[ ]:


confusion_matrix(y_test,predlog)


# In[ ]:


print(classification_report(y_test,predlog))


# In[ ]:


accuracy_score(y_test,predlog)


# In[ ]:


predlog = logreg.predict(df2)


# In[ ]:


predlog


# In[ ]:


log1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
log1.fit().summary()


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 25)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:





# # Random Forest

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# In[ ]:


rf = RandomForestClassifier(n_estimators=100)


# In[ ]:


rf.fit(X_train,y_train)


# In[ ]:


predrf = rf.predict(X_test)


# In[ ]:


confusion_matrix(y_test,predrf)


# In[ ]:


print(classification_report(y_test,predrf))


# In[ ]:


accuracy_score(y_test,predrf)


# In[ ]:





# In[ ]:


scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1_macro')


# In[ ]:


scores.mean()


# In[ ]:


rf.fit(X_train, y_train)
train_predictions = rf.predict(X_train)
test_predictions = rf.predict(X_test)


# In[ ]:


rf


# In[ ]:


print('The Training F1 Score is', f1_score(train_predictions, y_train,average='macro'))
print('The Testing F1 Score is', f1_score(test_predictions, y_test,average='macro'))


# HERE WE CAN SEE THAT OUR MODEL IN OVER FITTING

# In[ ]:





# # Random Forest with HyperParameter Tuning

# In[ ]:


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' %                   (method.__name__, (te - ts) * 1000))
        return result
    return timed


# In[ ]:


parameters = {   
              'max_depth':[10,20,30,40,50], 
              'min_samples_leaf':[1,2,3,4,5], 
              'min_samples_split':[2,3,4,5],
              'n_estimators': [10, 30, 50, 100,200],
              'criterion' : ['gini','entropy']}
scorer = make_scorer(f1_score,average ='macro')


# In[ ]:


@timeit
def generate_model_from_search(grid_or_random,model, parameters, scorer, X, y):
    if grid_or_random == "Grid":
        search_obj = GridSearchCV(model, parameters, scoring=scorer)
    elif grid_or_random == "Random":
        search_obj = RandomizedSearchCV(model, parameters, scoring=scorer)
    fit_obj = search_obj.fit(X, y)
    best_model = fit_obj.best_estimator_
    return best_model


# In[ ]:


best_model_random = generate_model_from_search("Random", 
                                           rf, 
                                           parameters, 
                                           scorer, 
                                           X_train, 
                                           y_train,
                                            )


# In[ ]:


scores = cross_val_score(best_model_random, X_train, y_train, cv=5,n_jobs=-1, verbose=1, scoring='f1_macro')
scores.mean()


# In[ ]:


best_model_random.fit(X_train, y_train)
best_train_predictions = best_model_random.predict(X_train)
best_test_predictions = best_model_random.predict(X_test)

print('The training F1 Score is', f1_score(best_train_predictions, y_train,average='macro'))
print('The testing F1 Score is', f1_score(best_test_predictions, y_test,average='macro'))


# In[ ]:


print(classification_report(y_test,best_test_predictions))


# In[ ]:


confusion_matrix(y_test,best_test_predictions)


# In[ ]:


accuracy_score(y_test,best_test_predictions)


# In[ ]:


best_test_predictions_Rf = best_model_random.predict(df2)


# In[ ]:


best_test_predictions_Rf


# In[ ]:





# # Decision Tree

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# In[ ]:


dt_default = DecisionTreeClassifier(max_depth=5)
dt_default.fit(X_train, y_train)


# In[ ]:


y_pred_default = dt_default.predict(X_test)
print(classification_report(y_test, y_pred_default))


# In[ ]:


print(confusion_matrix(y_test,y_pred_default))
print(accuracy_score(y_test,y_pred_default))


# In[ ]:


features = list(df1.columns[:-1])
features


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

n_folds = 5
parameters = {'max_depth': range(1, 40)}
dtree = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",return_train_score=True)
tree.fit(X_train, y_train)


# In[ ]:


scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[ ]:


plt.figure()
plt.plot(scores["param_max_depth"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[ ]:


n_folds = 5
parameters = {'min_samples_leaf': range(5, 200, 20)}
dtree = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",return_train_score=True)
tree.fit(X_train, y_train)


# In[ ]:


scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[ ]:


plt.figure()
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[ ]:


n_folds = 5
parameters = {'min_samples_split': range(5, 200, 20)}
dtree = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",return_train_score=True)
tree.fit(X_train, y_train)


# In[ ]:


scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[ ]:


plt.figure()
plt.plot(scores["param_min_samples_split"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_split"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_split")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# # Decision Tree With  Parameters

# In[ ]:


param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
    'criterion': ["entropy", "gini"]
}
n_folds = 5
dtree = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 
                          cv = n_folds, verbose = 1,return_train_score=True)
grid_search.fit(X_train,y_train)


# In[ ]:


cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results


# In[ ]:


plt.figure()
plt.plot(scores["param_min_samples_split"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_split"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_split")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[ ]:


print("best accuracy", grid_search.best_score_)
print(grid_search.best_estimator_)


# # Decision Tree With Best Parameters

# In[ ]:


clf_gini = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=10, 
                                  min_samples_leaf=50,
                                  min_samples_split=50)
clf_gini.fit(X_train, y_train)


# In[ ]:


clf_gini.score(X_test,y_test)


# # Decision Tree With Best Parameters - 2

# In[ ]:


clf_gini = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=3, 
                                  min_samples_leaf=50,
                                  min_samples_split=50)
clf_gini.fit(X_train, y_train)

print(clf_gini.score(X_test,y_test))


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
y_pred = clf_gini.predict(X_test)
print(classification_report(y_test, y_pred))


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


y_pred = clf_gini.predict(df2)


# In[ ]:


y_pred


# In[ ]:





# # Xg Boost

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# In[ ]:


classifier=xgboost.XGBRegressor()


# In[ ]:


booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]

n_estimators = [100, 500, 900, 1100]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]


hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }


# In[ ]:


random_cv = RandomizedSearchCV(estimator=classifier,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=100)


# In[ ]:


random_cv.fit(X_train,y_train)


# In[ ]:


random_cv.best_estimator_


# In[ ]:


regressor=xgboost.XGBClassifier(base_score=1, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.2, max_delta_step=0,
             max_depth=15, min_child_weight=1, missing=None, n_estimators=900,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)


# In[ ]:


regressor.fit(X_train,y_train)


# In[ ]:


pred = regressor.predict(X_test)


# In[ ]:


print(classification_report(y_test, pred))


# In[ ]:


print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))


# In[ ]:





# # Xg Boost With Right Parameters

# In[ ]:


booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]

n_estimators = [50,100]
max_depth = [1,2,3,4,5]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4,5]


hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }
scorer = make_scorer(f1_score,average ='macro')


# In[ ]:


@timeit
def generate_model_from_search(grid_or_random,regressor, hyperparameter_grid, scorer, X, y):
    if grid_or_random == "Grid":
        search_obj = GridSearchCV(regressor,hyperparameter_grid, scoring=scorer)
    elif grid_or_random == "Random":
        search_obj = RandomizedSearchCV(regressor,hyperparameter_grid, scoring=scorer)
    fit_obj = search_obj.fit(X, y)
    best_model = fit_obj.best_estimator_
    return best_model


# In[ ]:


best_model_random = generate_model_from_search("Random", 
                                           regressor, 
                                           hyperparameter_grid, 
                                           scorer, 
                                           X_train, 
                                           y_train,
                                            )


# In[ ]:


scores = cross_val_score(best_model_random, X_train, y_train, cv=5,n_jobs=-1, verbose=1, scoring='f1_macro')
scores.mean()


# In[ ]:


best_model_random.fit(X_train, y_train)
best_train_predictions = best_model_random.predict(X_train)
best_test_predictions = best_model_random.predict(X_test)

print('The training F1 Score is', f1_score(best_train_predictions, y_train,average='macro'))
print('The testing F1 Score is', f1_score(best_test_predictions, y_test,average='macro'))


# In[ ]:


print(classification_report(y_test,best_test_predictions))


# In[ ]:


print(confusion_matrix(y_test,best_test_predictions))
print(accuracy_score(y_test,best_test_predictions))


# In[ ]:


best_test_predictionsX = best_model_random.predict(df2)
best_test_predictionsX

