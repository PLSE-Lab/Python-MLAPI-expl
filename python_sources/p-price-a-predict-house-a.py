#!/usr/bin/env python
# coding: utf-8

# # Price Predict House (Regression Problem)

# In this notebook we use to method: 
# * By traditional Machine Learning
# * Try to use Deep Learning

# Actually, the datasets provided for this notebook are quite similar to [House Sales in King County, USA](https://www.kaggle.com/harlfoxem/housesalesprediction) but without time(date) relation and also a little bit customization value in the datas.

# # The Datas

# In[ ]:


import numpy as np 
import pandas as pd 
import warnings # Just to remove verbose maybe...
warnings.filterwarnings('ignore')


# In[ ]:


train_df = pd.read_csv('../input/dasprodatathon/train.csv')
test_df = pd.read_csv('../input/dasprodatathon/test.csv')

train_df.shape, test_df.shape


# In[ ]:


test_df.head()


# In[ ]:


test_df.isnull().sum().sum()


# In[ ]:


test_df.info()


# In[ ]:


test_df.describe().T


# In[ ]:


train_df.head()


# In[ ]:


train_df.isnull().sum().sum()


# In[ ]:


train_df.info()


# In[ ]:


train_df.describe().T


# In[ ]:


train_df.columns


# # EDA

# In[ ]:


# data visualization
import matplotlib.pyplot as plt
import seaborn as sns 
import chart_studio.plotly as ply
import plotly.graph_objs as go
import plotly.offline as offln
from plotly import tools

offln.init_notebook_mode(connected=False)

# Statistics
from scipy.stats import norm, skew 
from scipy import stats 

from sklearn.preprocessing import RobustScaler, robust_scale, MinMaxScaler


# In[ ]:


fig, axs = plt.subplots(ncols=2, figsize=(17,3))
sns.stripplot(x='Waterfront', data=train_df, ax=axs[0])
sns.stripplot(x='Year Renovated', data=train_df, ax=axs[1])


# In[ ]:


plt.figure(figsize=(19,7))
sns.stripplot(x='Zipcode', hue='Waterfront', data=train_df)


# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(train_df['Price'], color='Purple', fit=norm)
mu, sigma = norm.fit(train_df['Price'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} // $\sigma=$ {:.2f} // \nSkewness: {:.2f} // Kurtosis: {:.2f} )'
            .format(mu, sigma, train_df['Price'].skew(), train_df['Price'].kurt())])


# In[ ]:


sns.boxplot(train_df['Price'])


# In[ ]:


# n_rows = 6
# n_cols = 2

# plot_feat = np.array([ 'Price', 'Living Area', 'Total Area', 'Above the Ground Area',
#        'Basement Area', 'Neighbors Living Area', 'Neighbors Total Area',
#        'Bedrooms', 'Bathrooms', 'Floors', 'Year Built', 'Zipcode'])

# fig, axs = plt.subplots(n_rows, n_cols, figsize=(15,25))

# for r in range(0,n_rows):
#     for c in range(0,n_cols):  
#         i = r*n_cols+c
#         if i < len(plot_feat):
#             sns.distplot(train_data[plot_feat[i]], color='Purple', ax = axs[r][c], fit=norm)
#             mu, sigma = norm.fit(train_data[plot_feat[i]])
#             axs[r][c].legend(['Normal dist. ($\mu=$ {:.2f} // $\sigma=$ {:.2f} // \nSkewness: {:.2f} // Kurtosis: {:.2f} )'
#                         .format(mu, sigma, train_data[plot_feat[i]].skew(), train_data[plot_feat[i]].kurt())])

# plt.tight_layout()


# In[ ]:


# fig = plt.figure(figsize = (17,10))

# fig.add_subplot(1,3,1)
# res = stats.probplot(train_data['Price'], plot=plt)
# plt.title('Default')
# plt.legend(['Std: {}'.format(train_data['Price'].std())])

# fig.add_subplot(1,3,2)
# res = stats.probplot(robust_scale(train_data['Price']), plot=plt)
# plt.title('Scaling')
# plt.legend(['Std: {}'.format(robust_scale(train_data['Price']).std())])

# fig.add_subplot(1,3,3)
# res = stats.probplot(np.log1p(train_data['Price']), plot=plt)
# plt.title('Log1p')
# plt.legend(['Std: {}'.format(np.log1p(train_data['Price']).std())])


# In[ ]:


# scales=pd.array(robust_scale(train_data['Price']))
# logss=np.log1p(train_data['Price'])

# fig = plt.figure(figsize=(15,7))

# fig.add_subplot(1,2,1)
# sns.distplot(scales, color='Red', fit=norm)

# mu, sigma = norm.fit(scales)
# plt.legend(['Normal dist. ($\mu=$ {:.2f} // $\sigma=$ {:.2f} // \nSkewness: {:.2f} // Kurtosis: {:.2f} )'
#             .format(mu, sigma, scales.skew(), scales.kurt())])
# plt.title('Scaling')
# plt.xlabel('Price')

# fig.add_subplot(1,2,2)
# sns.distplot(logss, color='Red', fit=norm)

# mu, sigma = norm.fit(logss)
# plt.legend(['Normal dist. ($\mu=$ {:.2f} // $\sigma=$ {:.2f} // \nSkewness: {:.2f} // Kurtosis: {:.2f} )'
#             .format(mu, sigma, logss.skew(), logss.kurt())], loc='upper right')
# plt.title('Log1p')


# In[ ]:


# Interactive visualization
trace = dict(type = 'scattergeo',
             lon = train_df['Longitude'],
             lat = train_df['Latitude'],
             marker = dict(size=12),
             text = train_df['Zipcode'],
             mode = 'markers')

data = [trace]
layout = dict(title='House Location (Cognixy Region)',
              showlegend=False,
              geo=dict(
                  scope='usa',
                  showland=True)
             )

fig=dict(data=data,
         layout=layout)

offln.iplot(fig)


# In[ ]:


train_df.plot(kind='scatter',x='Longitude', y='Latitude', c='Price', alpha=0.4, title='Price Distribution', 
                cmap='terrain', figsize=(10,7), colorbar=True, sharex=False)


# In[ ]:


# plt.figure(figsize=(10,7))
# sns.scatterplot(x='Longitude', y='Latitude', hue='Zipcode', data=train_data, c='Zipcode')
train_df.plot(kind='scatter',x='Longitude', y='Latitude', c='Zipcode', alpha=0.4, title='Zipcode', 
                 cmap='terrain', figsize=(10,7), colorbar=True, sharex=False)


# In[ ]:


fig1, axs = plt.subplots(ncols=3, figsize=(20, 4))

train_df['Bathrooms'].value_counts().plot(kind='bar', ax=axs[0], title='Num of Bathrooms')
sns.countplot(x='Bedrooms', data=train_df, ax=axs[1])
sns.countplot(x='Floors', data=train_df, ax=axs[2])

# plt.tight_layout()


# In[ ]:


# Interactive visualization
train_unique = []
cols = ['Bedrooms','Bathrooms','Floors','Waterfront','View','Condition','Grade', 'Year Built', 'Zipcode', 'Year Renovated']

for i in cols:
    train_unique.append(train_df[i].nunique())
unique_train = pd.DataFrame()
unique_train['Columns'] = cols
unique_train['Unique_Value'] = train_unique

data = [
    go.Bar(
        x = unique_train['Columns'],
        y = unique_train['Unique_Value'],
        name = 'Unique value in features',
        textfont=dict(size=17),
        marker=dict(
        line=dict(
            color= 'yellow',
            #width= 2,
        ), opacity = 0.45
    )
    ),
    ]
layout= go.Layout(
        title= "Unique Value By Column",
        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),
        yaxis= dict(title='Value Count', ticklen=5, gridwidth=2),
        showlegend=True
)
fig = go.Figure(data=data, layout=layout)
offln.iplot(fig, filename='skin')


# In[ ]:


corr_matrix = train_df.corr(method='spearman')
fig, axes = plt.subplots(figsize=(17,17))
sns.heatmap(corr_matrix.sort_values(by='Price', ascending=False).sort_values(by='Price', ascending=False, axis=1), 
            cmap='YlGnBu', annot=True,linewidths= .5, vmax=1, ax=axes)

corr_matrix['Price'].sort_values(ascending=False)


# In[ ]:


num_feat = np.array([ 'Living Area','Neighbors Living Area','Above the Ground Area', 'Basement Area', 'Total Area', 'Neighbors Total Area'])
cat_feat = np.array(['Grade','Bathrooms','Bedrooms', 'Floors', 'View', 'Condition'])

def plot_bivar(features, the_plot, nrows, ncols):
    '''
    Use to plot some categorical features that seems look high correlated to the label ('Price')
    with 4 parameters:
    * features : The feature for visualization (pass as string in List)
    * the_plot : the type of the graph
    * nrows : rows number (for subplotting)
    * ncols : cols number (for subplotting)
    '''
    fig, axs = plt.subplots(nrows, ncols, figsize=(20,15))

    for r in range(0,nrows):
        for c in range(0,ncols):  
            i = r*ncols+c
            if i < len(features):
                the_plot(x = train_df[features[i]], y=train_df['Price'], ax = axs[r][c])

    sns.despine()
    plt.tight_layout()
    
plot_bivar(num_feat,sns.regplot,3,2)


# In[ ]:


plot_bivar(cat_feat,sns.boxplot,3,2)


# In[ ]:


train_df.columns


# # Preprocessing

# In[ ]:


from sklearn.feature_selection import SelectKBest, f_regression, rfe
from sklearn.cluster import KMeans


# In[ ]:


train_temp = train_df.copy()
test_temp = test_df.copy()


# In[ ]:


# We gonna try to cluster the zipcodes, maybe..
wcss = []
for i in range(1, 16):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(np.array(test_df['Zipcode']).reshape(-1,1))
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(19,10))
plt.plot(range(1, 16), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Squares')
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters=20)
kmeanstr = kmeans.fit(np.array(train_df['Zipcode']).reshape(-1,1))
labelstr = kmeans.predict(np.array(train_df['Zipcode']).reshape(-1,1))
kmeansts = kmeans.fit(np.array(test_df['Zipcode']).reshape(-1,1))
labelsts = kmeans.predict(np.array(test_df['Zipcode']).reshape(-1,1))


# In[ ]:


plt.figure(figsize=(10,7))
sns.scatterplot(train_df['Longitude'], train_df['Latitude'], labelstr, palette='rainbow', legend='full')
plt.title('Zipcode Cluster -- train')
plt.legend(loc=[1.1,.1]);


# In[ ]:


plt.figure(figsize=(10,7))
sns.scatterplot(test_df['Longitude'], test_df['Latitude'], labelsts, palette='rainbow', legend='full')
plt.title('Zipcode Cluster -- test')
plt.legend(loc=[1.1,.1]);


# In[ ]:


train_df.columns


# Feature Generation

# In[ ]:


train_df['zipcode_cluster'] = pd.Series(labelstr, index=train_df.index)
test_df['zipcode_cluster'] = pd.Series(labelsts, index=test_df.index)
train_df = train_df.drop('Zipcode', axis=1)
test_df = test_df.drop('Zipcode', axis=1)


# In[ ]:


harga = train_df.groupby(['zipcode_cluster'], as_index=False).agg({'Price': {'min_price_z': np.min,
                                                                             'mean_price_z': np.mean,
                                                                             'max_price_z': np.max}})
harga.columns = ['zipcode_cluster', 'min_price_z', 'mean_price_z', 'max_price_z']


# In[ ]:


merged_train = train_df.merge(harga, on='zipcode_cluster', how='left', right_index=False)
merged_test = test_df.merge(harga, on='zipcode_cluster', how='left', right_index=False)


# In[ ]:


merged_train.head()


# In[ ]:


train_df = merged_train
test_df = merged_test


# In[ ]:


train_df.columns


# In[ ]:


train_df['yard_area_ratio_w_neighbors'] = (train_df['Total Area'] - train_df['Living Area']) / (train_df['Neighbors Total Area'] - train_df['Neighbors Living Area'])
train_df['room_ratio_per_floor'] = (train_df['Bedrooms'] + train_df['Bathrooms']) / train_df['Floors']

test_df['yard_area_ratio_w_neighbors'] = (test_df['Total Area'] - test_df['Living Area']) / (test_df['Neighbors Total Area'] - test_df['Neighbors Living Area'])
test_df['room_ratio_per_floor'] = (test_df['Bedrooms'] + test_df['Bathrooms']) / test_df['Floors']


# Assume that we have outliers

# In[ ]:


train_df[train_df['Bedrooms'] == 9]


# In[ ]:


# plt.figure(figsize=(15,5))
# sns.boxplot(train_df['Basement Area'])
train_df[train_df['Bedrooms'] > 9]


# In[ ]:


# train_df = train_df[train_df['Bedrooms'] < 10]


# # Validation model

# Do it before train the data with full model and predict on real test set

# In[ ]:


from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


# In[ ]:


X = train_df.drop(['ID', 'Price'], axis=1)
y = train_df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[ ]:


class AvgMods(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        for model in self.models_:
            model.fit(X, y)

        return self
    
    def predict(self, X):
        prdct = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(prdct, axis=1)  


# In[ ]:


#Validation function
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).get_n_splits(X)
def rmse_cv(model):
    rmse= -cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv = skf)
    return print(f'{rmse}\n--> {rmse.mean():.4f}')

def acc_cv(model):
    score = cross_val_score(model, X, y, cv = skf)
    return print(f'{score}\n--> {score.mean()*100:.2f}%')

def rmse(y, y_pred):
    print(np.sqrt(mean_squared_error(y, y_pred)))

def get_best_param(model, param, X, y, theSearch='Random', n_iter=10):
    if theSearch == 'Grid':
        grSCV = GridSearchCV(model, param, scoring="neg_root_mean_squared_error", cv=skf, verbose=0)
        grSCV.fit(X, y)
        return grSCV.best_params_, '--> {:.4f}'.format(-grSCV.best_score_)
    elif theSearch == 'Random':
        ranSCV = RandomizedSearchCV(model, param, scoring="neg_root_mean_squared_log_error", cv=skf, verbose=0, n_iter=n_iter, refit=True, return_train_score=False)
        ranSCV.fit(X, y)
        return ranSCV.best_params_, '--> {:.4f}'.format(-ranSCV.best_score_)
    return None


# In[ ]:


param_gridsearch = {
    'learning_rate' : [0.01, 0.1, 1],
    'max_depth' : [5, 10, 15],
    'n_estimators' : [5, 20, 35], 
    'num_leaves' : [5, 25, 50],
    'boosting_type': ['gbdt', 'dart'],
    'colsample_bytree' : [0.6, 0.75, 1],
    'reg_lambda': [0.01, 0.1, 1],
}

param_random = {
    'learning_rate': list(np.logspace(np.log(0.01), np.log(1), num = 500, base=3)),
    'max_depth': list(range(5, 15)),
    'n_estimators': list(range(5, 35)),
    'num_leaves': list(range(5, 50)),
    'boosting_type': ['gbdt', 'dart'],
    'colsample_bytree': list(np.linspace(0.6, 1, 500)),
    'reg_lambda': list(np.linspace(0, 1, 500)),
}
param_hyperopt= {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'max_depth': hp.quniform('max_depth', 5, 15, 1),
    'n_estimators': 1500,
    'num_leaves': 10,
    'boosting_type': 'gbdt',
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
}

params = {
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 5000 ,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'mse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': True,
        'seed':0,        
    }


# In[ ]:


lm= make_pipeline(RobustScaler(), LinearRegression(copy_X=True, fit_intercept=True, normalize=False))

lasR = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

eNet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

gbReg = GradientBoostingRegressor(n_estimators=500, max_depth=7, loss='huber', learning_rate=0.05)

xgbMod = XGBRegressor(max_depth = 3, n_estimators = 1000, colsample_bytree=0.5, missing=True)

lgbMod = LGBMRegressor(num_leaves=10,learning_rate=0.05, n_estimators=3500,max_bin = 500, bagging_fraction = 0.9,
                       bagging_freq = 5, feature_fraction = 0.3,feature_fraction_seed=1, bagging_seed=3,min_data_in_leaf =1, 
                       min_sum_hessian_in_leaf = 11, tree_learner='data', force_col_wise=True, max_depth=500)


# Hyperparameter Tuning

# In[ ]:


# p={'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
#  get_best_param(lm, p, x_train, y_train, theSearch='sd')
# par={'min_sum_hessian_in_leaf':np.array([0,1,11,20,50,30])}
# get_best_param(lgbMod, par, x_train, y_train)


# In[ ]:


# acc_cv(lm)
# rmse_cv(lm)

# acc_cv(lasR)
# rmse_cv(lasR)

# acc_cv(eNet)
# rmse_cv(eNet)

# acc_cv(gbReg)
# rmse_cv(gbReg)

# acc_cv(xgbMod)
# rmse_cv(xgbMod)

acc_cv(lgbMod)
rmse_cv(lgbMod)


# In[ ]:


# averaged_models = AvgMods(models = (lgbMod, gbReg, xgbMod))
# acc_cv(averaged_models)


# # Evaluation again

# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[ ]:


lgbMod.fit(X_train, y_train)
y_pred = lgbMod.predict(X_test)


# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot((y_test-y_pred))


# In[ ]:


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2s = r2_score(y_test, y_pred)

print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'>> RMSE: {np.sqrt(mse):.4f}')
print(f'R squared: {r2s * 100 :.2f} %')


# # Res...

# In[ ]:


# gb_pred = gbReg.predict(X_test_real)
# xgb_pred = xgbMod.predict(X_test_real)


# In[ ]:


# sns.distplot((y_test-y_predict), bins=30)
# coeffecients = pd.DataFrame(lm.coef_,x_train.columns)
# coeffecients.columns = ['Coeffecient']
# coeffecients.sort_values(by='Coeffecient', ascending=False)


# # Use Deep Learning

# Preprocess for use in the Deep Learning

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# In[ ]:


scaler = RobustScaler()


# In[ ]:


X_train_scaled = scaler.fit_transform(X_train)
y_train_scaled = scaler.fit_transform(np.array(y_train).reshape(-1, 1))
X_test_scaled = scaler.transform(X_test)
y_test_scaled = scaler.transform(np.array(y_test).reshape(-1, 1))


# In[ ]:


X_train_scaled.shape, y_train_scaled.shape, X_test_scaled.shape, y_test_scaled.shape


# # Model Deep Learning

# In[ ]:


early_stop = EarlyStopping(monitor='val_loss', 
                           patience=10, 
                           mode='min', 
                           restore_best_weights=True)

check_point = ModelCheckpoint('house_price_pred_z.h5', 
                              monitor='val_loss', 
                              save_best_only=True)

lr_plateau = ReduceLROnPlateau(monitor='val_loss', 
                               patience=2,
                               factor=.2, 
                               min_lr=1e-6)


# In[ ]:


model = Sequential()

model.add(Dense(18, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(3, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(2, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')


# In[ ]:


model.fit(X_train_scaled, y_train_scaled, 
          validation_data=(X_test_scaled,y_test_scaled), 
          batch_size=128, 
          epochs=175,
          callbacks=[lr_plateau, early_stop])


# # Evaluate Deep Learning

# In[ ]:


loss_df = pd.DataFrame(model.history.history)


# In[ ]:


loss_df.head()


# In[ ]:


loss_df[['loss','val_loss']].plot()


# In[ ]:


dl_pred = model.predict(X_test_scaled)


# In[ ]:


mae = mean_absolute_error(y_test, dl_pred)
mse = mean_squared_error(y_test, dl_pred)
r2s = r2_score(y_test, dl_pred)

print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'>> RMSE: {np.sqrt(mse):.4f}')
print(f'R squared: {r2s * 100 :.2f} percent')


# # Prediction on Real Test Set

# Use the model that has the lowest rmlse

# In[ ]:


X_test_real = test_df.drop('ID', axis=1)


# In[ ]:


lgbMod.fit(X, y)

print('Success:)')


# In[ ]:


real_pred = lgbMod.predict(X_test_real)


# In[ ]:


# X = scaler.fit_transform(X)
# X_test_real = scaler.transform(X_test_real)

# model = Sequential()

# model.add(Dense(18, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(.2))

# model.add(Dense(12, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dropout(.2))

# model.add(Dense(3, activation='relu'))
# model.add(Dropout(.2))
# model.add(Dense(2, activation='relu'))

# model.add(Dense(1))

# model.compile(optimizer='adam', loss='mse')


# In[ ]:


# model.fit(X.values, y, batch_size=128, epochs=300)


# In[ ]:


# y_predict = model.predict(X_test_real)


# # Finishing

# The data to submit

# In[ ]:


submission_data = pd.read_csv('../input/dasprodatathon/sample_submission.csv')
submission_data['ID'] = test_df['ID']
submission_data['Price'] = real_pred

submission_data.to_csv('submit_house_price_predict.csv', index=False)


# In[ ]:


submission_data.tail()

