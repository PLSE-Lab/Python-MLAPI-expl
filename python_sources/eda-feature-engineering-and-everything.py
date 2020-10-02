#!/usr/bin/env python
# coding: utf-8

# Reference
# 
# 
# https://www.kaggle.com/artgor/eda-feature-engineering-and-everything
# 
# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# 
# https://www.kaggle.com/christofhenkel/market-data-nn-baseline

# ### Getting data and importing libraries

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[ ]:


#from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.kernel_ridge import KernelRidge
#from sklearn.svm import NuSVR
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# In[ ]:


# official way to get the data
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
print('Done!')


# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


pd.set_option("display.max_rows",10)
market_train_df


# We have two datasets, let's explore them separately.

# ## Market data
# 
# We have a really interesting dataset which contains stock prices for many companies over a decade!
# 
# For now let's have a look at the data itself and not think about the competition. We can see long-term trends, appearing and declining companies and many other things.

# In[ ]:


print(f'{market_train_df.shape[0]} samples and {market_train_df.shape[1]} features in the training market dataset.')


# In[ ]:


market_train_df.head()


# At first let's take 10 random assets and plot them.

# I plot data for all periods because I'd like to show long-term trends.
# Assets are sampled randomly, but you should see that some companies' stocks started trading later, some dissappeared. Disappearence could be due to bankruptcy, acquisition or other reasons.

# Well, these were some random companies. But it would be more interesting to see general trends of prices.

# It is cool to be able to see how markets fall and rise again.
# I have shown 4 events when there were serious stock price drops on the market.
# You could also notice that higher quantile prices have increased with time and lower quantile prices decreased.
# Maybe the gap between poor and rich increases... on the other hand maybe more "little" companies are ready to go to market and prices of their shares isn't very high.

# Now, let's look at these price drops in details.

# In[ ]:



market_train_df['price_diff'] = market_train_df['close'] - market_train_df['open']
grouped = market_train_df.groupby('time').agg({'price_diff': ['std', 'min']}).reset_index()
market_train_df


# In[ ]:


print(f"Average standard deviation of price change within a day in {grouped['price_diff']['std'].mean():.4f}.")


# We can see huge price fluctiations when market crashed. Just think about it... **But this is wrong!** There was no huge crash on January 2010... Let's dive into the data!

# ### Possible data errors
# 
# At first let's simply sort data by the difference between open and close prices.

# In[ ]:


market_train_df.sort_values('price_diff')[:10]


# In[ ]:


market_train_df['close_to_open'] =  np.abs(market_train_df['close'] / market_train_df['open'])


# In[ ]:


print(f"In {(market_train_df['close_to_open'] >= 1.2).sum()} lines price increased by 20% or more.")
print(f"In {(market_train_df['close_to_open'] <= 0.8).sum()} lines price decreased by 20% or more.")


# Well, this isn't much considering we have more than 4 million lines and a lot of these cases are due to price falls during market crash. Well just need to deal with outliers.

# In[ ]:


print(f"In {(market_train_df['close_to_open'] >= 2).sum()} lines price increased by 100% or more.")
print(f"In {(market_train_df['close_to_open'] <= 0.5).sum()} lines price decreased by 100% or more.")


# For a quick fix I'll replace outliers in these lines with mean open or close price of this company.

# In[ ]:



market_train_df['assetName_mean_open'] = market_train_df.groupby('assetName')['open'].transform('mean')
market_train_df['assetName_mean_close'] = market_train_df.groupby('assetName')['close'].transform('mean')

# if open price is too far from mean open price for this company, replace it. Otherwise replace close price.
for i, row in market_train_df.loc[market_train_df['close_to_open'] >= 2].iterrows():
    if np.abs(row['assetName_mean_open'] - row['open']) > np.abs(row['assetName_mean_close'] - row['close']):
        market_train_df.iloc[i,5] = row['assetName_mean_open']
    else:
        market_train_df.iloc[i,4] = row['assetName_mean_close']
        
for i, row in market_train_df.loc[market_train_df['close_to_open'] <= 0.5].iterrows():
    if np.abs(row['assetName_mean_open'] - row['open']) > np.abs(row['assetName_mean_close'] - row['close']):
        market_train_df.iloc[i,5] = row['assetName_mean_open']
    else:
        market_train_df.iloc[i,4] = row['assetName_mean_close']


# In[ ]:


market_train_df


# In[ ]:


market_train_df.drop(columns=['price_diff', 'close_to_open', 'assetName_mean_open', 'assetName_mean_close'], inplace=True)


# ## Modelling
# 
# It's time to build a model!
# I think that in this case we should build a binary classifier - we will simply predict whether the target goes up or down.

# In[ ]:


#numerical columns
cat_cols = ['assetCode']
num_cols = ['volume', 'close', 'open',
       'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
       #'price_diff', 'close_to_open', 'assetName_mean_open', 'assetName_mean_close'
           ]

from sklearn.model_selection import train_test_split

train_indices, val_indices = train_test_split(market_train_df.index.values,test_size=0.25, random_state=23)


# In[ ]:


def encode(encoder, x):
    len_encoder = len(encoder)
    try:
        id = encoder[x]
    except KeyError:
        id = len_encoder
    return id

encoders = [{} for cat in cat_cols]


for i, cat in enumerate(cat_cols):
    print('encoding %s ...' % cat, end=' ')
    encoders[i] = {l: id for id, l in enumerate(market_train_df.loc[train_indices, cat].astype(str).unique())}
    market_train_df[cat] = market_train_df[cat].astype(str).apply(lambda x: encode(encoders[i], x))
    print('Done')

embed_sizes = [len(encoder) + 1 for encoder in encoders] #+1 for possible unknown assets


# In[ ]:


from sklearn.preprocessing import StandardScaler
 
market_train_df[num_cols] = market_train_df[num_cols].fillna(0)
print('scaling numerical columns')

scaler = StandardScaler()

#col_mean = market_train[col].mean()
#market_train[col].fillna(col_mean, inplace=True)
scaler = StandardScaler()
market_train_df[num_cols] = scaler.fit_transform(market_train_df[num_cols])


# In[ ]:


market_train_df


# In[ ]:



class NN_base:        
        
    def __init__(self):
        
        from keras.models import Model
        from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, Dropout
        from keras.losses import binary_crossentropy, mse

        categorical_inputs = []
        for cat in cat_cols:
            categorical_inputs.append(Input(shape=[1], name=cat))

        categorical_embeddings = []
        for i, cat in enumerate(cat_cols):
            categorical_embeddings.append(Embedding(embed_sizes[i], 10)(categorical_inputs[i]))

            
        #categorical_logits = Concatenate()([Flatten()(cat_emb) for cat_emb in categorical_embeddings])
        categorical_logits = Flatten()(categorical_embeddings[0])
        categorical_logits = Dense(32,activation='relu')(categorical_logits)

        #categorical_logits = Flatten()(categorical_embeddings[0])
        #categorical_logits = Dense(32,activation='relu')(categorical_logits)
        #categorical_logits = Dropout(0.5)(categorical_logits)
        #categorical_logits = BatchNormalization()(categorical_logits)
        #categorical_logits = Dense(32,activation='relu')(categorical_logits)
        
        
        numerical_inputs = Input(shape=(11,), name='num')
        numerical_logits = numerical_inputs
        numerical_logits = BatchNormalization()(numerical_logits)

        #numerical_logits = Dense(128,activation='relu')(numerical_logits)
        #numerical_logits = Dropout(0.5)(numerical_logits)
        #numerical_logits = BatchNormalization()(numerical_logits)
        #numerical_logits = Dense(128,activation='relu')(numerical_logits)
        #numerical_logits = Dense(64,activation='relu')(numerical_logits)
         
        numerical_logits = Dense(128,activation='relu')(numerical_logits)
        numerical_logits = Dense(64,activation='relu')(numerical_logits)

        logits = Concatenate()([numerical_logits,categorical_logits])
        logits = Dense(64,activation='relu')(logits)
        out = Dense(1, activation='sigmoid')(logits)

        self.model = Model(inputs = categorical_inputs + [numerical_inputs], outputs=out)
        self.model.compile(optimizer='adam',loss=binary_crossentropy)
        
    def fit(self,X_train,y_train):
        from keras.callbacks import EarlyStopping, ModelCheckpoint

        check_point = ModelCheckpoint('model.hdf5',verbose=True, save_best_only=True)
        early_stop = EarlyStopping(patience=5,verbose=True)
        self.model.fit(X_train,y_train.astype(int),
                  #validation_data=(X_valid,y_valid.astype(int)),
                  epochs=3,
                  verbose=True,
                  callbacks=[early_stop,check_point]) 
    
    def predict(self,X_test):
        return self.model.predict(X_test)
    
    def summary(self):
        self.model.summary()


# In[ ]:


def get_input(market_train, indices):
    X_num = market_train.loc[indices, num_cols].values
    X = {'num':X_num}
    for cat in cat_cols:
        X[cat] = market_train.loc[indices, cat_cols].values
    y = (market_train.loc[indices,'returnsOpenNextMktres10'] >= 0).values
    r = market_train.loc[indices,'returnsOpenNextMktres10'].values
    u = market_train.loc[indices, 'universe']
    d = market_train.loc[indices, 'time'].dt.date
    return X,y,r,u,d

# r, u and d are used to calculate the scoring metric
X_train,y_train,r_train,u_train,d_train = get_input(market_train_df, train_indices)
X_valid,y_valid,r_valid,u_valid,d_valid = get_input(market_train_df, val_indices)


# In[ ]:


NN_tmp = NN_base()


# In[ ]:


NN_tmp.fit(X_train,y_train)


# In[ ]:


#model_lgb_ = lgb.LGBMClassifier(objective='binary',learning_rate=0.05, bagging_fraction = 0.8,
#                                bagging_freq = 5, n_estimators=100,boosting_type = 'dart',
#                                num_leaves = 2452, min_child_samples = 212, reg_lambda=0.01)


# In[ ]:


#model_lgb_.fit(X_train['num'],y_train)


# In[ ]:


#model_xgb_ = xgb.XGBClassifier(colsample_bytree=0.4603, gamma=0.0468, 
#                             learning_rate=0.05, max_depth=6, 
#                             min_child_weight=1.7817, n_estimators=100,
#                             reg_alpha=0.4640, reg_lambda=0.8571,
#                             subsample=0.5213, silent=1,
#                             random_state =7, nthread = -1)


# In[ ]:


#model_xgb_.fit(X_train['num'],y_train)


# In[ ]:


from sklearn.metrics import accuracy_score
confidence_valid = NN_tmp.predict(X_valid)[:,0]*2 -1
print(accuracy_score(confidence_valid>0,y_valid))


# In[ ]:


# calculation of actual metric that is used to calculate final score
r_valid = r_valid.clip(-1,1) # get rid of outliers. Where do they come from??
x_t_i = confidence_valid * r_valid * u_valid
data = {'day' : d_valid, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)


# In[ ]:


'''
import time
import copy

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        tmp_num = []
        tmp_cat = []
        for ty, model in base_models:
            if ty == 'num':
                tmp_num.append(model)
            elif ty == 'cat':
                tmp_cat.append(model)
            else:
                continue
        
        self.base_models_num = tuple(tmp_num)
        self.base_models_cat = tuple(tmp_cat)
        self.meta_model = meta_model
        self.n_folds = n_folds
        
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_num_ = [list() for x in self.base_models_num]
        self.base_models_cat_ = [list() for x in self.base_models_cat]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        X_num = X['num']
        X_cat = X['assetCode']
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X['num'].shape[0], len(self.base_models_num)))
        for i, model in enumerate(self.base_models_num):
            for train_index, holdout_index in kfold.split(X_num, y):
                ts = time.time()
                instance = clone(model)
                self.base_models_num_[i].append(instance)
                instance.fit(X_num[train_index], y[train_index])
                y_pred = instance.predict(X_num[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                print("{} model... complete at {}".format(i,(time.time()-ts)))
        
        out_of_fold_predictions_c = np.zeros((X['num'].shape[0], len(self.base_models_cat)))
        for i, model in enumerate(self.base_models_cat):
            for train_index, holdout_index in kfold.split(X_cat, y):
                ts = time.time()
                instance = copy.deepcopy(model)
                self.base_models_cat_[i].append(instance)
                
                X_t = {'assetCode' : X_cat[train_index], 'num': X_num[train_index]}
                X_h = {'assetCode' : X_cat[holdout_index], 'num': X_num[holdout_index]}
                instance.fit(X_t, y[train_index])
                y_pred = (instance.predict(X_h) > 0.5 )
                out_of_fold_predictions_c[holdout_index, i] = y_pred.flatten()
                print("{} model... complete at {}".format(i,(time.time()-ts)))        
        
        out_of_fold_predictions = np.concatenate((out_of_fold_predictions, out_of_fold_predictions_c), axis=1)
        
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
    
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        X_num = X['num']
        X_cat = X['assetCode']
        
        meta_features_num = np.column_stack([
            np.column_stack([model.predict(X_num) for model in base_models_num]).mean(axis=1)
            for base_models_num in self.base_models_num_ ])
        
        X_t = {'num': X_num, 'assetCode': X_cat}
        if not self.base_model_cat:
            meta_features_cat = np.column_stack([
                np.column_stack([model.predict(X_t) for model in base_models_cat]).mean(axis=1)
                for base_models_cat in self.base_models_cat_ ])
            meta_features = np.concatenate((meta_features_num, meta_features_cat), axis=1)
        else:
            meta_features = meta_features_num
        return self.meta_model_.predict_proba(meta_features)
    
    
'''


# In[ ]:


'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

NN = NN_base()

KN_2 = KNeighborsClassifier(n_neighbors=2)
KN_4 = KNeighborsClassifier(n_neighbors=4)
KN_8 = KNeighborsClassifier(n_neighbors=8)
lr = LogisticRegression()


#GBoost = GradientBoostingClassifier(n_estimators=10, learning_rate=0.05,
#                                   max_depth=6,min_samples_leaf=15, min_samples_split=10,random_state =5, verbose=2)
model_lgb_ = lgb.LGBMClassifier(objective='binary',learning_rate=0.05, bagging_fraction = 0.8,
                                bagging_freq = 5, n_estimators=100,boosting_type = 'dart',
                                num_leaves = 2452, min_child_samples = 212, reg_lambda=0.01)

model_xgb_ = xgb.XGBClassifier(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=6, 
                             min_child_weight=1.7817, n_estimators=100,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
'''


# In[ ]:


#model_lgb_meta = lgb.LGBMClassifier(objective='binary',learning_rate=0.05, n_estimators=100, bagging_fraction = 0.8,
#                              bagging_freq = 5, boosting_type = 'dart')


# In[ ]:


#stacked_averaged_models = StackingAveragedModels(base_models = (('num',model_lgb_),('num',model_xgb_)),
#                                                 meta_model = model_lgb_meta)


# In[ ]:


#stacked_averaged_models.fit(X_train,y_train)


# In[ ]:


#test={'num': X_train['num'][:10], 'assetCode': X_train['assetCode'][:10]}


# In[ ]:


#stacked_averaged_models.predict(test)


# In[ ]:


days = env.get_prediction_days()
import time

n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    if n_days % 50 == 0:
        print(n_days,end=' ')
    
    t = time.time()
    assetCode = market_obs_df['assetCode']

    #market_obs_df['price_diff'] = market_obs_df['close'] - market_obs_df['open']
    #market_obs_df['close_to_open'] =  np.abs(market_obs_df['close'] / market_obs_df['open'])
    #market_obs_df['assetName_mean_open'] = market_obs_df.groupby('assetName')['open'].transform('mean')
    #market_obs_df['assetName_mean_close'] = market_obs_df.groupby('assetName')['close'].transform('mean')
    market_obs_df[num_cols] = market_obs_df[num_cols].fillna(0)
    market_obs_df[num_cols] = scaler.fit_transform(market_obs_df[num_cols])
    #market_obs_df = market_obs_df.loc[:, num_cols].fillna(0).values
    X = {'num': market_obs_df[num_cols].values}
    for i,cat in enumerate(cat_cols):
        market_obs_df[cat+'_encoded'] = market_obs_df[cat].astype(str).apply(lambda x: encode(encoders[i],x))
        X[cat] = market_obs_df[cat+'_encoded'].values
    
    
    prep_time += time.time() - t
    
    t = time.time()
    #lp = stacked_averaged_models.predict(X)
    lp = NN_tmp.predict(X)[:,0]
    #lp = model_lgb_.predict(X['num'])[:]
    #lp = model_xgb_.predict(X['num'])[:]
    prediction_time += time.time() -t
    
    t = time.time()
    confidence = 2 * lp -1
    preds = pd.DataFrame({'assetCode':assetCode,'confidence':confidence})
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t
    
env.write_submission_file()


# In[ ]:


n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
predicted_confidences = np.array([])
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    print(n_days,end=' ')
    
    t = time.time()

    market_obs_df['assetCode_encoded'] = market_obs_df[cat].astype(str).apply(lambda x: encode(encoders[i], x))

    market_obs_df[num_cols] = market_obs_df[num_cols].fillna(0)
    market_obs_df[num_cols] = scaler.transform(market_obs_df[num_cols])
    X_num_test = market_obs_df[num_cols].values
    X_test = {'num':X_num_test}
    X_test['assetCode'] = market_obs_df['assetCode_encoded'].values
    
    prep_time += time.time() - t
    
    t = time.time()
    market_prediction = model.predict(X_test)[:,0]*2 -1
    predicted_confidences = np.concatenate((predicted_confidences, market_prediction))
    prediction_time += time.time() -t
    
    t = time.time()
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':market_prediction})
    # insert predictions to template
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t

env.write_submission_file()
total = prep_time + prediction_time + packaging_time
print(f'Preparing Data: {prep_time:.2f}s')
print(f'Making Predictions: {prediction_time:.2f}s')
print(f'Packing: {packaging_time:.2f}s')
print(f'Total: {total:.2f}s')


# In[ ]:




