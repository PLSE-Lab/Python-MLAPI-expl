#!/usr/bin/env python
# coding: utf-8

# # Market Data Only Baseline (XGBRegressor)
# 
# 
# This is a fit of market data only (no news data used) showing relatively good results. 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from kaggle.competitions import twosigmanews
from collections import Counter
from sklearn import linear_model
from xgboost import XGBRegressor
import scipy
import lightgbm as lgb
import itertools
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, truncnorm, uniform
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

pd.options.mode.chained_assignment = None


# In[ ]:


env = twosigmanews.make_env()
(market_train, _) = env.get_training_data()


# # Data Visu

# In[ ]:


market_train.shape


# In[ ]:


market_train["time"] = market_train["time"].apply(lambda x: pd.Timestamp(x))


# In[ ]:


# first date
min_time = np.min(market_train["time"])
print(min_time)


# In[ ]:


# last date
max_time = np.max(market_train["time"])


# In[ ]:


def get_score(df_):
    assert "pred" in df_.columns
    df_["score"] = df_["returnsOpenNextMktres10"] *         df_["pred"] * df_["universe"]
    x_t_sum = df_.groupby("time").sum()["score"]
    score = x_t_sum.mean() / x_t_sum.std()
    return score

def sigma_score_lgb_f(preds, valid_data, df_valid_):
    df_valid_["pred"] = preds
    score = get_score(df_valid_)
    
    return 'sigma_score', score, True

def get_market_X(df_,
                 X_columns=['volume',
                            'close',
                            'open',
                            'returnsClosePrevRaw1',
                            'returnsOpenPrevRaw1',
                            'returnsClosePrevMktres1',
                            'returnsOpenPrevMktres1',
                            'returnsClosePrevRaw10',
                            'returnsOpenPrevRaw10',
                            'returnsClosePrevMktres10',
                            'returnsOpenPrevMktres10'],
                 fillna_value=-1000):

    X = df_[X_columns]
    X.fillna(fillna_value, inplace=True)
    X = X.values
    return X


def get_lgb_dataset(df_,
                    X_columns=['volume',
                            'close',
                            'open',
                            'returnsClosePrevRaw1',
                            'returnsOpenPrevRaw1',
                            'returnsClosePrevMktres1',
                            'returnsOpenPrevMktres1',
                            'returnsClosePrevRaw10',
                            'returnsOpenPrevRaw10',
                            'returnsClosePrevMktres10',
                            'returnsOpenPrevMktres10']):
    

    X = get_market_X(df_)
    y = df_['returnsOpenNextMktres10'].clip(-1, 1)
    return lgb.Dataset(X, y, feature_name=X_columns, free_raw_data=False)




def get_trained_model_and_score(train_beginning,
                                train_end,
                                valid_beginning,
                                valid_end,
                                base_df,
                                model):
    # Selecting time

    df_train = base_df.loc[(base_df["time"] > pd.Timestamp(train_beginning, tz='UTC')) & (base_df["time"] < pd.Timestamp(train_end, tz='UTC'))]
    df_valid = base_df.loc[(base_df["time"] > pd.Timestamp(valid_beginning, tz='UTC')) & (base_df["time"] < pd.Timestamp(valid_end, tz='UTC'))]

    # Creating X and y arrays

    X_train = get_market_X(df_train)
    y_train = df_train[['returnsOpenNextMktres10']].values
    X_valid = get_market_X(df_valid)

    # Training the model, storing the prediction, and getting score
    model.fit(X_train, y_train)
    df_valid["pred"] = np.clip(model.predict(X_valid), -1, 1)
    score = get_score(df_valid)

    return model, score


def get_trained_model_and_score_lgb(train_beginning,
                                    train_end,
                                    valid_beginning,
                                    valid_end,
                                    base_df,
                                    lgb_params_):
    # Selecting time

    df_train = base_df.loc[(base_df["time"] > pd.Timestamp(train_beginning, tz='UTC')) & (base_df["time"] < pd.Timestamp(train_end, tz='UTC'))]
    df_valid = base_df.loc[(base_df["time"] > pd.Timestamp(valid_beginning, tz='UTC')) & (base_df["time"] < pd.Timestamp(valid_end, tz='UTC'))]
    
    sigma_score_lgb = lambda x,y: sigma_score_lgb_f(x,y,df_valid)
    
    lgb_train =  get_lgb_dataset(df_train)
    lgb_valid =  get_lgb_dataset(df_valid)

    evals_result = {}
    model = lgb.train(lgb_params_,
                      lgb_train,
                      num_boost_round=1000,
                      valid_sets=(lgb_valid,),
                      valid_names=('valid',),
                      verbose_eval=25,
                      early_stopping_rounds=100,
                      feval=sigma_score_lgb,
                      evals_result=evals_result)

    score = np.max(evals_result['valid']['sigma_score'])

    return model, score


# In[ ]:


# lgb_params = dict(
#     objective = 'regression_l1',
#     learning_rate = 0.1,
#     num_leaves = 10,
#     max_depth = -1,
# #     min_data_in_leaf = 1000,
# #     min_sum_hessian_in_leaf = 10,
#     bagging_fraction = 0.75,
#     bagging_freq = 2,
#     feature_fraction = 0.5,
#     lambda_l1 = 0.0,
#     lambda_l2 = 1.0,
#     metric = 'None', # This will ignore the loss objetive and use sigma_score instead,
#     seed = 42 # Change for better luck! :)
# )


# In[ ]:


# year = 2008
# train_beginning="{}-01-01".format(year)
# train_end="{}-01-01".format(year+1)
# valid_beginning="{}-01-01".format(year+1)
# valid_end="{}-01-01".format(year+2)

# get_trained_model_and_score_lgb(train_beginning=train_beginning,
#                                     train_end=train_end,
#                                     valid_beginning=valid_beginning,
#                                     valid_end=valid_end,
#                                     base_df=market_train,
#                                     lgb_params_=lgb_params)


# In[ ]:


# year = 2012
# train_beginning="{}-01-01".format(year)
# train_end="{}-01-01".format(year+1)
# valid_beginning="{}-01-01".format(year+1)
# valid_end="{}-01-01".format(year+2)

# m0,score = get_trained_model_and_score_lgb(train_beginning=train_beginning,
#                                     train_end=train_end,
#                                     valid_beginning=valid_beginning,
#                                     valid_end=valid_end,
#                                     base_df=market_train,
#                                     lgb_params_=lgb_params)


# In[ ]:


last_valid_df = market_train.loc[(market_train["time"] > pd.Timestamp("2016-01-01", tz='UTC')) & (market_train["time"] < pd.Timestamp("2017-01-01", tz='UTC'))]
X_valid_last = get_market_X(last_valid_df) 


# # making predictions and writting submissions

# In[ ]:


def predict_market_using_model(market_obs_df_, predictions_template_df_, model_):
    X = get_market_X(market_obs_df_)
    market_obs_df_["pred"] = np.clip(model_.predict(X), -1, 1)
    pred_dict = (market_obs_df_.set_index("assetCode")["pred"]).to_dict()
    pred_dict_f = lambda x: pred_dict[x] if x in pred_dict else 0.0 
    predictions_template_df_["confidenceValue"] = predictions_template_df_["assetCode"].apply(pred_dict_f)
    return predictions_template_df_

def write_sub_using_model(model_):
    days = env.get_prediction_days()
    for (market_obs_df, _ , predictions_template_df) in days:
        predictions_template_df_pred = predict_market_using_model(market_obs_df, predictions_template_df, model_)
        env.predict(predictions_template_df_pred)
    env.write_submission_file()
    print('Done!')


# 1. ## training different models XGBRegressor

# In[ ]:


years = range(2007, 2016)

all_XGB_models = []
all_XGB_scores = []

for year in years:
    lmodel = XGBRegressor()
    train_beginning="{}-01-01".format(year)
    train_end="{}-01-01".format(year+1)
    valid_beginning="{}-01-01".format(year+1)
    valid_end="{}-01-01".format(year+2)
    model, score = get_trained_model_and_score(train_beginning=train_beginning,
                                               train_end=train_end,
                                               valid_beginning=valid_beginning,
                                               valid_end=valid_end,
                                               base_df=market_train,
                                               model=lmodel)
    all_XGB_models.append(model)
    all_XGB_scores.append(score)
    print("train: {}   -- {}".format(train_beginning, train_end))
    print("valid: {}   -- {}".format(valid_beginning, valid_end))

    print("valid score = {:.5f}".format(score))
    print()
    
print("mean score = {:.5f}".format(np.mean(all_XGB_scores)))
print("std score = {:.5f}".format(np.std(all_XGB_scores)))


# ## training different models lightgbm

# In[ ]:


years = range(2007, 2016)

all_lgb_models = []
all_lgb_scores = []

lgb_params = dict(
    objective = 'regression_l1',
    learning_rate = 0.1,
    num_leaves = 10,
    max_depth = -1,
#     min_data_in_leaf = 1000,
#     min_sum_hessian_in_leaf = 10,
    bagging_fraction = 0.75,
    bagging_freq = 2,
    feature_fraction = 0.5,
    lambda_l1 = 0.0,
    lambda_l2 = 1.0,
    metric = 'None', # This will ignore the loss objetive and use sigma_score instead,
    seed = 42 # Change for better luck! :)
)


for year in years:
    train_beginning="{}-01-01".format(year)
    train_end="{}-01-01".format(year+1)
    valid_beginning="{}-01-01".format(year+1)
    valid_end="{}-01-01".format(year+2)
    model, score = get_trained_model_and_score_lgb(train_beginning=train_beginning,
                                                   train_end=train_end,
                                                   valid_beginning=valid_beginning,
                                                   valid_end=valid_end,
                                                   base_df=market_train,
                                                   lgb_params_=lgb_params)

    all_lgb_models.append(model)
    all_lgb_scores.append(score)
    print("train: {}   -- {}".format(train_beginning, train_end))
    print("valid: {}   -- {}".format(valid_beginning, valid_end))

    print("valid score = {:.5f}".format(score))
    print()
    
print("mean score = {:.5f}".format(np.mean(all_lgb_scores)))
print("std score = {:.5f}".format(np.std(all_lgb_scores)))


# In[ ]:


print(all_XGB_scores)
print("mean score = {:.5f}".format(np.mean(all_XGB_scores)))
print("std score = {:.5f}".format(np.std(all_XGB_scores)))


# In[ ]:


print(all_lgb_scores)
print("mean score = {:.5f}".format(np.mean(all_lgb_scores)))
print("std score = {:.5f}".format(np.std(all_lgb_scores)))


# In[ ]:


class CombinedModel:
    def __init__(self, model_list, weigths=None):
        self.model_list = model_list
        if weigths is None:
            weigths = np.random.randint(1,100,len(model_list))
            self.weigths = weigths / np.sum(weigths)
        else:
            self.weigths =  weigths

    
    def predict(self, X):
        pred = np.zeros((X.shape[0],))
        for model, weigth in zip(self.model_list,  self.weigths):
            pred += model.predict(X) *  weigth
        return pred


# In[ ]:


model2score ={}


# In[ ]:


w = np.clip(all_lgb_scores, 0, float("inf"))
w = w * 10

combined_lgb = CombinedModel(all_lgb_models, weigths=w)

last_valid_df["pred"] = np.clip(combined_lgb.predict(X_valid_last), -1, 1)
score = get_score(last_valid_df)
print(score)
model2score["combined_lgb"] = score


# In[ ]:


w = np.clip(all_XGB_scores, 0, float("inf"))
w = w * 10

combined_XGB = CombinedModel(all_XGB_models, weigths=w)

last_valid_df["pred"] = np.clip(combined_XGB.predict(X_valid_last), -1, 1)
score = get_score(last_valid_df)
print(score)
model2score["combined_XGB"] = score


# In[ ]:


w = np.array([np.mean(all_lgb_scores), np.mean(all_XGB_scores)])
w = w * 10
# w = [1,1]
w = w / np.sum(w)

combined_list = [combined_lgb, combined_XGB]


combined_m = CombinedModel(combined_list, weigths=w)
last_valid_df["pred"] = np.clip(combined_m.predict(X_valid_last), -1, 1)
score = get_score(last_valid_df)
print(score)
model2score["combined_m"] = score


# In[ ]:


lmodel = XGBRegressor(n_jobs=4, n_estimators=374, max_depth=7, eta=0.51, reg_lambda=5.0, gamma=0.127)
train_beginning="2007-01-01"
train_end="2016-01-01"
valid_beginning="2016-01-01"
valid_end="2017-01-01"

XGBRmodel_all, score = get_trained_model_and_score(train_beginning=train_beginning,
                                                   train_end=train_end,
                                                   valid_beginning=valid_beginning,
                                                   valid_end=valid_end,
                                                   base_df=market_train,
                                                   model=lmodel)

print("train: {}   -- {}".format(train_beginning, train_end))
print("valid: {}   -- {}".format(valid_beginning, valid_end))

print("valid score = {:.5f}".format(score))
print()
last_valid_df["pred"] = np.clip(XGBRmodel_all.predict(X_valid_last), -1, 1)
score = get_score(last_valid_df)
print(score)
model2score["XGBRmodel_all"] = score


# In[ ]:


train_beginning="2007-01-01"
train_end="2016-01-01"
valid_beginning="2016-01-01"
valid_end="2017-01-01"

lgbmodel_all, score = get_trained_model_and_score_lgb(train_beginning=train_beginning,
                                               train_end=train_end,
                                               valid_beginning=valid_beginning,
                                               valid_end=valid_end,
                                               base_df=market_train,
                                               lgb_params_=lgb_params)

print("train: {}   -- {}".format(train_beginning, train_end))
print("valid: {}   -- {}".format(valid_beginning, valid_end))

print("valid score = {:.5f}".format(score))
print()
last_valid_df["pred"] = np.clip(lgbmodel_all.predict(X_valid_last), -1, 1)
score = get_score(last_valid_df)
print(score)
model2score["lgbmodel_all"] = score


# In[ ]:


lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}

def prep_data(market_data):
    # add asset code representation as int (as in previous kernels)
    market_data['assetCodeT'] = market_data['assetCode'].map(lbl)
    market_col = ['assetCodeT', 'volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1', 
                        'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 
                        'returnsOpenPrevMktres10']
    # select relevant columns, fillna with zeros (where dropped in previous kernels that I saw)
    # getting rid of time, assetCode (keep int representation assetCodeT), assetName, universe
    #market_data = market_data[market_data['universe'] == True]
    X = market_data[market_col].fillna(0).values
    if "returnsOpenNextMktres10" in list(market_data.columns):#if training data
        up = (market_data.returnsOpenNextMktres10 >= 0).values
        r = market_data.returnsOpenNextMktres10.values
        universe = market_data.universe
        day = market_data.time.dt.date
        assert X.shape[0] == up.shape[0] == r.shape[0] == universe.shape[0] == day.shape[0]
    else:#observation data without labels
        up = []
        r = []
        universe = []
        day = []
    return X, up, r, universe, day


# In[ ]:


train_beginning="2007-01-01"
train_end="2016-01-01"
train_df = market_train.loc[(market_train["time"] > pd.Timestamp(train_beginning, tz='UTC')) & (market_train["time"] < pd.Timestamp(train_end, tz='UTC'))]


# In[ ]:


X, up, r, universe, day = prep_data(train_df)

# r, u and d are used to calculate the scoring metric on test
X_train, X_test, up_train, up_test, _, r_test, _, u_test, _, d_test = train_test_split(X, up, r, universe, day, test_size=0.25, random_state=99)


# In[ ]:


xgb_market = XGBClassifier(n_jobs=4, n_estimators=374, max_depth=7, eta=0.51, reg_lambda=5.0, gamma=0.127)
t = time.time()
print('Fitting Up')
xgb_market.fit(X_train,up_train)
print(f'Done, time = {time.time() - t}s')


# In[ ]:


def gen_conf(prediction_probs):
    new_conf  = np.empty([prediction_probs.shape[0]])
    for i in range(prediction_probs.shape[0]):
        if abs(prediction_probs[i][0] - prediction_probs[i][1]) < 0.01:
            new_conf[i] = 0
        elif prediction_probs[i][0] > prediction_probs[i][1]:
            new_conf[i] = - 1
        else:
            new_conf[i] = 1
    return new_conf

class ClassificationModel:
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        pred = gen_conf(self.model.predict_proba(X))
        return pred


# In[ ]:


xgb_classification =  ClassificationModel(model=xgb_market)


# In[ ]:


X, up, r, universe, day = prep_data(last_valid_df)

last_valid_df["pred"] = np.clip(xgb_classification.predict(X), -1, 1)
score = get_score(last_valid_df)
print(score)
model2score["xgb_classification"] = score
last_valid_df["xgb_classification"] = last_valid_df["pred"]


# In[ ]:


for i,m in enumerate(all_XGB_models):
    name = "XGB_model_{}".format(i) 
    last_valid_df["pred"] = np.clip(m.predict(X_valid_last), -1, 1)
    score = get_score(last_valid_df)    
    last_valid_df[name] = last_valid_df["pred"]
    model2score[name] = score


# In[ ]:


for i,m in enumerate(all_lgb_models):
    name = "lgb_model_{}".format(i) 
    last_valid_df["pred"] = np.clip(m.predict(X_valid_last), -1, 1)
    score = get_score(last_valid_df)    
    last_valid_df[name] = last_valid_df["pred"]
    model2score[name] = score


# In[ ]:


model2score


# In[ ]:


last_valid_df["combined_lgb"] = np.clip(combined_lgb.predict(X_valid_last), -1, 1)
last_valid_df["combined_m"] = np.clip(combined_m.predict(X_valid_last), -1, 1)
last_valid_df["XGBRmodel_all"] = np.clip(XGBRmodel_all.predict(X_valid_last), -1, 1)
last_valid_df["lgbmodel_all"] = np.clip(lgbmodel_all.predict(X_valid_last), -1, 1)
last_valid_df["combined_XGB"] = np.clip(combined_XGB.predict(X_valid_last), -1, 1)


# In[ ]:


select = list(model2score.keys())
select = [m for m in select if model2score[m] > 0.4]
last_valid_df_simple = last_valid_df[select]
pred_corr = last_valid_df_simple.corr()


# In[ ]:


def plot_corr(names_,
              corr_,
              title,
              cmap=plt.cm.Oranges,
              figsize=(9, 9)):
    """
    Plot a correlation matrix.
    
    cmap reference:
    https://matplotlib.org/examples/color/colormaps_reference.html
    
    :param names_: row/collum names
    :type names_: [str]
    :param corr_: matrix with correlations
    :type corr_: np.array
    :param title: image title
    :type title: str
    :param cmap: plt color map
    :type cmap: plt.cm
    :param figsize: plot's size
    :type figsize: tuple
    """
    plt.figure(figsize=figsize)
    plt.imshow(corr_, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=24, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(names_))
    plt.xticks(tick_marks, names_, rotation=45)
    plt.yticks(tick_marks, names_)
    thresh = corr_.max() / 2.
    for i, j in itertools.product(range(corr_.shape[0]), range(corr_.shape[1])):
        plt.text(j, i, format(corr_[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if corr_[i, j] > thresh else "black")

    plt.tight_layout()


# In[ ]:


for m in select:
    print(m, model2score[m])

plot_corr(pred_corr.columns,
           pred_corr.values,
           "correlation between models\n(all above 0.4)",
           cmap=plt.cm.Oranges,
           figsize=(10, 10))


# In[ ]:


# w = np.array([np.mean(all_lgb_scores), np.mean(all_XGB_scores)])
# w = w * 10
w = [1,1]
w = w / np.sum(w)

combined_list = [xgb_classification, all_lgb_models[2]]

Xclass , _, _, _, _ = prep_data(last_valid_df)

last_valid_df["pred"] = (np.clip(xgb_classification.predict(Xclass), -1, 1) + np.clip(all_lgb_models[2].predict(X_valid_last), -1, 1)) / 2 
score = get_score(last_valid_df)
print(score)


# In[ ]:


def write_sub_using_class_and_ref_models(class_model, reg_model):
    days = env.get_prediction_days()
    for (market_obs_df, _ , predictions_template_df) in days:
        Xreg = get_market_X(market_obs_df)
        Xclass , _, _, _, _ = prep_data(market_obs_df)
        market_obs_df["pred"] = (np.clip(class_model.predict(Xclass), -1, 1) + np.clip(reg_model.predict(Xreg), -1, 1)) / 2 
        pred_dict = (market_obs_df.set_index("assetCode")["pred"]).to_dict()
        pred_dict_f = lambda x: pred_dict[x] if x in pred_dict else 0.0 
        predictions_template_df["confidenceValue"] = predictions_template_df["assetCode"].apply(pred_dict_f)
        env.predict(predictions_template_df)
    env.write_submission_file()
    print('Done!')


# In[ ]:


write_sub_using_class_and_ref_models(class_model=xgb_classification, reg_model=all_lgb_models[2])


# In[ ]:


# write_sub_using_model(model_=combined_XGB)


# In[ ]:




