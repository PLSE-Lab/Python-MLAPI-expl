#!/usr/bin/env python
# coding: utf-8

# In this notebook, we will focus on filling the missing values. We define a function that will do us all the data cleansing stuff for us. We will then use it to preprocess our data and then for prediction.

# In[ ]:


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as mplt
import seaborn as sns
import xgboost as xgb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv("../input/train.csv", parse_dates=["timestamp"], date_parser=lambda x: pd.datetime.strptime(x, "%Y-%m-%d"))


# In[ ]:


test = pd.read_csv("../input/test.csv", parse_dates=["timestamp"], date_parser=lambda x: pd.datetime.strptime(x, "%Y-%m-%d"))


# In[ ]:


def data_cleaning(data, ret_f=True):
    features = list()
    
    data["t_year"] = data.timestamp.map(lambda x: x.year)
    data["t_year_month"] = data.timestamp.map(lambda x: x.year * 100 + x.month)
    data["t_month"] = data.timestamp.map(lambda x: x.month)
    
    features.append("t_year_month")
    
    # build_year
    data.loc[np.absolute(data.build_year - data.t_year) > 150, "build_year"] = data.t_year
    build_dict = data[["build_year", "sub_area"]].groupby("sub_area").aggregate(lambda x: stats.mode(x).mode[0]).to_dict()["build_year"]
    data.loc[data.build_year.isnull(), "build_year"] = data.sub_area.map(lambda x: build_dict[x])
    data["age_at_transact"] = data.t_year - data.build_year
    features.append("age_at_transact")
    
    # full_sq
    data.loc[data.full_sq < 10, "full_sq"] = 10
    features.append("full_sq")
    
    # product-type
    data["building_type"] = data.product_type.map(lambda x: 0 if x == "Investment" else 1)
    features.append("building_type")
    
    # life_sq
    life_sq_mode = stats.mode(data.life_sq).mode[0]
    data["life_sq"] = data.life_sq.fillna(life_sq_mode)
    data.loc[data.life_sq > data.full_sq, "life_sq"] = data.full_sq
    features.append("life_sq")
    
    # floor
    data["floor"] = data.floor.fillna(method="ffill")
    
    # max_floor
    max_by_area = data[["sub_area", "max_floor"]].groupby("sub_area").aggregate(np.mean).to_dict()["max_floor"]
    data["max_floor"] = data.sub_area.map(lambda x: max_by_area[x])
    data.loc[data.floor > data.max_floor, "max_floor"] = data.floor
    data["home_height"] = data.floor / data.max_floor
    features.append("home_height")
    
    # kitch_sq
    kitch_mean = data.kitch_sq.mean()
    data["kitch_sq"] = data.kitch_sq.fillna(kitch_mean)
    data.loc[data.kitch_sq > data.full_sq, "kitch_sq"] = data.life_sq / 2
    features.append("kitch_sq")
    
    # num_room
    clean_room = data.num_room.mode()
    data.loc[data.num_room > 8, "num_room"] = clean_room
    fill_val = int(data.num_room.median())
    data.loc[data.num_room == 0, "num_room"] = fill_val
    data["num_room"] = data.num_room.fillna(fill_val)
    features.append("num_room")
    
    # material
    md_mtrl = data.material.mode()
    data["material"] = data.material.fillna(md_mtrl)
    features.append("material")
    
    # product_type
    data["product_category"] = data.product_type.map(lambda x: 0 if x == "Investment" else 1)
    features.append("product_category")
    
    # state
    data.loc[(data.state < 1) & (data.state > 4), "state"] = np.nan
    fil_val = data.state.mode()
    data["building_state"] = data.state.fillna(fil_val)
    features.append("building_state")
    
    #
    # NEIGHBOURHOOD FEATURES
    #
    # area_m
    min_area = data.area_m.mean()
    data["area_m"] = data.area_m.fillna(min_area)
    features.append("area_m")
    
    # male_f, female_f
    data["pop_ratio"] = data["male_f"] / data["female_f"]
    features.append("pop_ratio")
    
    if ret_f:
        return data, features
    else:
        return data


# In[ ]:


train, features = data_cleaning(train, ret_f=True)
test = data_cleaning(test, ret_f=False)
train["target"] = np.log(train.price_doc)


# In[ ]:


cols = features[:]
cols.append("target")
corr = train[cols].corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig, ax = mplt.subplots()
fig.set_size_inches(10, 10)
sns.heatmap(corr, mask=mask, annot=True, ax=ax, square=True)


# Still there are is some multicollinearity. Will try to handle them....

# In[ ]:


train_arr = xgb.DMatrix(train[features], train["target"])
test_arr = xgb.DMatrix(test[features])


# In[ ]:


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
cv_output = xgb.cv(xgb_params, train_arr, num_boost_round=100, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)
cv_output[["train-rmse-mean", "test-rmse-mean"]].plot()


# In[ ]:


num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), train_arr, num_boost_round= num_boost_rounds)


# In[ ]:


featureImportance = model.get_fscore()
ftrs = pd.DataFrame()
ftrs['features'] = featureImportance.keys()
ftrs['importance'] = featureImportance.values()
ftrs.sort_values(by=['importance'],ascending=False,inplace=True)
fig,ax= mplt.subplots()
fig.set_size_inches(20,10)
mplt.xticks(rotation=60)
sns.barplot(data=ftrs.head(30),x="features",y="importance",ax=ax,orient="v")


# In[ ]:


test["predicted_price"] = model.predict(test_arr)
test["price_doc"] = np.exp(test["predicted_price"])


# In[ ]:


fig, ax = mplt.subplots(ncols=2)
stats.probplot(np.log(train.price_doc), plot=ax[0])
stats.probplot(test.predicted_price, plot=ax[1])
ax[0].set_title("Training Data")
ax[1].set_title("Test Data")


# In[ ]:


test[["id", "price_doc"]].head()


# In[ ]:


test[["id", "price_doc"]].to_csv("submission_xgb.csv", index=False)


# Kindly, upvote if you find it useful.
