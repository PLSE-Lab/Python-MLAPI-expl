#!/usr/bin/env python
# coding: utf-8

# # TMDB Revenue prediction - simple baseline using only numeric features + length of complex ones

# ### imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PowerTransformer, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_log_error
import holoviews as hv
hv.extension('bokeh')

from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression, ElasticNetCV

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')
print(os.listdir("../input"))


# ### utils

# In[ ]:


relu = np.vectorize(lambda x: x if x>0.0 else 0.0)

def create_feat_len(df, feat_name):
    _df = df.copy()
    _df[f"{feat_name}_count"] = _df[feat_name].str.count("name")
    return _df

def split_x_y(df, target="revenue"):
    return df.drop(columns=[target]), df[target]

def cross_val_pipeline(pipe, X, y):
    res = cross_val_score(pipe, X, y, scoring="neg_mean_squared_error", cv=5, n_jobs=2)
    res = np.sqrt(-res)
    mean_res, std_res = np.mean(res), np.std(res)
    print(f"RMSLE: {mean_res:2.3} +- {std_res:3.2}")


# ## 0. Data extraction: import and show data

# In[ ]:


df_train = pd.read_csv("../input/tmdb-box-office-prediction/train.csv")
df_test = pd.read_csv("../input/tmdb-box-office-prediction/test.csv")

df_train.head()


# ## Missing values

# In[ ]:


df_train.isna().mean()


# # 1. Data exploration: Hypothesis validation

# # Is the date of release relevant?

# In[ ]:


df_aux = df_train.copy()
df_aux['release_date'] = pd.to_datetime(df_aux.release_date)
df_dates_to_correct = df_aux['release_date'] > pd.datetime(2019, 2, 1)
df_aux.loc[df_dates_to_correct, 'release_date'] = df_aux.loc[df_dates_to_correct, 'release_date']                                                            .apply(lambda t: pd.datetime(t.year-100, t.month, t.day))
df_aux = df_aux.set_index("release_date").sort_index()
# df_aux.resample("Y").mean().revenue.plot()
df_aux["is_american"] = df_aux.production_countries.str.contains("United States of America")


# In[ ]:


plt.figure(figsize=(18,5))
df_aux.revenue.rolling('180d').mean().plot(label='90 rolling mean')
df_aux.revenue.rolling('180d').quantile(0.9).plot(label='90 rolling quantile 90')
plt.grid()
plt.legend()
plt.show()

df_aux2 = df_aux[~df_aux.is_american.isna()]
df_aux2["is_american"]=df_aux2["is_american"].astype(np.bool)
plt.figure(figsize=(18,5))
df_aux2[df_aux2.is_american].resample("y").mean().revenue.dropna().plot(label="American movie")
df_aux2[~df_aux2.is_american].resample("y").mean().revenue.dropna().plot(label='Non American movie')
plt.ylabel("Revenue", fontsize=14)
plt.xlabel("Release date", fontsize=14)
plt.title("Average revenue per year of American and Non-american movies", fontsize=14)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# # top 5 movies in rev and 

# In[ ]:


df_train_clean = pd.read_csv('../input/moviestmdb-datapreparation/train_prep.csv')


# In[ ]:


best_movies = df_train.sort_values(by='revenue', ascending=False).iloc[:10]


# In[ ]:


best_movies[["original_title", "popularity", "budget", "release_date","revenue"]].set_index(np.arange(1,11))


# In[ ]:


## 5 best actors


# In[ ]:


l_actors = [c for c in df_train_clean.columns if "cast_name" in c]
l_series_of_actors=[]
for c in l_actors:
    asd =df_train_clean[df_train_clean[c]==1].agg({"revenue":"mean","budget":"mean","popularity":"mean"})
    asd.name=c.replace("cast_name_","")
    l_series_of_actors.append(asd)
df_actors = pd.DataFrame(l_series_of_actors).reset_index()
df_actors.sort_values(by='revenue', ascending=False).iloc[:10].set_index(np.arange(1,11)).rename(columns={"index":"Actor"})


# In[ ]:


plt.scatter(df_actors.popularity, 
            df_actors.budget)
# plt.scatter(df_actors.popularity, 
#             df_actors.revenue)


# In[ ]:


asd =df_train_clean[df_train_clean[l_actors[0]]==1].agg({"revenue":"mean","budget":"mean","popularity":"mean"})
asd.name=l_actors[0].replace("cast_name_","")
pd.DataFrame([asd])


# # American vs non american movies

# In[ ]:





# In[ ]:


df_aux = df_train.copy()
df_aux["is_american"] = df_aux.production_countries.str.contains("United States of America")
df_aux["revenue"]=df_aux.revenue.apply(np.log1p)
df_aux = df_aux.dropna(subset=["is_american"])
df_aux["is_american"] = df_aux["is_american"].astype(np.bool)

plt.figure(figsize=(7,4))
sns.distplot(df_aux[df_aux['is_american']].revenue, label='American')
sns.distplot(df_aux[~df_aux['is_american']].revenue, label='Non American')
plt.legend()
plt.show()


# In[ ]:



plt.hist(df_aux[df_aux['is_american']].revenue, label='Non American', alpha=0.5)
plt.hist(df_aux[~df_aux['is_american']].revenue, label='Non American', alpha=0.5)


# In[ ]:


df_aux2 = df_aux[~df_aux.is_american.isna()]
df_aux2["is_american"]=df_aux2["is_american"].astype(np.bool)
plt.figure(figsize=(18,5))
df_aux2[df_aux2.is_american].resample("y").mean().revenue.dropna().plot(label="american movie")
df_aux2[~df_aux2.is_american].resample("y").mean().revenue.dropna().plot(label='non american movie')
plt.ylabel("Revenue", fontsize=14)
plt.xlabel("Release Date", fontsize=14)
plt.title("Revenue for American vs Non-american movies", fontsize=14)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.ylim(0,2)
plt.show()


# In[ ]:


plt.figure(figsize=(18,5))
df_aux[df_aux.original_language=='en'].revenue.rolling('30d').mean().plot(label='90 rolling mean revenue of english movies')
df_aux[df_aux.original_language!='en'].revenue.rolling('30d').mean().plot(label='90 rolling mean revenue of non-english movies')
plt.grid()
plt.legend()
plt.show()


# # Released feature - is it useful?

# In[ ]:


df_train.status.value_counts()


# Nope...

# ## Homepage exists is relevant?

# In[ ]:


df_train["has_homepage"] = (~df_train.homepage.isna()).astype(int)


# In[ ]:


sns.distplot(df_train[1==df_train.has_homepage].revenue, kde=False)
sns.distplot(df_train[0==df_train.has_homepage].revenue, kde=False)
plt.yscale("log")


# Yes!

# ## Does it matter if the film is american or not?

# In[ ]:


df_aux = df_train.copy().dropna(subset=["production_countries"])
df_aux["is_american"] = df_aux.production_countries.str.contains("United States of America")

sns.distplot(df_aux[df_aux.is_american].revenue, kde=False, label="Yes")
sns.distplot(df_aux[~df_aux.is_american].revenue, kde=False, label="No")
plt.title("Is the film american?")
plt.yscale("log")
plt.legend()
plt.show()


# Yes!

# ## What about being in english?

# In[ ]:


df_aux = df_train.copy().dropna(subset=["original_language"])
df_aux["is_english"] = df_aux.original_language=="en"

sns.distplot(df_aux[df_aux.is_english].revenue, kde=False, label="Yes")
sns.distplot(df_aux[~df_aux.is_english].revenue, kde=False, label="No")
plt.title("Is the film english?")
plt.yscale("log")
plt.legend()
plt.show()


# Yes!!

# ## Does it matter if belongs to collection?

# In[ ]:


df_aux = df_train.copy()
df_aux["belongs_to_a_collection"] = ~df_aux.belongs_to_collection.isna()

sns.distplot(df_aux[df_aux.belongs_to_a_collection].revenue, kde=False, label="Yes")
sns.distplot(df_aux[~df_aux.belongs_to_a_collection].revenue, kde=False, label="No")
plt.title("Is the film english?")
plt.yscale("log")
plt.legend()
plt.show()


# Yes!!

# ## Adding count of complex features (crew, cast, etc)

# In[ ]:


ref_date = pd.datetime(2019, 2, 1)
l_complex_feats = ["production_companies", "production_countries", "spoken_languages", "cast", "crew"]
def apply_custom_transformations(df):
    df_res = reduce(create_feat_len, l_complex_feats, df.copy())
    df_res["has_homepage"] = (~df_res.homepage.isna()).astype(np.float)
    df_res["is_english"] = (df_res.original_language=="en").astype(np.float)
    df_res["is_american"] = df_res.production_countries.str.contains("United States of America").astype(np.float)
    df_res["belongs_to_a_collection"] = (~df_res.belongs_to_collection.isna()).astype(np.float)
    df_res['release_date'] = pd.to_datetime(df_res.release_date)
    df_dates_to_correct = df_res['release_date'] > ref_date
    df_res.loc[df_dates_to_correct, 'release_date'] = df_res.loc[df_dates_to_correct, 'release_date']                                                                .apply(lambda t: pd.datetime(t.year-100, t.month, t.day))
    df_res['days_since_release'] = (df_res.release_date - ref_date).dt.days
    df_res = df_res.select_dtypes(np.number).set_index("id")
    
    # if train move target column to last
    if 'revenue' in df_res.columns:
        df_res = df_res[df_res.columns.drop("revenue").tolist() + ["revenue"]]
        df_res['revenue'] = np.log(df_res['revenue'])
    return df_res

custom_transformer = FunctionTransformer(apply_custom_transformations, validate=False)
df_FE=custom_transformer.fit_transform(df_train)


# In[ ]:


# sns.pairplot(df_FE, diag_kind='kde', markers = '+')
# plt.show()


# In[ ]:


sns.heatmap(df_FE.corr())


# # 2. Training: Time to start predicting

# In[ ]:


X, y = split_x_y(df_train)
logy = np.log(y)


# ### Linear regression

# In[ ]:


pipe = make_pipeline(custom_transformer, SimpleImputer(strategy='median'), StandardScaler(), 
                    ElasticNetCV(cv = 5, n_jobs=2))
cross_val_pipeline(pipe, X, logy)


# ### XGB with linear regressor booster

# In[ ]:


pipe = make_pipeline(custom_transformer, SimpleImputer(strategy='median'), StandardScaler(), 
                    XGBRegressor(booster='gblinear', n_estimators=500, n_jobs=2))
cross_val_pipeline(pipe, X, logy)


# ### XGB with regressor trees

# In[ ]:


model = XGBRegressor(max_depth=5, n_estimators=50)
pipe = make_pipeline(custom_transformer, SimpleImputer(strategy='median'), StandardScaler(), model)
cross_val_pipeline(pipe, X, logy)


# In[ ]:


# points = hv.Points(pd.DataFrame([y_pred, np.log(y_val)], index=["y_pred","y_true"]).T)
# points


# In[ ]:


pipe.fit(X, logy)
logy_pred_test = pipe.predict(df_test)
y_pred_test = np.exp(logy_pred_test)

df_pred = df_test[['id']].copy()
df_pred['revenue']=y_pred_test
df_pred.to_csv("submission_1.csv", index=False)

