#!/usr/bin/env python
# coding: utf-8

# # Simple tutorial of TMDB Box Office Competition using XGB, LGB and CatGB

# In this Kernel we will see how to perform feature engineering (transform features, handle missing values and add new features) and model selection and validation for the competition "TMDB Box Office" in a very simple way.
# 
# I am also very grateful to user Serigne for its Kernel about the House Prices competition: it taught me a lot and I really recommend it to you.

# # Part 0: Import libraries and read databases

# In[ ]:


import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm,skew
import ast
from sklearn.preprocessing import MultiLabelBinarizer,LabelEncoder
from scipy.special import boxcox1p,inv_boxcox1p,boxcox
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin,clone
from sklearn.metrics import mean_squared_log_error


# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
print("Shape of train set = {}, shape of test set = {}".format(train.shape,test.shape))


# In[ ]:


train.info()


# # Part 1: Feature Engineering

# Notice that there are many non-numeric features, which we will want to transform into dummy variables. In order to do so we will need to cancatenate *train* and *test* datasets into a unique dataset *df*.
# 
# We will also drop the feature *id*, since it is useless for the analysis.

# In[ ]:


test.index=test.index+3000

df=pd.concat([train.drop("revenue",axis=1),test]).drop("id", axis=1)
y_train=train["revenue"]
print("Shape of df = {}, shape of y_train = {}".format(df.shape,y_train.shape))


# ## Part 1.1: Modify the target variable

# We want our model to perform as good as possible and in order to do so the target variable of *y_train* must be as close as possible to the normal distribution (and its probability plot as close as linear as possible). 
# 
# We will thus check which power transformation of boxcox fulfills this task better.

# In[ ]:


def visualize_distribution(y):
    sns.distplot(y,fit=norm)
    mu,sigma=norm.fit(y)
    plt.legend(["Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})".format(mu,sigma)])
    plt.title("Distribution of revenue")
    plt.ylabel("Frequency")
    plt.show()
    
    
def visualize_probplot(y):
    stats.probplot(y,plot=plt)
    plt.show()


# In[ ]:


visualize_distribution(y_train)
visualize_probplot(y_train)


# It turns out that the boxcox transformation with coefficient 0.2 is the one which makes the probability plot closer to linear: we thus apply it to *y_train*.

# In[ ]:


y_train=boxcox1p(y_train,0.2)


# **Of course, before submitting the prediction we will need to remember to perform the inverse transformation!**

# In[ ]:


visualize_distribution(y_train)
visualize_probplot(y_train)


# ## Part 1.2: Fix the features with JSON-like formatting

# Some features are in a JSON-like format and will need to be converted to dictionaries or lists before being used. We will do it with *ast.literal_eval()*.

# The features of *features_to_fix* will be transformed in lists, since we will only need the informations regarding the name.

# In[ ]:


features_to_fix=["belongs_to_collection", "genres", "production_companies", "production_countries",                 "Keywords"]

for feature in features_to_fix:
    df.loc[df[feature].notnull(),feature]=    df.loc[df[feature].notnull(),feature].apply(lambda x : ast.literal_eval(x))    .apply(lambda x : [y["name"] for y in x])


# Instead, the features *cast* and *crew* will be transformed in dictionaries

# In[ ]:


df.loc[df["cast"].notnull(),"cast"]=df.loc[df["cast"].notnull(),"cast"].apply(lambda x : ast.literal_eval(x))
df.loc[df["crew"].notnull(),"crew"]=df.loc[df["crew"].notnull(),"crew"].apply(lambda x : ast.literal_eval(x))


# ## Part 1.3: Create new features and drop some useless ones

# Now it is time to create new useful features for our model to use

# **New features involving lengths**

# In[ ]:


df["cast_len"] = df.loc[df["cast"].notnull(),"cast"].apply(lambda x : len(x))
df["crew_len"] = df.loc[df["crew"].notnull(),"crew"].apply(lambda x : len(x))

df["production_companies_len"]=df.loc[df["production_companies"].notnull(),"production_companies"].apply(lambda x : len(x))

df["production_countries_len"]=df.loc[df["production_countries"].notnull(),"production_countries"].apply(lambda x : len(x))

df["Keywords_len"]=df.loc[df["Keywords"].notnull(),"Keywords"].apply(lambda x : len(x))
df["genres_len"]=df.loc[df["genres"].notnull(),"genres"].apply(lambda x : len(x))

df['original_title_letter_count'] = df['original_title'].str.len() 
df['original_title_word_count'] = df['original_title'].str.split().str.len() 
df['title_word_count'] = df['title'].str.split().str.len()
df['overview_word_count'] = df['overview'].str.split().str.len()
df['tagline_word_count'] = df['tagline'].str.split().str.len()


# **Features underlining important characteristics of films**

# In[ ]:


df.loc[df["homepage"].notnull(),"homepage"]=1
df["homepage"]=df["homepage"].fillna(0)  # Note that we only need to know if the film has a webpage or not!

df["in_collection"]=1
df.loc[df["belongs_to_collection"].isnull(),"in_collection"]=0

df["has_tagline"]=1
df.loc[df["tagline"].isnull(),"has_tagline"]=0

df["title_different"]=1
df.loc[df["title"]==df["original_title"],"title_different"]=0

df["isReleased"]=1
df.loc[df["status"]!="Released","isReleased"]=0


# **New features from** *release_date*

# In[ ]:


release_date=pd.to_datetime(df["release_date"])
df["release_year"]=release_date.dt.year
df["release_month"]=release_date.dt.month
df["release_day"]=release_date.dt.day
df["release_wd"]=release_date.dt.dayofweek
df["release_quarter"]=release_date.dt.quarter


# **Modify** *cast* **and** *crew*
# 
# For each film we only want to consider the six most important actors (*order* is the order of importance in ascending order starting from 0)

# In[ ]:


df.loc[df["cast"].notnull(),"cast"]=df.loc[df["cast"].notnull(),"cast"].apply(lambda x : [y["name"] for y in x if y["order"]<6]) 


# From *crew* we create new features *Director*, *Producer* and *Executive Producer*

# In[ ]:


df["Director"]=[[] for i in range(df.shape[0])]
df["Producer"]=[[] for i in range(df.shape[0])]
df["Executive Producer"]=[[] for i in range(df.shape[0])]

df["Director"]=df.loc[df["crew"].notnull(),"crew"].apply(lambda x : [y["name"] for y in x if y["job"]=="Director"])

df["Producer"]=df.loc[df["crew"].notnull(),"crew"].apply(lambda x : [y["name"] for y in x if y["job"]=="Producer"])

df["Executive Producer"]=df.loc[df["crew"].notnull(),"crew"].apply(lambda x : [y["name"] for y in x if y["job"]=="Executive Producer"])


# **Now it is very important to delete all useless features which will not be used in the analysis**

# In[ ]:


df=df.drop(["imdb_id","original_title","overview","poster_path","tagline","status","title",           "spoken_languages","release_date","crew"],axis=1)


# ## Part 1.4: Handle missing values and add even more new features

# First of all let's print the list of percentages of missing values in descending order

# In[ ]:


mis_val=((df.isnull().sum()/df.shape[0])*100).sort_values(ascending=False)
mis_val=mis_val.drop(mis_val[mis_val==0].index)
print(mis_val)


# The features of *to_empty_list* are lists and consequently their missing values will be replaced with empty lists

# In[ ]:


to_empty_list=["belongs_to_collection","Keywords","production_companies","production_countries",              "Director","Producer","Executive Producer","cast","genres"]

for feature in to_empty_list:
    df[feature] = df[feature].apply(lambda d: d if isinstance(d, list) else [])


# The missing values of time and length will be replaced with 0

# In[ ]:


to_zero=["runtime","release_month","release_year","release_wd","release_quarter","release_day"]+["Keywords_len","production_companies_len","production_countries_len","crew_len","cast_len","genres_len",
    "tagline_word_count","overview_word_count","title_word_count"]

for feat in to_zero:
    df[feat]=df[feat].fillna(0)


# **Now it is the best time to add new features, since they will not be affected by the missing values of other features**

# In[ ]:


df['_budget_popularity_ratio'] = df['budget']/df['popularity']
df['_releaseYear_popularity_ratio'] = df['release_year']/df['popularity']
df['_releaseYear_popularity_ratio2'] = df['popularity']/df['release_year']


# Finally, we can see there are no missing values!

# In[ ]:


mis_val=((df.isnull().sum()/df.shape[0])*100).sort_values(ascending=False)
mis_val=mis_val.drop(mis_val[mis_val==0].index)
print(mis_val)


# ## Part 1.6: Modify numeric features

# As we did in Part 1.1 for the target variable, it is now important to modify the numeric features in order to make them as close as possible to the normal distribution.
# 
# First of all let's select the numeric features which has a high skewness

# In[ ]:


numeric=[feat for feat in df.columns if df[feat].dtype!="object"]

skewness=df[numeric].apply(lambda x : skew(x)).sort_values(ascending=False)
skew=skewness[skewness>2.5]
print(skew)


# Let's reduce the  skewness of these features by applying a power transformation

# In[ ]:


high_skew=skew[skew>10].index
medium_skew=skew[skew<=10].index

for feat in high_skew:
    df[feat]=np.log1p(df[feat])

for feat in medium_skew:
    df[feat]=df[feat]=boxcox1p(df[feat],0.15)


# Indeed, we can se the skewness is really improved:

# In[ ]:


skew=df[skew.index].skew()
print(skew)


# Finally, let's visualize the distribution of these features

# In[ ]:


plt.figure(figsize=(15,40))
for i,feat in enumerate(skew.index):
    plt.subplot(7,2,i+1)
    sns.distplot(df[feat],fit=norm)
    mu,sigma=norm.fit(df[feat])
    plt.legend(["Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})".format(mu,sigma)])
    plt.title("Distribution of "+feat)
    plt.ylabel("Frequency")
plt.show()


# It is also useful to feed the feature *release_year* to the label encoder

# In[ ]:


lbl=LabelEncoder()
lbl.fit(df["release_year"].values)
df["release_year"]=lbl.transform(df["release_year"].values)


# ## Get dummy variables and separate train and test set

# We are going to get dummy variables from all the features which are not numeric

# In[ ]:


to_dummy = ["belongs_to_collection","genres","original_language","production_companies","production_countries",           "Keywords","cast","Director","Producer","Executive Producer"]


# Notice that the entries of the features of the list *to_dummy* are lists: in each feature, for each element of each list we want to get a dummy variable only if such element is "famous" enough, which is if it appears in a sufficient number of lists. In order to do so we use the list *limits*, whose entries are, for each feature, the minimum numbers of appearences which an element must have in order to get a dummy variable

# In[ ]:


limits=[4,0,0,35,10,40,10,5,10,12] 

for i,feat in enumerate(to_dummy):
    mlb = MultiLabelBinarizer()
    s=df[feat]
    x=pd.DataFrame(mlb.fit_transform(s),columns=mlb.classes_, index=df.index)
    y=pd.DataFrame(mlb.fit_transform(s),columns=mlb.classes_, index=df.index).sum().sort_values(ascending=False)
    rare_entries=y[y<=limits[i]].index
    x=x.drop(rare_entries,axis=1)
    df=df.drop(feat,axis=1)
    df=pd.concat([df, x], axis=1, sort=False)


# In[ ]:


print("The final total number of features is {}".format(df.shape[1]))


# Finally, after having obtained all the dummy variables we needed, we can separate *df* back in *train* and *test*

# In[ ]:


ntrain=train.shape[0]

train=df.iloc[:ntrain,:]
test=df.iloc[ntrain:,:]
print("The shape of train DataFrame is {} and the shape of the test DataFrame is {}".format(train.shape,test.shape))


# # Part 2: Build the model and make the predictions

# In this Kernel we will use XGB,LGBM and CatBoost regressors, since they are the state of the art in prediction of tabular data

# In[ ]:


model_xgb=xgb.XGBRegressor(max_depth=5,
                           learning_rate=0.1, 
                           n_estimators=2000, 
                           objective='reg:linear', 
                           gamma=1.45, 
                           verbosity=3,
                           subsample=0.7, 
                           colsample_bytree=0.8, 
                           colsample_bylevel=0.50)


# In[ ]:


model_lgb=lgb.LGBMRegressor(n_estimators=10000, 
                             objective="regression", 
                             metric="rmse", 
                             num_leaves=20, 
                             min_child_samples=100,
                             learning_rate=0.01, 
                             bagging_fraction=0.8, 
                             feature_fraction=0.8, 
                             bagging_frequency=1, 
                             subsample=.9, 
                             colsample_bytree=.9,
                             use_best_model=True)


# In[ ]:


model_cat = cat.CatBoostRegressor(iterations=10000,learning_rate=0.01,depth=5,eval_metric='RMSE',                              colsample_bylevel=0.7,
                              bagging_temperature = 0.2,
                              metric_period = None,
                              early_stopping_rounds=200)


# ## Part 2.1: Perform cross-validation

# Before training the models on the whole *train* DataFrame, it is interesting to perform some cross validation, in order to get an idea of how well the models are performing.
# 
# Here the cross validations are commented since they require a lot of time

# In[ ]:


n_folds=5

def cross_val(model):
    cr_val=np.sqrt(-cross_val_score(model,train.values,y_train.values,scoring="neg_mean_squared_log_error",cv=5))
    return cr_val


# In[ ]:


#print "\n mean XGB score = {:.3f} with std = {:.3f}".format(cross_val(model_xgb).mean(),cross_val(model_xgb).std())


# In[ ]:


#print "\n mean LGB score = {:.3f} with std = {:.3f}".format(cross_val(model_lgb).mean(),cross_val(model_lgb).std())


# In[ ]:


#print "\n mean LGB score = {:.3f} with std = {:.3f}".format(cross_val(model_cat).mean(),cross_val(model_cat).std())


# ## Part 2.2: Train the models

# We train the model on the whole *train* DataFrame and compute the loss function using the function *msle*

# In[ ]:


def msle(y,y_pred):
    return np.sqrt(mean_squared_log_error(y,y_pred))


# In[ ]:


ti=time.time()
model_lgb.fit(train.values,y_train)
print("Number of minutes of training of model_lgb = {:.2f}".format((time.time()-ti)/60))

lgb_pred_train=model_lgb.predict(train.values)
print("Mean square logarithmic error of lgb model on whole train = {:.4f}".format(msle(y_train,lgb_pred_train)))


# In[ ]:


ti=time.time()
model_xgb.fit(train.values,y_train)
print("Number of minutes of training of model_xgb = {:.2f}".format((time.time()-ti)/60))

xgb_pred_train=model_xgb.predict(train.values)
print("Mean square logarithmic error of xgb model on whole train = {:.4f}".format(msle(y_train,xgb_pred_train)))


# In[ ]:


ti=time.time()
model_cat.fit(train.values,y_train,verbose=False)
print("Number of minutes of training of model_cal = {:.2f}".format((time.time()-ti)/60))

cat_pred_train=model_cat.predict(train.values)
cat_pred_train[cat_pred_train<0]=0
print("Mean square logarithmic error of cat model on whole train = {:.4f}".format(msle(y_train,cat_pred_train)))


# Finally, our prediction will be the weighted mean (using the coefficients of c) of the predictions of the three models. Clearly, the sum of the entries of c must be equal to 1.

# In[ ]:


c = np.array([0.333334,0.333333,0.333333])

print("The sum of the entries of c is {}".format(c.sum()))

train_pred=xgb_pred_train*c[0]+lgb_pred_train*c[1]+cat_pred_train*c[2]
print("Mean square logarithmic error of chosen model on whole train = {:.4f}".format(msle(y_train,train_pred)))


# In[ ]:


plt.figure(figsize=(30,10))
plt.plot(y_train[:500],label="y_train")
plt.plot(train_pred[:500],label="train_pred")
plt.legend(fontsize=15)
plt.title("Real and predicted revenue of first 500 entries of train set",fontsize=24)
plt.show()


# In[ ]:


plt.figure(figsize=(30,10))
plt.plot(y_train.values[-500:],label="y_train")
plt.plot(train_pred[-500:],label="train_pred")
plt.legend(fontsize=15)
plt.title("Real and predicted revenue of last 500 entries of train set",fontsize=24)
plt.show()


# ## Part 2.3: Make the predictions and write them to the submission form

# In[ ]:


lgb_pred=model_lgb.predict(test)
xgb_pred=model_xgb.predict(test.values)
cat_pred=model_cat.predict(test)


# When computing the predictions we must remember to apply the inverse of the tranformation we applied on *y_train* in Part 1.1

# In[ ]:


pred=inv_boxcox1p((xgb_pred*c[0]+lgb_pred*c[1]+cat_pred*c[2]),0.2)

sub=pd.DataFrame({"id":np.arange(test.shape[0])+3001,"revenue":pred})
sub.to_csv("my_submission.csv",index=False)


# In[ ]:




