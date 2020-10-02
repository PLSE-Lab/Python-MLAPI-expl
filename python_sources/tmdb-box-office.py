#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.model_selection import train_test_split
import lightgbm as lgb


# In[ ]:


train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')
test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')
sample_submission = pd.read_csv('../input/tmdb-box-office-prediction/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


print("Number of rows(train): "+str(len(train)))
print("Number of rows(test): "+str(len(test)))


# In[ ]:


train.info()


# #### Collection

# In[ ]:


train['has_collection'] = train['belongs_to_collection'].apply(lambda x: 1 if str(x) != 'nan' else 0)
train['collection_id'] = train['belongs_to_collection'].apply(lambda x: eval(x)[0]['id'] if str(x) != 'nan' else 0)

test['has_collection'] = test['belongs_to_collection'].apply(lambda x: 1 if str(x) != 'nan' else 0)
test['collection_id'] = test['belongs_to_collection'].apply(lambda x: eval(x)[0]['id'] if str(x) != 'nan' else 0)


# In[ ]:


train = train.drop(['belongs_to_collection'], axis=1)
test = test.drop(['belongs_to_collection'], axis=1)


# #### Genres

# In[ ]:


list_of_genres = list(train['genres'].apply(lambda x: [i['name'] for i in eval(x)] if str(x) != 'nan' else []).values)


# In[ ]:


Counter([i for j in list_of_genres for i in j]).most_common(15)


# In[ ]:


top_genres = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common(15)]

train['all_genres'] = train['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in eval(x)])) if (isinstance(x,int) or isinstance(x,str)) == True else '')
for gen in top_genres:
    train['genre_' + gen] = train['all_genres'].apply(lambda x: 1 if gen in x else 0)
    
test['all_genres'] = test['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in eval(x)])) if (isinstance(x,int) or isinstance(x,str)) == True else '')
for gen in top_genres:
    test['genre_' + gen] = test['all_genres'].apply(lambda x: 1 if gen in x else 0)
    
train = train.drop(['genres'], axis=1)
test = test.drop(['genres'], axis=1)


# In[ ]:


lang_encoder = LabelEncoder()
train['all_genres'] = lang_encoder.fit_transform(train['all_genres'])
test['all_genres'] = lang_encoder.fit_transform(test['all_genres'])


# #### Original Language

# In[ ]:


train['original_language'].unique()


# In[ ]:


lang_encoder = LabelEncoder()
train['original_language'] = lang_encoder.fit_transform(train['original_language'])
test['original_language'] = lang_encoder.fit_transform(test['original_language'])


# #### Production Companies

# In[ ]:


prod_companies = list(train['production_companies'].apply(lambda x: [i['name'] for i in eval(x)] if str(x) != 'nan' else '').values)


# In[ ]:


train['prod_companies_count'] = train['production_companies'].apply(lambda x: len([i for i in eval(x)]) if str(x) != 'nan' else 0)
test['prod_companies_count'] = test['production_companies'].apply(lambda x: len([i for i in eval(x)]) if str(x) != 'nan' else 0)


# Let's try to make a new feature called "production score". Score will be calculated on the popularity of production company( that is given by the number of times it has appeared in the dataset). If the movie have multiple productions then the score will be assigned by the most popular production.

# In[ ]:


pop_production = Counter([i for j in prod_companies for i in j])


# In[ ]:


train['production_score'] = train['production_companies'].apply(lambda x: np.tanh(max([pop_production[i['name']] for i in eval(x)])) if str(x) != 'nan' else 0)
test['production_score'] = test['production_companies'].apply(lambda x: np.tanh(max([pop_production[i['name']] for i in eval(x)])) if str(x) != 'nan' else 0)


# In[ ]:


train = train.drop(['production_companies'], axis=1)
test = test.drop(['production_companies'], axis=1)


# #### Production Countries

# In[ ]:


train['production_countries'] = train['production_countries'].apply(lambda x: [i['name'] for i in eval(x)][0] if str(x) != 'nan' else '')
test['production_countries'] = test['production_countries'].apply(lambda x: [i['name'] for i in eval(x)][0] if str(x) != 'nan' else '')


# In[ ]:


prod_country_encoder = LabelEncoder()
train['production_countries'] = prod_country_encoder.fit_transform(train['production_countries'])
test['production_countries'] = prod_country_encoder.fit_transform(test['production_countries'])


# In[ ]:


train['production_countries'].head()


# #### Release Date

# In[ ]:


train['release_date'] = train['release_date'].apply(lambda x: pd.to_datetime(x))
test['release_date'] = test['release_date'].apply(lambda x: pd.to_datetime(x))


# In[ ]:


train['year'] = train['release_date'].apply(lambda x: x.year)
train['month'] = train['release_date'].apply(lambda x: x.month)
train['day_of_week'] = train['release_date'].apply(lambda x: x.weekday())

test['year'] = test['release_date'].apply(lambda x: x.year)
test['month'] = test['release_date'].apply(lambda x: x.month)
test['day_of_week'] = test['release_date'].apply(lambda x: x.weekday())


# In[ ]:


train = train.drop(['release_date'], axis=1)
test = test.drop(['release_date'], axis=1)


# #### Anomalies in release date

# From the plot below we can observe a very weird trend that there are a lot of movies that have a release date like 2067, 2024 and so on. It turns out that this is just a typo. For example, the movie *Major Dundee* was released in 1965 whereas in the dataset it's release year is 2065. So we can just replace it with 1965.
# 
# ![Major Dundee](https://image.tmdb.org/t/p/w600_and_h900_bestv2/skv6Jsw6YyPKV4oQhs88zvwDCAL.jpg)

# In[ ]:


_ = sns.lineplot(x=train['year'], y=train['revenue'], color='r')


# In[ ]:


train['year'] = train['year'].apply(lambda x: x-100 if x>2020 else x)
test['year'] = test['year'].apply(lambda x: x-100 if x>2020 else x)


# In[ ]:


_ = sns.lineplot(x=train['year'], y=train['revenue'], color='g')


# #### Runtime

# In[ ]:


avg_runtime_train = train['runtime'].mean()
train['runtime'] = train['runtime'].apply(lambda x: x if str(x) != 'nan' else avg_runtime_train)

avg_runtime_test = test['runtime'].mean()
test['runtime'] = test['runtime'].apply(lambda x: x if str(x) != 'nan' else avg_runtime_test)


# In[ ]:


_ = sns.distplot(train['runtime'])


# #### Budget

# In[ ]:


train[train['budget']==0].head() 


# #### Spoken Languages

# In[ ]:


train['spoken_languages_count'] = train['spoken_languages'].apply(lambda x: len(eval(x)) if str(x) != 'nan' else 0)
test['spoken_languages_count'] = test['spoken_languages'].apply(lambda x: len(eval(x)) if str(x) != 'nan' else 0)


# In[ ]:


_ = sns.countplot(x=train['spoken_languages_count'])


# In[ ]:


train = train.drop(['spoken_languages'], axis=1)
test = test.drop(['spoken_languages'], axis=1)


# #### Keywords

# In[ ]:


train['Keywords']


# In[ ]:


list_of_keywords = list(train['Keywords'].apply(lambda x: [i['name'] for i in eval(x)] if str(x) != 'nan' else []).values)


# In[ ]:


train['num_Keywords'] = train['Keywords'].apply(lambda x: len(eval(x)) if str(x) != 'nan' else 0)
train['all_Keywords'] = train['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in eval(x)])) if str(x) != 'nan' else '')

top_keywords = [m[0] for m in Counter([i for j in list_of_keywords for i in j]).most_common(30)]

for g in top_keywords:
    train['keyword_'+g] = train['all_Keywords'].apply(lambda x: 1 if g in x else 0)
    
    
    
test['num_Keywords'] = test['Keywords'].apply(lambda x: len(eval(x)) if str(x) != 'nan' else 0)
test['all_Keywords'] = test['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in eval(x)])) if str(x) != 'nan' else '')

top_keywords = [m[0] for m in Counter([i for j in list_of_keywords for i in j]).most_common(30)]

for g in top_keywords:
    test['keyword_'+g] = test['all_Keywords'].apply(lambda x: 1 if g in x else 0)


# In[ ]:


keywords_encoder = LabelEncoder()
train['all_Keywords'] = keywords_encoder.fit_transform(train['all_Keywords'])
test['all_Keywords'] = keywords_encoder.fit_transform(test['all_Keywords'])


# In[ ]:


train = train.drop(['Keywords'], axis=1)
test = test.drop(['Keywords'], axis=1)


# #### Cast

# In[ ]:


train['cast'] = train['cast'].apply(lambda x: eval(x) if str(x) != 'nan' else [])
test['cast'] = test['cast'].apply(lambda x: eval(x) if str(x) != 'nan' else [])


# In[ ]:


train['gender_0_count'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
train['gender_1_count'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
train['gender_2_count'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

test['gender_0_count'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
test['gender_1_count'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
test['gender_2_count'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))


# In[ ]:


list_of_cast_members = list(train['cast'].apply(lambda x: [i['name'] for i in x] if str(x) != 'nan' else []).values)


# In[ ]:


top_cast_members = [m[0] for m in Counter([i for j in list_of_cast_members for i in j]).most_common(50)]


# In[ ]:


for g in top_cast_members:
    train['cast_member_'+g] = train['cast'].apply(lambda x: 1 if g in str(x) else 0)
    
for g in top_cast_members:
    test['cast_member_'+g] = test['cast'].apply(lambda x: 1 if g in str(x) else 0)


# In[ ]:


train = train.drop(['cast'], axis=1)
test = test.drop(['cast'], axis=1)


# #### Crew

# In[ ]:


train['crew'] = train['crew'].apply(lambda x: eval(x) if str(x) != 'nan' else [])
test['crew'] = test['crew'].apply(lambda x: eval(x) if str(x) != 'nan' else [])


# In[ ]:


list_of_crew_members = list(train['crew'].apply(lambda x: [i['name'] for i in x] if str(x) != 'nan' else []).values)


# In[ ]:


top_crew_members = [m[0] for m in Counter([i for j in list_of_crew_members for i in j]).most_common(50)]


# In[ ]:


for g in top_crew_members:
    train['crew_member_'+g] = train['crew'].apply(lambda x: 1 if g in str(x) else 0)
    
for g in top_crew_members:
    test['crew_member_'+g] = test['crew'].apply(lambda x: 1 if g in str(x) else 0)


# In[ ]:


train = train.drop(['crew'], axis=1)
test = test.drop(['crew'], axis=1)


# #### Homepage

# In[ ]:


train['has_homepage'] = train['homepage'].apply(lambda x: 1 if str(x) != 'nan' else 0)
test['has_homepage'] = test['homepage'].apply(lambda x: 1 if str(x) != 'nan' else 0)


# In[ ]:


train = train.drop(['homepage'], axis=1)
test = test.drop(['homepage'], axis=1)


# #### Poster Path

# In[ ]:


train = train.drop(['poster_path'], axis=1)
test = test.drop(['poster_path'], axis=1)


# #### Status

# In[ ]:


train = train.drop(['status'], axis=1)
test = test.drop(['status'], axis=1)


# #### Title, Tagline, Overview, Original Title

# In[ ]:


for col in ['title', 'tagline', 'overview', 'original_title']:
    train['len_' + col] = train[col].fillna('').apply(lambda x: len(str(x)))
    train['words_' + col] = train[col].fillna('').apply(lambda x: len(str(x.split(' '))))
    
    test['len_' + col] = test[col].fillna('').apply(lambda x: len(str(x)))
    test['words_' + col] = test[col].fillna('').apply(lambda x: len(str(x.split(' '))))


# #### Creating a model

# In[ ]:


train = train.drop(["imdb_id", "original_title", "overview", "tagline", "title"], axis=1)
test = test.drop(["imdb_id", "original_title", "overview", "tagline", "title"], axis=1)


# In[ ]:


X = train.drop(['id', 'revenue'], axis=1)
Y = np.log1p(train['revenue'])
X_test = test.drop(['id'], axis=1)


# In[ ]:


X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1)


# In[ ]:


params = {'num_leaves': 30,
         'min_data_in_leaf': 20,
         'objective': 'regression',
         'max_depth': 5,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2,
         "verbosity": -1}
model = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)
model.fit(X_train, Y_train, 
        eval_set=[(X_train, Y_train), (X_valid, Y_valid)], eval_metric='rmse',
        verbose=1000, early_stopping_rounds=200)


# In[ ]:


y_pred_valid = model.predict(X_valid)
y_pred = model.predict(X_test, num_iteration=model.best_iteration_)


# In[ ]:


sample_submission['revenue'] = np.expm1(y_pred)
sample_submission.to_csv("submission.csv", index=False)


# In[ ]:




