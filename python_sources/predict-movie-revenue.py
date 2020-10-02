#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ast
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import operator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import eli5
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import make_scorer
import lightgbm as lgb
from PIL import Image
from urllib.request import urlopen
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
plt.style.use('ggplot')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Overview**
# 
# This kernel will use data from [The Movie Database](https://www.themoviedb.org/) (TMDB) to predict worldwide movie revenue. The movie industry is on the rise and box office sales have skyrocketed, can we use data about past movie revenue to predict future revenue? Could this information be used to create more profitable movies

# **Load and review the training data**

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

train_df.head()


# In[ ]:


train_df.shape


# The release data is stored as an object, convert the object to a datetime and create additional variables for year, month, day of week, week of year, quarter and day.

# In[ ]:



def fix_date(x):
    """
    Fixes dates which are in 20xx
    """
    year = x.split('/')[2]
    if int(year) <= 19:
        return x[:-2] + '20' + year
    else:
        return x[:-2] + '19' + year
train_df['release_date'] = pd.to_datetime(train_df['release_date'].apply(lambda x: fix_date(x)))
test_df.loc[test_df['release_date'].isnull() == True, 'release_date'] = '01/01/98'
test_df['release_date'] = pd.to_datetime(test_df['release_date'].apply(lambda x: fix_date(x)))
train_df['release_date'].head()


# In[ ]:


# creating features based on dates
def process_date(df):
    date_parts = ["year", "weekday", "month", 'weekofyear', 'day', 'quarter']
    for part in date_parts:
        part_col = 'release_date' + "_" + part
        df[part_col] = getattr(df['release_date'].dt, part).astype(int)
    
    return df

train_df = process_date(train_df)
test_df = process_date(test_df)


# **DATA CLEANING AND VARIABLE EXTRACTION**
# 
# * Some of the fields contain lists with dictionaries, but these columns are stored as text, we need to convert them back to dictionaries to view all the data
# * Extract relevant fields from the dictionaries, for genre, language, production countires and production companies
# * Most do not belong to a collection so I will just create a flag for those that do belong to a collection and keep the collection name.

# In[ ]:


dict_columns = ['belongs_to_collection', 'genres', 'production_companies', 'production_countries', 
                'spoken_languages', 'Keywords', 'cast', 'crew']
def convert_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: ast.literal_eval(x) if not pd.isnull(x) else np.NaN)
    return df


# In[ ]:


train_df = convert_dict(train_df)
test_df = convert_dict(test_df)


# In[ ]:


collections = train_df['belongs_to_collection']
collections[:5]


# In[ ]:


train_df['belongs_to_collection'].isnull().sum()


# Since almost 80% of the records do not belong to a collection, most of this information will not be useful for modeling. For now, we will create two additional variables:
# 
# * has_collection: This will indicate wheather the movie has a collection or not
# * collection_name: We will just keep the name of the collection for the movies with a collection

# In[ ]:


train_df['has_collection'] = train_df['belongs_to_collection'].apply(lambda x: 1 if not pd.isnull(x) else 0)
test_df['has_collection'] = test_df['belongs_to_collection'].apply(lambda x: 1 if not pd.isnull(x) else 0)
train_df['has_collection'][:5]


# In[ ]:


train_df['collection_name'] = train_df['belongs_to_collection'].apply(lambda x: x[0]['name'] if not pd.isnull(x) else 0)
test_df['collection_name'] = test_df['belongs_to_collection'].apply(lambda x: x[0]['name'] if not pd.isnull(x) else 0)
train_df['collection_name'][:5]


# In[ ]:


train_df = train_df.drop("belongs_to_collection", axis =1)
test_df = test_df.drop("belongs_to_collection", axis =1)


# In[ ]:


def dictToInd(var, num):
    train_df[var] = train_df[var].fillna(0)
    test_df[var] = test_df[var].fillna(0)
    
    train_df['num_' + var] = train_df[var].apply(lambda x: len(x) if x!= 0 else 0) 
    test_df['num_' + var] = test_df[var].apply(lambda x: len(x) if x!= 0 else 0) 
    
    train_df['all_'+var] = train_df[var].apply(lambda x: [i['name'] for i in x] if x !=0 else 0)
    test_df['all_'+var] = test_df[var].apply(lambda x: [i['name'] for i in x] if x !=0 else 0)
    
    d = dict()
    
    for i, e in enumerate(train_df['all_' + var]):
        if e != 0:
            for k in e:
                if k in d:
                    d[k] += 1
                else:
                    d[k] = 1
    top_d = dict(sorted(d.items(), key = operator.itemgetter(1), reverse = True)[:num])
    
    for i in top_d:
        train_df[var +'_' + i] = train_df['all_' + var].apply(lambda x: 1 if x!=0 and i in x else 0)
        test_df[var +'_' + i] = test_df['all_' + var].apply(lambda x: 1 if x!=0 and i in x else 0)


# Create variables to count the number of genres, production companies, production countries, and spoken languages. In addition I will create dummy variables for the different genres, companies, countries and languages. 

# In[ ]:


dictToInd('genres', 30)
train_df.head()


# In[ ]:


train_df['num_genres'].value_counts()


# The movies have between 0 and 7 genres, with most having between 2 and 5 genres. below we can see there are 20 possible genres

# In[ ]:


genres = []

for i, e in enumerate(train_df['all_genres']):
    if e != 0:
        for genre in e:
            if genre not in genres:
                genres.append(genre)
            
print(genres)


# In[ ]:


train_df['genres'].head()
list_of_genres = train_df['genres'].apply(lambda x: x if x != {} else [])
list_of_genres


# In[ ]:


train_df['production_companies'].fillna(0).apply(lambda x: len(x) if x != 0 else 0).value_counts()


# The movies have between 1 and 17 production companies with most having 6 or less. We'll add the top 6 most popular production companies

# In[ ]:


dictToInd('production_companies', 6)
train_df.head()


# In[ ]:


train_df['production_countries'].fillna(0).apply(lambda x: len(x) if x != 0 else 0).value_counts()


# Most the movies only have one production country, but we'll make indicators for the top 5 most popular ones

# In[ ]:


dictToInd('production_countries',5)
train_df.head()


# In[ ]:


train_df['spoken_languages'].fillna(0).apply(lambda x: len(x) if x != 0 else 0).value_counts()


# In[ ]:


dictToInd('spoken_languages', 5)
train_df.head()


# In[ ]:


train_df['Keywords'].fillna(0).apply(lambda x: len(x) if x !=0 else 0).value_counts()


# In[ ]:


train_df['Keywords']


# In[ ]:


list_of_keywords = list(train_df['Keywords'].fillna(0).apply(lambda x: [i['name'] for i in x] if x != 0 else []).values)
list_of_keywords[:19]


# In[ ]:


plt.figure(figsize=(12,18))
text = ' '.join([i for j in list_of_keywords for i in j])
wordcloud = WordCloud(max_font_size = None, background_color='white', collocations=False,
                      width = 1200, height=1000).generate(text)
plt.imshow(wordcloud)


# There's lots of possible keywords, but some that appear most often are relationship, woman, love, director, based, murder and film. 
# I'll start by pulling in the top 10 most popular key words to see if the have any predictive power.

# In[ ]:


dictToInd('Keywords', 10)
train_df.head()


# In[ ]:


train_df['cast'].fillna(0).apply(lambda x: len(x) if x != 0 else 0).value_counts()


# There's lots of possibilities for cast and number of cast, we'll create indicators for the 10 most popular actors, we'll do the same for crew. Perhaps popular actors or crew memebers impact the amount of revenue the movie will make.

# In[ ]:


dictToInd('cast', 10)
train_df.head()


# In[ ]:


dictToInd('crew', 10)
train_df.head()


# Now we need to eliminate the nulls by filling them in or removing columns with too many NAs. 

# In[ ]:


for i in train_df.columns:
    num_nas = train_df[i].isnull().sum()
    if num_nas > 0:
        print(i, " :", num_nas)


# In[ ]:


for i in test_df.columns:
    num_nas = test_df[i].isnull().sum()
    if num_nas > 0:
        print(i, " :", num_nas)


# In[ ]:


test_df['status'].value_counts()


# **Columns with NAN values**
# * homepage: This column is mostly null so we will drop the column
# * overview: We can serch the overview for popular key words to see if we can utilize this column
# * poster_path: This is a link to the moview poster, we will remove this column from this analysis
# * runtime: We can fill in the 2 NAN values with the median
# * tagline: This is the tagline for the movie, we will create an indicator for if the movie has a tagline and remove this column

# In[ ]:


train_df['runtime'] = train_df['runtime'].fillna(train_df['runtime'].median())
test_df['runtime'] = test_df['runtime'].fillna(test_df['runtime'].median())
train_df['tagline'] = train_df['tagline'].fillna('')
test_df['tagline'] = test_df['tagline'].fillna('')

train_df['homepage'] = train_df['homepage'].fillna(0)
test_df['homepage'] = test_df['homepage'].fillna(0)

train_df['has_tagline'] = train_df['tagline'].apply(lambda x: 1 if x != '' else 0)
test_df['has_tagline'] = test_df['tagline'].apply(lambda x: 1 if x != '' else 0)

train_df['has_homepage'] = train_df['homepage'].apply(lambda x: 1 if x != 0 else 0)
test_df['has_homepage'] = test_df['homepage'].apply(lambda x: 1 if x != 0 else 0)

train_df['budget'] = train_df['budget'].apply(lambda x: x if x > 0 else np.NaN)
test_df['budget'] = test_df['budget'].apply(lambda x: x if x > 0 else np.NaN)

train_df['budget'] = train_df['budget'].fillna(train_df['budget'].median())
test_df['budget'] = test_df['budget'].fillna(test_df['budget'].median())

test_df['status'] = test_df['status'].fillna(test_df['status'].mode()[0])


# We need to address the remaining characters values we want to utilize as predictors, we'll use sklearn LabelEncoder to convert them to numeric. Most of the movies are released so status will probbably not be a useful predictor.

# In[ ]:


le = LabelEncoder()
train_df['status'] = le.fit_transform(train_df['status'])
test_df['status'] = le.fit_transform(test_df['status'])
train_df['status'].value_counts()


# **TARGET**
# * The distribution of Revenue is highly skewed, It will be better to use Log Revenue for modeling purposes.

# In[ ]:


fig, axes = plt.subplots(1,2, figsize=(16,6))
sns.set(color_codes=True, style='darkgrid')
sns.distplot(train_df['revenue'], color ='y', ax=axes[0]).set_title('Revenue Distribution')
sns.distplot(np.log1p(train_df['revenue']), color='b', ax=axes[1]).set_title('Log Revenue Distribution')


# **Budget**

# In[ ]:


fig, axes = plt.subplots(2,2, figsize=(16,8))
sns.set(color_codes=True, style='darkgrid')
sns.distplot(train_df['budget'], color ='y', ax=axes[0,0]).set_title('Budget Distribution')
sns.distplot(np.log1p(train_df['budget']), ax=axes[0,1]).set_title('Log Budget Distribution')

sns.scatterplot(train_df['budget'], train_df['revenue'],color ='y', ax=axes[1,0]).set_title('Revenue vs. Budget')
sns.scatterplot(np.log1p(train_df['budget']), np.log1p(train_df['revenue']), ax=axes[1,1]).set_title('Log Revenue vs. Log Budget')


# In[ ]:


fig, axes = plt.subplots(1,2, figsize=(16,6))
plt.subplots_adjust(top=0.9)
fig.suptitle('Revenue for film with and without tagline/homepage')
sns.catplot(x='has_homepage', y='revenue', ax=axes[0], data=train_df);

sns.catplot(x='has_tagline', y='revenue', ax=axes[1], data=train_df)
plt.close(2)
plt.close(3)


# It does appear that movies with a homepage and/or tagline produce greater revenue, maybe this is because people are able to learn more about the movie.

# In[ ]:


train_df['log_revenue'] = np.log1p(train_df['revenue'])
train_df['log_budget'] = np.log1p(train_df['budget'])
test_df['log_budget'] = np.log1p(test_df['budget'])


# In[ ]:


plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
sns.boxplot(x='original_language', y='revenue', data=train_df.loc[train_df['original_language'].isin(train_df['original_language'].value_counts().head(10).index)]);
plt.title('Mean revenue per language');
plt.subplot(1, 2, 2)
sns.boxplot(x='original_language', y='log_revenue', data=train_df.loc[train_df['original_language'].isin(train_df['original_language'].value_counts().head(10).index)]);
plt.title('Mean log revenue per language');


# It does seem as though some languages produce greater revenue than others, it does make sense that langagues spoken in more places of the world (ie. English) would have the oppurtunity for greater revenue.

# In[ ]:


vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 2),
            min_df=5)

overview_text = vectorizer.fit_transform(train_df['overview'].fillna(''))
linreg = LinearRegression()
linreg.fit(overview_text, train_df['log_revenue'])
eli5.show_weights(linreg, vec=vectorizer, top=20, feature_filter=lambda x: x != '<BIAS>')


# We can see above that some words within the movies overview have a correlation with the amount of revenue the movie made and some words reduce the amount of revenue the movie made.

# In[ ]:


print('Target value:', train_df['log_revenue'][10])
eli5.show_prediction(linreg, doc=train_df['overview'].values[10], vec=vectorizer)


# In[ ]:


train_texts = train_df[['title', 'tagline', 'overview', 'original_title']]
test_texts = test_df[['title', 'tagline', 'overview', 'original_title']]

train_df[['title', 'tagline', 'overview', 'original_title']].head()


# In[ ]:


for col in ['title','overview', 'tagline','original_title']:
    train_df['len_' + col] = train_df[col].fillna('').apply(lambda x: len(str(x)))
    train_df['words_' + col] = train_df[col].fillna('').apply(lambda x: len(str(x.split(' '))))
    train_df = train_df.drop(col, axis=1)
    test_df['len_' + col] = test_df[col].fillna('').apply(lambda x: len(str(x)))
    test_df['words_' + col] = test_df[col].fillna('').apply(lambda x: len(str(x.split(' '))))
    test_df = test_df.drop(col, axis=1)


# In[ ]:


f, axes = plt.subplots(4, 5, figsize=(24, 16))
plt.suptitle('Violinplot of revenue vs genres')
for i, e in enumerate([col for col in train_df.columns if 'genres_' in col]):
    sns.violinplot(x=e, y='revenue', data=train_df, ax=axes[i // 5][i % 5]);


# The distribution of revenue varies by the genre of the movie. 

# In[ ]:


plt.style.use('ggplot')
d1 = train_df['release_date_year'].value_counts().sort_index()
d2 = test_df['release_date_year'].value_counts().sort_index()
data = [go.Scatter(x=d1.index, y=d1.values, name='train'), go.Scatter(x=d2.index, y=d2.values, name='test')]
layout = go.Layout(dict(title = "Number of films per year",
                  xaxis = dict(title = 'Year'),
                  yaxis = dict(title = 'Count'),
                  ),legend=dict(
                orientation="v"))
py.iplot(dict(data=data, layout=layout))


# More movies are being produced each year so the total movie revenue by year is going up.

# In[ ]:


f = ['budget', 'popularity', 'runtime', 'revenue', 'log_revenue', 'log_budget']
sns.pairplot(train_df[f].dropna())


# The above plots show that budget is a good predictor of revenue and popularity has a weak correlation with revenue. Movies that are very short or very long tend to make less revenue than movies of an averga (1.5-2hr) length.

# In[ ]:


corr = train_df.drop('release_date', axis=1).corr()
corr.style.background_gradient(cmap='Blues')


# The correlation matrix above shows the variables with the strongest correlation to revenue are:
# * Budget: 0.752965
# * Popularity: 0.46146
# * Num Crew: 0.37211
# * Has Collection: 0.339425
# * Num Cast: 0.335737
# * Genre Adevenuture: 0.328439
# * Has Homepage: 0.263179
# * Runtime: 0.216417

# Turn month and day of week into dummy variables for modeling and create some additional variables.

# In[ ]:


train_df = pd.concat([train_df, pd.get_dummies(train_df['release_date_weekday'], prefix='release_weekday')], axis=1)
train_df = pd.concat([train_df, pd.get_dummies(train_df['release_date_month'], prefix='release_month')], axis=1)
train_df = pd.concat([train_df, pd.get_dummies(train_df['release_date_quarter'], prefix='release_quarter')], axis=1)

test_df = pd.concat([test_df, pd.get_dummies(test_df['release_date_weekday'], prefix='release_weekday')], axis=1)
test_df = pd.concat([test_df, pd.get_dummies(test_df['release_date_month'], prefix='release_month')], axis=1)
test_df = pd.concat([test_df, pd.get_dummies(test_df['release_date_quarter'], prefix='release_quarter')], axis=1)


# In[ ]:


def new_features(df):
    df['budget_to_popularity'] = df['budget'] / df['popularity']
    df['budget_to_runtime'] = df['budget'] / df['runtime']
    
    # some features from https://www.kaggle.com/somang1418/happy-valentines-day-and-keep-kaggling-3
    df['_budget_year_ratio'] = df['budget'] / (df['release_date_year'] * df['release_date_year'])
    df['_releaseYear_popularity_ratio'] = df['release_date_year'] / df['popularity']
    df['_releaseYear_popularity_ratio2'] = df['popularity'] / df['release_date_year']
    
    df['runtime_to_mean_year'] = df['runtime'] / df.groupby("release_date_year")["runtime"].transform('mean')
    df['popularity_to_mean_year'] = df['popularity'] / df.groupby("release_date_year")["popularity"].transform('mean')
    df['budget_to_mean_year'] = df['budget'] / df.groupby("release_date_year")["budget"].transform('mean')
        
    return df


# In[ ]:


train_df =new_features(train_df)
test_df =new_features(test_df)
train_df.head()


# **MODELING**
# 
# Lets create our X and y, the y variable is revenue and we'll drop the predictors that we wont be using for our X variable.
# 
# Model Evaluation:
# * Simple Random Forest Model: mse=3.79
# * Gradient Boosting Model: mse: mse=3.32
# * Random Forest Model Optimized with RandomizedSearchCV: mse=3.55

# In[ ]:


y = train_df['log_revenue']
X = train_df.drop(['id', 'genres', 'homepage', 'imdb_id','original_language', 'poster_path', 'production_companies', 'production_countries', 'spoken_languages',
                  'Keywords', 'cast', 'crew', 'revenue', 'collection_name','all_genres', 'all_production_companies', 'all_production_countries', 'all_spoken_languages', 
                        'all_Keywords', 'all_cast', 'all_crew', 'release_date', 'revenue', 'log_revenue', 'budget', 'release_date_quarter',
                  'release_date_day', 'release_date_weekofyear', 'release_date_month', 'release_date_weekday', 
                   'budget_to_runtime' 
                  ], axis = 1)
X_test= test_df.drop(['id', 'genres', 'homepage', 'imdb_id','original_language',  'poster_path', 'production_companies', 'production_countries', 'spoken_languages',
                   'Keywords', 'cast', 'crew', 'collection_name','all_genres', 'all_production_companies', 'all_production_countries', 'all_spoken_languages', 
                        'all_Keywords', 'all_cast', 'all_crew', 'release_date', 'budget', 'release_date_quarter', 'release_date_day',
                     'release_date_weekofyear', 'release_date_month', 'release_date_weekday', 
                       'budget_to_runtime'
                     ], axis = 1)
X.head()


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)


# In[ ]:


simple_forest = RandomForestRegressor(criterion = 'mse',
                              n_estimators =100,
                              random_state = 1)


# In[ ]:


simple_forest.fit(X_train, y_train)
simple_forest_pred = simple_forest.predict(X_valid)


# In[ ]:


mse = mean_squared_error(simple_forest_pred, y_valid)
mse


# In[ ]:


simple_forest_pred = simple_forest.predict(X_test)
simple_forest_rev = np.expm1(simple_forest_pred)
os.chdir('../working')
submit = pd.DataFrame({'Id': test_df['id'], 'revenue': simple_forest_rev})
submit.to_csv('submission_simple_forest.csv', index=False)
submit.head()


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
light_gbm = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1, random_state=1)
light_gbm.fit(X_train, y_train, 
        eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',
        verbose=1000, early_stopping_rounds=200)


# In[ ]:


np.square(1.82195)


# In[ ]:


eli5.show_weights(light_gbm, feature_filter=lambda x: x != '<BIAS>')


# In[ ]:


light_gbm_pred = light_gbm.predict(X_test)
light_gbm_rev = np.expm1(light_gbm_pred)
os.chdir('../working')
submit = pd.DataFrame({'Id': test_df['id'], 'revenue': light_gbm_rev})
submit.to_csv('submission_light_gbm.csv', index=False)
submit.head()


# In[ ]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 8, 10, 15]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 3, 5, 7, 10]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# #Use the random grid to search for best hyperparameters**
# #First create the base model to tune
# rf = RandomForestRegressor()
# #Random search of parameters, using 3 fold cross validation, 
# #search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# #Fit the random search model
# rf_random.fit(X_train, y_train)
# 
# rf_random.best_params_

# In[ ]:


rf_best = RandomForestRegressor(n_estimators=1400,
                               max_features= 'auto',
                               max_depth= 80,
                               min_samples_split= 2,
                               min_samples_leaf= 3,
                               bootstrap= True,
                               random_state=1)


# In[ ]:


rf_best.fit(X_train, y_train)
rf_best_pred = rf_best.predict(X_valid)
mse_best= mean_squared_error(rf_best_pred, y_valid)
mse_best


# In[ ]:


rf_best_pred = rf_best.predict(X_test)
rf_best_pred[:5]


# In[ ]:


feature_importances = pd.DataFrame(rf_best.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
plt.barh(feature_importances.index[:15],feature_importances['importance'][:15])


# In[ ]:


rf_best_rev = np.expm1(rf_best_pred)
os.chdir('../working')
submit = pd.DataFrame({'Id': test_df['id'], 'revenue': rf_best_rev})
submit.to_csv('submission_rf_best.csv', index=False)
submit.head()


# The Gradient Boosting Model had the best results and I would recommend performing a full GridSearchCV on that model to optimize the model and obtain the best possible parameters.

# In[ ]:




