#!/usr/bin/env python
# coding: utf-8

# # TMDB movie binary classification

# ![imglink](https://fortunedotcom.files.wordpress.com/2017/08/movies.gif)
# 
# (Image taken from [imglink](https://fortunedotcom.files.wordpress.com/2017/08/movies.gif))

# # Introduction
# 
# From TMDB movie dataset, I decide to make some simple binary classification for predicting a **nice movie** <br>or movie that will get a** good rating** before movie release <br>
# and try to explore, visualize and wrangling the dataset.<br> 
# 
# 
# In this kernel,
# 
#  - Simple Exploratory Data Analysis
#  - Data wrangling
#  - Create model (without tuning parameter)
#  - Comparing model

# In[ ]:


import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph.


from numpy import median
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pylab import rcParams

get_ipython().run_line_magic('matplotlib', 'inline')


# **** Change Format from json file  ****
#     
#   Credit >>  [getting imdb kernels working with tmdb data](https://www.kaggle.com/sohier/getting-imdb-kernels-working-with-tmdb-data/)

# In[ ]:


def load_tmdb_movies(path):
    df = pd.read_csv(path)
    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())
    json_columns = ['genres', 'keywords', 'production_countries', 'production_companies', 'spoken_languages']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df


def load_tmdb_credits(path):
    df = pd.read_csv(path)
    json_columns = ['cast', 'crew']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df


# In[ ]:


# Columns that existed in the IMDB version of the dataset and are gone.
LOST_COLUMNS = [
    'actor_1_facebook_likes',
    'actor_2_facebook_likes',
    'actor_3_facebook_likes',
    'aspect_ratio',
    'cast_total_facebook_likes',
    'color',
    'content_rating',
    'director_facebook_likes',
    'facenumber_in_poster',
    'movie_facebook_likes',
    'movie_imdb_link',
    'num_critic_for_reviews',
    'num_user_for_reviews'
                ]

# Columns in TMDb that had direct equivalents in the IMDB version. 
# These columns can be used with old kernels just by changing the names
TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES = {
    'budget': 'budget',
    'genres': 'genres',
    'revenue': 'gross',
    'title': 'movie_title',
    'runtime': 'duration',
    'original_language': 'language',  # it's possible that spoken_languages would be a better match
    'keywords': 'plot_keywords',
    'vote_count': 'num_voted_users',
                                         }

IMDB_COLUMNS_TO_REMAP = {'imdb_score': 'vote_average'}


def safe_access(container, index_values):
    # return a missing value rather than an error upon indexing/key failure
    result = container
    try:
        for idx in index_values:
            result = result[idx]
        return result
    except IndexError or KeyError:
        return pd.np.nan


def get_director(crew_data):
    directors = [x['name'] for x in crew_data if x['job'] == 'Director']
    return safe_access(directors, [0])


def pipe_flatten_names(keywords):
    return '|'.join([x['name'] for x in keywords])


def convert_to_original_format(movies, credits):
    # Converts TMDb data to make it as compatible as possible with kernels built on the original version of the data.
    tmdb_movies = movies.copy()
    tmdb_movies.rename(columns=TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES, inplace=True)
    tmdb_movies['title_year'] = pd.to_datetime(tmdb_movies['release_date']).apply(lambda x: x.year)
    # I'm assuming that the first production country is equivalent, but have not been able to validate this
    tmdb_movies['country'] = tmdb_movies['production_countries'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['language'] = tmdb_movies['spoken_languages'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['director_name'] = credits['crew'].apply(get_director)
    tmdb_movies['actor_1_name'] = credits['cast'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['actor_2_name'] = credits['cast'].apply(lambda x: safe_access(x, [1, 'name']))
    tmdb_movies['actor_3_name'] = credits['cast'].apply(lambda x: safe_access(x, [2, 'name']))
    tmdb_movies['companies_1'] = tmdb_movies['production_companies'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['companies_2'] = tmdb_movies['production_companies'].apply(lambda x: safe_access(x, [1, 'name']))
    tmdb_movies['companies_3'] = tmdb_movies['production_companies'].apply(lambda x: safe_access(x, [2, 'name']))
    tmdb_movies['genres'] = tmdb_movies['genres'].apply(pipe_flatten_names)
    tmdb_movies['plot_keywords'] = tmdb_movies['plot_keywords'].apply(pipe_flatten_names)
    return tmdb_movies


# **** Read files ****

# In[ ]:


movies = load_tmdb_movies("../input/tmdb_5000_movies.csv")
credits = load_tmdb_credits("../input/tmdb_5000_credits.csv")
data =convert_to_original_format(movies, credits)


# In[ ]:


data.head()


# In[ ]:


#missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(5)


# **Missing value around 64 % for homepage features**

# In[ ]:


data.drop(['homepage'], axis=1, inplace=True)


# ## Exploratory Data Analysis

# In[ ]:


data.columns


# ## Drop features that only know after movie release
# 
# **Objective of this kernel is predict movie before it release**

# In[ ]:


data.drop(['num_voted_users','gross','popularity'], axis=1, inplace=True)


# In[ ]:


# Correlation matrix between numerical values
plt.figure(figsize = (10,8))
g = sns.heatmap(data[list(data)].corr(),annot=True, fmt = ".2f", cmap = "coolwarm",linewidths= 0.01)


# **From heatmap**, 
#              <br>we can notice that title_year and duration has an impact to vote_average score.
#             

# ### Title_year and vote_average

# In[ ]:


plt.figure(figsize = (10,10))
sns.jointplot(x="title_year", y="vote_average", data=data);


# ** A lot of movie in our dataset was made between 2000 and 2017. **
#  
# 
#     many rows in our dataset contain 0 vote average and I decide to delete them

# In[ ]:


data = data[data['vote_average'] != 0]


# In[ ]:


plt.figure(figsize = (10,10))
sns.regplot(x="title_year", y="vote_average", data=data);


# **and seem like their avg vote_average are lower than old movie **

# ### duration and vote_average

# In[ ]:


plt.figure(figsize = (10,10))
sns.jointplot(x="duration", y="vote_average", data=data, color= 'Red');


# ** Most of directors decide to make 90-120 mins movie **
# 
#     some movie in our data set has 0 duration and I think it not possible

# In[ ]:


data = data[data['duration'] != 0]


# In[ ]:


plt.figure(figsize = (10,10))
sns.regplot(x="duration", y="vote_average", data=data, color= 'Red');


# **Long movie look like to get higher score than short movie **

# ## Data wrangling
#      Because we have many useful features that can't use to create a model
#     Then we need to edit the data format
# 

# In[ ]:


data.columns


# # As I said in this kernel name, this is binary classification kernel.

# ## We need some target binary variable which is a criteria for seperate nice movie or not

# *** For This kernel, I decide to declare nice movies as a movie which got vote_score more than vote_score mean ***

# In[ ]:


data['vote_average'].mean()


# In[ ]:


data['Nice'] = data['vote_average'].map(lambda s :1  if s >= data['vote_average'].mean() else 0)


# In[ ]:


data.loc[:, ['vote_average', 'Nice']].head()


# In[ ]:


rcParams['figure.figsize'] = 13,10


# In[ ]:


data['Nice'].value_counts(sort = False)


# In[ ]:


# Data to plot
labels =["not","nice movie"]
sizes = data['Nice'].value_counts(sort = False)
colors = ["pink","whitesmoke"]
explode = (0.1,0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140,)
plt.axis('equal')
plt.show()


# In[ ]:


#### 54.1% of movies in our dataset are nice movie


# ## Budget

# In[ ]:


# budget distibution 
g = sns.kdeplot(data.budget[(data["Nice"] == 0) ], color="Red", shade = True)
g = sns.kdeplot(data.budget[(data["Nice"] == 1) ], ax =g, color="Blue", shade= True)
g.set_xlabel("budget")
g.set_ylabel("Frequency")
g = g.legend(["Not","Nice"])


# In[ ]:


import statistics
sd = statistics.stdev(data.budget)
mean = data.budget.mean()
max = data.budget.max()
min = data.budget.min()


# In[ ]:


# Create new feature of budget 

data['VeryLowBud'] = data['budget'].map(lambda s: 1 if s < 10000000 else 0)
data['LowBud'] = data['budget'].map(lambda s: 1 if 10000000 <= s < mean else 0)
data['MedBud'] = data['budget'].map(lambda s: 1 if  mean <= s < mean+sd  else 0)
data['HighBud'] = data['budget'].map(lambda s: 1 if mean+sd <= s < 100000000 else 0)
data['VeryHighBud'] = data['budget'].map(lambda s: 1 if s >= 100000000 else 0)


# In[ ]:


g = sns.factorplot(x="VeryLowBud",y="Nice",data=data,kind="bar",palette = "husl")
g = g.set_ylabels("Nice Probability")
g = sns.factorplot(x="LowBud",y="Nice",data=data,kind="bar",palette = "husl")
g = g.set_ylabels("Nice Probability")
g = sns.factorplot(x="MedBud",y="Nice",data=data,kind="bar",palette = "husl")
g = g.set_ylabels("Nice Probability")
g = sns.factorplot(x="HighBud",y="Nice",data=data,kind="bar",palette = "husl")
g = g.set_ylabels("Nice Probability")
g = sns.factorplot(x="VeryHighBud",y="Nice",data=data,kind="bar",palette = "husl")
g = g.set_ylabels("Nice Probability")


# ## Title year 

# In[ ]:


g = sns.factorplot(y="title_year",x="Nice",data=data,kind="violin", palette = "Set2")


# In[ ]:


data.columns


# ## Duration

# In[ ]:


data = data[np.isfinite(data['duration'])]


# In[ ]:


# duration distibution 
g = sns.kdeplot(data.duration[(data["Nice"] == 0) ], color="blueviolet", shade = True)
g = sns.kdeplot(data.duration[(data["Nice"] == 1) ], ax =g, color="gold", shade= True)
g.set_xlabel("Duration")
g.set_ylabel("Frequency")
g = g.legend(["Not","Nice"])


# In[ ]:


g = sns.factorplot(x="Nice", y = "duration",data = data, kind="box", palette = "Set3")


# In[ ]:


# Create new feature of duration

data['ShortMovie'] = data['duration'].map(lambda s: 1 if s < 90 else 0)
data['NotTooLongMovie'] = data['duration'].map(lambda s: 1 if 90 <= s < 120 else 0)
data['LongMovie'] = data['duration'].map(lambda s: 1 if   s >= 120  else 0)


# ## Genres 

# In[ ]:


data['genres'].head()


# **Many genres in one movie is the first problem that we met !!**
# 
#        It can't use even make some simple tree !!.
#          Then, I decide to make this column to binary format.

# In[ ]:



def Obtain_list_Occurences(columnName):
    # Obtaining a list of columnName
    list_details = list(map(str,(data[columnName])))
    listOcc = []
    for i in data[columnName]:
        split_genre = list(map(str, i.split('|')))
        for j in split_genre:
            if j not in listOcc:
                listOcc.append(j)
    return listOcc


# In[ ]:


genre = []
genre = Obtain_list_Occurences("genres")


# In[ ]:


for word in genre:
    data[word] = data['genres'].map(lambda s: 1 if word in str(s) else 0)


# In[ ]:


data.loc[:,'Action': 'Foreign'].head(5)


# **Oh!!, It's better ??**
# <br> but, we have anoter column that very similar to genre...
# <br> That is 'keyword'

# In[ ]:


data['plot_keywords'].head(20)


# **But..** <br>It's has many keyword, How many keywords should I use.

# Credit : https://www.kaggle.com/fabiendaniel/film-recommendation-engine

# In[ ]:


def count_word(df, ref_col, liste):
    keyword_count = dict()
    for s in liste: keyword_count[s] = 0
    for liste_keywords in df[ref_col].str.split('|'):        
        if type(liste_keywords) == float and pd.isnull(liste_keywords): continue        
        for s in [s for s in liste_keywords if s in liste]: 
            if pd.notnull(s): keyword_count[s] += 1
    #______________________________________________________________________
    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count


# In[ ]:


set_keywords = set()
for liste_keywords in data['plot_keywords'].str.split('|').values:
    if isinstance(liste_keywords, float): continue  # only happen if liste_keywords = NaN
    set_keywords = set_keywords.union(liste_keywords)
#_________________________
# remove null chain entry
set_keywords.remove('')


# In[ ]:


keyword_occurences, dum = count_word(data, 'plot_keywords', set_keywords)


# **Okays, I got the most 10 keywords that contains in our dataset **
# <br> Then, I use this Top 10 repeated keywords.

# In[ ]:


## Funtion to find top 10 in list

def TopTen(theList):
    TopTen = list()

    for i in range(0, 10):
        TopTen.append(theList[i][0])
    
    return TopTen


# In[ ]:


from wordcloud import WordCloud

def makeCloud(Dict,name,color):
    words = dict()

    for s in Dict:
        words[s[0]] = s[1]

        wordcloud = WordCloud(
                      width=1500,
                      height=750, 
                      background_color=color, 
                      max_words=50,
                      max_font_size=500, 
                      normalize_plurals=False)
        wordcloud.generate_from_frequencies(words)


    fig = plt.figure(figsize=(12, 8))
    plt.title(name)
    plt.imshow(wordcloud)
    plt.axis('off')

    plt.show()


# In[ ]:


makeCloud(keyword_occurences[0:15],"Keywords","White")


# In[ ]:


for word in TopTen(keyword_occurences):
    data[word] = data['plot_keywords'].map(lambda s: 1 if word in str(s) else 0)

    
data.drop('plot_keywords',axis=1,inplace=True)
data.loc[:,'woman director':].head()


# ### Do the same way with Actor, Director and Company 

# ## Director

# In[ ]:


data['director_name'].fillna('unknown',inplace=True)


# In[ ]:


def to_frequency_table(data):
    frequencytable = {}
    for key in data:
        if key in frequencytable:
            frequencytable[key] += 1
        else:
            frequencytable[key] = 1
    return frequencytable


# In[ ]:


director_dic = to_frequency_table(data['director_name'])
director_list = list(director_dic.items())
director_list.sort(key=lambda tup: tup[1],reverse=True)


# In[ ]:


makeCloud(director_list[0:10],"director","whitesmoke")


# In[ ]:


for word in TopTen(director_list):
    data[word] = data['director_name'].map(lambda s: 1 if word in str(s) else 0)

data.loc[:,'Steven Spielberg': ].head()


# ## Actor

#     In this dataset, it contain 3 actor_name columns and a lot of missing value for second and third actor
#     we need to combine it first

# In[ ]:


data['actor_1_name'].fillna('unknown',inplace=True)
data['actor_2_name'].fillna('unknown',inplace=True)
data['actor_3_name'].fillna('unknown',inplace=True)


# In[ ]:


data[['actor_1_name','actor_2_name','actor_3_name']].head()


# In[ ]:


data['actors_name'] = data[['actor_1_name', 'actor_2_name','actor_3_name']].apply(lambda x: '|'.join(x), axis=1)


# In[ ]:


data[['actors_name']].head()


# In[ ]:


actor = []
actor = Obtain_list_Occurences("actors_name")


# In[ ]:


for word in actor:
    data[word] = data['actors_name'].map(lambda s: 1 if word in str(s) else 0)


# ## Company
# 
#     do the same with method which we do in actor

# In[ ]:


data['companies_1'].fillna('unknown',inplace=True)
data['companies_2'].fillna('unknown',inplace=True)
data['companies_3'].fillna('unknown',inplace=True)


# In[ ]:


data['companies_name'] = data[['companies_1', 'companies_2','companies_3']].apply(lambda x: '|'.join(x), axis=1)
company = []
company = Obtain_list_Occurences("companies_name")

for word in company:
    data[word] = data['companies_name'].map(lambda s: 1 if word in str(s) else 0)


#    **Let's delete data that not effect to model or already have a same information but in tidy format column**

# In[ ]:


data.drop(['id','budget','original_title','overview','spoken_languages','production_companies','production_countries','release_date','status',
          'tagline','movie_title','vote_average','language','director_name','actor_1_name','actor_2_name','actor_3_name',
          'companies_1','companies_2','companies_3','country','genres','duration','actors_name','companies_name'], axis=1, inplace=True)


# In[ ]:


data.info()


# ### Check missing data

# In[ ]:


total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(5)


# #  Create model step 

#     Split data to train and test set

# In[ ]:


data['is_train'] = np.random.uniform(0, 1, len(data)) <= .75

# View the top 5 rows
data.head()


# In[ ]:


# Create two new dataframes, one with the training rows, one with the test rows
train, test = data[data['is_train']==True], data[data['is_train']==False]

train.drop(['is_train'], axis=1, inplace=True)
test.drop(['is_train'], axis=1, inplace=True)

train["Nice"] = train["Nice"].astype(int)

Y_train = train["Nice"]
X_train = train.drop(labels = ["Nice"],axis = 1)
Y_test = test["Nice"]
X_test = test.drop(labels = ["Nice"],axis = 1)


# ** I split our dataset to 2 part **
# <br> *75% for train dataset *
# <br> *and 25% for test dataset *

# In[ ]:


print(len(train))


# In[ ]:


print(len(test))


# ## Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)


# ** Ok, test model performance by cross validation method **

# In[ ]:


from sklearn.model_selection import  cross_val_score

c_dec = cross_val_score(clf, X_train, Y_train, cv=10)
c_dec.mean()


# **However, why not try with test dataset**

# In[ ]:


result = clf.predict_proba(X_test)[:]
test_result = np.asarray(Y_test)


# In[ ]:


Dec_result = pd.DataFrame(result[:,1])
Dec_result['Predict'] = Dec_result[0].map(lambda s: 1 if s >= 0.6  else 0)
Dec_result['testAnswer'] = pd.DataFrame(test_result)

Dec_result['Correct'] = np.where((Dec_result['Predict'] == Dec_result['testAnswer'])
                     , 1, 0)
Dec_result.head()


# In[ ]:


Dec_result['Correct'].mean()


# ## Ok... Let, Create next models

# ## Try with another common classification algorithm

# ### K-Nearest Neighbors
# 
#  check with 60 neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors= 60 )
knn.fit(X_train, Y_train)


# * Cross validation score

# In[ ]:


c_knn = cross_val_score(knn, X_train, Y_train, cv=10)
c_knn.mean()


# * Test score

# In[ ]:


result = knn.predict_proba(X_test)[:]


# In[ ]:


knn_result = pd.DataFrame(result[:,1])
knn_result['Predict'] = knn_result[0].map(lambda s: 1 if s >= 0.6  else 0)
knn_result['testAnswer'] = pd.DataFrame(test_result)

knn_result['Correct'] = np.where((knn_result['Predict'] == knn_result['testAnswer'])
                     , 1, 0)
knn_result.head()


# In[ ]:


knn_result['Correct'].mean()


#     In this case, K-NN gave a low accuracy, because of a lot of unimportant features. It made error in distance between neighbors 

# 
# ###  It's time to try with ensembel model

# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

Rfclf = RandomForestClassifier()
Rfclf.fit(X_train, Y_train)


# ** Ok finish for create simple randomforest. **
# <br> Let see the performance by cross validation method

# * Cross validation score

# In[ ]:


c_rf =  cross_val_score(Rfclf, X_train, Y_train, cv=10)
c_rf.mean()


# * Test score

# In[ ]:


result = Rfclf.predict_proba(X_test)[:]


# In[ ]:


Rf_result = pd.DataFrame(result[:,1])
Rf_result['Predict'] = Rf_result[0].map(lambda s: 1 if s >= 0.6  else 0)
Rf_result['testAnswer'] = pd.DataFrame(test_result)

Rf_result['Correct'] = np.where((Rf_result['Predict'] == Rf_result['testAnswer'])
                     , 1, 0)
Rf_result.head()


# In[ ]:


Rf_result['Correct'].mean()


# ## Gradient Boosting 

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier  

gb = GradientBoostingClassifier()
gb.fit(X_train, Y_train)


# * Cross validation score

# In[ ]:


c_gb = cross_val_score(gb, X_train, Y_train, cv=10)
c_gb.mean()


# * Test score

# In[ ]:


result = gb.predict_proba(X_test)[:]


# In[ ]:


gb_result = pd.DataFrame(result[:,1])
gb_result['Predict'] = gb_result[0].map(lambda s: 1 if s >= 0.6  else 0)
gb_result['testAnswer'] = pd.DataFrame(test_result)

gb_result['Correct'] = np.where((gb_result['Predict'] == gb_result['testAnswer'])
                     , 1, 0)
gb_result.head()


# In[ ]:


gb_result['Correct'].mean()


# ** I think it's enough for my creating model step **

# # Conclusion model performance
#     
#         by using cross validation score

# In[ ]:


cv_means = []
cv_means.append(c_dec.mean())
cv_means.append(c_knn.mean())
cv_means.append(c_rf.mean())
cv_means.append(c_gb.mean())


# In[ ]:


cv_std = []
cv_std.append(c_dec.std())
cv_std.append(c_knn.std())
cv_std.append(c_rf.std())
cv_std.append(c_gb.std())


# In[ ]:


res1 = pd.DataFrame({"ACC":cv_means,"Std":cv_std,"Algorithm":["DecisionTree","K-Nearest Neighbors","Random Forest","Gradient Boosting"]})
res1["Type"]= "CrossValid"
g = sns.barplot("ACC","Algorithm",data = res1, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# ## Model Test 

# In[ ]:


tv_means = []
tv_means.append(Dec_result['Correct'].mean())
tv_means.append(knn_result['Correct'].mean())
tv_means.append(Rf_result['Correct'].mean())
tv_means.append(gb_result['Correct'].mean())


# In[ ]:


res2 = pd.DataFrame({"ACC":tv_means,"Algorithm":["DecisionTree","K-Nearest Neighbors","Random Forest","Gradient Boosting"]})
res2['Type'] = "Test";

g = sns.barplot("ACC","Algorithm",data = res2, palette="Set2",orient = "h")
g.set_xlabel("Mean Accuracy")
g = g.set_title("Test scores")


# In[ ]:


res = pd.concat([res1,res2])


# In[ ]:


g = sns.factorplot(x='Algorithm', y='ACC', hue='Type',palette="coolwarm", data=res, kind='bar')
g.set_xticklabels(rotation=90)


# # Ok let check the feature importance
# ## How many weight in each feature that each tree model give.

# In[ ]:


dec_fea = pd.DataFrame(clf.feature_importances_)
dec_fea["name"] = list(X_train) 
dec_fea.sort_values(by=0, ascending=False).head()


# In[ ]:


g = sns.barplot(0,"name",data = dec_fea.sort_values(by=0, ascending=False)[0:10], palette="Set2",orient = "h")
g.set_xlabel("Weight")
g = g.set_title("Decision Tree")


# In[ ]:


rf_fea = pd.DataFrame(Rfclf.feature_importances_)
rf_fea["name"] = list(X_train) 
rf_fea.sort_values(by=0, ascending=False).head()


# In[ ]:


g = sns.barplot(0,"name",data = rf_fea.sort_values(by=0, ascending=False)[0:10], palette="Set2",orient = "h")
g.set_xlabel("Weight")
g = g.set_title("Random Forest")


# In[ ]:


gb_fea = pd.DataFrame(gb.feature_importances_)
gb_fea["name"] = list(X_train) 
gb_fea.sort_values(by=0, ascending=False).head()


# In[ ]:


g = sns.barplot(0,"name",data = gb_fea.sort_values(by=0, ascending=False)[0:10], palette="Set2",orient = "h")
g.set_xlabel("Weight")
g = g.set_title("Gradient Boosting")


# ## How will it affect accuracy
# ### if I combine Top 3 performance models together by voting prediction

# ## Hard Voting

# In[ ]:


voting = pd.DataFrame()


# In[ ]:


voting["knn"] =knn_result['Predict']
voting["GB"] = gb_result['Predict']
voting["RF"] = Rf_result['Predict']
voting['sum'] = voting.sum(axis=1)

voting['Predict'] = voting['sum'].map(lambda s: 1 if s >= 2 else 0)

voting['testAnswer'] = pd.DataFrame(test_result)

voting['Correct'] = np.where((voting['Predict'] == voting['testAnswer'])
                     , 1, 0)


# In[ ]:


voting.head()


# In[ ]:


voting['Correct'].mean()


# In[ ]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix

print(confusion_matrix(voting['testAnswer'], voting['Predict']))


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(voting['testAnswer'], voting['Predict']))


# **Thank you for reading until the end : )** 
# 
#       This is my first kernel. I will try to update new version
#     please vote or comment If you like it ^_^
#     I am new kaggler, If you have any suggestion let me know in comment.
