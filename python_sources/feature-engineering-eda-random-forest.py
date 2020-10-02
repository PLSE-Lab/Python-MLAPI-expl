#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import packages and read in data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from ast import literal_eval
from datetime import datetime

train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')
test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')
sub=test.copy()


# In[ ]:


print (train.shape)
print (test.shape)


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


data=train.append(test)
data.reset_index(inplace=True)
data.drop('index',axis=1,inplace=True)
data.head(3)


# ### Data Descriptive Analysis

# In[ ]:


data.shape


# In[ ]:


data.describe(include='all')


# In[ ]:


data[data['revenue']==data['revenue'].max()]


# In[ ]:


data[data['budget']==data['budget'].max()]


# In[ ]:


data.head()


# # Feature Engineering

# In[ ]:


# Let's see how much missing data we have here first
data.isnull().sum()


# In[ ]:


# There are severl columns with high percentage of missing values. I may consider to drop them afterward.
print ('The percentage of missing value of each column:')
print ('*'*50)
print (round(data.isnull().sum()/data.shape[0]*100,2))


# In[ ]:


# For some categorical data, I'd like to fill in na with mode. I check the mode of them first to see if it is reasonable.
print (data['genres'].mode()[0])
print (data['spoken_languages'].mode()[0])
print (data['production_companies'].mode()[0])
print (data['production_countries'].mode()[0])


# In[ ]:


# There is only one missing value in 'release_date'. I'll google it and fill na manually. 
data[data['release_date'].isnull()]


# In[ ]:


# The most ideal way to fill in missing value of 'spoken_language' is to use 'original_language'. However, the data 
# format is quite different, may lead to redundant process. There are only 7 of them are not English. So I'll stick
# to using mode to fill na for 'spoken_languages'.
data[data['spoken_languages'].isnull()]['original_language'].value_counts()


# In[ ]:


# Since I will convert 'cast' and 'crew' into no. of cast/crew after feature engineering, I just fill in 0 here. 
data['cast'].fillna('0',inplace=True)
data['crew'].fillna('0',inplace=True)

# As mentioned before, these four columns fill in with mode.
data['genres'].fillna(data['genres'].mode()[0],inplace=True)
data['production_countries'].fillna(data['production_countries'].mode()[0],inplace=True)
data['production_companies'].fillna(data['production_companies'].mode()[0],inplace=True)
data['spoken_languages'].fillna(data['spoken_languages'].mode()[0],inplace=True)

# Google says this movie's release date is 3/20/01. I choose to believe Google.
data['release_date'].fillna('3/20/01',inplace=True)

# For the continuous variable, fill in with mean value.
data['runtime'].fillna(data['runtime'].mean(),inplace=True)

# Just using 'original_title' to fill in 'title'.
data['title'].fillna(data['original_title'],inplace=True)


# In[ ]:


# Beautiful!! We have sorted most of the missing values.
data.isnull().sum()


# In[ ]:


# Convert 'belongs_to_collection' to binary value: is or is not serial movie.
data['belongs_to_collection'].fillna(0,inplace=True)
data['belongs_to_collection']=data['belongs_to_collection'].apply(lambda x:1 if x!=0 else x)


# In[ ]:


# Almost all of movies are released. This variable maybe is not so useful.
data['status'].value_counts()


# In[ ]:


# I'm not gonna dig into analysing 'words' stuffs. These are the variables I'm not gonna use in my model. 
notusing=['Keywords',
         'homepage',
         'id',
         'imdb_id',
         'original_language',
         'original_title',
         'overview',
         'poster_path',
         'status',
         'tagline']
data.drop(notusing,axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


# Now let's create some functions dealing with the columns with json-like values.
def find_name(string):
    s=eval(string) # list of dict
    l=[]
    for i in s:
        l.append(i['name'])
    return l

def find_language(string):
        t=eval(string)
        l=[]
        for i in t:
            l.append(i['iso_639_1'])
        return l

def find_actors(string):
    if eval(string)==0:
        return 0
    else:
        t=eval(string)
        l=[]
        for i in t:
            l.append(i['name'])
        return l


# In[ ]:


# Apply the functions to those json-like columns.
data['cast']=data['cast'].apply(find_actors)
data['crew']=data['crew'].apply(find_actors)
data['genres']=data['genres'].apply(find_name)
data['production_companies']=data['production_companies'].apply(find_name)
data['production_countries']=data['production_countries'].apply(find_name)
data['spoken_languages']=data['spoken_languages'].apply(find_language)

# I converted 'cast' and 'crew' into the no. of cast/crew, which is doable after the previous process.
data['no_of_cast']=data['cast'].apply(lambda x:len(x) if x!=0 else 0)
data['no_of_crew']=data['crew'].apply(lambda x:len(x) if x!=0 else 0)

data.drop(['cast','crew'],axis=1,inplace=True)

data.head()


# In[ ]:


# Most of the movies containing 1 to 3 genres. Some have more. 
print ('Movies with each no. of genres')
print ('*'*50)
print (data['genres'].apply(lambda x:len(x)).value_counts())


# In[ ]:


# Convert the 'genres' into dummy variables.
# The logic behind this transformation can be found here. https://stackoverflow.com/questions/29034928/pandas-convert-a-column-of-list-to-dummies
# It is quite clear to me, doing me a huge favor.
data=pd.get_dummies(data['genres'].apply(pd.Series).stack()).sum(level=0).merge(data,left_index=True,right_index=True)


# In[ ]:


data.head()


# ### The way I fill in missing value in 'budget', which is 0, is fill in with the mean budget of the genres that movie contains. I calculate mean budget of each genre and then put it back to the missing budget. 

# In[ ]:


# Firtly, calculate the mean budget of each genre.

list_of_genres=[]
for i in data['genres']:
    for j in i:
        if j not in list_of_genres:
            list_of_genres.append(j)

d={}
for i in list_of_genres:
    genre=i
    mean_budget=data.groupby(i)['budget'].mean()
    d[genre]=mean_budget[1]
    
pd.Series(d).sort_values()


# In[ ]:


# This part is just for inspection. To see how many countries/companies/languages in total
list_of_companies=[]
for i in data['production_companies']:
    for j in i:
        if j not in list_of_companies:
            list_of_companies.append(j)

list_of_countries=[]
for i in data['production_countries']:
    for j in i:
        if j not in list_of_countries:
            list_of_countries.append(j)
len(list_of_countries)

list_of_language=[]
for i in data['spoken_languages']:
    for j in i:
        if j not in list_of_language:
            list_of_language.append(j)
len(list_of_language)

print ('The total number of company occurs is {}'.format(len(list_of_companies)))
print ('The total number of country occurs is {}'.format(len(list_of_countries)))
print ('The total number of language occurs is {}'.format(len(list_of_language)))


# In[ ]:


# Replace the 0 budget with nan.
data['budget'].replace(0,np.nan,inplace=True)
data[data['budget'].isnull()][['budget','genres']].head(10)


# In[ ]:


# This function will calculate the mean budget value of that movie.
# For example, for the index 4 movie, the function will calculate the mean of the mean budget of Action and the mean
# budget of Thriller. 

def fill_budget(l):
    el=[]
    for i in l:
        if d[i] not in el:
            el.append(d[i])
    return (np.mean(el))


# In[ ]:


data['budget'].fillna(data['genres'].apply(fill_budget),inplace=True)


# In[ ]:


# Most of the movies are produced by 1 to 3 companies. Some have more. 
print ('Movies with each no. of production company')
print ('*'*50)
data['production_companies'].apply(lambda x:len(x)).value_counts()


# In[ ]:


# Most of the movies was shoot in under 2 countries. Some have more. 
print ('Movies with each no. of production_countries')
print ('*'*50)
data['production_countries'].apply(lambda x:len(x)).value_counts()


# In[ ]:


# Surprisingly, the budget doesn't have much to do with how many countries the movie was shoot, but how many companies
# involved.

data['no_of_country']=data['production_countries'].apply(lambda x:len(x))
data['no_of_company']=data['production_companies'].apply(lambda x:len(x))
data[['budget','no_of_country','no_of_company']].corr()


# ### Deal with release date

# In[ ]:


data['release_date'].head()


# In[ ]:


# If we just apply the datetime function of panda, there will be some year like 2048, 2050, 2072 happen. I'm pretty
# sure the time traveling has not been invented yet. The year has to be handled first. If it is greater than 18, it 
# must be 19xx.

def fix_year(x):
    year=x.split('/')[2]
    if int(year)>18:
        return x[:-2]+'20'+year
    else:
        return x[:-2]+'19'+year
data['release_date']=data['release_date'].apply(fix_year)
data['release_date']=pd.to_datetime(data['release_date'],infer_datetime_format=True)


# In[ ]:


# There still might be some ambiguities like 11, 15, 09. How does computer know it refers to 2011 or 1911? It don't. 
# So eventually I decided not to use the 'year'. But the month and date and weekday can still be really informative. 

#data['year']=data['release_date'].dt.year
data['month']=data['release_date'].dt.month
data['day']=data['release_date'].dt.day
data['weekday']=data['release_date'].dt.weekday

# Mapping weekday and month.
data['weekday']=data['weekday'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thur',4:'Fri',5:'Sat',6:'Sun'})
data['month']=data['month'].map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'July',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})


# In[ ]:


data[['release_date','month','day','weekday']].head()


# In[ ]:


data.drop(['release_date'],axis=1,inplace=True)
data.iloc[:5,20:]


# In[ ]:


# There are nearly 90 movies are English. 
# I'd like to know will the movie is/is not foreign language affect the revenue?
l=[]
for i in data['spoken_languages']:
    if 'en' in i:
        l.append(i)

len(l)/data.shape[0]


# In[ ]:


# Convert 'spoken_languages' into binary variable 'language_en'.
def en_or_not(l):
    if 'en' in l:
        return 1
    else:
        return 0
data['language_en']=data['spoken_languages'].apply(en_or_not)
data.drop('spoken_languages',axis=1,inplace=True)


# In[ ]:


# Same situation in 'production_countries'. Nearly 80% movies were shoot in USA.
u=[]
for i in data['production_countries']:
    if 'United States of America' in i:
        u.append(i)
        
len(u)/data.shape[0]


# In[ ]:


# Convert 'production_countries' into binary variable 'produce_in_USA'
def usa_or_not(l):
    if 'United States of America' in l:
        return 1
    else:
        return 0
data['produce_in_USA']=data['production_countries'].apply(usa_or_not)
data.drop('production_countries',axis=1,inplace=True)


# In[ ]:


top_company=pd.read_csv('../input/top-company-list/top_company.csv')
top_company=top_company['Production Company'].tolist()
top_company[:5]


# In[ ]:


data.iloc[:,20:].head()


# In[ ]:


#data['top_director']=data['director'].apply(lambda x:1 if x in top_director else 0)
def get_top_company(l):
    n=0
    for i in l:
        if i in top_company:
            n+=1
    return n

data['top_production_company']=data['production_companies'].apply(get_top_company)
data.drop('production_companies',axis=1,inplace=True)


# ## Normalisation

# In[ ]:


# To avoid the model being affected by the scale of each variable, normalise the continuous variable.

data['budget']=data['budget'].apply(lambda x:(x-np.min(data['budget']))/(np.max(data['budget']-np.min(data['budget']))))
data['popularity']=data['popularity'].apply(lambda x:(x-np.min(data['popularity']))/(np.max(data['popularity']-np.min(data['popularity']))))
data['runtime']=data['runtime'].apply(lambda x:(x-np.min(data['runtime']))/(np.max(data['runtime']-np.min(data['runtime']))))


# In[ ]:


# Set the index to movie title, and we are ready to go!!

data.set_index('title',inplace=True)
data.head()


# In[ ]:


plt.figure(figsize=(20,12))
plt.title('Violin plot of revenue of each month',fontsize=20)
sns.violinplot(x=data[data['revenue'].notnull()]['month'],y=data[data['revenue'].notnull()]['revenue'],scale='count')


# In[ ]:


plt.figure(figsize=(20,12))
plt.title('Violin plot of revenue of each weekday',fontsize=20)
sns.violinplot(x=data[data['revenue'].notnull()]['weekday'],y=data[data['revenue'].notnull()]['revenue'],scale='count')


# In[ ]:


data.drop('genres',axis=1,inplace=True)

# Month and weekday need to be converted to dummy variables.
data=pd.get_dummies(data)


# In[ ]:


data.iloc[:5,20:29]


# ### All done!!! We're good to do some exploratory analysis!

# # EDA

# In[ ]:


# For the EDA, I only use the training dataset.

Train=data[data['revenue'].notnull()]
Train.head(5)


# In[ ]:


# Does a movie is serial or not matter?
print (Train.groupby('belongs_to_collection')['revenue'].mean())
Train.groupby('belongs_to_collection')['revenue'].mean().plot.barh()


# In[ ]:


Train['belongs_to_collection'].value_counts()


# In[ ]:


sns.swarmplot(x=Train['belongs_to_collection'],y=Train['revenue'])


# In[ ]:


list_of_genres


# In[ ]:


# Similar to creating a series of mean budget of each genre, here we create 'revenue'.

g={}
for i in list_of_genres:
    mean_rev=Train.groupby(i)['revenue'].mean()
    g[i]=mean_rev[1]

g


# In[ ]:


plt.figure(figsize=(20,8))
pd.Series(g).sort_values().plot.barh()
plt.title('Mean revenue of each genre',fontsize=20)
plt.xlabel('Revenue',fontsize=20)


# In[ ]:


print (pd.DataFrame(Train.groupby('language_en')['revenue'].mean()))

plt.figure(figsize=(10,4))
Train.groupby('language_en')['revenue'].mean().sort_values().plot.barh()
plt.title('Mean revenue of is or is not foreign film.',fontsize=20)
plt.xlabel('Revenue',fontsize=20)


# In[ ]:


plt.figure(figsize=(10,4))
sns.swarmplot(x=Train['language_en'],y=Train['revenue'])
plt.title('Swarm plot of is or is not foreign film',fontsize=20)


# In[ ]:


print (pd.DataFrame(Train.groupby('produce_in_USA')['revenue'].mean()))

plt.figure(figsize=(10,4))
Train.groupby('produce_in_USA')['revenue'].mean().sort_values().plot.barh()
plt.title('Mean revenue of shoot in USA or not')
plt.xlabel('revenue')


# In[ ]:


plt.figure(figsize=(10,4))
sns.swarmplot(x=Train['produce_in_USA'],y=Train['revenue'])
plt.title('Swarm plot of movie produced in USA or not',fontsize=20)


# In[ ]:


plt.figure(figsize=(10,4))
plt.title('Mean revenue of each no. of top production company',fontsize=20)
Train.groupby('top_production_company')['revenue'].mean().plot.bar()
plt.xlabel('No. of top production company',fontsize=20)
plt.ylabel('Revenue',fontsize=20)


# In[ ]:


plt.figure(figsize=(10,4))
plt.title('Swarm plot of mean revenue of each no. of top production company',fontsize=20)
plt.xlabel('No. of top production company',fontsize=20)
plt.ylabel('Revenue',fontsize=20)

sns.swarmplot(x=Train['top_production_company'],y=Train['revenue'])


# In[ ]:


plt.figure(figsize=(8,8))
plt.scatter(Train['runtime'],Train['revenue'])
plt.title('Scatter plot of runtime vs revenue',fontsize=20)
plt.xlabel('runtime',fontsize=20)
plt.ylabel('Revenue',fontsize=20)


# In[ ]:


plt.figure(figsize=(8,8))
plt.scatter(Train['budget'],Train['revenue'])
plt.title('Scatter plot of budget vs revenue',fontsize=20)
plt.xlabel('budget',fontsize=20)
plt.ylabel('Revenue',fontsize=20)


# In[ ]:


plt.figure(figsize=(8,8))
plt.scatter(Train['popularity'],Train['revenue'])
plt.title('Scatter plot of popularity vs revenue',fontsize=20)
plt.xlabel('popularity',fontsize=20)
plt.ylabel('Revenue',fontsize=20)


# In[ ]:


month=['Jan','Feb','Mar','Apr','May','Jun','July','Aug','Sep','Oct','Nov','Dec']
m={}
for i in month:
    mean=Train.groupby('month_'+i)['revenue'].mean()
    m[i]=mean[1]
pd.Series(m)


# In[ ]:


for i in month:
    print (i,Train['month_'+i].value_counts()[1])


# In[ ]:


plt.figure(figsize=(20,8))
pd.Series(m).plot.bar()
plt.title('Mean revenue of each month',fontsize=20)
plt.xlabel('Revenue',fontsize=20)


# In[ ]:


plt.figure(figsize=(20,8))
Train.groupby('day')['revenue'].mean().sort_values().plot.bar()
plt.title('Mean revenue of each day',fontsize=20)
plt.xlabel('Revenue',fontsize=20)


# In[ ]:


w={}
weekday=['Mon','Tue','Wed','Thur','Fri','Sat','Sun']
for i in weekday:
    mean=Train.groupby('weekday_'+i)['revenue'].mean()
    w[i]=mean[1]
w


# In[ ]:


for i in weekday:
    print (i,Train['weekday_'+i].value_counts()[1])


# In[ ]:


plt.figure(figsize=(20,8))
pd.Series(w).plot.bar()
plt.title('Mean revenue of each weekday',fontsize=20)
plt.xlabel('Revenue',fontsize=20)


# In[ ]:


plt.figure(figsize=(10,4))
sns.distplot(Train['budget'])
plt.title('Distribution of budget')


# In[ ]:


plt.figure(figsize=(10,4))
sns.distplot(Train['revenue'])
plt.title('Distribution of revenue')


# In[ ]:


plt.figure(figsize=(10,4))
sns.distplot(data['popularity'])
plt.title('Distribution of popularity')


# In[ ]:


Train.iloc[:5,20:30]


# In[ ]:


num_var=['budget','popularity','runtime','no_of_cast','no_of_crew','language_en','produce_in_USA',
         'revenue','no_of_country','no_of_company','top_production_company','belongs_to_collection']
corr_table=Train[num_var].corr()
corr_table['revenue'].sort_values(ascending=False)


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(corr_table,vmax=True,annot=True,square=True,cmap='YlGnBu')
plt.title('Heatmap of the correlation between each numerical variable')
sns.set(font_scale=1.5)


# # Machine Learning Model

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, r2_score


# In[ ]:


rf = RandomForestRegressor(n_estimators=2000,
                           min_samples_split=15, 
                           min_samples_leaf=5, 
                           oob_score=True, 
                           n_jobs=-1, 
                           random_state=101)
X=Train.loc[:,Train.columns!='revenue']
y=Train['revenue']
X_train,X_test,y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=101)


# In[ ]:


rf.fit(X_train,y_train)


# In[ ]:


pred = rf.predict(X_test)
explained_variance_score(y_test,pred)


# In[ ]:


pd.DataFrame({'variable':X.columns.tolist(),
              'importance':rf.feature_importances_}).sort_values(by='importance',
                                                                 ascending=False).head(10)


# In[ ]:


rf.oob_score_


# In[ ]:


r2_score(y_test,pred)


# In[ ]:


Test=data[data['revenue'].isnull()]


# In[ ]:


Test['revenue']=rf.predict(Test.loc[:,Test.columns!='revenue']).astype(int)


# In[ ]:


sub['revenue']=rf.predict(Test.loc[:,Test.columns!='revenue']).astype(int)


# In[ ]:


sub=sub[['id','revenue']]
sub.to_csv('submission.csv',index=False)


# In[ ]:




