#!/usr/bin/env python
# coding: utf-8

# Get the Google Maps API key https://developers.google.com/places/web-service/?hl=de (for more than 1000 requests/sec add payment information)
# To install populartimes package https://github.com/m-wrzr/populartimes 
# Get the movie show time API key https://api.internationalshowtimes.com
# 
# Briefly Explanation
# The analysis purpose for this project is to identify the significant factors for cinemas popularity.
# 
# The first step I have done is to use populartimes API package to collect hourly popularity values in 9 cinemas on the 2 weekend May 2018. Then I collected several potential factors would influence popularity such as cinema ratings, local population, the number of showing movies, movie ratings, etc. After finishing the data collection part, I combined all the dataset into a single one. Then I used multiple models such as linear, logistic, random forest, cross-validation and found the hour is the most significant factors to cinemas popularity. 

# In[ ]:


import csv
import sys
#from googleplaces import GooglePlaces, types, lang
from decimal import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Apply for your api key

# In[ ]:


#YOUR_API_KEY = 'YOUR_API_KEY'
#google_places = GooglePlaces(YOUR_API_KEY)


# Get the location information about the location of cities in NY state.

# In[ ]:


reader = csv.reader(open('../input/nycity/NYcities.csv', 'rU'), dialect='excel')
gpstrip = []
for line in reader:
    gpstrip.append(line)
gpstrip.pop(0)
print (gpstrip[:5])


# Use below code to collect local features through google place API

# In[ ]:


# with open('../input/movietheater/555/movie_theater.csv', 'w') as csvfile:
#     reload(sys)
#     sys.setdefaultencoding('utf-8')
#     spamwriter = csv.writer(csvfile, dialect='excel')
#     spamwriter.writerow(['item','name','place_id','latitude','longitude'])
#     for i in range(len(gpstrip)):
#         query_result = google_places.nearby_search(
#        lat_lng={u'lat': Decimal(str(gpstrip[i][1])), u'lng': Decimal(str(gpstrip[i][2]))},
#         radius=10000, types=[types.TYPE_MOVIE_THEATER])
#         for n in query_result.places:
#             spamwriter.writerow([i+1,n.name,n.place_id,n.geo_location['lat'],n.geo_location['lng']])
            


# follow https://github.com/m-wrzr/populartimes to install populartimes package

# In[ ]:


#from googleplaces import GooglePlaces, types, lang
#import populartimes
import time
from datetime import timedelta
from datetime import datetime


# Use below code to automatically collect real time theater popular per hour 

# In[ ]:


# theater=pd.read_csv('../input/movietheater/theaterlocation.csv')
# store=[]
# while time.localtime(time.time()).tm_mday < 7 and time.localtime(time.time()).tm_hour <=23:
#     print 'working'
#     for i in range(len(theater)):
#         a=populartimes.get_id(YOUR_API_KEY, theater['place_id'].iloc[i])
#         mon=time.localtime(time.time()).tm_mon
#         day=time.localtime(time.time()).tm_mday
#         hour=time.localtime(time.time()).tm_hour
#         store.append([a['name'],a['rating'],a['rating_n'],a['current_popularity'],
#         mon,day,hour])
#     print 'sleeping'
#     df=pd.DataFrame(data=store,columns=['name','rating','rating_n','current_popularity',
#     'recordtime_mon','recordtime_day','recordtime_hour'])
#     df.to_csv('theaterpopularityMay6.csv')
#     time.sleep(60*59+30)


# use api.internationalshowtimes.com to apply your api key

# In[ ]:


# import requests
# import urllib, json
# key="your key"
# def getcinemaid(location,key):
#     try:
#         response = requests.get(
#             url="https://api.internationalshowtimes.com/v4/cinemas/",
#             params={
#                 "distance":'2',
#                 "location":location,
#             },
#             headers={
#                 "X-API-Key": key,
#             },
#         )
#         return (response.json())
#     except requests.exceptions.RequestException:
#         print('HTTP Request failed')

# def getmovieid(cinemaid,location,key):
#     try:
#         response = requests.get(
#             url="https://api.internationalshowtimes.com/v4/movies/",
#             params={
#                 "distance":'2',
#                 "location":location,
#                 "cinema_id":cinemaid,
#                 "time-to":'2018-05-07T04:00:00-08:00',
#             },
#             headers={
#                 "X-API-Key": key,
#             },
#         )
#         return (response.json())
#     except requests.exceptions.RequestException:
#         print('HTTP Request failed')

# def getshowtime(movieid,location,key):
#     try:
#         response = requests.get(
#             url="https://api.internationalshowtimes.com/v4/showtimes/",
#             params={
#                 "distance":'2',
#                 "location":location,
#                 "movie_id":movieid,
#                 "time_to":'2018-05-07T04:00:00-08:00',
#             },
#             headers={
#                 "X-API-Key": key,
#             },
#         )
#         return (response.json())
#     except requests.exceptions.RequestException:
#         print('HTTP Request failed')
       


# In[ ]:


dataset=[]
df=pd.read_csv('../input/theater-location/theaterlocation.csv')


# Use below code to get movie show time data

# In[ ]:


# for i in range(len(df)):
#     location= "%s , %s" % (df["latitude"][i],df["lngitude"][i])
#     cinema=getcinemaid(location,key)
#     cinemaid=cinema['cinemas'][0]['id']
#     city=df['city'][i]
#     theater=df['theater'][i]
#     movie=getmovieid(cinemaid,location,key)
#     for j in range(len(movie["movies"])):
#         title=movie["movies"][j]['title']
#         movieid=movie["movies"][j]['id']
#         show=getshowtime(movieid,location,key)       
#         for n in range(len(show['showtimes'])):
#             showtime=show['showtimes'][n]['start_at']
#             is_3d=show['showtimes'][n]['is_3d']
#             is_imax=show['showtimes'][n]['is_imax']
#             language=show['showtimes'][n]['language']
#             dataset.append([city,theater,title,showtime,is_3d,is_imax,language])     


# In[ ]:


movieshow=pd.DataFrame(data=dataset,columns=["city","theater","movie","showtime","is_3d","is_imax","language"])
print (movieshow.head())


# In[ ]:


import csv
import sys
import pandas as pd
movie=pd.read_csv('../input/movietheater/movieshowtimenew.csv', header=0)
popular=pd.read_csv('../input/movietheater/theaterpopularityMay.csv', header=0)
movie.head()


# In[ ]:


movie.groupby('audience score').size().sort_values(ascending=False).plot(kind='bar')


# In[ ]:


movie.groupby(['city','theater']).size()


# In[ ]:


movie.groupby('city').size().sort_values(ascending=False).plot(kind='bar')


# In[ ]:


movie.groupby(['city','theater']).size().sort_values(ascending=False).plot(kind='bar')


# In[ ]:


movie.groupby(['movie']).size().sort_values(ascending=False).plot(kind='bar')


# In[ ]:


popular.head()


# In[ ]:


movie['Start Day'] = pd.DatetimeIndex(movie['start time (local)']).day
movie['Start Hour'] = pd.DatetimeIndex(movie['start time (local)']).hour
movie['End Day'] = pd.DatetimeIndex(movie['end time (local)']).day
movie['End Hour'] = pd.DatetimeIndex(movie['end time (local)']).hour
movie['week Day'] = pd.DatetimeIndex(movie['start time (local)']).weekday
movie.head()


# In[ ]:


movie.groupby(['Start Hour']).movie.size().plot(kind='bar')


# In[ ]:


movie.groupby(['End Hour']).movie.size().plot(kind='bar')


# In[ ]:


sm=movie.groupby(['Start Hour','movie']).size()
sm=sm.reset_index()
sm.columns=['Hour','Movie','Count']
sm.head()


# In[ ]:


sm.pivot(index="Hour", columns="Movie", values="Count").plot()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


movie.groupby('week Day').size().sort_values(ascending=False).plot(kind='bar')


# In[ ]:


store=[]
for i in range(len(popular)):
    tp=movie[movie['theater']==popular['name'][i]]
    tp1=tp[tp['Start Day']==popular['recordtime_day'][i]] 
    tp2=tp1[tp1['End Day']==popular['recordtime_day'][i]]
    tp3=tp2[tp2['Start Hour']<=popular['recordtime_hour'][i]]
    tp4=tp3[tp3['End Hour'] > popular['recordtime_hour'][i]]
    movienm=len(tp4)
    localpopulation=max(tp['local population'])
    attraction=0    
    attraction += sum(tp4['average rating'] * tp4['rating number'] * tp4['audience score'])
    store.append([popular['current_popularity'][i],popular['recordtime_hour'][i],localpopulation,movienm,attraction,
                      popular['rating'][i],popular['rating_n'][i]])
newdata=pd.DataFrame(data=store,columns=['current_popularity','Hour','Local_population','Movies','Attraction','cinemarating','cinemarating_nm'])
dataset=newdata[newdata['Movies'] != 0] 
dataset.head()   


# In[ ]:


dataset.groupby('Local_population').current_popularity.sum().sort_values(ascending=False).plot(kind='bar')


# In[ ]:


dataset['logpopulation']=np.log(dataset['Local_population'])
dataset['logAttraction']=np.log(dataset['Attraction'])
dataset['logcinemarating_nm']=np.log(dataset['cinemarating_nm'])
del dataset['Local_population']
del dataset['Attraction']
del dataset['cinemarating_nm']
dataset.head()


# In[ ]:


dataset.describe()


# In[ ]:


dataset.groupby('Hour').current_popularity.sum().plot(kind='bar')


# In[ ]:


plt.scatter(data=dataset,x='Movies',y='current_popularity')
plt.xlabel('Movies')
plt.ylabel('Current_popularity')


# In[ ]:


dataset.groupby('Movies').size().plot(kind='line')


# In[ ]:


corr=dataset.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[ ]:


from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
import statsmodels.formula.api as smf


# In[ ]:


# create a fitted model with all features
lm1 = smf.ols(formula='current_popularity ~ Hour + Movies + cinemarating + logpopulation + logAttraction + logcinemarating_nm',
              data=dataset).fit()

# print the coefficients
lm1.summary()


# In[ ]:


# remove movies 
lm1 = smf.ols(formula='current_popularity ~ Hour + cinemarating + logpopulation + logcinemarating_nm',
              data=dataset).fit()

# print the coefficients
lm1.summary()


# In[ ]:


# use cross validation 
from sklearn.model_selection import KFold
from scipy import stats
from sklearn.metrics import r2_score


# In[ ]:


predictors=['Hour','cinemarating', 'logpopulation', 'logcinemarating_nm']
alg=LogisticRegression()


# In[ ]:


#use cross validation model to select 3-49 parts
# parts=[]
# r2=[]
# for j in range(3,50):
#     kf=KFold(dataset.shape[0],n_folds=j,random_state=1)
#     predictions=[]
#     for train, test in kf:
#         train_predictors=(dataset[predictors].iloc[train,:])
#         train_target=dataset['current_popularity'].iloc[train]
#         alg.fit(train_predictors,train_target)
#         test_predictors=alg.predict(dataset[predictors].iloc[test,:])
#         predictions.append(test_predictors)
#     x=np.concatenate(predictions,axis=0)
#     y=dataset['current_popularity']
#     slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
#     def linefitline(b):
#         return intercept + slope * b
#     line1 = linefitline(x)
#     line2 = np.full(313,[y.mean()])
#     differences_line1 = linefitline(x)-y
#     line1sum = 0
#     for i in differences_line1:
#         line1sum = line1sum + (i*i)
#     line1sum
#     differences_line2 = line2 - y
#     line2sum = 0
#     for i in differences_line2:
#         line2sum = line2sum + (i*i)
#         line2sum
#     r2.append (r2_score(y, linefitline(x)))
#     parts.append(j)    


# In[ ]:


# productDataset = list(zip(parts, r2))
# cd= pd.DataFrame(data=productDataset, columns=['Parts', 'Accuracy'])
# plt.scatter(cd['Parts'],cd['Accuracy'])


# In[ ]:


# # random forest
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import model_selection
# predictors=['Hour','cinemarating', 'logpopulation', 'logcinemarating_nm']
# alg=RandomForestClassifier(random_state=1,n_estimators=60,min_samples_split=4,min_samples_leaf=2)
# kf=KFold(dataset.shape[0],n_folds=21,random_state=1)
# scores=cross_validation.cross_val_score(alg,dataset[predictors],dataset['current_popularity'],cv=kf)
# scores.mean()


# In[ ]:


# select important features
from sklearn.feature_selection import SelectKBest, f_classif
predictors=['Hour','cinemarating', 'logpopulation', 'logcinemarating_nm','Movies']
selector=SelectKBest(f_classif,k=3)
selector.fit(dataset[predictors],dataset['current_popularity'])
scores=-np.log10(selector.pvalues_ )
plt.bar(range(len(predictors)),scores)
plt.xticks(range(len(predictors)),predictors,rotation='vertical')

