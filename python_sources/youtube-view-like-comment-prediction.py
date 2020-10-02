#!/usr/bin/env python
# coding: utf-8

# # Youtube Likes,views Prediction
# 
# ## Table of content
# * 1.Machine Learning Formulation
#     * 1.1 Data Overview
#     * 1.2  Attribute-information
# * 2.EDA
# * 3.Feature Engineering
#     * 3.1 publish_weekday-Day at which video is publish
#     * 3.2 No of Tags-No of tag video contain
#     * 3.3 Length of description-Length of video description
#     * 3.4 Ratio's
#          * 3.4.1 Ratio of View and likes
#          * 3.4.2 Ratio of view and dislikes
#          * 3.4.3 Ratio of view and comment_count
#          * 3.4.4 Ratio of likes and dislikes
# * 4.Correlation matrix
# * 5.Machine Learning(metric=r^2 score )
#     * 5.1-View Predicition
#          * 5.1.1-Splitting the data into train and Test(80:20)
#          * 5.1.2-Linear Regression
#          * 5.1.3-Random Forest
#               
#   * 5.2 -Like Predicition
#        * 5.2.1-Splitting the data into train and Test(80:20)
#        * 5.2.2-Linear Regression
#        * 5.2.3-Random Forest
#        
#   * 5.3-comment Count Predicition
#       * 5.3.1-Splitting the data into train and Test(80:20)
#       * 5.3.2-Linear Regression
#       * 5.3.3-Random Forest
#       
# * 6.Conclusion
# 
# ## 1.Machine Learning Formulation
# ### 1.1 Data Overviews
# Contain one file
# 
# #### 1.2 Attribute-information
# * video_id-Unique video id
# * trending_date-the date at which video start trending
# * title-Title of video
# * channel_title-video posted by channel
# * category_id-there are 15 Category value
# * publish_time-at what time video is uplaoded
# * tags-tag given to video
# * views-no of views
# * likes-no of likes
# * dislikes-no of dislikes
# * comment_count-no of comment
# 
# 
# 

# ## 2.EDA

# ### Loading Libary

# In[ ]:


#Loading library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from nltk.stem import PorterStemmer
import nltk
from sklearn.metrics import r2_score


# In[ ]:


youtube = pd.read_csv("../input/youtube-new/INvideos.csv")


# In[ ]:


youtube.head()


# In[ ]:


print(youtube.shape)


# In[ ]:


print(youtube.isnull().values.any())


# In[ ]:


youtube = youtube.dropna(how='any',axis=0)


# In[ ]:


youtube.describe()


# In[ ]:


youtube.drop(['video_id','thumbnail_link'],axis=1,inplace=True)


# In[ ]:


youtube.apply(lambda x: len(x.unique()))


# In[ ]:


for x in (['comments_disabled','ratings_disabled','video_error_or_removed','category_id']):
    count=youtube[x].value_counts()
    print(count)
    plt.figure(figsize=(7,7))
    sns.barplot(count.index, count.values, alpha=0.8)
    plt.title('{} vs No of video'.format(x))
    plt.ylabel('No of video')
    plt.xlabel('{}'.format(x))
    plt.show()


# ## 3.Feature Engineering

#  * 3.1 publish_weekday-Day at which video is publish
#  * 3.2 No of Tags-No of tag video contain
#  * 3.3 Length of description-Length of video description
#  * 3.4 Ratio's
#     * 3.4.1 Ratio of View and likes
#     * 3.4.2 Ratio of view and dislikes
#     * 3.4.3 Ratio of view and comment_count
#     * 3.4.4 Ratio of likes and dislikes

# In[ ]:


#No of tags
tags=[x.count("|")+1 for x in youtube["tags"]]
youtube["No_tags"]=tags


# In[ ]:


#length of desription
desc_len=[len(x) for x in youtube["description"]]
youtube["desc_len"]=desc_len


# In[ ]:


#length of title
title_len=[len(x) for x in youtube["title"]]
youtube["len_title"]=title_len


# In[ ]:


publish_time = pd.to_datetime(youtube['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
youtube['publish_time'] = publish_time.dt.time
youtube['publish_date'] = publish_time.dt.date

#day at which video is publish
youtube['publish_weekday']=publish_time.dt.weekday_name


# In[ ]:


#ratio of view/likes  upto 3 decimal
youtube["Ratio_View_likes"]=round(youtube["views"]/youtube["likes"],3)
#ratio of view/dislikes  upto 3 decimal
youtube["Ratio_View_dislikes"]=round(youtube["views"]/youtube["dislikes"],3)
#ratio of view/comment_count  upto 3 decimal
youtube["Ratio_views_comment_count"]=round(youtube["views"]/youtube["comment_count"],3)
#ratio of likes/dislikes  upto 3 decimal
youtube["Ratio_likes_dislikes"]=round(youtube["likes"]/youtube["dislikes"],3)


# In[ ]:


print(max(youtube["Ratio_View_likes"]))
print(max(youtube["Ratio_View_dislikes"]))
print(max(youtube["Ratio_views_comment_count"]))
print(max(youtube["Ratio_likes_dislikes"]))


# In[ ]:


#removing the infinite values
youtube=youtube.replace([np.inf, -np.inf], np.nan)
youtube = youtube.dropna(how='any',axis=0)


# In[ ]:


youtube['publish_weekday'] = youtube['publish_weekday'].replace({'Monday':1,
                                                             'Tuesday':2,
                                                             'Wednesday':3,
                                                             'Thursday':4,
                                                             'Friday':5,
                                                             'Saturday':6,
                                                             'Sunday':7})


# In[ ]:


count=youtube["publish_weekday"].value_counts()
print(count)
plt.figure(figsize=(7,7))
sns.barplot(count.index, count.values, alpha=0.8)
plt.title('No of videos vs weekdays')
plt.ylabel('no of videos')
plt.xlabel('weekdays')
plt.show()


# ## 4.Correlation  Matrix
# 

# In[ ]:


data = youtube

corr = data.corr()
plt.figure(figsize=(12, 12))
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# ### Removing non Correlated coloumns

# In[ ]:


youtube.drop(['trending_date','publish_date','publish_time','tags','title','description','channel_title'],axis=1,inplace=True)


# ## 5.Machine Learning Models

# ## 5.1 Prediciting Views

# ### 5.1.1 spliting the data into train and test in ratio of  80:20 

# In[ ]:


views=youtube['views']
youtube_view=youtube.drop(['views'],axis=1,inplace=False)


# In[ ]:


train,test,y_train,y_test=train_test_split(youtube_view,views, test_size=0.2,shuffle=False)


# In[ ]:


print(train.shape,test.shape,y_train.shape,y_test.shape)


# ## 5.1.2 Linear Regression

# In[ ]:


# REGRESSION ANALYSIS

# LINEAR REGRESSION

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

model = LinearRegression()
model.fit(train, y_train)

# predicting the  test set results
y_pred = model.predict(test)
print('Root means score', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Variance score: %.2f' % r2_score(y_test, y_pred))
print("Result :",model.score(test, y_test))
d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}
SK = pd.DataFrame(data = d1)
print(SK)


# In[ ]:


lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data = SK, size = 10)
fig1 = lm1.fig 
fig1.suptitle("Sklearn ", fontsize=18)
sns.set(font_scale = 1.5)


# ## 5.1.3 Random Forest

# ## 5.1.3.1 Hyper-parameter Turning

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
nEstimator = [140,160,180,200,220]
depth = [10,15,20,25,30]

RF = RandomForestRegressor()
hyperParam = [{'n_estimators':nEstimator,'max_depth': depth}]
gsv = GridSearchCV(RF,hyperParam,cv=5,verbose=1,scoring='r2',n_jobs=-1)
gsv.fit(train, y_train)
print("Best HyperParameter: ",gsv.best_params_)
print(gsv.best_score_)
scores = gsv.cv_results_['mean_test_score'].reshape(len(nEstimator),len(depth))
plt.figure(figsize=(8, 8))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.colorbar()
plt.xticks(np.arange(len(nEstimator)), nEstimator)
plt.yticks(np.arange(len(depth)), depth)
plt.title('Grid Search r^2 Score')
plt.show()
maxDepth=gsv.best_params_['max_depth']
nEstimators=gsv.best_params_['n_estimators']


# ## 5.1.3.2 Random Forest using Optimal Hyperparameter

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = nEstimators,max_depth=maxDepth)
model.fit(train, y_train)


# predicting the  test set results
y_pred = model.predict(test)
print('Root means score', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Variance score: %.2f' % r2_score(y_test, y_pred))
print("Result :",model.score(test, y_test))
d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}
SK = pd.DataFrame(data = d1)
print(SK)


# In[ ]:


lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data = SK, size = 10)
fig1 = lm1.fig 
fig1.suptitle("Sklearn ", fontsize=18)
sns.set(font_scale = 1.5)


# ## 5.2 Prediciting Likes

# ### 5.2.1 spliting the data into train and test in ratio of  80:20 

# In[ ]:


likes=youtube['likes']
youtube_like=youtube.drop(['likes'],axis=1,inplace=False)


# In[ ]:


train,test,y_train,y_test=train_test_split(youtube_like,likes, test_size=0.2,shuffle=False)


# In[ ]:


print(train.shape,test.shape,y_train.shape,y_test.shape)


# ### 5.2.2 Linear Regression

# In[ ]:


# REGRESSION ANALYSIS

# LINEAR REGRESSION

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

model = LinearRegression()
model.fit(train, y_train)

# predicting the  test set results
y_pred = model.predict(test)
print('Root means score', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Variance score: %.2f' % r2_score(y_test, y_pred))
print("Result :",model.score(test, y_test))
d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}
SK = pd.DataFrame(data = d1)
print(SK)


# In[ ]:


lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data = SK, size = 10)
fig1 = lm1.fig 
fig1.suptitle("Sklearn ", fontsize=18)
sns.set(font_scale = 1.5)


# ### 5.2.3 Random Forest

# ### 5.2.3.1 Hypermeter Turning

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

nEstimator = [140,160,180,200,220]
depth = [10,15,20,25,30]

RF = RandomForestRegressor()
hyperParam = [{'n_estimators':nEstimator,'max_depth': depth}]
gsv = GridSearchCV(RF,hyperParam,cv=5,verbose=1,scoring='r2',n_jobs=-1)
gsv.fit(train, y_train)
print("Best HyperParameter: ",gsv.best_params_)
print(gsv.best_score_)
scores = gsv.cv_results_['mean_test_score'].reshape(len(nEstimator),len(depth))

plt.figure(figsize=(8, 8))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.colorbar()
plt.xticks(np.arange(len(nEstimator)), nEstimator)
plt.yticks(np.arange(len(depth)), depth)
plt.title('Grid Search r^2 Score')
plt.show()
maxDepth=gsv.best_params_['max_depth']
nEstimators=gsv.best_params_['n_estimators']


# ### 5.2.3.2 Random Forest using the optimal hypermeter

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = nEstimators,max_depth=maxDepth)
model.fit(train, y_train)


# predicting the  test set results
y_pred = model.predict(test)
print('Root means score', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Variance score: %.2f' % r2_score(y_test, y_pred))
print("Result :",model.score(test, y_test))
d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}
SK = pd.DataFrame(data = d1)
print(SK)


# In[ ]:


lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data = SK, size = 10)
fig1 = lm1.fig 
fig1.suptitle("Sklearn ", fontsize=18)
sns.set(font_scale = 1.5)


# ## 5.3 Prediciting No of Comment

# ### 5.3.1 spliting the data into train and test in ratio of  80:20 

# In[ ]:


comment_count=youtube['comment_count']
youtube_comment=youtube.drop(['comment_count'],axis=1,inplace=False)


# In[ ]:


train,test,y_train,y_test=train_test_split(youtube_comment,comment_count, test_size=0.2,shuffle=False)


# In[ ]:


print(train.shape,test.shape,y_train.shape,y_test.shape)


# ### 5.3.2 Linear Regression

# In[ ]:


# REGRESSION ANALYSIS

# LINEAR REGRESSION

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

model = LinearRegression()
model.fit(train, y_train)

# predicting the  test set results
y_pred = model.predict(test)
print('Root means score', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Variance score: %.2f' % r2_score(y_test, y_pred))
print("Result :",model.score(test, y_test))
d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}
SK = pd.DataFrame(data = d1)
print(SK)


# In[ ]:


lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data = SK, size = 10)
fig1 = lm1.fig 
fig1.suptitle("Sklearn ", fontsize=18)
sns.set(font_scale = 1.5)


# ### 5.3.3 Random Forest

# ### 5.3.3.1 Hypermeter Turning

# In[ ]:


nEstimator = [140,160,180,200,220]
depth = [10,15,20,25,30]

RF = RandomForestRegressor()
hyperParam = [{'n_estimators':nEstimator,'max_depth': depth}]
gsv = GridSearchCV(RF,hyperParam,cv=5,verbose=1,scoring='r2',n_jobs=-1)
gsv.fit(train, y_train)
print("Best HyperParameter: ",gsv.best_params_)
print(gsv.best_score_)
scores = gsv.cv_results_['mean_test_score'].reshape(len(nEstimator),len(depth))

plt.figure(figsize=(8, 8))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.colorbar()
plt.xticks(np.arange(len(nEstimator)), nEstimator)
plt.yticks(np.arange(len(depth)), depth)
plt.title('Grid Search r^2 Score')
plt.show()
maxDepth=gsv.best_params_['max_depth']
nEstimators=gsv.best_params_['n_estimators']


# ### 5.3.3.1 RandomForest optimal Hyper-Parameter

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = nEstimators,max_depth=maxDepth)
model.fit(train, y_train)


# predicting the  test set results
y_pred = model.predict(test)
print('Root means score', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Variance score: %.2f' % r2_score(y_test, y_pred))
print("Result :",model.score(test, y_test))
d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}
SK = pd.DataFrame(data = d1)
print(SK)


# In[ ]:


lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data = SK, size = 10)
fig1 = lm1.fig 
fig1.suptitle("Sklearn ", fontsize=18)
sns.set(font_scale = 1.5)


# ## 6.Conclusion
# ### View Predicition
# 
# |Model|Variance|Result|
# |-----|--------|------|
# |Linear Regression|0.73|0.734|
# |Random Forests|0.98|0.984|
# 
# ### Like Predicition
# 
# |Model|Variance|Result|
# |-----|--------|------|
# |Linear Regression|0.41|0.411|
# |Random Forests|0.96|0.958|
# 
# 
# ### No of Comment
# 
# |Model|Variance|Result|
# |-----|--------|------|
# |Linear Regression|0.35|0.35|
# |Random Forests|0.81|0.81|
# 

# In[ ]:




