#!/usr/bin/env python
# coding: utf-8

# # Predicting TED Talks Views with ML Models
# In this notebook I've done some simple feature engineering on the TED Talks dataset and I've built machine learning models (Random Forest, XGBRegressor, ExtraTreesRegressor, and LGBMRegressor) and optimised their hyperparameters to predict the number of TED Talks views. I've made use of the following kernels to create this notebook:
# * https://www.kaggle.com/rounakbanik/ted-data-analysis
# * https://www.kaggle.com/holfyuen/what-makes-a-popular-ted-talk
# * https://www.kaggle.com/tristanmoser/predicting-a-powerful-idea-a-ted-talk-analysis
# 
# I may have used some of the other available notebooks and forgot to add them here, I apologise in advance if that's the case. Please don't shy away from providing feedback and making suggestions on how to improve the accuracy of the models. At the end of the notebook I've listed my plans for future work, feel free to make suggestions on what I should include there.
# 
# I start off by loading the TED Talks dataset and libraries which will be needed to analyse the dataset and to build a model to predict the views of the talks. I will load the dataset first, then check if the data has any null values, and pick out the parameters that I'll use for building the machine learning models.

# In[ ]:


# input data files are available in the "../input/" directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# load libraries
import time
import warnings
import random
import pandas as pd
import datetime
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import metrics
from mlxtend.regressor import StackingCVRegressor

# ignore warnings
warnings.filterwarnings('ignore')

# load data
data = pd.read_csv("../input/ted-talks/ted_main.csv")
data.shape


# In[ ]:


data.head()


# To perform well on predicting TED Talks video views, I try to use as much features from the dataset as possible. Nevertheless, I have decided not to use some of the parameters (e.g., url, main speaker name, etc.) because they won't be much useful in predicting the views. Using some of the features (such as description, tags) I've left for further work.
# - **duration** - duration of the video
# - **event** - name of the event of which the talk is part of
# - **languages** - number of languages in which the talk is available in
# - **num_speaker** - number of speakers in the talk
# - **film_date**, **published_date** - date of filming and publishing the talk, from which I extract:
#   - **day of the week**
#   - **month**
#   - **year**
# - **related-talks** - an array that consists of 6 related talks, from which I extract the average number of views
# 
# I've excluded the **comments** and **ratings** features, as using those I consider cheating. The point of the task is to predict the number of views for a video which has just been released or is yet to be released. After going through the data analysis notebooks I mentined earlier, I decided to exclude the following features:
# - **comments** - number of comments on the video
# - **ratings** - number of times the video has been rated
# - **name** - name of the talk, which includes the name main speaker and title of the talk
# - **main speaker** - name of the main speaker that leads the talk, we rarely see the same speaker do more than 1 talk 
# - **title** - title of the talk
# - **url** - url link to the talk
# 
# The following features I leave for future work:
# - **description** - description of the talk, will need to encode this information
# - **tags** - tags that are associated with the talk
# - **speaker_occupation** - occupation of the main speaker

# # TED Talks Data Analysis
# ## Cleaning up the data
# Various datasets frequently have missing values, so I start off by checking whether the TED Talks dataset has any. 

# In[ ]:


pd.isnull(data).sum()


# There are only 6 null values in the **speaker_occupation** feature, I will fill in those missing values with a default 'Other' value.

# In[ ]:


for index, row in data.iterrows():
    if pd.isnull(row['speaker_occupation']):
        data['speaker_occupation'][index] = 'Other'


# ## related_talks
# Here I print out the **related_talks** feature so I check what it looks like.

# In[ ]:


data['related_talks'][0]


# After that I split the string by its commas and then by the semi-column to get the middle element, which is the views of all related talks.

# In[ ]:


data['related_views'] = 0

for index, row in data.iterrows():
    vids = row['related_talks'].split(',')
    counter = 0
    total = 0
    for views in vids:
        if 'viewed_count' in views:
            view = views.split(':')
            # get rid of brackets and spaces
            view[1] = view[1].replace("]", "")
            view[1] = view[1].replace(" ", "")
            view[1] = view[1].replace("}", "")
            total+=int(view[1])
            counter+=1
    data['related_views'][index] = total/counter


# ## published_date, filmed_date
# From these two features I extract day of the week, month, and year.

# In[ ]:


data['published_date'] = data['published_date'].apply(lambda x: datetime.date.fromtimestamp(int(x)))
data['day'] = data['published_date'].apply(lambda x: x.weekday())
data['month'] = data['published_date'].apply(lambda x: x.month)
data['year'] = data['published_date'].apply(lambda x: x.year)
data['film_date'] = data['film_date'].apply(lambda x: datetime.date.fromtimestamp(int(x)))
data['day_film'] = data['film_date'].apply(lambda x: x.weekday())
data['month_film'] = data['film_date'].apply(lambda x: x.month)
data['year_film'] = data['film_date'].apply(lambda x: x.year)


# Here I categorise the data which is preferable over using numbers.

# In[ ]:


to_cat = {"day":   {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thurday", 4: "Friday", 5: "Saturday",
                    6: "Sunday" },
          "month": {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August", 
                    9: "September", 10: "October", 11: "November", 12: "December"},
          "year":  {2006: "2006", 2007: "2007", 2008: "2008", 2009: "2009", 2010: "2010", 2011: "2011", 2012: "2012", 
                    2013: "2013", 2014: "2014", 2015: "2015", 2016: "2016", 2017: "2017"},
          "day_film":   {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thurday", 4: "Friday", 5: "Saturday",
                    6: "Sunday" },
          "month_film": {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August", 
                    9: "September", 10: "October", 11: "November", 12: "December"},
          "year_film":  {2006: "2006", 2007: "2007", 2008: "2008", 2009: "2009", 2010: "2010", 2011: "2011", 2012: "2012", 
                    2013: "2013", 2014: "2014", 2015: "2015", 2016: "2016", 2017: "2017"}}

data.replace(to_cat, inplace=True)


# ## event
# I check the number of unique event names then list all of them:

# In[ ]:


print('Number of unique events: ',data['event'].unique().shape[0])
data['event'].unique()


# The dataset has 355 unique event names but from the looks of it, lots of these names can be categorised together as they are quite similar. I break down the event names in the following 11 categories (each consisting of at least 5 samples).

# In[ ]:


# initialise all values as 'Other' to assign this category
# to all entries that don't fit into the chosen categories
data['event_category'] = 'Other'

for i in range(len(data)):
    if data['event'][i][0:5]=='TED20':
        data['event_category'][i] = 'TED2000s'
    elif data['event'][i][0:5]=='TED19':
        data['event_category'][i] = 'TED1900s'
    elif data['event'][i][0:4]=='TEDx':
        data['event_category'][i] = "TEDx"
    elif data['event'][i][0:7]=='TED@BCG':
        data['event_category'][i] = 'TED@BCG'
    elif data['event'][i][0:4]=='TED@':
        data['event_category'][i] = "TED@"
    elif data['event'][i][0:8]=='TEDSalon':
        data['event_category'][i] = "TEDSalon"
    elif data['event'][i][0:9]=='TEDGlobal':
        data['event_category'][i] = 'TEDGlobal'
    elif data['event'][i][0:8]=='TEDWomen':
        data['event_category'][i] = 'TEDWomen'
    elif data['event'][i][0:6]=='TEDMED':
        data['event_category'][i] = 'TEDMED'
    elif data['event'][i][0:3]=='TED':
        data['event_category'][i] = 'TEDOther'


# I check whether each categoies can be found in the dataset.

# In[ ]:


data['event_category'].unique()


# ## tags
# Using KeyedVectors to encode and categorise the information from the provided tags.

# In[ ]:


import ast
destring = []
for number in range(len(data)):
    #Remove string
    destring.append(ast.literal_eval(data['tags'][number]))
data['Tags'] = pd.Series(destring)


# In[ ]:


from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format("../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin", binary=True)


# In[ ]:


listed = [item for sublist in destring for item in sublist]
listed = pd.Series(listed)
lists = list(listed.unique())
lists2 = [ x for x in lists if " " not in x ]
lists2 = [ x for x in lists2 if "-" not in x ]


# In[ ]:


lists2.remove('archaeology')
lists2.remove('TEDYouth')
lists2.remove('deextinction')
lists2.remove('blockchain')
lists2.remove('TEDNYC')


# In[ ]:


from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

labels = []
tokens = []

for word in lists2:
    tokens.append(model[word])
    labels.append(word)

tsne_model = TSNE(perplexity=50, n_components=2, init='pca', n_iter=105000, random_state=17,learning_rate=5500)
new_values = tsne_model.fit_transform(tokens)

kmeans = KMeans(n_clusters=15,n_init=200)
kmeans.fit(tokens)
clusters = kmeans.predict(tokens)

df_tsne = pd.DataFrame(new_values, columns=['1st_Comp', '2nd_Comp'])
df_tsne['Cluster'] = clusters

sns.lmplot(x='1st_Comp', y='2nd_Comp', data=df_tsne, hue='Cluster', fit_reg=False)
plt.title("Tag Clusters")


# In[ ]:


convert = {labels[word]: clusters[word] for word in range(len(labels))}


# In[ ]:


comp = pd.DataFrame(labels)
comp['cluster'] = clusters


# In[ ]:


comp_conver = {0:'Organizing/Perceiving Information',1:'animals/organisms',2:'exploration',3:'Scientific Fields',
              4:'media/entertainment',5:'arts/creativity',6:'Epidemics',7:'Humanity/Progress',8:'Vices/Prejudices',
              9:'robots/prosthetics',10:'music',11:'philanthropy/religion',12:'Middle East',13:'Global issues',
              14:'Outer-Space',15:'NA'}


# In[ ]:


comp['group'] = 'None'
for ii in range(len(comp)):
    comp['group'][ii] = comp_conver[comp['cluster'][ii]]
    
unique = comp['group'].unique()


# In[ ]:


for group in unique:
    data[group+'_tag'] = 0
    for item in range(len(data['Tags'])):
        for ii in data['Tags'][item]:
            try:
                clust = convert[ii]
            except KeyError:
                clust = 15
            grouping = comp_conver[clust]
            if grouping == group:
                data[group+'_tag'][item] = 1


# In[ ]:


data.filter(like='_tag', axis=1).head()


# ## Final touches on the dataset
# I take out the views and comments and the rest of the features I won't be using.

# In[ ]:


views = data['views']
comments = data['comments']
data = data.drop(['comments', 'description', 'event', 'film_date', 'main_speaker', 'name', 'published_date', 'ratings', 
           'related_talks', 'tags', 'title', 'url', 'views', 'speaker_occupation', 'Tags'], 1)
data.head()


# I apply **One-Hot-Encoding** on the categorical attributes and get the data ready for training machine learning models. Then I print out the dimensions of the final dataset.

# In[ ]:


# data2 = data.filter(like='_tag', axis=1)
# data = data.drop(data2.columns, 1)
# data.head()
data_final = pd.get_dummies(data)
data_final.shape


# # Machine Learning
# ## Random Forest
# First model I will be testing with is **Random Forest** as this is the one I'm most comfortable with. I will then explore some other ML models, optimise the hyperparameters of each model, and combine those into an ensebmle model.
# 
# I split the dataset in training (90%) and test (10%) sets. The test data will be later used to validate the ML models on unseen data. I start off with a **Random Forest** as it's quite a powerful model that can be used as a baseline. I will use **Mean Absolute Error (MAE)** to measure the error as it will give us a more intuitive understanding of how accurate the model is. Additionally, using **Mean Squared Error (MSE)** to predict target variables with large values (such as the TED Talks views I'm working with) can lead to problems.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data_final, views, test_size=0.1, random_state=121212)


# The baseline Random Forest model seems to have overfitted. Together with the **MAE** of the model I also print the **mean** and **std** of the target variables. Judging by the high variance of the data, it's safe to conclude that the model is performing reasonably well.

# In[ ]:


rf = RandomForestRegressor(criterion='mae',max_depth=15, max_features=45, n_estimators=500, min_samples_leaf=2, min_samples_split=2,
                           random_state=2019)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)
print('Training MAE: {:0.2f}'.format(metrics.mean_absolute_error(y_train, y_pred)))
print('Test MAE: {:0.2f}'.format(metrics.mean_absolute_error(y_test, y_test_pred)))
print('Views mean: {:0.2f}'.format(views.mean()))
print('Views std: {:0.2f}'.format(views.std()))


# I plot the feature importances that are derived from the model that was just trained. I will use this in the future to exclude the unimportant features with the hope of boosting performance.

# In[ ]:


importances = pd.DataFrame({'Features': X_train.columns, 
                                'Importances': rf.feature_importances_})
    
importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)
fig = plt.figure(figsize=(14, 4))
sns.barplot(x='Features', y='Importances', data=importances)
plt.xticks(rotation='vertical')
plt.show()


# ## XGBRegressor
# Good accuracy, the model tends to overfit quite easily when **n_estimators > 20**. Maybe it's worth exploring wether we can use a higher **n_estimators** value while using the other hyperparameters to reguralise the model.

# In[ ]:


xgbr = xgb.XGBRegressor(criterion='mae', earning_rate=0.1, max_depth=10, subsample=0.5, n_estimators=20, min_child_weight=2, random_state=2019)
xgbr.fit(X_train, y_train)
y_pred = xgbr.predict(X_train)
y_test_pred = xgbr.predict(X_test)
print('Training MAE: {:0.2f}'.format(metrics.mean_absolute_error(y_train, y_pred)))
print('Test MAE: {:0.2f}'.format(metrics.mean_absolute_error(y_test, y_test_pred)))


# ## ExtraTreesRegressor
# ExtraTreesRegressor yields the best accuracy, maybe can reguralise the model better as the gap between Training and Test MAE is quite big?

# In[ ]:


et = ExtraTreesRegressor(criterion='mae', max_depth=30, n_estimators=1000, random_state=2019, min_samples_leaf=2, min_samples_split=6)
et.fit(X_train, y_train)
y_pred = et.predict(X_train)
y_test_pred = et.predict(X_test)
print('Training MAE: {:0.2f}'.format(metrics.mean_absolute_error(y_train, y_pred)))
print('Test MAE: {:0.2f}'.format(metrics.mean_absolute_error(y_test, y_test_pred)))


# ## LGBMRegressor
# Couldn't find other hypermarameters except for **max_depth** and **n_estimators** which improve the model's accuracy, more work to be done here..

# In[ ]:


lgbm = lgb.LGBMRegressor(max_depth=5, n_estimators=50, random_state=2019)
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_train)
y_test_pred = lgbm.predict(X_test)
print('Training MAE: {:0.2f}'.format(metrics.mean_absolute_error(y_train, y_pred)))
print('Test MAE: {:0.2f}'.format(metrics.mean_absolute_error(y_test, y_test_pred)))


# ## StackingCVRegressor
# Will implement this later..

# In[ ]:


# stacking_regressor = StackingCVRegressor(regressors=[xgbr,
#                                             rf,
#                                             et,
#                                             lgbm],
#                                            cv=3,
#                             use_features_in_secondary=True,
#                             verbose=2,
#                             random_state=0,
#                             n_jobs=-1,
#                             meta_regressor=rf)


# In[ ]:


# stacking_regressor.fit(X_train, y_train)


# In[ ]:


# y_pred = stacking_regressor.predict(X_train)
# y_test_pred = stacking_regressor.predict(X_test)
# print('Training MAE: {:0.2f}'.format(metrics.mean_absolute_error(y_train, y_pred)))
# print('Test MAE: {:0.2f}'.format(metrics.mean_absolute_error(y_test, y_test_pred)))


# # Conclusion
# Among the ML models I experimented with, ExtraTreesRegressor returns the best **Test MAE** of 597858.20. It seems feasible to decrease the MAE down to 550000 or even further if I experiment more with the model hyperparameters and features.

# # Future Work
# - Improve feature engineering
# - Remove unimportant and correlated features
# - Normalise the data
# - Improve the hyperparameters of the models
# - Use PCA
# 

# # Any Suggestions?
# Should you have any ideas or suggestions on how to improve this notebook (do some actual data analysis and feature engineering, improve the accuracy of the models, or anything else), please don't hesitate to give me a shout in the comments!
