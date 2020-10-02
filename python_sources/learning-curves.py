#!/usr/bin/env python
# coding: utf-8

# # Building Lerning curves for Logit

# This notebook doesn't beat the second baseline for the in-class competition. The idea here is to try use lerning curves for model selecting and to see how adding new features improves model's performance . This notebook consists of:
# * 1) Data loading and transforming
# * 2) EDA and feature creation 
# * 3) LogisticRegression and it's performance inspection
# * 4) Adding polynomial features
# * 5) Conclusion

# ## Data loading and transformation

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import pandas as pd
import numpy as np
import random
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit, TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack
import eli5


# In[ ]:


times = ['time%s' % i for i in range(1, 11)]
train_df = pd.read_csv('../input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/train_sessions.csv',
                       index_col='session_id', parse_dates=times)
test_df = pd.read_csv('../input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/test_sessions.csv',
                      index_col='session_id', parse_dates=times)


# In[ ]:


train_df = train_df.sort_values(by='time1')
train_df.head()


# We can see, that all sessions have different length, there are codded names for sites and the time when user navigated to each site. We will transform our data in the following way:
# * 1) All sites we will have in one text feature, so we will be able to treat them as text feature
# * 2) We will save times when session begins and ends
# * 3) We will create a session_len feature

# In[ ]:


# Here we make a new DataFrame and store sites in one feature
sites = ['site%s' % i for i in range(1, 11)]
train_df[sites] = train_df[sites].fillna(0).astype(np.uint16).astype(str)
model_df = train_df[sites].apply(lambda x: " ".join(x), axis=1)


# In[ ]:


# Reset index, delete 0 from the string and change type to np.str_
model_df = model_df.reset_index(name='sites').set_index('session_id')
model_df = model_df.replace({'sites':r'( 0)*'},{'sites':''}, regex=True)
model_df['sites'] = model_df['sites'].apply(lambda x: np.str_(x))


# In[ ]:


# Here we add new features: start, end and the length of the session
model_df['target'] = train_df['target']
model_df['start'] = train_df[times].min(axis=1)
model_df['end'] = train_df[times].max(axis=1)
model_df['session_len'] = (model_df['end'] - model_df['start']) / np.timedelta64(1, 's')


# In[ ]:


model_df.head()


# Here we can see, that the dataset is unbalanced 

# In[ ]:


model_df.groupby('target')['sites'].count()


# ## EDA part

# Let's explore our dataset and try to find any patterns in the Alice's behavior

# In[ ]:


sns.boxplot(x='target', y='session_len', data = model_df)


# We can see here, that in average Alice spend less time for session, so the session_len can help us to detect Alice. Next we check sites and find the most popular

# In[ ]:


vec1 = CountVectorizer(ngram_range=(1,3), max_features=100000)


# In[ ]:


def freq_sites(vectorizer, data):
    X = vectorizer.fit_transform(data)
    freqs = zip(vectorizer.get_feature_names(), np.asarray(X.sum(axis=0)).ravel())
    return sorted(freqs, key = lambda x: x[1], reverse=True)[:10]


# In[ ]:


# First column for Alice, second - not.
l = [freq_sites(vec1, model_df[model_df['target']==1]['sites']),
     freq_sites(vec1, model_df[model_df['target']==0]['sites'])]
list(map(list, zip(*l)))


# We can see here a significant difference in favorite sites for  Alice and not-Alice, so this feature will be really useful. We also can use TdIdf vectorizer to decrease the impact of common sites and then compare the results. Let's add more features: hour, day_of_week and month

# In[ ]:


model_df['day_of_week'] = model_df['start'].apply(lambda x: x.dayofweek).astype(int)
model_df['hour'] = model_df['start'].apply(lambda x: x.hour).astype(int)
model_df['month'] = model_df['start'].apply(lambda x: x.month).astype(int)


# In[ ]:


model_df['day'] = model_df['start'].apply(lambda x: x.day).astype(int)


# And now we visualize them

# In[ ]:


sns.countplot(x='day_of_week', hue='target', data=model_df[model_df['target']==0])


# In[ ]:


sns.countplot(x='day_of_week', hue='target', data=model_df[model_df['target']==1])


# Here we can see that Alice has some kind of pattern: she has almost no sessions on 2,5,6 days. Maybe we should use it as feature

# In[ ]:


model_df['active_days'] = model_df['day_of_week'].apply(lambda x: x in [0,1,3,4]).astype(int)


# In[ ]:


sns.countplot(x='hour', data=model_df[model_df['target']==0])


# In[ ]:


sns.countplot(x='hour', data=model_df[model_df['target']==1])


# Well here again there is a pattern. Alice's sessions are from 9 till 18 hours, and the most active hours are 12,13,16,17,18.  We definetelly should take it as feature

# In[ ]:


model_df['active_hours']= model_df['hour'].apply(lambda x: x in [12,13,16,17,18]).astype(int)


# In[ ]:


sns.countplot(x='day', data=model_df[model_df['target']==0])


# In[ ]:


sns.countplot(x='day', data=model_df[model_df['target']==1])


# No special info here, but let's keep this feature

# In[ ]:


sns.countplot(x='month', data=model_df[model_df['target']==0])


# In[ ]:


sns.countplot(x='month', data=model_df[model_df['target']==1])


# Here we have active months, let's add them to the features

# In[ ]:


model_df['active_month']= model_df['month'].apply(lambda x: x in [1,2,3,4,9,11,12]).astype(int)


# ## LogisticRegression and useful functions

# Here we initialize our model, scaler, TfIdf vectorizer, also write useful functions to not repeat the same code again

# In[ ]:


log_reg = LogisticRegression(random_state=17, solver='liblinear')
scaler = StandardScaler()
vec2=TfidfVectorizer(ngram_range=(1,3), max_features=100000)


# In[ ]:


cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
time_split = TimeSeriesSplit(n_splits=10)


# In[ ]:


# Function for printing metrics for our predicted result
def print_report(model,train_x, train_y, test_x, test_y):
    pred_y = model.predict(test_x)
    report = metrics.classification_report(test_y, pred_y)
    print(report)
    cv_scores = cross_val_score(model, train_x, train_y, cv=time_split, scoring='roc_auc', n_jobs=4)
    print("cv_scores (train)",cv_scores)
    print("cv_scores (train) mean: {:0.3f}".format(cv_scores.mean()))
    print("roc_auc_score (test):  {:0.3f}".format(roc_auc_score(test_y, pred_y)))


# In[ ]:


# This function helps us not to repeat the same lines of code many times.
def model_cycle(model, train_x, test_x, train_y, test_y):
    model.fit(train_x,train_y)
    print_report(model, train_x, train_y, test_x , test_y)
    return model


# In[ ]:


def data_preparing(train_df, test_df, vec, feature_names):
    train_vec = vec.fit_transform(train_df['sites'])
    test_vec = vec.transform(test_df['sites'])
    if feature_names!=[]:
        scaled_features_train = scaler.fit_transform(train_df[feature_names])
        joined_train = hstack([train_vec, scaled_features_train])
        scaled_features_test = scaler.transform(test_df[feature_names])
        joined_test = hstack([test_vec, scaled_features_test])
        return (joined_train, joined_test)
    return (train_vec, test_vec)    


# In[ ]:


def plot_learning_curves(model, train_x, train_y,  cv, n_jobs=4):
    #train_vec = vectorizer.fit_transform(train_x)
    train_sizes, train_scores, valid_scores = learning_curve(model, train_x, train_y,
                                                             train_sizes=np.linspace(.1, 1.0, 5), scoring='roc_auc', cv=cv, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    # Plot learning curve 
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, valid_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.ylim(0.9,1)
    plt.legend(loc="best")
    plt.xlabel("training_examples")
    plt.ylabel("Score")
    plt.show()    


# In[ ]:


# Here we split our data for the train and test sets
train_x, test_x, train_y, test_y = train_test_split(model_df, model_df['target'], test_size=0.33, random_state=17)


# ## Model performance and learning curves

# Here our first attempt to train model. We use only sites here, no additional information

# In[ ]:


train_temp, test_temp = data_preparing(train_x, test_x, vec1, [])
model = model_cycle(log_reg, train_temp, test_temp, train_y, test_y )
plot_learning_curves(log_reg, train_temp, train_y, cv,4)


# In[ ]:


eli5.show_weights(model, feature_names=vec1.get_feature_names())


# Here we can see low bias on training set and high variance between train and validation scores. We also have the test score, that differs significantly. What also concerns here - low recall of Alice's sessions, which means that our model misses most sessions of Alice. Next we will try TfidfVectorizer for converting sites to features and compare results.

# In[ ]:


train_temp, test_temp = data_preparing(train_x, test_x, vec2, [])
model = model_cycle(log_reg, train_temp, test_temp, train_y, test_y )
plot_learning_curves(log_reg, train_temp, train_y,  cv,4)


# In[ ]:


eli5.show_weights(model, feature_names=vec2.get_feature_names())


# Here the results are even worse, and we have the classic problem here, when comparing different models: first one has lower bias and higher variance and the second - vice-versa. Let's add more features and look at the results. We will add only the part of features, that we have created, just mearures without attempts to detect pattern 

# In[ ]:


train_temp, test_temp = data_preparing(train_x, test_x, vec1, ['hour', 'session_len', 'day_of_week','day','month'])
model = model_cycle(log_reg, train_temp, test_temp, train_y, test_y )
plot_learning_curves(log_reg, train_temp, train_y,  cv,4)


# In[ ]:


eli5.show_weights(model, feature_names=vec1.get_feature_names()+['hour', 'session_len', 'day_of_week','day','month'])


# We can that result improves significanlty, 0.676 vs 0.629 on test and the variance is smaler,recall is also goes up, so we're on the right way. Now we check our second vectorizer

# In[ ]:


train_temp, test_temp = data_preparing(train_x, test_x, vec2, ['hour', 'session_len', 'day_of_week', 'day','month'])
model = model_cycle(log_reg, train_temp, test_temp, train_y, test_y )
plot_learning_curves(log_reg, train_temp, train_y,  cv,4)


# In[ ]:


eli5.show_weights(model, feature_names=vec2.get_feature_names()+['hour', 'session_len', 'day_of_week','day','month'])


# Here our model also shows better scores 0.558 vs 0.529, but it's bias and variance remains the same. We can see, that it's learning curve looks different and shows us the potential for improvement. Now we add our special features, that describe Alise's pattern

# In[ ]:


train_temp, test_temp = data_preparing(train_x, test_x, vec1, ['hour', 'session_len', 'day_of_week','day','month','active_hours','active_days','active_month'])
model = model_cycle(log_reg, train_temp, test_temp, train_y, test_y )
plot_learning_curves(log_reg, train_temp, train_y,  cv,4)


# In[ ]:


eli5.show_weights(model, feature_names=vec1.get_feature_names()+['hour', 'session_len', 'day_of_week','day','month','active_hours','active_days','active_month'])


# More improvement: variance becomes lower and test results and recall goes up: were are able to detect almost half of the Alice's sessions.

# In[ ]:


train_temp, test_temp = data_preparing(train_x, test_x, vec2, ['hour', 'session_len', 'day_of_week','day','month','active_hours','active_days','active_month'])
model = model_cycle(log_reg, train_temp, test_temp, train_y, test_y )
plot_learning_curves(log_reg, train_temp, train_y,  cv,4)


# In[ ]:


eli5.show_weights(model, feature_names=vec2.get_feature_names()+['hour', 'session_len', 'day_of_week','day','month','active_hours','active_days','active_month'])


# Here we can see the improvement too: both bias and variance goes down, but in average it shows worse results than the first one. And the main difference between model remains too: first gas lower bias and higher variance and vice-versa. Now we add polynomial features and look at the result  

# ## Polynomial features

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures


# The degree of polynom should be selected within the cross-validation, but here we will use degree=2 just to show the idea.

# In[ ]:


poly = PolynomialFeatures(2, include_bias=False)


# We will apply polynomial features on all non-sites features

# In[ ]:


train_temp, test_temp = data_preparing(train_x, test_x, vec1, [])
poly_feat = poly.fit_transform(train_x[['hour', 'session_len', 'day_of_week','day','month','active_hours','active_days','active_month']])
poly_scaled = scaler.fit_transform(poly_feat)
train_poly = hstack([train_temp, poly_scaled])
poly_feat = poly.fit_transform(test_x[['hour', 'session_len', 'day_of_week','day','month','active_hours','active_days','active_month']])
poly_scaled = scaler.fit_transform(poly_feat)
test_poly = hstack([test_temp, poly_scaled])


# And here are the result for first model. The calculation takes some time, please, be patient

# In[ ]:


model = model_cycle(log_reg, train_poly, test_poly, train_y, test_y )
plot_learning_curves(log_reg, train_poly, train_y,  cv,4)


# Here are the results: you can see the improvement again: 0.771 vs 0.739 on test, higher recall 0.54 vs 0.48 and lower variance. Now let's look at the second model.

# In[ ]:


train_temp, test_temp = data_preparing(train_x, test_x, vec2, [])
poly_feat = poly.fit_transform(train_x[['hour', 'session_len', 'day_of_week','day','month','active_hours','active_days','active_month']])
poly_scaled = scaler.fit_transform(poly_feat)
train_poly = hstack([train_temp, poly_scaled])
poly_feat = poly.fit_transform(test_x[['hour', 'session_len', 'day_of_week','day','month','active_hours','active_days','active_month']])
poly_scaled = scaler.fit_transform(poly_feat)
test_poly = hstack([test_temp, poly_scaled])


# In[ ]:


model = model_cycle(log_reg, train_poly, test_poly, train_y, test_y )
plot_learning_curves(log_reg, train_poly, train_y,  cv,4)


# So we can see, that polynomial features also improve our results: 0.67 vs 0.616 on test and lower bias. But still we have have here relatively low recall, that concerns me.

# ## Conclusion

# What do we have in the end of the work? We shoul use as many tools as possible to make the right decision in model selecting. The next steps to do:
# * 1) Select the model. I'd rather take the firt one, beacuse of it's higher recall, but the second one looks like it still has reserve for improvement. If you want to add more features - better to select the second.
# * 2) Tuning of the parameters: regularization C for the logistic regression and degree for polynom. This part is missed here, because calculations take a long time.
# * 3) Transform the test set in the same way, that train if you want to make the submission to the competition (I have this part for you commented)

# That's it, thanks for reading!

# In[ ]:


"""
test_df[sites] = test_df[sites].fillna(0).astype(np.uint16).astype(str)
test_sites_df = test_df[sites].apply(lambda x: " ".join(x), axis=1)
test_sites_df = test_sites_df.reset_index(name='sites').set_index('session_id')
test_sites_df = test_sites_df.replace({'sites':r'( 0)*'},{'sites':''}, regex=True)
test_model_df = pd.DataFrame(index=test_df.index)
#
test_model_df['start'] = test_df[times].min(axis=1)
test_model_df['end'] = test_df[times].max(axis=1)
#
test_model_df['session_len'] = (test_model_df['end'] - test_model_df['start']) / np.timedelta64(1, 's')
test_model_df['sites'] = test_sites_df['sites']
test_model_df['sites'] = test_model_df['sites'].apply(lambda x: np.str_(x))
#
test_model_df['day_of_week'] = test_model_df['start'].apply(lambda x: x.dayofweek).astype(int)
test_model_df['hour']= test_model_df['start'].apply(lambda x: x.hour).astype(int)
test_model_df['active_hours']= test_model_df['hour'].apply(lambda x: x in [12,13,16,17,18]).astype(int)
test_model_df['active_days'] = test_model_df['day_of_week'].apply(lambda x: x in [0,1,3,4]).astype(int)
test_model_df['month']= test_model_df['start'].apply(lambda x: x.month).astype(int)
test_model_df['active_month']= test_model_df['month'].apply(lambda x: x in [1,2,3,4,9,11,12]).astype(int)
test_model_df['day'] = test_model_df['start'].apply(lambda x: x.day).astype(int)
#
test_vec = vec1.transform(test_model_df['sites'])
poly_feat = poly.transform(test_model_df[['hour', 'session_len', 'day_of_week','day','month','active_hours','active_days','active_month']])
poly_scaled = scaler.transform(poly_feat)
final_test = hstack([test_vec, poly_scaled])
"""

