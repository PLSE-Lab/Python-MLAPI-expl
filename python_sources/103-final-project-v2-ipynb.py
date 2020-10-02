#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import numpy as np
import pandas as pd
import csv
import re
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from yellowbrick.text import FreqDistVisualizer
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
import seaborn as sns
import xgboost

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# ## read in the tweet data (since 2017-01-01)

# In[ ]:


tweets = pd.read_csv(r'../input/market-volatility/TrumpTweets.csv') #MB removed dash in market-volatility
tweets['Date']  = pd.to_datetime(tweets['created_at']).dt.date
tweets.head()


# ### cleanup the tweets text

# In[ ]:


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
STOPWORDS.add('tco') #additional manual word removal
STOPWORDS.add('https') #additional manual word removal

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text

tweets['text'] = tweets['text'].apply(clean_text)

#Source: https://stackoverflow.com/questions/58603042/easier-way-to-clean-up-text


# ## filter to only consider tweets with certain topic, in this case, trade.

# In[ ]:


trade_words = r'\A(trade|xi|tariff|billion|vietnam|war|deal|talk|dollar|export|farm|agri\w*|producer|duty|global|china|chinese|market|deal|soy|economy)'
trade_tweets = tweets[tweets['text'].str.contains(trade_words)]
print('total of {} trade-related tweets'.format(trade_tweets.shape[0]))
trade_tweets.head(5)


# ## extract the sentiment of the tweets

# In[ ]:


from textblob import TextBlob 
trade_tweets.loc[:, 'sentiment'] = trade_tweets['text'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
daily_tweets = trade_tweets[['Date', 'sentiment']].groupby(['Date'], as_index=False).mean()


# ## read in the options data and narrow down to only certain commodity options of certain types

# In[ ]:


options_df = pd                           .read_csv(r'/kaggle/input/market-volatility/options.csv')     .query("commodity == 'soybeans'")     .query("ticker == 'CZO'")             .query('volatility > 0')              .query('month_remaining == 2 or month_remaining == 4')       .query("dt > '2016-12-31'")           .drop_duplicates(subset=['dt', 'ticker', 'call', 'month_remaining'], keep = 'first')

options_df.head()


# ## plot these options' vol data against the backdrop of the tweets

# In[ ]:


options_df['Date'] = pd.to_datetime(options_df.dt)
daily_tweets['Date'] = pd.to_datetime(daily_tweets.Date)
fig, [ax1, ax2] = plt.subplots(2, figsize=(16, 8), sharex=True)
sns.lineplot(data=options_df, x='Date', y='volatility', hue = 'month_remaining', ax=ax1)
sns.scatterplot(data=daily_tweets, x='Date', y='sentiment', ax=ax2)
ax1.set(xlim=('2017-01-01', '2019-10-01'))
plt.show()


# ## augment options dataset with tweet context

# In[ ]:


def days_since_last_tweet(row, tweets):
    options_date = row['Date']
    date_row = tweets[tweets.Date < options_date].tail(1)
    return (options_date -  date_row.iloc[0, 0]).days

def last_tweet_sentiment(row, tweets):
    options_date = row['Date']
    date_row = tweets[tweets.Date < options_date].tail(1)
    return date_row.iloc[0, 1]

options_df['days_since_last_tweet'] = None
options_df['last_tweet_sentiment'] = None
options_df['latest_volatility'] = None
options_df['days_since_last_tweet'] = options_df.apply(lambda row: days_since_last_tweet(row, daily_tweets), axis=1)
options_df['last_tweet_sentiment'] = options_df.apply(lambda row: last_tweet_sentiment(row, daily_tweets), axis=1)


# In[ ]:


options_df.head()


# In[ ]:


def latest_volatility(row, options):
    date_row = options[(options.Date < row['Date']) & 
                       (options.call == row['call']) & 
                       (options.month_remaining == row['month_remaining']) & 
                       (options.commodity == row['commodity'])].tail(1)
    return row['volatility'] if date_row.empty else date_row.iloc[0, 4]
 
options_df['latest_volatility'] = options_df.apply(lambda row: latest_volatility(row, options_df), axis=1)


# In[ ]:


sns.pairplot(options_df)


# In[ ]:


sns.scatterplot(data=options_df, x='latest_volatility', y='volatility')


# In[ ]:


fig, axs = plt.subplots(2, figsize=(10, 10))
sns.scatterplot(data=options_df, x='future_open_interest', y='volume', hue='month_remaining', alpha=0.5, ax=axs[0])
sns.scatterplot(data=options_df, x='future_open_interest', y='volatility', hue='month_remaining', alpha=0.5, ax=axs[1])


# ## fit a regression model to predict the market volatility

# In[ ]:


X = options_df[['call', 'month_remaining', 'volume', 
                   'underlying_price', 'future_open_interest', 
                   'tnote_rate', 'days_since_last_tweet', 
                   'last_tweet_sentiment']]
y = options_df['volatility']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    random_state = 42) 


# In[ ]:


def output_accurancy(y_hat, y):
    errors = np.mean(abs(y_hat - y))
    accuracy = round((1-np.mean(errors / y)) * 100, 1) 
    print('Mean Absolute Error: {} with accurancy of {}%.'.format(errors, accuracy))


# In[ ]:


model = LinearRegression()
model.fit(X_train, y_train)

y_train_hat = model.predict(X_train)
y_test_hat = model.predict(X_test)

print('---- Train --- ')
output_accurancy(y_train_hat, y_train)
print('---- Test --- ')
output_accurancy(y_test_hat, y_test)


# In[ ]:


best_depth = 5
model = RandomForestRegressor(max_depth=best_depth)
model.fit(X_train, y_train)


# ## evaluate the model

# In[ ]:


y_train_hat = model.predict(X_train)
y_test_hat = model.predict(X_test)

print('---- Train --- ')
output_accurancy(y_train_hat, y_train)
print('---- Test --- ')
output_accurancy(y_test_hat, y_test)


# In[ ]:


#def output_accurancy2(y_hat, y):
#    errors = np.mean(abs(y_hat - y))
#    accuracy = round((1-np.mean(errors / y)) * 100, 3) 
#    print('Mean Absolute Error: {} with accurancy of {}%.'.format(errors, accuracy))
    
def output_accurancy2(y_hat, y):
    errors = np.mean(abs(y_hat - y))
    accuracy = round((1-np.mean(errors / y)) * 100, 3) 
    return accuracy

acc_train = []
acc_test = []
for i in range(1,50):
    model = RandomForestRegressor(max_depth=i)
    model.fit(X_train, y_train)

    y_train_hat = model.predict(X_train)
    y_test_hat = model.predict(X_test)

    acc_train.append(output_accurancy2(y_train_hat, y_train))
    acc_test.append(output_accurancy2(y_test_hat, y_test))

plt.plot(acc_train,c='r',label='Train')
plt.plot(acc_test,c='b',label='Test')
plt.ylabel('Accuracy')
plt.xlabel('Max Depth')
plt.title('Random Forest Depth Accuracy')
plt.legend()
plt.show()


# In[ ]:


best_depth = 18
model = RandomForestRegressor(max_depth=best_depth)
model.fit(X_train, y_train)

y_train_hat = model.predict(X_train)
y_test_hat = model.predict(X_test)

print('---- Train --- ')
output_accurancy(y_train_hat, y_train)
print('---- Test --- ')
output_accurancy(y_test_hat, y_test)


# In[ ]:


sorted_feature_importance = sorted(zip(model.feature_importances_, X_train.columns), reverse=True)
print (*sorted_feature_importance, sep = "\n")


# > ## fit a regression model on volatility changes (a.k.a. volality of volatility)

# In[ ]:


np.array(sorted_feature_importance)[:,0].astype(float)
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title('important features for predicting volatility')
b = sns.barplot(y=np.array(sorted_feature_importance)[:,1], 
                x=np.array(sorted_feature_importance)[:,0].astype(float))
plt.xticks(rotation=45)

plt.show()


# In[ ]:


options_df['delta_volatility'] = options_df['volatility'] - options_df['latest_volatility']


# In[ ]:


delta_options_df = options_df.query('delta_volatility != 0')
X_delta = delta_options_df[['call', 'month_remaining', 'volume', 
                   'underlying_price', 'future_open_interest', 
                   'tnote_rate', 'days_since_last_tweet', 
                   'last_tweet_sentiment']]
    
y_delta = delta_options_df['delta_volatility']

X_train_delta, X_test_delta, y_train_delta, y_test_delta = train_test_split(X_delta, y_delta, 
                                                    test_size=0.2,
                                                    random_state = 42) 


# In[ ]:


X_train_delta.head()


# In[ ]:


y_train_delta.head()


# In[ ]:


best_depth = 20
model_delta = RandomForestRegressor(max_depth=best_depth)
model_delta.fit(X_train_delta, y_train_delta)


# In[ ]:


fig, [ax1, ax2] = plt.subplots(2, figsize=(16, 8), sharex=True)
sns.lineplot(data=delta_options_df, x='Date', y='delta_volatility', ax=ax1, legend="full", alpha=0.7)
sns.lineplot(x=delta_options_df['Date'], y=model_delta.predict(X_delta), ax=ax1, legend="full", alpha=0.7)
ax1.legend(labels=["actual vol changes","predicted vol changes"])
sns.scatterplot(data=daily_tweets, x='Date', y='sentiment', ax=ax2)
ax1.set_title('volatility delta actual vs predicted')
plt.setp(ax1.get_legend().get_texts(), fontsize=22)  # for legend text
ax1.set(xlim=('2017-01-01', '2019-10-01'))
plt.show()


# In[ ]:


y_train_hat_delta = model_delta.predict(X_train_delta)
y_test_hat_delta = model_delta.predict(X_test_delta)

def output_accurancy(y_hat, y):
    errors = np.mean(abs(y_hat - y))
    print(np.mean(errors/ abs(y)))
    accuracy = round((1-np.mean(errors / abs(y))) * 100, 1) 
    print('Mean Absolute Error: {} with accurancy of {}%.'.format(errors, accuracy))

print('---- Train --- ')
output_accurancy(y_train_hat_delta, y_train_delta)
print('---- Test --- ')
output_accurancy(y_test_hat_delta, y_test_delta)


# In[ ]:


from sklearn import metrics
def output_accurancy2(y_hat, y):
    errors = np.mean(abs(y_hat - y))
    #print(np.mean(errors/ abs(y)))
    accuracy = round((1-np.mean(errors / abs(y))) * 100, 1) 
    #print('Mean Absolute Error: {} with accurancy of {}%.'.format(errors, accuracy))
    return errors

model_delta = RandomForestRegressor(max_depth=10)
model_delta.fit(X_train_delta, y_train_delta)
y_train_hat_delta = model_delta.predict(X_train_delta)
output_accurancy2(y_train_hat_delta, y_train_delta)*100
#df = pd.DataFrame({'Actual': y_train_delta, 'Predicted': y_train_hat_delta})
#df
print('Mean Squared Error:', metrics.mean_squared_error(y_train_delta, y_train_hat_delta))  


# In[ ]:


from sklearn import metrics
def output_accurancy2(y_hat, y):
    #errors = np.mean(abs(y_hat - y))
    #print(np.mean(errors/ abs(y)))
    #accuracy = round((1-np.mean(errors / abs(y))) * 100, 1) 
    #print('Mean Absolute Error: {} with accurancy of {}%.'.format(errors, accuracy))
    return metrics.mean_squared_error(y,y_hat)

mse_train_delta = []
mse_test_delta = []
for i in range(1,50):
    model_delta = RandomForestRegressor(max_depth=i)
    model_delta.fit(X_train_delta, y_train_delta)

    y_train_hat_delta = model_delta.predict(X_train_delta)
    y_test_hat_delta = model_delta.predict(X_test_delta)

    mse_train = output_accurancy2(y_train_hat_delta, y_train_delta)*100
    mse_test = output_accurancy2(y_test_hat_delta, y_test_delta)*100

    mse_train_delta.append(mse_train)
    mse_test_delta.append(mse_test)

plt.plot(mse_train_delta,c='r',label='Train')
plt.plot(mse_test_delta,c='b',label='Test')
plt.ylabel('MSE')
plt.xlabel('Max Depth')
plt.title('Random Forest Regressor Error')
plt.legend()
plt.show()


# In[ ]:


model_delta = RandomForestRegressor(max_depth=20)
model_delta.fit(X_train_delta, y_train_delta)
sorted_feature_importance_delta = sorted(zip(model_delta.feature_importances_, X_train_delta.columns), reverse=True)
print (*sorted_feature_importance_delta, sep = "\n")


# In[ ]:


np.array(sorted_feature_importance_delta)[:,0].astype(float)
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title('important features for predicting changes in volatility')
b = sns.barplot(y=np.array(sorted_feature_importance_delta)[:,1], 
                x=np.array(sorted_feature_importance_delta)[:,0].astype(float))
plt.xticks(rotation=45)
plt.show()


# ## fit a classifier to determine if the volatility is going up or down

# In[ ]:


delta_options_df['delta_volatility_up'] = 1*(delta_options_df['volatility'] > delta_options_df['latest_volatility'])


# In[ ]:


delta_options_df.head()


# In[ ]:


X_direction = delta_options_df[['call', 'month_remaining', 'volume', 
                   'underlying_price', 'future_open_interest', 
                   'tnote_rate', 'days_since_last_tweet', 
                   'last_tweet_sentiment']]
y_direction = delta_options_df['delta_volatility_up']

X_train_direction, X_test_direction, y_train_direction, y_test_direction = train_test_split(X_direction, y_direction, 
                                                    test_size=0.2,
                                                    random_state = 42,
                                                    stratify = y_direction) 


# In[ ]:


best_depth = 8
model_direction = RandomForestClassifier(max_depth=best_depth)
model_direction.fit(X_train_direction, y_train_direction)
y_train_hat_direction = model_direction.predict(X_train_direction)
y_test_hat_direction = model_direction.predict(X_test_direction)


#Perfromance Evaluation
acc_random_forest_training = accuracy_score(y_train_direction, y_train_hat_direction)*100
acc_random_forest_testing = accuracy_score(y_test_direction, y_test_hat_direction)*100

print("Random Forest: Direction Prediction Accuracy, Training Set : {:0.2f}%".format(acc_random_forest_training))
print("Random Forest: Direction Prediction Accuracy, Testing Set :  {:0.2f}%".format(acc_random_forest_testing))


# In[ ]:


sorted_feature_importance_delta = sorted(zip(model_delta.feature_importances_, X_train_delta.columns), reverse=True)

np.array(sorted_feature_importance_delta)[:,0].astype(float)
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title('important features for predicting change direction in volatility')
b = sns.barplot(y=np.array(sorted_feature_importance_delta)[:,1], 
                x=np.array(sorted_feature_importance_delta)[:,0].astype(float))
plt.xticks(rotation=45)
plt.show()


# In[ ]:


#    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
#    LinearSVC(),
#    MultinomialNB(),
#    LogisticRegression(random_state=0),

acc_train_direction = []
acc_test_direction = []
for i in range(1,50):
    model_direction = RandomForestClassifier(max_depth=i)
    model_direction.fit(X_train_direction, y_train_direction)

    y_train_hat_direction = model_direction.predict(X_train_direction)
    y_test_hat_direction = model_direction.predict(X_test_direction)

    acc_random_forest_training = accuracy_score(y_train_direction, y_train_hat_direction)*100
    acc_random_forest_testing = accuracy_score(y_test_direction, y_test_hat_direction)*100

    acc_train_direction.append(acc_random_forest_training)
    acc_test_direction.append(acc_random_forest_testing)
    
plt.plot(acc_train_direction,c='r',label='Train')
plt.plot(acc_test_direction,c='b',label='Test')
plt.ylabel('Accuracy')
plt.xlabel('Max Depth')
plt.title('Random Forest Depth Accuracy')
plt.legend()
plt.show()


# In[ ]:


#plt.hist(X_train_direction)
#X_train_direction
X_train_direction_nonnegative = X_train_direction[(X_train_direction > 0).all(1)]


# In[ ]:


#delete
from sklearn.model_selection import cross_val_score

models = [
    RandomForestClassifier(max_depth=15, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    xgboost.XGBClassifier(),
    LogisticRegression(random_state=0),
]

CV = 20
cv_df = pd.DataFrame(index=range(CV * len(models)))

features = X_train_direction+1
labels = y_train_direction

entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
    
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
          ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']
acc


# In[ ]:


plt.figure(figsize=(8,5))
sns.boxplot(x='model_name', y='accuracy', 
            data=cv_df, 
            color='lightblue', 
            showmeans=True)
plt.title("MEAN ACCURACY (cv = 20)\n", size=14);


# In[ ]:


#import pandas as pd
#TrumpTweets = pd.read_csv("../input/TrumpTweets.csv")
#options = pd.read_csv("../input/options.csv")

