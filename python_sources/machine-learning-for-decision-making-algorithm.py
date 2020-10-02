#!/usr/bin/env python
# coding: utf-8

# # Analyzing Business.json

# Firstly importing all the libraries that we need

# In[ ]:


import gc # garbage collector
import numpy as np # linear algebra
from collections import Counter # for counting commong words
import pandas as pd # data processing, JSON file I/O (e.g. pd.read_json)
import matplotlib.pyplot as plt # visualization
plt.style.use('fivethirtyeight') # use ggplot ploting style
import json
import seaborn as sns # visualization 
from wordcloud import WordCloud, STOPWORDS # this module is for making wordcloud in python
import os
import re # regular expression
import string # for finding punctuation in text
import nltk # preprocessing text
from textblob import TextBlob
# import ploty for visualization
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
from plotly.graph_objs import *
import plotly.tools as tls
import plotly.figure_factory as fig_fact
plotly.tools.set_config_file(world_readable=True, sharing='public')

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# Now that we have imported the libraries, I am going to read the datasets with pandas.

# In[ ]:


df_businesses = pd.read_json('../input/yelp_academic_dataset_business.json', lines=True)


# This is how the Business dataset looks like

# In[ ]:


df_businesses.head()


# In[ ]:


print('In total there are ', df_businesses.isnull().sum().sum(), ' missing values')
print('Missing values on each column:')
df_businesses.apply(lambda x: sum(x.isnull()),axis=0)


# In[ ]:


print("1 for Opened Businesses - 0 for Closed Businesses")
df_businesses['is_open'].value_counts()


# In[ ]:


print('Let\'s analyze the ratings of businesses')
df_businesses.stars.describe()


# In[ ]:


plt.hist(df_businesses.stars, bins=np.linspace(1,5,10))
plt.xlabel('Stars')
plt.ylabel('# of Businesses')
plt.title('Businesses by Stars')


# Since there are many different Business on this dataset, we only want to work with Restaurants.
# Lets filter them out

# In[ ]:


print('All businesses: ',df_businesses.shape)
df_businesses.fillna('NA', inplace=True)
df_businesses = df_businesses[df_businesses['categories'].str.contains('Restaurants')]
print('After we have filtered only Restaurants: ',df_businesses.shape)


# There are a total of 59371 Restaurants. I also checked for the empty values and I could have removed them but I filled them inplace with NA. Lets see them

# In[ ]:


df_businesses[df_businesses['attributes'].str.contains('NA') | df_businesses['categories'].str.contains('NA') | df_businesses['hours'].str.contains('NA')]


# Lets see how the filtered dataset with only Restaurnats looks like

# In[ ]:


df_businesses


# In[ ]:


print("1 for Opened Restaurants - 0 for Closed Restaurants")
df_businesses['is_open'].value_counts()


# In[ ]:


print('Let\'s analyze the ratings of Restaurants')
df_businesses.stars.describe()


# Lets see how many different states are there

# In[ ]:


print('There are a total of {} states where there are Restaurants'.format(len(df_businesses['state'].unique())))
df_businesses['state'].unique()


# Check the number of Restaurants on each state

# In[ ]:


df_businesses['state'].value_counts()


# In[ ]:


df_states = df_businesses.groupby('state').count()
df_top_states = df_states['name']
df_top_states_sorted = df_top_states.sort_values(ascending = False)
df_top_states_sorted[:20].plot(kind = 'bar')


# Lets see how many different cities are there

# In[ ]:


print('There are a total of {} cities where there are Restaurants'.format(len(df_businesses['city'].unique())))
df_businesses['city'].unique()


# Check the number of Restaurants on each city

# In[ ]:


df_businesses['city'].value_counts()


# In[ ]:


df_cities = df_businesses.groupby('city').count()
df_top_cities = df_cities['name']
df_top_cities_sorted = df_top_cities.sort_values(ascending = False)
df_top_cities_sorted[:20].plot(kind = 'bar')


# Now lets see the reviews and ratings for Restaurants

# In[ ]:


plt.hist(df_businesses.review_count, bins=range(0,200,5))
plt.xlabel('Review Count')
plt.ylabel('# of Restaurants')
plt.title('Restaurants by Review Count')


# In[ ]:


plt.hist(df_businesses.stars, bins=np.linspace(1,5,10))
plt.xlabel('Stars')
plt.ylabel('# of Restaurants')
plt.title('Restaurants by Stars')


# Lets see what we dont need anymore so we can delete them from Memory.

# In[ ]:


import sys

# These are the usual ipython objects, including this one you are creating
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# Get a sorted list of the objects and their sizes
sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


# Lets delete this dataframes df_cities,df_top_cities,df_top_cities_sorted,df_states,df_top_states,df_top_states_sorted because we dont need them anymore

# In[ ]:


del df_cities,df_top_cities,df_top_cities_sorted,df_states,df_top_states,df_top_states_sorted
gc.collect()


# In[ ]:


import sys

# These are the usual ipython objects, including this one you are creating
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# Get a sorted list of the objects and their sizes
sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


# # Analyzing Review.json and Tip.json

# Yelp is a collection of different business in different areas. In Yelp the most popular business is Restaurants business. Yelp has a huge collection of restaurants. I take top 20 most occurrences restaurants and calculate their mean of ratings reviews polarity and plot them and see which is most popular restaurants since it's quite impossible to plot that thousands of business ratings. 

# In[ ]:


df_reviews_dir = pd.read_json('../input/yelp_academic_dataset_review.json', chunksize=100000, lines=True)
df_tips = pd.read_json('../input/yelp_academic_dataset_tip.json', lines=True)


# Because reviews are too big, we will read them in chunks, and make sure we only take reviews for the Restaurants that we filtered earlier. I choose 10 chunks, (larger numbers will give MemoryError).

# In[ ]:


df_reviews = pd.DataFrame()
i=0
for df in df_reviews_dir:
    df = df[df['business_id'].isin(df_businesses['business_id'])]
    df_reviews = pd.concat([df_reviews, df])
    i=i+1
    print(i)
    if i==10: break


# In[ ]:


print('Reviews: ',df_reviews.shape)
print('Tips: ',df_tips.shape)


# Lets check the ratings on reviews

# In[ ]:


df_reviews["stars"].value_counts()


# In[ ]:


plt.hist(df_reviews.stars, bins=np.linspace(1,5,10))
plt.xlabel('Stars')
plt.ylabel('# of Reviews')
plt.title('Reviews by Stars')


# Lets analyze the columns of reviews

# In[ ]:


df_reviews.describe()


# In[ ]:


print('Missing values of Reviews dataset ', df_reviews.isnull().sum().sum())
print('Missing values of Tips dataset ', df_tips.isnull().sum().sum())


# Make sure we only get businesses that are in our review list and delete the rest, since we dont have reviews for them.

# In[ ]:


df_businesses = df_businesses[df_businesses['business_id'].isin(df_reviews['business_id'])]


# In[ ]:


print('Final businesses shape: ', df_businesses.shape)
print('Final review shape: ', df_reviews.shape)


# Since Review dataset doesnt have Business name but only Business id, lets add the name column in Review dataset too, after that I will do a text cleansing and add it to another column called cleared_text. And then I will select 20 Restaurants which have more reviews, and put them in a new dataframe, then I will calculate the mean of Ratings for each Restaurant

# In[ ]:


df_reviews['name'] = df_reviews['business_id'].map(df_businesses.set_index('business_id')['name'])


# Lets also see the Reviews with the added column name

# In[ ]:


def preprocess(text):
    text = re.sub('[^a-z\s]', '', text.lower())                  # get rid of noise
    text = [word for word in text.split() if word not in set(stopwords)]  # remove stopwords
    return ' '.join(text) # then join the text again
# let's find out which stopwords need to remove. We'll use english stopwords.
i = nltk.corpus.stopwords.words('english')
# punctuations to remove
j = list(string.punctuation)
# finally let's combine all of these
stopwords = set(i).union(j)

df_reviews['cleared_text'] = df_reviews['text'].apply(preprocess)


# In[ ]:


df_reviews


# Lets get all reviews for top 20 restaurants and calculate the mean of their rating

# In[ ]:


top_restaurants = df_reviews.name.value_counts().index[:20].tolist()
df_top_reviews = df_reviews.loc[df_reviews['name'].isin(top_restaurants)]
df_top_reviews.groupby(df_top_reviews.name)['stars'].mean().sort_values(ascending=True).plot(kind='barh',figsize=(12, 10))
plt.yticks(fontsize=18)
plt.title('Top rated restaurants on Yelp',fontsize=20)
plt.ylabel('Restaurants names', fontsize=18)
plt.xlabel('Ratings', fontsize=18)
plt.show()


# Lets select One specific Restaurant by business_id and see its reviews

# In[ ]:


specific_restaurant_reviews = 'QXAEGFB4oINsVuTFxEYKFQ'

df_specific_restaurant_reviews = df_reviews[df_reviews['business_id'] == specific_restaurant_reviews]


# In[ ]:


df_specific_restaurant_reviews


# Now lets add sentiment polarity for Restaurants with most Reviews

# In[ ]:


def sentiment(text):
    sentiment = TextBlob(text)
    return sentiment.sentiment.polarity

df_top_reviews['sentiment_polarity'] = df_top_reviews['cleared_text'].apply(sentiment)


# Now lets see the dataset

# In[ ]:


df_top_reviews


# Lets calculate the mean of sentiment polarity for each restaurant

# In[ ]:


df_top_reviews.groupby(df_top_reviews.name)['sentiment_polarity'].mean().sort_values(ascending=True).plot(kind='barh',figsize=(12, 10))
plt.yticks(fontsize=18)
plt.title('Top customers satisfied restaurants on Yelp',fontsize=20)
plt.ylabel('Restaurants names', fontsize=18)
plt.xlabel('Reviews polarity', fontsize=18)
plt.show()


# Lets check the mean of useful, funny and cool of each Restaurant

# In[ ]:


df_top_reviews.groupby(df_top_reviews.name)[['useful','funny', 'cool']].mean().sort_values('useful',ascending=True).plot(kind='barh', figsize=(15, 14),width=0.7)
plt.yticks(fontsize=18)
plt.title('Top useful, funny and cool restaurants',fontsize=28)
plt.ylabel('Restaurants names', fontsize=18)
plt.legend(fontsize=22)
plt.show()


# # Analyzing Tip.json
# 
# Since Tip dataset doesnt have Business name but only Business id, lets add the name column in Tip dataset too. And then I will select 20 Restaurants which have more tips, and put them in a new dataframe and then I will do text cleansing for Tips. 

# In[ ]:


df_tips['name'] = df_tips['business_id'].map(df_businesses.set_index('business_id')['name'])
df_top_tips = df_tips.loc[df_tips['name'].isin(top_restaurants)]
df_top_tips['cleared_text'] = df_top_tips['text'].apply(preprocess)


# Lets have a look in the dataset

# In[ ]:


df_top_tips


# Lets see most used words in the tips

# In[ ]:


wc = WordCloud(width=1600, height=800, random_state=1, max_words=200000000)
wc.generate(str(df_top_tips['cleared_text']))
plt.figure(figsize=(20,10), facecolor='k')
plt.title("Tips for top reviewed restaurant", fontsize=40,color='white')
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=10)
plt.show()


# # ANALYZING SOME SPECIFIC RESTAURANTS

# **Lets analyze Little Miss BBQ Restaurant Reviews and Ratings**

# In[ ]:


df_little_miss_bbq_only = df_businesses.loc[df_businesses['name'] == "Little Miss BBQ"]
df_little_miss_bbq_review = df_top_reviews.loc[df_top_reviews['business_id'].isin(df_little_miss_bbq_only.business_id)]


# In[ ]:


wc = WordCloud(width=1600, height=800, random_state=1, max_words=200000000)
wc.generate(str(df_little_miss_bbq_review['cleared_text']))
plt.figure(figsize=(20,10), facecolor='k')
plt.title("Customers reviews about 'Little Miss BBQ'", fontsize=40,color='white')
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=10)
plt.show()


# In[ ]:


# convert date column to pandas datatime 
df_little_miss_bbq_review.date = pd.to_datetime(df_little_miss_bbq_review.date)
df_little_miss_bbq_review.groupby(df_little_miss_bbq_review.date.dt.year)['sentiment_polarity'].mean().plot(kind='bar', figsize=(12, 7))
plt.title("Customer satisfaction of 'Little Miss BBQ' in different years", fontsize=20)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Polarity of reviews(satisfaction)', fontsize=18)
plt.show()


# In[ ]:


df_little_miss_bbq_review.date = pd.to_datetime(df_little_miss_bbq_review.date)
df_little_miss_bbq_review.groupby(df_little_miss_bbq_review.date.dt.year)['stars'].mean().plot(kind='bar', figsize=(12, 7))
plt.title("'Little Miss BBQ' ratings in different years", fontsize=20)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Stars', fontsize=18)
plt.show()


# In[ ]:


df_little_miss_bbq_review.date = pd.to_datetime(df_little_miss_bbq_review.date)
df_little_miss_bbq_review.groupby(df_little_miss_bbq_review.date.dt.year)[['useful','funny','cool']].mean().plot(kind='bar', figsize=(12, 7))
plt.title("'Little Miss BBQ' usefulness, funniness and coolness in different years", fontsize=20)
plt.xlabel('Year', fontsize=18)
plt.show()


# Lets delete the dataframe that we don't need now.

# In[ ]:


del df_little_miss_bbq_only,df_little_miss_bbq_review
gc.collect()


# **Lets analyze The Venetian Las Vegas Restaurant Reviews and Ratings**

# In[ ]:


df_the_venetian_las_vegas_only = df_businesses.loc[df_businesses['name'] == "The Venetian Las Vegas"]
df_the_venetian_las_vegas_review = df_top_reviews.loc[df_top_reviews['business_id'].isin(df_the_venetian_las_vegas_only.business_id)]


# In[ ]:


wc = WordCloud(width=1600, height=800, random_state=1, max_words=200000000)
wc.generate(str(df_the_venetian_las_vegas_review['cleared_text']))
plt.figure(figsize=(20,10), facecolor='k')
plt.title("Customers reviews about 'The Venetian Las Vegas'", fontsize=40,color='white')
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=10)
plt.show()


# In[ ]:


# convert date column to pandas datatime 
df_the_venetian_las_vegas_review.date = pd.to_datetime(df_the_venetian_las_vegas_review.date)
df_the_venetian_las_vegas_review.groupby(df_the_venetian_las_vegas_review.date.dt.year)['sentiment_polarity'].mean().plot(kind='bar', figsize=(12, 7))
plt.title("Customer satisfaction of 'The Venetian Las Vegas' in different years", fontsize=20)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Polarity of reviews(satisfaction)', fontsize=18)
plt.show()


# In[ ]:


df_the_venetian_las_vegas_review.date = pd.to_datetime(df_the_venetian_las_vegas_review.date)
df_the_venetian_las_vegas_review.groupby(df_the_venetian_las_vegas_review.date.dt.year)['stars'].mean().plot(kind='bar', figsize=(12, 7))
plt.title("'The Venetian Las Vegas' ratings in different years", fontsize=20)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Stars', fontsize=18)
plt.show()


# In[ ]:


df_the_venetian_las_vegas_review.date = pd.to_datetime(df_the_venetian_las_vegas_review.date)
df_the_venetian_las_vegas_review.groupby(df_the_venetian_las_vegas_review.date.dt.year)[['useful','funny','cool']].mean().plot(kind='bar', figsize=(12, 7))
plt.title("'The Venetian Las Vegas' usefulness, funniness and coolness in different years", fontsize=20)
plt.xlabel('Year', fontsize=18)
plt.show()


# In[ ]:


del df_the_venetian_las_vegas_only,df_the_venetian_las_vegas_review
gc.collect()


# **Lets analyze Juan's Flaming Fajitas & Cantina Restaurant Reviews and Ratings**

# In[ ]:


df_juans_flaming_fajitas_cantina_only = df_businesses.loc[df_businesses['name'] == "Juan's Flaming Fajitas & Cantina"]
df_juans_flaming_fajitas_cantina_review = df_top_reviews.loc[df_top_reviews['business_id'].isin(df_juans_flaming_fajitas_cantina_only.business_id)]


# In[ ]:


wc = WordCloud(width=1600, height=800, random_state=1, max_words=200000000)
wc.generate(str(df_juans_flaming_fajitas_cantina_review['cleared_text']))
plt.figure(figsize=(20,10), facecolor='k')
plt.title("Customers reviews about 'Juan's Flaming Fajitas & Cantina'", fontsize=40,color='white')
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=10)
plt.show()


# In[ ]:


# convert date column to pandas datatime 
df_juans_flaming_fajitas_cantina_review.date = pd.to_datetime(df_juans_flaming_fajitas_cantina_review.date)
df_juans_flaming_fajitas_cantina_review.groupby(df_juans_flaming_fajitas_cantina_review.date.dt.year)['sentiment_polarity'].mean().plot(kind='bar', figsize=(12, 7))
plt.title("Customer satisfaction of 'Juan's Flaming Fajitas & Cantina' in different years", fontsize=20)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Polarity of reviews(satisfaction)', fontsize=18)
plt.show()


# In[ ]:


df_juans_flaming_fajitas_cantina_review.date = pd.to_datetime(df_juans_flaming_fajitas_cantina_review.date)
df_juans_flaming_fajitas_cantina_review.groupby(df_juans_flaming_fajitas_cantina_review.date.dt.year)['stars'].mean().plot(kind='bar', figsize=(12, 7))
plt.title("'Juan's Flaming Fajitas & Cantina' ratings in different years", fontsize=20)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Stars', fontsize=18)
plt.show()


# In[ ]:


df_juans_flaming_fajitas_cantina_review.date = pd.to_datetime(df_juans_flaming_fajitas_cantina_review.date)
df_juans_flaming_fajitas_cantina_review.groupby(df_juans_flaming_fajitas_cantina_review.date.dt.year)[['useful','funny','cool']].mean().plot(kind='bar', figsize=(12, 7))
plt.title("'Juan's Flaming Fajitas & Cantina' usefulness, funniness and coolness in different years", fontsize=20)
plt.xlabel('Year', fontsize=18)
plt.show()


# In[ ]:


del df_juans_flaming_fajitas_cantina_only,df_juans_flaming_fajitas_cantina_review,df_top_reviews,df_tips,df_top_tips
gc.collect()


# Lets see what we dont need anymore so we can delete them from Memory.

# In[ ]:


import sys

# These are the usual ipython objects, including this one you are creating
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# Get a sorted list of the objects and their sizes
sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


# In[ ]:


del df,df_reviews_dir,df_specific_restaurant_reviews,top_restaurants, specific_restaurant_reviews
gc.collect()


# In[ ]:


import sys

# These are the usual ipython objects, including this one you are creating
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# Get a sorted list of the objects and their sizes
sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


# # Predicting Ratings from Review text

# 1. ** Linear SVM Algorithm**

# Importing the libraries

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import itertools


# Vectorizer - breaks text into single words and bi-grams and then calculates the TF-IDF representation. The 'fit' builds up the vocabulary from all the reviews and the 'transform' turns each indivdual text into a matrix of numbers.

# In[ ]:


vectorizer = TfidfVectorizer(ngram_range=(1,3))

vectors = vectorizer.fit_transform(df_reviews['cleared_text'])


# Splitting dataset into Train and test Data where 20% of data is going to be used for testing and 80% for training.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(vectors, df_reviews['stars'], test_size=0.2, random_state=42, shuffle=False)


# Building Linear SVM model and train our classifier

# In[ ]:


from sklearn.svm import LinearSVC

LSVMclassifier = LinearSVC()

LSVMclassifier.fit(X_train, y_train)


# Making predictions

# In[ ]:


LSVMpredictions = LSVMclassifier.predict(X_test)

print("Actual Ratings: ")
print(y_test[:10])
print("Predicted Ratings: ",end = "")
print(LSVMpredictions[:10])


# **Evaluating our classifier**

# Lets calculate the accuracy of our classifier by comparing the predicted ratings and the real ratings, if they are the same, our classifier predicted the ratings correctly. We sum up all of the correct answers and divide by the total number of reviews in our test set

# In[ ]:


print('Accuracy score: ', accuracy_score(y_test, LSVMpredictions))


# Precision and Recall of the Model

# In[ ]:


print ('Precision: ' + str(precision_score(y_test, LSVMpredictions, average='weighted')))
print ('Recall: ' + str(recall_score(y_test, LSVMpredictions, average='weighted')))


# Classification Report of the Model

# In[ ]:


print(classification_report(y_test, LSVMpredictions))


# Confusion Metrics

# In[ ]:


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Greens):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


from sklearn import metrics
names = ['1','2','3','4','5']

confusion_matrix = metrics.confusion_matrix(y_test, LSVMpredictions)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(confusion_matrix, classes=names, title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(confusion_matrix, classes=names, normalize=True, title='Normalized confusion matrix')

plt.show()


# Lets make predictions for the whole dataset using the classifier that we trained 

# In[ ]:


X_null_train, X_full_test, y_null_train, y_full_test = train_test_split(vectors, df_reviews['stars'], test_size=0.999995, random_state=42, shuffle=False)

LSVMfullpredictions = LSVMclassifier.predict(X_full_test)


# In[ ]:


print("Actual Ratings: ")
print(y_full_test[-10:])
print("\nPredicted Ratings: ",end = "")
print(LSVMfullpredictions[-10:])


# Lets save this to a new dataframe called df_LSVM_reviews. Before saving it lets make a copy of a reviews dataset since we need some modifications on the new dataset and we will need the original dataset later again. We create a copy of dataset and drop first 3 rows, since we couldnt predict ratings for the first 3 rows. After the copy is made change the real stars column with the predicted ones.

# In[ ]:


df_LSVM_reviews = df_reviews.copy()

df_LSVM_reviews.drop(df_LSVM_reviews.head(3).index, inplace=True)

df_LSVM_reviews['stars'] = LSVMfullpredictions

df_LSVM_reviews


# **Predicting Positive and Negative sentiments **

# In[ ]:


sentiments = []
for star in df_reviews['stars']:
    if star <= 3:
        sentiments.append('n')
    if star > 3:
        sentiments.append('p')


# Splitting dataset into Train and test Data where 20% of data is going to be used for testing and 80% for training.

# In[ ]:


X2_train, X2_test, y2_train, y2_test = train_test_split(vectors, sentiments, test_size=0.20, random_state=42)


# Building Linear SVM model and train our classifier

# In[ ]:


LSVMclassifier2 = LinearSVC()

LSVMclassifier2.fit(X2_train, y2_train)


# Making predictions

# In[ ]:


LSVMpredictions2 = LSVMclassifier2.predict(X2_test)

print("Actual Rating Category: ")
print(y2_test[:10])
print("\nPredicted Rating Category: ",end = "")
print(list(LSVMpredictions2[:10]))


# Accuracy of the model

# In[ ]:


print('Accuracy score: ', accuracy_score(y2_test, LSVMpredictions2))


# Precision and Recall of the Model

# In[ ]:


print ('Precision: ' + str(precision_score(y2_test, LSVMpredictions2, average='weighted')))
print ('Recall: ' + str(recall_score(y2_test, LSVMpredictions2, average='weighted')))


# Classification Report of the Model

# In[ ]:


print(classification_report(y2_test, LSVMpredictions2))


# Confusion Metrics

# In[ ]:


from sklearn import metrics
names = ['Negative','Positive']

confusion_matrix = metrics.confusion_matrix(y2_test, LSVMpredictions2)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(confusion_matrix, classes=names, title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(confusion_matrix, classes=names, normalize=True, title='Normalized confusion matrix')

plt.show()


# 2. **Naive Bayes Algorithm**

# * Naive Bayes Algorithm using (1,2,3,4 and 5 Ratings)

# Selecting review text and rating from reviews dataset and make a copy of them, since we will vectorize the text, and the original dataset is needed again later.

# In[ ]:


x = df_reviews['text'].copy()
y = df_reviews['stars'].copy()


# In[ ]:


def text_preprocessing(text):
    no_punctuation = [ch for ch in text if ch not in string.punctuation]
    no_punctuation = ''.join(no_punctuation)
    return [w for w in no_punctuation.split() if w.lower() not in stopwords]


# Vectorization - Converting each review into a vector using bag-of-words approach

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

vector = CountVectorizer(analyzer=text_preprocessing).fit(x)

x = vector.transform(x)


# Splitting dataset into Train and test Data where 20% of data is going to be used for testing and 80% for training.

# In[ ]:


X3_train, X3_test, y3_train, y3_test = train_test_split(x, y, test_size=0.20, random_state=0, shuffle=False)


# Building Multinomial Naive Bayes model and train our classifier

# In[ ]:


from sklearn.naive_bayes import MultinomialNB

NBclassifier = MultinomialNB()

NBclassifier.fit(X3_train, y3_train)


# Making predictions

# In[ ]:


NBpredictions = NBclassifier.predict(X3_test)

print("Actual Ratings: ")
print(y3_test[:10])
print("Predicted Ratings: ",end = "")
print(NBpredictions[:10])


# Accuracy of the model

# In[ ]:


print('Accuracy score: ', accuracy_score(y3_test, NBpredictions))


# Precision and Recall of the Model

# In[ ]:


print ('Precision: ' + str(precision_score(y3_test, NBpredictions, average='weighted')))
print ('Recall: ' + str(recall_score(y3_test, NBpredictions, average='weighted')))


# Classification Report of the Model

# In[ ]:


print(classification_report(y3_test, NBpredictions))


# Confusion Metrics

# In[ ]:


from sklearn import metrics
names = ['1','2','3','4','5']

confusion_matrix = metrics.confusion_matrix(y3_test, NBpredictions)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(confusion_matrix, classes=names, title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(confusion_matrix, classes=names, normalize=True, title='Normalized confusion matrix')

plt.show()


# Lets make predictions for the whole dataset using the classifier that we trained 

# In[ ]:


X3_null_train, X3_full_test, y3_null_train, y3_full_test = train_test_split(x, y, test_size=0.999995, random_state=0, shuffle=False)

NBfullpredictions = NBclassifier.predict(X3_full_test)


# In[ ]:


print("Actual Ratings: ")
print(y3_full_test[-10:])
print("\nPredicted Ratings: ",end = "")
print(NBfullpredictions[-10:])


# Lets save this to a new dataframe called df_NB_reviews. Before saving it lets make a copy of a reviews dataset since we need some modifications on the new dataset and we will need the original dataset later again. We create a copy of dataset and drop first 3 rows, since we couldnt predict ratings for the first 3 rows. After the copy is made change the real stars column with the predicted ones.

# In[ ]:


df_NB_reviews = df_reviews.copy()

df_NB_reviews.drop(df_NB_reviews.head(3).index, inplace=True)

df_NB_reviews['stars'] = NBfullpredictions

df_NB_reviews


# * Naive Bayes Classifier using (1 and 5 Rating: Positive & Negative Reviews)

# Lets classify Ratings from 1 to 3 as Negatives and Ratings from 4 to 5 as Positives

# In[ ]:


df_copied_reviews = df_reviews.copy()

df_copied_reviews['stars'][df_copied_reviews.stars == 3] = 1
df_copied_reviews['stars'][df_copied_reviews.stars == 2] = 1
df_copied_reviews['stars'][df_copied_reviews.stars == 4] = 5


# Lets put them in respective variables rating1 and rating5

# In[ ]:


rating1 = df_copied_reviews[df_copied_reviews['stars'] == 1]


# On the analysis section that we saw earlier, there are more positives ratings than negatives. Because of that the dataset is unbalanced. To make undersampling of the dataset lets check how many ratings are Negative and select the same amount for Positive.

# In[ ]:


len(rating1)


# Since there are 216737 Negative ratings from (1 to 3) lets also select only 216737 for Positive (4 and 5)

# In[ ]:


rating5 = df_copied_reviews[df_copied_reviews['stars'] == 5][0:216737]


# In[ ]:


frames = [rating1, rating5]
df_copied_reviews = pd.concat(frames)


# In[ ]:


x2 = df_copied_reviews['text']
y2 = df_copied_reviews['stars']


# Vectorization - Converting each review into a vector using bag-of-words approach

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

vector2 = CountVectorizer(analyzer=text_preprocessing).fit(x2)

x2 = vector.transform(x2)


# Splitting dataset into Train and test Data where 20% of data is going to be used for testing and 80% for training.

# In[ ]:


X4_train, X4_test, y4_train, y4_test = train_test_split(x2, y2, test_size=0.20, random_state=0)


# Building Multinomial Naive Bayes model and train our classifier

# In[ ]:


NBclassifier2 = MultinomialNB()

NBclassifier2.fit(X4_train, y4_train)


# Making predictions

# In[ ]:


NBpredictions2 = NBclassifier2.predict(X4_test)

print("Actual Rating Category: ")
print(y4_test[:10])
print("\nPredicted Rating Category: ",end = "")
print(list(NBpredictions2[:10]))


# Accuracy of the model

# In[ ]:


print('Accuracy score: ', accuracy_score(y4_test, NBpredictions2))


# Precison and Recall of the Model

# In[ ]:


print ('Precision: ' + str(precision_score(y4_test, NBpredictions2, average='weighted')))
print ('Recall: ' + str(recall_score(y4_test, NBpredictions2, average='weighted')))


# Classification Report of the Model

# In[ ]:


print(classification_report(y4_test, NBpredictions2))


# Confusion Metrics

# In[ ]:


from sklearn import metrics
names = ['Negative','Positive']

confusion_matrix = metrics.confusion_matrix(y4_test, NBpredictions2)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(confusion_matrix, classes=names, title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(confusion_matrix, classes=names, normalize=True, title='Normalized confusion matrix')

plt.show()


# Check for df that are on memory and we dont need anymore so we free up some space

# Lets see what we dont need anymore so we can delete them from Memory.

# In[ ]:


import sys

# These are the usual ipython objects, including this one you are creating
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# Get a sorted list of the objects and their sizes
sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


# In[ ]:


del df_copied_reviews,rating1,rating5,sentiments,TfidfVectorizer,CountVectorizer,LinearSVC,MultinomialNB,accuracy_score,precision_score,recall_score,classification_report,confusion_matrix,vectors,vector,vector2,y,y2,y_test,y2_test,y3_test,y4_test,y_full_test,y3_full_test,y_train,y2_train,y3_train,y4_train,y_null_train,y3_null_train,LSVMpredictions,LSVMpredictions2,LSVMfullpredictions,NBpredictions,NBpredictions2,NBfullpredictions,LSVMclassifier,LSVMclassifier2,NBclassifier,NBclassifier2,x,x2,X_test,X2_test,X3_test,X4_test,X_full_test,X3_full_test,X_train,X2_train,X3_train,X4_train,X_null_train,X3_null_train
gc.collect()


# In[ ]:


import sys

# These are the usual ipython objects, including this one you are creating
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# Get a sorted list of the objects and their sizes
sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


# # Recommendation Algorithms

# **1. Collaborative Filtering**

# Importing the libraries

# In[ ]:


from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt


# Prediction Function

# In[ ]:


def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_difference = (ratings - mean_user_rating[:, np.newaxis])
        prediction = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_difference) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        prediction = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return prediction


# Root Mean Square Error Algorithm

# In[ ]:


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


# Mean Absolute Error Algorithm

# In[ ]:


def mae(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return mean_absolute_error(prediction, ground_truth)


# Collaborative Filtering Algorithm

# Due to Memory Error I am using just 4200 reviews for 1,3,4,5 stars rated and 2500 for 2 stars rated 

# In[ ]:


def collaborativeFiltering(dataset):
    reviews_dataframe = dataset.copy()

    print("Starting undersampling of the dataset")
    
    # Undersampling the dataset to get a balanced dataset
    rating1 = reviews_dataframe[reviews_dataframe['stars'] == 1][0:4000]
    rating2 = reviews_dataframe[reviews_dataframe['stars'] == 2][0:2500]
    rating3 = reviews_dataframe[reviews_dataframe['stars'] == 3][0:4000]
    rating4 = reviews_dataframe[reviews_dataframe['stars'] == 4][0:4000]
    rating5 = reviews_dataframe[reviews_dataframe['stars'] == 5][0:4000]
    frames = [rating1, rating2, rating3, rating4, rating5]
    reviews_dataframe = pd.concat(frames)
    
    print("Completed undersampling the dataset")
    
    # converting user_id and business_id to integers for the matrix
    reviews_dataframe['user_id'] = pd.factorize(reviews_dataframe.user_id)[0]
    reviews_dataframe['business_id'] = pd.factorize(reviews_dataframe.business_id)[0]
    
    # getting the number unique users and restaurants
    unique_users = reviews_dataframe.user_id.unique().shape[0]
    unique_restaurants = reviews_dataframe.business_id.unique().shape[0]
    
    train_data, test_data = train_test_split(reviews_dataframe, test_size=0.2)
    
    #Create two user-item matrices, one for training and another for testing
    train_data_matrix = np.zeros((unique_users, unique_restaurants))
    
    print("Starting creation of user-item matrix")
    
    # train_data_matrix
    for line in train_data.itertuples():
         train_data_matrix[line[9], line[1]] = line[6]
            
    # test_data_matrix
    test_data_matrix = np.zeros((unique_users, unique_restaurants))
    for line in test_data.itertuples():
        test_data_matrix[line[9], line[1]] = line[6]
    
    print("Completed creating user-item matrix")
    
    
    print("Starting creation of similarity matrix")
    
    # similarity between users and items
    user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
    item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
    
    print("Completed creating similarity matrix")
    
    
    print("Starting creation of prediction matrix")
    
    item_prediction = predict(train_data_matrix, item_similarity, type='item')
    user_prediction = predict(train_data_matrix, user_similarity, type='user')
    
    print("Completed creating prediction matrix")
    
    
    print('Printing the RMSE and MAE' + '\n')
    
    if dataset.equals(df_reviews):
        rating_type = 'biased rating'
    elif dataset.equals(df_LSVM_reviews):
        rating_type = 'unbiased rating for Linear SVM Dataframe'
    else:
        rating_type = 'unbiased rating for Naive Bayes Dataframe'
    
    print('Root Mean Square Error while testing the model using ' + rating_type)
    print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
    print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)) + '\n')

    print('Root Mean Square Error while training the model using ' + rating_type)
    print('User-based CF RMSE: ' + str(rmse(user_prediction, train_data_matrix)))
    print('Item-based CF RMSE: ' + str(rmse(item_prediction, train_data_matrix)) + '\n')
    
    print('Mean Absolute Error while testing the model using ' + rating_type)
    print('User-based CF MAE: ' + str(mae(user_prediction, test_data_matrix)))
    print('Item-based CF MAE: ' + str(mae(item_prediction, test_data_matrix)) + '\n')

    print('Mean Absolute Error while training the model using ' + rating_type)
    print('User-based CF MAE: ' + str(mae(user_prediction, train_data_matrix)))
    print('Item-based CF MAE: ' + str(mae(item_prediction, train_data_matrix)) + '\n')


# Collaborative filtering for original Reviews dataset

# In[ ]:


collaborativeFiltering(df_reviews)


# Collaborative filtering for Linear SVM Reviews dataset

# In[ ]:


collaborativeFiltering(df_LSVM_reviews)


# Collaborative filtering for Naive Bayes Reviews dataset

# In[ ]:


collaborativeFiltering(df_NB_reviews)


# In[ ]:


del df_LSVM_reviews,df_NB_reviews
gc.collect()

