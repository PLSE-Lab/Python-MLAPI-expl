#!/usr/bin/env python
# coding: utf-8

# The task at hand is to predict Airbnb prices given certain metrics in NYC. 
# 
# **NOTE**: This project is done solely for personal practice. Please let me know where I can improve on or if there are any questions to my thinking. Thank you!

# In[ ]:


#from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from matplotlib import pyplot


# <h3> Obtaining Data

# In[ ]:


df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head()


# In[ ]:


columns = df.columns
columns, len(columns)


# <h3> Exploring Data

# In[ ]:


df.shape


# We have 16 total features and ~49000 samples. Not all of the features are useful and we are also unsure if all samples are valid (i.e. presence of NaN) so we will most likely have to perform feature engineering to obtain more useful data. Let's see if we can get any insights from the data before making any changes to it. <br>
# 
# Note that majority of the features are self-explanatory, but just for clarification:
# 1. **'calculated_host_listings_count'** indicates the total number of apartments and bedrooms referred to the same landlord (one landlord can have more than one property in NY) [1]
# 2. **'availability_365'** indicates the # of days in an year that the airbnb is available. It is possible for this value to be 0 [2]
# 
# [1]https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data/discussion/120300 <br>
# [2]https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data/discussion/111835

# In[ ]:


# checking data types
df.info()


# In[ ]:


# checking # of NaN in each column
if(df.isnull().values.any() == True):
    for name in columns:
        print('%s: %i' %(name, df[name].isnull().sum()) )


# Majority of the NaN are in the last_review and reviews_per_month feature. This makes sense because last_review is a date type and some error most likely occured during the data gathering portion. reviews_per_month are NaN because of a division by 0 error. 
# 
# As of now, we can drop a couple of columns that (in my honest opinion) are not that important:
# 1. id
# 2. host_id
# 3. host_name
# 4. last_review
# 
# 

# In[ ]:


df.drop(columns = ['id', 'host_id', 'host_name', 'last_review'], inplace = True)
df.head()


# We left some columns over that might potentially be useful. If not, we will simply drop them later. First, let's deal with the NaN and potentially invalid data.
# 
# As mentioned earlier, availability_365 can equal 0, meaning that the owner is never open for an year. It is unknown whether the host is just closed or if it's bad data, but we will remove samples where availability_365 = 0. There are no NaN values in this column so we won't need to handle that case. 

# In[ ]:


df_1 = df[df['availability_365'] > 0]
df_1


# **NOTE**: ~17000 samples were removed

# Now changing NaN values in the reviews_per_month into 0 

# In[ ]:


df_1['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df_1


# We had 4 NaN values in the name category as well. We will simply fill these with blanks (i.e. '')

# In[ ]:


df = df_1.copy()
df['name'] = df_1['name'].fillna('to') # we are using 'to' bc it is a stopword -> explained later


# In[ ]:


# checking our NaN values again
df.isnull().values.any() 


# As of this moment, I have no idea how to incorporate latitude and longitude into my model. I think it's a fair assumption to make that latitude and longitude are not normal metrics used when people are considering prices n such. In addition, these two columns are essentially incorporated in the neighbourhood_group and neighbourhood columns. 

# In[ ]:


df.drop(columns = ['latitude', 'longitude'], inplace = True)
df.head()


# <h3> Encoding + Feature Engineering 

# We will not be able to use string types in our model, so we will have to encode room_type, neighbourhood_group and neighbourhood. 
# 
# In addition, the name that the airbnb is titled is (once again in my opinion) an important factor that people consider when going through AirBnb's (no one wants to rent an airbnb called craphole). So we will perform natural language processing on this column and perform sentiment analysis to obtain some kind of score. Note that there are several flaws with this plan such as the fact that there is no dictionary available to map terms specific for real estate and property to a value. Also, given how small the strings are in the name column (character wise), there might be large variance in sentiment values.
# 
# 

# <h4> Encoding

# In[ ]:


df['room_type'].unique()


# In[ ]:


df['neighbourhood_group'].unique()


# In[ ]:


df['neighbourhood'].unique()


# There are no copied values or invalid values (from a glance) so we will not have to perform any cleaning

# In[ ]:


# taking columns with an object type only
object_cols = df.select_dtypes(include = [object])

# dropping the name column bc we do not want to perform encoding on it
object_cols.drop(columns = ['name'], inplace = True)
object_cols.head()


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
object_cols = encoder.fit_transform(object_cols)


# <h4> Sentiment Analysis

# Now we will perform sentiment analysis on the name column. Remember that we added 'to' to the NaN in the name column.  VADER will be our chosen library for sentiment analysis. It is capable of picking up on use of capitalization, slangs and emjois to some extent, making it more accurate for informal writings such as this.

# In[ ]:


# creating a function to remove stop words to decrease our runtime
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
def stop_words(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return filtered_sentence


# Now let's go through our name column one last time. Since our sentiment analysis will only be applied to english, we should check to see if all the titles are in english. We will be using langdetect for this.

# In[ ]:


from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

description = df['name'].astype(str)

errors = []
not_english = []
english = 0
other = 0
for title in description:
    try:
        #print(title)
        if detect(title) == 'en':
            english += 1
        else:
            not_english += [title]
            other += 1
    except LangDetectException:
        other += 1
        errors += [title]
english, other


# Not all of the descriptions of the airbnbs get detected as english even though they are in english, which might create problems for our sentiment analysis. This issue is most likely caused by langdetect having too many languages it is capable of detecting. Couple other factors is that people use slangs, emojis and mix languages together, making it a difficult task to pinpoint the exact language. 

# In[ ]:


not_english


# A method around this is to limit the # of possible languages (using langid) and for entries that are not in english, we can remove those titles and give a sentiment score equivalent to the average of the respective neighborhood. From the world_atlas, the most common languages spoken in NYC are: English, Spanish/Spanish Creole and Chinese [2]. To get more accurate results, we should remove any characters that are not used in proper english (i.e. hypens, slashes, asterisks etc.)
# 
# [2] https://www.worldatlas.com/articles/how-many-languages-are-spoken-in-nyc.html

# In[ ]:


def grammar(string):
    # add whatever else you think you might have to account for
    result = str(string)
    result = result.replace('/', ' ')
    result = result.replace('*', ' ')
    result = result.replace('&', ' ')
    result = result.replace('>', ' ')
    result = result.replace('<', ' ')
    result = result.replace('-', ' ')
    result = result.replace('...', ' ')
    result = result.replace('@', ' ')
    result = result.replace('#', ' ')
    result = result.replace('-', ' ')
    result = result.replace('$', ' ')
    result = result.replace('%', ' ')
    result = result.replace('+', ' ')
    result = result.replace('=', ' ')
    
    return result


# In[ ]:


import langid
desc = description.apply(grammar)
langid.set_languages(['en', 'es', 'zh'])
not_en = []
not_en_index = []
i = 0
for title in desc:
    if langid.classify(title)[0] != 'en':
        not_en += [title]
        not_en_index += [desc.index[i]]
    i += 1
    
len(not_en)


# While a couple of the names that were categorized as not english are still in english, we were still able to obtain a greater # of accurate samples.

# In[ ]:


not_en


# Now we will turn these into empty string lists so when the sentiment analysis is performed, a score of 0 will be obtained.

# In[ ]:


for i in desc.index:
    if desc[i] in not_en:
        desc[i] = ''


# Now removing stop words

# In[ ]:


description = desc.apply(stop_words)
description 


# To use VADER, we have to put our names into a single string

# In[ ]:


# creating a function to convert a list of strings to single string
def to_single_string(list_of_strings):
    result = ''
    for string in list_of_strings:
        result += ' ' +string
    return result

# applying above function
description = description.apply(to_single_string)
description


# In[ ]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sentiment_analyzer = SentimentIntensityAnalyzer()

def sentiment_score(string):
        result = sentiment_analyzer.polarity_scores(string)
        return result


# In[ ]:


sentiment = description.apply(sentiment_score)
sentiment


# VADER uses 4 main metrics to measure sentiment for words. Positive, negative, and neutral represent the proportion of text falling into these categories. The last metric is a sum of all the lexicon ratings that have been normalized between -1 (for most negative) to +1 (for most positive). A general rule is:
# 1. *positive sentiment*: compound score >= 0.05
# 2. *neutral sentiment*: -0.05 < compound score < 0.05
# 3. *negative sentiment*: compound score <= -0.05
# 
# Simply to experiment, I will try two different sentiment scoring methods:
# 1. sentiment value = compound score
# 2. sentiment value = {-1, 0 ,1} depending on polarity of the sentiment from the ranges above
# 
# 

# In[ ]:


# method 1
def compound_score(sent):
    return sent.get('compound')

sentiment_M1 = sentiment.apply(compound_score)
sentiment_M1


# In[ ]:


# method 2
def polarity(sent):
    compound = sent.get('compound')
    if(compound >= 0.05):
        return 1
    elif(compound <= -0.05):
        return -1
    return 0

sentiment_M2 = sentiment.apply(polarity)
sentiment_M2


# Now that we obtained the sentiment scores for the english titles, we have to obtain the average in each neighborhood and apply that average as the sentiment values for the non-english titles.

# In[ ]:


df.columns


# In[ ]:


# creating temporary df
temporary = pd.DataFrame()
temporary['location'] = df['neighbourhood']
temporary['sent_M1'] = sentiment_M1.to_frame()
temporary['sent_M2'] = sentiment_M2.to_frame()
temporary['name'] = description.to_frame()
temporary


# In[ ]:


# removing rows that are in not_en, with not_en_index (which was obtained in the same block of code as not_en)
temporary.drop(index = not_en_index, inplace = True)
temporary


# In[ ]:


neighborhood_sent = temporary.groupby(['location']).mean()
neighborhood_sent


# In[ ]:


def polarity_range(score):
    if(score >= 0.05):
        return 1
    elif(score <= -0.05):
        return -1
    return 0


# obtaining our different sentiment scores
nhood_sent_M1 = neighborhood_sent['sent_M1']
nhood_sent_M2 = neighborhood_sent['sent_M2']

# applying our function to sent_M2 to turn it into -1, 0 and +1 only
nhood_sent_M2 = nhood_sent_M2.apply(polarity_range)


# In[ ]:


nhood_sent_M1


# Now let's combine everything we have 

# Now subbing these averages back into our original Series of sentiment scores

# In[ ]:


sent_m1 = sentiment_M1.copy()
sent_m2 = sentiment_M2.copy()

for index in not_en_index:
    sent_m1[index] = nhood_sent_M1[df['neighbourhood'][index]]
    sent_m2[index] = nhood_sent_M2[df['neighbourhood'][index]]


# Now that we have all the required features, we can put our final dataset together and begin creating our models
# 

# <h3> Preparing Data

# Let's take another look at our dataframe

# In[ ]:


df.head()


# Number_of_reviews and reviews_per_month does not give us much help since we don't know if the reviews were positive or negative. Because it is so ambiguous, we can drop these 2 columns

# In[ ]:


df.drop(columns = ['number_of_reviews',	'reviews_per_month'], inplace = True)
df.head()


# Dropping off all of our features that are not used or have been transformed.

# In[ ]:


df.drop(columns = ['name', 'neighbourhood_group', 'neighbourhood', 'room_type'], inplace = True)
df.head()


# Extracting our target label

# In[ ]:


# creating a copy of df for later
df_copy = df.copy()
#################################
y = df['price']
df.drop(columns = ['price'], inplace = True)


# Now adding our features from before. Two dataframes are created to account for both methods used to calculate sentiment

# In[ ]:


df_sent_m1 = df.copy()
df_sent_m2 = df.copy()

df_sent_m1['sentiment'] = sent_m1
df_sent_m2['sentiment'] = sent_m2


# adding onto our copy of df
df_copy['sent_m1'] = sent_m1
df_copy['sent_m2'] = sent_m2


# Let's take a quick look at the correlation coefficient that price has with every other feature

# In[ ]:


corr_matrix_m1 = df_copy.corr()
corr_matrix_m1["price"].sort_values(ascending = False)


# So there is a very low correlation coefficient with price and our other features (reaching even negative values for our sentiment values. This can indicate one of 2 things:
# 1. there is little to no relationship between price and the other features
# 2. the relationship between the other features is nonlinear to price since correlation coefficient is only a measure of linear relationship

# In[ ]:


object_cols


# In[ ]:


# we will change df_sent_m1 and _m2 into sparse matrices as this data type get processed better
from scipy import sparse
#from scipy.sparse import hstack

X_m1 = sparse.hstack([object_cols, sparse.csr_matrix(df_sent_m1.to_numpy())])
X_m2 = sparse.hstack([object_cols, sparse.csr_matrix(df_sent_m2.to_numpy())])


# <h4> Train/Test

# Now splitting the data by using train/test

# In[ ]:


from sklearn.model_selection import train_test_split
X_train_m1, X_test_m1, y_train_m1, y_test_m1 = train_test_split(X_m1, y, test_size = 0.2, random_state = 42)
X_train_m2, X_test_m2, y_train_m2, y_test_m2 = train_test_split(X_m2, y, test_size = 0.2, random_state = 42)

y_train_m1 = y_train_m1.tolist()
y_train_m2 = y_train_m2.tolist()
y_test_m1 = y_test_m1.tolist()
y_test_m2 = y_test_m2.tolist()


# To obtain more accurate results, normalization will be applied to the appropiate columns

# In[ ]:


temp_train_m1 = X_train_m1[:, X_train_m1.shape[1]-4:].toarray()
temp_train_m2 = X_train_m2[:, X_train_m2.shape[1]-4: X_test_m2.shape[1]-1].toarray()

temp_test_m1 = X_test_m1[:, X_test_m1.shape[1]-4:].toarray()
temp_test_m2 = X_test_m2[:, X_test_m2.shape[1]-4: X_test_m2.shape[1]-1].toarray()

from sklearn.preprocessing import StandardScaler
scaler_m1 = StandardScaler().fit(temp_train_m1)
scaler_m2 = StandardScaler().fit(temp_train_m2)

temp_train_m1 = scaler_m1.transform(temp_train_m1)
temp_train_m2 = scaler_m2.transform(temp_train_m2)

temp_test_m1 = scaler_m1.transform(temp_test_m1)
temp_test_m2 = scaler_m2.transform(temp_test_m2)


# In[ ]:


X_train_m1[:,X_train_m1.shape[1]-4:] = sparse.csr_matrix(temp_train_m1)
X_train_m2[:,X_train_m2.shape[1]-4: X_test_m2.shape[1]-1] = sparse.csr_matrix(temp_train_m2)

X_test_m1[:,X_test_m1.shape[1]-4: ] = sparse.csr_matrix(temp_test_m1)
X_test_m2[:,X_test_m2.shape[1]-4: X_test_m2.shape[1]-1] = sparse.csr_matrix(temp_test_m2)


# Now that we have prepared our train and test data, we can begin to create models

# <h3> Model Building

# The models that we will try are:
# 1. SVM
# 2. Lasso Regression
# 3. Ridge Regression
# 4. Linear Regression
# 5. Random Forests
# 6. Neural network
# 
# Cross validation with stratified k fold will be used to determine the most appropiate model.

# In[ ]:


from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from keras.initializers import RandomNormal

#### FOR NEURAL NETWORK ####
initializer = RandomNormal(mean=0., stddev=1.)

def neural_net():
    model = Sequential()
    model.add(Dense(int(2/3 * X_test_m1.shape[1]), kernel_initializer = initializer, activation = 'relu', input_dim = X_test_m1.shape[1]))
    model.add(Dense(int(4/9 * X_test_m1.shape[1]), kernel_initializer = initializer, activation = 'relu'))
    model.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))
    model.compile(optimizer = 'SGD', loss = 'mse', metrics = ['mae'])
    return model
############################    

models = []
models += [['SVM', SVR(kernel = 'linear')]]
models += [['Lasso', Lasso(alpha = 0.9, normalize = False, selection = 'cyclic')]]
models += [['Ridge', Ridge(alpha = 0.9, normalize = False, solver = 'auto')]]
models += [['Linear', LinearRegression(normalize = False)]]
models += [['Random Forests', RandomForestClassifier(n_estimators = 100, max_features = X_test_m1.shape[1], random_state = 42, max_depth = 9)]]

# for the k fold cross validation
kfold = KFold(n_splits = 10, random_state = 1, shuffle = True)


# Performing cross validation

# <h4> Method 1

# In[ ]:


from sklearn.model_selection import cross_val_score

result_m1 =[]
names = []

for name, model in models:
    cv_score = -1 * cross_val_score(model, X_train_m1, y_train_m1, cv = kfold, scoring = 'neg_mean_absolute_error')
    result_m1 +=[cv_score]
    names += [name]
    print('%s: %f (%f)' % (name,cv_score.mean(), cv_score.std()))


# In[ ]:


from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

estimator = KerasRegressor(build_fn = neural_net, epochs = 1000, batch_size = 2000, verbose=0)
results = -1 * cross_val_score(estimator, X_train_m1, y_train_m1, cv = kfold, scoring = 'neg_mean_absolute_error')
print("Neural net: %f (%f) MSE" % (results.mean(), results.std()))


# <h4> Method 2

# In[ ]:


result_m2 =[]
names = []

for name, model in models:
    cv_score = -1 * cross_val_score(model, X_train_m2, y_train_m2, cv = kfold, scoring = 'neg_mean_absolute_error')
    result_m2 +=[cv_score]
    names += [name]
    print('%s: %f (%f)' % (name,cv_score.mean(), cv_score.std()))


# In[ ]:


estimator = KerasRegressor(build_fn = neural_net, epochs = 1000, batch_size = 2000, verbose=0)
results = -1 * cross_val_score(estimator, X_train_m2, y_train_m2, cv = kfold, scoring = 'neg_mean_absolute_error')
print("Neural net: %f (%f) MSE" % (results.mean(), results.std()))


# It seems that a linear SVM has the lowest mean score, and interestingly the 2nd method of using strict values of +1, 0 or -1 for the sentiment resulted in marginally better results for all models.

# <h3> Final Evaluation

# So our final chosen model will be a linear SVM, using the 2nd method for sentiment

# In[ ]:


model = SVR(kernel = 'linear').fit(X_train_m2, y_train_m2)

# obtaining predictions
predictions = model.predict(X_test_m2)


# Evaluating our model

# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
print("Final mean squared error: %f      R^2 value: %f" %(mean_absolute_error(y_test_m2, predictions), r2_score(y_test_m2, predictions)))


# This is a very high MSE and has a very poor fit to our data.  
# 
# **NOTE**: To be quite honest, tweaking hyperparameters is something that I am still not entirely comfortable doing, so it will be skipped here for now. Will pick this notebook back up once I learn more. 
