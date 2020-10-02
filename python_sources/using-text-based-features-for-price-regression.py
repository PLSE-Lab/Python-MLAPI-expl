#!/usr/bin/env python
# coding: utf-8

# # Introduction

# This kernel is meant to give a simple example of feature extraction from text data and its usefulness for modeling. This is a subject I am just beginning to learn about, and I think that the NYC AirBnB dataset provides a nice practical demonstration. So hopefully this kernel can be useful to others interested in learning about extracting useful features from text.
# 
# Many other kernels for this dataset go through exploratory data analysis (EDA) and fit a variety of models. I will restrict my focus to using ridge regression to predict AirBnB prices-- first without using the textual data, and then again using text features derived from a Bag of Words representation. The latter gives a significant improvement in target score, and the feature coefficients have interesting interpretations.

# # Import packages

# In[ ]:


import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
import math
import statistics as stats
import scipy
import seaborn


# # Import raw data and inspect

# Here's our first look at the data. Note that there are no feature columns for things like "number of bedrooms" or other descriptions of the rental besides the "room_type." But in the 'name' column each row has a description written by the host, which seems like it could be a very rich source of additional features.

# In[ ]:


data_raw = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data_raw.head()


# ### Show dimensions, columns, and datatypes

# In[ ]:


print("Dimensions: ", data_raw.shape)
num_row = data_raw.shape[0]
num_col = data_raw.shape[1]
print(data_raw.dtypes)


# # Cleaning the data

# In[ ]:


data = data_raw.copy() # Make new copy for cleaned data


# ### Drop some unused columns

# In[ ]:


data = data.drop(columns=['host_name']) # Not needed thanks to host_id. Anonymizes the data.


# ### Check for NaN values

# In[ ]:


#Number of NaN values
print(data_raw.isna().sum())
print("Out of",num_row,"rows")


# ### Fill some NaNs

# In[ ]:


data['name'] = data['name'].fillna('')
data['reviews_per_month'] = data['reviews_per_month'].fillna(0)


# ### Convert date strings to DateTime and fill NaNs

# In[ ]:


data['last_review_DT'] = pd.to_datetime(data['last_review'])

#Fill 
mean_date = data['last_review_DT'].mean()
max_date = data['last_review_DT'].max()
data['last_review_DT'] = data['last_review_DT'].fillna(mean_date)

def how_many_days_ago(datetime):
    return (max_date - datetime).days

data['days_since_last_review'] = data['last_review_DT'].apply(how_many_days_ago)


# ### Add one-hot encoding for categorical features

# We'll need to encode all the categorical features in a way that is friendly for the linear model we will use (ridge regression). One-hot encoding is the default choice. The one-hot encoded features are very sparse so we'll leave them in a SciPY CSR format.

# In[ ]:


from sklearn import preprocessing
from sklearn.compose import ColumnTransformer

# Here are the categorical features we are going to create one-hot encoded features for
categorical_features = ['neighbourhood_group','room_type','neighbourhood'] 

encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
one_hot_features = encoder.fit_transform(data[categorical_features])
one_hot_names = encoder.get_feature_names()

print("Type of one_hot_columns is:",type(one_hot_features))


# Just to get an idea of what the one-hot encoded features look like, let's take a peek at them in DataFrame format

# In[ ]:


one_hot_df = pd.DataFrame.sparse.from_spmatrix(one_hot_features)
one_hot_df.columns = one_hot_names # Now we can see the actual meaning of the one-hot feature in the DataFrame
one_hot_df.head()


# # Some very minimal EDA: Dealing with skewness/outliers

# I'm not going to go through much EDA in this kernel, though it's of course an important first step for handling any dataset. So if you haven't done any EDA with this data yet I recommend checking out some of the other kernels for this data on Kaggle. Here I'll just focus on the issue of skewness in the data and a simple fix to make it more suitable for fitting a linear model.

# ### Look at scaled box plot for outliers etc.

# In[ ]:


import seaborn as sns
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

numerical_features = ['latitude','longitude','price','minimum_nights','number_of_reviews','reviews_per_month',
                      'days_since_last_review', 'calculated_host_listings_count','availability_365']

dataScaled = pd.DataFrame(min_max_scaler.fit_transform(data[numerical_features]), columns=numerical_features)
#viz_2=sns.violinplot(data=data, y=['price','number_of_reviews'])
ax = sns.boxplot(data=dataScaled, orient="h")
ax.set_title("Box plots for min-max scaled features")


# In[ ]:


sns.distplot(data['price']).set_title("Distribution of AirBnB prices")


# We see that several features (including the eventual target, price) have very skewed distributions. Most of these make sense: For example, "calculated_host_listings_count" is skewed since most AirBnB hosts simply rent our their own residence occasionally, but some are professionals managing many listings. 
# 
# These skewed distributions are not ideal for fitting linear models. There are many ways to handle this, but I'll take the simple approach of logarithmically transforming the skewed variables, including price. We'll then use a linear model to fit log(1 + price).

# ### Transform data and repeat

# In[ ]:


# I'll transform the following columns by taking log(1+x)
transform_cols = ['price','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count']
for col in transform_cols:
    col_log1p = col + '_log1p'
    data[col_log1p] = data[col].apply(math.log1p)


# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler()

# Now let's plot the numerical features, but take the transformed values for the columns we applied log1p to
numerical_features_log1p = numerical_features
def take_log_col(col):
    if col in transform_cols: return col + '_log1p'
    else: return col
numerical_features_log1p[:] = [take_log_col(col) for col in numerical_features_log1p]

dataScaled_log1p = pd.DataFrame(min_max_scaler.fit_transform(data[numerical_features_log1p]), columns=numerical_features_log1p)
ax = sns.boxplot(data=dataScaled_log1p, orient="h")
ax.set_title("Box plots for min-max scaled features")


# In[ ]:


sns.distplot(data['price_log1p']).set_title("Distribution of log(1 + price)")


# ![](http://)This is a little nicer-- the outliers are a lot less extreme (the max values are only few times larger than the quartiles, rather than 100x larger).  

# # Ridge Regression without using text features

# Now we're ready to start training a model to predict prices.

# ### Define feature and target vectors and perform train-test split

# First we'll use only the numerical features and one-hot-encoded categorical features as predictors. The numerical features will need to be scaled to be used in a regularized linear model.

# In[ ]:


from sklearn.model_selection import train_test_split

numerical_feature_names = ['latitude', 'longitude', 'minimum_nights_log1p','number_of_reviews_log1p','reviews_per_month_log1p', 
                          'days_since_last_review', 'calculated_host_listings_count_log1p', 'availability_365']
numerical_features = data[numerical_feature_names]
scaler = preprocessing.MinMaxScaler()
numerical_features = scaler.fit_transform(numerical_features) # Need to scale numerical features for ridge regression

# Combine numerical features with one-hot-encoded features
features = scipy.sparse.hstack((numerical_features, one_hot_features),format='csr') 
all_feature_names = np.hstack((numerical_feature_names,one_hot_names)) # Store names of all features for later interpretation

target_column = ['price_log1p'] # We will fit log(1 + price) 
target = data[target_column].values

# Perform train and test split of data
rand_seed = 51 # For other models we will use the same random seed, so that we're always using the same train-test split
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=rand_seed)


# ### Fit the model and check $R^2$ scores

# Ridge regression has one hyperparameter $\alpha$, controlling the amount of regularization. SciKit-learn has a handy method "RidgeCV" which automatically selects the best value of $\alpha$ by cross-validation.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn import linear_model\n\nridge_fit = linear_model.RidgeCV(cv=5)\nridge_fit.fit(features_train, target_train)\nprint("RidgeCV found an optimal regularization parameter alpha =",ridge_fit.alpha_)\ntest_score_no_text = ridge_fit.score(features_test,target_test)\nprint("Test score for Ridge Regression without text features:", test_score_no_text)')


# Note that the above test score is an $R^2$ score, which is derived from mean-squared-error or MSE, which is what ridge regression aims to minimize. But we're actually fitting log(1 + price) with this model, so in terms of price we've actually been trying to minimize the mean-square-logarithmic-error or MSLE. This is a reasonable goal for this problem since we probably care more about fractional accuracy of prices rather than absolute accuracy.  

# # Adding features from text using the "Bag of Words" vectorization

# Now let's try to use the rich text data in the 'name' column to improve our predictions. I'm going to use one of the simplest approaches to feature extraction from text, namely the "Bag of Words" model. This simply counts the occurence of words (e.g. "spacious" or "budget") in each text entry, without attempting to keep track of order or sentence structure. For each word it creates a feature column which records the count of that word in each text sample. Since our data has many short text entries, the resulting feature space will be high-dimensional but sparse.
# 
# See https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction for an explanation of Bag of Words and its implementation in scikit-learn.

# ### Vectorize the text feature 'name'

# To properly validate the model, we should define our set of feature vectors by looking at the training set only, and then apply the feature extraction procedure to the test set as well. So for example if some word like "splendiferous" appears in the test set but NOT in the training set, we should not create a feature corresponding to it.
# 
# Apart from setting a "min_df" (see below), I use scikit-learn's CountVectorizer "out of the box."

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

# Same train-test split as before (same random seed)
data_train, data_test = train_test_split(data, test_size=0.2, random_state=rand_seed)

training_corpus = data_train['name'].values # Only use the training set to define the features we are going to extract
vectorizer = CountVectorizer(min_df=3) 
# min_df is the minimum number of times a word needs to appear in the corpus in order to be assigned a vector
vectorizer.fit(training_corpus)
num_words = len(vectorizer.vocabulary_) # Total number of words 
print("Number of distinct words to be used as features:",num_words)


# ### Apply vectorizer to full dataset

# Now we'll apply the vectorizer (which was fit to just the training set) to the whole dataset.

# In[ ]:


full_corpus = data['name'].values
word_features = vectorizer.transform(full_corpus) # This is a sparse matrix of our word-occurrence features 
words = vectorizer.get_feature_names() # The actual words corresponding to the columns of the above feature matrix
word_frequencies = np.array(word_features.sum(axis=0))[0] # The total number of occurrences of each word in the dataset
print("Shape of word-occurrence feature matrix:",word_features.shape)


# ### Combine previous features and vectorized text features

# Let's just add these vectorized text features to the features we use previously.

# In[ ]:


num_non_text = features.shape[1]
features_with_text = scipy.sparse.hstack((features, word_features),format='csr') 
# We want to keep the feature matrix in a sparse format for efficiency
feature_names = np.hstack((all_feature_names, words))   

# Same train-test split as before (same random seed)
features_with_text_train, features_with_text_test, target_train, target_test = train_test_split(
    features_with_text, target, test_size=0.2, random_state=rand_seed)

num_features = num_non_text + num_words

print("Number of non-text features: ",num_non_text)
print("Number of vectorized text features (word occurrences): ",num_words)
print("Features shape including text features: ",features_with_text.shape)


# ### Fit ridge regression again and compare

# We'll fit using ridge regression exactly as before (with the hyperparameter $\alpha$ automatically chosen by cross-validation).

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn import linear_model\n\nridge_fit = linear_model.RidgeCV(cv=5)\nridge_fit.fit(features_with_text_train, target_train)\nprint("RidgeCV found an optimal regularization parameter alpha =",ridge_fit.alpha_)\ntest_score_with_text = ridge_fit.score(features_with_text_test,target_test)\nprint("Test score for Ridge Regression WITHOUT text features:", test_score_no_text)\nprint("Test score for Ridge Regression WITH text features:", test_score_with_text)')


# A pretty nice improvement! And still pretty quick to train despite the large number of features. Using a sparse format for the features was important for efficiency, I find that it takes about 50x longer if we use an ordinary dense numpy array.

# # Interpreting the effect of features, including text

# Just for fun let's take a look at how the different features actually contribute the fit, i.e. what are their coefficients in the linear fit. For example, if the feature corresponding to the word "luxurious" as a large positive coefficient, then it is correlated with higher prices, and vice versa for negative coefficients.

# In[ ]:


coefs = ridge_fit.coef_[0] # Coefficients of the linear fit

# I'll make a num_features-sized array of zeros, and then fill the indices corresponding to the word-occurrence features
# with the total number of counts for the word in that dataset. So features that don't correspond to words have 
# word_counts = 0.
num_features = features_with_text.shape[1]
word_counts = np.zeros(num_features, dtype=int)
word_counts[num_non_text:] = word_frequencies

# Make a DataFrame of feature names, coefficients, and word counts, and sort it by magnitude of the coefficient.
coef_df = pd.DataFrame(data={'names': feature_names, 'coefs': coefs, 'total_word_counts': word_counts})
coef_df_sorted = coef_df.reindex(coef_df['coefs'].abs().sort_values(ascending=False).index)


# Let's scan the top 200 features

# In[ ]:


with pd.option_context('display.max_rows', None): 
    print(coef_df_sorted.head(200))


# This gives some insight into what factors drive the AirBnB prices. "superbowl" having the largest positive coefficient probably indicates that prices were greatly inflated around the time of the 2014 Superbowl which took place in the area. Locations suitable for "events" or "shoots" charge a premium. We can also see which neighborhoods have higher or lower prices. 
# 
# We also note that features like "4bd" and "5bath" are useful, but actually most listings are probably conveying that information in multiple words, i.e. "4 bedroom". The approach of counting invidiual words does not capture info like this that is usually expressed in two words. This motivates us to try extracting features corresponding to two-word phrases as well.

# # Using bigrams to extract better features

# Motivated by the above, let's repeat the analysis, but this time ask CountVectorizer to return occurrences of two-word phrases or "bigrams" as well as single words (1-grams). Also, by default CountVectorizer only considers words with two or more characters, but since we also want to count phrases like "3 bedroom" we will change the token_pattern used to identify words to include single-character words.

# In[ ]:


vectorizer = CountVectorizer(ngram_range=(1, 2), # Tokenize both one-word and two-word phrases
                             min_df=3, # Token should occur at least 3 times in training set
                             token_pattern="[a-zA-Z0-9]{1,30}" # Regular expression defining the form of a word or 1-gram
                            ) 
vectorizer.fit(training_corpus)
num_words = len(vectorizer.vocabulary_) # Total number of words 
print("Number of distinct tokens to be used as features:",num_words)


# In[ ]:


word_features = vectorizer.transform(full_corpus) # This is is a sparse matrix of our word-occurrence features 
words = vectorizer.get_feature_names() # The actual words corresponding to the columns of the above feature matrix
word_frequencies = np.array(word_features.sum(axis=0))[0] # The total number of occurrences of each word in the dataset

features_with_text = scipy.sparse.hstack((features, word_features),format='csr') 
feature_names = np.hstack((all_feature_names, words))   

features_with_text_train, features_with_text_test, target_train, target_test = train_test_split(
    features_with_text, target, test_size=0.2, random_state=rand_seed)

num_features = num_non_text + num_words

print("Number of non-text features: ",num_non_text)
print("Number of vectorized text features (word occurrences): ",num_words)
print("Features shape including text features: ",features_with_text.shape)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn import linear_model\n\nridge_fit = linear_model.RidgeCV(cv=5)\nridge_fit.fit(features_with_text_train, target_train)\nprint("RidgeCV found an optimal regularization parameter alpha =",ridge_fit.alpha_)\ntest_score_with_bigrams = ridge_fit.score(features_with_text_test,target_test)\nprint("Test score for Ridge Regression WITHOUT text features:", test_score_no_text)\nprint("Test score for Ridge Regression with single-word tokens:", test_score_with_text)\nprint("Test score for Ridge Regression with bigram tokens:", test_score_with_bigrams)')


# We get a bit of improvement in the score using bigrams. Let's see what the coefficients of the features look like now.

# In[ ]:


coefs = ridge_fit.coef_[0] # Coefficients of the linear fit

# I'll make a num_features-sized array of zeros, and then fill the indices corresponding to the word-occurrence features
# with the total number of counts for the word in that dataset. So features that don't correspond to words have 
# word_counts = 0.
num_features = features_with_text.shape[1]
word_counts = np.zeros(num_features, dtype=int)
word_counts[num_non_text:] = word_frequencies

# Make a DataFrame of feature names, coefficients, and word counts, and sort it by magnitude of the coefficient.
coef_df = pd.DataFrame(data={'names': feature_names, 'coefs': coefs, 'total_word_counts': word_counts})
coef_df_sorted = coef_df.reindex(coef_df['coefs'].abs().sort_values(ascending=False).index)

with pd.option_context('display.max_rows', None): 
    print(coef_df_sorted.head(200))


# We see that phrases like "5 bedroom" are indeed being used as features, which may account for he improvement in score. However, certain features are redundant, like "bowl" and "super bowl" which always occur simultaneously. An improved approach to feature extraction might avoid such redundancies.

# # Afterword

# We have seen that the NYC AirBnB dataset is a nice playground for applying some simple approaches for extracting text-based features. Clearly I've only scratched the surface of how one could use the text data. For example, one could try combining the counts for tokens like "3 bedroom", "3bd", "3br", "three bedroom" etc. into one feature. Or one may try to extract the size of the rental by looking for numbers preceding the strings "square feet", "sqft", etc. And of course, one could try fitting different regression models to compare how they perform and how well they utilize these text-based features.
# 
# Please feel free to comment on or critique anything in this kernel!
