#!/usr/bin/env python
# coding: utf-8

# **Goal: Using the Kaggle Boston Airbnb dataset, I attempt to use sentiment analysis on written reviews to predict the numerical Airbnb rating. (SPOILER ALERT: It doesn't go very well)**
# 
# The Airbnb dataset has 3 csv files: "calendar.csv", "listings.csv", "reviews.csv". I first create an LSTM model in Keras and train it on the IMDB sentiment dataset. Then I use the trained model on the comments from "reviews.csv" to predict sentiment: either positive (1), or negative (0). Taking the average of these, I see how closely this correlates to the actual ratings, found in "listings.csv".
# 
# A couple things of note: the initial sentiment analysis is a classification task, whereas predicting the rating is really a regression task. My intention is to see how closely we can correlate these.
# 
# Also, since this is my first Kaggle submission, note that I really like functions. I primarily program in Spyder, so writing functions allows for easy and discrete code chunks, and makes debugging much easier.

# In[ ]:


import numpy as np
import pandas as pd
import os
import string
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, save_model, load_model
from keras.layers import Embedding, LSTM, Dense, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import datetime
import math


# First, we load in the IMDB dataset. This is also available via `keras.datasets`, but it comes preprocessed and already encoded there. I want to also work on the preprocessing and cleaning, so I got it directly from the source:
# http://ai.stanford.edu/~amaas/data/sentiment/
# The dataset has 50,000 reviews: 25,000 for the training set and 25,000 for the testing set. Each set is further divided into positive ('pos') and negative ('neg') reviews.

# In[ ]:


def clean_imdb(directory):
    '''
    Returns cleaned dataframe of IMDB reviews with columns ['review', 'sentiment']
    '''
    sentiment = {'neg': 0, 'pos': 1}
    df_columns = ['review', 'sentiment']
    reviews_with_sentiment = pd.DataFrame(columns = df_columns)
    for i in ('test', 'train'):
        for j in ('neg', 'pos'):
            file_path = directory + i + '/' + j
            for file in os.listdir(file_path):
                with open((file_path + '/' + file), 'r',
                          encoding = 'utf-8') as text_file:
                    text = text_file.read()
                review = pd.DataFrame([[text, sentiment[j]]],
                                      columns = df_columns)
                reviews_with_sentiment = reviews_with_sentiment.                                         append(review, ignore_index = True)
    return reviews_with_sentiment

directory = '/kaggle/input/standford-imdb-review-dataset/aclimdb_v1/aclImdb/'
cleaned_imdb = clean_imdb(directory)


# Let's take a look at what a single review with sentiment looks like:

# In[ ]:


cleaned_imdb.iloc[13]


# Next we will load in one of the GloVe word embeddings (https://nlp.stanford.edu/projects/glove/). A word embedding represents words as vectors, which allows them to actually be interpreted by a computer. There are several different versions of GloVe depending on how many dimensions - and therefore memory - you would like to use. I used one of the smaller ones, with 50 dimensions and 6 billion tokens.

# In[ ]:


def load_GloVe(file_path):
    '''
    Loads word embedding .txt file
    Returns word embedding as dictionary
    '''
    GloVe_dict = dict()
    with open(file_path, encoding = 'utf-8') as GloVe_file:
        for line in GloVe_file:
            values = line.split()
            word = values[0]
            coef = np.asarray(values[1:], dtype = 'float32')
            GloVe_dict[word] = coef
    return GloVe_dict

GloVe_file_path = '../input/glove6b50d/glove.6B.50d.txt'
embedding_dict = load_GloVe(GloVe_file_path)


# Let's check out this word embedding a bit. You can see that it includes 400,000 entries of all lower case words. We'll also look at a random slice of the `embedding_dict` keys.

# In[ ]:


len(embedding_dict)


# In[ ]:


[print(word) for word in list(embedding_dict.keys()) if word != word.lower()]


# In[ ]:


list(embedding_dict.keys())[100:125]


# Let's check out how `'under'`, the last word in our list we just printed, is represented in vector space.

# In[ ]:


embedding_dict['under']


# Now that we have our reviews with sentiment and our word embedding loaded, it's time to start cleaning up and preprocessing our reviews. The reviews have all sorts of non-alphanumeric and uppercase characters, as shown in this rather eloquent review:

# In[ ]:


cleaned_imdb.iloc[200].values


# Next, we will strip off all non-alphanumeric characters. This process takes a while, so I also added a print statement to update us on how far we've gotten. Note that while the `cleaned_imdb` reviews are a `pd.DataFrame`, this function is also used later when we are stripping the Airbnb reviews, which are a `pd.Series`, hence the `if` statement. We first strip off all the punctuation and replace it with a single space. Then we replace all whitespace characters, **except** actual spaces.
# 
# It's important to note that this is not always best practice. Consider the following sentence from the review above:
# 
# > WOW!!! DO NOT SEE THIS MOVIE!!! IT SUCKS!!!
# 
# You can tell that this carries a much stronger sentiment than just one (or no) exclamation points. There are ways of dealing with punctuation (see [VADER](https://github.com/cjhutto/vaderSentiment) for example), but I did not use them here.

# In[ ]:


def strip_punctuation_and_whitespace(reviews_df, verbose = True):
    '''
    Strips all punctuation and whitespace from reviews EXCEPT spaces (i.e. ' ')
    Removes "<br />"
    Returns dataframe of cleaned IMDB reviews
    '''
    trans_punc = str.maketrans(string.punctuation,
                               ' ' * len(string.punctuation))
    whitespace_except_space = string.whitespace.replace(' ', '')
    trans_white = str.maketrans(whitespace_except_space,
                                ' ' * len(whitespace_except_space))
    stripped_df = pd.DataFrame(columns = ['review', 'sentiment'])
    for i, row in enumerate(reviews_df.values):
        if i % 5000 == 0 and verbose == True:
            print('Stripping review: ' + str(i) + ' of ' + str(len(reviews_df)))
        if type(reviews_df) == pd.DataFrame:
            review = row[0]
            sentiment = row[1]
        elif type(reviews_df) == pd.Series:
            review = row
            sentiment = np.NaN
        try:
            review.replace('<br />', ' ')
            for trans in [trans_punc, trans_white]:
                review = ' '.join(str(review).translate(trans).split())
            combined_df = pd.DataFrame([[review, sentiment]],
                                       columns = ['review', 'sentiment'])
            stripped_df = pd.concat([stripped_df, combined_df],
                                    ignore_index = True)
        except AttributeError:
            continue
    return stripped_df

stripped_imdb = strip_punctuation_and_whitespace(cleaned_imdb)


# Now let's take another look at our surfer's review from above:

# In[ ]:


stripped_imdb.iloc[200].values


# We are going to need to know how many words we want to include in our tokenizer, so let's see how many words are in each review.

# In[ ]:


def get_length_all_reviews(sentences):
    '''
    Returns a list of length of all reviews
    Used for plotting histogram
    '''
    lengths = [len(i.split(' ')) for i in sentences]
    return lengths

imdb_lengths = get_length_all_reviews(stripped_imdb['review'])


# In[ ]:


max(imdb_lengths)


# Is that a lot of words or not? Let's plot a histogram and see where the majority of reviews are in terms of length. We can tell from the first plot below that 2525 is an outlier, so we also plot up to 1200 to get a better view.

# In[ ]:


def plot_histogram(sentence_lengths, x_dim):
    '''
    Plots histogram of length of all sentences
    '''
    plt.hist(sentence_lengths, 50, [0, x_dim])
    plt.xlabel('Review length (words)')
    plt.ylabel('Frequency')
    plt.title('Review Lengths (Words per review)')
    plt.show()

plot_histogram(imdb_lengths, 2600)
plot_histogram(imdb_lengths, 1200)


# From these plots, it's obvious the 2525 was an outlier, so we will choose a maximum sequence length of 1000 for our tokenizer. Additionally, we will use a `vocabulary_length` of 10,000. By setting `lower = True` within the `Tokenizer` object, we ensure that all words fed to the tokenizer are converted to lowercase. In practice, this may not always be the best idea. Consider the same sentence we looked at above:
# 
# > WOW!!! DO NOT SEE THIS MOVIE!!! IT SUCKS!!!
# 
# Once again, this clearly carries a stronger sentiment than if it had been written in all lowercase. However, the GloVe model we are using only recognizes lowercase words, so we force all words into lowercase.

# In[ ]:


def create_tokenizer(max_words_to_keep, words_review_df):
    '''
    Creates tokenizer
    Returns a tokenizer object and reviews converted to integers
    '''
    tokenizer = Tokenizer(num_words = max_words_to_keep,
                          lower = True,
                          split = ' ')
    tokenizer.fit_on_texts(words_review_df['review'].values)
    return tokenizer,            tokenizer.texts_to_sequences(words_review_df['review'].values)

imdb_sequence_length = 1000
vocabulary_length = 10000
tokenizer, integer_reviews = create_tokenizer(vocabulary_length, stripped_imdb)


# And now looking at the same surfer review, converted to integers:

# In[ ]:


integer_reviews[200]


# However, the reviews are all still different lengths:

# In[ ]:


print(len(integer_reviews[100]))
print(len(integer_reviews[200]))


# We are going to need to ensure all the reviews are the same length when we ultimately feed it to our LSTM model. There are primarily 2 ways of doing this: pad `0` at the front (`pre`) of your sequence or at the end (`post`) of your sequence. Since an LSTM model forgets a little bit of it's training during each step, we will pad the `0`'s at the front of our sequence, ensuring most of the information is carried through the model.

# In[ ]:


def pad_zeros(encoded_reviews, padding_length, padding = 'pre'):
    '''
    Pads integer reviews either left ('pre') or right ('post')
    '''
    return pad_sequences(encoded_reviews,
                         maxlen = padding_length,
                         padding = padding)

padded_reviews = pad_zeros(integer_reviews,
                           imdb_sequence_length,
                           padding = 'pre')


# Once again, same review, now with `0` padding at the start of the review

# In[ ]:


padded_reviews[200]


# We are finally ready to start training out LSTM model. Before we do that, let's recap what we've done to get us this far:
# 
# * We loaded in our IMDB dataset, which has 25,000 positive and 25,000 negative reviews. Each of the reviews is assigned their respective sentiment (negative = 0, positive = 1). We loaded these all into one dataframe, `cleaned_imdb`.
# * We loaded our GloVe model as `embedding_dict`, which has 400,000 lowercase words. Each of these words is mapped to a 50-dimensional vector space.
# * We stripped all punctuation and extra whitespace from the IMDB reviews and created the `stripped_imdb` dataframe. This allows us to actually represent the words within the reviews in the vector space given by the GloVe model.
# * By plotting a histogram of the length of each of the IMDB reviews, we were able to determine an appropriate maximum length to feed our `tokenizer`. This prevents it from having to carry around a bunch of `0`'s just to match the length of one very wordy review.
# * We converted the words in each review to integers and captured them in the list `integer_reviews`. Computers like numbers a lot more than words, so this will allow our model to actually train on the reviews.
# * Finally, we padded `0`'s at the beginning of every review - up to our `imdb_sequence_length` of 1000 - determined from our histogram plot and created the array `padded_reviews`

# Great! Now we can start creating and training our LSTM model. First, we will need to split our `padded_reviews` array into training and testing sets. We split it right in half, but you can adjust this to have more or less samples captured in your testing set. We've also set the `random_state` for reproducibility.

# In[ ]:


split = 0.5
X_train, X_test, y_train, y_test = train_test_split(padded_reviews,
                                                    stripped_imdb['sentiment'],
                                                    test_size = split,
                                                    random_state = 42)


# In[ ]:


def create_LSTM_model(vocab_length, in_length, opt = 'Adam',
                      learning_rate = 0.001):
    '''
    Returns 1-layer LSTM model
    '''
    model = Sequential()
    model.add(Embedding(vocab_length, 32))
    model.add(LSTM(32))
    model.add(Dense(1, activation = 'sigmoid'))
    optimizer = getattr(keras.optimizers, opt)(lr = learning_rate)
    model.compile(loss = 'binary_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy'])
    return model

LSTM_model = create_LSTM_model(vocabulary_length,
                               imdb_sequence_length,
                               opt = 'Adam',
                               learning_rate = 0.001)
print(LSTM_model.summary())


# This is about as simple an LSTM model as one can make, but before we try to complicate matters, let's see how well it performs. We will save the training performance in `LSTM_history` to reference after training.

# In[ ]:


ep = 10
LSTM_history = LSTM_model.fit(X_train, y_train,
                              validation_data = (X_test, y_test),
                              batch_size = 1000, epochs = ep, verbose = 1)


# In[ ]:


plt.plot(range(10), LSTM_history.history['val_acc'], '--o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy After {} Epochs'.format(ep))
plt.show()


# Looks like we hit our maximum accuracy of ~88% after only 4 epochs! This is without any optimization of hyperparameters whatsoever, and one of the simplest LSTM models you can put together. Since this is only an exploration to see if we can correlate sentiment with actual ratings, this accuracy is sufficient for now. We could easily go back and alter any of the hyperparameters, add more layers to the model, etc. if we so desire.

# Now, finally on to the fun part! Let's see how well this model can predict the sentiment on the Airbnb reviews. First, we'll load in the datasets.

# In[ ]:


def load_airbnb_datasets():
    '''
    Run this if you need to load in the Boston Airbnb datasets
    '''
    df_calendar = pd.read_csv('../input/boston/calendar.csv')
    df_listings = pd.read_csv('../input/boston/listings.csv')
    df_reviews = pd.read_csv('../input/boston/reviews.csv')
    return df_calendar, df_listings, df_reviews

df_calendar, df_listings, df_reviews = load_airbnb_datasets()


# Let's see how these are broken down.

# In[ ]:


df_reviews.iloc[13]


# Ok, so `df_reviews` has both a `listing_id` and a review in the `comments` column. We will use these to match the reviews up to the listings. Let's see how `df_listings` is setup.

# In[ ]:


df_listings.iloc[13]


# The listings have a ton more information, but all we are going to focus on for this exploration are the `id` and `review_scores_rating`. By matching up the `listing_id` in `df_reviews` with the `id` in `df_listings`, we can try to correlate an average sentiment calculated from our model with the `review_scores_rating`.

# First, let's see how many reviews a single listing might have. We need to have some threshold so that we can try to apply the central limit theorem.

# In[ ]:


ids, counts = np.unique(df_reviews['listing_id'], return_counts = True)
print('Minimum number of reviews: ' + str(min(counts)))
print('Maximum number of reviews: ' + str(max(counts)))


# Ok...so this is a pretty big range. In theory, we could make a `for` loop, run from 1 to 404 and see when (if?) we approach a good approximation. For now, let's make a somewhat arbitrary cutoff and only look at listings that have at least 100 reviews. If we want, we can come back later and experiment with this.

# In[ ]:


gt_100 = np.where(counts > 100)[0]
ids_gt_100 = ids[gt_100]

print('Number of listings with greater than 100 reviews: ' + str(len(gt_100)))
print('\nIndices of listings with greater than 100 reviews:\n' + str(gt_100))
print('\nAssociated listings with greater than 100 reviews:\n' + str(ids_gt_100))


# We now have everything we need:
# 
# * LSTM model with relatively good accuracy
# * Listing IDs of Airbnbs with greater than 100 reviews
# 
# Let's start analyzing these reviews! We create a `for` loop to run through each of the listing IDs in `ids_gt_100`, find them in the `df_reviews` dataframe, and then run through the exact same sequence of preprocessing as we did for the IMDB dataset - makes it helpful that we defined all of these functions above. At the end, however, instead of training a model, we use our `LSTM_model` to make predictions of sentiment and collect them in a `ratings` dictionary. The `ratings` dictionary will have the `listing_id` as the key (which we call `temp_id` in the loop), and a list of `[actual_rating, predicted_rating]` as the values.
# 
# One thing I will note, I chose an `airbnb_sequence_length` of 250, as opposed to the `imdb_sequence_length` of 1000. When I ran this through originally, I plotted similar histograms to the IMDB dataset of each of the listing IDs, and found 250 was a pretty consistent cutoff in terms of length. I don't want to include 133 histograms in this notebook, but you can feel free to uncomment the following line of code in the `for` loop and see for yourself (also feel free to adjust 1000 to whatever you'd like):
# 
# `plot_histogram(airbnb_lengths, 1000)`

# In[ ]:


ratings = {}

for temp_id in ids_gt_100:
    temp_comments = df_reviews.loc[df_reviews['listing_id'] ==                                    temp_id]['comments']
    
    # Rename for function, then strip punctuation and whitespace
    temp_comments.rename('review', inplace = True)
    stripped_airbnb = strip_punctuation_and_whitespace(temp_comments,
                                                       verbose = False)
    
    # Plot histogram of review length. Find sequence cutoff length
    airbnb_lengths = get_length_all_reviews(stripped_airbnb['review'])
    #plot_histogram(airbnb_lengths, 1000)
    airbnb_sequence_length = 250
    
    # Tokenizer with 10000 word vocabulary
    airbnb_tokenizer, airbnb_integer_reviews =                                             create_tokenizer(vocabulary_length,
                                                             stripped_airbnb)
    # Pad zeros up to airbnb_sequence_length
    airbnb_padded_reviews = pad_zeros(airbnb_integer_reviews,
                                      airbnb_sequence_length,
                                      padding = 'pre')
    
    # Predict sentiment
    airbnb_sentiments = LSTM_model.predict_classes(airbnb_padded_reviews)
    predicted_rating = round(airbnb_sentiments.mean() * 100, 1)

    # Print comparisons
    actual_rating = df_listings.loc[df_listings['id'] == temp_id]                    ['review_scores_rating'].values[0]
    print('--- Listing ID ' + str(temp_id) + ' ---\nPredicted Rating: [' +           str(predicted_rating) + '] vs. Actual Rating: [' +           str(actual_rating) + ']')
    ratings[temp_id] = [actual_rating, predicted_rating]


# All right, so this...isn't great. Predicted Ratings aren't matching up that well with Actual Ratings. Let's make this visual just to see what exactly we're dealing with. First, we'll sort the ratings in ascending order of Actual Rating. Since we created a list of `[actual_rating`, `predicted_rating`] as the values in the `ratings` dictionary, we can easily sort them while also ensuring both ratings are still referring to the same `listing_id`. What we don't want to do is sort `actual_rating` and `predicted_rating` *separately* - if we did that, we would likely see a false correlation as both ratings would obviously increase.

# In[ ]:


sorted_ratings = [ratings[i] for i in ratings]
sorted_ratings.sort()
sorted_ratings


# Now, we can separate them feeling confident the ratings are in the correct order. You can easily match these lists up with the values above.

# In[ ]:


plot_actual_ratings = [rating[0] for rating in sorted_ratings]
plot_predicted_ratings = [rating[1] for rating in sorted_ratings]

print('First 10 Actual Ratings: \n' + str(plot_actual_ratings[0:10]))
print('\nFirst 10 Predicted Ratings: \n' + str(plot_predicted_ratings[0:10]))


# Lastly, we'll plot the ratings and see how well we did! Note that the y-axes do not display comparable ranges. Since the model gets trained differently each time we run through this, the predicted ratings tend to fluctuate, so we can't anticipate and set limits on the ranges. However, we want to show how ratings fluctuate within one dataset and see if any potential fluctuations are comparable, so this usually works to our advantage since the plots automatically get overlaid.

# In[ ]:


ax1_min = int(math.floor(min(plot_predicted_ratings)/5) * 5)

fig, ax1 = plt.subplots()
predicted_line = ax1.plot(range(len(plot_predicted_ratings)),
                          plot_predicted_ratings,
                          color = 'orange',
                          label = 'Predicted Ratings')
ax1.set_xlabel('Listing')
ax1.set_ylabel('LSTM Predicted Rating', color = 'orange')
ax1.tick_params(axis = 'y', color = 'orange')
plt.setp(ax1.get_yticklabels(), color = 'orange')

ax2 = ax1.twinx()
actual_line = ax2.plot(range(len(plot_actual_ratings)),
                       plot_actual_ratings,
                       color = 'black',
                       label = 'Actual Ratings')
ax2.set_ylabel('Actual Ratings', color = 'black')
ax2.set_ylim(70, 100)
ax2.spines['left'].set_color('orange')

ax1.legend((predicted_line + actual_line),
           ['Predicted Rating', 'Actual Rating'],
           loc = 'upper center',
           bbox_to_anchor = (0.5, -0.15),
           fancybox = True,
           shadow = True,
           ncol = 2)
plt.title('Predicted Ratings vs. Actual Ratings for Boston Airbnbs')
plt.show()


# Obviously, this was a failed experiment. I'll be honest, it went a lot better in my head. However, it was a super fun exploration and I learned a lot about both LSTMs and presenting my info logically in a notebook. From a data analysis perspective, I don't think correlating a binary classifier (i.e. "positive" or "negative") with what should really be a regression task (predicting a numerical rating) was the best approach. However, I do think this can be used to show how differently people rate things when given a numerical assessment versus a verbal/written assessment.
