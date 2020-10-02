#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import norm

from datetime import datetime
import re
from wordcloud import WordCloud, STOPWORDS

from collections import Counter

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

import re
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[ ]:


df_reviews = pd.read_json('../input/imdb-spoiler-dataset/IMDB_reviews.json', lines=True)
df_details = pd.read_json('../input/imdb-spoiler-dataset/IMDB_movie_details.json', lines=True)


# In[ ]:


df_reviews = pd.merge(df_reviews,df_details.drop('rating',axis=1),on='movie_id')[df_reviews.columns.values]


# In[ ]:


df_reviews.is_spoiler.value_counts()


# In[ ]:


le = preprocessing.LabelEncoder()
df_reviews.is_spoiler = le.fit_transform(df_reviews.is_spoiler)


# Since the train_test_split function hangs on this large dataset, i will stratify and sample manually.

# In[ ]:


# stratify
spoilers = df_reviews[df_reviews.is_spoiler == 1].sample(frac=1).reset_index(drop=True)
others = df_reviews[df_reviews.is_spoiler == 0].sample(frac=1).reset_index(drop=True)

# train set
df_reviews = pd.concat([spoilers.head(35000), others.head(100000)]).sample(frac=1).reset_index(drop=True)

# test set
test = pd.concat([spoilers.tail(1500), others.tail(3000)]).sample(frac=1).reset_index(drop=True)


# # Movie spoiler ratio
# What hints can the movie name give about its review? I use a metric which I will call Movie Spoiler Ratio. This measures the fraction of each movie's total reviews which are spoilers. It gives us the tendency of a movie to have revealing reviews.

# In[ ]:


movie_spoiler_counts = df_reviews.groupby(by='movie_id').is_spoiler.agg(['sum']).reset_index().rename(columns={'sum':'movie_spoiler_count'})
movie_total_counts = df_reviews.groupby(by='movie_id').is_spoiler.agg(['count']).reset_index().rename(columns={'count':'movie_total_count'})
df_ratio = pd.merge(movie_spoiler_counts, movie_total_counts, on='movie_id')
df_ratio['movie_spoiler_ratio'] = df_ratio.movie_spoiler_count / df_ratio.movie_total_count
df_ratio.drop(['movie_spoiler_count','movie_total_count'], axis=1, inplace=True)
display(df_ratio)
df_reviews = pd.merge(df_reviews, df_ratio, on='movie_id')


# In[ ]:


df_ratio.shape


# In[ ]:


df_temp = pd.DataFrame({"label":["spoiler", "non spoiler"],
                      "movie_spoiler_ratio":[df_reviews[df_reviews.is_spoiler==1].movie_spoiler_ratio.mean(), 
                                            df_reviews[df_reviews.is_spoiler==0].movie_spoiler_ratio.mean()]})
sns.barplot(x = "label", y = "movie_spoiler_ratio", data = df_temp);


# # User spoiler ratio
# This measures the fraction of each users total reviews which are spoilers. It tells us how likely a user is to give spoiler review.

# In[ ]:


user_spoiler_counts = df_reviews.groupby(by='user_id').is_spoiler.agg(['sum']).reset_index().rename(columns={'sum':'user_spoiler_count'})
user_total_counts = df_reviews.groupby(by='user_id').is_spoiler.agg(['count']).reset_index().rename(columns={'count':'user_total_count'})
user_spoiler_ratio = pd.merge(user_spoiler_counts, user_total_counts, on='user_id')
user_spoiler_ratio['user_spoiler_ratio'] = user_spoiler_ratio.user_spoiler_count / user_spoiler_ratio.user_total_count
user_spoiler_ratio.drop(['user_spoiler_count','user_total_count'], axis=1, inplace=True)
display(user_spoiler_ratio.sample(frac=1))
df_reviews = pd.merge(df_reviews, user_spoiler_ratio, on='user_id')


# In[ ]:


df_temp = pd.DataFrame({"label":["spoiler", "non spoiler"], 
                        "user_spoiler_ratio":[df_reviews[df_reviews.is_spoiler==1].user_spoiler_ratio.mean(), 
                                            df_reviews[df_reviews.is_spoiler==0].user_spoiler_ratio.mean()]})
sns.barplot(x = "label", y = "user_spoiler_ratio", data = df_temp);


# Let's set 0.1 as the cut-off point. If user-spoiler-ratio is above 0.1, the user is prone to writing spoilers, else not.

# In[ ]:


df_reviews.user_spoiler_ratio = (df_reviews.user_spoiler_ratio >= 0.1) + 0


# # Review recency
# We have the date the review was given and also the release date of the movie. Let me define a metric called Review relevance which is the number of days elapsed between the movie's release and the date of review. In other words, it tells us how recent the review is. I suspect that more recent reviews could have spoilers (probably an emotional outburst :-p)

# In[ ]:


def formatReviewDate(review_date):
    return datetime.strptime(review_date, '%d %B %Y').date()

def formatReleaseDate(release_date):
    date = None
    try:
        date = datetime.strptime(release_date, '%Y-%m-%d').date()
    except:
        try:
            date = datetime.strptime(release_date+'-01', '%Y-%m-%d').date()
        except:
            date = datetime.strptime(release_date+'-01-01', '%Y-%m-%d').date()
    return date


# In[ ]:


df_reviews['review_date'] = df_reviews.review_date.apply(formatReviewDate)
df_details['release_date'] = df_details.release_date.apply(formatReleaseDate)


# In[ ]:


merged_df_reviews = pd.merge(df_reviews,df_details,on='movie_id')
merged_df_reviews['review_relevance'] = (merged_df_reviews.review_date - merged_df_reviews.release_date).apply(lambda x: abs(x.days))


# In[ ]:


print("Mean recency (spoilers) =",merged_df_reviews[merged_df_reviews.is_spoiler==1].review_relevance.mean())
print("Mean recency (non-spoilers) =",merged_df_reviews[merged_df_reviews.is_spoiler==0].review_relevance.mean())


# It appears that the mean recency of spoilers and non-spoilers is very close. This is not likely to play a role in the classification.

# # Genre
# Once again, I suspect that some genres are more likely to have spoiler reviews.

# In[ ]:


genre_names = np.unique(np.array(' '.join(df_details.genre.str.join(' ')).split()))
print(genre_names)


# In[ ]:


df_genre = pd.DataFrame();
for genre in genre_names:
    df_genre[genre.lower()] = 0


# In[ ]:


for index,row in df_details.iterrows():
    details = df_details[df_details.movie_id == row['movie_id']]
    df_genre.at[index,'movie_id'] = row['movie_id']
    for genre in genre_names:
        df_genre.at[index, genre.lower()] = int(genre in details['genre'].tolist()[0])


# In[ ]:


df_reviews_temp = pd.merge(df_reviews, df_genre, on="movie_id")


# In[ ]:


e = 0.001
genre_spoiler_ratio = np.zeros(len(genre_names))
for i,g in enumerate(genre_names):
    genre_spoiler_ratio[i] = df_reviews_temp[(df_reviews_temp.is_spoiler==1) & (df_reviews_temp[g.lower()]==1)].shape[0] / (df_reviews_temp[df_reviews_temp[g.lower()]==1].shape[0]+e)


# In[ ]:


ax = sns.barplot(x=genre_spoiler_ratio,y=genre_names)
ax.set(xlabel="Spoiler ratio");
plt.show();


# **Action, Adventure, Fantasy, Horror, Mystery, Sci-Fi, Thriller** have the highest spoiler ratios. But we don't find substantial evidence of spoilers. However, each movie might belong to more than one genre. So let's try combinations using these to see if they yield any evidence.

# In[ ]:


e = 0.001
selected = ["Action","Adventure","Fantasy","Horror","Mystery","Sci-Fi","Thriller"]

genre_2_labels = []
genre_2_ratios = []

i=0
while i < len(selected):
    j = i+1
    while j < len(selected):
        genre_2_labels.append(selected[i]+"+"+selected[j])
        spoilers = df_reviews_temp[(df_reviews_temp[selected[i].lower()]==1) & (df_reviews_temp[selected[j].lower()]==1) & df_reviews_temp.is_spoiler].shape[0]
        total = df_reviews_temp[(df_reviews_temp[selected[i].lower()]==1) & (df_reviews_temp[selected[j].lower()]==1)].shape[0]
        genre_2_ratios.append(spoilers / (total+e))
        j+=1
    i+=1
    
ax = sns.barplot(x=genre_2_ratios,y=genre_2_labels)
ax.set(xlabel="Spoiler ratio");
plt.show();


# In[ ]:


e = 0.001
selected = ["Action","Adventure","Fantasy","Horror","Mystery","Sci-Fi","Thriller"]

genre_3_labels = []
genre_3_ratios = []

i=0
while i < len(selected):
    j = i+1
    while j < len(selected):
        k = j+1
        while k < len(selected):
            genre_3_labels.append(selected[i]+"+"+selected[j]+"+"+selected[k])
            spoilers = df_reviews_temp[(df_reviews_temp[selected[k].lower()]==1) & (df_reviews_temp[selected[i].lower()]==1) & (df_reviews_temp[selected[j].lower()]==1) & df_reviews_temp.is_spoiler].shape[0]
            total = df_reviews_temp[(df_reviews_temp[selected[k].lower()]==1) & (df_reviews_temp[selected[i].lower()]==1) & (df_reviews_temp[selected[j].lower()]==1)].shape[0]
            genre_3_ratios.append(spoilers / (total+e))
            k+=1
        j+=1
    i+=1
    
plt.figure(figsize=(10,10))
ax = sns.barplot(x=genre_3_ratios,y=genre_3_labels)
ax.set(xlabel="Spoiler ratio");
plt.show();


# Genres prone to spoilers:
# * Fantasy + Sci-Fi
# * Adventure + Mystery
# * Action + Mystery + Sci-Fi
# * Adventure + Horror + Thriller
# * Adventure + Sci-Fi + Thriller
# * Fantasy + Mystery + Thriller

# In[ ]:


def isListSubset(a,b):
    count = 0
    for i,v in enumerate(a):
        if v in b:
            count += 1
    return count == len(a)

def getGenreRatio(genres):
    return isListSubset(["Fantasy","Sci-Fi"], genres) or isListSubset(["Adventure","Mystery"], genres) or isListSubset(["Action","Mystery","Sci-Fi"], genres) or isListSubset(["Adventure","Horror","Thriller"], genres) or isListSubset(["Adventure","Thriller"], genres) or isListSubset(["Fantasy","Mystery"], genres)


# In[ ]:


df_reviews['genre_spoiler_ratio'] = pd.merge(df_reviews,df_details,on="movie_id").genre.apply(getGenreRatio)+0


# # Rating

# In[ ]:


sns.boxplot(x=df_reviews.is_spoiler,y=df_reviews.rating);


# Review rating doesn't seem to offer much.

# # Review text

# It is difficult to process the text to find a spoiler. Sequence models may not necessarily perform well because we are not identifying the tone or sentiment here. The only way to tell a spoiler from a non-spoiler is to compare the review to the plot of the movie.
# 
# We have Review text in the reviews dataset and plot summary in the details dataset.
# 
# I will calculate the similarities between the review and corresponding plot summary using average word embeddings of GloVe vectors. 
# 
# Both review and plot summary is made up of many words - an average vector can summarize information about a collection of words. Calculating the euclidean distance between two average vectors can tell us the similarity (or difference) between the two texts.
# 
# This technique is based on the assumption that spoiler reviews are more similar to their plot summaries.

# In[ ]:


stopwords = set(stopwords.words('english'))
stopwords_dict = Counter(stopwords)

def preprocess_text(review):
    review = review.lower() # Convert to lowercase
    review = re.sub('[^a-zA-Z]',' ', review) # Remove words with non-letter characters
    words = review.split()
    words = [word for word in words if word not in stopwords_dict] # Remove stop words
    review = " ".join(words)
    return review


# In[ ]:


df_reviews.review_text = df_reviews.review_text.apply(preprocess_text)


# In[ ]:


embeddings_index = dict()
f = open('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# In[ ]:


def getAverageWordEmbedding(text):
    words = text.split()
    n = 0
    sumEmbed = np.zeros(50)
    
    for word in words:
        if word in embeddings_index:
            sumEmbed += embeddings_index[word]
            n += 1
    
    avgEmbed = sumEmbed / n
    return avgEmbed
    
def EuclideanDist(e1, e2):
    return np.sum(np.square(e1-e2), axis = 1)


# In[ ]:


plot_review_merge = pd.merge(df_reviews, df_details[['movie_id','plot_summary']], on = 'movie_id')
review_embed = plot_review_merge.review_text.apply(getAverageWordEmbedding)
plot_embed = plot_review_merge.plot_summary.apply(getAverageWordEmbedding)


# In[ ]:


euclideans = EuclideanDist(np.stack(review_embed), np.stack(plot_embed))


# In[ ]:


plot_review_merge['euclideans'] = euclideans

df_temp = pd.DataFrame({"label":["spoiler", "non spoiler"], 
                        "euclidean_dist":[plot_review_merge[plot_review_merge.is_spoiler==1].euclideans.mean(), 
                                            plot_review_merge[plot_review_merge.is_spoiler==0].euclideans.mean()]})
sns.barplot(x = "label", y = "euclidean_dist", data = df_temp);


# # Model Train

# Using a SVM model.

# In[ ]:


x = df_reviews[['movie_spoiler_ratio','user_spoiler_ratio','genre_spoiler_ratio','rating']]
x['euclideans'] = euclideans
x = x.fillna(0)


# In[ ]:


model_svm = svm.SVC(gamma='scale',C=10)
model_svm.fit(x, df_reviews.is_spoiler)


# In[ ]:


predictions = model_svm.predict(x)


# In[ ]:


print(classification_report(df_reviews.is_spoiler, predictions))
print(confusion_matrix(df_reviews.is_spoiler, predictions))


# The model correctly identified ~34,593 out of 35,000 spoilers. However, it falsely classified ~407 spoiler reviews as spoiler-free in the training data. The training recall was ~99%. This could be overfitting, let's verify on unseen data.

# # Model Test

# Since we cannot recalculate the features like user_spoiler_ratio, we have to borrow movie and user likelihoods (of having spoilers) from the train data.

# In[ ]:


# user_spoiler_ratio
test1 = pd.merge(test, user_spoiler_ratio, how = 'left', on = 'user_id')

# movie_spoiler_ratio
test1 = pd.merge(test1, df_ratio, how = 'left', on = 'movie_id')

# genre_spoiler_ratio
test1['genre_spoiler_ratio'] = pd.merge(test1, df_details,on="movie_id").genre.apply(getGenreRatio)+0

# euclidean distances between review and plot
test1.review_text = test1.review_text.apply(preprocess_text)
plot_review_merge_test = pd.merge(test1, df_details[['movie_id','plot_summary']], on = 'movie_id')
review_embed_test = plot_review_merge_test.review_text.apply(getAverageWordEmbedding)
plot_embed_test = plot_review_merge_test.plot_summary.apply(getAverageWordEmbedding)
test1['euclideans'] = EuclideanDist(np.stack(review_embed_test), np.stack(plot_embed_test))


# In[ ]:


x_test = test1[['movie_spoiler_ratio','user_spoiler_ratio','genre_spoiler_ratio','euclideans','rating']]
x_test = x_test.fillna(0)


# In[ ]:


predictions_new = model_svm.predict(x_test)


# In[ ]:


print(classification_report(test1.is_spoiler, predictions_new))
print(confusion_matrix(test1.is_spoiler, predictions_new))


# The model correctly identified ~463 out of 1500 spoilers. However, it falsely classified ~1037 spoiler reviews as spoiler-free in the training data. The training recall was ~31%.
# 
# This means we were able to correctly classify only 31% of the spoilers. 
# 
# * It is challenging to identify spoilers because the exact definition of a spoiler is blurry. Not every detail in a review is necessarily a spoiler. So there is no well-defined criteria to classify it.  
# * But the good news is, we can make some guesses based on historical data about the user who wrote the review, the movie and other common knowledge.  
# * We could get better results by training on more data.  
