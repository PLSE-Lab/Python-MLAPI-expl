#!/usr/bin/env python
# coding: utf-8

# ## Let us perform Web Scraping using BeautifulSoup 

# Importing the required libraries and extracting the Movie reviews and their ratings

# In[ ]:


from bs4 import BeautifulSoup


# In[ ]:


soup = BeautifulSoup(open("../input/1.html",encoding="utf8"), "html.parser")


# Let us take a look at the html structure 

# In[ ]:


movie_containers = soup.find_all('div' , class_ = 'review-container')
print(type(movie_containers))
print(len(movie_containers))


# We can see there are 1124 containers consisting of the reviews and the ratings

# Let us try to extract the reviews

# In[ ]:


first_movie = movie_containers[0]
first_movie.a.text


# Let us try to find the ratings of the reviews

# In[ ]:


temp = first_movie.span.text


# In[ ]:


temp


# We can see that the reviews and ratings require cleaning which we will deal with it later

# Let us try to extract all the reviews and the ratings

# In[ ]:


# Lists to store the scraped data in
reviews = []
ratings = []

# Extract data from individual movie container
for container in movie_containers:
    
    #review
    review = container.a.text
    reviews.append(review)
    
    #rating
    rating = container.span.text
    ratings.append(rating)
   


# Let us try to put all the ratings into a dataframe

# In[ ]:


import pandas as pd
import numpy as np

test_df = pd.DataFrame({'Review': reviews,'Rating': ratings})
print(test_df.info())
test_df.head()


# And woah we have scrapped all the reviews and the ratings 
# Pandas do actualy make our work easier!

# Let us perform cleaning on the reviews and the ratings

# In[ ]:


test_df.loc[:, 'Rating'] = test_df['Rating'].str[6:8]


# In[ ]:


test_df.loc[:, 'Rating'] = test_df['Rating'].str.replace('/', '')
test_df.loc[:, 'Review'] = test_df['Review'].str.replace('\n', '')
test_df.loc[:, 'Rating'] = test_df['Rating'].str.replace('-', '')


# In[ ]:


import re
def split_it(rating):
    return re.sub('[a-zA-Z]+','NaN', rating)


# In[ ]:


test_df['Rating'] = test_df['Rating'].apply(split_it)


# In[ ]:


test_df = test_df[test_df.Rating.str.contains("NaN") == False]


# In[ ]:


test_df['Rating'] = test_df['Rating'].apply(pd.to_numeric)


# In[ ]:


test_df.head()


# We can see that we have cleaned the Ratings and the Reviews
# There might be some redundency in the cleaning which I will update later

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.countplot(test_df['Rating'])


# We can see that there are some Ratings above 10 that we need to get rid of.

# In[ ]:


test_df = test_df[test_df.Rating <= 10]


# In[ ]:


sns.countplot(test_df['Rating'])


# Let us look at the descriptions

# In[ ]:


test_df.describe()


# Let us look if there is a relation between a review length and a Rating

# In[ ]:


test_df['Review']=test_df['Review'].astype(str)
test_df['Review Length']=test_df['Review'].apply(len)

g = sns.FacetGrid(data=test_df, col='Rating')
g.map(plt.hist, 'Review Length', bins=50)


# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x='Rating', y='Review Length', data=test_df)


# We can see that Reviews with ratings 1 and and 8 are the longest

# Let us try to use machine learning and use NLP analytics

# In[ ]:


from collections import Counter
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize


# In[ ]:


nltk.download('punkt')


# Let us try to extract all the words and try to perform analysis on it

# In[ ]:


a = test_df['Review'].str.lower().str.cat(sep=' ')


# In[ ]:


# removes punctuation,numbers and returns list of words
b = re.sub('[^A-Za-z]+', ' ', a)


# In[ ]:


#remove all the stopwords from the text
stop_words = list(get_stop_words('en'))         
nltk_words = list(stopwords.words('english'))   
stop_words.extend(nltk_words)

newStopWords = ['game','thrones', 'bran', 'stark', 'dragons']
stop_words.extend(newStopWords)


# In[ ]:


word_tokens = word_tokenize(b)


# We have extracted all the words in the reviews. Let us find the total length

# In[ ]:


len(word_tokens)


# We will now remove the stop words from the reviews

# In[ ]:


filtered_sentence = []
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)


# In[ ]:


len(filtered_sentence)


# We have removed all the stop words and reduced the size by half.

# In[ ]:


# Remove characters which have length less than 2  
without_single_chr = [word for word in filtered_sentence if len(word) > 2]

# Remove numbers
cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]   


# Let us look at the top 10 most used words in a review.

# In[ ]:


top_N = 100
word_dist = nltk.FreqDist(cleaned_data_title)
rslt = pd.DataFrame(word_dist.most_common(top_N),
                    columns=['Word', 'Frequency'])

plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.barplot(x="Word",y="Frequency", data=rslt.head(7))


# We can see that the most common words are positive indicating how great Game of thrones is!

# In[ ]:


from wordcloud import WordCloud, STOPWORDS


# In[ ]:


def wc(data,bgcolor,title):
    plt.figure(figsize = (100,100))
    wc = WordCloud(background_color = bgcolor, max_words = 1000,  max_font_size = 50)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis('off')


# Let us visualise the most common words

# In[ ]:


wc(cleaned_data_title,'black','Most Used Words')


# Let us try to perform analysis on the entire review rather than all the words.
# For this we make use of the TextBlob

# In[ ]:


from textblob import TextBlob

bloblist_desc = list()

df_review_str=test_df['Review'].astype(str)


# In[ ]:


for row in df_review_str:
    blob = TextBlob(row)
    bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
    df_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['Review','sentiment','polarity'])


# We will try to perform sentimental analysis and try to classify whether a review is a positive or negative

# In[ ]:


df_polarity_desc.head()


# In[ ]:


def f(df_polarity_desc):
    if df_polarity_desc['sentiment'] >= 0:
        val = "Positive Review"
    elif df_polarity_desc['sentiment'] >= -0.09:
        val = "Neutral Review"
    else:
        val = "Negative Review"
    return val


# After looking at the sentiments I have used the above values. This is a personal perference which you can change according to your choice.

# In[ ]:


df_polarity_desc['Sentiment_Type'] = df_polarity_desc.apply(f, axis=1)

plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.countplot(x="Sentiment_Type", data=df_polarity_desc)


# In[ ]:


positive_reviews=df_polarity_desc[df_polarity_desc['Sentiment_Type']=='Positive Review']
negative_reviews=df_polarity_desc[df_polarity_desc['Sentiment_Type']=='Negative Review']


# In[ ]:


negative_reviews.head()


# Let us look at the most used words in all the positive reviews

# In[ ]:


wc(positive_reviews['Review'],'black','Most Used Words')


# Let us look at the most used words in all the negative reviews

# In[ ]:


wc(negative_reviews['Review'],'black','Most Used Words')


# In[ ]:


import string
def text_process(review):
    nopunc=[word for word in review if word not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# Let us now train a model 
# We are taking only review with ratings 1 and 10 to perform the analysis to make the analysis more simple.

# In[ ]:


test_df=test_df.dropna(axis=0,how='any')
rating_class = test_df[(test_df['Rating'] == 1) | (test_df['Rating'] == 10)]
X_review=rating_class['Review']
y=rating_class['Rating']


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
bow_transformer=CountVectorizer(analyzer=text_process).fit(X_review)


# In[ ]:


X_review = bow_transformer.transform(X_review)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_review, y, test_size=0.3, random_state=101)


# Let us train a model using Multinomial Naive Bayes as it works great on text.

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)
predict=nb.predict(X_test)


# Let us find the accuracy, precision and recall

# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, predict))
print('\n Accuracy:')
print(accuracy_score(y_test, predict))
print(classification_report(y_test, predict))


# We have a 81 percent accuracy but a very recall score indicating that movies with ratings 1's
# are not classified properly this maybe due to a large ratio of positive review. 
# Hence it has a bad recall.
# 

# Let us test our model

# In[ ]:


rating_positive=test_df['Review'][6]
rating_positive


# In[ ]:


rating_postive_transformed = bow_transformer.transform([rating_positive])
nb.predict(rating_postive_transformed)[0]


# In[ ]:


rating_negative=test_df['Review'][54]
rating_negative


# In[ ]:


rating_negative_transformed = bow_transformer.transform([rating_negative])
nb.predict(rating_negative_transformed)[0]


# As we can see that we have a skewed data set and let us try to improve recall by performing undersampling

# In[ ]:


ratings_1 = (rating_class['Rating']==1).sum()
ratings_1_indices = np.array(rating_class[rating_class.Rating == 1].index)


# In[ ]:


ratings_10_indices = rating_class[rating_class.Rating == 10].index


random_normal_indices = np.random.choice(ratings_10_indices, ratings_1, replace = False)
random_normal_indices = np.array(random_normal_indices)

under_sample_indices = np.concatenate([ratings_1_indices,random_normal_indices])




# In[ ]:


undersample = rating_class.ix[under_sample_indices]

X_undersample = undersample.ix[:, undersample.columns != 'Rating']
y_undersample = undersample.ix[:, undersample.columns == 'Rating']


# In[ ]:


print("Percentage of 10 ratings: ", len(undersample[undersample.Rating == 10])/len(undersample))
print("Percentage of 1 ratings: ", len(undersample[undersample.Rating == 1])/len(undersample))
print("Total number of examples in resampled data: ", len(undersample))


# In[ ]:


X_review_us = X_undersample['Review']


# In[ ]:


X_review_us = bow_transformer.transform(X_review_us)


# In[ ]:


X_train_us, X_test_us, y_train_us, y_test_us = train_test_split(X_review_us, y_undersample, test_size=0.3, random_state=101)


# In[ ]:


nb.fit(X_train_us, y_train_us)
predict_us=nb.predict(X_test_us)


# In[ ]:


print(confusion_matrix(y_test_us, predict_us))
print('\n Accuracy:')
print(accuracy_score(y_test_us, predict_us))
print(classification_report(y_test_us, predict_us))


# We can see a reduced accuracy but we have increased the recall now let us try to use this model on the entire data

# In[ ]:


nb.fit(X_train_us, y_train_us)
predict_entire=nb.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test, predict_entire))
print('\n Accuracy:')
print(accuracy_score(y_test, predict_entire))
print(classification_report(y_test, predict_entire))


# Although, our overall accuracy has decreased. We have increased the recall to 90 percent.
# Hence our model can classify review with 1 rating better. <br>
# Comparing with the original model without undersampling

# In[ ]:


print(confusion_matrix(y_test, predict))
print('\n Accuracy:')
print(accuracy_score(y_test, predict))
print(classification_report(y_test, predict))


# In[ ]:




