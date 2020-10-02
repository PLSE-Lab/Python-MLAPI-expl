#!/usr/bin/env python
# coding: utf-8

# #### NLP stands for Natural Language Processing which is the task of mining the text and find out meaningful insights like Sentiments, Named Entity, Topics of Discussion and even Summary of the text.
# 
# #### With this dataset we will explore Sentiment Analaysis.
# 
# #### Since texts are free form, we need to apply a set of text cleaning techniques.
# 
# #### We can't apply texts to our ML model; we have to convert them in mathematical form and we will explore different techniques of Text Encoding.
# 
# #### I hope you will like it and please upvote!

# #### Import the basic libraries.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Load the dataset.

# In[ ]:


dataset = pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")


# #### Explore the dataset.

# In[ ]:


dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


dataset['sentiment'].value_counts()


# #### So we can see there are 2 columns - review and sentiment. sentiment is the target column that we need to predict. The dataset is completely balanced and it has equal number of positive and negative sentiments.

# #### Let's take one review as sample and understand why we need to clean the text.

# In[ ]:


review = dataset['review'].loc[1]
review


# #### Normally any NLP task involves following text cleaning techniques -
# 
# 1. Removal of HTML contents like "< br>".
# 2. Removal of punctutions, special characters like '\'.
# 3. Removal of stopwords like is, the which do not offer much insight.
# 4. Stemming/Lemmatization to bring back multiple forms of same word to their common root like 'coming', 'comes' into 'come'.
# 5. Vectorization - Encode the numeric values once you have cleaned it.
# 6. Fit the data to the ML model.
# 
# #### We will apply all these techniques on this sample review and understand how it works.
# 
# #### First of all we will remove HTML contents.

# In[ ]:


from bs4 import BeautifulSoup

soup = BeautifulSoup(review, "html.parser")
review = soup.get_text()
review


# #### We can see HTML tags are removed; so in the next step we will remove everything except lower/upper case letters using Regular Expressions.

# In[ ]:


import re

review = re.sub('\[[^]]*\]', ' ', review)
review = re.sub('[^a-zA-Z]', ' ', review)
review


# #### Next we will bring everything into lowercase.

# In[ ]:


review = review.lower()
review


# #### Stopwords removal - since stopwords removal works on every word in your text we need to split the text.

# In[ ]:


review = review.split()
review


# In[ ]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

review = [word for word in review if not word in set(stopwords.words('english'))]
review


# #### Stemming/Lemmatization - we will apply both and see the difference.

# In[ ]:


from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
review_s = [ps.stem(word) for word in review]
review_s


# In[ ]:


from nltk.stem import WordNetLemmatizer

lem = WordNetLemmatizer()
review = [lem.lemmatize(word) for word in review]
review


# #### We can see that 'little' has become 'littl' after Stemming but remained 'little' after Lemmatization. We will use Lemmatization.

# #### Merge the words to form cleaned up version of the text.

# In[ ]:


review = ' '.join(review)
review


# #### Our next step will be to bring this text in mathematical forms and to do so we will create a Corpus first.

# In[ ]:


corpus = []
corpus.append(review)


# #### To vectorize the text we will apply -
# 
# 1. CountVectorizer (Bag of Words Model)
# 2. TfidfVectorizer (Bag of Words Model)
# 3. Keras Tokenizer (Embedding)

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

count_vec = CountVectorizer()
review_count_vec = count_vec.fit_transform(corpus)

review_count_vec.toarray()


# #### So we can see the data has become numeric with 1,2 and 3s based on the number of times they appear in the text.
# 
# #### There is another variation of CountVectorizer with binary=True and in that case all zero entries will have 1.

# In[ ]:


count_vec_bin = CountVectorizer(binary=True)
review_count_vec_bin = count_vec_bin.fit_transform(corpus)

review_count_vec_bin.toarray()


# #### So there is no 2s and 3s in the vector.
# 
# #### We will now explore TF-IDF - TF stands for Text Frequency which means how many times a word (term) appears in a text (document). IDF means Inverse Document Frequency and is calculated as log(# of documents in corpus/# of documents containing the term).
# 
# #### Finally TF-IDF score is calculated as TF * IDF.
# 
# #### IDF acts as a balancing factor and diminishes the weight of terms that occur very frequently in the document set and increases the weight of terms that occur rarely.

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer()
review_tfidf_vec = tfidf_vec.fit_transform(corpus)

review_tfidf_vec.toarray()


# #### We will now apply all the techniques that we discussed on the whole dataset but there is no test dataset so we will keep 25% of the data aside to test the performance of the model.

# In[ ]:


from sklearn.model_selection import train_test_split

dataset_train, dataset_test, train_data_label, test_data_label = train_test_split(dataset['review'], dataset['sentiment'], test_size=0.25, random_state=42)


# #### Convert sentiments to numeric forms.

# In[ ]:


train_data_label = (train_data_label.replace({'positive': 1, 'negative': 0})).values
test_data_label  = (test_data_label.replace({'positive': 1, 'negative': 0})).values


# #### Clean the text and build the train and test corpus.

# In[ ]:


corpus_train = []
corpus_test  = []

for i in range(dataset_train.shape[0]):
    soup = BeautifulSoup(dataset_train.iloc[i], "html.parser")
    review = soup.get_text()
    review = re.sub('\[[^]]*\]', ' ', review)
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    lem = WordNetLemmatizer()
    review = [lem.lemmatize(word) for word in review]
    review = ' '.join(review)
    corpus_train.append(review)
    
for j in range(dataset_test.shape[0]):
    soup = BeautifulSoup(dataset_test.iloc[j], "html.parser")
    review = soup.get_text()
    review = re.sub('\[[^]]*\]', ' ', review)
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    lem = WordNetLemmatizer()
    review = [lem.lemmatize(word) for word in review]
    review = ' '.join(review)
    corpus_test.append(review)


# #### Let's validate one sample entry.

# In[ ]:


corpus_train[-1]


# In[ ]:


corpus_test[-1]


# #### We will now vectorize using TF-IDF technique.

# In[ ]:


tfidf_vec = TfidfVectorizer(ngram_range=(1, 3))

tfidf_vec_train = tfidf_vec.fit_transform(corpus_train)
tfidf_vec_test = tfidf_vec.transform(corpus_test)


# #### I am going to use LinearSVC as my first model.

# In[ ]:


from sklearn.svm import LinearSVC

linear_svc = LinearSVC(C=0.5, random_state=42)
linear_svc.fit(tfidf_vec_train, train_data_label)

predict = linear_svc.predict(tfidf_vec_test)


# #### Let's see the performance.

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Classification Report: \n", classification_report(test_data_label, predict,target_names=['Negative','Positive']))
print("Confusion Matrix: \n", confusion_matrix(test_data_label, predict))
print("Accuracy: \n", accuracy_score(test_data_label, predict))


# #### Now we will vectorize using CountVectorizer(binary=False) and fit it on LinearSVC model.

# In[ ]:


count_vec = CountVectorizer(ngram_range=(1, 3), binary=False)
count_vec_train = count_vec.fit_transform(corpus_train)
count_vec_test = count_vec.transform(corpus_test)


# In[ ]:


linear_svc_count = LinearSVC(C=0.5, random_state=42, max_iter=5000)
linear_svc_count.fit(count_vec_train, train_data_label)

predict_count = linear_svc_count.predict(count_vec_test)


# #### Let's measure its performance.

# In[ ]:


print("Classification Report: \n", classification_report(test_data_label, predict_count,target_names=['Negative','Positive']))
print("Confusion Matrix: \n", confusion_matrix(test_data_label, predict_count))
print("Accuracy: \n", accuracy_score(test_data_label, predict_count))


# #### Now we will vectorize using CountVectorizer(binary=True) and fit it on LinearSVC model.

# In[ ]:


ind_vec = CountVectorizer(ngram_range=(1, 3), binary=True)
ind_vec_train = ind_vec.fit_transform(corpus_train)
ind_vec_test = ind_vec.transform(corpus_test)


# In[ ]:


linear_svc_ind = LinearSVC(C=0.5, random_state=42)
linear_svc_ind.fit(ind_vec_train, train_data_label)

predict_ind = linear_svc_ind.predict(ind_vec_test)


# #### Let's measure its performance.

# In[ ]:


print("Classification Report: \n", classification_report(test_data_label, predict_ind,target_names=['Negative','Positive']))
print("Confusion Matrix: \n", confusion_matrix(test_data_label, predict_ind))
print("Accuracy: \n", accuracy_score(test_data_label, predict_ind))


# #### So we are getting maximum accuracy using TF-IDF vectorizer.

# #### We will now fit the data to Multinomial Naive Bayes classifier. Bayesian model uses prior probabilities to predict posterior probabilites which is helpful for classification with discrete features like text classification.

# In[ ]:


tfidf_vec_NB = TfidfVectorizer(ngram_range=(1, 1))
tfidf_vec_train_NB = tfidf_vec_NB.fit_transform(corpus_train)

tfidf_vec_test_NB = tfidf_vec_NB.transform(corpus_test)

print(tfidf_vec_train_NB.toarray().shape, tfidf_vec_test_NB.toarray().shape)


# #### So there are 81301 terms in the corpus and we will use a *Chi-Square* test to select top 50000 features.

# In[ ]:


from sklearn.feature_selection import SelectKBest, chi2

ch2 = SelectKBest(chi2, k=50000)
tfidf_vec_train_NB = ch2.fit_transform(tfidf_vec_train_NB, train_data_label)
tfidf_vec_test_NB  = ch2.transform(tfidf_vec_test_NB)


# #### We can see the top features as well.

# In[ ]:


feature_names = tfidf_vec_NB.get_feature_names()
feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
feature_names = np.asarray(feature_names)
feature_names[32245]


# #### Let's fit the data to Multinomial Naive Bayes model.

# In[ ]:


from sklearn.naive_bayes import MultinomialNB

multi_clf = MultinomialNB()
multi_clf.fit(tfidf_vec_train_NB, train_data_label)

predict_NB = multi_clf.predict(tfidf_vec_test_NB)


# #### Let's measure its performance.

# In[ ]:


print("Classification Report: \n", classification_report(test_data_label, predict_NB,target_names=['Negative','Positive']))
print("Confusion Matrix: \n", confusion_matrix(test_data_label, predict_NB))
print("Accuracy: \n", accuracy_score(test_data_label, predict_NB))


# #### The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work. We will now use CountVectorizer as our Bag-of-Words model before applying MultinomialNB model.

# In[ ]:


count_vec_NB = CountVectorizer(ngram_range=(1, 3), binary=False)
count_vec_train_NB = count_vec_NB.fit_transform(corpus_train)
count_vec_test_NB = count_vec_NB.transform(corpus_test)


# In[ ]:


multi_clf_count = MultinomialNB()
multi_clf_count.fit(count_vec_train_NB, train_data_label)

predict_NB_count = multi_clf_count.predict(count_vec_test_NB)


# In[ ]:


print("Classification Report: \n", classification_report(test_data_label, predict_NB_count,target_names=['Negative','Positive']))
print("Confusion Matrix: \n", confusion_matrix(test_data_label, predict_NB_count))
print("Accuracy: \n", accuracy_score(test_data_label, predict_NB_count))


# #### So LinearSVC using TF-IDF vectorization gives the maximum accuracy and we can see the outcome on our makeshift test dataset.

# In[ ]:


dataset_predict = dataset_test.copy()
dataset_predict = pd.DataFrame(dataset_predict)
dataset_predict.columns = ['review']
dataset_predict = dataset_predict.reset_index()
dataset_predict = dataset_predict.drop(['index'], axis=1)
dataset_predict.head()


# In[ ]:


test_actual_label = test_data_label.copy()
test_actual_label = pd.DataFrame(test_actual_label)
test_actual_label.columns = ['sentiment']
test_actual_label['sentiment'] = test_actual_label['sentiment'].replace({1: 'positive', 0: 'negative'})


# In[ ]:


test_predicted_label = predict.copy()
test_predicted_label = pd.DataFrame(test_predicted_label)
test_predicted_label.columns = ['predicted_sentiment']
test_predicted_label['predicted_sentiment'] = test_predicted_label['predicted_sentiment'].replace({1: 'positive', 0: 'negative'})


# In[ ]:


test_result = pd.concat([dataset_predict, test_actual_label, test_predicted_label], axis=1)
test_result.head()


# #### We can apply Recurrent Neural Networks like LSTM to perform sentiment analysis and we have a different vectorization technique called *Word Embeddings*.
# 
# #### Word embeddings give us a way to use an efficient, dense representation in which similar words have a similar encoding. Importantly, we do not have to specify this encoding by hand. An embedding is a dense vector of floating point values (the length of the vector is a parameter you specify). Instead of specifying the values for the embedding manually, they are trainable parameters (weights learned by the model during training, in the same way a model learns weights for a dense layer). It is common to see word embeddings that are 8-dimensional (for small datasets), up to 1024-dimensions when working with large datasets. A higher dimensional embedding can capture fine-grained relationships between words, but takes more data to learn.
# 
# Reference: https://www.tensorflow.org/tutorials/text/word_embeddings

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Masking, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# #### Let's understand how it works.

# In[ ]:


text = ['subha is a good boy', 'swati is a very good girl', 'yes...we will go home for sure', 'India"s gdp is less than USA.']


# #### Think this is our text which we need to vectorize.

# In[ ]:


text


# In[ ]:


tokenizer = Tokenizer(num_words=20)

tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
sequences


# #### So we can see we are able to tokenize the textual data.

# In[ ]:


tokenizer.word_index


# #### Now we can see that the 4 lines in the text have different length; so we need to standardize their length thru a technique called *Padding*.
# 
# #### padding='post' means zeroes will be appended after actual text and vice versa if padding='pre'.

# In[ ]:


data = pad_sequences(sequences, maxlen=8, padding='post')
data


# In[ ]:


word_index = tokenizer.word_index
len(word_index)


# In[ ]:


embedding = Embedding(input_dim=20, output_dim=4, mask_zero=True)
masked_output = embedding(data)

print(masked_output.numpy())


# In[ ]:


print("Shape before Embedding: \n", data.shape)
print("Shape after Embedding: \n", masked_output.shape)


# #### So we can see that the 4x8 array is converted to a 4x8x4 tensor.

# #### Let's proceed with the LSTM model.

# In[ ]:


max_features = 20000
maxlen = 200
tokenizer = Tokenizer(num_words=max_features)


# In[ ]:


train = pd.DataFrame(dataset_train)
train.columns = ['review']

test = pd.DataFrame(dataset_test)
test.columns = ['review']


# #### Apply the Tokenizer.

# In[ ]:


tokenizer.fit_on_texts(train['review'])
X_train_token = tokenizer.texts_to_sequences(train['review'])

tokenizer.fit_on_texts(test['review'])
X_test_token = tokenizer.texts_to_sequences(test['review'])


# #### Apply the Padding.

# In[ ]:


X_train = pad_sequences(X_train_token, maxlen=maxlen, padding='post')
X_test  = pad_sequences(X_test_token, maxlen=maxlen, padding='post')
print(X_train.shape, X_test.shape)


# In[ ]:


y_train = train_data_label.copy()
y_test  = test_data_label.copy()


# In[ ]:


model = Sequential([Embedding(max_features, 64, mask_zero=True),
                    Bidirectional(LSTM(64, dropout=0.2)),
                    Dense(64, activation='sigmoid'),
                    Dense(1)])


# In[ ]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train, y_train,
                    batch_size=50,
                    epochs=3,
                    validation_data=(X_test, y_test))


# #### Plotting model performance.

# In[ ]:


history.history


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'], '')
plt.xlabel("Epochs")
plt.ylabel('Accuracy')
plt.title('Change of Accuracy over Epochs')
plt.legend(['accuracy', 'val_accuracy'])
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], '')
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.title('Change of Loss over Epochs')
plt.legend(['loss', 'val_loss'])
plt.show()


# #### We can conclude that Bi-directional LSTM takes more time to train and is performing poorly compared to TF-IDF vectorization and Linear Classifier or Multinomial Naive Bayes Classifier.
