#!/usr/bin/env python
# coding: utf-8

# # Studycheck dataset. 
# 
# Download it from https://www.kaggle.com/longnguyen2306/germany-universities-reviews-and-recommendation

# In[ ]:


import numpy as np
import pandas as pd
import warnings
import pandarallel
import nltk

pandarallel.pandarallel.initialize()

warnings.filterwarnings('ignore')

df = pd.read_csv('study/studycheck.csv')

del df["url"]

df.head()


# In[ ]:


df.info()


# In[ ]:


len(df)


# # Divide the dataset into positive and negative reviews

# In[ ]:


import matplotlib.pyplot as plt
from collections import Counter

df_positive = df[df["weiter_empfehlung"] == True]
df_negative = df[df["weiter_empfehlung"] == False]

counter = Counter(df['weiter_empfehlung'])
plt.bar(counter.keys(), counter.values())
plt.show()


# In[ ]:


positive_corpus = df_positive["inhalt"].str.cat(sep = '\n')
positive_corpus[0:1000]


# In[ ]:


len(positive_corpus)


# In[ ]:


negative_corpus = df_negative["inhalt"].str.cat(sep = '\n')
negative_corpus[0:1000]


# # Quantivative comparision of negative and negative reviews.

# In[ ]:


plt.bar([0, 1], [len(negative_corpus), len(positive_corpus)])
plt.show()


# In[ ]:


positive_text = nltk.Text(positive_corpus)
negative_text = nltk.Text(negative_corpus)
positive_text, negative_text


# # Lexical diversity comparision between negative and positive. Negative reviews tend to use richer text

# In[ ]:


def lexical_diversity(text):
    return len(text) / len(set(text))

negative_diversity = lexical_diversity(negative_text)
positive_diversity = lexical_diversity(positive_text)

plt.bar([0, 1], [negative_diversity, positive_diversity])


# In[ ]:


from nltk.corpus import stopwords

stop_words = set(stopwords.words('german'))


# In[ ]:


from nltk.probability import FreqDist


# # Warning: Don't run the following cells. They are incredible slow so I installed joblib to accelerate the code

# In[ ]:


negative_tokens = nltk.word_tokenize(negative_corpus)
negative_tokens = [x for x in negative_tokens if x not in stop_words]
negative_tokens = [x for x in negative_tokens if len(x) > 5]
negative_freq = FreqDist(negative_tokens)
negative_freq


# In[ ]:


positive_tokens = nltk.word_tokenize(positive_corpus)
positive_tokens = [x for x in positive_tokens if x not in stop_words]
positive_tokens = [x for x in positive_tokens if len(x) > 5]
positive_freq = FreqDist(positive_tokens)

positive_freq


# In[ ]:


plt.figure(figsize=(20, 12))
negative_freq.plot(25)
plt.show()


# In[ ]:


plt.figure(figsize=(20, 12))
positive_freq.plot(25)
plt.show()


# In[ ]:


get_ipython().system('pip install joblib')


# # Comparing sequential word tokenizer with parallel word tokenizer

# In[ ]:


from nltk import word_tokenize

negative_tokens = word_tokenize(negative_corpus)


# In[ ]:


from joblib import Parallel, delayed

negative_tokens = Parallel(n_jobs=24)(delayed(word_tokenize)(line) for line in negative_corpus.split("\n"))


# # For the negative dataset, sequential takes twice amout of time what joblib can do with 24 cores processor

# In[ ]:


import itertools

negative_tokens = list(itertools.chain(*negative_tokens))

negative_tokens = [x for x in negative_tokens if x not in stop_words]
negative_tokens = [x for x in negative_tokens if len(x) > 5]

negative_tokens[:10]


# In[ ]:


plt.figure(figsize=(20, 12))
negative_freq = FreqDist(negative_tokens)
negative_freq.plot(50)
plt.show()


# # The bigger the dataset, the better joblib can utilize multiple cores. 

# In[ ]:


positive_tokens = word_tokenize(positive_corpus)


# In[ ]:


positive_tokens = Parallel(n_jobs=24)(delayed(word_tokenize)(line) for line in positive_corpus.split("\n"))


# In[ ]:


positive_tokens = list(itertools.chain(*positive_tokens))


positive_tokens = [x for x in positive_tokens if x not in stop_words]
positive_tokens = [x for x in positive_tokens if len(x) > 5]

positive_tokens[:10]


# In[ ]:


plt.figure(figsize=(20, 12))
positive_freq = FreqDist(positive_tokens)
positive_freq.plot(50)
plt.show()


# # Same process but with word stemming

# In[ ]:


from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("german")

positive_tokens = [stemmer.stem(x) for x in positive_tokens]

plt.figure(figsize=(20, 12))

positive_freq = FreqDist(positive_tokens)

positive_freq.plot(50)

plt.show()


# In[ ]:


negative_tokens = [stemmer.stem(x) for x in negative_tokens]

plt.figure(figsize=(20, 12))

negative_freq = FreqDist(negative_tokens)

negative_freq.plot(50)

plt.show()


# # I want to see how many words each reviews has in average

# In[ ]:


def count_words(text):
    return len(text.split(" "))

df["word_count"] = df["inhalt"].parallel_apply(count_words)
df.head()


# # Pretty many outliers. 

# In[ ]:


import seaborn as sns
import numpy as np

plt.figure(figsize=(20, 10))
sns.distplot(df['word_count'])
plt.title(f"Range from {np.min(df['word_count'])} to {np.max(df['word_count'])}")
plt.plot()


# In[ ]:


df_positive = df[df["weiter_empfehlung"] == True]

plt.figure(figsize=(20, 10))
sns.distplot(df_positive['word_count'])
plt.title(f"Range from {np.min(df_positive['word_count'])} to {np.max(df_positive['word_count'])}")
plt.plot()


# In[ ]:


df_negative = df[df["weiter_empfehlung"] == False]

plt.figure(figsize=(20, 10))
sns.distplot(df_negative['word_count'])
plt.title(f"Range from {np.min(df_negative['word_count'])} to {np.max(df_negative['word_count'])}")
plt.plot()


# In[ ]:


df_without_outlier = df[df['word_count'] < 301]
plt.figure(figsize=(20, 10))
sns.distplot(df_without_outlier['word_count'])
plt.title(f"Range from {np.min(df_without_outlier['word_count'])} to {np.max(df_without_outlier['word_count'])}")
plt.plot()


# In[ ]:


df_without_outlier_positive = df_without_outlier[df_without_outlier["weiter_empfehlung"] == True]
plt.figure(figsize=(20, 10))
sns.distplot(df_without_outlier_positive['word_count'])

min_range = np.min(df_without_outlier_positive['word_count'])
max_range = np.max(df_without_outlier_positive['word_count'])
standard_deviation = np.std(df_without_outlier_positive['word_count'])

plt.title(f"Range from {min_range} to {max_range} with a standard deviation of {standard_deviation}")
plt.plot()


# In[ ]:


df_without_outlier_negative = df_without_outlier[df_without_outlier["weiter_empfehlung"] == False]
plt.figure(figsize=(20, 10))
sns.distplot(df_without_outlier_negative['word_count'])

min_range = np.min(df_without_outlier_negative['word_count'])
max_range = np.max(df_without_outlier_negative['word_count'])
standard_deviation = np.std(df_without_outlier_negative['word_count'])

plt.title(f"Range from {min_range} to {max_range} with a standard deviation of {standard_deviation}")
plt.plot()


# # There is also something I want to discover. How often does the word "ich" appear in each review? (Which means "I" in german)

# In[ ]:


def count_ich(text: str):
    ret = 0
    stems = ["ich", "mir", "mich"]
    for stem in stems:
        appearances = str(text).lower().split(" ").count(stem)
        ret += appearances
    return ret

df["ich_appearance"] = df["inhalt"].parallel_apply(count_ich)
df.head()


# In[ ]:


plt.figure(figsize=(20, 12))
sns.distplot(df["ich_appearance"])
plt.show()


# # Is there a correlation between count of "I" and length of the text? Well, seems like that

# In[ ]:


df["word_count"].corr(df["ich_appearance"], method="pearson")


# In[ ]:


df["word_count"].corr(df["ich_appearance"], method="kendall")


# In[ ]:


df["word_count"].corr(df["ich_appearance"], method="spearman")


# In[ ]:


threshold = 30
df_many_ich= df[df["ich_appearance"] > threshold]
print(f"There are about {len(df_many_ich)} entries with {threshold} or more appearances of ich")
df_many_ich.head()


# # In order the make the graph more readable, let's filter out the outliers

# In[ ]:


df["ich_appearance"].quantile(0.99)


# In[ ]:


threshold = 9
df_many_ich= df[df["ich_appearance"] <= threshold]
sns.distplot(df_many_ich["ich_appearance"])
plt.show()


# In[ ]:


df_many_ich_positive = df_many_ich[df_many_ich["weiter_empfehlung"] == True]
sns.distplot(df_many_ich["ich_appearance"])
plt.show()


# In[ ]:


df_many_ich_positive = df_many_ich[df_many_ich["weiter_empfehlung"] == False]
sns.distplot(df_many_ich["ich_appearance"])
plt.show()


# # In conclusion, my thesis is wrong. I wanted to know if students with negative reviews are more egocentric.

# In[ ]:


data = df["inhalt"]
label = np.array(df["weiter_empfehlung"]).astype(np.int32)


# In[ ]:


from nltk.corpus import stopwords

stop_words = set(stopwords.words('german'))
def remove_stopwords(text):
    tokens = text.split(" ")
    tokens = [x.lower() for x in tokens if x not in stop_words]
    return " ".join(tokens)
data = data.parallel_apply(remove_stopwords)


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, x_test, Y_train, y_test = train_test_split(data, label)

X_train.shape, x_test.shape, Y_train.shape, y_test.shape


# # TfIdf Matrix

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfv = TfidfVectorizer() 

X_train = tfv.fit_transform(X_train)

x_test = tfv.transform(x_test)

X_train.shape, x_test.shape


# # Imbalance data, the good old problem. The accuracy of logistic regression of unbalanced data is terrible

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

lr = LogisticRegression()

lr.fit(X_train, Y_train)

print(classification_report(lr.predict(x_test), y_test))


# # LoL, Naives Bayes doesn't even care to train at all

# In[ ]:


from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train, Y_train)

print(classification_report(nb.predict(x_test), y_test))


# # Let's undersamling the positive dataset so we can have more balanced training and testing data

# In[ ]:


df_positive.head()


# In[ ]:


df_negative.head()


# In[ ]:


df_positive_sampled = df_positive.sample(n=len(df_negative), random_state=42)
print(len(df_positive_sampled))
df_positive_sampled.head()


# In[ ]:


balanced_df = pd.concat([df_negative, df_positive_sampled])
print(len(balanced_df))
balanced_df.head()


# In[ ]:


data = balanced_df["inhalt"]
target = balanced_df["weiter_empfehlung"]


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, x_test, Y_train, y_test = train_test_split(data, target, test_size=0.15)

X_train.shape, x_test.shape, Y_train.shape, y_test.shape


# # Make a pipeline so we don't have to repeat the whole thing again and again

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

class DataTransformer(TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('german'))
        
    def remove_stopwords(self, text):
        tokens = text.split(" ")
        tokens = [x.lower() for x in tokens if x not in stop_words]
        return " ".join(tokens)
    
    def fit(self, X,  y=None, **kwargs):
        return self
        
    def transform(self, X,  y=None, **kwargs):
        return X.parallel_apply(self.remove_stopwords)

class TfIdfTransformer(TransformerMixin):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
    
    def fit(self, X, y=None, **kwargs):
        self.tfidf.fit(X)
        return self
    
    def transform(self, X,  y=None, **kwargs):
        return self.tfidf.transform(X)
        

class TargetTransformer(TransformerMixin):
    def fit(self, X, y=None, **kwargs):
        return self
    
    def transform(self, X, y=None, **kwargs):
        return np.array(y).astype(np.int32)
    
data_transformer = DataTransformer()
tfidf_transformer = TfIdfTransformer()
target_transformer = TargetTransformer()

X_train = data_transformer.fit_transform(X_train)
x_test = data_transformer.fit_transform(x_test)
X_train = tfidf_transformer.fit_transform(X_train)
x_test = tfidf_transformer.transform(x_test)
Y_train = target_transformer.transform(X=X_train, y=Y_train)
y_test = target_transformer.transform(X=x_test, y=y_test)

X_train.shape, Y_train.shape, x_test.shape, y_test.shape


# # Thanks god, it ain't that hard, is it?

# In[ ]:


lr = LogisticRegression()

lr.fit(X_train, Y_train)

print(classification_report(lr.predict(x_test), y_test))


# In[ ]:


nb = MultinomialNB()

nb.fit(X_train, Y_train)

print(classification_report(nb.predict(x_test), y_test))


# # We can apply some other algorithms to see how the model works

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train, Y_train)

print(classification_report(rf.predict(x_test), y_test))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()

gb.fit(X_train, Y_train)

print(classification_report(gb.predict(x_test), y_test))


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

ab = AdaBoostClassifier()

ab.fit(X_train, Y_train)

print(classification_report(ab.predict(x_test), y_test))


# In[ ]:


from sklearn.ensemble import BaggingClassifier

bc = BaggingClassifier()

bc.fit(X_train, Y_train)

print(classification_report(bc.predict(x_test), y_test))


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier()

et.fit(X_train, Y_train)

print(classification_report(et.predict(x_test), y_test))


# In[ ]:


from sklearn.ensemble import VotingClassifier

vc = VotingClassifier(estimators=[
    ("lr", lr),
    ("rf", rf),
    ("nb", nb),
    ("gb", gb),
    ("ab", ab),
])

vc.fit(X_train, Y_train)

print(classification_report(vc.predict(x_test), y_test))


# # Can not seem to get out of 80%. Let's try some neural network magic. For this purpose I will restart the notebook and read the data again.

# In[ ]:


from nltk.corpus import stopwords
import pandas as pd

df = pd.read_csv('study/studycheck.csv') 

stop_words = set(stopwords.words('german'))
def remove_stopwords(text):
    tokens = text.split(" ")
    tokens = [x.lower().replace(",", "").replace(".", "").replace(":", "") for x in tokens if x not in stop_words]
    return " ".join(tokens)

df_positive = df[df["weiter_empfehlung"] == True]

df_negative = df[df["weiter_empfehlung"] == False]

df_positive_sampled = df_positive.sample(n=len(df_negative), random_state=42)

balanced_df = pd.concat([df_negative, df_positive_sampled])

balanced_df["inhalt"] = balanced_df["inhalt"].parallel_apply(remove_stopwords)

balanced_df.head()


# In[ ]:


from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(balanced_df, 
                                     test_size=0.15,
                                     random_state=42)

len(df_train), len(df_test)


# In[ ]:


documents = [review.split() for review in df_train.inhalt] 
documents[:1]


# In[ ]:


import gensim
from gensim.models.word2vec import Word2Vec


W2V_SIZE = 300
W2V_WINDOW = 10
W2V_MIN_COUNT = 5
SEQUENCE_LENGTH = 300
w2v_model = Word2Vec(size=W2V_SIZE, 
                     window=W2V_WINDOW, 
                     min_count=W2V_MIN_COUNT, 
                     workers=24)

w2v_model.build_vocab(documents)


# In[ ]:


words = w2v_model.wv.vocab.keys()
len(words)


# In[ ]:


W2V_EPOCH = 20

w2v_model.train(documents, 
                total_examples=len(documents), 
                epochs=W2V_EPOCH)


# In[ ]:


w2v_model.most_similar("schlecht")


# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train.inhalt)

vocab_size = len(tokenizer.word_index)

print("Total words", vocab_size)


# In[ ]:


from itertools import islice

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

for k, v in take(10, tokenizer.word_index.items()):
    print(k, v)


# In[ ]:


from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(
    tokenizer.texts_to_sequences(df_train.inhalt), 
    maxlen=SEQUENCE_LENGTH)

x_test = pad_sequences(
    tokenizer.texts_to_sequences(df_test.inhalt), 
    maxlen=SEQUENCE_LENGTH)

x_train[0]


# In[ ]:


tokenizer.sequences_to_texts(x_train)[0]


# In[ ]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(df_train.weiter_empfehlung.tolist())

y_train = encoder.transform(df_train.weiter_empfehlung.tolist())
y_test = encoder.transform(df_test.weiter_empfehlung.tolist())

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("y_train",y_train.shape)
print("y_test",y_test.shape)


# In[ ]:


embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
embedding_matrix


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM

embedding_layer = Embedding(vocab_size, 
                            W2V_SIZE, 
                            weights=[embedding_matrix], 
                            input_length=SEQUENCE_LENGTH, 
                            trainable=False)

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

model.summary()


# In[ ]:


EPOCHS = 5
BATCH_SIZE = 64

history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS)


# In[ ]:


import pickle

model.save('study/model.h5')
w2v_model.save('study/w2v_model.w2v')
pickle.dump(tokenizer, open("study/tokenizer.pkl", "wb"), protocol=0)
pickle.dump(encoder, open("study/encoder.pkl", "wb"), protocol=0)


# In[ ]:


score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print("ACCURACY:",score[1])
print("LOSS:",score[0])


# In[ ]:


def predict(text, ):
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    score = model.predict([x_test])[0]
    return score

predict("Informatik ist schon ganz gut.")


# In[ ]:


get_ipython().system('shutdown now')

