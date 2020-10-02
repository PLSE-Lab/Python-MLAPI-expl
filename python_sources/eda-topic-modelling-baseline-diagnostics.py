#!/usr/bin/env python
# coding: utf-8

# # Notebook Contents and Expectations
# 
# The notebook is aimed at understanding EDA, pre-processing, creating a baseline for sentiment classification. We'll understand where did the model go wrong and why. This is for people who know how to do fit() and predict(), but want to move beyond that.
# <br><br>
# Please follow the comments for code explanations and insights explanation
# 
# If you want to skip EDA, please jumpy to section 3.0-Modelling

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import random
import string
import nltk
from gensim import corpora, models
from wordsegment import load, segment
from nltk.tokenize.casual import TweetTokenizer
from itertools import chain, groupby
import scikitplot as skplt
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import re
from nltk import SnowballStemmer
from nltk.corpus import words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, plot_confusion_matrix
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # 1.0 Reading the Data

# In[ ]:


df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv') #reading the data


# ## 1.1 Super high-level summary

# In[ ]:


print("Columns in the data: ", list(df.columns))
print("Size of the data set: ", df.shape[0]) #df.shape = tuple(rows, columns)


# In[ ]:


print("Taking a gist of the first few rows:\n--------------------------------------\n")
df.head() #give the first 5 rows in the data


# In[ ]:


#counts of the predicted variable 'sentiment'
sentiment_counts = dict(df['sentiment'].value_counts()) #.value_counts() provides count of each unique value. dict() transforms
#it into {unique_value: counts} form

#making a barplot
fig = px.bar(
    x=list(sentiment_counts.keys()),
    y=list(sentiment_counts.values()),
)
fig.show()

#Neutral dominates. Positive and Negative are kinda equal in number. No harsh class imbalance though


# In[ ]:


#check if we have any empty rows
null_counts = df.isna().sum()
print("Null values counts column-wise\n------------------------------\n")
print(null_counts)
#There is one.let's remove that one row


# In[ ]:


df.dropna(inplace=True) #drops the row with empty columns. Inplace=True modifies df. if False, do df=df.dropna(inplace=False)


# In[ ]:


#Now check again if we have any empty rows. Just to make sure
null_counts = df.isna().sum()
print("Null values counts column-wise\n------------------------------\n")
print(null_counts)
print("New number of rows: ", df.shape[0])


# In[ ]:


#also, we understand that the 'selected_text' column is a phrase from the original sentence. Let's just verify it
for i in range(df.shape[0]):
    if i in df.index: #to avoid indexes of dropped row
        assert df.loc[i, 'selected_text'] in df.loc[i, 'text'] #checks if selected column is verbatim taken from text column


# ## 1.2 What do we know so far?
# 1. Our data has three classes - Neutral, Positive and Negative.
# 2. Neutral is the majority, although the imbalance is not too harsh
# 3. There was just one empty row, which is removed
# 4. The selected text is part of the main text body
# <br>
# 
# ## 1.3 Next steps?
# 1. Analyze the text vocabulary and sentences for all the three classes
# 2. Analyze the selection and the main text. How these two are related apart from being subset of the other

# # 2.0 EDA
# 
# ## 2.1 text column

# In[ ]:


#take a look at few sample sequences

def random_text(df, sentiment, n_samples):
    sub_df = df[df['sentiment']==sentiment]
    indexes = random.sample(list(sub_df.index), n_samples)
    print(df['text'][indexes])


# In[ ]:


#neutral sentiment samples
random_text(df, 'neutral', 5)


# In[ ]:


#negative sentiment samples
random_text(df, 'negative', 5)


# In[ ]:


#positive sentiment samples
random_text(df, 'positive', 5)


# ### Question: What words occur more commonly in each sentiment than the rest?

# In[ ]:


def filter_df(df, column, condition):
    '''
    returns list of values of a column based on condition
    example - df[df['some_column']==some_value]['column']
    
    params:
        - df: dataframe
        - column: column name in df (str)
        - condition: series of Boolean values
        
    returns:
        - list
    
    '''
    return list(df[condition][column])

def decontracted(phrase):
    '''
    Preprocessing step to replace 'won't' with 'will not' and such
    
    params:
        - phrase: a string
        
    returns:
        - transformed phrase (str)
    
    '''
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def preprocess(sentence, tokenizer, lemmatizer):
    '''
    Perform elementary preprocessing steps
    
    params:
        - sentence: string
        - tokenizer: a callable that takes a string and returns a list of tokens
        - lemmatizer: a callable that takes a word (string) and returns lemmatized word
        
    returns:
        -tokens: list of strings
    '''
    stopwords_en = stopwords.words('english')
    stopwords_en.remove('not')
    stopwords_en.remove('nor')

    sentence = sentence.lower()
    sentence = re.sub(r'http\S+'," ", sentence)
    sentence = re.sub(r'www\S+'," ", sentence)
    sentence = decontracted(sentence)
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    tokens = tokenizer(sentence)
    tokens = [token for token in tokens if token not in stopwords_en]
    return tokens


def generate_vocab(text_list, tokenizer, lemmatizer):
    '''
    Generates vocabulary based on a list of sentences
    
    params:
        - text_list: list of strings
        - tokenizer: a callable that takes a string and returns a list of tokens
        - lemmatizer: a callable that takes a word (string) and returns lemmatized word
    
    returns:
        vocab_counts: a dictionary with keys as words in the vocabulary and counts as number of times they occur in the text_list
    
    '''
    tokens_list = map(lambda x: preprocess(x, tokenizer, lemmatizer), text_list)
    vocab_counts = Counter(chain.from_iterable(tokens_list))
    return vocab_counts


# In[ ]:


#example
tokenizer = TweetTokenizer().tokenize #a callable
lemmatizer = WordNetLemmatizer().lemmatize # a callable

sentence = "I am telling you I really can't do that anymore!!"
print(preprocess(sentence, tokenizer, lemmatizer))


# In[ ]:


positive_text = filter_df(df, 'text', df['sentiment'] == 'positive')
postive_vocab_counts = generate_vocab(positive_text, tokenizer, lemmatizer)#vocab for positive sentiment labelled text

negative_text = filter_df(df, 'text', df['sentiment'] == 'negative')
negative_vocab_counts = generate_vocab(negative_text, tokenizer, lemmatizer) #vocab for negative sentiment labelled text

neutral_text = filter_df(df, 'text', df['sentiment'] == 'neutral')
neutral_vocab_counts = generate_vocab(neutral_text, tokenizer, lemmatizer)#vocab for neutral sentiment labelled text


# In[ ]:


def top_n(count_dict, n=20):
    '''
    Takes a count dictionary and returns n top words (keys) by count
    
    params:
        -count_dict: dict with str keys and int values
        
    returns:
        -None
    
    '''
    count_dict = sorted(count_dict.items(), key=lambda x:x[1], reverse=True)
    print("Word -----> Counts\n------------------------\n")
    for i in range(n):
        print(count_dict[i][0], "----->", count_dict[i][1])


# In[ ]:


top_n(postive_vocab_counts)


# In[ ]:


top_n(neutral_vocab_counts)


# In[ ]:


top_n(negative_vocab_counts)


# In[ ]:


#Is there any distinction among sentiment text with respect to number of words
sub_df = df[['text', 'sentiment']]
sub_df['lengths'] = sub_df['text'].apply(lambda x: len(tokenizer(x)))

fig = px.histogram(
        sub_df, 
        x="lengths", 
        color="sentiment",
        opacity=0.8,
        marginal="box",
        barmode="overlay"
      )

fig.show()
    
# distribution of sentences for each sentiment. Not anything useful here


# In[ ]:


#how many words in our data are actual english words (present in dictionary)?
word_set = set([lemmatizer(word) for word in words.words()])
def is_english(word_set, word):
    '''
    Checks if a word is in word_set
    
    params:
        -word_set - set() of strings
        -word -string
        
    returns:
        -bool
    '''
    return lemmatizer(word) in word_set

postive_en_vocab = [is_english(word_set, word.lower()) for word in postive_vocab_counts.keys()]
print("Proportion of non-english words in positive vocab: ", round(sum(postive_en_vocab)/len(postive_en_vocab),2))

negative_en_vocab = [is_english(word_set, word.lower()) for word in negative_vocab_counts.keys()]
print("Proportion of non-english words in negative vocab: ", round(sum(negative_en_vocab)/len(negative_en_vocab),2))

neutral_en_vocab = [is_english(word_set, word.lower()) for word in neutral_vocab_counts.keys()]
print("Proportion of non-english words in neutral vocab: ", round(sum(neutral_en_vocab)/len(neutral_en_vocab),2))


# In[ ]:


#Perform topic modelling to see what the data is talking about
tokens = list(map(lambda x: preprocess(x, tokenizer, lemmatizer), list(df['text'])))


# In[ ]:


dictionary_LDA = corpora.Dictionary(tokens)
dictionary_LDA.filter_extremes(no_below=3, no_above=1)
print("Size of dictionary:", len(dictionary_LDA.itervalues()))


# In[ ]:


NUM_TOPICS = 10
corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in tokens]
lda_model = models.LdaModel(
                    corpus, 
                    num_topics=NUM_TOPICS,
                    id2word=dictionary_LDA,
                    passes=4, alpha='auto',
                    eta='auto'
                )


# In[ ]:


#Looking at topic distribution for positive text
score_dict = {i:0 for i in range(NUM_TOPICS)}

indexes = list(df[df['sentiment'] == 'positive'].index)
sub_corpus = [corpus[i] for i in indexes if i<len(corpus)]
for seq in sub_corpus:
    scores = lda_model[seq]
    for topic, prob in scores:
        score_dict[topic] += prob/len(indexes)
        
fig = px.bar(
    x=list(score_dict.keys()),
    y=list(score_dict.values()),
)
fig.show()
    


# In[ ]:


#Looking at topic distribution for negative text
score_dict = {i:0 for i in range(NUM_TOPICS)}

indexes = list(df[df['sentiment'] == 'negative'].index)
sub_corpus = [corpus[i] for i in indexes if i<len(corpus)]
for seq in sub_corpus:
    scores = lda_model[seq]
    for topic, prob in scores:
        score_dict[topic] += prob/len(indexes)
        
fig = px.bar(
    x=list(score_dict.keys()),
    y=list(score_dict.values()),
)
fig.show()
    


# In[ ]:


#Looking at topic distribution for neutral text
score_dict = {i:0 for i in range(NUM_TOPICS)}

indexes = list(df[df['sentiment'] == 'neutral'].index)
sub_corpus = [corpus[i] for i in indexes if i<len(corpus)]
for seq in sub_corpus:
    scores = lda_model[seq]
    for topic, prob in scores:
        score_dict[topic] += prob/len(indexes)
        
fig = px.bar(
    x=list(score_dict.keys()),
    y=list(score_dict.values()),
)
fig.show()
    


# In[ ]:


lda_model.print_topics(num_words=10)


# In[ ]:


print("Model log perplexity is: ", lda_model.log_perplexity(corpus))


# ## What do we know?
# 
# 1. All the three sentiments have similar distribution for lengths
# 2. It's hard to differentiate topics based on sentiments. All sentiments discuss similar topics
# 3. Around 40-50% of the vocabulary is not even in English. Possibly slang words
# 
# 
# ## Next Steps
# 
# 1. Analyze selection and text together

# In[ ]:


#How does my selection length vary with my text length
tags_text = list(map(lambda x: nltk.pos_tag(tokenizer(x)), list(df['text'])))
tags_selection = list(map(lambda x: nltk.pos_tag(tokenizer(x)), list(df['selected_text'])))


# In[ ]:


lengths_text = [len(seq) for seq in tags_text]
lengths_selection = [len(seq) for seq in tags_selection]


# In[ ]:


fig = px.scatter(x=lengths_text, y=lengths_selection)
fig.show()


# In[ ]:


#I'm curious about examples where the whole text is the selected text
count=0
index = []
for i, (text, selection) in enumerate(zip(tags_text, tags_selection)):
    if i>100:
        break
    if len(text) == len(selection):
        index.append(i)
        count+=1
        print(len(text))
        print(" ".join([word for word,_ in text]))
        print(" ".join([word for word,_ in text]))
        print("------------------------------------------\n")
        


# In[ ]:


print("Proportion of such pairs is: ", round(count/len(tags_text),2))


# In[ ]:


sentiment_dict = dict(Counter(df.iloc[index,:]['sentiment']))

fig = px.bar(
    x=list(sentiment_dict.keys()),
    y=list(sentiment_dict.values()),
)
fig.show()


#Around 10k of the 11k neutral text have the same text and selected text.
# Majority positive and negative data have nothing of that sort. Only 1/4th data have exactly same text and selected text


# In[ ]:


pos_tags_sel = []
pos_tags_text = []
for i, (text, selection) in enumerate(zip(tags_text, tags_selection)):
    pos_tags_sel.extend([tag[0] for _, tag in selection if tag[0].isalnum()])
    pos_tags_text.extend([tag[0] for _, tag in text if tag[0].isalnum()])


# In[ ]:


#refer https://www.learntek.org/blog/categorizing-pos-tagging-nltk-python/ for POS tags and their meanings
pos_tag_dict = dict(Counter(pos_tags_text))

fig = px.bar(
    x=list(pos_tag_dict.keys()),
    y=list(pos_tag_dict.values()),
)
fig.show()

#


# In[ ]:


pos_select_dict = dict(Counter(pos_tags_sel))

pos_prop = {}

for key in pos_select_dict:
    pos_prop[key] = pos_select_dict[key]/pos_tag_dict[key]
    

fig = px.bar(
    x=list(pos_prop.keys()),
    y=list(pos_prop.values()),
)
fig.show()


# ## EDA Notes
# We have a good idea about our data. Let's summarize
# 
# 1. The sentiments are not associated to any topics. One cannot guess the sentiment based on the informtation about the topic it talks about
# 2. The selected text for neutral sentiment text is the whole text itself. Just by classifying a text as neutral, we can score well on the jaccard similarity between the text and selected text
# 
# ## Next Steps
# 
# 1. Create a baseline for classification task

# ## 3.0 Modelling

# In[ ]:


#start from scratch

df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')


df.dropna(inplace=True)
text = list(df['text'])


# In[ ]:


#same as functions above
def filter_df(df, column, condition):
    return list(df[condition][column])

def decontracted(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def remove_trails(word):
    '''
    Process words like wowwwwwwww to wow
    '''
    groups = groupby(word)
    result = [(label, sum(1 for _ in group)) for label, group in groups]
    new_string = ""
    for char, count in result:
        if count<3:
            new_string+=char*count
        else:
            new_string+=char
    return new_string


def preprocess(sentence, tokenizer, lemmatizer, stemmer):
    stopwords_en = stopwords.words('english')
    stopwords_en.remove('not')
    stopwords_en.remove('nor')
    sentence = sentence.lower()
    sentence = re.sub(r'http\S+'," ", sentence)
    sentence = re.sub(r'www\S+'," ", sentence)
    sentence = decontracted(sentence)
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    tokens = tokenizer(sentence)
    tokens = [remove_trails(stemmer(lemmatizer(token))) for token in tokens if token not in stopwords_en]
    
    return tokens


tokenizer = TweetTokenizer().tokenize #a callable
lemmatizer = WordNetLemmatizer().lemmatize # a callable
stemmer = SnowballStemmer('english').stem


# In[ ]:


tokens = list(map(lambda x: preprocess(x, tokenizer, lemmatizer, stemmer), text))


# In[ ]:


vocabulary_counts = Counter(chain.from_iterable(tokens))


# In[ ]:


count_lt3 = [key for key,_ in vocabulary_counts.items() if _<3]
print("Proportion of words with count < 3:", round(len(count_lt3)/len(vocabulary_counts),3))
print("------------------------------------------")
print(count_lt3[:100]) #remove [:100] to view all

#too many words with trailing sequence of letters like eeeeeee or oooooo


# In[ ]:


count_gt3 = [key for key,_ in vocabulary_counts.items() if _>=3]
print("Proportion of words with count > 3:", round(len(count_gt3)/len(vocabulary_counts),3))
print("------------------------------------------")
print(count_gt3[:100]) #remove [:100] to view all


# In[ ]:


#total counts by rare words and common words
total = sum(vocabulary_counts.values())

lt3 = sum(count for _, count in vocabulary_counts.items() if count<3)
gt3 = sum(count for _, count in vocabulary_counts.items() if count>=3)

print("Total counts for --")
print("Less than 3:", lt3)
print("Greater than 3:", gt3)
print("Total: ", total)


# In[ ]:


vocabulary_counts = Counter(chain.from_iterable(tokens))
count_lt3 = [key for key,_ in vocabulary_counts.items() if _<3]
print("Proportion of words with count < 3:", round(len(count_gt3)/len(vocabulary_counts),3))
print("------------------------------------------")
print(count_lt3[:100]) #remove [:100] to view all


# In[ ]:


MAX_FEATURES=int(0.27*len(vocabulary_counts)) #since 27% have counts greater than 3. Leaving rest of the vocab loses like 10% of total occurences

vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES
    )


sentences = [" ".join(token_list) for token_list in tokens]
transformed = vectorizer.fit_transform(sentences)
X = transformed.toarray()


# In[ ]:


print("X shape: ", X.shape)
# number of features ~= number of data point. Not very good


# In[ ]:


y = list(df['sentiment']) #encoding predicted valriable
encoder = LabelEncoder()
y = encoder.fit_transform(y)


# In[ ]:


#using pd.DataFrame to get indexes of data in train and test. Helps lookup the text column for test examples from the original df
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(pd.DataFrame(X), pd.DataFrame(y),test_size=0.2, shuffle=True)

X_train = X_train_df.to_numpy()
y_train = y_train_df.to_numpy()
X_test = X_test_df.to_numpy()
y_test = y_test_df.to_numpy()


# In[ ]:


model = LogisticRegression(max_iter=2000, solver='saga', penalty='l1', C=1)
model.fit(X_train,y_train)


# In[ ]:


y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
#test results


# In[ ]:


y_pred = model.predict(X_train)
print(classification_report(y_train, y_pred))
#train predict


# In[ ]:


import matplotlib.pyplot as plt

y_probas = model.predict_proba(X_train)# predicted probabilities generated by sklearn classifier
skplt.metrics.plot_roc_curve(y_train, y_probas)
plt.show()

#negative --> 0
#neutral --> 1
#positive --> 2


# In[ ]:


#Let's look at some missclassified cases

y_pred = model.predict(X_test)
incorrect = np.where(y_pred!=y_test.ravel())[0]


# In[ ]:


plot_confusion_matrix(model, X_test, y_test, display_labels=list(encoder.classes_))
#most of the missclassifications are between negative and neutral


# In[ ]:


#printing missclassified examples with their actual and predicted label
PRINT = 30
count=0
for i in incorrect:
    if count>PRINT:
        break
    count+=1
    index = int(X_test_df.index[i])
    print("-------------------------------------------")
    print(text[index], "----> Actual:", encoder.classes_[int(y_test[i])], "----> Predicted:", encoder.classes_[int(y_pred[i])])


# In[ ]:



#Fin.


# In[ ]:





# In[ ]:




