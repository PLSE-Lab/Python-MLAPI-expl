#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import nltk 
import string

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[ ]:


data = pd.read_csv("../input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv")
data = data.iloc[:,2:]
data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


filter1 = data['Rating'] >= 4
df1 = data[filter1]

filter2 = data['Rating'] < 4
df2 = data[filter2]


# In[ ]:


df1


# In[ ]:


text1 = df1['Review Text'].str.lower().str.strip().str.cat(sep = " ")
text1 = text1.translate(str.maketrans("","", string.punctuation))

text2 = df2['Review Text'].str.lower().str.strip().str.cat(sep = " ")
text2 = text2.translate(str.maketrans("","", string.punctuation))


# In[ ]:


words_rating_above4 = nltk.word_tokenize(text1)
words_rating_below4 = nltk.word_tokenize(text2)
stopwords = nltk.corpus.stopwords.words("english")

li1 = []
li2 = []

word_net_lemmatizer = WordNetLemmatizer()

for word in words_rating_above4: 
    if(word in stopwords):
        continue
    li1.append(word_net_lemmatizer.lemmatize(word, pos='v'))
    
for word in words_rating_below4: 
    if(word in stopwords):
        continue
    li2.append(word_net_lemmatizer.lemmatize(word, pos='v'))


# In[ ]:


freq1 = nltk.FreqDist(li1)

freq2 = nltk.FreqDist(li2)


# In[ ]:



fig = plt.figure(figsize = (20,10))
plt.gcf().subplots_adjust() # to avoid x-ticks cut-off
freq1.plot(100, cumulative=False)
plt.show()


# In[ ]:


fig = plt.figure(figsize = (20,10))
plt.gcf().subplots_adjust() # to avoid x-ticks cut-off
freq2.plot(100, cumulative=False)
plt.show()


# # Topics with rating above 4

# In[ ]:


df1=df1.fillna("")
df2=df2.fillna("")


# In[ ]:


vectorizer = CountVectorizer(stop_words= stopwords)
matrix = vectorizer.fit_transform(df1['Review Text'])
feature_names = np.array(vectorizer.get_feature_names())


# In[ ]:


lda = LatentDirichletAllocation(n_components=10, learning_method="batch",
                                max_iter=200, random_state=0) 
document_topics = lda.fit_transform(matrix)


# In[ ]:


# Set n to your desired number of tokens 
n = 8
# Find top n tokens
topics = dict()
for idx, component in enumerate(lda.components_): 
    top_n_indices = component.argsort()[:-(n + 1): -1] 
    topic_tokens = [feature_names[i] for i in top_n_indices] 
    topics[idx] = topic_tokens

topics


# # Topic modeling with rating below 4

# In[ ]:


vectorizer2 = CountVectorizer(stop_words= stopwords)
matrix2 = vectorizer2.fit_transform(df2['Review Text'])
feature_names2 = np.array(vectorizer2.get_feature_names())


# In[ ]:


lda2 = LatentDirichletAllocation(n_components=10, learning_method="batch",
                                max_iter=200, random_state=0) 
document_topics2 = lda2.fit_transform(matrix2)


# In[ ]:


# Set n to your desired number of tokens 
n = 8
# Find top n tokens
topics2 = dict()
for idx, component in enumerate(lda2.components_): 
    top_n_indices2 = component.argsort()[:-(n + 1): -1] 
    topic_tokens2 = [feature_names2[i] for i in top_n_indices2] 
    topics2[idx] = topic_tokens2

topics2


# # Creating a Classification model

# In[ ]:


data.head()


# In[ ]:


data1 = data.drop(['Age', 'Positive Feedback Count', 'Division Name', 'Department Name'], axis = 1)

data1.head()


# In[ ]:


data1['Rating'] = data1['Rating'].apply(lambda x: 1 if(x>=4) else 0)


# In[ ]:


X = data1.drop(['Rating'], axis = 1)
y = data1['Rating']


# In[ ]:



X['Title'] = X['Title'].apply(lambda x: str(x))
X['Review Text'] = X['Review Text'].apply(lambda x: str(x))
X.head()


# In[ ]:


X['Title_Review'] = X['Title'] + " " + X['Review Text']
X['Title_Review'] = X['Title_Review'].apply(lambda x: x.replace("nan","").strip())
X = X.drop(['Review Text', "Title"], axis = 1)
X.head()


# In[ ]:


from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

word_net_lemmatizer = WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

stopwords1 = STOP_WORDS
st2 = string.digits+string.punctuation

def lemmatize_text(text):
    lis=[]
    words = text.split()
    for word in words:
        if(word in stopwords1):
            continue
        lis.append(word_net_lemmatizer.lemmatize(word, pos='v'))
    return lis


X['Title_Review'] = X['Title_Review'].apply(lambda x: " ".join(lemmatize_text(x)))
X['Title_Review'] = X['Title_Review'].apply(lambda x: x.translate(str.maketrans("","", st2)))
X['Title_Review'] = X['Title_Review'].apply(lambda x: x.lower())


# In[ ]:


from textblob import TextBlob

X['Sentiment_Polarity'] = X['Title_Review'].apply(lambda x: TextBlob(x).sentiment[0])
X['Sentiment_Subjectivity'] = X['Title_Review'].apply(lambda x: TextBlob(x).sentiment[1])


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(stop_words='english')
df_text1 = vect.fit_transform(X['Title_Review'])
df_text1 = pd.DataFrame(df_text1.toarray(), columns= vect.get_feature_names())


# In[ ]:


X1 = X.drop(['Title_Review'], axis = 1)
X1 = pd.concat([X1, df_text1], axis = 1)

X1 = pd.get_dummies(columns = ['Class Name'], data = X1)
X1 = X1.iloc[:, :-1]
X1


# # 1st model with ratings as target variable

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X1, y, random_state = 0, test_size = 0.3)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("accuracy score is: {}".format(accuracy_score(y_test, y_pred)))


# In[ ]:


print("Confusion matrix is:")
print((confusion_matrix(y_test, y_pred)))
print("Classification report is:")
print((classification_report(y_test, y_pred)))


# # 2nd model with Recommended as target variable

# In[ ]:


X2 = X1.copy()
X2['Rating'] = data['Rating']
y2 = data['Recommended IND']
X2= X2.drop(['Recommended IND'], axis = 1)
X2


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X2, y2, random_state = 0, test_size = 0.3)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("accuracy score is: {}".format(accuracy_score(y_test, y_pred)))


# In[ ]:


print("Confusion matrix is:")
print((confusion_matrix(y_test, y_pred)))
print("Classification report is:")
print((classification_report(y_test, y_pred)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




