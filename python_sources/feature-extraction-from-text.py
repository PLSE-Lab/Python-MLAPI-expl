#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np 
import pandas as pd
import spacy
import seaborn as sns
import string


# ### Logloss for this competition
# 

# In[ ]:


def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


# * https://spacy.io/
# spaCy is the best way to prepare text for deep learning. It interoperates seamlessly with TensorFlow, PyTorch, scikit-learn, Gensim and the rest of Python's awesome AI ecosystem. With spaCy, you can easily construct linguistically sophisticated statistical models for a variety of NLP problems.

# In[ ]:


nlp = spacy.load("en_core_web_sm")


# In[ ]:


#READING INPUT
data = pd.read_csv("/kaggle/input/spooky-author-identification/train.csv")
data.head()


# In[ ]:


test = pd.read_csv("/kaggle/input/spooky-author-identification/test.csv")


# # Tokenisation
# 
# * the process of breaking up the original text into components (tokens)

# In[ ]:


doc = nlp(data["text"][0])
for token in doc[0:5]:
    print(token.text, token.pos , token.pos_, token.dep_) # part of speach and syntax dependency


# # Lemmatisation
# 
# * in linguistics is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form.

# In[ ]:


for token in doc[0:5]:
    print(token.text,  token.pos_, token.lemma_) # part of speach and syntax dependency


# # target

# In[ ]:


sns.barplot(x=['Edgar Allen Poe', 'Mary Wollstonecraft Shelley', 'H.P. Lovecraft'], y=data['author'].value_counts())


# In[ ]:


data['author_num'] = data["author"].map({'EAP':0, 'HPL':1, 'MWS':2})
data.head()


# we map "EAP" to 0 "HPL" to 1 and "MWS" to 2 as it will be more convenient for our classifier. 
# In other words we are just telling our computer that if classifier predicts 0 for the text then it means that it is preicting "EAP", if 1 then it means that it is predicting "HPL", if 2 then it means that it is predicting "MWS".

# # Meta features
# 
# * features that are extracted from the text like number of words, number of stop words, number of punctuations etc Number of words in the text
# * Number of unique words in the text
# * Number of characters in the text
# * Number of stopwords
# * Number of punctuations
# * Number of upper case words
# * Number of title case words
# 

# In[ ]:


from nltk.corpus import stopwords
stopwords = stopwords.words('english')
print (stopwords)


# In[ ]:


## Number of words in the text ##
data["num_words"] = data["text"].apply(lambda x: len(str(x).split()))
test["num_words"] = test["text"].apply(lambda x: len(str(x).split()))
## Number of unique words in the text ##
data["num_unique_words"] = data["text"].apply(lambda x: len(set(str(x).split())))
test["num_unique_words"] = test["text"].apply(lambda x: len(set(str(x).split())))
## Number of characters in the text ##
data["num_chars"] = data["text"].apply(lambda x: len(str(x)))
test["num_chars"] = test["text"].apply(lambda x: len(str(x)))
## Number of stopwords in the text ##
data["num_stopwords"] = data["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords]))
test["num_stopwords"] = test["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords]))
## Number of punctuations in the text ##
data["num_punctuations"] =data['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test["num_punctuations"] =test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
## Number of title case words in the text ##
data["num_words_upper"] = data["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["num_words_upper"] = test["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
## Number of title case words in the text ##
data["num_words_title"] = data["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test["num_words_title"] = test["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
## Max length of the words in the text ##
data["max_word_len"] = data["text"].apply(lambda x: np.max([len(w) for w in str(x).split()]))
test["max_word_len"] = test["text"].apply(lambda x: np.max([len(w) for w in str(x).split()]))


# # Spacy
# 
# 

# In[ ]:




# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
def cleanup_text(docs, logging=False):
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in string.punctuation]
        #tokens = [tok for tok in tokens if tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)


# In[ ]:


print('Original training data shape: ', data['text'].shape)
data["text_cleaned"]= cleanup_text(data['text'], logging=True)
print('Cleaned up training data shape: ', data["text_cleaned"].shape)


# In[ ]:


print('Original training data shape: ', test['text'].shape)
test["text_cleaned"] = cleanup_text(test['text'], logging=True)
print('Cleaned up training data shape: ', test["text_cleaned"].shape)


# ## feature extraction on Spacy

# In[ ]:



data["num_unique_words_clenaed"] = data["text_cleaned"].apply(lambda x: len(set(str(x).split())))
test["num_unique_words_cleaned"] = test["text_cleaned"].apply(lambda x: len(set(str(x).split())))


# In[ ]:


def numberOfADV(docs, logging=False):
    numberOfADV = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.pos_ == 'ADP']
        #tokens = [tok for tok in tokens if tok not in stopwords and tok not in string.punctuation]
        #tokens = [tok for tok in tokens if tok not in punctuations]
        #tokens = ' '.join(tokens)
        
        numberOfADV.append(len(tokens))
    return pd.Series(numberOfADV)


# In[ ]:


data["num_of_ADV"] = numberOfADV(data['text_cleaned'], logging=True)


# In[ ]:


test["num_of_ADV"] = numberOfADV(test['text_cleaned'], logging=True)


# In[ ]:


def numberOfADJ(docs, logging=False):
    numberOfADV = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.pos_ == 'ADJ']
        
        numberOfADV.append(len(tokens))
    return pd.Series(numberOfADV)


# In[ ]:


data["num_of_ADJ"] = numberOfADJ(data['text_cleaned'], logging=True)


# In[ ]:


test["num_of_ADJ"] = numberOfADJ(test['text_cleaned'], logging=True)


# In[ ]:


data.head()


# In[ ]:


test.head()


# In[ ]:


sns.barplot(x= data["author"], y = data["num_of_ADJ"])


# In[ ]:


sns.barplot(x= data["author"], y = data["num_of_ADV"])


# ## Vectorisation

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


vect = CountVectorizer()


# In[ ]:


X_train_matrix = vect.fit_transform(data["text_cleaned"]) 
X_test_matrix = vect.transform(test["text_cleaned"]) 


# In[ ]:


features = vect.get_feature_names()
df_X_train_matrix = pd.DataFrame(X_train_matrix.toarray(), columns=features)
df_X_train_matrix.head()


# In[ ]:


df_X_test_matrix = pd.DataFrame(X_test_matrix.toarray(), columns=features)
df_X_test_matrix.head()


# ### Concatenate

# In[ ]:


data_df = data.drop(["id","text", "text_cleaned", "author"], axis = 1)

df_train = pd.concat([data_df, df_X_train_matrix], axis=1)

test_df = test.drop(["id","text", "text_cleaned"], axis = 1)

df_test = pd.concat([test_df, df_X_test_matrix], axis=1)


# In[ ]:


df_train.head()


# # X and y

# In[ ]:


X = df_train.drop("author_num", axis = 1)
y = data['author_num']


# ## Split training and test data

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify = y)


# ## Model

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))

print (clf.score(X_test, y_test))


# In[ ]:


predicted_result=clf.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predicted_result))


# In[ ]:


predictions = clf.predict_proba(X_test)

print ("logloss: %0.3f " % multiclass_logloss(y_test, predictions))


# # Submission

# In[ ]:


sample = pd.read_csv("/kaggle/input/spooky-author-identification/sample_submission.csv")
sample.head()


# In[ ]:



predicted_result = clf.predict_proba(df_test)


# In[ ]:


result=pd.DataFrame()
result["id"]=test["id"]
result["EAP"]=predicted_result[:,0]
result["HPL"]=predicted_result[:,1]
result["MWS"]=predicted_result[:,2]
result.head()


# In[ ]:


result.to_csv("submission_v3.csv", index=False)


# In[ ]:




