# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import operator
tqdm.pandas()


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
train = pd.read_csv("../input/train.csv")
# test = pd.read_csv("../input/test.csv")
# data_train, data_test, target_train, target_test = train_test_split(train["question_text"], train["target"],
#                                                                     test_size=0.33, random_state=42)
# # print(train.shape)
# # print(test.shape)
# # print(train.columns.values)
# training_data = list(data_train)
# training_data_labels = np.array(target_train)

# test_data = list(data_test)
# test_data_labels = np.array(target_test)

# # Performance of SVM classifier
# text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()),
#                          ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))])

# text_clf_svm = text_clf_svm.fit(training_data, training_data_labels)
# predicted_svm = text_clf_svm.predict(test_data)
# print("efficiency of Support Vector machine: ", np.mean(predicted_svm == test_data_labels))

# text_clf_rf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf-svm', RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0))])


# text_clf_rf = text_clf_rf.fit(training_data, training_data_labels)
# predicted_rf = text_clf_rf.predict(test_data)
# print("efficiency of Random Forest: ", np.mean(predicted_rf == test_data_labels))



# predicted_svm_new = text_clf_svm.predict(text)
# print("predicted label", predicted_svm_new)
# print(unique_skills[17])

# print(test.columns.values)
# dfList = test['question_text'].values[50101]
# print(dfList)
# insincere = train.loc[train['target'] == 1]
# print(train['question_text'].values[1306071])
# print(insincere.count)
# print(insincere['question_text'])
# import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#To track our training vocabulary, which goes through all our text and counts the occurance of the contained words.
def build_vocab(sentences, verbose=True):
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] =1
                
    return vocab
    
sentences = train["question_text"].progress_apply(lambda x: x.split()).values
vocab = build_vocab(sentences)
print({k: vocab[k] for k in list(vocab)[:20]})

# Next we import the embeddings we want to use in our model later.

from gensim.models import KeyedVectors

news_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)

def check_coverage(vocab, embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k +=vocab[word]
            
        except:
            oov[word] = vocab[word]
            i += vocab[word]
            pass
        
    print ('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print ('Found embeddings for {:.2%} of all text'.format(k / (k+i)))
    sorted_x = sorted(oov.items(), key = operator.itemgetter(1))[::-1]
    
    return sorted_x

oov = check_coverage(vocab,embeddings_index)











