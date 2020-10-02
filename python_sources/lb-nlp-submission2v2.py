# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import string
from sklearn.linear_model import SGDClassifier
import gc
from gensim.models import KeyedVectors
from wordcloud import STOPWORDS
import zipfile

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

def find_best_threshold_result(y_target, y_predicted, score=f1_score, thresholds=np.arange(.01, .81, .01)):
    threshold_scores = np.array([(threshold, score(y_target, y_predicted[:, 1] > threshold)) 
                                 for threshold in thresholds])
    best = threshold_scores[np.argmax(threshold_scores[:, 1])]
    return best

def print_results(y_target, y_predicted, score=f1_score, thresholds=np.arange(.01, .81, .01)):
    roc_auc = roc_auc_score(y_target, y_predicted[:, 1])
    best = find_best_threshold_result(y_target, y_predicted, score, thresholds)
    clf_report = classification_report(y_target, y_predicted[:, 1] > best[0])
    
    print("ROC_AUC: ", roc_auc)
    print("Threshold: ", best[0])
    print("Best score: ", best[1])
    print(clf_report)
    
    return best


train_df = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")
test_df = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")

print("Evaluating additions features")
words_counter = lambda x: len(x.split()) 
unique_counter = lambda x: len(set(x.split()))
char_counter = len
stopwords_counter = lambda x: len([w for w in x.lower().split() if w in STOPWORDS])
punctuation_counter = lambda x: len([c for c in x if c in string.punctuation])
uppers_counter = lambda x: len([w for w in x.split() if w.isupper()])
title_words_counter = lambda x: len([w for w in x.split() if w.istitle()])
mean_word_len_counter = lambda x: np.mean([len(w) for w in x.split()])

train_df["n_words"] = train_df.question_text.map(words_counter)
train_df["n_words_unique"] = train_df.question_text.map(unique_counter)
train_df["n_chars"] = train_df.question_text.map(char_counter)
train_df["n_stopwords"] = train_df.question_text.map(stopwords_counter)
train_df["n_punctuations"] = train_df.question_text.map(punctuation_counter)
train_df["n_words_upper"] = train_df.question_text.map(uppers_counter)
train_df["n_words_title"] = train_df.question_text.map(title_words_counter)
train_df["mean_word_len"] = train_df.question_text.map(mean_word_len_counter)

test_df["n_words"] = test_df.question_text.map(words_counter)
test_df["n_words_unique"] = test_df.question_text.map(unique_counter)
test_df["n_chars"] = test_df.question_text.map(char_counter)
test_df["n_stopwords"] = test_df.question_text.map(stopwords_counter)
test_df["n_punctuations"] = test_df.question_text.map(punctuation_counter)
test_df["n_words_upper"] = test_df.question_text.map(uppers_counter)
test_df["n_words_title"] = test_df.question_text.map(title_words_counter)
test_df["mean_word_len"] = test_df.question_text.map(mean_word_len_counter)

additional_features = ["n_words", "n_words_unique", "n_chars", 
                       "n_stopwords", "n_punctuations", "n_words_upper", 
                       "n_words_title", "mean_word_len"]
F_train = train_df.loc[:, additional_features]
F_test = test_df.loc[:, additional_features]

print("Scaling additional features")
scaler = StandardScaler()
scaler.fit(F_train)
F_train = scaler.transform(F_train)
F_test = scaler.transform(F_test)

y_train = train_df.target.values

print("Using TfidfVectorizer")
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.91, min_df=2)
X_train = vectorizer.fit_transform(train_df.question_text.values).astype(np.float32)
X_test = vectorizer.transform(test_df.question_text.values).astype(np.float32)

print("Train shape: ", X_train.shape)
print("Test shape: ", X_test.shape)

print("Stacking tfidf with additional features")
X_train = sparse.hstack((F_train, X_train), format='csr', dtype='float32')
X_test = sparse.hstack((F_test, X_test), format='csr', dtype='float32')

n_splits = 3
cv = StratifiedKFold(n_splits, random_state=21, shuffle=True)

sgd = SGDClassifier(alpha=5*10**-7, loss="log", penalty="elasticnet", l1_ratio=.29, 
                     class_weight="balanced", random_state=21, n_jobs=-1)

print("Evaluationg predictions part 1")
test_preds1 = []
train_preds1 = np.zeros(train_df.shape[0])
for train_idx, valid_idx in cv.split(X_train, y_train):
    sgd.fit(X_train[train_idx], y_train[train_idx])
    train_preds1[valid_idx] = sgd.predict_proba(X_train[valid_idx])[:, 1]
    test_preds1.append(sgd.predict_proba(X_test)[:, 1])
    
test_preds1 = np.average(test_preds1, axis=0)

del vectorizer
del X_train
del X_test
del sgd
del F_train
del F_test
gc.collect()

print("Starting w2v part")

with zipfile.ZipFile("../input/quora-insincere-questions-classification/embeddings.zip","r") as z:
    z.extract("GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin")
    
w2v = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)
tokenizer = TfidfVectorizer().build_tokenizer()

def build_text_vector(text, tokenizer):
    v = []
    for token in tokenizer(text.lower()):
        try:
            v.append(w2v[token])
        except KeyError:
            pass
    
    if len(v) != 0:
        w = np.average(v, axis=0)
        w = w / np.sqrt(w.dot(w))
    else:
        w = np.zeros(w2v.vector_size)
        
    return w

print("Building mean doc vectors")
X_train = np.stack(train_df.question_text.map(lambda x: build_text_vector(x, tokenizer)).values.ravel())
X_test = np.stack(test_df.question_text.map(lambda x: build_text_vector(x, tokenizer)).values.ravel())
print("Train shape: ", X_train.shape)
print("Test shape: ", X_test.shape)

print("Deleting w2v")
del w2v
gc.collect()

sgd = SGDClassifier(alpha=3*10**-6, average=True, loss="log", penalty="elasticnet", 
                    l1_ratio=.28, random_state=21, class_weight="balanced", n_jobs=-1)

print("Evaluating predictions part 2")
test_preds2 = []
train_preds2 = np.zeros(train_df.shape[0])
for train_idx, valid_idx in cv.split(X_train, y_train):
    sgd.fit(X_train[train_idx], y_train[train_idx])
    train_preds2[valid_idx] = sgd.predict_proba(X_train[valid_idx])[:, 1]
    test_preds2.append(sgd.predict_proba(X_test)[:, 1])
    
test_preds2 = np.average(test_preds2, axis=0)

y_mix_train = np.hstack([train_preds1.reshape(-1, 1),
                         train_preds2.reshape(-1, 1)])
y_mix_test = np.hstack([test_preds1.reshape(-1, 1), 
                        test_preds2.reshape(-1, 1)])

print("Mixing step")
clf = SGDClassifier(loss="log", fit_intercept=False, class_weight="balanced", random_state=21, n_jobs=-1)

print("Evaluating final predictions")
test_preds = []
train_preds = np.zeros(train_df.shape[0])
for train_idx, valid_idx in cv.split(y_mix_train, y_train):
    clf.fit(y_mix_train[train_idx], y_train[train_idx])
    train_preds[valid_idx] = clf.predict_proba(y_mix_train[valid_idx])[:, 1]
    test_preds.append(clf.predict_proba(y_mix_test)[:, 1])

print("Searching best threshold")
test_preds = np.average(test_preds, axis=0)
best = print_results(y_train, np.hstack([np.zeros((train_preds.shape[0], 1)), train_preds.reshape(-1, 1)]))
threshold = best[0]

print("Forming submission")
predictions = (test_preds > threshold).astype(int)
submission = pd.DataFrame({"qid": test_df.qid.values, "prediction": predictions}, columns=["qid", "prediction"])
submission.to_csv("submission.csv", index=False)

