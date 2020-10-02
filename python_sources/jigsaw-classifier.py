#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, time
import tensorflow as tf
import tensorflow_hub as hub
from kaggle_datasets import KaggleDatasets
import sys
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB, GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.model_selection import train_test_split
# We'll use a tokenizer for the BERT model from the modelling demo notebook.
get_ipython().system('pip install bert-tensorflow')
import bert.tokenization

print(tf.version.VERSION)


# In[ ]:


SEQUENCE_LENGTH = 128

DATA_PATH =  "../input/jigsaw-multilingual-toxic-comment-classification"
BERT_PATH = "../input/bert-multi"
BERT_PATH_SAVEDMODEL = os.path.join(BERT_PATH, "bert_multi_from_tfhub")

OUTPUT_PATH = "/kaggle/working"


# # Examples
# 
# Load and look at examples from [our first competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/). These are comments from Wikipedia with a variety of annotations (toxic, obscene, threat, etc).

# In[ ]:


# Training data from our first competition,
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
wiki_toxic_comment_data = "jigsaw-toxic-comment-train.csv"
wiki_toxic_comment_data = "jigsaw-toxic-comment-train-processed-seqlen128.csv"

wiki_toxic_comment_train = pd.read_csv(os.path.join(
    DATA_PATH, wiki_toxic_comment_data))
wiki_toxic_comment_train.head()


# # Generating Balance Data

# In[ ]:


wiki_toxic_comment_train['comment_text'][0]


# In[ ]:


toxic = wiki_toxic_comment_train[wiki_toxic_comment_train.toxic ==1]
not_toxic = wiki_toxic_comment_train[wiki_toxic_comment_train.toxic == 0]

downsampled = resample(not_toxic,
                       replace = False, # sample without replacement
                       n_samples = len(toxic), # match minority n
                       random_state = 10) # reproducible results
train = pd.concat([downsampled, toxic])
train.toxic.value_counts().plot(kind = 'bar')


# In[ ]:


vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18,
                               alternate_sign=False)

X = train['comment_text']
y = train['toxic']


# In[ ]:


X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
X_train = vectorizer.transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# In[ ]:


def benchmark(clf, name):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    
    auc = metrics.roc_auc_score(y_test, pred)
    print("auc:      %0.3f" % auc)
    
    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                           ))

        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = name
    return clf_descr, auc, train_time, test_time


# In[ ]:


results = []
for clf, name in (
        (LogisticRegression(C = 1), "Logistic Regression C = 1"),
        (LogisticRegression(C = 10), "Logistic Regression C = 10")):
#        (LogisticRegression(C = 100), "Logistic Regression C = 100")):
#        (RidgeClassifier, "Ridge Classifier"),
#        (KNeighborsClassifier(n_neighbors=10), "kNN"),
#        (RandomForestClassifier(), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf,name))


# In[ ]:


for clf, name in (
        (RandomForestClassifier(n_estimators = 10), "Random forest tree=10"),
#       (RandomForestClassifier(n_estimators = 100), "Random forest tree=100"),
        (RandomForestClassifier(n_estimators = 1000), "Random forest tree=100")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf,name))
    
tuned_parameters = {'n_estimators':[10,100,1000],'max_features':["auto","log2"],'criterion': ['gini', 'entropy']}


# In[ ]:


for clf, name in (
        (RandomForestClassifier(criterion = 'entropy'), "Random forest criterion = entropy"),
        (RandomForestClassifier(max_features = 'log2'), "Random forest max_features = log2")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf,name))


# In[ ]:


for clf, name in (
    (LinearSVC(penalty='l2',  dual=False, tol=1e-3), "LinearSVC L2"),
    (LinearSVC(penalty = 'l1',C = 0.01, dual=False, tol=1e-3), "LinearSVC L1 C = 0.01"),
    (LinearSVC(penalty = 'l1',C = 0.1, dual=False, tol=1e-3), "LinearSVC L1 C = 0.1"),    
    (LinearSVC(penalty = 'l1',C = 10, dual=False, tol=1e-3), "LinearSVC L1 C = 10"),
#    (SVC(kernel = 'poly'), "SVC Poly Kernel"),
 (SVC(kernel = 'rbf'),"SVC RBF Kernel" )):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf,name))


# In[ ]:


# Train sparse Naive Bayes classifiers
print("Naive Bayes")
for alpha in [0.001, 0.01, 0.1, 1,10]:
    results.append(benchmark(MultinomialNB(alpha=alpha), "MultinomialNB Alpha="+str(alpha)))
    results.append(benchmark(BernoulliNB(alpha=alpha), "BernoulliNB Alpha="+str(alpha)))
    results.append(benchmark(ComplementNB(alpha=alpha), "ComplementNB Alpha="+str(alpha)))


# # Hyperparameter Tuning

# In[ ]:



# Set the parameters by cross-validation
#LogisticRegression
tuned_parameters = {'C': [0.1,1,10,100]}

clf = GridSearchCV(LogisticRegression(), tuned_parameters, scoring= 'accuracy')
clf.fit(X_train, y_train)
print("Best parameters set found on development set:")
print(clf.best_params_)
print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))


# In[ ]:


#RandomForestClassifier
# Set the parameters by cross-validation
tuned_parameters = {'n_estimators':[10,100,1000],'max_features':["auto","log2"],'criterion': ['gini', 'entropy']}
clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, scoring= 'accuracy')
clf.fit(X_train, y_train)
print("Best parameters set found on development set:")
print(clf.best_params_)
print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))


# In[ ]:


#SVM
tuned_parameters = [{'kernel': ['rbf','poly','sigmoid'], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

clf = GridSearchCV(SVC(), tuned_parameters, scoring= 'accuracy')
clf.fit(X_train, y_train)
print("Best parameters set found on development set:")
print(clf.best_params_)
print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))


# In[ ]:


res_df = pd.DataFrame(results, columns = ["Classifer", "AUC", "Train Time", "Test Time"])
res_df = res_df.sort_values("AUC", ascending = False)
res_df


# In[ ]:


indices = np.arange(len(results))
res = [[x[i] for x in results] for i in range(4)]
clf_names, score, training_time, test_time = res
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()


# # Ensemble

# Voting

# In[ ]:


clf1 = LinearSVC(penalty='l1', dual=False, tol=1e-3)
clf2 = LogisticRegression(C= 10)
clf3 = ComplementNB(alpha=0.01)
clf4 = RandomForestClassifier(n_estimators = 10)
clf5 = MultinomialNB(alpha=0.01)

clf1.fit(X_train, y_train)
pred1 = clf1.predict(X_test)

clf2.fit(X_train, y_train)
pred2 = clf2.predict(X_test)

clf3.fit(X_train, y_train)
pred3 = clf3.predict(X_test)

clf4.fit(X_train, y_train)
pred4 = clf4.predict(X_test)

clf5.fit(X_train, y_train)
pred5 = clf5.predict(X_test)


# In[ ]:


pred = (pred1 + pred2 + pred3 + pred4 + pred5)/5
auc = metrics.roc_auc_score(y_test, pred)
print("auc:      %0.3f" % auc)


# In[ ]:


pred1_prob = pred1
pred2_prob = [x[1] for x in clf2.predict_proba(X_test)]
pred3_prob = [x[1] for x in clf3.predict_proba(X_test)]
pred4_prob = [x[1] for x in clf4.predict_proba(X_test)]
pred5_prob = [x[1] for x in clf5.predict_proba(X_test)]

pred_prob = (pred1_prob + pred2_prob + pred3_prob + pred4_prob + pred5_prob)/5
auc = metrics.roc_auc_score(y_test, pred_prob)
print("auc:      %0.3f" % auc)


# Boosting

# In[ ]:





# # Top useful feature

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
feature_names = vectorizer.get_feature_names()
feature_names = np.asarray(feature_names)
feature_names


# In[ ]:


clf1 = LogisticRegression()
clf1.fit(X_train,y_train)
top100 = np.argsort(clf1.coef_)[-100:]
pd.DataFrame(feature_names[top100]).transpose()


# # GloVe

# In[ ]:





# ## Plot

# In[ ]:





# # BERT Tokenizer
# 
# Get the tokenizer corresponding to our multilingual BERT model. See [TensorFlow 
# Hub](https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1) for more information about the model.

# In[ ]:


def get_tokenizer(bert_path=BERT_PATH_SAVEDMODEL):
    """Get the tokenizer for a BERT layer."""
    bert_layer = tf.saved_model.load(bert_path)
    bert_layer = hub.KerasLayer(bert_layer, trainable=False)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    cased = bert_layer.resolved_object.do_lower_case.numpy()
    tf.gfile = tf.io.gfile  # for bert.tokenization.load_vocab in tokenizer
    tokenizer = bert.tokenization.FullTokenizer(vocab_file, cased)
  
    return tokenizer

tokenizer = get_tokenizer()


# We can look at one of our example sentences after we tokenize it, and then again after converting it to word IDs for BERT.

# In[ ]:


example_sentence = wiki_toxic_comment_train.iloc[1].comment_text[:150]
print(example_sentence)

example_tokens = tokenizer.tokenize(example_sentence)
print(example_tokens[:17])

example_input_ids = tokenizer.convert_tokens_to_ids(example_tokens)
print(example_input_ids[:17])


# In[ ]:





# # Preprocessing
# 
# Process individual sentences for input to BERT using the tokenizer, and then prepare the entire dataset. The same code will process the other training data files, as well as the validation and test data.

# In[ ]:


def process_sentence(sentence, max_seq_length=SEQUENCE_LENGTH, tokenizer=tokenizer):
    """Helper function to prepare data for BERT. Converts sentence input examples
    into the form ['input_word_ids', 'input_mask', 'segment_ids']."""
    # Tokenize, and truncate to max_seq_length if necessary.
    tokens = tokenizer.tokenize(sentence)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]

    # Convert the tokens in the sentence to word IDs.
    input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    pad_length = max_seq_length - len(input_ids)
    input_ids.extend([0] * pad_length)
    input_mask.extend([0] * pad_length)

    # We only have one input segment.
    segment_ids = [0] * max_seq_length

    return (input_ids, input_mask, segment_ids)

def preprocess_and_save_dataset(unprocessed_filename, text_label='comment_text',
                                seq_length=SEQUENCE_LENGTH, verbose=True):
    """Preprocess a CSV to the expected TF Dataset form for multilingual BERT,
    and save the result."""
    dataframe = pandas.read_csv(os.path.join(DATA_PATH, unprocessed_filename),
                                index_col='id')
    processed_filename = (unprocessed_filename.rstrip('.csv') +
                          "-processed-seqlen{}.csv".format(SEQUENCE_LENGTH))

    pos = 0
    start = time.time()

    while pos < len(dataframe):
        processed_df = dataframe[pos:pos + 10000].copy()

        processed_df['input_word_ids'], processed_df['input_mask'], processed_df['all_segment_id'] = (
            zip(*processed_df[text_label].apply(process_sentence)))

        if pos == 0:
            processed_df.to_csv(processed_filename, index_label='id', mode='w')
        else:
            processed_df.to_csv(processed_filename, index_label='id', mode='a',
                                header=False)

        if verbose:
            print('Processed {} examples in {}'.format(
                pos + 10000, time.time() - start))
        pos += 10000
    return
  
# Process the training dataset.
preprocess_and_save_dataset(wiki_toxic_comment_data)

