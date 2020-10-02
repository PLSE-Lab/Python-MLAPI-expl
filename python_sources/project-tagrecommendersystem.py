#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import pickle
import itertools
from collections import Counter

import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.preprocessing import text

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from google.cloud import bigquery


# In[ ]:


SPACE = ' '
pattern_code = re.compile('<pre><code>.*?</code></pre>', re.DOTALL)
pattern_link = re.compile('[a-z]+://')
pattern_atag = re.compile('<a[^>]+>(.*)</a>')
pattern_tags = re.compile('<[^>]+>')
pattern_qout_n_t_r = re.compile('["\n\t\r]')
pattern_more_space = re.compile(' +')

def clean_text(text):
    if not isinstance(text, str):
        return text
    text = pattern_code.sub(SPACE, text)

    def replace_link(match):
        return SPACE if pattern_link.match(match.group(1)) else match.group(1)
    
    text = pattern_atag.sub(replace_link, text)
    text = pattern_tags.sub(SPACE, text)
    text = pattern_qout_n_t_r.sub(SPACE, text)
    text = pattern_more_space.sub(SPACE, text)
    return text.lower()


def fetch_data(limit):
    client = bigquery.Client()
    query = """
        SELECT title, body, tags
        FROM `bigquery-public-data.stackoverflow.posts_questions`
        LIMIT {} """.format(limit)
    return client.query(query).to_dataframe()


def get_commons(tdf, limit):
    df = tdf.reset_index(drop=True)
    tuple_of_tags = (tag for tags in df['tags'] for tag in tags.split('|'))
    common_tags =  Counter(tuple_of_tags).most_common(limit)
    common_tags_name = set(tag[0] for tag in common_tags)
    tags = dict(common_tags)

    common_tags_boolean_flags = []
    for index in range(df.shape[0]):
        count = 0
        elem = ''
        for tag in df.at[index, 'tags'].split('|'):
            if tag in tags and count < tags[tag]:
                count = tags[tag]
                elem = tag
        if count != 0:
            df.at[index, 'tags'] = elem
            common_tags_boolean_flags.append(True)
        else:
            common_tags_boolean_flags.append(False)
    
    return df[common_tags_boolean_flags].reset_index(drop=True), common_tags_name



def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This utility function is from the sklearn docs: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)


def plot_cnf_from_predicts(title, text_labels, y_test_1d, y_pred_1d):
    cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
    plt.figure(figsize=(24,20))
    plot_confusion_matrix(cnf_matrix, classes=text_labels, title=title)
    plt.show()
    return cnf_matrix


# In[ ]:


# fetch and clean the data

posts_questions = fetch_data(100000)
posts, common_tags_name = get_commons(posts_questions, 10)
posts.title = posts.title.apply(clean_text)
posts.body = posts.body.apply(clean_text)


# In[ ]:


# split dataset (train:test = 70:30)

train_size = int(len(posts) * .7)
train_posts = posts[:train_size]
test_posts = posts[train_size:].reset_index(drop=True)


# ## using neural net

# In[ ]:


x_train_title = train_posts.title.copy()
x_test_title = test_posts.title.copy()
x_train_body = train_posts.body.copy()
x_test_body = test_posts.body.copy()

y_train = train_posts.tags.copy()
y_test = test_posts.tags.copy()


# In[ ]:


max_title_words = 1000
max_body_words = 10000
tokenize_title = text.Tokenizer(num_words=max_title_words, char_level=False)
tokenize_title.fit_on_texts(x_train_title) # only fit on train
tokenize_body = text.Tokenizer(num_words=max_body_words, char_level=False)
tokenize_body.fit_on_texts(x_train_body) # only fit on train

x_train_title = tokenize_title.texts_to_matrix(x_train_title)
x_train_body = tokenize_body.texts_to_matrix(x_train_body)
x_train = np.concatenate((x_train_title, x_train_body), axis=1)

x_test_title = tokenize_title.texts_to_matrix(x_test_title)
x_test_body = tokenize_body.texts_to_matrix(x_test_body)
x_test = np.concatenate((x_test_title, x_test_body), axis=1)


# In[ ]:


encoder = LabelEncoder()
encoder.fit(list(common_tags_name))
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)
text_labels = encoder.classes_

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)


# In[ ]:


get_ipython().run_cell_magic('time', '', "neural_net = Sequential()\nneural_net.add(Dense(2000, activation='relu', input_dim=x_train.shape[1]))\nneural_net.add(Dropout(0.1))\nneural_net.add(Dense(600, activation='relu'))\nneural_net.add(Dropout(0.1))\nneural_net.add(Dense(y_train.shape[1], activation='softmax'))\nneural_net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])\n\nhistory = neural_net.fit(x_train, y_train, epochs=4, batch_size=1024, verbose=1)\nloss, accuracy = neural_net.evaluate(x_test, y_test, batch_size=1024, verbose=1)\nprint('loss: {}, accuracy: {}'.format(loss, accuracy))")


# In[ ]:


def predict_results_neural_net():
    y_test_1d, y_pred_1d = [], []
    y_softmax = neural_net.predict(x_test)

    for i in range(len(y_test)):
        probs = y_test[i]
        index_arr = np.nonzero(probs)
        one_hot_index = index_arr[0].item(0)
        y_test_1d.append(one_hot_index)

    for i in range(len(y_softmax)):
        probs = y_softmax[i]
        predicted_index = np.argmax(probs)
        y_pred_1d.append(predicted_index)
    return y_test_1d, y_pred_1d


y_test_1d, y_pred_1d = predict_results_neural_net()
cnf_matrix_neural_net = plot_cnf_from_predicts("Confusion matrix for neural net", text_labels, y_test_1d, y_pred_1d)


# In[ ]:


def save_neural_net():
    with open("neural_net.json", "w") as neural_net_json_file:
        neural_net_json_file.write(neural_net.to_json())
    neural_net.save_weights("neural_net.h5")

    with open('extras.pkl', 'wb') as extras:
        pickle.dump({
            'tokenize_title': tokenize_title,
            'tokenize_body': tokenize_body,
            'text_labels': text_labels,
            'encoder': encoder
        }, extras)

def load_neural_net():
    global tokenize_title, tokenize_body, text_labels, encoder
    with open('neural_net.json', 'r') as neural_net_json_file:
        neural_net = model_from_json(neural_net_json_file.read())
    neural_net.load_weights("neural_net.h5")

    with open('extras.pkl', 'rb') as extras:
        extra = pickle.load(extras)
    tokenize_title = extra['tokenize_title']
    tokenize_body = extra['tokenize_body']
    text_labels = extra['text_labels']
    encoder = extra['encoder']

def predict_single_tag_neural_net(data_title=None, data_body=None):
    if data_title is None:
        data_title = input('Enter title: ')
    if data_body is None:
        data_body = input('Enter body: ')

    tokenized_input = np.concatenate((tokenize_title.texts_to_matrix([data_title]), tokenize_body.texts_to_matrix([data_body])), axis=1)
    prediction = neural_net.predict(np.array(tokenized_input))
    predicted_tag = text_labels[np.argmax(prediction)]
    return predicted_tag


predict_single_tag_neural_net('', 'lifecycle of activity')


# ## using lsvm

# In[ ]:


x_train = train_posts.body.copy()
x_test = test_posts.body.copy()

y_train = train_posts.tags.copy()
y_test = test_posts.tags.copy()


# In[ ]:


get_ipython().run_cell_magic('time', '', "linear_svm = Pipeline([\n    ('vec', CountVectorizer()),\n    ('tfidf', TfidfTransformer()),\n    ('clf', svm.SVC(kernel='linear', verbose=True)),\n])\nlinear_svm.fit(x_train, y_train)\ny_pred_lsvm = linear_svm.predict(x_test)\n\nprint('accuracy {}'.format(accuracy_score(y_pred_lsvm, y_test)))\nprint(classification_report(y_test, y_pred_lsvm, target_names=text_labels))")


# In[ ]:


def predict_results_lsvm(y_pred_lsvm):
    y_test_1d, y_pred_1d = [], []

    labels = list(text_labels)
    for ytest, ypred in zip(y_test, y_pred_lsvm):
        y_test_1d.append(labels.index(ytest))
        y_pred_1d.append(labels.index(ypred))
    return y_test_1d, y_pred_1d


y_test_1d, y_pred_1d = predict_results_lsvm(y_pred_lsvm)
cnf_matrix_lsvm = plot_cnf_from_predicts("Confusion matrix for linear svm", text_labels, y_test_1d, y_pred_1d)


# In[ ]:


def save_lsvm():
    linear_svm_file = "linear_svm_model.pkl"
    joblib.dump(linear_svm, linear_svm_file)


def load_lsvm():
    linear_svm = joblib.load(linear_svm_file)

def predict_single_tag_linear_svm(data_title=None, data_body=None):
    if data_title is None:
        data_title = input('Enter title: ')
    if data_body is None:
        data_body = input('Enter body: ')
    return linear_svm.predict([data_body])[0]


predict_single_tag_linear_svm('', 'lifecycle of activity')


# In[ ]:


def plot_confusion_matrix_for_one_class(cm, classes, title, cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=12)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=10)
    plt.xlabel('Predicted label', fontsize=10)
    
    
def summary_plot(cnf_matrix):
    total_sensitivity = np.array([np.array([0, 0]), np.array([0, 0])])
    cnf_total = cnf_matrix.sum()
    cnf_row = cnf_matrix.sum(axis=1)
    cnf_col = cnf_matrix.sum(axis=0)

    for i in range(len(text_labels)):
        tp = cnf_matrix[i][i]
        fn = cnf_row[i] - tp
        fp = cnf_col[i] - tp
        tn = cnf_total - (fn + fp + tp)
        total_sensitivity[0][0] += tp
        total_sensitivity[0][1] += fn
        total_sensitivity[1][0] += fp
        total_sensitivity[1][1] += tn
    plt.figure(figsize=(2,2))
    plot_confusion_matrix_for_one_class(total_sensitivity, classes=["true class", "non-true class"], title="Confusion matrix for overall")
    plt.show()


# In[ ]:


summary_plot(cnf_matrix_neural_net)


# In[ ]:


summary_plot(cnf_matrix_lsvm)


# # some other algorithms
# ## using naive bayes

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nnaive_bayes = Pipeline([\n    ('vec', CountVectorizer()),\n    ('tfidf', TfidfTransformer()),\n    ('clf', MultinomialNB())\n])\nnaive_bayes.fit(x_train, y_train)\ny_pred_nb = naive_bayes.predict(x_test)\n\nprint('accuracy of naive bayes: {}'.format(accuracy_score(y_pred_nb, y_test)))\nprint(classification_report(y_test, y_pred_nb, target_names=text_labels))")


# ## using sdg

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nstochastic_gradient_descent = Pipeline([\n    ('vec', CountVectorizer()),\n    ('tfidf', TfidfTransformer()),\n    ('clf', SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)),\n])\nstochastic_gradient_descent.fit(x_train, y_train)\ny_pred_sdg = stochastic_gradient_descent.predict(x_test)\n\nprint('accuracy of SDGClassifier: {}'.format(accuracy_score(y_pred_sdg, y_test)))\nprint(classification_report(y_test, y_pred_sdg, target_names=text_labels))")

