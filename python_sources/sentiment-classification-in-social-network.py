#!/usr/bin/env python
# coding: utf-8

# # Sentiment Classification

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read data

# In[ ]:


data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')


# In[ ]:


data_train.head()


# In[ ]:


print(data_train.dtypes)
print(data_train.describe())
print(data_train.info())


# In[ ]:


data_train.label.value_counts()


# In[ ]:


print(data_train.shape, data_test.shape)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


vectorizer = CountVectorizer()


# In[ ]:


train_vector = vectorizer.fit_transform(data_train.sentence)
test_vector = vectorizer.transform(data_test.sentence)


# ### Visualize Word Frequency

# In[ ]:


WordFrequency = pd.DataFrame({'Word': vectorizer.get_feature_names(), 'Count': train_vector.toarray().sum(axis=0)})


# In[ ]:


WordFrequency['Frequency'] = WordFrequency['Count'] / WordFrequency['Count'].sum()


# In[ ]:


plt.plot(WordFrequency.Frequency)
plt.xlabel('Word Index')
plt.ylabel('Word Frequency')
plt.show()


# ### Sort WordFrequency in descending order

# In[ ]:


WordFrequency_sort = WordFrequency.sort_values(by='Frequency', ascending=False)
WordFrequency_sort.head()


# ## Model 1: Naive Bayes

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score


# In[ ]:


clf1 = MultinomialNB()


# In[ ]:


cross_val_acc = cross_val_score(clf1, train_vector, data_train.label.values, cv=10, scoring='accuracy')
print(cross_val_acc)
print(cross_val_acc.mean())


# In[ ]:


clf1.fit(train_vector, data_train.label.values)
predictions = clf1.predict(test_vector)


# In[ ]:


solution1 = pd.DataFrame(list(zip(data_test.sentence, predictions)), columns=['sentence', 'label'])


# In[ ]:


solution1.to_csv('./solution1_naive_bayes.csv', index=False)
# Accuracy in testing data: 0.97461


# ## Model 2: Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier(n_jobs=-1)


# In[ ]:


cross_val_acc2 = cross_val_score(clf2, train_vector, data_train.label.values, cv=10, scoring='accuracy')
print(cross_val_acc2)
print(cross_val_acc2.mean())


# In[ ]:


clf2.fit(train_vector, data_train.label.values)
prediction2 = clf2.predict(test_vector)


# In[ ]:


solution2 = pd.DataFrame(list(zip(data_test.sentence, prediction2)), columns=['sentence','label'])


# In[ ]:


solution2.to_csv('./solution2_random_forest.csv', index=False)
# Accuracy in testing data: 0.97884


# ### Use GridSearchCV

# In[ ]:


from pprint import pprint
from sklearn.model_selection import GridSearchCV


# In[ ]:


pprint(clf2.get_params())


# In[ ]:


param_grid = {
             'class_weight': ['balanced', None],
             'criterion': ['gini', 'entropy'],
             'max_depth': [None, 1, 5, 10],
             'max_features': ['auto', 'log2', None],
             'n_estimators': [5, 10, 20]}
cv_clf2 = GridSearchCV(estimator=clf2, param_grid=param_grid, scoring='accuracy', verbose=0, n_jobs=-1)
cv_clf2.fit(train_vector, data_train.label.values)
best_parameters = cv_clf2.best_params_
print('The best parameters for using RF model is: ', best_parameters)


# In[ ]:


clf2_balanced_gini = RandomForestClassifier(class_weight='balanced', n_estimators=20)
clf2_entropy = RandomForestClassifier(criterion='entropy', n_estimators=20)
clf2_gini = RandomForestClassifier(n_estimators=20)


# In[ ]:


RF_score1 = cross_val_score(clf2_balanced_gini, train_vector, data_train.label.values, cv=10, scoring='accuracy')
print(RF_score1)
print(RF_score1.mean())


# In[ ]:


RF_score2 = cross_val_score(clf2_entropy, train_vector, data_train.label.values, cv=10, scoring='accuracy')
print(RF_score2)
print(RF_score2.mean())


# In[ ]:


RF_score3 = cross_val_score(clf2_gini, train_vector, data_train.label.values, cv=10, scoring='accuracy')
print(RF_score3)
print(RF_score3.mean())


# In[ ]:


clf2_balanced_gini.fit(train_vector, data_train.label.values)
prediction2_tuned = clf2_balanced_gini.predict(test_vector)
solution2_tuned = pd.DataFrame(list(zip(data_test.sentence, prediction2_tuned)), columns=['sentence', 'label'])


# In[ ]:


solution2_tuned.to_csv('./solution2_RF_tuned.csv', index=False)


# ## Model 3: Logistic Regression (Use GridSearchCV to tune hyper-parameters)

# In[ ]:


# Use Logistic Regression directly
from sklearn.linear_model import LogisticRegression
clf3_1 = LogisticRegression()


# In[ ]:


cross_val_acc3_1 = cross_val_score(clf3_1, train_vector, data_train.label.values, cv=10, scoring='accuracy')
print(cross_val_acc3_1)
print(cross_val_acc3_1.mean())


# ### Use GridSearchCV

# In[ ]:


pprint(clf3_1.get_params())


# In[ ]:


param_grid = {'penalty': ['l1', 'l2'],
             'class_weight': ['balanced', None],
             'C': [0.1, 1, 10]
             }
clf3_2 = GridSearchCV(estimator=clf3_1, param_grid=param_grid, scoring='accuracy', verbose=1, n_jobs=-1)
clf3_2.fit(train_vector, data_train.label.values)
best_param = clf3_2.best_params_
print('The best parameters for using LR model is: ', best_param)


# In[ ]:


clf3_2 = LogisticRegression(C=9.4)
cross_val_acc3_2 = cross_val_score(clf3_2, train_vector, data_train.label.values, cv=10, scoring='accuracy')
print(cross_val_acc3_2)
print(cross_val_acc3_2.mean())


# In[ ]:


clf3_1.fit(train_vector, data_train.label.values)
clf3_2.fit(train_vector, data_train.label.values)
prediction3_1 = clf3_1.predict(test_vector)
prediction3_2 = clf3_2.predict(test_vector)


# In[ ]:


solution3_origin_LR = pd.DataFrame(list(zip(data_test.sentence, prediction3_1)), columns=['sentence', 'label'])
solution3_CV_LR = pd.DataFrame(list(zip(data_test.sentence, prediction3_2)), columns=['sentence', 'label'])


# In[ ]:


solution3_origin_LR.to_csv('./solution3_origin_LR.csv', index=False)
# Accuracy in testing data: 0.99083


# In[ ]:


solution3_CV_LR.to_csv('./solution3_CV_LR.csv', index=False)
# Accuracy in testing data:0.99083


# ## Model 4: RNN

# In[ ]:


import collections
import tensorflow as tf
import os
import nltk
from keras.preprocessing import sequence


# In[ ]:


data_train.head()


# In[ ]:


num_sentences = len(data_train)
print(num_sentences)


# ### Applied nltk.word_tokenize to do words spliting

# In[ ]:


maxLength = 0
word_frequency = collections.Counter()


# In[ ]:


for idx, row in data_train.iterrows():
    words = nltk.word_tokenize(row['sentence'].lower())
    if len(words) > maxLength:
        maxLength = len(words)
    for word in words:
        word_frequency[word] += 1
print(len(word_frequency))
print(maxLength)


# In[ ]:


maxFeatures = 2074
vocab_size = maxFeatures + 2
word2index = {x[0]: i+2 for i, x in enumerate(word_frequency.most_common(maxLength))}

word2index['PAD'] = 0
word2index['UNK'] = 1
index2word = {i:w for w, i in word2index.items()}


# In[ ]:


data_X_in = np.empty((num_sentences, ), dtype=list)
data_y = np.zeros((num_sentences, ))
i = 0

for index, row in data_train.iterrows():
    words = nltk.word_tokenize(row['sentence'].lower())
    seqs = []
    for word in words:
        if word in word2index:
            seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])
    data_X_in[i] = seqs
    data_y[i] = int(row['label'])
    i += 1


# In[ ]:


data_X_in = sequence.pad_sequences(data_X_in, padding='post', value=word2index['PAD'], maxlen=maxLength)


# In[ ]:


print(data_X_in[:5])
print(data_X_in.shape)


# In[ ]:


print(data_train.sentence.head())


# In[ ]:


print(data_y[:5])
print(data_y.shape)


# In[ ]:


def data_generator(batch_size):
    while True:
        for i in range(0,len(data_X_in),batch_size):
            if i + batch_size < len(data_X_in):
                yield data_X_in[i:i + batch_size], data_y[i:i + batch_size]


# In[ ]:


batch_size = 24
embedding_size = 100
vocab_size = maxFeatures + 2
num_units = 64
NUM_EPOCHS = 10


# In[ ]:


import tflearn
tf.reset_default_graph()
config = tf.ConfigProto(log_device_placement=True,allow_soft_placement = True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


# In[ ]:


with tf.device('/gpu:1'):
    initializer = tf.random_uniform_initializer(
        -0.08, 0.08)
    tf.get_variable_scope().set_initializer(initializer)
    x = tf.placeholder("int32", [None, None])
    y = tf.placeholder("int32", [None])
    x_len = tf.placeholder("int32",[None])
    
    learning_rate = tf.placeholder(tf.float32, shape=[])
    
    # embedding
    embedding_encoder = tf.get_variable(
        "embedding_encoder", [vocab_size, embedding_size],dtype=tf.float32)
    encoder_emb_inp = tf.nn.embedding_lookup(
        embedding_encoder, x)
    
    # Build RNN cell
    encoder_cell = tf.contrib.rnn.BasicLSTMCell(num_units)
    
    encoder_cell = tf.contrib.rnn.DropoutWrapper(cell=encoder_cell, output_keep_prob=0.75)
    # Run Dynamic RNN
    #   encoder_outputs: [max_time, batch_size, num_units]
    #   encoder_state: [batch_size, num_units]
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        encoder_cell, encoder_emb_inp,
        sequence_length=x_len, time_major=False,dtype=tf.float32)
    
    model_logistic = tf.layers.dense(encoder_state[0],1)
    model_pred = tf.nn.sigmoid(model_logistic)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y,tf.float32),logits=tf.reshape(model_logistic,(-1,)))
    loss = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(loss)


# In[ ]:


session.run(tf.global_variables_initializer())


# In[ ]:


import os
import sys
import time

class Dataset():
    def __init__(self,data,label):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._label = label
        assert(data.shape[0] == label.shape[0])
        self._num_examples = data.shape[0]
        pass

    @property
    def data(self):
        return self._data
    
    @property
    def label(self):
        return self._label

    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = self.data[idx]  # get list of `num` random samples
            self._label = self.label[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            label_rest_part = self.label[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of `num` random samples
            self._label = self.label[idx0]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch  
            data_new_part =  self._data[start:end]  
            label_new_part = self._label[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0),np.concatenate((label_rest_part, label_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end],self._label[start:end]

class ProgressBar():
    def __init__(self,worksum,info="",auto_display=True):
        self.worksum = worksum
        self.info = info
        self.finishsum = 0
        self.auto_display = auto_display
    def startjob(self):
        self.begin_time = time.time()
    def complete(self,num):
        self.gaptime = time.time() - self.begin_time
        self.finishsum += num
        if self.auto_display == True:
            self.display_progress_bar()
    def display_progress_bar(self):
        percent = self.finishsum * 100 / self.worksum
        eta_time = self.gaptime * 100 / (percent + 0.001) - self.gaptime
        strprogress = "[" + "=" * int(percent // 2) + ">" + "-" * int(50 - percent // 2) + "]"
        str_log = ("%s %.2f %% %s %s/%s \t used:%ds eta:%d s" % (self.info,percent,strprogress,self.finishsum,self.worksum,self.gaptime,eta_time))
        sys.stdout.write('\r' + str_log)

def get_dataset(paths):
    dataset = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths = [os.path.join(facedir,img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

def split_dataset(dataset, split_ratio, mode):
    if mode=='SPLIT_CLASSES':
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes*split_ratio))
        train_set = [dataset[i] for i in class_indices[0:split]]
        test_set = [dataset[i] for i in class_indices[split:-1]]
    elif mode=='SPLIT_IMAGES':
        train_set = []
        test_set = []
        min_nrof_images = 2
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            split = int(round(len(paths)*split_ratio))
            if split<min_nrof_images:
                continue  # Not enough images for test set. Skip class...
            train_set.append(ImageClass(cls.name, paths[0:split]))
            test_set.append(ImageClass(cls.name, paths[split:-1]))
    else:
        raise ValueError('Invalid train/test split mode "%s"' % mode)
    return train_set, test_set


# In[ ]:


losses = []
beginning_lr = 0.1
gen = data_generator(batch_size)
for one_epoch in range(0,1):
    pb = ProgressBar(worksum=len(data_X_in))
    pb.startjob()
    for one_batch in range(0,len(data_X_in),batch_size):
        batch_x,batch_y = gen.__next__()
        batch_x_len = np.asarray([len(i) for i in batch_x])
        batch_lr = beginning_lr 
        
        _,batch_loss = session.run([optimizer,loss],feed_dict={
            x:batch_x,
            y:batch_y,
            x_len:batch_x_len,
            learning_rate:batch_lr,
        })
        pb.info = "EPOCH {} batch {} lr {} loss {}".format(one_epoch,one_batch,batch_lr,batch_loss)
        pb.complete(batch_size)
        losses.append(batch_loss)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.DataFrame(losses).plot()


# In[ ]:


def predict_result(sent):
    words = nltk.word_tokenize(row['sentence'].lower())
    senttoken = [word2index.get(word,word2index['UNK']) for word in words]
    inputx = np.asarray([senttoken])
    inputx_len = np.asarray([len(senttoken)])
    batch_predict = session.run(model_pred,feed_dict={
            x:inputx,
            x_len:inputx_len,
        })[0]
    return 1 if batch_predict > 0.5 else 0


# In[ ]:


labels = []
for index, row in data_test.iterrows():
    label = predict_result(row['sentence'])
    labels.append(label)


# In[ ]:


print(len(labels))


# In[ ]:


solution_RNN1 = pd.DataFrame(list(zip(data_test.sentence, labels)), columns=['sentence', 'label'])


# In[ ]:


solution_RNN1.to_csv('./solution_RNN1.csv', index=False)


# ### Applied sentence.split() to do words spliting

# In[ ]:


max_len = 0
word_freq = collections.Counter()
for i in data_train.sentence.values:
    words = [j.lower() for j in i.strip('\n').split()]
    if len(words) > max_len:
        max_len = len(words)
    for word in words:
        word_freq[word] += 1
print(len(word_freq))
print(max_len)


# In[ ]:


maxFeatures = 2673
vocab_size = maxFeatures + 2
word2index = {x[0]: i+2 for i, x in enumerate(word_freq.most_common(maxLength))}

word2index['PAD'] = 0
word2index['UNK'] = 1
index2word = {i:w for w, i in word2index.items()}


# In[ ]:


data_X_in = np.empty((num_sentences, ), dtype=list)
data_y = np.zeros((num_sentences, ))
i = 0

for index, row in data_train.iterrows():
    words = [j.lower() for j in row['sentence'].split()]
    seqs = []
    for word in words:
        if word in word2index:
            seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])
    data_X_in[i] = seqs
    data_y[i] = int(row['label'])
    i += 1


# In[ ]:


data_X_in = sequence.pad_sequences(data_X_in, padding='post', value=word2index['PAD'], maxlen=max_len)


# In[ ]:


print(data_X_in[:5])
print(data_X_in.shape)


# In[ ]:


print(data_train.head())


# In[ ]:


print(data_y[:5])
print(data_y.shape)


# In[ ]:


tf.reset_default_graph()
config = tf.ConfigProto(log_device_placement=True,allow_soft_placement = True)
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)

with tf.device('/gpu:1'):
    initializer = tf.random_uniform_initializer(
        -0.08, 0.08)
    tf.get_variable_scope().set_initializer(initializer)
    x = tf.placeholder("int32", [None, None])
    y = tf.placeholder("int32", [None])
    x_len = tf.placeholder("int32",[None])
    
    learning_rate = tf.placeholder(tf.float32, shape=[])
    
    # embedding
    embedding_encoder = tf.get_variable(
        "embedding_encoder", [vocab_size, embedding_size],dtype=tf.float32)
    encoder_emb_inp = tf.nn.embedding_lookup(
        embedding_encoder, x)
    
    # Build RNN cell
    encoder_cell = tf.contrib.rnn.BasicLSTMCell(num_units)
    
    encoder_cell = tf.contrib.rnn.DropoutWrapper(cell=encoder_cell, output_keep_prob=0.75)
    # Run Dynamic RNN
    #   encoder_outputs: [max_time, batch_size, num_units]
    #   encoder_state: [batch_size, num_units]
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        encoder_cell, encoder_emb_inp,
        sequence_length=x_len, time_major=False,dtype=tf.float32)
    
    model_logistic = tf.layers.dense(encoder_state[0],1)
    model_pred = tf.nn.sigmoid(model_logistic)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y,tf.float32),logits=tf.reshape(model_logistic,(-1,)))
    loss = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(loss)


# In[ ]:


session.run(tf.global_variables_initializer())


# In[ ]:


losses = []
beginning_lr = 0.1
gen = data_generator(batch_size)
for one_epoch in range(0,1):
    pb = ProgressBar(worksum=len(data_X_in))
    pb.startjob()
    for one_batch in range(0,len(data_X_in),batch_size):
        batch_x,batch_y = gen.__next__()
        batch_x_len = np.asarray([len(i) for i in batch_x])
        batch_lr = beginning_lr 
        
        _,batch_loss = session.run([optimizer,loss],feed_dict={
            x:batch_x,
            y:batch_y,
            x_len:batch_x_len,
            learning_rate:batch_lr,
        })
        pb.info = "EPOCH {} batch {} lr {} loss {}".format(one_epoch,one_batch,batch_lr,batch_loss)
        pb.complete(batch_size)
        losses.append(batch_loss)


# In[ ]:


pd.DataFrame(losses).plot()


# In[ ]:


def predict_result(sent):
    words = [j.lower() for j in row['sentence'].split()]
    senttoken = [word2index.get(word,word2index['UNK']) for word in words]
    inputx = np.asarray([senttoken])
    inputx_len = np.asarray([len(senttoken)])
    batch_predict = session.run(model_pred,feed_dict={
            x:inputx,
            x_len:inputx_len,
        })[0]
    return 1 if batch_predict > 0.5 else 0


# In[ ]:


labels = []
for index, row in data_test.iterrows():
    label = predict_result(row['sentence'])
    labels.append(label)


# In[ ]:


print(len(labels))


# In[ ]:


COLUMN_NAMES = ['sentence', 'label']
solution_RNN2 = pd.DataFrame(columns=COLUMN_NAMES)

solution_RNN2['sentence'] = data_test['sentence']
ll = pd.Series(labels)
solution_RNN2['label'] = ll.values
print(solution_RNN2.shape)


# In[ ]:


solution_RNN2.to_csv('./solution_RNN2.csv', index=False)

