# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import re
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from sklearn.utils import shuffle
from gensim.models import Word2Vec

tf.reset_default_graph()
sentences = []
original = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

neg_pd = pd.DataFrame(original[(original.toxic == 1) | (original.severe_toxic == 1) | (original.obscene == 1) | (original.threat == 1) | (original.insult == 1) | (original.identity_hate == 1)])
pos_pd = pd.DataFrame(original[(original.toxic == 0) & (original.severe_toxic == 0) & (original.obscene == 0) & (original.threat == 0) & (original.insult == 0) & (original.identity_hate == 0)])
neg_pd.reset_index(inplace=True)
pos_pd.reset_index(inplace=True)
neg_pd_len = len(neg_pd)
pos_pd_len = len(pos_pd)
for i in range(neg_pd_len*2) :
    pos_row = pos_pd.sample(n=1)
    pos_pd.drop(pos_row.index,inplace=True)
    neg_row = neg_pd.sample(n=1)
    pos_pd = pos_pd.append(neg_row,ignore_index=True)

train = pos_pd.append(neg_pd,ignore_index=True)
train = shuffle(train)

def get_sentences(dataset) :
    for index,row in dataset.iterrows() :
        comment_text = dataset.loc[index,'comment_text']
        text_no_punc = re.sub(r'[^a-zA-Z.]',' ',comment_text)
        text_no_punc = text_no_punc.lower()
        row_sent = text_no_punc.split(".")
        for sent in row_sent :
            words = sent.split()
            word_count = len(words)
            if word_count < 200 :
                for j in range(word_count,200) :
                    words.append('<eos>')
                for i in range(word_count) :
                    if(words[i]) == 's' :
                        words[i] = 'is'
                    elif(words[i]) == 'll' :
                        words[i] = 'will'
                    elif(words[i]) == 'm' :
                        words[i] = 'am'
                    elif(words[i]) == 'u' :
                        words[i] = 'you'
                if word_count > 0 :
                    sentences.append(words)
                    
def get_200_words(dataset) :
    normalised_con = []
    for index,row in dataset.iterrows() :
        content = dataset.loc[index,'comment_text']
        stripped_con = re.sub(r'[^a-zA-Z]',' ',content)
        con_lower = stripped_con.lower()
        row_con = []
        row_con = con_lower.split()
        if(len(row_con) < 200) :
            for i in range(len(row_con),200) :
                row_con.append('<eos>')
        else :
            row_con = row_con[:200]
        normalised_con.append(np.asarray(row_con))
    return normalised_con
    
def get_word_embeddings(dataset,model) :
    embeddings_list = []
    for index,row in dataset.iterrows() :
        word_list = dataset.loc[index,'words']
        embeddings = []
        for word in word_list :
            if word in model.wv.vocab :
                embeddings.append(model.wv[word])
            else :
                embeddings.append(model.wv['<eos>'])
        embeddings_list.append(embeddings)
    return embeddings_list
    
train['words'] = get_200_words(train)
test['words'] = get_200_words(test)
get_sentences(train)
get_sentences(test)
embedding_size = 30
model = Word2Vec(sentences,size=embedding_size)
train['embeddings'] = get_word_embeddings(train,model)
test['embeddings'] = get_word_embeddings(test,model)
n_layers = 2
n_outputs = 6
n_neurons = 150
learning_rate = 0.001
X = tf.placeholder(tf.float32,[None,200,embedding_size])
y = tf.placeholder(tf.float32,[None,6])
is_training = tf.placeholder(tf.bool,shape=(),name='is_training')
def make_cell() :
    def f1() :
        return 0.5
    def f2() :
        return 1.0
    keep_prob = tf.cond(tf.equal(is_training,tf.constant(True)),f1,f2)
    cell = tf.contrib.rnn.GRUCell(num_units=n_neurons)
    cell_drop = tf.contrib.nn.DropoutWrapper(cell,input_keep_prob=keep_prob)
    return cell_drop
multi_layer_cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(n_layers)],state_is_tuple=False)
gru_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell,X,dtype=float32)
logits = fully_connected(states,n_outputs,activation_fn=None)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
predictions = tf.round(tf.nn.sigmoid(logits))
correct = tf.equal(predictions,tf.round(y))
accuracy = tf.reduce_mean(tf.cast(correct,float32))
all_labels_true = tf.reduce_min(tf.cast(correct,tf.float32),1)
accuracy2 = tf.reduce_mean(all_labels_true)
init = tf.global_variables_initializer()

def get_next_batch(batch_size,iteration) :
    if (iteration+1)*batch_size > len(train) :
        X_batch = train.iloc[iteration*batch_size:]
    else :
        X_batch = train.iloc[iteration*batch_size:(iteration+1)*batch_size]
    X_batch_col = []
    for index,row in X_batch.iterrows() :
        X_batch_col.append(X_batch.loc[index,'embeddings'])
    y_batch = X_batch.loc[:,['toxic','severe_toxic','obscene','threat','insult','identity_hate']].applymap(np.float32)
    return (X_batch_col,y_batch)
    
def get_test_data() :
    X_test = []
    for index,rows in test.iterrows() :
        X_test.append(test.loc[index,'embeddings'])
    return X_test
    
n_epochs = 10
batch_size = 64
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess :
    init.run()
    X_test = get_test_data()
    for epoch in range(n_epochs) :
        for iteration in range(len(train)//batch_size):
            X_batch,y_batch = get_next_batch(batch_size,iteration)
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch,is_training:True})
        acc_train = accuracy.eval(feed_dict={X:X_batch,y:y_batch,is_training:True})
        print(epoch,"Training accuracy:",acc_train)
    pred = predictions.eval(feed_dict={X:X_batch,y:y_batch,is_training:False})
    pred_DF = pd.DataFrame(pred,columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate'])
pred_DF = pred_DF.applymap(np.int)
submissionDF = pred_DF.join(test)
submissionDF = pd.DataFrame(pred,columns=['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'])
submissionDF.to_csv('submission.csv',index=False)