#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.contrib import rnn
import numpy as np
import random
import pickle
from collections import Counter
import tensorflow as tf
import pandas as pd



# In[ ]:


#kaggle datasets download -d krabhi07/positive-and-negative-sentence-only-ascii
pos_filepath= "../input/positive-and-negative-sentences/positive.txt"
neg_filepath= "../input/positive-and-negative-sentences/negative.txt"
#pos_data=pd.read_csv(pos_filepath, sep=" ", header=None)
#neg_data=pd.read_clipboard(neg_filepath)

lemmatizer=WordNetLemmatizer()
hm_lines=10000000



# In[ ]:



        


def create_lexicon(pos,neg):
    lexicon=[]
    for fi in [pos,neg]:
        with open(fi,'r',encoding='utf-8',errors='ignore') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words=word_tokenize(l.lower())
                lexicon +=list(all_words)
                
                
    lexicon=[lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2=[]
    for w in w_counts:
        if 1000 >w_counts[w]> 50:
            l2.append(w)
            
            
    print(len(l2))      
    return l2
                


# In[ ]:


def sample_handling(sample,lexicon,classification):
    featureset=[]
    with open(sample,'r',encoding='utf-8',errors='ignore') as f:
        contents=f.readlines()#.decode('windows-1252')    here utf-8 encoding error
        for l in contents[:hm_lines]:
            current_words=word_tokenize(l.lower())
            current_words=[lemmatizer.lemmatize(i) for i in current_words]
            features=np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value=lexicon.index(word.lower())
                    features[index_value] += 1
                    
            features= list(features)
            featureset.append([features,classification])
            
    return featureset       


# In[ ]:


def create_feature_sets_and_labels(pos,neg,test_size=0.1):
    lexicon = create_lexicon(pos_filepath,neg_filepath)
    features=[]
    features += sample_handling(pos_filepath,lexicon,[1,0])
    features += sample_handling(neg_filepath,lexicon,[0,1])
    random.shuffle(features)
    features=np.array(features)
    testing_size=int(test_size*len(features))
    
    train_x=list(features[:,0][:-testing_size])
    train_y=list(features[:,1][:-testing_size])
    
    test_x=list(features[:,0][-testing_size:])
    test_y=list(features[:,1][-testing_size:])
    

    return train_x,train_y,test_x,test_y,lexicon    


# In[ ]:


if __name__ == '__main__' :
   train_x,train_y,test_x,test_y,lexicon=create_feature_sets_and_labels(pos_filepath,neg_filepath)
    #with open('sentiment_set.pickle','wb') as f:
       #pickle.dump([train_x,train_y,test_x,test_y],f)
      
   


# In[ ]:


n_nodes_hl1=500
n_nodes_hl2=500
n_nodes_hl3=500

n_classes=2
batch_size=100

x=tf.placeholder('float',shape=[None,len(train_x[0])])
y=tf.placeholder('float')


# In[ ]:


def neural_network(data):

    hidden_layer_1={'Weights':tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_hl1])),
                    'Biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_layer_2={'Weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                    'Biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_layer_3={'Weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                    'Biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer={'Weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                  'Biases':tf.Variable(tf.random_normal([n_classes]))}
    
    l1=tf.add(tf.matmul(data,hidden_layer_1['Weights']), hidden_layer_1['Biases'])
    l1=tf.nn.relu(l1)
    
    l2=tf.add(tf.matmul(l1,hidden_layer_2['Weights']), hidden_layer_2['Biases'])
    l2=tf.nn.relu(l2)
    
    l3=tf.add(tf.matmul(l2,hidden_layer_3['Weights']), hidden_layer_3['Biases'])
    l3=tf.nn.relu(l3)
    
    output=tf.add(tf.matmul(l3,output_layer['Weights']), output_layer['Biases'])
    
    
    return output
    


# In[ ]:


def train_neural_network(x):
    prediction=recurrent_neural_network(x)                       
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits= prediction, labels= y,name=None))
    optimizer=tf.train.AdamOptimizer().minimize(cost)
    hm_epoch=3
    print(hm_epoch)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(hm_epoch):
            epoch_loss=0
            
            i=0
            while i < len(train_x):
                start=i
                end=i+batch_size
                batch_x=np.array(train_x[start:end])
                batch_y=np.array(train_y[start:end])
            
                _,c=sess.run([optimizer,cost], feed_dict = {x:batch_x,y: batch_y})
                epoch_loss+=c
                
                i += batch_size
            print('Epoch :',epoch,'completed out of :',hm_epoch,'loss :',epoch_loss)   
           
        
        correct=tf.equal(tf.argmax(prediction,1), tf.argmax(y ,1))
        accuracy=tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy :',accuracy.eval({x:test_x, y:test_y}))
        input_data=input("enter the test sentence : ")
        features = get_features_for_input(input_data,lexicon)
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:features}),1)))
        if result[0] == 0:
            print('Positive:',input_data)
            f1=open('../input/positive.txt','r')
            
            if f1.mode == 'r':
                    
                    contents=f1.read()
                    print(contents)
                
                
       
            
        elif result[0] == 1:
            print('Negative:',input_data)
            f2=open('../input/negative.txt','r')
            if f2.mode =='r':
                
                contents=f2.read()
                print(contents)
            
                
                
        

 


# In[ ]:


def train_neural_network(x):
    prediction=neural_network(x)                       
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits= prediction, labels= y,name=None))
    optimizer=tf.train.AdamOptimizer().minimize(cost)
    hm_epoch=7
    print(hm_epoch)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(hm_epoch):
            epoch_loss=0
            
            i=0
            while i < len(train_x):
                start=i
                end=i+batch_size
                batch_x=np.array(train_x[start:end])
                batch_y=np.array(train_y[start:end])
            
                _,c=sess.run([optimizer,cost], feed_dict = {x:batch_x,y: batch_y})
                epoch_loss+=c
                
                i += batch_size
            print('Epoch :',epoch,'completed out of :',hm_epoch,'loss :',epoch_loss)   
           
        
        correct=tf.equal(tf.argmax(prediction,1), tf.argmax(y ,1))
        accuracy=tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy :',accuracy.eval({x:test_x, y:test_y}))
        input_data=input("enter the test sentence :  ")
        features = get_features_for_input(input_data,lexicon)
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:features}),1)))
        if result[0] == 0:
            print('Positive:',input_data)
            '''' f1=open('../input/positive.txt','r')
            
            if f1.mode == 'r':
                    
                    contents=f1.read()
                    print(contents)'''
                
                
       
            
        elif result[0] == 1:
            print('Negative:',input_data)
            '''f2=open('../input/negative.txt','r')
             if f2.mode =='r':
                
                contents=f2.read()
                print(contents)'''
            
                
                
        

 


# In[ ]:


def get_features_for_input(input,lexicon):
    featureset = []
    lemmatizer=WordNetLemmatizer()
    current_words = word_tokenize(input.lower())
    current_words = [lemmatizer.lemmatize(i) for i in current_words]
    features = np.zeros(len(lexicon))
    for word in current_words:
        if word.lower() in lexicon:
            index_value = lexicon.index(word.lower())
            features[index_value] += 1
    featureset.append(features)
    return np.asarray(featureset)

train_neural_network(x)

