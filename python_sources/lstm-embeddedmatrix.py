#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
@author : Rajat Shukla
This is is a script which classify the comments,
data cab be downloaded fron URL https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
The objective of the task is to classify the comments. 
Here comment's type can be toxic severe_toxic obscene threat insult identity_hate
"""
from __future__ import print_function
import os
import numpy as np
import pandas as pd
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D
from keras.layers import Dropout, Embedding
from keras.preprocessing import text, sequence
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from keras.models import Model
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Sequential
from keras import regularizers
from keras.callbacks import Callback
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from matplotlib.backends.backend_pdf import PdfPages


#Set here Global variables
max_features=100000
maxlen=150
embed_size=300
embedding_dimension = 300
glove_data = '../input/glove840b300dtxt/glove.840B.300d.txt'
batch_size = 32
epochs = 1
num_filters = 256
weight_decay = 1e-4
num_classes = 6


class roc_callback(Callback):
    """
    This is class which implements ROC callback for Keras.
    Reference URL: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
    https://hackernoon.com/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier-2ecc6c73115a
    """
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        #####################################
        self.train_x = training_data[0]
        self.train_y = training_data[1]
        self.test_x = validation_data[0]
        self.test_y = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return
    
    def generate_auc_curve(self, ):
        #train_y_pred = self.model.predict(self.train_x)
        test_y_pred = self.model.predict(self.test_x)
        print(test_y_pred)
        # Plot linewidth
        lw = 2
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = 6
        y_score = test_y_pred
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.test_y[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(self.test_y.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        # Plot all ROC curves
        plt.figure(1)
        plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]),color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"],label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'
''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        print("Save figure")
        pp = PdfPages('multipage1.pdf')
        plt.savefig(pp, format='pdf')
        plt.show()


        # Zoom in view of the upper left corner.
        plt.figure(2)
        plt.xlim(0, 0.2)
        plt.ylim(0.8, 1)
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        print("Save figure")
        pp = PdfPages('multipage2.pdf')
        plt.savefig(pp, format='pdf')
        plt.show()

    
    
    
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        ############################## New code for curve ###################
        self.generate_auc_curve()
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return



class CommentPrediction():
    """
    This class predict the type of comment based on comment.
    This class does following task.
    1. Read the train.csv and test.csv by pandas
    2. Fill NA for the comment which doesn't have any comment
    3. Analyse the data
    4. Creates token, convert text to sequence and pad sequences
    5. Create embedding matrix
    6. Crate a layer of model, layer is Sequential --> Embedding --> Conv1D --> MaxPolling1D --> Conv1D --> MaxPolling1D --> Dropout -- > Dense
    7. Train the model
    6. Predict the model
    """
    def __init__(self,
                 ):
        """ This method initialize the object"""
        pass
    
    def __loadData(self):
        """ 
        This method loads the train and test csv file
        Reference URL:
        https://www.dataquest.io/blog/pandas-python-tutorial/
        https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python
        """
        print(os.listdir("."))
        self.train = pd.read_csv('../input/cleaned-toxic-comments/train_preprocessed.csv').fillna(" ")
        self.test  = pd.read_csv('../input/cleaned-toxic-comments/test_preprocessed.csv').fillna(" ")
    
    def __cleanData(self):
        """ 
        This method clean the data
        """        
        self.train_x = self.train['comment_text'].fillna(' ')
        self.test_x  = self.test['comment_text'].fillna(' ')
        self.train_y = self.train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
        self.train_x = self.train_x.str.lower()
        self.test_x = self.test_x.str.lower()
        
        
    def __toknizeData(self):
        """ 
        This method tokenize data.
        Reference URL : 
        http://www.orbifold.net/default/2017/01/10/embedding-and-tokenizer-in-keras/
        https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/
        """
        self.tokenizer = text.Tokenizer(num_words=max_features, )
        self.tokenizer.fit_on_texts(list(self.train_x))
        self.train_x = self.tokenizer.texts_to_sequences(self.train_x)
        self.test_x = self.tokenizer.texts_to_sequences(self.test_x)
        self.train_x = sequence.pad_sequences(self.train_x, maxlen=maxlen)
        self.test_x = sequence.pad_sequences(self.test_x, maxlen=maxlen)
        
    
    
    def __generateEmbeddingMatrix(self,):
        """ 
        This method generates the embedding matrix for the given token using 
        global glob vector 
        Reference I used are
        https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12
        http://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/
        https://nlp.stanford.edu/projects/glove/
        http://www.orbifold.net/default/2017/01/10/embedding-and-tokenizer-in-keras/
        """
        embeddings_index = {}
        f = open(glove_data)
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            value = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = value
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))
        word_index = self.tokenizer.word_index
        num_words = min(max_features, len(word_index))
        print("No of words : %s, embedding dimension : %s" % (num_words, embedding_dimension))
        self.embedding_matrix = np.zeros((num_words , embedding_dimension))
        for word, i in word_index.items():
            if i > num_words : continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector[:embedding_dimension]
        
    def __createModel(self):
        """ 
        This method creates the model for LSTM.
        Reference URL
        https://faroit.github.io/keras-docs/0.3.3/examples/
        http://philipperemy.github.io/keras-stateful-lstm/
        https://towardsdatascience.com/understanding-lstm-and-its-quick-implementation-in-keras-for-sentiment-analysis-af410fd85b47
        https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py
        https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
        https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
        https://keras.io/getting-started/sequential-model-guide/
        http://adventuresinmachinelearning.com/keras-lstm-tutorial/
        """
        self.model = Sequential()
        self.model.add(Embedding(max_features, embed_size, weights=[self.embedding_matrix], trainable=True))
        self.model.add(Bidirectional(LSTM(256, dropout=0.15, recurrent_dropout=0.15)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='sigmoid')) 
        self.model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
        print(self.model.summary())
        
    def __trainModel(self):
        """ This method train the model """
        X_train, X_test, y_train, y_test = train_test_split(self.train_x, self.train_y, test_size=0.33, random_state=42)
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                       callbacks=[roc_callback(training_data=(X_train, y_train), validation_data=(X_test, y_test))])
        
    def __predictModel(self):
        self.predictions = self.model.predict(self.test_x, batch_size=batch_size, verbose=1)
        
        
    def __resultGeneration(self):
        """ This method generates result for the test file"""
        submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
        submission[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = self.predictions
        submission.to_csv('submission.csv', index=False)
        print(os.listdir("."))
    
    def run(self):
        self.__loadData()
        self.__cleanData()
        self.__toknizeData()
        self.__generateEmbeddingMatrix()
        self.__createModel()
        self.__trainModel()
        self.__predictModel()
        self.__resultGeneration()
        
        
        
if __name__ == '__main__' :
    obj = CommentPrediction()
    obj.run()
        
        


# In[ ]:




