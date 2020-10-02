# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
"""Import crap starts here skip down to the next comment!"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gensim as gs 
import tensorflow as tf
import os
import sys
import datetime
from gensim.models import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.parsing.preprocessing import preprocess_string 
from gensim.parsing.preprocessing import  strip_tags 
from gensim.parsing.preprocessing import  remove_stopwords
from gensim.parsing.preprocessing import  strip_short
from gensim.parsing.preprocessing import  strip_multiple_whitespaces
from gensim.parsing.preprocessing import  strip_punctuation
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM, BatchNormalization, Dropout, Conv1D, Conv2D,Input, MaxPool1D, MaxPool2D, Flatten, Reshape, ConvLSTM2D
from tensorflow.keras.optimizers import SGD, Nadam
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.models import Model       # Keras is the new high level API for TensorFlow
import math
# Input data files are available in the "../input/" directry.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""The gravy starts here. This is basically a crappy conversion of input data to GLOVE embeddings attached in gloveicg fed into an LSTM."""
"""This is your cockpit for hyperparameter tuning and debugging ease without having to scroll down"""
WORD_DIM = 300 #dimension of the word embeddings to use, determines which glove embedding is read in, e.g. 100 = 100 dimensional word embedding 
PAD_LENGTH = 21 #the maximum sequence length, i.e. the maximum permitted sentence length for processed text
EPOCHS = 2 #number of epochs for training the lstm network 
BATCH_SIZE = 32 # batch size for training the lstm network 
BASE_NUM_FILTERS = 100 #the number of filters to use for the conv1d layer for the lst mnetwork
BASE_NUM_UNITS = 50 #the number of hidden units 
KERNEL_SIZE = 10 #kernel size for the 1D convolution in the network
VERBOSE = 0 #if set to 1 you will see your sea monkeys grow
OPTIMIZER = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999) #optimizer for your network 
def build_network(unit_multiplier = 1, num_lstm_stacks = 5):
    """
    Builds a stacked LSTM network without anything fancy, allows you to say how deep the stack is and the multiple of hidden units to use 
    
    Parameters: 
    
        unit_multiplier: a number to multiply the number of hidden units by 
        
        num_lstm_stacks: the number of lstm stacks, passing the whole sequence through an LSTM with unit_mulitplier* hidden units  
        
    Returns: 
    
        Keras LSTM stacked network 
    """
    print("got to training!")
    model = Sequential()
    model.add(Conv1D(filters= BASE_NUM_FILTERS*unit_multiplier, kernel_size = KERNEL_SIZE, activation='relu', input_shape=(PAD_LENGTH, WORD_DIM)))
    model.add(MaxPool1D())
    for i in range(1,num_lstm_stacks): 
        model.add(LSTM(units=BASE_NUM_UNITS*unit_multiplier, return_sequences = True, recurrent_dropout = .20, dropout = .20))
        model.add(BatchNormalization())
    model.add(LSTM(units=BASE_NUM_UNITS*unit_multiplier, return_sequences = False, recurrent_dropout = .20, dropout = .20))
    model.add(BatchNormalization())
    model.add(Dense(units=1, activation="sigmoid"))
    return model 
def sub_sample(df_input):
    """ 
    Performs undersampling for imbalanced samples where positive findings are underrepresented, a cheap hack for having too little data 
    
    Returns: A random sampling of df_input as a dataframe where target == 0 and target == 1 are balanced 
    
    """
    count_negative = len(df_input[df_input["target"] == 0])
    print("Number of negative samples", count_negative)
    count_positive = len(df_input[df_input["target"] == 1])
    print("Number of positive samples", count_positive)
    sample_fraction = count_positive/count_negative
    print("Resampling negative as fraction", sample_fraction)
    sample_zero = df_input[df_input["target"] == 0].sample(frac=sample_fraction, random_state = 20)
    sample_one = df_input[df_input["target"] == 1]
    result_frame = pd.concat([sample_zero, sample_one], axis = 0)
    result_frame = result_frame.sample(frac=1.0, random_state = 30).reset_index(drop=True)
    return result_frame
def read_in_train():
    """Reads in training data"""
    df_input = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
    return df_input
def read_in_test(): 
    """Reads in data to generate submission for"""
    df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
    return df_test
def read_in_glove():
    """
    Reads in the glove embeddings and stores them in a gensim model containing a mapping from words to WORD_DIM dimensional embeddings 
    If you run it once, it will save the model to the containers disk, if it doesn't exist, it will bitch then load it, it takes a little while
    WORD_DIM = 300 is a chunk of ram 
    
    Returns: a gensim model containing a mapping from words to WORD_DIM dimensional embeddings 
    """
    model = None
    path = "word2vec{}d.model".format(WORD_DIM)
    try:    
        model = KeyedVectors.load(os.path.abspath(path))
    except:
        print("could not load word2vec model, this is ok for a first run, or with new WORD_DIM ", sys.exc_info()[0])
        path_glove = '/kaggle/input/gloveicg/glove/Glove/glove.6B.{}d.txt'.format(WORD_DIM)
        print(path_glove)
        glove_file = datapath(path_glove)
        tmp_file = get_tmpfile(path)
        _ = glove2word2vec(glove_file, tmp_file)
        model = KeyedVectors.load_word2vec_format(tmp_file)
        model.save(path)
    return model
def pad_word_vector(word_vector,pad_length):
    """
    Pads to pad_length with 0s if shorter than pad_length, otherwise cuts off
    
    Parameters:
    
        word_vector: word embedding represented by np array of integers
        
        pad_length: the length to pad to if smaller, otherwise cut off size for longer sentences 
    
    Returns:
        
        vector of length pad_length where the original is preserved if <= length ,otherwise padded with 0s to pad_length
    """
    length = len(word_vector)
    if length >= pad_length: 
        return word_vector[:pad_length]
    else: 
        added_stuff = [np.zeros(WORD_DIM)]*(pad_length-length)
        word_vector.extend(added_stuff)
        return word_vector   
def preprocess_data(df_input, word_vectors, is_test_data = False):
    """
    Subsamples data if not test data, processes input text using gensim, returns tuple of text mapped to vector of word embeddings and target values
    
    Parameters:
    
        df_input: input dataframe 
        
        word_vectors: map from word to WORD_DIM dimensional embedding
        
        is_test_data: if for submission or not, deactivates subsampling for test data used for submission and returns ids instead of targets
        
    Returns: 
    
        returns tuple of text mapped to vector of word embeddings and target values or ids if for test_data
    """
    if is_test_data is False:
        df_input = sub_sample(df_input)
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, remove_stopwords, strip_multiple_whitespaces]
    feature_vectors = list()
    for text in df_input["text"]:
        processed = preprocess_string(text, CUSTOM_FILTERS)
        processed_text = list()
        for word in processed: 
            if word in word_vectors and word != 'http' and len(word) > 2: 
                processed_text.append(word_vectors[word])
        feature_vectors.append(processed_text)
    for i,vector in enumerate(feature_vectors): 
        feature_vectors[i] = pad_word_vector(vector,PAD_LENGTH)
    if is_test_data is False: 
        return (feature_vectors, df_input["target"])
    else: 
        return (feature_vectors, df_input["id"])
def flatten_inputs(x_total):
    """
    Flattens data, i.e. reshapes from three dimensions to two dimensions, so simple models can be used 
    """
    x_total = np.reshape(x_total,(x_total.shape[0],WORD_DIM*PAD_LENGTH))
    return x_total 
def convert_to_binary(predictions):
    """
    Converts data from proabilities to binary values
    """
    return np.round(predictions)
def simple_model():
    """
    Simple model for benchmarking the network 
    """
    return LogisticRegression(solver='liblinear')
def create_submission(model, word_vectors):
    """create submission file with test_data and save to container dir"""
    test_data = read_in_test()
    x_test, x_id  = preprocess_data(test_data, word_vectors, is_test_data = True)
    predictions = model.predict(np.array(x_test))
    predictions = convert_to_binary(predictions)
    submission_result = pd.concat([x_id, pd.Series(np.reshape(predictions,(predictions.shape[0],)), name = "target")],axis=1)
    submission_result.to_csv("submission.csv", index = False)
#deep frying commences here!!!!!
if __name__ == "__main__":
    df_input = read_in_train() #read in the dataframes 
    word_vectors = read_in_glove() #read in GloVe embeddings
    x_total,y_total = preprocess_data(df_input, word_vectors, is_test_data = False) #preprocess text
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=30)# split data preserving distributions of target == 0 and target == 1
    x_total = np.array(x_total)
    print("shape input", x_total.shape)
    y_total = np.array(y_total)
    split_generator = splitter.split(X=x_total,y=y_total)
    try: 
        train_indices, test_indices = next(split_generator)
    except:
        print("could not split model ", sys.exc_info()[1])
    print("split data")
    x_train, x_test, y_train, y_test = x_total[train_indices], x_total[test_indices], y_total[train_indices], y_total[test_indices]
    model = build_network(unit_multiplier= 3, num_lstm_stacks = 4)
    model.summary()
    try: 
        model.compile(optimizer=OPTIMIZER,loss='binary_crossentropy',metrics=['accuracy'])
    except:
        print("could not compile model ", sys.exc_info()[1])
    try:
        start_time = datetime.datetime.now()
        model.fit(x=x_train,y=y_train, epochs = EPOCHS, verbose = VERBOSE,batch_size = BATCH_SIZE)
        end_time = datetime.datetime.now()
        print("Training took:", (end_time-start_time).total_seconds(), " seconds for ", EPOCHS, " epochs")
    except: 
        print("could not train model ", sys.exc_info()[1])
    print("training network successful")
    predictions_network = convert_to_binary(model.predict(x=x_test))
    print("f1 score for network",f1_score(predictions_network,y_test))
    print("precision for network",precision_score(predictions_network, y_test))
    print("recall for network", recall_score(predictions_network, y_test))
    create_submission(model,word_vectors)
    print("successfully created submission")
    flattened_train = flatten_inputs(x_train)
    flattened_test = flatten_inputs(x_test)
    simple_model = simple_model()
    simple_model.fit(X=flattened_train,y=y_train)
    predictions_simple = simple_model.predict(flattened_test)
    print("f1 score for simple model", f1_score(predictions_simple,y_test))
    print("precision for simple model",precision_score(predictions_simple, y_test))
    print("recall for simple model", recall_score(predictions_simple, y_test))
 
# Any results you write to the current directory are saved as output.