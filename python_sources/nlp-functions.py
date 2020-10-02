# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from gensim.models import Word2Vec, Doc2Vec, FastText
from gensim.models.doc2vec import TaggedDocument
from gensim.models.phrases import Phraser, Phrases

from mpl_toolkits.mplot3d import Axes3D

from collections import defaultdict
import matplotlib.pyplot as plt
import tensorflow as tf




# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

def pre_process(df):
    
    """ Preprocess
        ----------

        The main preprocess pipeline for my NLP
        projects. This functional pipeline has
        proven affective over a number of NLP
        tasks
         and continues to serve me well (that
        is until the great rising when functions
        will rule the world).

        A number of steps are completed, accompanied
        by a small comment explaining what is happening.

        If you find this pipeline useful please use it. 

        Input: Df - Pandas Dataframe 
               Dataframe with a text column that needs
               to be cleaned. 

               col_name - String
               The name of the column that needs to be
               cleaned.
    
        Output: Df - Pandas Dataframe
                Dataframe that includes a new cleaned 
                "col_name" column. THis function does
                not remove the original feature. 

    
    """
    
    # First thing to do is reset the index for all entries, this needs to be done for
    # the final step where a list of lemmatised words is built up and then used to 
    # replace the text value in the twitter dataframe. 
    
    df = df.reset_index()
    
    # Next change is lower case using apply and lambda function
    
    df['text'] = df['text'].apply(lambda x: x.lower())
    
    # Next we need to tokenise the text in the tweets then remove them from list.
    
    df['text'] = df['text'].apply(lambda x: word_tokenize(x))

    # Cool, so all messages are now stored in seperate lists that have been tokenised
    
    # Now to remove stopwords, lemmatisation and stemming
    
    # We need p.o.s (part of speech) tags to understand if its a noun or a verb
    
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    
    # Next iterate over each entry in the text column
    
    for index, entry in enumerate(df['text']):
        
        # Create list to hold final lemmetized words
        
        final_words = ""
        
        # Instantiate the lemmatizer
        
        word_lem = WordNetLemmatizer()
        
        # Loop over the entry in the text column.
        
        for word, tag in pos_tag(entry):
            
            # If the word is not in the stopword and is not alphanumeric add it to final words
            
            if word not in stopwords.words('english') and word.isalpha() and word != 'http':
                
                # Run our lemmatizer on the word.
                
                word_final = str(word_lem.lemmatize(word, tag_map[tag[0]]))
                
                # Now add to final words
                
                final_words += word_final + " "
        
        # Once we've looped over all words in the entry we replace that entry with our
        # final words list. 
        
        df.loc[index, 'text'] = str(final_words)
        
    # Need a final adjustment to drop the newly created index column
    
    df = df.drop(columns=['index'])
    
    
    return df


def plot_vectors(svd_df, disaster_tweets):
    
    """ Plot Vectors
        ------------

        Creates a static 3d plot

        Input
        -----

        pcs_df: Pandas Dataframe
                Dataframe of pca_values
                Note: this function is
                not flexible.

                TODO: Generlise this. 

        Output
        ------

        Displays 3d plot.
    
    
    """
    
    fig = plt.figure()

    fig = plt.figure(figsize=(25, 20))
    ax = fig.add_subplot(111, projection='3d')

    xs = svd_df['svd_one']
    ys = svd_df['svd_two']
    zs = svd_df['svd_three']

    ax.scatter(xs, ys, zs, c=disaster_tweets['target'], cmap='plasma')
    plt.title("Visualisation of Word Embeddings Using Singular Value Decomposition: Hue - Target")

        
    ax.set_xlabel("SVD 1")
    ax.set_ylabel("SVD 2")
    ax.set_zlabel("SVD 3")

    plt.show()
    
def drop_columns(df):
    
    """ Drop Columns Function
        
        input: df
               DataFrame of disaster tweets
    
        output: df
                Transformed dataframe with columns dropped.
    
    """
    
    df = df.drop(columns=['id', 'keyword', 'location'])
    return df

def encode_train(text_tensor, label):
    text = text_tensor.numpy()[0]
    encoded_text = encoder.encode(text)
    return encoded_text, label

def encode_map_fn_train(text, label):
    return tf.py_function(encode_train, inp=[text, label], Tout=(tf.int64, tf.int64))

def encode_unseen(text_tensor):
    text = text_tensor.numpy()[0]
    encoded_text = encoder.encode(text)
    return encoded_text, 1

def encode_map_fn_unseen(text):
    return tf.py_function(encode_unseen, inp=[text], Tout=(tf.int64, 1))