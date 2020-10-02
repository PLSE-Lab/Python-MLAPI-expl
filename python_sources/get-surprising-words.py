# %% [code] {"_kg_hide-input":true,"_kg_hide-output":true}
## Imports (code & data)
import math
import itertools
import pandas as pd
import nltk

# forum-wide frequency info for identifying surprising words
FREQUENCY_TABLE = pd.read_csv("../input/kaggle-forum-term-frequency-unstemmed/kaggle_lex_freq.csv",
                              error_bad_lines=False)

# %% [code] {"_kg_hide-input":true,"_kg_hide-output":true}
## Utility functions
# get sample post info by #
def get_post_info_by_cluster(number, 
                             post_data,
                             cluster_object):
    """Post data in a dataframe, cluster_object from sklearn spectral clsutering"""
    return post_data[cluster_object.labels_ == number]

def post_info_preprocessing(cluster_index, post_data, cluster_object):
    """Custom preprocessing for our cluster text"""
    # create corpus from a cluster
    text = get_post_info_by_cluster(cluster_index, 
                                    post_data=post_data, 
                                    cluster_object=cluster_object)\
        .Message.astype(str).str.cat(sep=' ')

    # tokenize
    words = nltk.word_tokenize(text)

    # Remove single-character tokens (mostly punctuation)
    words = [word for word in words if len(word) > 1]

    # Remove numbers
    words = [word for word in words if not word.isnumeric()]

    # remove non-breaking space
    words = [word for word in words if word != "nbsp"]

    # Lowercase all words (default_stopwords are lowercase too)
    words = [word.lower() for word in words]
    
    return words

def get_cluster_saliency_dict(cluster_index, post_data, cluster_object):
    """Return dictionary & frequency distrobution for clusters"""
    words = post_info_preprocessing(cluster_index, post_data, cluster_object)
    
    # Calculate frequency distribution
    fdist = nltk.FreqDist(words)

    # create empty dictionary to store our info
    cluster_dict = dict() 

    # get saliency measures
    for word, frequency in fdist.most_common():
        saliency_measure_smoothed = math.log(frequency + 0.0001)/(math.log(fdist.most_common(1)[0][1] + 0.0001))
        cluster_dict[word] = saliency_measure_smoothed
        
    return cluster_dict, fdist

def get_surprising_words(cluster_index, post_data, cluster_object):
    """Returns words with higher normalized log frequency in cluster
    than in Kaggle forums overall"""
    cluster_dict, fdist = get_cluster_saliency_dict(cluster_index, post_data, cluster_object)
    
    words = []
    surprisal = []

    for word, _ in fdist.most_common():
        words.append(word)
        surprisal_measure = cluster_dict[word] - FREQUENCY_TABLE.saliency[FREQUENCY_TABLE.word == word]
        if surprisal_measure.empty:
            surprisal.append(cluster_dict[word] - .0001)
        else:
            surprisal.append(surprisal_measure.values[0])

    cluster_surprisal_measures = pd.DataFrame(list(zip(words, surprisal)), 
                                              columns=['Words', 'Surprisal']) 

    suprising_words = cluster_surprisal_measures.Words[cluster_surprisal_measures.Surprisal > 0]
    
    return suprising_words
