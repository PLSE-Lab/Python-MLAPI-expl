'''
Author: Varun D N, New York University, Center for Data Science
Email: vdn207@nyu.edu

This project is to deploy LDA on the US Presidential elections debate transcipts. 

CAVEAT: The results are not completely convincing at the moment. I am open to
any suggestions. The code can be used by anyone for obtaining better results.

The dataset was obtained from Kaggle datasets. https://www.kaggle.com/mrisdal/2016-us-presidential-debates

DATASET DESCRIPTION 
The dataset contains the statements made by the primary speakers during a debate which include:
	- Presidential/Vice-Presidential candidates
	- Moderators
	- Audience

This dataset has 1,025 statements in total along with a timestamp for every statement.  

The distribution of speakers in the dataset looks as follows:
	Trump         224
	Clinton       158
	Pence         134
	Kaine         124
	Holt           98
	Quijano        76
	Cooper         75
	Raddatz        62
	CANDIDATES     37
	Audience       29
	QUESTION        8

I will be using the Python library 'gensim' to perform topic modeling. 

'''

'''
This program takes the input in the following order:
	NUM TOPICS - int 
	NUM PASSES - int 
	OUTPUT FILE NAME - string

The program outputs the top 10 words in each topic. These numbers can be 
configured by tweaking the variable values. 

'''

import gensim
from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS

import sys
import pandas as pd 

def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

def iter_text(statements):

    '''
    'statements' - list of the debate statements collected from the corpus.
    '''
    
    for text in statements:
        tokens = tokenize(text.strip())
        yield tokens


num_topics = 3    # int(sys.argv[1])
num_passes =  10  # int(sys.argv[2])
output_file_name = 'lda_3topics' # sys.argv[3]

debate = pd.read_csv('../input/debate.csv', encoding = "ISO-8859-1")
transcipt = debate.Text.values.tolist()

print ("Tokenizing......")
doc_stream = (tokens for tokens in iter_text(transcipt))


print ("Building dictionary......")
# Dictionary
id2word = gensim.corpora.Dictionary(doc_stream)
id2word.save('debate.dictionary')

streamed_corpus = [id2word.doc2bow(doc) for doc in doc_stream]

print ("Serializing.....")
# Saving the serialized corpus
gensim.corpora.MmCorpus.serialize('wiki_bow.mm', streamed_corpus)

# Loading the saved corpus. Avoids building the corpus dictionary every time.
mm_corpus = gensim.corpora.MmCorpus('wiki_bow.mm')

clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus)

print ("")
print ("Learning topics from the corpus......")

print ("NUM TOPICS: ", num_topics)
print ("NUM PASSES: ", num_passes)

lda_model = gensim.models.LdaModel(clipped_corpus, num_topics = num_topics, id2word = id2word, passes = num_passes)

# If you want to save the LDA model
# lda_model.save(output_file_name + '_' + str(num_topics) + 't_' + str(num_passes) + 'p.model')

print (lda_model.print_topics(num_words = 10))



