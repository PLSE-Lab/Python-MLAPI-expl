#!/usr/bin/env python
# coding: utf-8

# # Building a paragraph search tool using word2vec
# This notebook builds a simple, general purpose, search tool based on word embeddings. We start by training word embeddings using word2vec, then calculate weighted average vectors for each paragraph in each of the papers, and finally use nearest neighbours to find related content. 
# 
# Training our own embeddings has a significant advantage in that the corpus we are dealing with has highly specific vocabulary containing many words that are unlikely to appear in standard pretrained embeddings. The disadvantage is that we don't have very much text with which to train the embeddings, and the vocabulary is large.
# 
# A few points about the general approach and assumptions I'm making:
# - I do some light cleaning of the text, e.g. removing url and doi links, and references numbers; but including other numbers. Additional cleaning rules might be justified, but I'm not familiar enough with the raw data to know what these should be.
# - I don't bother removing stop words. This has a knock-on effect on word2vec training - it might be worth having a slightly larger 'window' to get a more meaningful context. 
# - I'm maintaining casing of the text. My justification is that these are scientific documents where case may well be important. (I'm not so sure about this assumption...).
# - The vector for a chunk of text (sentence, paragraph, document, etc) is calculated in a manner analogous to TF-IDF - we take the average of the vectors for each word in the chunk, but weight each vector by the inverse of the log of the frequency of that word in the corpus (Note that this effectively removes stopwords from the weighted average). 

# Running the entire preprocessing and word2vec training takes about an hour. If you just want to test the interactive search tool, set `PREPROCESS_AND_TRAIN = False` below. The script will then load the saved versions of the embeddings and paragraph vectors.

# In[ ]:


PREPROCESS_AND_TRAIN = True


# In[ ]:


import numpy as np 
import pandas as pd 
import os
from glob import glob
from tqdm.auto import tqdm
import json
import re
from unidecode import unidecode
from collections import Counter
import pickle
import gensim 
import logging
import sys

# We create a logger handler so that gensim outputs its logging to stdout, which will then appear in the notebook cells.
# This is nice to have to be able to see the progress of the training process
logger = logging.getLogger()

if logger.hasHandlers():
        logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s : %(levelname)s : %(message)s'))
logger.addHandler(console_handler)
logger.info('hi')


# These are some helper functions for cleaning and tokenising the texts. Our aim is to clean up the text a bit, but without removing what could be important details in scientific papers - e.g. numbers. However, the text does include lots of footnote reference numbers, so we do try to remove those. 

# In[ ]:


# Regex used for cleaning and tokenisation
space = re.compile('\s+')
reference = re.compile(r'[\(\[]\d+(, ?\d+)?[\)\]]')
links = re.compile(r'https?://\S+')
sentence  = re.compile(r'(\S{3,})[.!?]\s')
hyphen_1 = re.compile(r'([A-Z0-9])\-(.)')
hyphen_2 = re.compile(r'(.)\-([A-Z0-9])')

license_phrases = [r'\(which was not peer-reviewed\)',
                    'The copyright holder for this preprint',
                    'It is made available under a is the author/funder',
                    'who has granted medRxiv a license to display the preprint in perpetuity',
                    'medRxiv preprint',
                    r'(CC(?:-BY)?(?:-NC)?(?:-ND)? \d\.\d (?:International license)?)',
                    'Submitted manuscript.', 
                   'Not peer reviewed.']

license_phrases = re.compile('|'.join(license_phrases), re.I)
PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE) # from gensim - removes digits - keeps only other alpha numeric and tokenises on everything else
PAT_ALL = re.compile(r'((\d+(,\d+)*(\.\d+)?)+|([\w_])+)', re.UNICODE) # Includes digits - tokenises on space and non alphanumeric characters (except _)

def clean_text(text):
    text = text.replace('\t', ' ').replace('\n', ' ')
    text = sentence.sub(r'\1 _SENT_ ', text)
    text = text.replace('doi:', ' http://a.website.com/')
    text = unidecode(text) # converts any non-unicode characters to a unicode equivalent
    text = hyphen_1.sub(r'\1\2', text)
    text = hyphen_2.sub(r'\1\2', text)
    text = links.sub(' ', text)
    text = license_phrases.sub(' ', text)
    text = reference.sub(' ', text)
    text = space.sub(' ', text)

    return text.strip()

def fetch_tokens(text, reg_pattern):
    for match in reg_pattern.finditer(text):
        yield match.group()

def tokenise(text, remove_stop=False, lowercase=False, include_digits=True):
    text = clean_text(text)
    
    if lowercase:
        text = text.lower()
    
    if include_digits:
        tokens = list(fetch_tokens(text, reg_pattern=PAT_ALL))
    else:
        tokens = list(fetch_tokens(text, reg_pattern=PAT_ALPHABETIC))
            
    if remove_stop:
        return ' '.join([x for x in tokens if x.lower() not in stopWords])
    else:
        return ' '.join(tokens)


# Let's test the cleaning and tokenisation on a sample bit of text. 
# 
# Note that we try to add a special token '\_SENT\_' between sentences to give us the option to split the paragraphs into sentences later on.

# In[ ]:


test_text = """A few test sentences for the tokeniser and cleaner.
With a web link http://something.co.uk/ and including a large number 100,000.50 and small ones 5.5.
But not including footnote references (10, 34) and [23].
Remove hyphens from Covid-19 and MERS-SARS to keep these as one token;
But add spaces for hyphenated-lowercase-words to keep the individual words as tokens."""


# In[ ]:


print(tokenise(test_text))


# These look OK, so now we will process all the JSON files to extract the body text of the papers and tokenise it. We will save everything into a pandas DataFrame for convenience. 
# 
# In addition to the body text, we will extract reference text and add it to the relevant paragraph. Where it doesn't appear to be linked to any paragraph (presumably a mistake in the extraction from the original document), we simple add the reference text as an extra paragraph. 

# In[ ]:


LICENSE_TYPES = ['comm_use_subset',
                 'noncomm_use_subset',
                 'pmc_custom_license',
                 'biorxiv_medrxiv']

# DATA_PATH = '/kaggle/input/CORD-19-research-challenge/2020-03-13'
DATA_PATH = '/kaggle/input/CORD-19-research-challenge'
PRE_PROCESSED_PATH = '/kaggle/input/paragraph-search-using-word2vec'


# In[ ]:


def iter_info():
    """
    Custom iterator to extract paragraphs from body text and reference text, alongside some of the document's meta data 
    """
    for licence in LICENSE_TYPES:
        for root, dirs, files in os.walk(os.path.join(DATA_PATH, licence)):
            for name in files: 
                fname = os.path.join(root, name)
                document = json.load(open(fname))
            
                refs = {}
                for para_num, para in enumerate(document['body_text']):
                    paragraph = {'title': document['metadata']['title'],
                                'id': document['paper_id'],
                                'contains_refs': False,
                                'section': para['section'],
                                'para_num': para_num,
                                'para_text': para['text'],
                                'licence': licence}

                    # check for references within this paragraph, but only add text once per paragraph
                    for ref in para['ref_spans']:
                        if ref['ref_id'] in document['ref_entries'].keys() and ref['ref_id'] not in refs.keys():
                            refs[ref['ref_id']] = True
                            paragraph['contains_refs'] = True 
                            paragraph['para_text'] += ' _STARTREF_ ' + document['ref_entries'][ref['ref_id']]['text'] + ' _ENDREF_ '
                    yield paragraph

                # Add any references text that has been missed
                for ref_id, ref_element in document['ref_entries'].items():
                    if ref_id not in refs.keys():
                        paragraph = {'title': document['metadata']['title'],
                                    'id': document['paper_id'],
                                    'contains_refs': False,
                                    'para_num': ref_id,
                                    'para_text': ref_element['text'],
                                    'section': 'References',
                                    'licence': licence}
                        yield paragraph


# In[ ]:


if PREPROCESS_AND_TRAIN:
    all_data = []
    texts = iter_info()
    for cc, t in tqdm(enumerate(texts)):
        t['tokenised'] = tokenise(t['para_text'])
        all_data.append(t)
#         if cc == 50000:
#             break
    # Convert to dataframe and save
    all_data = pd.DataFrame(all_data)
    all_data.to_pickle('CORD_19_all_papers.pkl')
else:
    all_data = pd.read_pickle(os.path.join(PRE_PROCESSED_PATH, 'CORD_19_all_papers.pkl'))


# In[ ]:


all_data.head()


# ## Train word embeddings 

# First we create an iterator which will yield lists of tokens to pass to the word2vec training.

# In[ ]:


class Paragraphs(object):
    def __init__(self, corpus_df, min_words_in_sentence=10):
        self.corpus_df = corpus_df
        self.min_words = min_words_in_sentence

    def __iter__(self): 
        for tot_paras, para in enumerate(self.corpus_df.tokenised, 1):
            if (tot_paras % 100000 == 0):
                logger.info("Read {0} paras".format(tot_paras))
            word_list = para.replace('\n', ' ').split(' ')
            if len(word_list) >= self.min_words:
                yield word_list


# Before training we will count how many times each word occurs in the corpus. We need these corpus frequencies later when calculating weighted average document, paragraph or sentence vectors.

# In[ ]:


if PREPROCESS_AND_TRAIN:
    documents = Paragraphs(all_data, min_words_in_sentence=10)

    vocab = Counter()
    for line in tqdm(documents):
        vocab.update(line)

    # Save these word frequencies
    vocab = dict(vocab)
    pickle.dump(vocab, open('covid_vocab_frequencies.pkl', 'wb'))
else:
    vocab = pickle.load(open(os.path.join(PRE_PROCESSED_PATH, 'covid_vocab_frequencies.pkl'), 'rb'))


# In[ ]:


vocab['COVID19']


# Let's look at the most common words in the corpus, and the size of the vocabulary.

# In[ ]:


print(f'{len(vocab)} unique tokens in total.')
print(f'Most common words are: {Counter(vocab).most_common(10)}')
print(f'COVID19 mentioned {vocab["COVID19"]} times.')


# Pretty much what we expected given that we haven't removed stopwords.

# ### Create and train a gensim word2vec model

# The next bit of code trains a word2vec model. There are lots of hyperparamter options. The key ones are:
# - _size_: the number of dimensions that the embedding will have;
# - _window_: the model looks this many tokens either side of a central token. These window tokens provide the context with which to predict the central word (or vica versa under the skipgram approach);
# - *min_count*: the minimum number of times a token must appear in the corpus in order to have an embedding;
# - _epoch_: the number of times the model passes over the corpus when training.

# In[ ]:


EMBEDDING_DIMS = 100


# In[ ]:


if PREPROCESS_AND_TRAIN:
    # build vocabulary within Gensim
    model = gensim.models.Word2Vec(
        size=EMBEDDING_DIMS,
        window=6,
        min_count=5)

    model.build_vocab(documents)
    print(f'Number of paragraphs in corpus: {model.corpus_count}')


# In[ ]:


if PREPROCESS_AND_TRAIN:
    model.train(sentences=documents, 
            epochs=5, 
            total_examples=model.corpus_count, 
            total_words=model.corpus_total_words)
    model.save('covid_w2v')    
else:
    model = gensim.models.Word2Vec.load(os.path.join(PRE_PROCESSED_PATH, 'covid_w2v'))

