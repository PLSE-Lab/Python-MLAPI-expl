#!/usr/bin/env python
# coding: utf-8

# # Setup Code

# In[ ]:


from IPython.utils import io
with io.capture_output() as captured:
    get_ipython().system('pip install scispacy')
    get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz')
    get_ipython().system('pip install whoosh')


# In[ ]:


import pandas as pd
import numpy as np
import os
from collections import defaultdict


# In[ ]:


import whoosh
from whoosh.qparser import *
from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED
from whoosh.analysis import StandardAnalyzer
from whoosh import index

from whoosh.analysis import Tokenizer, Token
from whoosh import highlight


# In[ ]:


from IPython.core.display import display, HTML


# In[ ]:


# def set_column_width(ColumnWidth, MaxRows):
#     pd.options.display.max_colwidth = ColumnWidth
#     pd.options.display.max_rows = MaxRows
#     print('Set pandas dataframe column width to', ColumnWidth, 'and max rows to', MaxRows)

# # this has the user interactively set the Pandas dataframe dimensions
# def user_sets_pandas_df_dimensions():
#     interact(set_column_width, 
#            ColumnWidth=widgets.IntSlider(min=50, max=400, step=50, value=200),
#            MaxRows=widgets.IntSlider(min=50, max=500, step=100, value=100));


# # Load Data

# In[ ]:


df = pd.read_csv('../input/cord-19-create-dataframe/cord19_df.csv')

df.shape


# In[ ]:


df.columns


# In[ ]:


import en_core_sci_lg

# medium model
nlp = en_core_sci_lg.load(disable=["tagger", "parser", "ner"])
nlp.max_length = 2000000

# New stop words list 
customize_stop_words = [
    'doi', 'preprint', 'copyright', 'org', 'https', 'et', 'al', 'author', 'figure', 'table',
    'rights', 'reserved', 'permission', 'use', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 'al.', 'Elsevier', 'PMC', 'CZI',
    '-PRON-', 'usually',
    r'\usepackage{amsbsy', r'\usepackage{amsfonts', r'\usepackage{mathrsfs', r'\usepackage{amssymb', r'\usepackage{wasysym',
    r'\setlength{\oddsidemargin}{-69pt',  r'\usepackage{upgreek', r'\documentclass[12pt]{minimal'
]

# Mark them as stop words
for w in customize_stop_words:
    nlp.vocab[w].is_stop = True


# In[ ]:


def my_spacy_tokenizer(sentence):
    # lowercase lemma, startchar and endchar of each word in sentence
    return [(word.lemma_.lower(), word.idx, word.idx + len(word)) for word in nlp(sentence) 
            if not (word.like_num or word.is_stop or word.is_punct or word.is_space or len(word)==1)]


# In[ ]:


class SpacyTokenizer(Tokenizer):
    """
    Customized tokenizer for Whoosh
    """
    
    def __init__(self, spacy_tokenizer):
        self.spacy_tokenizer = spacy_tokenizer
    
    def __call__(self, value, positions = False, chars = False,
                 keeporiginal = False, removestops = True,
                 start_pos = 0, start_char = 0, mode = '', **kwargs):
        """
        :param value: The unicode string to tokenize.
        :param positions: Whether to record token positions in the token.
        :param chars: Whether to record character offsets in the token.
        :param start_pos: The position number of the first token. For example,
            if you set start_pos=2, the tokens will be numbered 2,3,4,...
            instead of 0,1,2,...
        :param start_char: The offset of the first character of the first
            token. For example, if you set start_char=2, the text "aaa bbb"
            will have chars (2,5),(6,9) instead (0,3),(4,7).
        """
        
        assert isinstance(value, str), "%r is not unicode" % value
        
        spacy_tokens = self.spacy_tokenizer(value)
        
        t = Token(positions, chars, removestops=removestops, mode=mode)

        for pos, spacy_token in enumerate(spacy_tokens):
            t.text = spacy_token[0]
            if keeporiginal:
                t.original = t.text
            t.stopped = False
            if positions:
                t.pos = start_pos + pos
            if chars:
                t.startchar = start_char + spacy_token[1]
                t.endchar = start_char + spacy_token[2]
            yield t


# In[ ]:


#get hardcoded schema for the index
def get_search_schema(analyzer=StandardAnalyzer()):
    schema = Schema(paper_id=ID(stored=True),
                    doi = ID(stored=True),
                    authors=TEXT(analyzer=analyzer),
                    journal=TEXT(analyzer=analyzer),
                    title=TEXT(analyzer=analyzer, stored=True),
                    abstract=TEXT(analyzer=analyzer, stored=True),
                    methods=TEXT(analyzer=analyzer)
#                     body_text = TEXT(analyzer=analyzer)
                   )
    return schema

def add_documents_to_index(ix, df):
    #create a writer object to add documents to the index
    writer = ix.writer()

    #now we can add documents to the index
    for _, doc in df.iterrows():
        writer.add_document(paper_id = str(doc.paper_id),
                            doi = str(doc.doi),
                            authors = str(doc.authors),
                            journal = str(doc.journal),
                            title = str(doc.title),
                            abstract = str(doc.abstract),
                            methods = str(doc.methods)
#                             body_text = str(doc.body_text)                            
                           )
 
    writer.commit()

    return

def create_search_index(search_schema):
    if not os.path.exists('indexdir'):
        os.mkdir('indexdir')
        ix = index.create_in('indexdir', search_schema)
        add_documents_to_index(ix, df)
    else:           
        #open an existing index object
        ix = index.open_dir('indexdir')
    return ix


# # Create Search Index

# In[ ]:


# import shutil
# shutil.rmtree("/kaggle/working/indexdir") # to delete existing index


# In[ ]:


my_analyzer = SpacyTokenizer(my_spacy_tokenizer)


# In[ ]:


search_schema = get_search_schema(analyzer=my_analyzer)
ix = create_search_index(search_schema)


# In[ ]:


parser = MultifieldParser(['title', 'abstract'], schema=search_schema, group=OrGroup.factory(0.9))
parser.add_plugin(SequencePlugin())


# # Get Search Results

# In[ ]:


df.set_index('paper_id', inplace=True)


# In[ ]:


def search(query, lower=1950, upper=2020, only_covid19=False, kValue=5):
    query_parsed = parser.parse(query)
    with ix.searcher() as searcher:
        results = searcher.search(query_parsed, limit = None)
        output_dict = defaultdict(list)
        for result in results:
            output_dict['paper_id'].append(result['paper_id'])
            output_dict['score'].append(result.score)
        
    search_results = pd.Series(output_dict['score'], index=pd.Index(output_dict['paper_id'], name='paper_id'))
    search_results /= search_results.max()
    
    relevant_time = df.publish_year.between(lower, upper)

    if only_covid19:
        temp = search_results[relevant_time & df.is_covid19]

    else:
        temp = search_results[relevant_time]

    if len(temp) == 0:
        return -1

    # Get top k matches
    top_k = temp.nlargest(kValue)
    
    return top_k


# In[ ]:


def searchQuery(query, lower=1950, upper=2020, only_covid19=False, attr=['paper_id', 'title', 'abstract', 'url', 'authors'], kValue=3):
    search_results = search(query, lower, upper, only_covid19, kValue)

    if type(search_results) is int:
        return []
    #results = df.loc[search_results].reset_index()
    results = pd.merge(search_results.to_frame(name='similarity'), df, on='paper_id', how='left').reset_index() 

    return results[attr].to_dict('records')


# In[ ]:


search('title:hydroxychloroquine', kValue=3)


# In[ ]:


searchQuery('authors:drosten', only_covid19=True, kValue=2, attr=['authors', 'title'])


# In[ ]:


searchQuery("title:hydroxychloroquine AND methods:(randomized controlled trial)", only_covid19=True, kValue=2, attr=['title', 'abstract'])


# In[ ]:


searchQuery("authors:drosten NOT title:coronavirus", only_covid19=False, kValue=2, attr=['authors', 'title'])


# In[ ]:


searchQuery('doi:10.1101/2020.01.31.929042', only_covid19=True, kValue=3, attr=['doi', 'title'])


# In[ ]:


searchQuery('journal:(Studies in Natural Products Chemistry)', kValue=2, attr=['title', 'journal'])


# In[ ]:


user_query = 'abstract:(sars OR "sars-cov-2" OR coronavirus* OR ncov OR "covid-19" OR mers OR "mers-cov") NOT abstract:(animal OR equine OR porcine OR calves OR dog*) AND abstract:incubation AND abstract:("symptom onset" OR characteristics OR exposure)'


# In[ ]:


searchQuery(user_query, kValue=1, attr=['title', 'abstract'])


# # Highlight Excerpts

# In[ ]:


query_parsed = parser.parse('incubation period' )


# In[ ]:


with ix.searcher() as searcher:
    results = searcher.search(query_parsed, limit = 3)
    results.fragmenter = highlight.SentenceFragmenter(charlimit=100000, maxchars=200)
#     results.fragmenter = highlight.WholeFragmenter()
    for hit in results:
        print(hit['title'])
        display(HTML(hit.highlights("title", top=2)))
        display(HTML(hit.highlights("abstract", top=2)))

