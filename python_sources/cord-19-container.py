from cord_19_word_cleaner import post, lemmatize, valid_token
from cord_19_text_cleaner import Cleaner
from cord_19_lm import mle

from nltk.tokenize import sent_tokenize, word_tokenize

from gensim import matutils
from nltk import SimpleGoodTuringProbDist, FreqDist
import numpy as np
import pandas as pd
import json

import itertools
import re
import datetime

dictionary = None


class Sentence():
    ws_reg = re.compile(r'\s+')
    
    def __init__(self, text):
        # ~ Original text
        self.original_text = self.ws_reg.sub(' ', text)
        
        # List of str
        self._text = []
        
        # Lazy init
        self._bow = None
        # Sentence text tokens id
        self._tokensid = None
        # Unique tokens id
        self._tokensid_set = None
        
    def tokenize(self):
        # Rewrite proprely, at the right place (tokenizer class)
        self.text[:] = word_tokenize(self.original_text, preserve_line=True)
        self.text[:] = [token.lower() for token in self.text]
        self.text[:] = [post(token) for token in self.text]
        self.text[:] = [lemmatize(token) for token in self.text]
        self.text[:] = [token for token in self.text if valid_token(token)]
    
    @property
    def tokensid(self):
        if self._tokensid is None:
            self._tokensid = [tokenid for tokenid, _ in self.bow]
        return self._tokensid
    
    @property
    def tokensid_set(self):
        if self._tokensid_set is None:
            self._tokensid_set = set(self.tokensid)
        return self._tokensid_set
    
    @property
    def bow(self):
        if self._bow is None:
            self._bow = dictionary.doc2bow(self.text)
        return self._bow
    
    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        self._text = text
    
    def __hash__(self):
        return hash(self.original_text)

    def __eq__(self, other):
        return self.original_text == other.original_text
    
    def __repr__(self):
        return self.original_text


class Document():
    def __init__(self, txt_blocks):
        txt_blocks = txt_blocks if isinstance(txt_blocks,list) else [txt_blocks]
        
        # Lazy init
        # Document BoW
        self._bow = None
        # Document text tokens id
        self._tokensid = None
        # Unique tokens id
        self._tokensid_set = None
        
        self.sentences = list(set([Sentence(sentence) for txt in txt_blocks for sentence in sent_tokenize(txt)]))
    
    def tokenize(self):
        for sent in self.sentences:
            sent.tokenize()
    
    """
    Commun with Sentence class, merge them.
    """
    @property
    def bow(self):
        if self._bow is None:
            self._bow = dictionary.doc2bow(self.text)
        return self._bow
    
    @property
    def tokensid(self):
        if self._tokensid is None:
            self._tokensid = [tokenid for tokenid, _ in self.bow]
        return self._tokensid
    
    @property
    def tokensid_set(self):
        if self._tokensid_set is None:
            self._tokensid_set = set(self.tokensid)
        return self._tokensid_set
    """
    End commun
    """
    
    @property
    def text(self):
        return list(itertools.chain(*[sentence.text for sentence in self]))
    
    def __iter__(self):
        return iter(self.sentences)
            
    def __len__(self):
        return len(self.sentences)


# Paper ex FileReader class, they are so different now but it helps!
# https://www.kaggle.com/maksimeren/covid-19-literature-clustering
class Paper(Document):
    def __init__(self, txt_blocks=None):
        pass
    
    def get_content(self, INPUT_DIR):
        with open(f'{INPUT_DIR}/{self.file_path}') as file:
            content = json.load(file)
            return content
    
    def add_entry(self, container, text, cleaner=None, is_bib=False, min_len=24, max_len=2e4):
        text = text.strip()
        
        n_chars = len(text)
        if n_chars>min_len and n_chars<max_len:
            if cleaner is not None:
                text = cleaner.clean(text, is_bib)
            container.add(text)
            
            return text
        return None
    
    def from_json(self, row, INPUT_DIR):
        """
        Parse JSON paper
        Better to put this function away from this class
        
        Parameters
            row : Pandas Series (title, doi, abstract, publish_time, pmc_json_files)
        returns
            Paper
        """
        
        # Paper text blocks
        txt_blocks = set()
        
        self.file_path = row.pmc_json_files
        self.doi = row.doi
        if 'cord_uid' in row:
            self.cord_uid = row.cord_uid
        
        # Epoch time
        self.publish_time = (row.publish_time - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        
        
        # Old code, now all papers are recent
        pub_date = datetime.datetime.fromtimestamp(self.publish_time).date()
        is_recent = pub_date > datetime.date(2019, 12, 1)
        cleaner = Cleaner(is_recent)
        
        self.title = self.add_entry(txt_blocks, row.title, cleaner)
        self.abstract = self.add_entry(txt_blocks, row.abstract, cleaner)
        
        if self.file_path != '':
            content = self.get_content(INPUT_DIR)

            for entry in content['body_text']:
                self.add_entry(txt_blocks, entry['section'], cleaner) #sub title
                self.add_entry(txt_blocks, entry['text'], cleaner)

            for entry in content['ref_entries']:
                self.add_entry(txt_blocks, content['ref_entries'][entry]['text'], cleaner)

            """
            Not used now
            We keep bib entries to build a graph, and set PageRank
                We need prior paper relevance, can be combination of authority and newest
            Make it outside this class
            Remember to not change papers title use 'cleaner.clean(text, is_bib=True)'
            """
            bib_entries = set()
            for entry in content['bib_entries']:
                self.add_entry(bib_entries, content['bib_entries'][entry]['title'], cleaner, is_bib=True)
            self.bib_entries = list(bib_entries)
        
        # When debuging keep paper text...
        #self.txt_blocks = list(txt_blocks)
        
        super(Paper, self).__init__(list(txt_blocks))
        self.tokenize()
        
        return self
    
    def __repr__(self):
        return f'{self.title}:\n{self.abstract[:200]}...\n'


class Corpus:
    """
    Corpus class
    """
    def __init__(self, docs):
        # List of Document
        self.docs = docs
        self.dictionary = dictionary
        
        """
        Lazy init
        Corpus Term frequency & Term relative frequency
        Shape [n, m]
          n = #documents
          m = #terms
        """
        self._TF = None
        self._TRF = None
        
        self._p_coll = None

    @property
    def p_coll(self):
        """Maximum likelihood estimate of word w in the collection.
        """
        
        if not hasattr(self, '_p_coll'):
            self._p_coll = None
        
        if self._p_coll is None:
            self._p_coll = mle(self.TF)
            
        return self._p_coll
        
        
    @property
    def TF(self):
        if self._TF is None:
            # Like in gensim:docsim:SparseMatrixSimilarity
            # But transform to dense array... explain!
            self._TF = matutils.corpus2csc([doc.bow for doc in self], num_terms=len(dictionary)).T
            self._TF = np.asarray(self._TF.todense()) # currently no-op, CSC.T is already CSR
        
        return self._TF
    
    @property
    def TRF(self):
        if self._TRF is None:
            self._TRF = self.TF/self.TF.sum(axis=1).reshape(-1,1)
            
            # Will do next Simple Good Turing
            '''
            self._TRF = np.zeros_like(self.TF)
            
            unreliables = 0
            for i, paper in enumerate(self):
                x = dict(paper.bow)
                fd = FreqDist(x)
                p = SimpleGoodTuringProbDist(fd)
                if p._slope >= -1:
                    unreliables += 1
                for k in x.keys():
                    self._TRF[i,k] = p.prob(k)
                    
            if unreliables > 0:
                print(f'SimpleGoodTuring did not find a proper best fit for {unreliables}/{len(self)} documents, we handle it!')
            '''
        return self._TRF
    
    @property
    def bow(self):
        return [doc.bow for doc in self.docs]
    
    @property
    def text(self):
        return [doc.text for doc in self.docs]
    
    # https://gaopinghuang0.github.io/2018/11/17/python-slicing
    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return Corpus([self.docs[i] for i in range(start, stop, step)])
        elif isinstance(key, list):
            assert len(key)>0, f'Empty list provided'
            assert type(key[0]) == int, f'Only list of integers supported'
            
            return Corpus([self.docs[i] for i in key])
        elif isinstance(key, int):
            return self.docs[key]
        else:
            raise Exception(f'Invalid argument type: {type(key)}')

    def __iter__(self):
        return iter(self.docs)
            
    def __len__(self):
        return len(self.docs)
