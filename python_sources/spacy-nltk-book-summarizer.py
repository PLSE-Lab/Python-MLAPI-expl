# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import re
import itertools
import gensim
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

class topicModel:
    def __init__(self, path):
        self.path = path
        self.book = []
        self.chapter = []

    def print_data(self):
        with open(self.path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                print(line)
                
    def store_data(self):
        with open(self.path, 'r', encoding='utf-8', errors='ignore') as f:
            
            for line in f:
                self.book.append(line)
            return self
        
    def book_getChapters(self):
        print("Breaking down by chapters...")
        #get chapter blocks
        chapBlockStart = [i for i, elem in enumerate(self.book) if re.search('(chapter)',elem.lower())]
        #print(len(self.book))
        chapBlockEnd = chapBlockStart[1:]
        chapBlockEnd.append(len(self.book))
        #chapBlockEnd = [chapBlockStart[1:],len(self.book)] #creates lists of lists : [[1,2],[10]]
        #print(chapBlockEnd)
        #chapBlockEnd = list(itertools.chain.from_iterable(chapBlockEnd)) #flattens [1,2,10]
        chapBlocks = zip(chapBlockStart,chapBlockEnd)
        
        for chapter in chapBlocks:
            #print(chapter)
            self.chapter.append(self.book[chapter[0]:chapter[1]])
        return self.chapter
        
    def gutenberg_getBook(self):
        indices = [i for i, elem in enumerate(self.book) if re.search(r'(START|END) OF THIS PROJECT GUTENBERG EBOOK THE SCARLET PIMPERNEL',elem)]
        #print(indices)
        self.book = self.book[indices[0]:indices[1]] #removes all info regarding the project
        return self.book_getChapters()
        
class topicModelPrep:
    
    def __init__(self, sents):
        self.sents = sents
        self.texts = []
        self.bigrams = []
        self.trigrams = []
        self.textsLemmatized = []
        print("preprocessing...")
        
    def sent_to_words(self):
        print("sents to words...")
        
        for sentence in self.sents:

            self.texts = gensim.utils.simple_preprocess(str(self.sents), deacc=True)  # deacc=True removes punctuations
            return self
    
    def remove_stopwords(self):
        print("removing stopwords...")
        stop_words = stopwords.words('english')
        stop_words.extend(['chapter',''])
        self.texts = [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in self.texts]
        self.texts = [word for word in self.texts if len(word)>0]
        return self

    def make_bitrigrams(text,minCount,thres):
        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(text, min_count=minCount, threshold=thres) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[text], threshold=thres)  
        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        bigrams = [bigram_mod[doc] for doc in text]
        trigrams = [trigram_mod[bigram_mod[doc]] for doc in text]
        return bigrams,trigrams
    
    def lemmatization(self, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        print("lemmatizing...")
        for word in self.texts:
            doc = nlp(" ".join(word)) 
            self.textsLemmatized = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        return self

class topicModelFeatExtract:
    
    def __init__(self,tokens):
        self.tokens = tokens
        self.spacyTag = []
        self.spacyToken = []
        self.df = pd.DataFrame()
        self.nltkBigram = []
        self.nltkTrigram = []
        
    def nltk_grams(self):
        check2 = " ".join(itertools.chain.from_iterable(self.tokens))
        nltkPairs = nltk.pos_tag(nltk.word_tokenize(check2))
        #print(nltkPairs[:5])
        self.nltkBigram = nltk.ngrams(nltkPairs,2)
        self.nltkTrigram = nltk.ngrams(nltkPairs,3)
        return self

    def spacy_tags(self,keep_pos=['NOUN']):
        check2 = " ".join(itertools.chain.from_iterable(self.tokens))
        # post cleaning getting noun pharses and noun qualifier phrases
        for eachWord in nlp(check2):
            if eachWord.pos_ in keep_pos:
                self.spacyTag.append(eachWord.pos_)
                self.spacyToken.append(eachWord.text)
        return self
        
    def getphrasesdf(self):
        phraseWord = ["{} {}".format(pair[0],pair[1]) for pair in zip(self.spacyToken,self.spacyToken[1:])]
        phraseTags = [self.spacyTag[i] +" "+self.spacyTag[i+1] for i in range(0,len(self.spacyTag)-1)]
        #print(phraseTags)
        #print(len(phraseTags))
        #print(len(phraseWord))
        self.df['spacyPOS']=phraseTags#[:-1]
        self.df['Word']=phraseWord
        return self
    
    def grams(gramlist,min,max):
        filler = []
        for i in range(min,max+1):
            grams = [gramlist[j:] for j in range(i)]
            #print(grams)
            #print([pair for pair in zip(*grams)])
            filler.append(["_".join(pair) for pair in zip(*grams)])
            #filler = [*itertools.chain.from_iterable(filler)]
        return filler
        
    def compiledf(self,min,max):
        xWord=topicModelFeatExtract.grams(self.spacyToken,min,max)
        xTag=topicModelFeatExtract.grams(self.spacyTag,min,max)
        
        self.df['Word']=[val for sublist in xWord for val in sublist]#[*itertools.chain.from_iterable(xWord)]
        self.df['spacyPOS']=[val for sublist in xTag for val in sublist]#[*itertools.chain.from_iterable(xTag)]
        return self
        
    
    
if __name__ == "__main__":
    scarletPimp = topicModel("/kaggle/input/scarletpimpernel-book-nlp/SP.txt")
    scarletPimp = scarletPimp.store_data()
    scarletPimpbyChapter = scarletPimp.gutenberg_getBook()
    
    #processing texts
    texts=[]
    for i,chapterName in enumerate(scarletPimpbyChapter):
        print("chapter {} is being processed".format(i+1))
        processed = topicModelPrep(chapterName)
        processed.sent_to_words()
        processed.remove_stopwords()
        print("\n")
        texts.append(processed.texts)
    
    fe = pd.DataFrame()
    allgramsToken = []
    for i in range(len(texts)):    
        #print(texts)
        fe = topicModelFeatExtract(texts[i]).spacy_tags()
        
        allgramsToken.append(topicModelFeatExtract.grams(fe.spacyToken,1,3))
        #dfbyChapter = df2.getphrasesdf().df
        dfbyChapter = fe.compiledf(1,3).df
        dfbyChapter['chapNum'] = i+1

vizdata = []
for i in range(len(allgramsToken)):
    id2word = gensim.corpora.Dictionary(allgramsToken[i])
    corpus = [id2word.doc2bow(text) for text in allgramsToken[i]]

    ldamodel = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                            id2word=id2word,
                                            num_topics=1,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

    print(ldamodel.print_topics())
    #ldamodel.show_topic(0)[1][0]
    
    for topic_num in range(1):
        wp = ldamodel.show_topic(topic_num)
        #topic_keywords = ", ".join([word for word, prop in wp])
        topic_keywords = ", ".join(["\'{}\'".format(word) for word, prop in wp])
        print(topic_keywords)
        vizdata.append(["Chapter"+str(i+1),[topic_keywords]])
    