import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk import tokenize, stem
from nltk.corpus import stopwords

def tokenizer(sentence):
    #convert all characters to lowercase
    lowered = sentence.lower()
    tokenized = tokenize.word_tokenize(lowered)
    lemmatizer = stem.WordNetLemmatizer()
    lemmas = list(map(lemmatizer.lemmatize, tokenized))
    return lemmas

def pos_tagger(lemmas):
    tagged = nltk.pos_tag(lemmas)
    return tagged

def remove_stop_words(lemmas, st_list=None):
    stop_words = stopwords.words('english')
    if st_list:
        stop_words.extend(st_list)
    #remove non-alphabetical characters like '(', '.' or '!'
    alphas = [w for w in lemmas if w.isalpha()]
    # remove stopwords and words with too short characters
    sentence_without_stop_words = [r for r in alphas if r not in stop_words and len(r) > 1]
    return sentence_without_stop_words

def bow_features(docs,_max_features=10):
    if type(docs) != np.ndarray:
        docs = np.array(docs)
    docs_preprocessed = []
    for doc in docs:
        if type(doc) == list:
            doc = " ".join(doc)
        docs_preprocessed.append(doc)
    vec_bow = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b', max_df=1.0, min_df=1, max_features=_max_features, lowercase=True, 
                              stop_words='english', analyzer='word', ngram_range=(1,2))
    bow_matrix = vec_bow.fit_transform(docs_preprocessed)
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=["BOW_" + n for n in vec_bow.get_feature_names()])
    return bow_df

def tfidf_features(docs_tfidf, docs_train=None, docs_test=None, _max_features=10):
    if type(docs_tfidf) != np.ndarray:
        docs_tfidf = np.array(docs_tfidf)
    docs_tfidf_preprocessed = []
    for doc in docs_tfidf:
        if type(doc) == list:
            doc = " ".join(doc)
        docs_tfidf_preprocessed.append(doc)
    vec_tfidf = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b', max_df=1.0, min_df=1, max_features=_max_features, norm='l2', 
                                stop_words='english', lowercase=True, use_idf=True, analyzer='word', ngram_range=(1,2), smooth_idf=True)
    vec_tfidf.fit(docs_tfidf_preprocessed)
    if docs_train == None or docs_test == None:
        tfidf_matrix = vec_tfidf.fit_transform(docs_tfidf_preprocessed)
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=["TFIDF_" + n for n in vec_tfidf.get_feature_names()])
        return tfidf_df
    else:
        tfidf_matrix_train = vec_tfidf.transform(docs_train)
        tfidf_matrix_test = vec_tfidf.transform(docs_test)
        tfidf_train = pd.DataFrame(tfidf_matrix_train.toarray(), columns=["TFIDF_" + n for n in vec_tfidf.get_feature_names()])
        tfidf_test = pd.DataFrame(tfidf_matrix_test.toarray(), columns=["TFIDF_" + n for n in vec_tfidf.get_feature_names()])
        return tfidf_train, tfidf_test

def ngram_func(ngram, text):
    dic = nltk.FreqDist(nltk.ngrams(text, ngram)).most_common(30)
    ngram_df = pd.DataFrame(dic, columns=['ngram','count'])
    ngram_df.index = [' '.join(i) for i in ngram_df.ngram]
    ngram_df.drop('ngram',axis=1, inplace=True)
    return ngram_df
    
if __name__ == '__main__':
    base_dir = '../input/testdata/'
    demotext = 'df_text_eng.csv'
    data = pd.read_csv(base_dir + demotext)
    data.drop("Unnamed: 0", axis=1, inplace=True)
    data.dropna(how='any', inplace=True)
    
    #TEST : tokenizer
    sentence = data['blurb'][0]
    ret = tokenizer(sentence)
    print(ret)
    
    #TEST : pos_tagger
    ret2 = pos_tagger(ret)
    print(ret2)
    
    #TEST : remove_stop_words
    ret3 = remove_stop_words(ret,['go','using'])
    print(ret3)
    
    # preprocessing pipeline
    sentences = data['blurb'][0:10000]
    sentences_preprocessed = []
    for sentence in sentences:
        lemmas = tokenizer(sentence)
        sentence_without_stop_words = remove_stop_words(lemmas)
        sentences_preprocessed.append(sentence_without_stop_words)
        
    #TEST : tfidf_features
    ret4 = tfidf_features(docs_tfidf=sentences_preprocessed,_max_features=20)
    print(ret4)
    
    #TEST : bow_features
    ret5 = bow_features(sentences_preprocessed,20)
    print(ret5)
    
    #TEST : n-gram
    text = data['blurb'][0:5000].sum().split()
    #Unigram
    ret6 = ngram_func(1,text)
    print(ret6)
    #Bigram
    ret7 = ngram_func(2,text)
    print(ret7)
    