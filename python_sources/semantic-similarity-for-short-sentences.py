#!/usr/bin/env python
# coding: utf-8

# http://sujitpal.blogspot.be/2014/12/semantic-similarity-for-short-sentences.html

# In[ ]:



import time
start = time.clock()

#open data
import pandas as pd
import numpy as np
import nltk
import codecs
from nltk.stem.snowball import SnowballStemmer
print(SnowballStemmer("english").stem("generously"))
from nltk.tokenize import word_tokenize

datas = pd.read_csv('../input/train.csv') #
datas = datas.fillna('leeg')

def cleantxt(x):    # aangeven sentence
    # Removing non ASCII chars
    x = x.replace(r'[^\x00-\x7f]',r' ') 
#    x = x.decode('utf-8').strip()
    x = x.lower()
    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        #x = x.replace(char, ' ' + char + ' ')
        x = x.replace(char, ' ')
    return x

datas['question1']=datas['question1'].map(cleantxt)
datas['question2']=datas['question2'].map(cleantxt)
print(datas.head())

end = time.clock()
print('open:',end-start)


# In[ ]:


import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import reuters
import math
import numpy as np
import sys

# Parameters to the algorithm. Currently set to values that was reported
# in the paper to produce "best" results.
ALPHA = 0.2
BETA = 0.45
ETA = 0.4
PHI = 0.2
DELTA = 0.85

brown_freqs = dict()
N = 0

######################### word similarity ##########################

def get_best_synset_pair(word_1, word_2):
    """ 
    Choose the pair with highest path similarity among all pairs. 
    Mimics pattern-seeking behavior of humans.
    """
    max_sim = -1.0
    synsets_1 = wn.synsets(word_1)
    synsets_2 = wn.synsets(word_2)
    if len(synsets_1) == 0 or len(synsets_2) == 0:
        return None, None
    else:
        max_sim = -1.0
        best_pair = None, None
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
               sim = wn.path_similarity(synset_1, synset_2)
               #print(synset_1,synset_2,sim,max_sim)
               if sim==None:
                sim=0
                
               if sim > max_sim:
                   max_sim = sim
                   best_pair = synset_1, synset_2
        return best_pair

def length_dist(synset_1, synset_2):
    """
    Return a measure of the length of the shortest path in the semantic 
    ontology (Wordnet in our case as well as the paper's) between two 
    synsets.
    """
    l_dist = sys.maxsize
    if synset_1 is None or synset_2 is None: 
        return 0.0
    if synset_1 == synset_2:
        # if synset_1 and synset_2 are the same synset return 0
        l_dist = 0.0
    else:
        wset_1 = set([str(x.name()) for x in synset_1.lemmas()])        
        wset_2 = set([str(x.name()) for x in synset_2.lemmas()])
        if len(wset_1.intersection(wset_2)) > 0:
            # if synset_1 != synset_2 but there is word overlap, return 1.0
            l_dist = 1.0
        else:
            # just compute the shortest path between the two
            l_dist = synset_1.shortest_path_distance(synset_2)
            if l_dist is None:
                l_dist = 0.0
    # normalize path length to the range [0,1]
    return math.exp(-ALPHA * l_dist)

def hierarchy_dist(synset_1, synset_2):
    """
    Return a measure of depth in the ontology to model the fact that 
    nodes closer to the root are broader and have less semantic similarity
    than nodes further away from the root.
    """
    h_dist = sys.maxsize
    if synset_1 is None or synset_2 is None: 
        return h_dist
    if synset_1 == synset_2:
        # return the depth of one of synset_1 or synset_2
        h_dist = max([x[1] for x in synset_1.hypernym_distances()])
    else:
        # find the max depth of least common subsumer
        hypernyms_1 = {x[0]:x[1] for x in synset_1.hypernym_distances()}
        hypernyms_2 = {x[0]:x[1] for x in synset_2.hypernym_distances()}
        lcs_candidates = set(hypernyms_1.keys()).intersection(
            set(hypernyms_2.keys()))
        if len(lcs_candidates) > 0:
            lcs_dists = []
            for lcs_candidate in lcs_candidates:
                lcs_d1 = 0
                if lcs_candidate in hypernyms_1:
                    lcs_d1 = hypernyms_1[lcs_candidate]
                lcs_d2 = 0
                if lcs_candidate in hypernyms_2:
                    lcs_d2 = hypernyms_2[lcs_candidate]
                lcs_dists.append(max([lcs_d1, lcs_d2]))
            h_dist = max(lcs_dists)
        else:
            h_dist = 0
    return ((math.exp(BETA * h_dist) - math.exp(-BETA * h_dist)) / 
        (math.exp(BETA * h_dist) + math.exp(-BETA * h_dist)))
    
def word_similarity(word_1, word_2):
    synset_pair = get_best_synset_pair(word_1, word_2)
    return (length_dist(synset_pair[0], synset_pair[1]) * 
        hierarchy_dist(synset_pair[0], synset_pair[1]))

######################### sentence similarity ##########################

def most_similar_word(word, word_set):
    """
    Find the word in the joint word set that is most similar to the word
    passed in. We use the algorithm above to compute word similarity between
    the word and each word in the joint word set, and return the most similar
    word and the actual similarity value.
    """
    max_sim = -1.0
    sim_word = ""
    for ref_word in word_set:
      sim = word_similarity(word, ref_word)
      if sim > max_sim:
          max_sim = sim
          sim_word = ref_word
    return sim_word, max_sim
    
def info_content(lookup_word):
    """
    Uses the Brown corpus available in NLTK to calculate a Laplace
    smoothed frequency distribution of words, then uses this information
    to compute the information content of the lookup_word.
    """
    global N
    if N == 0:
        # poor man's lazy evaluation
        for sent in reuters.sents():
            for word in sent:
                word = word.lower()
                if word not in brown_freqs:
                    brown_freqs[word] = 0
                brown_freqs[word] = brown_freqs[word] + 1
                N = N + 1
    lookup_word = lookup_word.lower()
    n = 0 if lookup_word not in brown_freqs else brown_freqs[lookup_word]
    return 1.0 - (math.log(n + 1) / math.log(N + 1))
    
def semantic_vector(words, joint_words, info_content_norm):
    """
    Computes the semantic vector of a sentence. The sentence is passed in as
    a collection of words. The size of the semantic vector is the same as the
    size of the joint word set. The elements are 1 if a word in the sentence
    already exists in the joint word set, or the similarity of the word to the
    most similar word in the joint word set if it doesn't. Both values are 
    further normalized by the word's (and similar word's) information content
    if info_content_norm is True.
    """
    sent_set = set(words)
    semvec = np.zeros(len(joint_words))
    i = 0
    for joint_word in joint_words:
        if joint_word in sent_set:
            # if word in union exists in the sentence, s(i) = 1 (unnormalized)
            semvec[i] = 1.0
            if info_content_norm:
                semvec[i] = semvec[i] * math.pow(info_content(joint_word), 2)
        else:
            # find the most similar word in the joint set and set the sim value
            sim_word, max_sim = most_similar_word(joint_word, sent_set)
            semvec[i] = PHI if max_sim > PHI else 0.0
            if info_content_norm:
                semvec[i] = semvec[i] * info_content(joint_word) * info_content(sim_word)
        i = i + 1
    return semvec                
            
def semantic_similarity(sentence_1, sentence_2, info_content_norm):
    """
    Computes the semantic similarity between two sentences as the cosine
    similarity between the semantic vectors computed for each sentence.
    """
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = set(words_1).union(set(words_2))
    vec_1 = semantic_vector(words_1, joint_words, info_content_norm)
    vec_2 = semantic_vector(words_2, joint_words, info_content_norm)
    return np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))

######################### word order similarity ##########################

def word_order_vector(words, joint_words, windex):
    """
    Computes the word order vector for a sentence. The sentence is passed
    in as a collection of words. The size of the word order vector is the
    same as the size of the joint word set. The elements of the word order
    vector are the position mapping (from the windex dictionary) of the 
    word in the joint set if the word exists in the sentence. If the word
    does not exist in the sentence, then the value of the element is the 
    position of the most similar word in the sentence as long as the similarity
    is above the threshold ETA.
    """
    wovec = np.zeros(len(joint_words))
    i = 0
    wordset = set(words)
    for joint_word in joint_words:
        if joint_word in wordset:
            # word in joint_words found in sentence, just populate the index
            wovec[i] = windex[joint_word]
        else:
            # word not in joint_words, find most similar word and populate
            # word_vector with the thresholded similarity
            sim_word, max_sim = most_similar_word(joint_word, wordset)
            if max_sim > ETA:
                wovec[i] = windex[sim_word]
            else:
                wovec[i] = 0
        i = i + 1
    return wovec

def word_order_similarity(sentence_1, sentence_2):
    """
    Computes the word-order similarity between two sentences as the normalized
    difference of word order between the two sentences.
    """
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = list(set(words_1).union(set(words_2)))
    windex = {x[1]: x[0] for x in enumerate(joint_words)}
    r1 = word_order_vector(words_1, joint_words, windex)
    r2 = word_order_vector(words_2, joint_words, windex)
    return 1.0 - (np.linalg.norm(r1 - r2) / np.linalg.norm(r1 + r2))

######################### overall similarity ##########################

def similarity(sentence_1, sentence_2, info_content_norm):
    """
    Calculate the semantic similarity between two sentences. The last 
    parameter is True or False depending on whether information content
    normalization is desired or not.
    """
    return DELTA * semantic_similarity(sentence_1, sentence_2, info_content_norm) +         (1.0 - DELTA) * word_order_similarity(sentence_1, sentence_2)
        
######################### main / test ##########################

# the results of the algorithm are largely dependent on the results of 
# the word similarities, so we should test this first...

for xi in range(20,25):
    print (datas.iloc[xi]['question1'], datas.iloc[xi]['question2'],datas.iloc[xi]['is_duplicate'])
    print( 
           similarity(datas.iloc[xi]['question1'], datas.iloc[xi]['question2'], False),
           similarity(datas.iloc[xi]['question1'],datas.iloc[xi]['question2'], True)
          ) 
    
end = time.clock()
print('five examples:',end-start)


# In[ ]:


from nltk.corpus import stopwords # Import the stop word list

from nltk import word_tokenize, ngrams
#print stopwords.words("english") simil_d=[]

simil_d=[]
simil_c=[]
simil_i=[]
simil_b=[]
for xyz in range(200,300):
    q1=datas.iloc[xyz].question1
    q2=datas.iloc[xyz].question2
    uni_sent1 = nltk.wordpunct_tokenize(q1) #tokenize sentence
    uni_sent2 = nltk.wordpunct_tokenize(q2) 
    #sims=pd.DataFrame(data=None, index=uni_sent1, columns=uni_sent2)  #abs(np.random.randn(len(sent1), len(sent2))/1000000)
    equq1 = [w for w in uni_sent1 if w in uni_sent2]
    difq1 = [w for w in uni_sent1 if w not in uni_sent2]
    difq2 = [w for w in uni_sent2 if w not in uni_sent1]
    diftot = difq1+difq2
    difton = [w for w in diftot if not w in stopwords.words("english")]
    Q2no = [w for w in uni_sent2 if not w in stopwords.words("english")]

    canar=len(equq1)/(len(equq1)+len(diftot))
    simil_c.append(canar)
    
    
    # get bigram features #
    bigrams_q1 = [i for i in ngrams(uni_sent1, 2)]
    bigrams_q2 = [i for i in ngrams(uni_sent2, 2)]
    common_bigrams_len = len(set(bigrams_q1).intersection(set(bigrams_q2)))
    simil_b.append( (common_bigrams_len)*2.0 / (len(set(bigrams_q1))+len(set(bigrams_q2)) )  )
    # get trigram features #
    trigrams_q1 = [i for i in ngrams(uni_sent1, 3)]
    trigrams_q2 = [i for i in ngrams(uni_sent2, 3)]
    common_trigrams_len = len(set(trigrams_q1).intersection(set(trigrams_q2)))
    simil_i.append( common_trigrams_len*2.0 /( len(set(trigrams_q1))+len(set(trigrams_q2)) ) )

    if len(difton)==0 and datas.iloc[xyz].is_duplicate==0:
        simil_d.append(2)
    elif difton==1 and datas.iloc[xyz].is_duplicate==0:
        simil_d.append(3)
    #elif Q2no<3 :
    #    simil_d.append(4)
    else:
        simil_d.append(datas.iloc[xyz].is_duplicate)
 
end = time.clock()
print('first canary:',end-start)
  
 


# In[ ]:


print(len(simil_d),len(simil_i))


# In[ ]:


import seaborn as sns
sns.set(style="white", color_codes=True)
simil_f=[]
simil_t=[]
#simil_d=simil_d
           
for xi in range(200,300):
    simil_f.append(similarity(datas.iloc[xi]['question1'], datas.iloc[xi]['question2'], False))
    simil_t.append(similarity(datas.iloc[xi]['question1'],datas.iloc[xi]['question2'], True))
    #simil_d.append(datas.iloc[xi]['is_duplicate'])

end = time.clock()
print('300 similarity:',end-start)

similXY=pd.DataFrame([])
similXY['sim_f']=simil_f
similXY['sim_t']=simil_t
similXY['unigr']=simil_c
similXY['bigram']=simil_b
similXY['trigram']=simil_i
similXY['is_duplicate']=simil_d
print(similXY.head())

sns.pairplot(similXY, hue="is_duplicate", size=3)
#sns.pairplot(similXY.drop("Id", axis=1), hue="is_duplicate", size=2, diag_kind="kde")


# In[ ]:



import time
start = time.clock()

#open data
import pandas as pd
import numpy as np
import nltk
import codecs
from nltk.stem.snowball import SnowballStemmer
print(SnowballStemmer("english").stem("generously"))
from nltk.tokenize import word_tokenize

datas = pd.read_csv('../input/train.csv') #
datas = datas.fillna('leeg')

def cleantxt(x):    # aangeven sentence
    # Removing non ASCII chars
    x = x.replace(r'[^\x00-\x7f]',r' ') 
#    x = x.decode('utf-8').strip()
    x = x.lower()
    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        #x = x.replace(char, ' ' + char + ' ')
        x = x.replace(char, ' ')
    return x

datas['question1']=datas['question1'].map(cleantxt)
datas['question2']=datas['question2'].map(cleantxt)
print(datas.head())

end = time.clock()
print('open:',end-start)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




