# ----------------------------------------------------------------------------------------
import pickle   
from os import listdir
from os.path import isfile, join
import spacy #load spacy
nlp = spacy.load('en_core_web_sm')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
from collections import defaultdict
import matplotlib.cm as cm
from copy import deepcopy
import math
# import sframe                           
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
# get_ipython().magic(u'matplotlib inline')



def spacy_cleansing(doc):
    # for token in doc:
    #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
    #             token.shape_, token.is_alpha, token.is_stop)
    doc = nlp(doc)
    lemmatized = []
    for token in doc:
        if not token.is_stop and token.pos_ not in ['SYM']:
            lemmatized.append(token.lemma_)            
    return lemmatized


def art_to_para(article,threshold=50):
    #     For fragmentation of article text into paragraphs of word-count >= threshold
    #         Parameters :
    #         -------------
    #         article : str
    #         threshold : int, optional, default 50
    para_list = article.split('\n\n')
    if len(para_list) == 1 and len(para_list[0]) > 3.2*threshold :
        temp = para_list[0].split('. ')
        para_list = [". ".join(temp[:math.floor(len(temp)/2)]), ". ".join(temp[math.floor(len(temp)/2):])]
    l = len(para_list)
    def para_thresholding(i=0, clustered_para=[]):
        temp = para_list[i]
        while len(temp.split(" ")) < threshold and i < l - 1:
            temp = temp + " " + para_list[i+1]
            i+=1
        clustered_para.append(temp) 
        if i == l-1:
            cluster_len = len(clustered_para[-1])
            if cluster_len < threshold and cluster_len > 1:
                clustered_para[-2] = clustered_para[-2] + " " + clustered_para[-1]
                del clustered_para[-1]
            return clustered_para
        else:
            return para_thresholding(i+1,clustered_para)
    return para_thresholding()

def load_and_break(keywords, filesave=False):
    mypath = 'news-analyzer/data/'+keywords+'/news/'
    news_text_file = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    usr_input = input("For whole article, Press a \nFor paragraph segmentation, press p\n")
    bow = [] #........cleaned documents in list of lists
    text = []
    if usr_input == 'a':
        for f in news_text_file:    
            text_article = pickle.load(open(f, 'rb'))
            text.append(text_article['content'])
            temp = spacy_cleansing(text_article['content'])
            bow.append(temp)
    #         print ('article no : ',news_text_file.index(f), sorted(temp),'\n')
    else:
        para_count_in_art = []
        for f in news_text_file:    
            text_article = pickle.load(open(f, 'rb'))
            temp = art_to_para(text_article['content'])
            text += temp
            para_count_in_art.append(len(temp))
            for t in temp:
                temp1 = spacy_cleansing(t)
                bow.append(temp1)
    store_file = {}
    store_file['para_count_in_art'] = para_count_in_art if usr_input != 'a' else 0
    store_file['text'] = text if usr_input != 'a' else 0
    store_file['clean_text_doclist'] = bow
    if filesave:    
        if usr_input == 'a':
            file = 'spacy_cleansing.pkl'
        else:
            file = 'spacy_cleansing_of_paras.pkl'
        pickle.dump(store_file,open(file,'wb'))
    return store_file


# ---------------------------------- tfidf ---------------------------------------------------------------------


def word_count(doc):
    doc_set_all = []
    for docc in doc:
        doc_set = sorted(set().union(docc))
        doc_dict = dict.fromkeys(doc_set, 0)
        for word in docc:
            doc_dict[word] += 1 
        doc_set_all.append(doc_dict.copy())
    return doc_set_all

# print ("wordSet  :   "," ".join(sorted(wordSet)))

def computeTF(doc_set_all,doc):
    tfidf_all = []
    for i in range(len(doc)):
        tfidf = {}
        doc_count = len(doc[i])
        for word, count in doc_set_all[i].items():
            tfidf[word] = round((count+1)/doc_count,4)
    #     print(str(tfDict),'\n'
        tfidf_all.append(tfidf.copy())
    return tfidf_all


def computeIDF(doc_set_all,wordSet):
#     import math
    idfDict = dict.fromkeys(wordSet, 0)
    N = len(doc_set_all)
    for word in wordSet:
        for i in range(N):
        	if word in doc_set_all[i].keys():
        		idfDict[word] += 1
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / val)
                        
    return idfDict        




def computeTFIDF(tfs, idfs, threshold=0.01):
    tfidfs = []
    new_wordSet = []
    for t in tfs:
        tfidf = {}
        for word, val in t.items():
            ti = round(val*idfs[word],4)
            if ti > threshold:
                tfidf[word] = ti #------rounded for error handeling of data(float64)
                new_wordSet.append(word)
        tfidfs.append(tfidf.copy())
#     print(str(tfidf),'\n')
#     print(str(new_wordset),'\n')
    return tfidfs, new_wordSet


def computeTFIDF_matrix(wordSet, tfidf, l_doc):
    tfidf_wordSet = [dict.fromkeys(wordSet, 0)]*l_doc
    i=0 
    tfidf_matrix = [] 
    for wrdF in tfidf_wordSet:
        tfidf_matrix.append(dict(wrdF, **tfidf[i])) 
        i += 1
    return tfidf_matrix
    
    
def main(stry_kywrds, threshold=0.01, filesave=False):
    processed_doc = load_and_break(stry_kywrds)
    doc = processed_doc['clean_text_doclist']
    wordSet = sorted(set().union(*doc))
    doc_set_all = word_count(doc)
    tfs = computeTF(doc_set_all, doc)
    idfs = computeIDF(doc_set_all, wordSet)
    tfidf, new_wordSet = computeTFIDF(tfs, idfs, threshold)
    wordSet = sorted(set().union(new_wordSet))
    df = pd.DataFrame(computeTFIDF_matrix(wordSet, tfidf, len(doc)))
    if filesave :
        df.to_csv('tfidf_of_paras.csv')
    return df, tfidf, processed_doc['text']


dataframe, tfidf, text = main('rafale deal', threshold=0.001, filesave=False)
print (dataframe.shape)

#------------------------------DIVISIVE CLUSTERING BY BIPARTITION------------------------------
# SOURCE : https://github.com/SSQ/Coursera-UW-Machine-Learning-Clustering-Retrieval/blob/master/Week%206%20PA%201/6_hierarchical_clustering.py
# wiki = sframe.SFrame('people_wiki.gl/')
# dataframe = pd.read_csv('tfidf_of_paras.csv')
X = dataframe.iloc[:,1:].values
X = np.nan_to_num(X)
m, feat = X.shape
tf_idf = X
tf_idf = normalize(tf_idf)
# str(dataframe.iloc[:,0].tolist())


def bipartition(cluster, maxiter=400, num_runs=4, seed=None):
    data_matrix = cluster['matrix']
    dataframe   = cluster['dataframe']
    kmeans_model = KMeans(n_clusters=2, max_iter=maxiter, n_init=num_runs, random_state=seed, n_jobs=1)
    kmeans_model.fit(data_matrix)
    centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_
    data_matrix_left_child, data_matrix_right_child = data_matrix[cluster_assignment==0], data_matrix[cluster_assignment==1]
#     cluster_assignment_sa = sframe.SArray(cluster_assignment) ##     AG
    cluster_assignment_sa = cluster_assignment                 ##     AG
    dataframe_left_child, dataframe_right_child = dataframe[cluster_assignment_sa==0], dataframe[cluster_assignment_sa==1]
    cluster_left_child  = {'matrix': data_matrix_left_child,
                           'dataframe': dataframe_left_child,
                           'centroid': centroids[0]}
    cluster_right_child = {'matrix': data_matrix_right_child,
                           'dataframe': dataframe_right_child,
                           'centroid': centroids[1]}
    return (cluster_left_child, cluster_right_child)


data = {'matrix': tf_idf, 'dataframe': dataframe} 
cluster_list = [bipartition(data, maxiter=100, num_runs=6, seed=1)]
i = 0
last_layer = []
while i != len(cluster_list):
    if len(cluster_list[i][0]['matrix']) > 80:
        cluster_list.append(bipartition(cluster_list[i][0], maxiter=100, num_runs=6, seed=1))
    else:
        last_layer.append(cluster_list[i][0])
    if len(cluster_list[i][1]['matrix']) > 80:
        cluster_list.append(bipartition(cluster_list[i][1], maxiter=100, num_runs=6, seed=1))
    else:
        last_layer.append(cluster_list[i][1])
    i += 1
print('PARA_COUNT IN LOWEST {}-CLUSTERS: \n{}'.format(len(last_layer),str([len(x['matrix']) for x in last_layer])))

ii = 1
for j in last_layer:
    for i in j['dataframe'].index.tolist():
        print('\033[1m'+'CLUSTER NO : {} with {} articles'.format(ii, len(j['matrix']))+'\033[0m')
        print (i, '\n', text[int(i)], '\n\n',tfidf[int(i)], '\n-----------------------------------------------------------------------------\n')
    ii += 1
    print('\n\n\n\n\n\n\n\n\n\n<<             ---- END OF CLUSTER --------             >>\n\n\n\n\n\n\n\n\n\n\n')