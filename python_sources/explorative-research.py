#!/usr/bin/env python
# coding: utf-8

# ## **Trying to find a solution**
# 
# searching a system to find the closest match with train

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline


# **i exported the data in a subdirectory**
# 
# 
# * with a previous script
# * i use wiki[:5000] characters versus question
# * since the wiki can be lengthy, i don't advise to cap off the text
# 

# In[ ]:


testW=pd.read_csv("/kaggle/input/qa-download/test_document_text.csv")
trainW=pd.read_csv("/kaggle/input/qa-download/train_document_text.csv")
testQ=pd.read_csv("/kaggle/input/qa-download/test_question_text.csv")
trainQ=pd.read_csv("/kaggle/input/qa-download/train_question_text.csv")
trainW.columns=['id','start','stop','short','wiki']
testW.columns=['id','wiki']
trainQ.columns=['id','start','stop','short','question']
testQ.columns=['id','question']
trainW


# In[ ]:


trainW.info()


# ## writing a splitter, that splits the wiki in paragraphs guided by the 'tags'
# 
# i so a difference of 1 position between the database and my script... I don't know why, but i find my positions corrector. Anyway we will need to adopt those script to make collide the positions
# see the difference between train1, and the train data
# 
# 

# 

# In[ ]:


' '.join(trainW.iloc[1]['wiki'].split(' ')[212:310])  #paragraph in wiki


# In[ ]:


' '.join(trainW.iloc[1]['wiki'].split(' ')[213:215])  #short answer


# In[ ]:


#  find the start stop  token position of a sentence 
def find_start_end_token(txt,sent):
    pos=txt.find(np.str(sent) )
    start=len(txt[:pos-1].split(' '))
    end=start+len(np.str(sent).split(' '))
    return start,end

sentence='How I Met Your Mother'
find_start_end_token(trainW.iloc[1]['wiki'],sentence),trainW.iloc[1]['wiki'].split(' ')[210:310],find_start_end_token(trainW.iloc[1]['wiki'],' '.join(trainW.iloc[1]['wiki'].split()[210:217]))


# In[ ]:


trainW


# In[ ]:


from bs4 import BeautifulSoup

def wiki_tagsplit(html,wi):
    temp=[]
    #html = trainW.iloc[wi]['wiki'] 
    for hi in html:
        
        soup = BeautifulSoup(hi, 'html.parser')
        for ti in ['h1','h2','p','table','tr']:  #splitting tags extracting this features
            allep=soup.find_all(ti) #p paragraph
            for pi in allep:
                start,stop=find_start_end_token(hi,pi.get_text())
                if start>1:
                    line=[wi,ti,start,stop,pi.get_text()]
                    temp.append(line)
                    
        wi=wi+1
    return pd.DataFrame(temp,columns=['id','tag','start','stop','txt'])
wiki_tagsplit(trainW.iloc[1:2].wiki,1)


# In[ ]:


import numpy as np
from scipy.sparse import csr_matrix
get_ipython().system('pip install sparse_dot_topn')
import sparse_dot_topn.sparse_dot_topn as ct

def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))


def get_matches_df(sparse_matrix, name_vectorQ,name_vectorR, top=100):
    non_zeros = sparse_matrix.nonzero()
    
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]
    
    if top<sparsecols.size:
        nr_matches = top
    else:
        nr_matchesQ = sparserows.size
        nr_matchesR =sparsecols.size
    
    left_row = np.empty([nr_matchesQ], dtype=object)
    left_side = np.empty([nr_matchesQ], dtype=object)
    right_row =np.empty([nr_matchesQ], dtype=object)
    right_side = np.empty([nr_matchesQ], dtype=object)
    similairity = np.zeros(nr_matchesQ)
    
    for index in range(0, nr_matchesQ):
        #print(index,sparserows[index],name_vector[sparserows[index]])
        left_row[index] =sparserows[index]
        right_row[index] =sparsecols[index]
        left_side[index] = name_vectorQ[sparserows[index]]
        right_side[index] = name_vectorR[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]
    
    return pd.DataFrame({'left_nr':left_row,
                        'left_side': left_side,
                         'right_nr':right_row,
                          'right_side': right_side,
                           'similarity': similairity})


# In[ ]:


get_ipython().run_cell_magic('time', '', 'vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5)\nvect.fit( trainQ.question.values )')


# In[ ]:



for qi in range(0,1):
    questions=trainQ[qi:qi+5].question.values
    wikis=trainW.iloc[qi:qi+5].wiki.values
    wiki_split=wiki_tagsplit(wikis,qi)
    print(questions)
    Qtfidf_tot = vect.transform( questions ) #.append( (testQ.question))) )
    Wtfidf_tot = vect.transform(wiki_split.txt) #.append((testQ.question)) ) )
    
    matches = awesome_cossim_top(Qtfidf_tot, Wtfidf_tot.T, 5,0.357)
    matches_df = get_matches_df(matches, questions,wiki_split.txt.values ,top=20000)
    matches_df = matches_df[matches_df['similarity'] < 1.] 

matches_df.merge(wiki_split, left_on='right_side', right_on='txt')


# In[ ]:


Qtfidf_tot = vect.transform(trainQ[:1000].question.values ) #.append( (testQ.question))) )
wikis=trainW.iloc[:1000].wiki.values
wiki_split=wiki_tagsplit(wikis,qi)

Wtfidf_tot = vect.transform(wiki_split.txt) #.append((testQ.question)) ) )


# In[ ]:


matches = awesome_cossim_top(Qtfidf_tot, Wtfidf_tot.T, 1,0.357)
matches


# top=pd.DataFrame(np.dot(W_tot,Q_tot[4])).sort_values(0)[-40:]
# top.reset_index().sort_values('index')

# ## **testing on top 1000 matching**
# 
# as you can see without training you can match text with questions
# but this is more a solution to find a needle in the haystack or to find the text that matches the question

# In[ ]:


leng=1000
matches = awesome_cossim_top(Qtfidf_tot[:leng], Wtfidf_tot[:leng].T, 10,0.37)
matches_df = get_matches_df(matches, trainQ[:leng].question.values,wiki_split.txt.values ,top=20000)
matches_df = matches_df[matches_df['similarity'] < 1.99999] 
print(len(matches_df))
matches_df


# In[ ]:





# ## Most simple method to match the wiki content with the question

# In[ ]:


for qi in range(100,110,1):
    print(trainQ.iloc[qi])
    Qtfidf_tot = vect.transform(trainQ.iloc[qi:qi+1].question ) #.append( (testQ.question))) )
    wikis=trainW.iloc[qi:qi+1].wiki
    wiki_split=wiki_tagsplit(wikis,qi)
    
    Wtfidf_tot = vect.transform(wiki_split.txt) #.append((testQ.question)) ) )
    
    wiki_split['pro']=Wtfidf_tot.dot(Qtfidf_tot.T.todense())
    print(wiki_split[wiki_split.pro>0][['start','stop','txt','pro']].sort_values('pro')[-3:])


# In[ ]:


for qi in range(100,110,1):
    print(trainQ.iloc[qi])
    Qtfidf_tot = vect.transform(trainQ.iloc[qi:qi+1].question ) #.append( (testQ.question))) )
    wikis=trainW.iloc[qi:qi+1].wiki
    wiki_split=wiki_tagsplit(wikis,qi)
    
    Wtfidf_tot = vect.transform(wiki_split.txt) #.append((testQ.question)) ) )
    
    wiki_split['pro']=Wtfidf_tot.dot(Qtfidf_tot.T.todense())
    print(wiki_split[wiki_split.pro>0][['start','stop','txt','pro']].sort_values('pro')[-1:])
    headerwords=wiki_split[wiki_split.pro>0][['start','stop','txt','pro']].sort_values('pro')[-1:].txt
    headerwords=[str.lower(wi) for wi in headerwords.values[0].split(' ') ]
    remainQ=[wi for wi in trainQ.iloc[qi].question.split(' ') if wi not in headerwords]
    remainQ=(' ').join(remainQ)
    print(remainQ)
    Qtfidf_tot = vect.transform([remainQ,''])
    wiki_split['pro']=Wtfidf_tot.dot(Qtfidf_tot.T.todense())
    print(wiki_split[wiki_split.pro>0][['start','stop','txt','pro']].sort_values('pro')[-1:])
    

