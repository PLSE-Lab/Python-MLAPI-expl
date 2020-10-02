#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train=pd.read_csv('https://www.kaggleusercontent.com/kf/31943753/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..2JkMtB524ykQVE-oXkjoUQ.U1qNTXY2QjcHQwoQGZeAlTW9aIlhBG38NSTcZVLbiMeLyJybAwdKuI611hl5aqDX16JXzzrKxfKVYbPqX5inu2Vgk2b_JS7FpnJzQ6isNOlMyqtdRG0EXW4ksMvGsPNq0BbfqR_rFvwMuHBkVfWZpyazdJW-99792-FPqk9Ve6pqQpyI2w-FvPkXkHsyFksE_HiFN71UcPmICS5oDIE8PeWXosqTZU7q7iLs1uht-NZVDAYg27roJ0FfHGtKL5o0ZFJB8sHCkGqXKua92q4B-B9V6M3C10gkaAzQev0zaD9LAt2xsScwkNSZELMDVXnp5cqDlE85dbj71hzrDITatv9KSugwoh3sJxXBVBOEJ5LRjwqZeGJimKSuQMEZdH5QCWFZqn40J5y5TB7s7f_RN77vCDUAprpRc2nFUft5vFs8gaiti-9dLqhlpWN4zHUJq538L00AWYNM3trolNdG5DbBGFJzg7WCUN-HVchwI-BaBqQZFDwWzZkf7yK6NzPJ5x1xgQi6M2dmxKg70C2bZ5EpAr5PjS3B7rjSMFRQAQOrmMauRPhuCC_uhNnwrni2cqUIOQgl505pLR383KaN3YYOznWtltjlpi4YaizHw7W22T4a1k4LWKzaWbfGPC4fDEv3EIWv8rJ7KWNxm_nyC2fOp0YEy8wbL3WrJ9fFpNY.wU_iwkwjQg2x3yliM5B6bA/entitypairs.csv')

train=pd.read_csv('https://www.kaggleusercontent.com/kf/31943753/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..kUHwkKzrJ7LowINLs4qVlg.kWE5Vko--KPokahzMlQOAkGViea9NEFlFK7Xr91lQj526mRWLSjie8L-bHnHx7Radufl08YCcALYeiViTs849kSMW3ir49B51W5YK0dfk7HCeW7sr8FQ328iucKm21GEbH-4ZORMmPjIf3cgCpIpESWzgihlM3JmPs68YvVUMyNfH5wst3p71uxUJjjY8RdRSUV0GKhn_7ZMEC2FvfOptx_HaSnuwcuxcbrwPi90mYT5Oel4AI982bYCFaF2s7iIPZJN_vxB6poAzX-qfdGdYOoYGzbhf77i3YKxWJiF4cSgIe4sP_WIZwluLKtgLe_8beFbQap7iXkFx4KwIAyn-FLazLhvYkpvEdOC7HuE54pUt4BrBebrQCl-RVFl8ofhzerDPmnXxpHnCF9SRTNjJ3EFpD6TWqVMP15X_HZLzDjRXFrJsE0tAEoUQK3REjoaMwS2asXBM1p3QGJ6Lk0lxC2splU_BcS9yqEAnVVHAK7IbJ--ayrpbshlN5NV1aoBKZDl4YBp2RjlPtLuEAVh_VDZ_mN3QVF_NPio4J_xrqQCVeFVFndzw0gR_7TxwxsTlu-zONSjzRHkJQJnp_x73d5_bfotuj4lFkTcNjjU_EIP4fLAT64QeX6UVy6xIU7BYHvCxk-UgfhCQ05Fowdjlaj8N_r4JI0cI9XPtTaATYg.WYndUynewGEs684wjezbJQ/Relevant_COVID-19_Abstracts.csv')


# In[ ]:


train


# In[ ]:


Questions=['Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.',
 'Prevalence of asymptomatic shedding and transmission (e.g., particularly children).',
 'Seasonality of transmission. Winter Summer Season effect Influenza',
 'Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).',
 'Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).',
 'Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).',
 'Natural history of the virus and shedding of it from an infected person',
 'Implementation of diagnostics and products to improve clinical processes',
 'Disease models, including animal models for infection, disease and transmission',
 'Tools and studies to monitor phenotypic change and potential adaptation of the virus',
 'Immune response and immunity',
 'Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings',
 'Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings',
 'Role of the environment in transmission']


# In[ ]:


test=pd.DataFrame(Questions,columns=['Title'])
test


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
tfidf = TfidfVectorizer(max_features=100000, stop_words='english')
X = tfidf.fit_transform( ( train.Title.fillna('')+' '+train.Abstract.fillna('') ).append(test.Title)  ) #corona_df.total.fillna(' '))
X


# In[ ]:



import argparse
import codecs
import logging
import time

import numpy as np
import tqdm

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.datasets.movielens import get_movielens
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         TFIDFRecommender, bm25_weight)

log = logging.getLogger("implicit")

# read in the input data file
start = time.time()
output_filename='output.txt'
model_name='bpr'
min_rating=-10.0,
#titles=corono_df..columns[1:]
# remove things < min_rating, and convert to implicit dataset
# by considering ratings as a binary preference only
X.data[X.data < min_rating] = 0
X.eliminate_zeros()
X.data = np.ones(len(X.data))

log.info("read data file in %s", time.time() - start)

# generate a recommender model based off the input params
if model_name == "als":
    model = AlternatingLeastSquares()

    # lets weight these models by bm25weight.
    print("weighting matrix by bm25_weight")
    ratings = (bm25_weight(ratings, B=0.9) * 5).tocsr()

elif model_name == "bpr":
    model = BayesianPersonalizedRanking()

elif model_name == "lmf":
    model = LogisticMatrixFactorization()

elif model_name == "tfidf":
    model = TFIDFRecommender()

elif model_name == "cosine":
    model = CosineRecommender()

elif model_name == "bm25":
    model = BM25Recommender(B=0.2)

else:
    raise NotImplementedError("TODO: model %s" % model_name)

# train the model
print("training model %s", model_name)
start = time.time()
model.fit(X)
print("trained model '%s' in %s", model_name, time.time() - start)
log.debug("calculating top movies")

user_count = np.ediff1d(X.indptr)
titles=( train.Title.fillna('')+' '+train.Abstract.fillna('') ).append(test.Title)
#to_generate = sorted(np.arange(len(titles)-13207), key=lambda x: -user_count[x])


# In[ ]:



    #titles=(train.source.fillna('')+' '+train.target.fillna('')+' '+train.edge.fillna('')).append(test.source)
    user_count = np.ediff1d(X.indptr)
    
    #to_generate = sorted(np.arange(len(titles)-5247), key=lambda x: -user_count[x])
    to_generate = [x for x in range(X.shape[0]) if x>25795 ]

    print("possible text related to questions")
    with tqdm.tqdm(total=len(to_generate)) as progress:
        with codecs.open(output_filename, "w", "utf8") as o:
            for movieid in to_generate:
                print("______________________________________________________________________")                    
                print(movieid,model.similar_items(movieid))
            
                # if this movie has no ratings, skip over (for instance 'Graffiti Bridge' has
                # no ratings > 4 meaning we've filtered out all data for it.
                if X.indptr[movieid] != X.indptr[movieid + 1]:
                    title = titles.iloc[movieid]#.Abstract
                    print('Q:',title)
                    print("______________________________________________________________________")                        
                    for other, score in model.similar_items(movieid,10):
                        #o.write("%s\t%s\t%s\n" % (title, titles[other], score))
                        try:
                            print(other,titles.iloc[other],'r2',score)
                            
                        except:
                            print(titles.iloc[other],other,score)
                progress.update(1)
                print('______________________________________________________________________')


# In[ ]:


user_count = np.ediff1d(X.indptr)
to_generate = [x for x in range(X.shape[0]) if x>25795]

print("possible text related to questions")
with tqdm.tqdm(total=len(to_generate)) as progress:
    with codecs.open(output_filename, "w", "utf8") as o:
        for movieid in to_generate:
            print("______________________________________________________________________")                    
            print(movieid,model.similar_items(movieid))
        
            # if this movie has no ratings, skip over (for instance 'Graffiti Bridge' has
            # no ratings > 4 meaning we've filtered out all data for it.
            if X.indptr[movieid] != X.indptr[movieid + 1]:
                title = titles.iloc[movieid]#.Abstract
                print('Q:',title)
                print("______________________________________________________________________")                        
                for other, score in model.similar_items(movieid,10):
                    #o.write("%s\t%s\t%s\n" % (title, titles[other], score))
                    try:
                        print(other,titles[other][:50],score)
                        print(generate_summary(titles.iloc[other],2))                            
                    except:
                        print(titles.iloc[other],other,score)
                        try:
                            len(titles.iloc[other].total)>0
                            print(generate_summary(titles.iloc[other],2)) 
                        except:
                            print('')
            progress.update(1)
            print('______________________________________________________________________')

