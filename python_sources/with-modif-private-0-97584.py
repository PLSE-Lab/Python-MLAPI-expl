#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np, pandas as pd, os
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.covariance import LedoitWolf, OAS
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, Birch
from sklearn.pipeline import Pipeline
from scipy.stats import multivariate_normal
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import rankdata
from scipy.optimize import minimize


# In[ ]:


get_ipython().system('pip install MulticoreTSNE')


# In[ ]:


from MulticoreTSNE import MulticoreTSNE as TSNE


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


modif_list =  [7, 24, 27,28, 36, 42,49,52,54, 71,72, 79,
    81, 85,87,89,92, 93, 94, 95, 96, 102, 103,
    104, 107,109, 114, 125, 129, 130, 131, 143,
    149,151, 152, 162,166, 177, 182,187, 193,194, 195,
    197, 198, 206,207, 208,211, 213, 220,
    221, 222, 230,233, 234, 235, 242,244,251,
    252,258, 262, 263, 264, 266, 268, 270, 271, 273,
    287, 289,293,308, 310, 312,313, 315,320, 321,
    328, 329, 332, 333, 335, 342,344,345,346,
    349, 335, 356, 357, 369, 372, 373,374, 375, 377,
    380, 381, 383, 384, 386, 387, 388, 393, 394, 404,
    411, 412, 415, 419, 422, 425, 426,427, 431,433,
    435, 436, 437, 443, 449, 451, 457, 458, 465, 471,
    486,487, 489, 490, 491,495, 498, 499, 501, 508, 509, 510]


# In[ ]:


params = {
7: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 0.01},
24: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.1},
27: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.1},
28: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01},
36: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.1},
42: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.1},
49: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.01},
52: {'init_params': 'random', 'n_init': 1, 'reg_covar': 1e-06},
54: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.0001},
71: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.0001},
72: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.0001},
79: {'init_params': 'kmeans', 'n_init': 5, 'reg_covar': 1e-08},
81: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.1},
85: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.1},
87: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.1},
89: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.1},
92: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.0001},
93: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.1},
94: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01},
95: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.0001},
96: {'init_params': 'random', 'n_init': 10, 'reg_covar': 0.01},
102: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.1}, 
103: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.1},
104: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 1e-08},
107: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 1e-06},
109: {'init_params': 'kmeans', 'n_init': 1, 'reg_covar': 0.0001},
114: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.01},
125: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.1},
129: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.01},
130: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01},
131: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.0001},
143: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01},
149: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01},
151: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.1},
152: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.0001},
162: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 0.01},
166: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 1e-08},
177: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 0.01},
182: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 0.01},
187: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01},
193: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.0001},
194: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.0001},
195: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 1e-08},
197: {'init_params': 'kmeans', 'n_init': 1, 'reg_covar': 0.0001},
198: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 1e-08},
206: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 1e-08},
207: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.1},
208: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01},
211: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 1e-06},
213: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 1e-06},
220: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.1},
221: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 0.1},
222: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.1},
230: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.01},
233: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.1},
234: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.0001},
235: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.1},
242: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 1e-08},
244: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.1},
251: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 0.0001},
252: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.01},
258: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 1e-06}, 
262: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.1},
263: {'init_params': 'kmeans', 'n_init': 5, 'reg_covar': 0.01},
264: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01},
266: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.1},
268: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.0001},
270: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01},
271: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.1},
273: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 0.0001},
287: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01},
289: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 1e-06},
293: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.1},
308: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 0.01},
310: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.1},
312: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 1e-06},
313: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 0.01},
315: {'init_params': 'random', 'n_init': 30, 'reg_covar': 0.0001},
320: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.0001},
321: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01},
328: {'init_params': 'random', 'n_init': 1, 'reg_covar': 0.0001},
329: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01},
332: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.01},
333: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 0.01},
335: {'init_params': 'random', 'n_init': 20, 'reg_covar': 0.01},
342: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.01},
344: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01},
345: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.1},
346: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.0001},
349: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01},
356: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 1e-06},
357: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01},
369: {'init_params': 'random', 'n_init': 30, 'reg_covar': 0.0001},
372: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 1e-06},
373: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01},
374: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 0.01},
375: {'init_params': 'kmeans', 'n_init': 5, 'reg_covar': 0.0001},
377: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.01},
380: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 1e-06},
381: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 0.01},
383: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 0.01},
384: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01},
386: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 1e-08},
387: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 1e-08},
388: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 0.1},
393: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 1e-06},
394: {'init_params': 'kmeans', 'n_init': 1, 'reg_covar': 1e-06},
404: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.0001},
411: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 0.0001},
412: {'init_params': 'kmeans', 'n_init': 1, 'reg_covar': 0.0001},
415: {'init_params': 'random', 'n_init': 1, 'reg_covar': 0.0001},
419: {'init_params': 'random', 'n_init': 30, 'reg_covar': 0.0001},
422: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.01},
425: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.01},
426: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.1},
427: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.0001},
431: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 1e-08},
433: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.0001},
435: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.1},
436: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 1e-08},
437: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 1e-06},
443: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 1e-06},
449: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.1},
451: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01},
457: {'init_params': 'kmeans', 'n_init': 1, 'reg_covar': 0.1},
458: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.01},
465: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 0.1},
471: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.1},
486: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01},
487: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.1},
489: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.0001},
490: {'init_params': 'kmeans', 'n_init': 1, 'reg_covar': 1e-06},
491: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.1},
495: {'init_params': 'kmeans', 'n_init': 5, 'reg_covar': 0.01},
498: {'init_params': 'kmeans', 'n_init': 1, 'reg_covar': 0.0001},
499: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.0001},
501: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 0.0001},
508: {'init_params': 'kmeans', 'n_init': 20, 'reg_covar': 0.1},
509: {'init_params': 'kmeans', 'n_init': 10, 'reg_covar': 0.1},
510: {'init_params': 'kmeans', 'n_init': 30, 'reg_covar': 0.01}}


# In[ ]:


# INITIALIZE VARIABLES
cols = [c for c in train.columns if c not in ['id', 'target']]
cols.remove('wheezy-copper-turtle-magic')
oof    = np.zeros((len(train),4))
preds  = np.zeros((len(test) ,4))
oof2   = np.zeros((len(train),4))
preds2 = np.zeros((len(test) ,4))
oof3   = np.zeros((len(train),1))
preds3 = np.zeros((len(test) ,1))
ooff   = np.zeros((len(train),4))
predsf = np.zeros((len(test) ,4))

# BUILD 512 SEPARATE MODELS
for i in tqdm(range(512)):
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    #pipe = Pipeline([('vt', VarianceThreshold(threshold=2)), ('scaler',  RobustScaler(quantile_range=(25, 75)))])
    pipe = Pipeline([('vt', VarianceThreshold(threshold=1.5)), ('scaler',  RobustScaler(quantile_range=(35, 65)))])
    sel = pipe.fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
    print(train3.shape)

    # embeddings1 = TSNE(n_jobs=4,
    #                    n_components=5,
    #                    perplexity=30.0,
    #                    early_exaggeration=12.0,
    #                    learning_rate=20.0,
    #                    n_iter=1000,
    #                    n_iter_without_progress=50
    #                   ).fit_transform( np.concatenate([train3,test3],axis = 0) )
    # train3 = np.hstack( (train3, embeddings1[:train3.shape[0],:]) )
    # test3  = np.hstack( ( test3, embeddings1[train3.shape[0]:,:]) )

    # km = KMeans(n_clusters=8, n_jobs=4 )
    # km = km.fit_transform( np.concatenate([train3,test3],axis = 0) )
    # train3 = np.hstack( (train3, km[:train3.shape[0],:]) )
    # test3  = np.hstack( ( test3, km[train3.shape[0]:,:]) )
    # print(train3.shape)

    NK = 2
    # STRATIFIED K-FOLD
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train3, train2['target']):
        oof_test_index = [t for t in test_index if t < len(idx1)]

        x = train3[train_index]
        y = train2.loc[train_index,'target'].values
        x1 = x[(y==1).astype(bool)]
        cc1 = Birch(threshold=0.5, branching_factor=150, n_clusters=NK, compute_labels=True).fit_predict(x1)
        x11 = x1[cc1==0]
        x12 = x1[cc1==1]
        model11 = OAS(assume_centered =False).fit(x11)
        p11 = model11.precision_
        m11 = model11.location_ 
        model12 = OAS(assume_centered =False).fit(x12)
        p12 = model12.precision_
        m12 = model12.location_ 

        x2 = x[(y==0).astype(bool)]
        cc2 = Birch(threshold=0.5, branching_factor=150, n_clusters=NK, compute_labels=True).fit_predict(x2)
        x21 = x2[cc2==0]
        x22 = x2[cc2==1]
        model21 =  OAS(assume_centered =False).fit(x21)
        p21 = model21.precision_
        m21 = model21.location_ 
        model22 =  OAS(assume_centered =False).fit(x22)
        p22 = model22.precision_
        m22 = model22.location_ 

        ms = np.stack([m11,m12,m21,m22])
        ps = np.stack([p11,p12,p21,p22])

        gm = GaussianMixture(n_components=4,random_state=42, covariance_type='full', means_init=ms, precisions_init=ps, tol=0.000001, max_iter=1000 )
        gm.fit(np.concatenate([train3[train_index],test3],axis = 0))
        oof[idx1[oof_test_index]] = gm.predict_proba(train3[oof_test_index,:])
        preds[idx2] += gm.predict_proba(test3) / skf.n_splits

        x = train3[train_index[:int(0.95*len(train_index))]]
        y = train2.loc[train_index[:int(0.95*len(train_index))],'target'].values
        x1 = x[(y==1).astype(bool)]
        cc1 = Birch(threshold=0.5, branching_factor=150, n_clusters=NK, compute_labels=True).fit_predict(x1)
        x11 = x1[cc1==0]
        x12 = x1[cc1==1]
        model11 = OAS(assume_centered =False).fit(x11)
        p11 = model11.precision_
        m11 = model11.location_ 
        model12 = OAS(assume_centered =False).fit(x12)
        p12 = model12.precision_
        m12 = model12.location_ 

        x2 = x[(y==0).astype(bool)]
        cc2 = Birch(threshold=0.5, branching_factor=150, n_clusters=NK, compute_labels=True).fit_predict(x2)
        x21 = x2[cc2==0]
        x22 = x2[cc2==1]
        model21 =  OAS(assume_centered =False).fit(x21)
        p21 = model21.precision_
        m21 = model21.location_ 
        model22 =  OAS(assume_centered =False).fit(x22)
        p22 = model22.precision_
        m22 = model22.location_ 

        ms = np.stack([m11,m12,m21,m22])
        ps = np.stack([p11,p12,p21,p22])    

        gm = GaussianMixture(n_components=4,random_state=43, covariance_type='full', means_init=ms, precisions_init=ps, tol=0.000001, max_iter=500 )
        gm.fit(np.concatenate([train3[train_index[:int(0.95*len(train_index))]],test3[:int(0.95*test3.shape[0])]],axis = 0))
        oof[idx1[oof_test_index]] += gm.predict_proba(train3[oof_test_index,:])
        preds[idx2] += gm.predict_proba(test3) / skf.n_splits



        x = train3[train_index[int(0.15*len(train_index)):]]
        y = train2.loc[train_index[int(0.15*len(train_index)):],'target'].values
        x1 = x[(y==1).astype(bool)]
        cc1 = Birch(threshold=0.5, branching_factor=150, n_clusters=NK, compute_labels=True).fit_predict(x1)
        x11 = x1[cc1==0]
        x12 = x1[cc1==1]
        model11 = OAS(assume_centered =False).fit(x11)
        p11 = model11.precision_
        m11 = model11.location_ 
        model12 = OAS(assume_centered =False).fit(x12)
        p12 = model12.precision_
        m12 = model12.location_ 

        x2 = x[(y==0).astype(bool)]
        cc2 = Birch(threshold=0.5, branching_factor=150, n_clusters=NK, compute_labels=True).fit_predict(x2)
        x21 = x2[cc2==0]
        x22 = x2[cc2==1]
        model21 =  OAS(assume_centered =False).fit(x21)
        p21 = model21.precision_
        m21 = model21.location_ 
        model22 =  OAS(assume_centered =False).fit(x22)
        p22 = model22.precision_
        m22 = model22.location_ 

        ms = np.stack([m11,m12,m21,m22])
        ps = np.stack([p11,p12,p21,p22])      

        gm = GaussianMixture(n_components=4,random_state=44, covariance_type='full', means_init=ms, precisions_init=ps, tol=0.000001, max_iter=500 )
        gm.fit(np.concatenate([train3[train_index[int(0.15*len(train_index)):]],test3[int(0.15*test3.shape[0]):]],axis = 0))
        oof[idx1[oof_test_index]] += gm.predict_proba(train3[oof_test_index,:])
        preds[idx2] += gm.predict_proba(test3) / skf.n_splits

#         clf = LGBMClassifier(num_leaves=7,
#                              max_depth=-1,
#                              learning_rate=0.01,
#                              n_estimators=1000,
#                              min_child_samples=50,
#                              subsample=0.34,
#                              subsample_freq=1,
#                              colsample_bytree=0.67,
#                              reg_alpha=0.1)
#         train_index2 = train_index[ int(0.05*len(train_index)): ] 
#         clf.fit(train3[train_index2,:],train2.loc[train_index2]['target'])
#         oof3[idx1[test_index],0] += clf.predict_proba(train3[oof_test_index,:])[:,1]
#         preds3[idx2,0]           += clf.predict_proba(test3)[:,1] / skf.n_splits

        train_index2 = train_index[ :int(0.95*len(train_index)) ] 
        clf = KNeighborsClassifier(n_neighbors=50, metric='braycurtis' )
        clf.fit(train3[train_index2,:],train2.loc[train_index2]['target'])
        oof3[idx1[test_index],0] += clf.predict_proba(train3[oof_test_index,:])[:,1]
        preds3[idx2,0]           += clf.predict_proba(test3)[:,1] / skf.n_splits



    print( roc_auc_score(train2['target'], oof[idx1,0]+oof[idx1,1] ) )
    # print('Optimize')
    # for train_index, test_index in skf.split(train3, train2['target']):
    #     def min_func( k ):
    #         tmp = ( oof[idx1[train_index],0]*k[0])
    #         tmp+= ( oof[idx1[train_index],1]*k[1])
    #         sc = roc_auc_score( train2['target'].values[train_index], tmp  )
    #         return -sc
    #     k = minimize(min_func, [1.,1.] , method='Nelder-Mead', tol=1e-6,  options={'maxiter': 100} ).x
    #     tmp = ( oof[idx1[test_index],0]*k[0])
    #     tmp+= ( oof[idx1[test_index],1]*k[1])
    #     ooff[idx1[test_index],0] = tmp

    # print( roc_auc_score(train2['target'], ooff[idx1,0]) )
    print( roc_auc_score(train2['target'], oof[idx1,0]+oof[idx1,1]+0.10*oof3[idx1,0] ) )
    print( roc_auc_score(train2['target'], oof[idx1,0]+oof[idx1,1]+0.20*oof3[idx1,0] ) )
    print( roc_auc_score(train2['target'], oof[idx1,0]+oof[idx1,1]+0.40*oof3[idx1,0] ) )
    print( roc_auc_score(train2['target'], oof[idx1,0]+oof[idx1,1]+0.50*oof3[idx1,0] ) )
    print( roc_auc_score(train2['target'], oof[idx1,0]+oof[idx1,1]+0.60*oof3[idx1,0] ) )
    print( roc_auc_score(train2['target'], oof[idx1,0]+oof[idx1,1]+0.80*oof3[idx1,0] ) )
    print( roc_auc_score(train2['target'], oof[idx1,0]+oof[idx1,1]+1.00*oof3[idx1,0] ) )

    px = np.where( train['wheezy-copper-turtle-magic']<=i )[0]
    print( roc_auc_score(train['target'].values[px], oof[px,0]+oof[px,1]+0.10*oof3[px,0] ) )


# In[ ]:


# INITIALIZE VARIABLES
cols = [c for c in train.columns if c not in ['id', 'target']]
cols.remove('wheezy-copper-turtle-magic')
oof    = np.zeros((len(train),4))
preds  = np.zeros((len(test) ,4))
oof2   = np.zeros((len(train),4))
preds2 = np.zeros((len(test) ,4))
oof3   = np.zeros((len(train),1))
preds3 = np.zeros((len(test) ,1))
ooff   = np.zeros((len(train),4))
predsf = np.zeros((len(test) ,4))
K2=[]
K3=[]
K4=[]

# BUILD 512 SEPARATE MODELS
for i in tqdm(range(512)):
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    #pipe = Pipeline([('vt', VarianceThreshold(threshold=2)), ('scaler',  RobustScaler(quantile_range=(25, 75)))])
    pipe = Pipeline([('vt', VarianceThreshold(threshold=1.5)), ('scaler',  RobustScaler(quantile_range=(35, 65)))])
    sel = pipe.fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
    print(train3.shape)

    NK = 2
    # STRATIFIED K-FOLD
    skf = StratifiedKFold(n_splits=16, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train3, train2['target']):
        oof_test_index = [t for t in test_index if t < len(idx1)]

        x = train3[train_index]
        y = train2.loc[train_index,'target'].values
        x1 = x[(y==1).astype(bool)]
        cc1 = Birch(threshold=0.5, branching_factor=100, n_clusters=NK, compute_labels=True).fit_predict(x1)
        x11 = x1[cc1==0]
        x12 = x1[cc1==1]
        model11 = OAS(assume_centered =False).fit(x11)
        p11 = model11.precision_
        m11 = model11.location_ 
        model12 = OAS(assume_centered =False).fit(x12)
        p12 = model12.precision_
        m12 = model12.location_ 

        x2 = x[(y==0).astype(bool)]
        cc2 = Birch(threshold=0.5, branching_factor=100, n_clusters=NK, compute_labels=True).fit_predict(x2)
        x21 = x2[cc2==0]
        x22 = x2[cc2==1]
        model21 =  OAS(assume_centered =False).fit(x21)
        p21 = model21.precision_
        m21 = model21.location_ 
        model22 =  OAS(assume_centered =False).fit(x22)
        p22 = model22.precision_
        m22 = model22.location_ 

        ms = np.stack([m11,m12,m21,m22])
        ps = np.stack([p11,p12,p21,p22])

        gm = GaussianMixture(n_components=4,random_state=42, covariance_type='full', means_init=ms, precisions_init=ps, tol=0.000001, max_iter=1000 )
        gm.fit(np.concatenate([train3[train_index],test3],axis = 0))
        oof[idx1[oof_test_index]] = gm.predict_proba(train3[oof_test_index,:])
        #preds[idx2] += gm.predict_proba(test3) / skf.n_splits
    sc2 = roc_auc_score(train2['target'], oof[idx1,0]+oof[idx1,1] )

    
    NK = 3
    # STRATIFIED K-FOLD
    for train_index, test_index in skf.split(train3, train2['target']):
        oof_test_index = [t for t in test_index if t < len(idx1)]

        x = train3[train_index]
        y = train2.loc[train_index,'target'].values
        x1 = x[(y==1).astype(bool)]
        cc1 = Birch(threshold=0.5, branching_factor=100, n_clusters=NK, compute_labels=True).fit_predict(x1)
        x11 = x1[cc1==0]
        x12 = x1[cc1==1]
        model11 = OAS(assume_centered =False).fit(x11)
        p11 = model11.precision_
        m11 = model11.location_ 
        model12 = OAS(assume_centered =False).fit(x12)
        p12 = model12.precision_
        m12 = model12.location_ 

        x2 = x[(y==0).astype(bool)]
        cc2 = Birch(threshold=0.5, branching_factor=100, n_clusters=NK, compute_labels=True).fit_predict(x2)
        x21 = x2[cc2==0]
        x22 = x2[cc2==1]
        model21 =  OAS(assume_centered =False).fit(x21)
        p21 = model21.precision_
        m21 = model21.location_ 
        model22 =  OAS(assume_centered =False).fit(x22)
        p22 = model22.precision_
        m22 = model22.location_ 

        ms = np.stack([m11,m12,m21,m22])
        ps = np.stack([p11,p12,p21,p22])

        gm = GaussianMixture(n_components=4,random_state=42, covariance_type='full', means_init=ms, precisions_init=ps, tol=0.000001, max_iter=1000 )
        gm.fit(np.concatenate([train3[train_index],test3],axis = 0))
        oof[idx1[oof_test_index]] = gm.predict_proba(train3[oof_test_index,:])
        #preds[idx2] += gm.predict_proba(test3) / skf.n_splits
    sc3 = roc_auc_score(train2['target'], oof[idx1,0]+oof[idx1,1] )

    NK = 4
    # STRATIFIED K-FOLD
    for train_index, test_index in skf.split(train3, train2['target']):
        oof_test_index = [t for t in test_index if t < len(idx1)]

        x = train3[train_index]
        y = train2.loc[train_index,'target'].values
        x1 = x[(y==1).astype(bool)]
        cc1 = Birch(threshold=0.5, branching_factor=100, n_clusters=NK, compute_labels=True).fit_predict(x1)
        x11 = x1[cc1==0]
        x12 = x1[cc1==1]
        model11 = OAS(assume_centered =False).fit(x11)
        p11 = model11.precision_
        m11 = model11.location_ 
        model12 = OAS(assume_centered =False).fit(x12)
        p12 = model12.precision_
        m12 = model12.location_ 

        x2 = x[(y==0).astype(bool)]
        cc2 = Birch(threshold=0.5, branching_factor=100, n_clusters=NK, compute_labels=True).fit_predict(x2)
        x21 = x2[cc2==0]
        x22 = x2[cc2==1]
        model21 =  OAS(assume_centered =False).fit(x21)
        p21 = model21.precision_
        m21 = model21.location_ 
        model22 =  OAS(assume_centered =False).fit(x22)
        p22 = model22.precision_
        m22 = model22.location_ 

        ms = np.stack([m11,m12,m21,m22])
        ps = np.stack([p11,p12,p21,p22])

        gm = GaussianMixture(n_components=4,random_state=42, covariance_type='full', means_init=ms, precisions_init=ps, tol=0.000001, max_iter=1000 )
        gm.fit(np.concatenate([train3[train_index],test3],axis = 0))
        oof[idx1[oof_test_index]] = gm.predict_proba(train3[oof_test_index,:])
        #preds[idx2] += gm.predict_proba(test3) / skf.n_splits
    sc4 = roc_auc_score(train2['target'], oof[idx1,0]+oof[idx1,1] )

        
    K2.append(sc2)
    K3.append(sc3)
    K4.append(sc4)

    print( sc2,sc3,sc4 )
    
    
    
    


# In[ ]:


print( ','.join(K2) )
print( ','.join(K3) )
print( ','.join(K4) )


# In[ ]:




