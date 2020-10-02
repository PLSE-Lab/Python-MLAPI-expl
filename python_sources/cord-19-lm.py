"""
Language models
"""

import numpy as np

# p_coll: Maximum likelihood estimate of word w in the collection.
def mle(TF):
    p = TF.sum(0)
    p = np.asarray(p).squeeze()
    p = p / p.sum()
    
    return p

def linearly_smooth(p_ml, p_coll, lambda_=0.6):
    return lambda_*p_ml + (1.-lambda_)*p_coll

def compute_ql(corpus, Q, lambda_=0.6):
    """Compute query likelihood
    Q: query BoW
    """
    Q_tokensid = [tokenid for tokenid, tf in Q]
    
    p_ml_q_d = corpus.TRF[:,Q_tokensid]
    # We should assert that p_coll_w are all != 0
    p_coll_w = corpus.p_coll[Q_tokensid]
    
    p_q_d = linearly_smooth(p_ml_q_d, p_coll_w, lambda_)
    
    query_likelihood = np.exp(np.sum(np.log(p_q_d), axis=1))
    
    query_likelihood /= query_likelihood.sum()
    
    return query_likelihood


def compute_rm1(rel_freq, p_coll, query_likelihood, lambda_=0.6):
    """Compute Relevance model RM1
    query language model
    query_likelihood = P(Q|D) ~ P(D|Q)
    P(D|Q) = [q,d]
    rel_freq = P(w|d) [d,w]
    return:
        P(w|Q)
    """
    if rel_freq.ndim==1:
        rel_freq = rel_freq.reshape(1,-1)
    
    if query_likelihood.ndim==1:
        query_likelihood = query_likelihood.reshape(1,-1)
    
    query_likelihood = query_likelihood / query_likelihood.sum(1).reshape(-1, 1)
    
    assert rel_freq.shape[0] == query_likelihood.shape[1], f'{rel_freq.shape} == {query_likelihood.shape}'
    assert rel_freq.shape[1] == len(p_coll), f'{rel_freq.shape} == {len(p_coll)}'
    
    p_w_d = linearly_smooth(rel_freq, p_coll, lambda_)
    
    #p_w_Q
    qlm = query_likelihood @ p_w_d

    qlm /= qlm.sum(1).reshape(-1,1)
    
    qlm=np.squeeze(qlm)
    
    return qlm

def compute_rm3(rm1, p_w_q, lambda_=0.6):
    """Compute Relevance model RM3
    """
    
    assert (p_w_q.ndim == 1) and (rm1.ndim == 1), f'p_w_q.ndim={p_w_q.ndim} rm1.ndim={rm1.ndim}'
    
    rm3 = linearly_smooth(rm1, p_w_q, lambda_)
    
    # Normalize to probality
    rm3[:] /= sum(rm3)
    
    return rm3

def compute_tm_rm(ranker, R, Q, query_likelihood, rm3, lambda_=0.6):
    """
    For LDA this is like the formula:
    gamma, phi = ldamodel.inference(corpus, collect_sstats=True)
    p_w_tm = phi
    p_tm_D = gamma[I] #Normalize
    p_Q_t, _ = ldamodel.inference(query, collect_sstats=False)
    p_Q_t /= p_Q_t.sum(1)[:, None]
    """
    
    # This is an approximation using LDI
    p_w_tm = ranker.B # P(w|t_m)
    p_tm_D = ranker.project(R) # P(t_m|D)
    p_Q_t  = ranker.project(Q) # P(Q|t_m)
    
    p_tm = p_tm_D*p_Q_t
    p_tm = p_tm@p_w_tm
    p_tm *= query_likelihood.reshape(-1,1)
    p_tm = p_tm.sum(0)
    p_tm[:] /= sum(p_tm)
    
    p_tm_rm = linearly_smooth(p_tm, rm3, lambda_)
    
    return p_tm_rm

def get_relevant(corpus, score, topk):
    """Return topk documents
    Old code...
    """
    I = np.argsort(score, kind='stable')[::-1]
    
    r = []
    for i in I:
        i = i.item()
        paper = corpus[i]
        r.append(i)
            
        if len(r) == topk:
            break
    
    return r, corpus[r]
