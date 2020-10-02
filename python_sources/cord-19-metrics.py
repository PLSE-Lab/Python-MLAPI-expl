from cord_19_container import Document, Corpus
import cord_19_lm as lm

from gensim import matutils

import numpy as np
from scipy.stats import pearsonr

# calculate the kl divergence
# https://machinelearningmastery.com/divergence-between-probability-distributions/
def kl_divergence(p, q):
    return sum(p*np.log2(p/q))

# calculate the js divergence
def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def compute_cs(qlm, p_coll):
    """ Clarity score
    """
    return js_divergence(qlm, p_coll)


# It's a framework should be transformed to a Class, and works with others metris to.
def compute_uef_cs(corpus, R, rm1, q, score, query_likelihood,
        rm1_lambda=0.6, rm3_lambda=0.6):
    """ UEF(CS)
    """
    
    TF = R.TF.copy()
    # Set to 0 words not in query
    mask = np.isin(np.arange(len(corpus.dictionary)), list(q.tokensid_set), invert=True)
    TF[:,mask] = 0
    TRF = TF / (TF.sum(1).reshape(-1,1)+1e-18)
    p_w_q = lm.compute_rm1(TRF, R.p_coll, query_likelihood, lambda_=1.)
    rm3 = lm.compute_rm3(rm1, p_w_q, lambda_=rm3_lambda)
    p_w_d = lm.linearly_smooth(R.TRF, corpus.p_coll, lambda_=rm1_lambda)
    Rqs_score = p_w_d@rm3
    
    c = pearsonr(score, Rqs_score)[0]
    #assert c>0, f'Bad correlation {c:.6f}'
    
    #[-1,1] -> [0,1]
    c = (c+1)/2
    
    clarity_score = compute_cs(rm3, corpus.p_coll)
    
    return c*clarity_score


def compute_queries_perf(corpus, q_corpus, scores, query_likelihoods,
                         kind='cs',
                         n_relevant = 150,
                         rm1_lambda=0.6, rm3_lambda=0.6):
    """
    scores shape
        1 model, 1 query: [n]
        1 model, q query: [1,q,n]
        m model, 1 query: [m,n]
        m model, q query: [m,q,n]
            n=#documents
    
    Return
        queries performance using Clarity or UEF(Clarity) [m, q]
    """

    assert kind in ['cs', 'uef']
    
    if isinstance(q_corpus, Document):
        q_corpus = Corpus([q_corpus])
    if query_likelihoods.ndim == 1:
        query_likelihoods = query_likelihoods[None,:]
    if scores.ndim == 1:
        scores = scores[None,:]
    if scores.ndim == 2:
        scores = scores[None,:,:]
    
    performances = np.empty([len(scores), len(q_corpus)])
    
    for qi, q in enumerate(q_corpus):
        # q = q.bow
        for i, score in enumerate(scores):
            q_score = score[qi]
            
            I, R = lm.get_relevant(corpus, q_score, n_relevant)
            
            query_likelihood = query_likelihoods[qi, I]
            rm1 = lm.compute_rm1(R.TRF, corpus.p_coll, query_likelihood, lambda_=rm1_lambda)
            
            if kind=='uef':
                perf = compute_uef_cs(corpus, R, rm1,
                                      q, q_score[I], query_likelihood,
                                      rm1_lambda=rm1_lambda, rm3_lambda=rm3_lambda)
            elif kind=='cs':
                perf = compute_cs(rm1, corpus.p_coll)
            
            performances[i, qi] = perf
    
    return performances.squeeze()
