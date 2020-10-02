from cord_19_container import Document, Corpus
from cord_19_lm import compute_ql

from gensim.models import TfidfModel
from gensim.similarities import MatrixSimilarity
from gensim.interfaces import TransformedCorpus
from gensim import matutils

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

dictionary = None

# Interface not used for the moment, it can help to produce most readble code and easy to maintain.
class RankerABC:
    @property
    def name(self):
        name = type(self).__name__
        return name
    
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name
    
    def __repr__(self):
        return self.name

class ranker_QL():
    def __init__(self, corpus, lambda_=0.6):
        self.corpus = corpus
        self.lambda_ = lambda_
    
    def set_params(self, lambda_):
        self.lambda_ = lambda_
    
    def __getitem__(self, q_corpus, renorm=False):
        if isinstance(q_corpus, Document):
            q_corpus = Corpus([q_corpus])
        
        sim = np.asarray([compute_ql(self.corpus, q.bow, lambda_=self.lambda_) for q in q_corpus])
        #sim /= sim.sum(1).reshape(-1,1)
        sim = np.squeeze(sim)
        
        return sim

class Ranker_TFIDF():
    def __init__(self, corpus, smartirs='nfc'):
        self.tfidf = TfidfModel(dictionary=dictionary, smartirs=smartirs)
        self.ranker = MatrixSimilarity(self.tfidf[corpus.bow], num_features=len(dictionary))
    
    def project(self, bow):
        return self.tfidf[bow]
        
    def __getitem__(self, corpus, renorm=False):
        if isinstance(corpus, Document):
            corpus = Corpus([corpus])
        
        x = self.project(corpus.bow)
        
        sim = self.ranker[x]
        if renorm:
            sim /= sim.sum(1).reshape(-1,1)
        sim = np.squeeze(sim)
        
        return sim

class Ranker_LSI():
    def __init__(self, corpus, model, pre_model=None):
        self.model = model
        self.pre_model = pre_model
        
        x = corpus.bow
        x = self.project(x)
        
        self.ranker = MatrixSimilarity(x, num_features=len(dictionary))
    
    def project(self, x):
        if self.pre_model:
            x = self.pre_model[x]
            
        x = self.model[x]
        
        return x
    
    def __getitem__(self, corpus, renorm=False):
        if isinstance(corpus, Document):
            corpus = Corpus([corpus])
        
        x = corpus.bow
        x = self.project(x)
        
        sim = self.ranker[x]
        if renorm:
            sim /= sim.sum(1).reshape(-1,1)
        sim=np.squeeze(sim)
        
        return sim

class Ranker_LDI():
    def __init__(self, ldamodel, corpus, subset=None):
        self.ldamodel = ldamodel
        
        B = self.ldamodel.state.get_lambda()
        
        if subset is not None:
            B=B[subset]
        
        excluded_words = (B.sum(axis=0)==0.)
        excluded_topics = (B.sum(axis=1)==0.)
        
        denom = B.sum(axis=1)
        self.B = B / denom[:, None]
        
        denom = self.B.sum(axis=0)
        denom[excluded_words] = 1.
        
        self.W = (self.B / denom)

        self.num_topics = self.W.shape[0]
        self.D = self.project(corpus)
    
    def project(self, corpus):
        if isinstance(corpus, Document):
            corpus = Corpus([corpus])
        
        if isinstance(corpus, (np.ndarray,)):
            D = (corpus @ self.W.T) / corpus.sum(1).reshape(-1,1)
        else:
            D = np.zeros([len(corpus), self.num_topics])

            for i, doc in enumerate(corpus):
                tokens_id = [x[0] for x in doc.bow]
                tokens_tf = [x[1] for x in doc.bow]

                D[i,:] = (self.W[:,tokens_id]*tokens_tf).sum(axis=1)/sum(tokens_tf)
    
        return D
    
    def __getitem__(self, corpus, renorm=False):
        if isinstance(corpus, Document):
            corpus = Corpus([corpus])
            
        Q = self.project(corpus)
        
        sim = cosine_similarity(Q, self.D)
        if renorm:
            sim /= sim.sum(1).reshape(-1,1)
        sim=np.squeeze(sim)
        
        return sim

"""
Temporary, Ranker_NMF is the same class as Ranker_LDI.
Even if here we use pre_model
We are still looking how to rename those classes to a generic name
    (Ranker_NMF, Ranker_LDI and even Ranker_LSI)
    then merge them
"""
class Ranker_NMF():
    def __init__(self, B, corpus, pre_model=None):
        self.pre_model = pre_model
        
        denom = B.sum(axis=1)
        self.B = B / denom[:, None]
        
        denom = self.B.sum(axis=0)
        
        self.W = self.B / denom

        self.num_topics = self.W.shape[0]
        self.D = self.project(corpus)
    
    def project(self, corpus):
        if isinstance(corpus, Document):
            corpus = Corpus([corpus])
        
        if self.pre_model:
            if isinstance(corpus, list):
                corpus = self.pre_model[ [d.bow for d in corpus] ]
            else:
                corpus = self.pre_model[corpus.bow]
        
        if isinstance(corpus, Document):
            corpus = Corpus([corpus])
        elif isinstance(corpus, TransformedCorpus):
            # Sparse to dense representation
            corpus = matutils.corpus2csc(corpus, len(dictionary))
            corpus = corpus.T.toarray()

        if isinstance(corpus, (np.ndarray, )):
            D = (corpus @ self.W.T) / corpus.sum(1).reshape(-1,1)
        else:
            D = np.zeros([len(corpus), self.num_topics])

            for i, doc in enumerate(corpus):
                tokens_id = [x[0] for x in doc.bow]
                tokens_tf = [x[1] for x in doc.bow]

                D[i,:] = (self.W[:,tokens_id]*tokens_tf).sum(axis=1)/sum(tokens_tf)
    
        return D
    
    def __getitem__(self, corpus, renorm=False):
        if isinstance(corpus, Document):
            corpus = Corpus([corpus])
            
        Q = self.project(corpus)
        
        sim = cosine_similarity(Q, self.D)
        if renorm:
            sim /= sim.sum(1).reshape(-1,1)
        sim = np.squeeze(sim)
        
        return sim

class Ranker_EnM():
    """
    Compute ensemble models score
    return 
    """
    def __init__(self, models, alpha=None, renorm=False):
        if alpha is None:
            alpha = np.ones(len(models))
            alpha /= sum(alpha)
        
        self.renorm = renorm
        self.alpha = alpha
        assert len(models)==len(alpha)
        
        self.scores = None
        
        self.models = models
    
    def set_alpha(self, alpha):
        self.alpha = np.asarray(alpha)
    
    @property
    def models_name(self):
        return list(self.models.keys())
    
    def get_models_index(self, names):
        my_names = self.models_name
        return [my_names.index(name) for name in names]
    
    def get_model_index(self, name):
        my_names = self.models_name
        if name in my_names:
            return my_names.index(name)
        else:
            return -1
    
    def get_model_score(self, name):
        i = self.get_model_index(name)
        if i != -1:
            return self.scores[i]
            
    
    def combine_scores(self):
        expand_dim = (-1,)+(1,)*(self.scores.ndim-1)
        scores = self.scores*self.alpha.reshape(expand_dim)
        score = scores.sum(0)
        score=np.squeeze(score)
        
        return score
    
    def __getitem__(self, corpus):
        if isinstance(corpus, Document):
            corpus = Corpus([corpus])
        
        self.scores = np.asarray([model.__getitem__(corpus, renorm=self.renorm) for model in self.models.values()])
        score = self.combine_scores()

        return score
    
# tmp class
class NMFModel():
    def __init__(self, U):
        self.U = U
        self.id2word = dictionary
        
    def get_topics(self):
        """Get the topic vectors.

        Notes
        -----
        The number of topics can actually be smaller than `self.num_topics`, if there were not enough factors
        in the matrix (real rank of input matrix smaller than `self.num_topics`).

        Returns
        -------
        np.ndarray
            The term topic matrix with shape (`num_topics`, `vocabulary_size`)

        """
        projections = self.U.T
        num_topics = len(projections)
        topics = []
        for i in range(num_topics):
            c = np.asarray(projections[i, :]).flatten()
            norm = np.sqrt(np.sum(np.dot(c, c)))
            topics.append(1.0 * c / norm)
        return np.array(topics)
        
    def show_topic(self, topicno, topn=10):
        """Get the words that define a topic along with their contribution.

        This is actually the left singular vector of the specified topic.

        The most important words in defining the topic (greatest absolute value) are included
        in the output, along with their contribution to the topic.

        Parameters
        ----------
        topicno : int
            The topics id number.
        topn : int
            Number of words to be included to the result.

        Returns
        -------
        list of (str, float)
            Topic representation in BoW format.

        """
        # size of the projection matrix can actually be smaller than `self.num_topics`,
        # if there were not enough factors (real rank of input matrix smaller than
        # `self.num_topics`). in that case, return an empty string
        if topicno >= len(self.U.T):
            return ''
        c = np.asarray(self.U.T[topicno, :]).flatten()
        norm = np.sqrt(np.sum(np.dot(c, c)))
        most = matutils.argsort(np.abs(c), topn, reverse=True)
        return [(dictionary.id2token[val], 1.0 * c[val] / norm) for val in most]
