#!/usr/bin/env python
# coding: utf-8

# # Bayesian Network 
# 
# A Bayesian Network (BN) is a specific Probabilistic Graphical Model (PGM) in the form of a Directed Acyclical Graph where 
# - a Node represents a Random Variable (RV)
# - an Edge represents Probabilistic Dependence between RVs 
# 
# so $ \mathcal{G} = (\{N\}_{i=1,...,n}, \{E\}_{j=1,...,m}) $ where 
# 
# - $ A \in \{N\} $ node, represents the node "A"
# - $ e_{A,B} \in \{E\} $ edge, represents the dependency of "A" from "B" 
# 
# The goal is to perform the **Model Status Estimation** which means performing the Estimation of the Latent Variables Joint PDF i.e. $ P(\{E\}) = P(A,B,C, ...) $ which factorizes according to the Probabilistic Relationships 
# 
# E.g. if A --> B then it means 
# 
# $ P(A,B) = P(A) P(B|A) $
# 
# 

# # Status Estimation  
# 
# The Latent Variables Joint PDF represents how the Information Processing System (that's what the PGM essentially is) evolves according to an External Stimulus (Observation) 
# 
# It means performing the BN Latent Variable Estimation according to the Observation $ P(Z |X) $ where
# 
# - $ Z $ : Latent Variable Set 
# - $ X $ : Observation Set 
# 
# From a purely Theoretical Bayesian Point of View the computation can be performed using Bayesian Rule 
# 
# $ P(Z|X) = \frac{P(X|Z)P(Z)}{P(X)} $
# 
# - $ P(X|Z) $ : Observation Likelihood (i.e. the interaction between the Observation and the Current State)
# - $ P(Z) $ : System State Prior 
# - $ P(X) $ : Observation Prior 
# 
# However this is typically intractable hence approximation strategies are used, namely 
# 
# - Variational Inference 
# - Monte Carlo Methods 
# 
# 

# # Variational Inference 
# 
# The Variational Inference Idea is to define an Approximation of the True Posterior $ P(Z|X) $ as a Meta Prior which is typically called **Variational Posterior** $ q_{\phi}(Z|X) $ where $ \phi $ is the Param Set resulting from an optimization process (e.g. learned in a Data Driven way)
# 
# In order to set up the Optimization Process to perform the Variational Posterior Estimation, a Similarity Distance Function to be minimized is needed and in case of PDF the most typical choice is KL Divergence hence the Minimization Problem becomes 
# 
# $$ \hat \phi = \arg\min_{\phi} KL(q_{\phi}(Z|X) || P(Z|X)) $$
# 
# 
# 

# # Dataset Generation 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import theano
floatX = theano.config.floatX
import pymc3 as pm
import theano.tensor as T
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
sns.set_style('white')
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons


# In[ ]:


X, Y = make_moons(noise=0.2, random_state=0, n_samples=1000)
X = scale(X)
X = X.astype(floatX)
Y = Y.astype(floatX)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)


# In[ ]:


fig, ax = plt.subplots()
ax.scatter(X[Y==0, 0], X[Y==0, 1], label='Class 0')
ax.scatter(X[Y==1, 0], X[Y==1, 1], color='r', label='Class 1')
sns.despine(); ax.legend()
ax.set(xlabel='X', ylabel='Y', title='Toy binary classification data set');


# # Dataset Processing 

# In[ ]:


from pymc3.theanof import set_tt_rng, MRG_RandomStreams
set_tt_rng(MRG_RandomStreams(42))


# In[ ]:


# Simple NN 
def construct_nn(ann_input, ann_output):
    n_hidden = 5

    # Initialize random weights between each layer
    init_1 = np.random.randn(X.shape[1], n_hidden).astype(floatX)
    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_out = np.random.randn(n_hidden).astype(floatX)

    with pm.Model() as neural_network:
        # Weights from input to hidden layer
        weights_in_1 = pm.Normal('w_in_1', 0, sd=1,
                                 shape=(X.shape[1], n_hidden),
                                 testval=init_1)

        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal('w_1_2', 0, sd=1,
                                shape=(n_hidden, n_hidden),
                                testval=init_2)

        # Weights from hidden layer to output
        weights_2_out = pm.Normal('w_2_out', 0, sd=1,
                                  shape=(n_hidden,),
                                  testval=init_out)

        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input,
                                         weights_in_1))
        act_2 = pm.math.tanh(pm.math.dot(act_1,
                                         weights_1_2))
        act_out = pm.math.sigmoid(pm.math.dot(act_2,
                                              weights_2_out))

        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli('out',
                           act_out,
                           observed=ann_output,
                           total_size=Y_train.shape[0] # IMPORTANT for minibatches
                          )
    return neural_network

# Trick: Turn inputs and outputs into shared variables.
# It's still the same thing, but we can later change the values of the shared variable
# (to switch in the test-data later) and pymc3 will just use the new data.
# Kind-of like a pointer we can redirect.
# For more info, see: http://deeplearning.net/software/theano/library/compile/shared.html
ann_input = theano.shared(X_train)
ann_output = theano.shared(Y_train)
neural_network = construct_nn(ann_input, ann_output)


# # Variational Inference 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nwith neural_network:\n    inference = pm.ADVI()\n    approx = pm.fit(n=30000, method=inference)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nwith neural_network:\n    advifit = pm.advi(n=30000)')


# In[ ]:




