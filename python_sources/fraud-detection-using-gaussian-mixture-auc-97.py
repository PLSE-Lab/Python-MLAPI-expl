#!/usr/bin/env python
# coding: utf-8

# # Credit card fraud detection
# 
# #### This notebook will use Gaussian Mixture model to fit and predict credit card fraud detection dataset
# 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading the dataset

# In[ ]:


df = pd.read_csv("../input/creditcard.csv")
df.head()


# # Checking the target classes

# In[ ]:


count_classes = pd.value_counts(df['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")


# # Gaussian model 

# In[ ]:


class Gaussian:
    def __init__(self, mu, sigma):
        # mean and standard deviation
        self.mu = mu
        self.sigma = sigma

    def pdf(self, datum):
        "probability of a data point given the current parameters"
        u = (datum - self.mu) / abs(self.sigma)
        y = (1 / (np.sqrt(2 * np.pi) * abs(self.sigma))) * np.exp(-u * u / 2)
        return y

    def log_pdf(self, datum):
        "log probability of a data point given the current parameters"
        u = (datum - self.mu) / abs(self.sigma)
        y = np.log((1 / (np.sqrt(2 * np.pi) * abs(self.sigma)))) + (-u * u / 2)
        return y

    def log_pdf_np(self, X):
        Y = (X - self.mu) / abs(self.sigma)
        Y = np.log((1 / (np.sqrt(2 * np.pi) * abs(self.sigma)))) + (-Y ** 2 / 2)
        return Y


# # Gaussian Mixture Model
# 

# In[ ]:


class GaussianMixture:
    """
        using numpy package for computation
        optimize the computational speed of the em algorithm
    """
    def __init__(self, n_components):
        # list of gaussian components
        self.g = None

        # weights of the gaussian components
        self.mix = None

        # the number of mixture components.
        self.n_components = n_components

        # the convergence threshold.
        # em iterations will stop when the lower bound average gain is below this threshold.
        self.tol = 0.001

        # number of step used by the best fit of EM to reach the convergence.
        self.n_iter_ = None

        # the method used to initialize the weights, the means and the precisions.
        self.init_params = 'kmeans'

    def _initialize_parameters(self, X, random_state=42):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance.
        """
        n_samples, _ = X.shape

        if self.init_params == 'kmeans':
            resp = np.zeros((n_samples, self.n_components))
            label = KMeans(n_clusters=self.n_components, n_init=1,
                                random_state=random_state).fit(X).labels_
            resp[np.arange(n_samples), label] = 1
        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)

        self.m_step(X, resp)

    def init_model(self, X):
        mu_min = min(X)
        mu_max = max(X)
        sigma_min = 1
        sigma_max = 1

        g = []
        mix = []
        for i in range(self.n_components):
            g.append(Gaussian(uniform(mu_min, mu_max), uniform(sigma_min, sigma_max)))
            mix.append(1 / self.n_components)

        self.g = g
        self.mix = mix

        return self

    def e_step(self, X):
        """e step.

        Parameters
        ----------
        X : array-like, shape (n_samples,)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        assert X is not None and len(X) > 0, 'X is none or empty'
        assert self.g is not None and len(self.g) > 0, 'g is none or empty'
        assert self.mix is not None and len(self.mix) > 0, 'mix is none or empty'
        assert len(self.g) == len(self.mix), 'length of g and mix is not equal'

        log_prob_norm, log_resp = self.estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp

    def estimate_log_prob_resp(self, X):
        """Estimate log probabilities and responsibilities for each sample.

        Refer to function "_estimate_log_prob_resp()" in sklearn\mixture\base.py

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like, shape (n_samples,)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """

        # weighted_log_prob : array, shape (n_samples, n_components)
        # weighted_log_prob = self.estimate_weighted_log_prob(X)
        weighted_log_prob = self.estimate_weighted_log_prob_np(X)

        # log_prob_norm: array, shape (n_samples,)
        #       i.e., log p(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        assert len(log_prob_norm) == len(X), 'length of log_prob_norm error'

        with np.errstate(under='ignore'):
            # ignore underflow
            # log_resp : array, shape (n_samples, n_components)
            #       logarithm of the responsibilities
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]

        return log_prob_norm, log_resp

    def estimate_weighted_log_prob(self, X):
        # refer to function "_estimate_weighted_log_prob()" in sklearn\mixture\base.py
        # estimate the weighted log-probabilities, log P(X | Z) + log weights.
        # weighted_log_prob : array, shape (n_samples, n_components)
        weighted_log_prob = []
        for i in range(len(self.g)):
            a = [self.g[i].log_pdf(x) + np.log(self.mix[i]) for x in X]
            # here we assume X is 1D array
            assert len(a) == len(X), 'length of array a error'
            weighted_log_prob.append(a)

        weighted_log_prob = np.array(weighted_log_prob)
        return weighted_log_prob.T

    def estimate_weighted_log_prob_np(self, X):
        # refer to function "_estimate_weighted_log_prob()" in sklearn\mixture\base.py
        # estimate the weighted log-probabilities, log P(X | Z) + log weights.
        # weighted_log_prob : array, shape (n_samples, n_components)
        X = np.array(X)
        X = X.flatten()
        weighted_log_prob = []
        for i in range(len(self.g)):
            a = self.g[i].log_pdf_np(X)
            # here we assume X is 1D array
            assert len(a) == len(X), 'length of array a error'
            weighted_log_prob.append(a)

        weighted_log_prob = np.array(weighted_log_prob)
        return weighted_log_prob.T

    def m_step(self, X, resp):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples,)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """

        # nk : array-like, shape (n_components,)
        #   The numbers of data samples in the current components.
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps

        resp = resp.T

        # compute new means
        for i in range(len(self.g)):
            self.g[i].mu = np.dot(resp[i], np.array(X)) / nk[i]

        # compute new sigmas
        for i in range(len(self.g)):
            self.g[i].sigma = np.sqrt(np.dot(resp[i], (np.array(X) - self.g[i].mu) ** 2) / nk[i])

        # compute new mix
        for i in range(len(self.g)):
            self.mix[i] = nk[i] / len(X)

    def pdf(self, x):
        v = 0
        for i in range(len(self.g)):
            v += self.g[i].pdf(x) * self.mix[i]
        return v

    def fit(self, X, max_iter):
        "perform n iterations, then compute log-likelihood"
        self._initialize_parameters(X)

        lower_bound = None

        for i in range(max_iter):
            self.n_iter_ = i
            # print("iter: " + str(i))
            prev_lower_bound = lower_bound

            # if verbose:
            #     print('iteration: ' + str(i))

            log_prob_norm, log_resp = self.e_step(X)
            self.m_step(X, np.exp(log_resp))

            lower_bound = log_prob_norm

            if prev_lower_bound is not None:
                change = lower_bound - prev_lower_bound
                if abs(change) < self.tol:
                    break

    def score_samples(self, X):
        """Compute the weighted log probabilities for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log probabilities of each data point in X.
        """
        # weighted_log_prob : array, shape (n_samples, n_components)
        # weighted_log_prob = self.estimate_weighted_log_prob(X)
        weighted_log_prob = self.estimate_weighted_log_prob_np(X)

        # log_prob_norm: array, shape (n_samples,)
        #       i.e., log p(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        assert len(log_prob_norm) == len(X), 'length of log_prob_norm error'

        return log_prob_norm


# # Fix the data using Gaussian Mixture model
# 

# In[ ]:


class GMM:
    def __init__(self):
        self._gmm_list = None
        self._log_prior = None

    def fit(self, X, y, n_components, max_iter=100):
        self._log_prior = np.log(np.bincount(y) / len(y))

        # shape of log_pdf
        shape = (len(self._log_prior), X.shape[1])

        self._gmm_list = np.empty(shape, dtype=object)
        for i in range(shape[0]):
            for j in range(shape[1]):
                print('fit model ({0},{1})'.format(i, j))
                model = GaussianMixture(n_components)
                a = X[y == i, j:j + 1]
                model.init_model(a)
                model.fit(a, max_iter)
                self._gmm_list[i, j] = model
                print('n_iter_: {0}'.format(model.n_iter_))

    def predict_proba(self, X):
        assert self._gmm_list is not None, 'gmm list is none'
        assert self._log_prior is not None, 'log prior is none'

        # shape of log_likelihood before summing
        shape = (len(self._log_prior), X.shape[1], X.shape[0])

        ll = [[self._gmm_list[i][j].score_samples(X[:, j:j + 1])
                    for j in range(shape[1])]
                    for i in range(shape[0])]

        log_likelihood = np.sum(ll, axis=1).T

        log_joint = self._log_prior + log_likelihood

        predicts = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))
        return predicts


# # Splitting data into train and test set

# In[ ]:


columns = df.columns.tolist()

target = "Class"
columns = [c for c in columns if c not in ["Class", "Time", "Amount"]]

X = df[columns]
y = df[target]

print(X.columns)

row_count = X.shape[0]

X_train = X[:int(row_count*0.8)]
X_test = X[int(row_count*0.8):]

y_train = y[:int(row_count*0.8)]
y_test = y[int(row_count*0.8):]


# # Train the model

# In[ ]:


n_components = 10
g = GMM()
g.fit(np.array(X_train), np.array(y_train), n_components)


# # Predict

# In[ ]:


predicts = g.predict_proba(np.array(X_test))
auc = roc_auc_score(y_test, predicts[:, 1])

print(auc)

