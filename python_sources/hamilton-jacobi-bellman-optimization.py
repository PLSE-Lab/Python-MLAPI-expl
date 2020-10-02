#!/usr/bin/env python
# coding: utf-8

# **Help requested**
# 
# This problem might be solved using optimal control mathematics. We want to control a detector of the system's output states (how many open channels) so that the f1 score of the detector is maximized. 
# 
# The Hamilton Jacobi Bellman equation offers a solution. The solution of the minumized partial derivative of the Hamiltonian is the solution for this problem. It should provide an estimate of "open_channels" at the highest possible f1 score. Actually, to frame the problem correctly for the HJB, we will seek to minumize (1-f1_score).
# 
# ![](https://www.elmtreegarden.com/wp-content/uploads/2020/04/ion-channel-receiver.gif)
# 
# **The HJB equation is composed with the following functions**:<br>
# **A**. The state equations relating input x and output y including history (x',y') in t = {0,tf). Train data.<br>
# **B**. Performance function: minumize(1- f1 score) modified to be differentiable<br>
# **C**. Constrains on the system: conservation of energy.<br>
# **D**. Initial conditions
# 
# **A.** y(t) is the output of the receiver. y(t) is an integer in the range: 0 <= y <= 10  with 11 possible states.<br>
# Based on the known physics of the problem and previous experiments with features, we can make a second order differencial equation as a model for yt). Since we will improve this with control I will shoot for a F1_score of about 0.90 or better before optimizing a control for the system. Let our model be:
# 
#                     y(t) = a0 + R f(t)^2 - R f(t-1)^2 + y(t-1)
#                                 
#                     y'(t) = R/2 f(t) 
#                     
# 
# **B.** The performance function, **J**(t) , is the f_1 score, f_1(y, y'). In our case we will 'soften' the function, using a function provided by reference 3. The natural form of the HJB is to minumize performance, so we will use the negative of the softened f1 score in the equation. <br>So **J**, the performance measure, is <br>
# 
#                     **J**(y,t) = 1 - soft_f1_score(y(t),y'(t))
# 
# **C.** Constrains on the system: conservation of energy. The Energy in the signal during transition is {x(t)^2 - x(t-1)^2} dt, where dt is time between samples (0.0001 s). The Energy in the transition is k dy(t)/dt, where k is a constant.. where k is the energy of each ion transition. "*Onsager phenomenological coefficients* ". <br>
# 
#                     ky'(t) - R f(t)^2 = 0
#                     y'(t) = R/k f(t)^2
# 
# **D.** Initial conditions will be assumed are<br>
# 
#                     y(0) = y'(0) = 0
#                     **J**(y(0),t=0) = 0
# 
# The HJB equation is:
# 
# 0 = **J**(y,t) + **H**{x, u"{x, **J**(y,t), t}, **J**(y,t), t}<br>
# 
# The **u"** that solves this partial diffencial equation (PDE) is the optimal control for the History. Now take the partial derivative with respect to x of the HJB equation.
# 
# where **H** is the Hamiltonion
# 
# 
# References:
# 1. Kirk, "Optimal Control Theory", 1970 Chapters 1-5<br>
# 2. [E.T. Jaynes: Minimum Entropy Principle](https://pdfs.semanticscholar.org/b326/6b25cb2ff34634aff48434652bacb3fede9c.pdf)
# 3. [Toward Data Science Blog](https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d)

# In[ ]:


import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
import pandas as pd
import seaborn as sns
import os
import scipy as sp
import scipy.fftpack
from scipy import signal
from scipy.signal import butter, sosfiltfilt, freqz, filtfilt
from sklearn import tree
import lightgbm as lgb
import xgboost as xgb
import gc
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer

DATA_PATH = "../input/liverpool-ion-switching"

train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
#test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
#submission_df = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))


# In[ ]:


class Equation(object):
    """Base class for defining PDE related function."""

    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError


class HJBLQ(Equation):
    """HJB equation in PNAS paper doi.org/10.1073/pnas.1718942115"""
    def __init__(self, eqn_config):
        super(HJBLQ, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.0)
        self.lambd = 1.0

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return -self.lambd * tf.reduce_sum(tf.square(z), 1, keepdims=True)

    def g_tf(self, t, x):
        return tf.math.log((1 + tf.reduce_sum(tf.square(x), 1, keepdims=True)) / 2)


# In[ ]:


from itertools import islice

def window(seq, n=2):
    "Sliding window width n from seq.  From old itertools recipes."""
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
        
pairs = pd.DataFrame(window(train_df.loc[:,'open_channels']), columns=['state1', 'state2'])
counts = pairs.groupby('state1')['state2'].value_counts()
alpha = 1 # Laplacian smoothing is when alpha=1
counts = counts.fillna(0)
probs = ((counts + alpha )/(counts.sum()+alpha)).unstack()

probs


# In[ ]:


# reference https://www.kaggle.com/friedchips/on-markov-chains-and-the-competition-data
def create_axes_grid(numplots_x, numplots_y, plotsize_x=6, plotsize_y=3):
    fig, axes = plt.subplots(numplots_y, numplots_x)
    fig.set_size_inches(plotsize_x * numplots_x, plotsize_y * numplots_y)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    return fig, axes


# In[ ]:


"""
train_df['rel_work'] = train_df['signal']**2 - (train_df['signal']**2).mean()
pairs = pd.DataFrame(window(train_df.loc[:,'rel_work']), columns=['state1', 'state2'])
means = pairs.groupby('state1','state2')['rel_work'].mean()
alpha = 1 # Laplacian smoothing is when alpha=1
means = means.unstack()
means
"""


# In[ ]:


train_df.loc[:,'oc'] = train_df['open_channels'].shift(1)
train_df.loc[:,'rel_work'] = train_df['signal']**2 - (train_df['signal']**2).mean()
means = train_df.groupby(['open_channels','oc'])['rel_work'].mean()
means = means.unstack()
means
fig, axes = create_axes_grid(1,1,10,5)
axes.set_title('Markov Transition Matrix P for all of train')
sns.heatmap(
    means,
    annot=True, fmt='.3f', cmap='Blues', cbar=False,
    ax=axes, vmin=0, vmax=0.5, linewidths=2);


# In[ ]:


train_df


# In[ ]:


print('Occurence Table of State Transitions')
ot = counts.unstack().fillna(0)
ot
fig, axes = create_axes_grid(1,1,10,5)
axes.set_title('Occurence Table of State Transitions')
sns.heatmap(
    ot,
    annot=True, fmt='.0f', cmap='Blues', cbar=False,
    ax=axes, vmin=0, vmax=0.5, linewidths=2);


# In[ ]:


P = (ot)/(ot.sum())
Cal = - P * np.log(P)
Cal
fig, axes = create_axes_grid(1,1,10,5)
axes.set_title('Markov Transition Matrix P for all of train')
sns.heatmap(
    Cal,
    annot=True, fmt='.3f', cmap='Blues', cbar=False,
    ax=axes, vmin=0, vmax=0.5, linewidths=2);


# In[ ]:


Caliber = Cal.sum().sum()
Caliber


# In[ ]:


#reference https://www.kaggle.com/teejmahal20/a-signal-processing-approach-low-pass-filtering
# some of TJ Klein's code, who helped me make this notebook with his examples
def butter_lowpass_filter(data, fc, fs, order=5):
    normal_cutoff = fc / (fs/2)
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    sig = pd.DataFrame(y, columns = ['sig_filt'])
    return sig


# In[ ]:


bs = 500000
fs = 10000
sfft = []
sfft_10 = []
sfft_20 = []
output = []
ws = 11
window = signal.blackmanharris(ws)
window_10 = signal.blackmanharris(ws*5)
window_small = signal.blackmanharris(5)
bs = int(bs)

for ii in range(10):  # perform band pass filter
    i = ii*bs
    fourier = rfft(train_df.iloc[i:i+bs,1])
    # filter the power spectrum
    fourierb = fourier
    # Apply Blackman-Harris high pass filter. Use first half of window
    for i in range(5):
        fourier[i] = fourier[i]*(window[i])
        fourier[i+int(bs/4)] = fourier[i+int(bs/4)]*(window[i])
    
    sf = irfft(fourier)
    sfft = np.append(sfft,sf)
    
    # Apply Blackman-Harris notch filter to reduce 50Hz buzz noise
    #
    fourier[2500-28:2500+27] = fourier[2500-28:2500+27]*(1-window_10)
    fourier[2500-28+int(bs/4):2500+27+int(bs/4)] = fourier[2500-28+int(bs/4):2500+27+int(bs/4)]*(1-window_10)
     
    sf_10 = irfft(fourier)
    sfft_10 = np.append(sfft_10,sf_10)
    
    # Apply Blackman-Harris notch filter to cut out sawtooth noise with period of 10 second 
    # the index of 0.1Hz is 5 and cut out odd harmonics of 0.1Hz
    for harm in range(1,11,2):# Use small Blackman window inverted for notching out harmonics of 0.1Hz
        fourier[harm*5+i-2:harm*5+i+3] = fourier[harm*5+i-2:harm*5+i+3]*(1-window_small[i])
        fourier[harm*5+i-2+int(bs/4):harm*5+i+3+int(bs/4)] = fourier[harm*5+i-2+int(bs/4):harm*5+i+3+int(bs/4)]*(1-window_small[i])
     
    sf_20 = irfft(fourier)
    sfft_20 = np.append(sfft_20,sf_20)
    
    
train_df['signal_f'] = 0.
train_df['signal_f'] = sfft
train_df['signal_f10'] = 0.
train_df['signal_f10'] = sfft_10
train_df['signal_f20'] = 0.
train_df['signal_f20'] = sfft_20


# Apply a low pass filter with 600Hz cutoff frequency
fc = 600  # Cut-off frequency of the filter
train_df['signal_f30'] = 0.
output = butter_lowpass_filter(train_df.iloc[:,4], fc, fs, 5)
train_df['signal_f30'] = output['sig_filt']


# In[ ]:


train_df.loc[:,'oc'] = train_df['open_channels'].shift(1)
train_df.loc[:,'rel_work_30'] = train_df['signal_f30']**2 - (train_df['signal_f30']**2).shift(-1)
means = train_df.groupby(['open_channels','oc'])['rel_work_30'].mean()
means = means.unstack()
means
fig, axes = create_axes_grid(1,1,10,5)
axes.set_title('Markov Transition Matrix P for all of train')
sns.heatmap(
    means,
    annot=True, fmt='.3f', cmap='Reds', cbar=False,
    ax=axes, vmin=0, vmax=0.5, linewidths=2);


# In[ ]:



fig, axes = create_axes_grid(1,1,10,5)
axes.set_title('Markov Transition Matrix P for all of train')
sns.heatmap(
    P,
    annot=True, fmt='.3f', cmap='Blues', cbar=False,
    ax=axes, vmin=0, vmax=0.5, linewidths=2);


# In[ ]:


eig_values, eig_vectors = np.linalg.eig(np.transpose(P))
print("Eigenvalues :", eig_values)


# Constrains on our system will define it's behavior. The system is constrained by the law of conservation of Energy. Input is signal x and output is open channels y.
# 
# Our signal x is current in pico-amperes (pA or 10^-12 Amps). The current squared x(t)^2 is proportional to instintainious Power is R * x(t)^2, where R is the resistance of the circuit. Transition Energy = TE = R*{x(t)^2-x(t-1)^2}* dt
# 
# Assuming that each Ion that transitions through the channel requires energy to make this transition, eT. The energy to make n state transitions is n * eT, where -10 <= n <= 10  so our constraint is:
# 
# TE - n * eT = 0   or
# 
# R*{x(t)^2-x(t-1)^2} * dt - n * eT = 0
# 
# Lagrangian analysis seeks to find minimums and maxima for prediction purposes. First find the Lagrangian L such that:
# 
# f(x,y) = L g(x,y)
# 

# In[ ]:


# reference: http://kitchingroup.cheme.cmu.edu/blog/2013/02/03/Using-Lagrange-multipliers-in-optimization/
def func(X):
    x = X[0]
    y = X[1]
    L = X[2] 
    return x + y + L * (x**2 + k * y)

def dfunc(X):
    dL = np.zeros(len(X))
    d = 1e-3 
    for i in range(len(X)):
        dX = np.zeros(len(X))
        dX[i] = d
        dL[i] = (func(X+dX)-func(X-dX))/(2*d);
    return dL


# In[ ]:


from scipy.optimize import fsolve

# this is the max
X1 = fsolve(dfunc, [1, 1, 0])
print(X1, func(X1))

# this is the min
X2 = fsolve(dfunc, [-1, -1, 0])
print(X2, func(X2))


# In[ ]:


# Reference https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
def macro_double_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)
    soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost) # average on all labels
    return macro_cost

