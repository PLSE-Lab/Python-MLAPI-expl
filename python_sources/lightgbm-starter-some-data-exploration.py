#!/usr/bin/env python
# coding: utf-8

# Full disclosure: I am actually a particle physicist, although I've never worked much with collider experiments.
# 
# In this challenge, we are tasked with predicting whether or not a given reconstructed event in a particle physics experiment is from a 
# 
# $$ \tau \rightarrow \mu^+\mu^-\mu^- $$
# 
# decay.
# 
# Tau leptons are basically like very heavy electrons, with a mass of 1.78 GeV rather than 511 keV. These are still quite light by the standards of proton colliders. The top quark, which is the heaviest known elementary particle, has a mass about 100 times higher. The LHC was built so that it would be able to create TeV-scale particles such as supersymmetric partners to regular Standard Model particles.
# 
# Taus are unstable and decay with a lifetime of $3\times10^{-13}$ s, or 300 femtoseconds. Due to conservation of lepton number, taus are created in $\tau^+\tau^-$ pairs or in conjunction with neutrinos. In a facility like the LHC, this means that the taus will typically have a large amount of kinetic energy in the lab frame. The apparent lifetime in the lab frame is boosted from $\tau = 300$~fs to $\gamma\tau$, where $\gamma$ is a relativistic factor equal to the ratio between the total energy in the lab frame and the rest mass. So, if $\gamma=10$ (that is, the $\tau$ has a total energy 17.8 GeV), it will have a lifetime of 3 picoseconds. Since the tau will be going nearly the speed of light, it will travel on average around 1 mm from the initial vertex before decaying.
# 
# If you can reconstruct the separation between the tau decay and the vertex (at least in the transverse dimension), this will let you do things like compare the estimated $\gamma$ of the tau (based on muon kinematics) to the distance traveled. This will give you some ability to at least start to select tau-like events.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


folder = '../input/'
train = pd.read_csv(folder+'training.csv', index_col=0)
test = pd.read_csv(folder+'test.csv',index_col=0)
#check_correlation = pd.read_csv(folder+'check_correlation.csv', index_col='id')
#check_agreement = pd.read_csv(folder+'check_agreement.csv', index_col='id')


# In[ ]:


train.head()


# In[ ]:


train.info()


# We see that there are no null values, which is good. This is all truth or reconstructed information, so we do not expect any null values anywhere.
# 
# There are quite a few different features here. One of the main challenges of this sort of dataset is figuring out how to select or construct features that do not show noticeable differences between simulated data and real data.
# 
# These sorts of tests are very common in particle physics. Most high energy physics experiments rely heavily on detailed simulations of both the physics they are trying to measure and also the interactions of particles in their detectors. The models going into these simulations always have various uncertainties and systematic errors, so you  need to be very careful that you are not training on the idiosyncracies of your simulation but rather on features that are fairly insensitive to data/simulation differences.

# In[ ]:


train.signal.value_counts()


# The data aren't perfectly balanced, but this looks like we are not in the regime of finding rare events over a large background. Either there are large weights that need to be applied somewhere to renormalize things or we are seeing a set of events that are preselected to have a decent signal to background ratio.

# In[ ]:


signal = train[train.signal==1]
nonsignal = train[train.signal==0]


# Now let's plot a few of the variables.
# 
# The transverse momentum ($p_t$) of the 3-muon system would be the magnitude of the vector sum of the transverse momenta of the muons. This shows how much momentum is going away from axis defining the beam direction. This is important in many collider analyses because when protons collide at high energies, we are really seeing subatomic particles withing the protons (i.e. quarks and gluons). These particles, called partons, carry only a fraction of the total proton momentum. As a result, there can be a significant overall boost in the longitudinal direction. There is going to be little transverse momentum in the collision, so high $p_t$ particles somehow got their transverse momentum from the collision. If you could measure all the particles from the collision, the total $p_t$ would be very close to 0. In practice, there can be missing $p_t$ from things like neutrinos, fragments of the protons that are too close to the beam to measure, and even exotic theoretical particles. 

# In[ ]:


plt.hist(signal.pt/1000, range=(0,25), bins=100, label='Signal', alpha=0.7,normed=True)
plt.hist(nonsignal.pt/1000, range=(0,25), bins=100, label='Background',alpha=0.7,normed=True)
plt.xlabel('Transverse Momentum [GeV?]')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.show()


# I already explained what the flight distance represents. This is the distance between the initial vertex and the location where the proposed $\tau$ lepton decayed.

# In[ ]:


plt.hist(signal.FlightDistance, range=(0,100), bins=100, label='Signal', alpha=0.7,normed=True)
plt.hist(nonsignal.FlightDistance, range=(0,100), bins=100, label='Background',alpha=0.7,normed=True)
plt.xlabel('Flight Distance')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.show()


# Dira gives the angle between the collision vertex/decay vertex vector and the direction of the 3-muon system. In $\tau\rightarrow 3\mu$ decays, all of the $\tau$'s momentum goes into the muons, so the momentum vector does not change. Too large an angle and we know that there is something not quite right with our proposed event type.

# In[ ]:


plt.hist(signal.dira, range=(0.998,1), bins=100, label='Signal', alpha=0.7,normed=True)
plt.hist(nonsignal.dira,range=(0.999,1),bins=100, label='Background',alpha=0.7,normed=True)
plt.xlabel('Dira')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.show()


# The vertex $\chi^2$ is a measure of the goodness of fit of the $\tau$ vertex. I believe this means the goodness of fit of the three muons converging on the same location. This will typically be low for true $\tau\rightarrow3\mu$ events and high for events where the 3 muons are not really from the same decay.
# 
# Here we see a sudden cutoff in the histograms around $\chi^2=15$. I did not put that there. It's in the data, so this looks like one of the variables used in the preselection process used to collect the data.
# 
# It's quite likely that this is not going to be a good feature to use in this challenge.

# In[ ]:


plt.hist(signal.VertexChi2, range=(0,16), bins=100, label='Signal', alpha=0.7,normed=True)
plt.hist(nonsignal.VertexChi2,range=(0,16),bins=100, label='Background',alpha=0.7,normed=True)
plt.xlabel(r'Vertex $\chi^2$')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.show()


# I'm not 100% sure of what the impact parameter is but my guess is that this is the estimated closest approach of the tau candidate to the vertex position. That is, if we take the momentum vector of the three muons as the tau momentum and then propagate the tau backward in time, this is the closest it gets to the vertex.

# In[ ]:


plt.hist(signal.IP, range=(0,0.5), bins=100, label='Signal', alpha=0.7,normed=True)
plt.hist(nonsignal.IP,range=(0,0.5),bins=100, label='Background',alpha=0.7,normed=True)
plt.xlabel('Impact Paraneter')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.show()


# There are a lot of track isolation variables. This likely show how far the muon tracks are from other tracks. Basically, these tell you how clean the muon tracks are. This is important because if we want precision measurements of where things are near the vertex, we don't want our tracks to be contaminated from stray hits from other particles. If the tracks aren't sufficiently isolated, you might assign a hit to the wrong track and throw off your whole reconstruction model.

# In[ ]:


plt.hist(signal.iso, range=(0,20), bins=20, label='Signal', alpha=0.7,normed=True)
plt.hist(nonsignal.iso,range=(0,20),bins=20, label='Background',alpha=0.7,normed=True)
plt.xlabel('Track Isolation Variable')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.show()


# Finally, the lifetime should show the estimated lifetime of each particle. We expect this took exponential (not including things like efficiency and resolution). This will be much too short to directly measure, so it is likely extrapolated by taking the flight distance, the 3-muon momentum, and the known tau mass to get a lifetime. This is calculated as:
# 
# $$\tau = \frac{\ell}{\sqrt{1+p^2/m_\tau^2}}$$
# 
# where $\ell$ is the flight distance, $p$ is the magnitude of the total momentum and $m_\tau$  is the tau mass.

# In[ ]:


plt.hist(signal.LifeTime, range=(0,1e-2), bins=50, label='Signal', alpha=0.7,normed=True)
plt.hist(nonsignal.LifeTime,range=(0,1e-2),bins=50, label='Background',alpha=0.7,normed=True)
plt.xlabel('Lifetime')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.show()


# To start out, we know from the examples that hte flight distance, lifetime, and transverse momentum should be safe variables to use that should typically pass the various tests. I've also checked that the impact parameter works as well. I'll train a simple untuned LightGBM model model.

# In[ ]:


import lightgbm as lgb
feature_names = ['FlightDistance', 'LifeTime', 'pt', 'IP']
features = train[feature_names]
target = train['signal']
train_set = lgb.Dataset(features,train.signal)
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'num_leaves': 31,
    'metric': {'auc'},
    'learning_rate': 0.01,
#    'feature_fraction': 0.8,
#    'bagging_fraction': 0.8     
    
}
cv_output = lgb.cv(
    params,
    train_set,
    num_boost_round=400,
    nfold=10,
)


# In[ ]:


best_niter = np.argmax(cv_output['auc-mean'])
best_score = cv_output['auc-mean'][best_niter]
print('Best number of iterations: {}'.format(best_niter))
print('Best CV score: {}'.format(best_score))
model = lgb.train(params, train_set, num_boost_round=best_niter)


# In[ ]:


test_features = test[feature_names]


# In[ ]:


predictions = model.predict(test_features)
test['prediction'] = predictions
test[['prediction']].to_csv('lightgbm_starter.csv')


# In[ ]:




