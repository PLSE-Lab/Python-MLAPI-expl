#!/usr/bin/env python
# coding: utf-8

# This competition was a fun one and I learned a lot from it mostly because it had a relatively small dataset that let me design various experiments and run them in a short time.
# 
# Here I am going to share how I achieved public LB score 0.97482; at this point I assume all of the competitiors who broke the 0.973 barrier found the same thing that I will describe here. If you still don't know how the dataset is generated first read my other kernel at https://www.kaggle.com/mhviraf/synthetic-data-for-next-instant-gratification
# 
# Moreover, you can find my complete code at https://www.kaggle.com/mhviraf/mhviraf-s-best-submission-in-instant-gratification
# 
# After the abovementioned kernel and Vlad's public kernel on QDA were published, everyone assumed that after removing useless columns in each `wheezy-copper-turtle-magic` the dataset must look like somehow to the following graph (of course in higher dimensions) because QDA gave us more accurate results than SVD, Logistic regression, LGBM, etc. etc. 

# In[ ]:


from sklearn.datasets import make_classification 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
np.random.seed(2019)

# generate dataset 
X, y = make_classification(1024, 2, n_informative=2, n_clusters_per_class=1, n_redundant=0, flip_y=0.05, random_state=125)


plt.scatter(X[y==0, 0], X[y==0, 1], label='target=0')
plt.scatter(X[y==1, 0], X[y==1, 1], label='target=1')
plt.legend()


# This is actually very close to what the competition dataset is. HOWEVER, there is one important and different thing. In `make_classification`, the default value of `n_clusters_per_class` is 2 not 1! and I think it is set to 3 in Instant Gratification competition dataset but for the sake of easier illustration i'm gonna stick with 2 here. So the above graph would turn into the following graph if `n_clusters_per_class=2`:

# In[ ]:


from sklearn.datasets import make_classification 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.random.seed(2019)

# generate dataset 
X, y = make_classification(1024, 2, n_informative=2, n_clusters_per_class=2, n_redundant=0, flip_y=0.05, random_state=5)


plt.scatter(X[y==0, 0], X[y==0, 1], label='target=0')
plt.scatter(X[y==1, 0], X[y==1, 1], label='target=1')
plt.legend()


# QDA would still perform ok on this dataset, particularly if you use the exact same parameters but a different `random_state`, the dataset might be much easier for QDA to model. refer to the figure below. Still 2 clusters per class but you can imagine fiting 1 gaussian distribution to each class too.

# In[ ]:


from sklearn.datasets import make_classification 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# generate dataset 
X, y = make_classification(1024, 2, n_informative=2, n_clusters_per_class=2, n_redundant=0, flip_y=0.05, random_state=0)


plt.scatter(X[y==0, 0], X[y==0, 1], label='target=0')
plt.scatter(X[y==1, 0], X[y==1, 1], label='target=1')
plt.legend()


# Now if we fit a guassian model to the former dataset with 2 clusters per class, whether it's a `QDA` or `mixture.GaussianMixture` we will get the model shown below. Still doing ok, accurate enough but not the best possible model.

# In[ ]:


from sklearn import mixture
from matplotlib.colors import LogNorm
from sklearn.covariance import OAS

def get_mean_cov(X):
    model = OAS(assume_centered=False)
    
    ms = []
    ps = []
    for xi in X:
        model.fit(xi)
        ms.append(model.location_)
        ps.append(model.precision_)
    return np.array(ms), np.array(ps)

knn_clf = mixture.GaussianMixture(n_components=2, init_params='random',
                          covariance_type='full',
                          n_init=1, 
                          random_state=0)


X, y = make_classification(1024, 2, n_informative=2, n_clusters_per_class=2, n_redundant=0, flip_y=0.05, random_state=5)

x_t =X.copy()
y_t =y.copy()
train3_pos = X[y==1]
train3_neg = X[y==0]

print(train3_pos.shape, train3_neg.shape)
ms, ps = get_mean_cov([train3_pos, train3_neg])

clf = mixture.GaussianMixture(n_components=2, covariance_type='full', means_init=ms, precisions_init=ps,)
clf.fit(X)

x = np.linspace(-5., 5.)
y = np.linspace(-5., 5.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

plt.figure()
plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),  levels=np.logspace(0, 3, 10))
plt.scatter(x_t[y_t==0, 0], x_t[y_t==0, 1], label='target=0', alpha=.3)
plt.scatter(x_t[y_t==1, 0], x_t[y_t==1, 1], label='target=1', alpha=.3)
plt.legend()


# ### The key!
# If we rather cluster each class into 2 or 3 clusters before fitting the model and then set `n_components=4 or 6` in `GaussianMixture` we will achieve an even more accurate model such as the one shown below.

# In[ ]:


from sklearn import mixture
from matplotlib.colors import LogNorm
from sklearn.covariance import OAS
X, y = make_classification(1024, 2, n_informative=2, n_clusters_per_class=2, n_redundant=0, flip_y=0.05, random_state=5)

x_t =X.copy()
y_t =y.copy()
train3_pos = X[y==1]
train3_neg = X[y==0]

cluster_num_pos = knn_clf.fit_predict(train3_pos)
train3_pos_1 = train3_pos[cluster_num_pos==0]
train3_pos_2 = train3_pos[cluster_num_pos==1]
#print(train3_pos.shape, train3_pos_1.shape, train3_pos_2.shape, train3_pos_3.shape)

## FIND CLUSTERS IN CHUNKS WITH TARGET = 0
cluster_num_neg = knn_clf.fit_predict(train3_neg)
train3_neg_1 = train3_neg[cluster_num_neg==0]
train3_neg_2 = train3_neg[cluster_num_neg==1]
        
    
print(train3_pos.shape, train3_neg.shape)
ms, ps = get_mean_cov([train3_pos_1, train3_pos_2, train3_neg_1, train3_neg_2])

clf = mixture.GaussianMixture(n_components=4, covariance_type='full', means_init=ms, precisions_init=ps,)
clf.fit(X)

x = np.linspace(-5., 5.)
y = np.linspace(-5., 5.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

plt.figure()
plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),  levels=np.logspace(0, 3, 10))
plt.scatter(x_t[y_t==0, 0], x_t[y_t==0, 1], label='target=0', alpha=.3)
plt.scatter(x_t[y_t==1, 0], x_t[y_t==1, 1], label='target=1', alpha=.3)
plt.legend()


#  Hopefully other competitiors will share their solutions soon but I think everyone with LB score > 0.974 has been using the same technique that is described here.
