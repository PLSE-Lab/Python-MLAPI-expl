#!/usr/bin/env python
# coding: utf-8

# In[6]:


from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


df = pd.read_csv("../input/samp_subm.csv")
df.head()


# # 1. Feature Engineering
# 
# - Generate some useful features for the sequences to distinguish random ones from the others
# 
# The description of the features below is given here: https://pdfs.semanticscholar.org/f534/658b60ce28cf58e9667f7cef6bdcf83e011b.pdf

# In[8]:


def repetitions(x):
    prev = None
    reps = 0
    for d in x:
        if d == prev and prev is not None:
            reps += 1
        prev = d
    return reps


def runs(x):
    num_ascending = 1
    nums = []
    prev = x[0]
    for d in x[1:]:
        if d > prev:
            num_ascending += 1
        else:
            nums.append(num_ascending)
            num_ascending = 1
    nums.append(num_ascending)
    return np.array(nums).var()


def poker(x):
    total = 0
    for d in map(str, range(10)):
        sum = 0
        for i in range(0, 100, 5):
            if x[i: i + 5].count(d) == 2:
                 sum += 1
        total += sum
    return total


def cluster_ratio(x):
    m = np.zeros((10, 10), dtype=np.float32)
    for d in map(str, range(10)):
        for d2 in map(str, range(10)):
            m[int(d), int(d2)] = x.count(d + d2)
            if d == d2:
                m[int(d), int(d2)] /= 2
    return m.std()


def coupon(x):
    s = set()
    for i, d in enumerate(x):
        s.add(d)
        if len(s) == 10:
            return i
    return len(x)


def series(x):
    sum = 0
    for d in range(9):
        digram = str(d) + str(d + 1)
        sum += x.count(digram)
    return sum


def gap(x):
    gaps = []
    for d in map(str, range(10)):
        ind = []
        for i, g in enumerate(x):
            if d == g:
                ind.append(i)
        for i in range(1, len(ind)):
            gaps.append(ind[i] - ind[i - 1])
        if len(ind):
            gaps.append(ind[0] - 0)
            gaps.append(99 - ind[-1])
    return np.median(gaps)

def digram_repetitions(x):
    s = set()
    sum = 0
    for i in range(len(x) - 1):
        digram = x[i: i + 2]
        if digram not in s:
            sum += x.count(digram) - 1
        s.add(digram)
    return sum


def variance(x):
    cnts = Counter(x)
    for d in map(str, range(10)):
        if d not in cnts:
            cnts[d] = 0
    return np.var(list(cnts.values()))


def runs_reversed(x):
    return runs(x[::-1])


def coupon_reversed(x):
    return coupon(x[::-1])


def series_reversed(x):
    return series(x[::-1])


funcs = [repetitions, runs, poker, cluster_ratio, coupon, series, gap, digram_repetitions, variance,
         runs_reversed, coupon_reversed, series_reversed]


# In[9]:


for f in funcs:
    df[f.__name__] = df.Id.apply(f)


# In[10]:


df.iloc[:, 2:].head()


# ## Visualization

# Standardize features

# In[11]:


from sklearn.preprocessing import StandardScaler

m = StandardScaler().fit_transform(df.iloc[:, 2:])


# ### PCA

# In[12]:


from sklearn.decomposition import PCA

pca = PCA(2, svd_solver='full').fit_transform(m)
plt.scatter(pca[:,0], pca[:,1]);


# ### KMeans Clustering 
# 
# (indicated by color on the graph of principal components)

# In[13]:


from sklearn.cluster import KMeans

labels = KMeans(3, random_state=42).fit_predict(m)
plt.scatter(pca[:,0], pca[:,1], c=labels);


# Clusters' sizes

# In[14]:


pd.Series(labels).value_counts()


# **Note** that we already can use clustering output for prediction at this step (label some clusters as non-random) 

# # 2. Density Estimation
# 
# - Generate samples of the negative class (random sequences)
# - Estimate densities of the features given these samples

# In[15]:


np.random.seed(42)
n_samples = 1000
seq_len = 100
gen = np.random.randint(0, 10, size=(n_samples, seq_len))
gen_seq = pd.Series(["".join(map(str, x)) for x in gen])
gen_seq.head()


# KDE

# In[16]:


from sklearn.neighbors import KernelDensity

kdes = {}
for f in funcs:
    kde = KernelDensity().fit(gen_seq.apply(f).values[:, None])
    kdes[f] = kde


# # 3. Likelihood
# 
# - Compute likelihood (its logarithm) of the initial examples individually assuming they are randomly generated
# - Naive assumption of features independence is made

# More specifically, compute log-likelihood of a sequence by summing up log-probability for every its feature by means of the above estimated densities

# In[17]:


score = np.zeros(df.shape[0], dtype=np.float32)  # log-likelihood
for f in funcs:
    score += kdes[f].score_samples(df[f.__name__].values[:, None])


# ## Visualization

# In[18]:


plt.hist(score[score > -100], bins=20);


# In the PCA graph

# In[19]:


plt.scatter(pca[:,0], pca[:,1], c=score);


# Remove 3 outliers

# In[20]:


pca_ = pca[df.index.isin(score.argsort()[3:])]
score_ = score[score > score[score.argsort()[2]]]
plt.scatter(pca_[:,0], pca_[:,1], c=score_);


# Check if our score (and potentially real label) correlates with position in dataset

# In[21]:


plt.plot(score[score > score[score.argsort()[2]]]);


# # Submission
# 
# The final score is negative log-likelihood as a measure of non-randomness

# In[22]:


df_final = df[["Id", "Label"]]
df_final.loc[:, "Label"] = -score
df_final.to_csv("submission.csv", index=False)

