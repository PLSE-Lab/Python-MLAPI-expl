#!/usr/bin/env python
# coding: utf-8

# Last year I took an amazing roller-coaster course on Modern Multivariate Statistics, which covered the longest laundry list of methods in Multivariate Statistics from Reduced Rank Regression and Biclustering to Structural Equation Modelling and Gaussian Mixture Models. The course followed Izenman's book on Modern multivariate statistical techniques and was amazing exposure for me and my peers. As our convenor works closely with biomedical researchers in the field of biostatistics, we spent valuable time exploring methods in the field of biostatistics and spent some time exploring one of Hastie, Trevor, et al.'s methods in Gene Clustering, called Principle Gene Shaving, which I thought I would explore. 
# 
# Sadly, Python is very poorly equipped for this method, as I was required to write a lot of it from scratch.  For R and S, there are [complete implementation](https://bioinformatics.mdanderson.org/public-software/geneclust/) which I would recommend for any serious research.  
# 
# 
# [1] Izenman, Alan Julian. "Modern multivariate statistical techniques." Regression, classification and manifold learning 10 (2008): 978-0.
# [2] Hastie, Trevor, et al. "'Gene shaving'as a method for identifying distinct sets of genes with similar expression patterns." Genome biology 1.2 (2000): research0003-1.

# # Data
# The data which I am using is from the [NCBI database](https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/virus?SeqType_s=Nucleotide&VirusLineage_ss=Severe%20acute%20respiratory%20syndrome%20coronavirus%202,%20taxid:2697049) which was accessed on the 7th of april and represents a collection of genes which researchers have sequenced of the SARS-COV-2 virus. 

# # Alignment
# I have very little exposure working with Genetic Data and so had to reach out to some friends here and there to answer some very basic questions. A major challenge I faced in this analysis was how to deal with the issue of alignment.  When sequencing gene, mistakes can arise which cause sequences to have gaps which lead to misalignment. For large sequences and large datasets of gene this is alignment process can be extremely computationally intensive and after struggling with multiple tools in ended up settling on Lassmann's Kalign 3 method, as it appeared the fastest and simplest method I could use on CPU hardware.  
# 
# I tried a number of different appoaches and software tools, but Kalign 3 seems to be the easiest to use with the time I had.  I if people have had experience in gene alignment and have some spare compute I would really appreciate feedback on this process and the do's and don't, as this notebook- for now- mainly focuses on the methodology and less the analysis. 
# 
# [1] Lassmann, Timo. Kalign 3: multiple sequence alignment of large data sets. Bioinformatics (2019)
# 
# I have an example of how to install and use the `kalign3` software locally below, this may not support avx2 acceleration- for that you will have to brew install or install from source:

# In[ ]:


# ! conda install -c bioconda kalign3
# ! kalign  -i sequences.fasta -o kalign_fast.fasta 


# In[ ]:


get_ipython().system(' conda install -y scikit-bio')


# In[ ]:


import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import holoviews as hv
from skbio import DNA
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

hv.extension('bokeh')


# In[ ]:


file = '/kaggle/input/2019ncov-sequences-wuhan-coronavirus/kalign.fasta'


# In[ ]:


class FastaSeq:
    def __init__(self, name, sequence):
        self.name = name
        self.sequence = DNA(sequence)

    def get_seqs(file):
        items = []
        index = 0
        start = False
        for line in file:
            if line.startswith(">"):
                start = True
                if index >= 1:
                    items.append(aninstance)
                index+=1
                name = line[:-1]
                seq = ''
                aninstance = FastaSeq(name, seq)
            else:
                if start:
                    seq += line[:-1]
                    aninstance = FastaSeq(name, seq)
        if start:
            items.append(aninstance)

        return items


# In[ ]:


with open(file, "r") as f:
    data = FastaSeq.get_seqs(file=f.readlines())


# After alignment I had a number of sequences with a very large portion of gaps which I opted to filter from the dataset. 

# In[ ]:


str(data[3].sequence)[:1000] + ' ...'


# In[ ]:


replace = {'A':'A',
             'C':'C',
             'G':'G',
             'T':'U',
             'R':'AG',
             'Y':'CU',
             'S':'GC',
             'W':'AU',
             'K':'GU',
             'M':'AC',
             'B':'CGU',
             'D':'AGU',
             'H':'ACU',
             'V':'ACG',
             'N':'ACGU',
             '.':'S',
            '-':'S'}


# # Preprocessing
# IUPAC has a [guide](https://www.bioinformatics.org/sms/iupac.html) on how sequenced genes should be represented. In order to best capture this structure I opted to produce a One-hot encoding of these nucleotide code allowing for the symbols which represent ambiguity in the sequencing. 
# 
# 
# | IUPAC nucleotide code	| Base |
# |---|------------|
# | A | 	Adenine  |  
# | C | 	Cytosine  |  
# | G | 	Guanine  |  
# | T |  (or U)	Thymine (or Uracil)  |  
# | R | 	A or G  |  
# | Y | 	C or T  |  
# | S | 	G or C  |  
# | W | 	A or T  |  
# | K | 	G or T  |  
# | M | 	A or C  |  
# | B | 	C or G or T  |  
# | D | 	A or G or T  |  
# | H | 	A or C or T  |  
# | V | 	A or C or G  |  
# | N | 	any base  |  
# | . |  or -	gap  |  

# Before filtering:

# In[ ]:


df = pd.DataFrame(list(map(lambda x: pd.Series(list(str(x.sequence))), data)))
df.shape


# After filtering:

# In[ ]:


gaps = (df == '-').mean(1)
gap_threshold = gaps < 0.25

sequences = (df
             .loc[gap_threshold, :]
             .replace(replace)
             .applymap(str))
sequences.shape


# In[ ]:


from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
encoder = CountVectorizer(analyzer='char', vocabulary=['S', 'C', 'A', 'G', 'U'], strip_accents=None, lowercase=False)
X = hstack([encoder.fit_transform(sequences.loc[:, i]) for i in sequences])


# In[ ]:


X = np.ascontiguousarray(X
                         .toarray()
                         .astype('float32'))


# # Dimensionality Reduction
# I opted to apply a number of methods in dimensionality reduction in order to best visualise the highly sparse data. To set my perplexity, I looked to the [original author's guide](https://lvdmaaten.github.io/tsne/) which recommends low perplexities for small, sparse datasets.  

# In[ ]:


# svd
svd = TruncatedSVD(2, algorithm='arpack')
Z = svd.fit_transform(X)
svd_components = [f'Component {i} ({round(e*100)}%)' for i, e in enumerate(svd.explained_variance_ratio_.tolist())]

# kpca
kpca = KernelPCA(2, kernel='cosine', n_jobs=-1)
T = kpca.fit_transform(X)
kpca_components = [f'Component {i}' for i in range(2)]

# TSNE
tsne = TSNE(2, perplexity=18, metric='hamming')
U = tsne.fit_transform(X)
tsne_high_components = [f'Component {i}' for i in range(2)]

tsne = TSNE(2, perplexity=5, metric='hamming')
W = tsne.fit_transform(X)
tsne_low_components = [f'Component {i}' for i in range(2)]


tsne = TSNE(2, perplexity=18, metric='cosine')
V = tsne.fit_transform(X)
tsne_cosine_high_components = [f'Component {i}' for i in range(2)]

tsne = TSNE(2, perplexity=5, metric='cosine')
S = tsne.fit_transform(X)
tsne_cosine_low_components = [f'Component {i}' for i in range(2)]

# plots
fig, axes = plt.subplots(3, 2, figsize=(10, 10))
svd_plot = pd.DataFrame((Z - Z.mean(0)) / Z.std(0), columns=svd_components).plot.scatter(x=svd_components[0], y=svd_components[1], title='SVD', ax=axes[0][0])
cosine_kpca = pd.DataFrame(T, columns=kpca_components).plot.scatter(x=kpca_components[0], y=kpca_components[1], title='Cosine KPCA', ax=axes[0][1])
tsne_high = pd.DataFrame(U, columns=tsne_high_components).plot.scatter(x=tsne_high_components[0], y=tsne_high_components[1], title='Hamming TSNE: Perplexity 18', ax=axes[1][0])
tsne_low = pd.DataFrame(W, columns=tsne_low_components).plot.scatter(x=tsne_low_components[0], y=tsne_low_components[1], title='Hamming TSNE: Perplexity 5', ax=axes[1][1])
tsne_high = pd.DataFrame(V, columns=tsne_cosine_high_components).plot.scatter(x=tsne_cosine_high_components[0], y=tsne_cosine_high_components[1], title='Cosine TSNE: Perplexity 18', ax=axes[2][0])
tsne_low = pd.DataFrame(S, columns=tsne_cosine_low_components).plot.scatter(x=tsne_cosine_low_components[0], y=tsne_cosine_low_components[1], title='Cosine TSNE: Perplexity 5', ax=axes[2][1])


# # Principle Gene Shaving
# 1. Start with the expression matrix X, with each row centred at zero  
# 2. Compute the leading Principle Component of the rows of X  
# 3. Shave off a portion \alpha (typically 10%) of the gene having the smallest absolute inner product with teh leading principal component.  
# 4. Repeat steps 2 and 3 until only one gene remains.  
# 5. This produces a nested sequence of gene clusters $S_1 \in ... \in S_{k_2} \in S_{K_1} \in S_{k} \in S_{N}$.  
# 6. Orthogonalize each row of X with respect to $\bar{X_{S_K}}$, the average gene in $X_{\hat{S_K}}$.  
# 7. Repeat steps 1-5 above with the orthogonalized data, until M clusters are found with M chosen a priori. 
#   
# \- from "'Gene shaving' as a method for identifying distinct sets of genes with similar expression patterns" original paper

# As my notebook is more an example of the method and less an thorough exploration of the data, I have opted to use larger shaving having $\alpha =0.25$, mainly due to the resource limitation of the notebook. 

# In[ ]:


from sklearn.metrics import pairwise_distances

def d_stat(X):
    return np.sum(pairwise_distances(X)**2).astype('float32')

def gap(data, labels=None):
    grouper = (pd.DataFrame(data)
     .groupby(labels))
    
    D_k = grouper.apply(d_stat)
    N_k = grouper.count()[0]
    
    W_k = (D_k/(2*N_k)).sum()
    
    D = d_stat(data)
    N = X.shape[0]
    W = D / (2 * N)
    
    return np.log(W) - np.log(W_k)

def orth(v, u):
    return v - (v@v)/(u@u) * u


# In[ ]:


from sklearn.decomposition import PCA
from functools import partial
from sklearn.metrics import pairwise_distances_argmin

class PrincipleGeneShaving:
    def __init__(self, n_clusters = 2, alpha = 0.1):
        self.n_clusters = n_clusters
        self.alpha = alpha
        
    def fit_transform(self, X):
        # centre each row at zero
        X  = X - X.mean(1).reshape(-1,1)
        svd = PCA(1)
        
        labels = np.full((X.shape[0], self.n_clusters), 0)
        for k in range(self.n_clusters - 1):
            
            # shave
            indexes = [np.arange(X.shape[0]).flatten()]
            S = [X]
            while S[-1].shape[0] > 1:
                P = svd.fit_transform(S[-1].T)

                inner = S[-1] @ P

                threshold = np.quantile(inner, self.alpha)
                not_shaved = (inner > threshold).reshape(-1,1).flatten().copy()
                
                if not_shaved.ndim == 2:
                    print('reshape')
                    ns = np.array(sum(not_shaved.tolist(), []))
                else:
                    ns = not_shaved
                                
                indexes.append(indexes[-1][ns])
                S_prime = S[-1][ns, :]

                if S_prime.shape[0] > 1:
                    S.append(S_prime)
                else:
                    break

            # score
            scores = []
            for index in indexes:
                l = np.full(X.shape[0], k)
                l[index] = k + 1

                scores.append(gap(X, l))

            max_score = np.argmax(scores)

            labels[indexes[max_score], k] += 1
            
            # orthogonalize
            X_bar = (np.ascontiguousarray(X[(labels[:, k] == 1), :].mean(0))
                     .ravel()
                     .astype('float32'))
            
            assert X.shape[1] == X_bar.shape[0]
            X = np.apply_along_axis(partial(orth, u=X_bar), 1, np.ascontiguousarray(X)).astype('float32')
        
        return pairwise_distances_argmin(labels, np.unique(labels, axis=0))


# In[ ]:


pgs = PrincipleGeneShaving(3, 0.25)
labels = pgs.fit_transform(X)
hv.Bars(pd.Series(labels, name='Count of labels').value_counts()).opts(xlabel='Label')


# In[ ]:


components = ['Component 1', 'Component 2']
(hv.Scatter(pd.DataFrame(U, columns=components)
            .assign(cluster=labels.astype(str)), 
            kdims=components[0], vdims=[components[1], 'cluster'])
 .opts(color='cluster', cmap='Category10',
       tools=['hover'], size=7,
       width=800, height=600, title='SARS-COV-2: Principle Gene Shaving'))


# Below, I have visualize the 'mode' gene produced by each cluster, which may be used for comparison. 

# In[ ]:


super_genes = (sequences
               .assign(label=labels)
               .groupby('label').apply(lambda df: df.mode())
               .dropna()
               .drop(columns=['label']))


# In[ ]:


hv.Raster(super_genes.iloc[[0], :]
           .replace({k: i for i, k in enumerate(list(replace.keys()) + ['S', 'U'])}).astype(np.int).values).opts(width=1000, height=100, title='Super-gene 1')


# In[ ]:


hv.Raster(super_genes.iloc[[1], :]
           .replace({k: i for i, k in enumerate(list(replace.keys()) + ['S', 'U'])}).astype(np.int).values).opts(width=1000, height=100, title='Super-gene 2')


# In[ ]:


hv.Raster(super_genes.iloc[[2], :]
           .replace({k: i for i, k in enumerate(list(replace.keys()) + ['S', 'U'])}).astype(np.int).values).opts(width=1000, height=100, title='Super-gene 3')

