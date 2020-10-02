#!/usr/bin/env python
# coding: utf-8

# # Genomic Visualization via Dimensionality Reduction
# This notebook will show you how to embed and visualize your 23andMe genotype data using samples from the [1000 Genomes Project](http://www.internationalgenome.org/) as a reference population.  Ancestry-informative snps (single nucleotide polymorphisms) are locations in the genome that have significant variance across global populations. There are several scientific publications which have shared these locations and this kernel uses 55 AISNPs from [Kidd et al.](https://www.ncbi.nlm.nih.gov/pubmed?db=pubmed&cmd=Retrieve&dopt=citation&list_uids=24508742) Dimensionality redcution techniques include PCA, t-SNE, and UMAP. 
# 
# This kernel depends on a few files in the `input/` directories:  
# `my23andme/` -- This directory will have **your** 23andMe file (keep it private)  
# `23andme2vcf/` -- perl script to convert your 23andMe file to a common genomics file format called [.vcf](https://samtools.github.io/hts-specs/VCFv4.2.pdf)  
# `ancestry_informative_snps/` -- This directory contains genotype data from the 1000 Genomes Project, the reference population that we'll fit our dimenionality reduction methods on. The directory also contains genomic coordinates for 55 locations in the genome that display significant variance between continental populations.
# 
# ![3d](https://s3.amazonaws.com/ancestry-snps-1kgp/tgviz3d.png)
# 
# # Follow these steps to visualize your genotype data:
# 1. Download your raw data from 23andMe using [this link](https://you.23andme.com/tools/data/download/).  
# 2. Check your email and unzip the `genome_Your_Name_.zip`  
# 3. Fork this kernel!   
# 4. Click "+ Add Data" to add your downloaded genotype file from 23andMe.  
#     a. Name the Dataset `my23andme`  
#     b. Rename the file to `my23andme.txt`  
#     c. You should keep this dataset private because multiple users can't have similarly named public datasets (I think)
# 5. Follow along below :)
#  
# ![gif](http://g.recordit.co/gcJtejs1NN.gif)
# 
# # Share screenshots of your 3D graph in the comments to show the diversity of the Kaggle community
# If you're interested in learning more about the pre-processing techniques or interactive visualization with plotly's Dash, I created an [app](http://tgviz.herokuapp.com/) with the [source code hosted on GitHub](https://github.com/arvkevi/tgviz). 

# In[ ]:


get_ipython().run_cell_magic('sh', '', 'pip install MulticoreTSNE pysam pyvcf;\napt-get install tabix;')


# In[ ]:


import bz2
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pysam
import requests
import seaborn as sns
import umap
import vcf

from plotly.offline import init_notebook_mode, iplot
from MulticoreTSNE import MulticoreTSNE as mTSNE
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')
init_notebook_mode(connected=True) #plotly


# # Convert the 1000 Genomes Project genotypes to a DataFrame

# In[ ]:


def vcf2df(vcf_fname):
    """Convert a subsetted vcf file to pandas DataFrame"""
    vcf_reader = vcf.Reader(filename=vcf_fname)
    
    df = pd.DataFrame(index=vcf_reader.samples)
    for variant in vcf_reader:
        df[variant.ID] = [call.gt_type if call.gt_type is not None else 3 for call in variant.samples]

    return df


# In[ ]:


df = vcf2df('/kaggle/input/ancestry-informative-snps/Kidd.55AISNP.1kG.vcf')


# # Inspect the encoded DataFrame
# Genotype Encodings:  
# 0 = reference allele / reference allele    
# 1 = reference allele / alternate allele  
# 2 = alternate allele / alternate allele  
# 3 = Unknown
# 
# Example:  
# Sample `HG00096` (row 1) at position `rs7554936` (column 2) is `1`, meaning this person has a `C/T` at this location.
# If you [lookup this snp in gnomad](https://gnomad.broadinstitute.org/variant/1-151122489-C-T), notice in the **Population Frequencies** table, this snp is an excellent discriminator between `East Asian`(85% allele frequency) and `South Asian` (0% allele frequency) populations.

# In[ ]:


df.head(3)


# These aren't really class labels, but they represent the populations and subpopulations that the 1000 Genomes Project samples identified as.  
# A more detailed description of these populations can be found [here](http://www.internationalgenome.org/faq/which-populations-are-part-your-study/).

# In[ ]:


samples = 'ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel'
dfsamples = pd.read_csv(samples, sep='\t')
dfsamples.set_index('sample', inplace=True)
dfsamples.drop(['Unnamed: 4', 'Unnamed: 5'], inplace=True, axis=1)

# Each sample is assigned a population, a super population, and a gender
dfsamples.head(3)


# # Read the 55 ancestry-informative snp locations into a DataFrame

# In[ ]:


dfaim = pd.read_csv('/kaggle/input/ancestry-informative-snps/Kidd_55_AISNPs.txt', sep='\t')
dfaim.head(3)


# # Convert your 23andMe file to a common genomics file format: .vcf
# Your downloaded file from 23andMe has one of  `v3`, `v4` or `v5` in the filename, you should appropriately change the `VERSION_23ANDME` variable below to be either `3, 4, or 5`. (check the file name of the `.zip` archive if you don't recall, since I instructed you to change the name...)
# 
# `"X sites were not included; ...` is OK to see from the output of the next cell!

# In[ ]:


get_ipython().run_cell_magic('sh', '', 'cd /kaggle/input/23andme2vcf/23andme2vcf/23andme2vcf/\n\n# you may need to change this variable\nVERSION_23ANDME=4\n\nperl /kaggle/input/23andme2vcf/23andme2vcf/23andme2vcf/23andme2vcf.pl \\\n/kaggle/input/my23andme/my23andme.txt \\\n/kaggle/working/my23andme.vcf \\\n"$VERSION_23ANDME"\n\ncd /kaggle/working\n\nbgzip /kaggle/working/my23andme.vcf -f && tabix /kaggle/working/my23andme.vcf.gz')


# # Convert your vcf to a DataFrame

# In[ ]:


# make sure the filename is the correct path to your created .vcf file
my_vcf_filename = '/kaggle/working/my23andme.vcf.gz'

vcf_reader = vcf.Reader(filename=my_vcf_filename)

# create a small DataFrame for your data
dfme = pd.DataFrame(index=['me'], columns=df.columns)
dfme.loc['me'] = np.repeat([3], dfme.shape[1])

# iterate over the AISNPs and query your genotypes at these locations
# assign them to dfme
for i, row in dfaim.iterrows():
    chrom = row['Chr']
    pos = row['Build 37 nt position'].replace(',', '')
    rsid = row['dbSNP rs#']
    # The file is indexed with tabix, so we can quickly extract the location with `fetch()`
    for variant in vcf_reader.fetch('chr{}'.format(chrom), int(pos)-1, int(pos)):
        dfme.loc['me', rsid] = [call.gt_type if call.gt_type is not None else 3 for call in variant.samples][0]


# In[ ]:


# how many missing genotypes? Count 3's
dfme.T['me'].value_counts()


# # One Hot Encode genotypes

# In[ ]:


ncols = len(df.columns)
ohe = OneHotEncoder(categories=[range(4)] * ncols, sparse=False)

X = ohe.fit_transform(df.values)
X_me = ohe.transform(dfme.values)


# # Reduce the dimensionality
# of the reference genotypes and your genotypes -- then add your data to the reference data.
# 
# Use `reduce_dim` and try selecting different algorithms from one of `{'PCA', 'TSNE', 'UMAP'}`

# In[ ]:


def reduce_dim(X, X_me, algorithm='PCA', n_components=3):
    """Reduce the dimensionality of the 55 AISNPs
    :param X: One-hot encoded 1kG 55 AISNPs.
    :type X: array
    :param X_me: Your one-hot encoded genotype array
    :type X_me: array
    :param algorithm: The type of dimensionality reduction to perform. 
        One of {PCA, UMAP, TSNE}
    :type algorithm: str 
    :param n_components: The number of components to return in X_red 
    :type n_components: int
    
    :returns: The transformed X[m, n] array - reduced to X[m, n_components] by `algorithm`
    """
    
    if algorithm == 'PCA':
        pca = PCA(n_components=n_components).fit(X)
        X_red = pca.transform(X)
        X_red_me = pca.transform(X_me)
        # merge your data into the same table as the reference data
        X_merged = np.vstack((X_red, X_red_me))
        df_red = pd.DataFrame(X_merged, 
                              index=df.index.append(dfme.index), 
                              columns=['component1', 'component2', 'component3'])
    elif algorithm == 'TSNE':
        # TSNE, Barnes-Hut have dim <= 3
        if n_components > 3:
            print('The Barnes-Hut method requires the dimensionaility to be <= 3')
            return None
        else:
            X_merged = np.vstack((X, X_me))
            X_red = mTSNE(n_components=n_components, n_jobs=4).fit_transform(X_merged)
            df_red = pd.DataFrame(X_red, 
                                  index=df.index.append(dfme.index), 
                                  columns=['component1', 'component2', 'component3'])
    elif algorithm == 'UMAP':
        umap_ = umap.UMAP(n_components=n_components).fit(X)
        X_red = umap_.transform(X)
        X_red_me = umap_.transform(X_me) 
        # merge your data into the same table as the reference data
        X_merged = np.vstack((X_red, X_red_me))
        df_red = pd.DataFrame(X_merged, 
                              index=df.index.append(dfme.index), 
                              columns=['component1', 'component2', 'component3'])
    else:
        return None
    
    return df_red


# In[ ]:


# Here, we choose PCA
df_dim = reduce_dim(X, X_me, algorithm='PCA')


# # Join the samples DataFrame with the reduced dimensions DataFrame
# 
# This will add the population "labels" in the dimensionality-reduced DataFrame

# In[ ]:


# add a new record for yourself in dfsamples
dfsamples.loc['me'] = ['me', 'me', 'male']
df_dim = df_dim.join(dfsamples)


# # Plot with Plotly!

# In[ ]:


def generate_figure_image(groups, layout):
    data = []

    for idx, val in groups:
        if idx == 'me':
            scatter = go.Scatter3d(
            name=idx,
            x=val.loc[:, 'component1'],
            y=val.loc[:, 'component2'],
            z=val.loc[:, 'component3'],
            text=[idx for _ in range(val.loc[:, 'component1'].shape[0])],
            textposition='middle right',
            mode='markers',
            marker=dict(
                size=12,
                symbol='diamond'
                )
            )
        else:
            scatter = go.Scatter3d(
                name=idx,
                x=val.loc[:, 'component1'],
                y=val.loc[:, 'component2'],
                z=val.loc[:, 'component3'],
                text=[idx for _ in range(val.loc[:, 'component1'].shape[0])],
                textposition='middle right',
                mode='markers',
                marker=dict(
                    size=4,
                    symbol='circle'
                )
            )
        data.append(scatter)

    figure = go.Figure(
        data=data,
        layout=layout
    )
    return figure


# In[ ]:


layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(
                    title='Component 1',
                    showgrid=True,
                    zeroline=False,
                    showticklabels=True
                ),
                yaxis=dict(
                    title='Component 2',
                    showgrid=True,
                    zeroline=False,
                    showticklabels=True
                ),
                zaxis=dict(
                    title='Component 3',
                    showgrid=True,
                    zeroline=False,
                    showticklabels=True
                )
            )
        )


# In[ ]:


pop_resolution = 'super_pop'
groups = df_dim.groupby(pop_resolution)
figure = generate_figure_image(groups, layout)


# In[ ]:


iplot(figure)


# In[ ]:


# output a static image
plt.figure(figsize=(8,8));
plt.title('Genomic Visualizations');
sns.scatterplot(x='component1', y='component2', data=df_dim, hue=pop_resolution);
plt.savefig('my23andme.png')


# # Plot with TSNE

# In[ ]:


df_dim = reduce_dim(X, X_me, algorithm='TSNE')
df_dim = df_dim.join(dfsamples)
groups = df_dim.groupby(pop_resolution)
figure = generate_figure_image(groups, layout)
iplot(figure)


# # Plot with UMAP

# In[ ]:


df_dim = reduce_dim(X, X_me, algorithm='UMAP')
df_dim = df_dim.join(dfsamples)
groups = df_dim.groupby(pop_resolution)
figure = generate_figure_image(groups, layout)
iplot(figure)


# In[ ]:


# remove your genetic data from the working directory
get_ipython().system('rm my23andme.vcf.gz my23andme.vcf.gz.tbi sites_not_in_reference.txt')

