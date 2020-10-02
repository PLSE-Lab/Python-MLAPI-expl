#!/usr/bin/env python
# coding: utf-8

# Explore some of the features in the Genetic Variant Classifications dataset. Each record represents a genetic "variant".
# For a more detailed description of the features please see the [dataset](https://www.kaggle.com/kevinarvai/clinvar-conflicting).

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style(style='whitegrid')
sns.set(font_scale=1.5);

import pandas as pd
import re


# In[ ]:


df = pd.read_csv('../input/clinvar_conflicting.csv', dtype={'CHROM': str, 38: str, 40: object})


# In[ ]:


df.CHROM.dtype


# In[ ]:


df.shape


# In[ ]:


df.groupby(['CHROM', 'POS', 'REF', 'ALT']).ngroups


# In[ ]:


df.CHROM.value_counts()


# The `CLASS` distribution is skewed a bit to the `0` class, meaning there are fewer variants with conflicting submissions.

# In[ ]:


ax = sns.countplot(x="CLASS", data=df)
ax.set(xlabel='CLASS', ylabel='Number of Variants');


# It's clear that conflicting variants are more common in some genes.   

# In[ ]:


gene_ct = pd.crosstab(df.SYMBOL, df.CLASS, margins=True)


# In[ ]:


gene_ct = pd.crosstab(df.SYMBOL, df.CLASS, margins=True)
gene_ct.drop('All', axis=0, inplace=True)

# limit to the 50 most submitted genes for visualization
gene_ct = gene_ct.sort_values(by='All', ascending=False).head(50)
gene_ct.drop('All', axis=1, inplace=True)

gene_ct.plot.bar(stacked=True, figsize=(12, 4));


# CLNVC (Variant Type)

# In[ ]:


vt_ct = pd.crosstab(df.CLNVC, df.CLASS, margins=True)
vt_ct.drop('All', axis=0, inplace=True)

# limit to the 50 most submitted genes for visualization
vt_ct = vt_ct.sort_values(by='All', ascending=False)
vt_ct.drop('All', axis=1, inplace=True)

vt_ct.plot.bar(stacked=True, figsize=(12, 4));


# Exons are features of genes that map sequences nucleotides that encode functional parts of DNA. Genes have differing numbers of exons, some have few, some have many. Let's see if, regardless of gene, whether or not conflicting variants are enriched in a general exon location.

# In[ ]:


df.EXON.fillna('0', inplace=True)
df['variant_exon'] = df.EXON.apply(lambda x: [int(s) for s in re.findall(r'\b\d+\b', x)][0])


# `variant_exon` = 0 represents that the variant is located in an **Intron**. Intron variants seem to be conflicting much more frequently than exon variants.

# In[ ]:


exondf = pd.crosstab(df['variant_exon'], df['CLASS'])
exondf.plot.bar(stacked=True, figsize=(20, 5));
plt.xlim(-0.5, 20.5);


# Parse and encode the `MC` (molecular consequence) field

# In[ ]:


MC_list = df.MC.dropna().str.split(',').apply(lambda row: list((c.split('|')[1] for c in row)))
MC_encoded = pd.get_dummies(MC_list.apply(pd.Series).stack()).sum(level=0)
MC_encoded = MC_encoded.reindex(index=MC_list.index)

# Incorporate the transformed MC feature into the existing DataFrame
df = df.join(MC_encoded).drop(columns=['MC'])

# Transformed MC feature
MC_encoded.head()


# Manually generate the `crosstab`, there is probably a faster method via `pandas`.

# In[ ]:


mccounts= {0: {},
           1: {},
           'All': {}
          }

for col in MC_encoded.columns:
    for class_ in [0, 1]:
        mccounts[class_][col] = df.loc[df['CLASS'] == class_][col].sum()
    
    mccounts['All'][col] = df[col].sum()
    
mc_ct = pd.DataFrame.from_dict(mccounts)

mc_ct_all = mc_ct.sum(axis=0)
mc_ct_all.name = 'All'
mc_ct = mc_ct.append(mc_ct_all, ignore_index=False)


# In[ ]:


mc_ct.drop('All', axis=0, inplace=True)

mc_ct = mc_ct.sort_values(by='All', ascending=False)
mc_ct.drop('All', axis=1, inplace=True)

mc_ct.plot.bar(stacked=True, figsize=(12, 4));


# Results from `SIFT` and `PolyPhen` software that predict the severity of a variant, in-silico.

# In[ ]:


sift_ct = pd.crosstab(df.SIFT, df.CLASS, margins=True)
sift_ct.drop('All', axis=0, inplace=True)

# limit to the 50 most submitted genes for visualization
sift_ct = sift_ct.sort_values(by='All', ascending=False)
sift_ct.drop('All', axis=1, inplace=True)

sift_ct.plot.bar(stacked=True, figsize=(12, 4));


# In[ ]:


pp_ct = pd.crosstab(df.PolyPhen, df.CLASS, margins=True)
pp_ct.drop('All', axis=0, inplace=True)

# limit to the 50 most submitted genes for visualization
pp_ct = pp_ct.sort_values(by='All', ascending=False)
pp_ct.drop('All', axis=1, inplace=True)

pp_ct.plot.bar(stacked=True, figsize=(12, 4));


# Encode `SIFT` and `PolyPhen`

# In[ ]:


df = pd.get_dummies(df, columns=['SIFT', 'PolyPhen'])


# Correlation for  categorical featuress by way of chi-square test
# 

# In[ ]:


from itertools import combinations
from scipy.stats import chi2_contingency


# In[ ]:


# select a few categorical features
categoricals_index = pd.MultiIndex.from_tuples(combinations(['CHROM', 'REF', 'ALT', 'IMPACT', 'Consequence', 'SYMBOL', 'CLASS'], 2))
categoricals_corr = pd.DataFrame(categoricals_index, columns=['cols'])


# In[ ]:


def chisq_of_df_cols(row):
    c1, c2 = row[0], row[1]
    groupsizes = df.groupby([c1, c2]).size()
    ctsum = groupsizes.unstack(c1)
    # fillna(0) is necessary to remove any NAs which will cause exceptions
    return chi2_contingency(ctsum.fillna(0))[1]


# In[ ]:


categoricals_corr[ 'chi2_p'] =  categoricals_corr.cols.apply(chisq_of_df_cols)


# In[ ]:


categoricals_corr


# In[ ]:


categoricals_corr.index = categoricals_index
categoricals_corr = categoricals_corr.chi2_p.unstack()


# I trid plotting a heatmap with `-np.log(p)` but it didn't look good as a visualization.

# In[ ]:


categoricals_corr


# The dark blue box in in the heatmap highlights the negatvie correlation with the **allele frequency** features. Commomn alleles are less likely to pathogenic (cause disease), therefore most labs agree they should be benign.

# In[ ]:


corr = df.select_dtypes(exclude='object').corr()

import numpy as np
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 12));

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True);

# Draw the heatmap with the mask and correct aspect ratio
g = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5});


from matplotlib.patches import Rectangle

g.add_patch(Rectangle((1, 6), 3, 1, fill=False, edgecolor='blue', lw=4));


# [One of the ways variants can be classified is by the amount (and type) of sequence change](https://www.ebi.ac.uk/training/online/course/human-genetic-variation-i-introduction/what-genetic-variation/types-genetic-variation).  
# A substitution of a nucleotide (letter) is considered a single nucleotide variant (SNV), these are sometimes referred to as single nucleotide polymorphisms (SNP).  
# When one or more nucleotides are inserted or deleted the variant is considered an insertion or deletion. Therefore, if the length of `REF` or `ALT` is >1 then the variant can be considered an Insertion or Deletion (indel), otherwise it can be considered a SNV.
# 

# In[ ]:


snvs = df.loc[(df.REF.str.len()==1) & (df.ALT.str.len()==1)]
indels = df.loc[(df.REF.str.len()>1) | (df.ALT.str.len()>1)]


# In[ ]:


len(df) == (len(snvs) + len(indels))


# SNVs are more likely to be conflicting than Indels

# In[ ]:


snp_indel = pd.concat([snvs.CLASS.value_counts(normalize=True).rename('snv_class'), 
                       indels.CLASS.value_counts(normalize=True).rename('indel_class')], 
                      axis=1).T


# In[ ]:


snp_indel.plot.bar(stacked=True, figsize=(12, 4));


# CLNDN are lists of diseases associated with the variant. It may be beneficial to treat both `not_specified` and/or `not_provided` as the same category..

# In[ ]:


clndn = pd.concat([df.CLASS.loc[(df.CLNDN=='not_specified') | (df.CLNDN=='not_provided') | (df.CLNDN=='not_specified|not_provided')].value_counts(normalize=True).rename('disease_not_specified'), 
                       df.CLASS.loc[(df.CLNDN!='not_specified') | (df.CLNDN!='not_provided') | (df.CLNDN!='not_specified|not_provided')].value_counts(normalize=True).rename('some_disease_specified')], 
                      axis=1).T


# In[ ]:


clndn.plot.bar(stacked=True, figsize=(12, 4));


# most AF values are very low

# In[ ]:


sns.distplot(df.AF_ESP, label="AF_ESP")
sns.distplot(df.AF_EXAC, label="AF_EXAC")
sns.distplot(df.AF_TGP, label="AF_TGP")
plt.legend();

