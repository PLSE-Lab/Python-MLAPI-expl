#!/usr/bin/env python
# coding: utf-8

# ---
# 
# # A Complete Analysis (Genomes and EDA)
# 
# ---

# This kernel will teach you how to sequence and perform EDA the coronavirus. Also, I'll try to explain the biology behind Biopython.

# # 1. **Introduction**
# 
# ---
# 
# ## 1a. A Biological Perspective.
# 
# The first thing you need to understand when using genome sequencing is that every DNA strand has four main bases. These are known as:
# * Adenine/A (C5H5N5)
# * Guanine/G (C5H5N5O)
# * Cytosine/C (C4H5N3O)
# * Thymine/T (C5H6N2O2)
# 
# You will also find the letter `U` being used to represent Uracil, which is a demethylated form of thymine. Its chemical formula is C4H4N2O2.

# ## 1b. Starting the sequence

# Let's import our library.

# In[ ]:


import Bio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as S


# We need to use the IUPAC alphabet for this.

# In[ ]:


from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
DNAseq = SeqIO.read('/kaggle/input/coronavirus-genome-sequence/MN908947.fna', "fasta")


# Our little sequence works a lot like a Python array.

# In[ ]:


from Bio.Data import CodonTable
print(CodonTable.unambiguous_rna_by_name['Standard'])


# In[ ]:


# Credit to Abish Pius
DNA = DNAseq.seq
mRNA = DNA.transcribe()
Amino_Acid = mRNA.translate()


# In[ ]:


Proteins = Amino_Acid.split('*')
#Remove chains smaller than 20 amino acids long
for i in Proteins[:]:
    if len(i) < 20:
        Proteins.remove(i)
Proteins


# In[ ]:


genome_dict = []
for i in range(80):
    genome_dict.append(i)


# In[ ]:


genomes = pd.DataFrame({'id': genome_dict, 'protein_id': Proteins})


# What is the most common single molecule in each protein?

# In[ ]:


genomes.protein_id = genomes.protein_id.astype('str')
genomes['unique_amino'] = genomes['protein_id'].apply(lambda x: len(set(str(x))))
genomes['num_amino'] = genomes['protein_id'].apply(lambda x: len(x))
genomes['unique_per_amino'] = genomes['unique_amino'] / genomes['num_amino']
genomes['unique_per_amino'].value_counts()


# We have an extremly high concentration of 0.4 and 0.5. Perhaps this can get us somewhere?

# In[ ]:


genomes


# In[ ]:


L = ['A', 'B', 'C', 'D','E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


# In[ ]:


for x in L:
    genomes[f'{x}_count'] = genomes['protein_id'].str.count(x)
    genomes[f'{x}_per_amino'] = genomes[f'{x}_count'] / genomes['num_amino']

