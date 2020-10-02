#!/usr/bin/env python
# coding: utf-8

# # COVID19 CoV-2 Protein Analysis (WIP)
# 
# ![genome variants](https://i.imgur.com/Ewd1d6u.png)
# 
# Todo:
# - BLAST charts
# - Seq Alignment

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Loading Complementary DNA Sequence into an alignable file
from Bio import SeqIO
DNAseq = SeqIO.read('/kaggle/input/coronavirus-genome-sequence/MN908947.fna', "fasta") # This is DNA Sequence


# Since input sequence is FASTA (DNA), and Coronavirus is RNA type of virus, we need to:
# 
# 1. Transcribe DNA to RNA (`ATTAAAGGTT...` => `AUUAAAGGUU...`)
# 2. Translate RNA to Amino acid sequence (`AUUAAAGGUU...` => `IKGLYLPR*Q...`)
# 
# In the current scenario, the `.fna` file starts with **ATTAAAGGTT**, then we call `transcribe()` so `T` (thymine) is replaced with `U` (uracil), so we get the RNA sequence which starts with **AUUAAAGGUU**.
# 
# When we `translate()` the RNA sequence to amino-acid sequence, we get something like **IKGLYLPR*Q** (* is so called STOP codon, effectively * is separator for proteins). In our scenario, sequence looks like this:
# `IKGLYLPR*QTNQLSISCRSVL*TNFKICV`, where:
# 
# 1. `IKGLYLPR` encodes first protein (every letter encodes single amino-acid)
# 2. `QTNQLSISCRSVL` encodes second protein, and so on

# In[ ]:


DNA = DNAseq.seq
print('DNA', DNA[:10], '...')
mRNA = DNA.transcribe()
print('RNA', mRNA[:10], '...')
print('Total nucleotides:', len(mRNA))
# Obtain Amino Acid Sequence from mRNA (translation)
amino_acids = mRNA.translate(table=1, cds=False) 
print('Amino Acid', amino_acids[:30])
print('Total Amino acids', len(amino_acids))


# In[ ]:


#Identify all the Proteins (chains of amino acids)
Proteins = amino_acids.split('*') # * is translated stop codon
df = pd.DataFrame(Proteins)
df.describe()
print('Total proteins:', len(df))

def conv(item):
    return len(item)

def to_str(item):
    return str(item)

df['sequence_str'] = df[0].apply(to_str)
df['length'] = df[0].apply(conv)
df.rename(columns={0: "sequence"}, inplace=True)
df.head()


# In[ ]:


# Take only longer than 20
functional_proteins = df.loc[df['length'] >= 20]
print('Total functional proteins:', len(functional_proteins))
functional_proteins.describe()


# In[ ]:


# Plot lengths
plt.figure(figsize=(20,5))

plt.subplot(111)
plt.hist(functional_proteins['length'])
plt.title('Length of proteins -- histogram')


# Remove the extremes
plt.figure(figsize=(20,5))
wo = functional_proteins.loc[functional_proteins['length'] < 60]
plt.subplot(121)
plt.hist(wo['length'])
plt.title('Lenght of proteins (where < 60)')

wo = functional_proteins.loc[functional_proteins['length'] > 1000]
plt.subplot(122)
plt.hist(wo['length'])
plt.title('Length of proteins (where > 1000)')


# In[ ]:


# See what's about that huge protein
large_prot = functional_proteins.loc[functional_proteins['length'] > 2700]
l = large_prot['sequence'].tolist()[0]
print('Sequence sample:', '...',l[1000:1150],'...')


# # Mapping
# 
# This is probably `orf1ab`:
# 
# https://www.ncbi.nlm.nih.gov/nuccore/MN908947.3?report=genbank&log$=seqview

# # Comparing Human Coronavirus RNA
# 
# Using pariwise alignment to find similarities between variants.

# In[ ]:


from Bio import pairwise2
# Define sequences to be aligned
SARS = SeqIO.read("/kaggle/input/coronavirus-accession-sars-mers-cov2/sars.fasta", "fasta")
MERS = SeqIO.read("/kaggle/input/coronavirus-accession-sars-mers-cov2/mers.fasta", "fasta")
COV2 = SeqIO.read("/kaggle/input/coronavirus-accession-sars-mers-cov2/cov2.fasta", "fasta")

print('Sequence Lengths:')
print('SARS:', len(SARS.seq))
print('COV2:', len(COV2.seq))
print('MERS:', len(MERS.seq))


# In[ ]:


# Alignments using pairwise2 alghoritm
SARS_COV = pairwise2.align.globalxx(SARS.seq, COV2.seq, one_alignment_only=True, score_only=True)
print('SARS/COV Similarity (%):', SARS_COV / len(SARS.seq) * 100)
MERS_COV = pairwise2.align.globalxx(MERS.seq, COV2.seq, one_alignment_only=True, score_only=True)
print('MERS/COV Similarity (%):', MERS_COV / len(MERS.seq) * 100)
MERS_SARS = pairwise2.align.globalxx(MERS.seq, SARS.seq, one_alignment_only=True, score_only=True)
print('MERS/SARS Similarity (%):', MERS_SARS / len(SARS.seq) * 100)


# In[ ]:


# Plot the data
X = ['SARS/COV2', 'MERS/COV2', 'MERS/SARS']
Y = [SARS_COV/ len(SARS.seq) * 100, MERS_COV/ len(MERS.seq)*100, MERS_SARS/len(SARS.seq)*100]
plt.title('Sequence identity (%)')
plt.bar(X,Y)

