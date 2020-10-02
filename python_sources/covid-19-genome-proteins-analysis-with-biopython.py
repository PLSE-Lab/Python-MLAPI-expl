#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# I am going to use Biopython to analyze the established sequence of the caronavirus.
# For documentation look here: [BioPython](https://biopython.org/)
# 
# Some facts regarding the virus itself: [COVID-19](https://www.cdc.gov/coronavirus/2019-ncov/hcp/faq.html)
# * It is of the (+)ssRNA classification of viruses, which means it is a single stranded virus that can be directly translated into protein.
# * The actual virus is called SARS-CoV-2, Covid-19 is the name for the respiratory disease it causes (I found this interesting)

# In[ ]:


# This code is from Paul Mooney, link here: https://www.kaggle.com/paultimothymooney/explore-coronavirus-sars-cov-2-genome
from Bio import SeqIO
for sequence in SeqIO.parse('/kaggle/input/coronavirus-genome-sequence/MN908947.fna', "fasta"):
    print('Id: ' + sequence.id + '\nSize: ' + str(len(sequence))+' nucleotides')


# In[ ]:


# Loading Complementary DNA Sequence into an alignable file
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
DNAseq = SeqIO.read('/kaggle/input/coronavirus-genome-sequence/MN908947.fna', "fasta")


# The loaded file is the **SARS-CoV-2** Complementary DNA sequence, which is the strand of DNA that is transcribed from the +ssRNA of **SARS-CoV-2**. In my analyses, I will take this complementary DNA obtain the RNA sequence and then translate it into protein subunits, amino acids. (Note: In all my analyses I am assuming there is no post-transcriptional modification - aka DNA is made fully into mRNA with no shuffling or deletions)

# In[ ]:


DNA = DNAseq.seq
print(DNA)


# In[ ]:


#Obtain mRNA Sequence
mRNA = DNA.transcribe()
print(mRNA)


# The difference between the complementary DNA and the mRNA is just that the bases T (for Thymine) is replaced with U (for Uracil).

# In[ ]:


# Obtain Amino Acid Sequence from mRNA
Amino_Acid = mRNA.translate()
print(Amino_Acid)
print("Length of Protein:",len(Amino_Acid))
print("Length of Original mRNA:",len(mRNA))


# **Explaining Translation to Protein** <br/>
# I will try to make this as simple as possible for those unfamiliar with biology. 
# First, note that there are fewer sequences in the protein than the mRNA that is because 3 mRNA's are used to produce a single subunit of a protein, known as an amino acid, using the codon table shown below. The * is used to denote a stop codon, in these regions the protein has finished its full length. Many of these occur frequently and result in short lengths of protein, more likely than not these play little biological role and will be excluded in the further analyses. <br/>
# 
# **Code Details** <br/>
# Can be found here: [translate](https://biopython.org/DIST/docs/api/Bio.Seq-module.html#translate)

# In[ ]:


from Bio.Data import CodonTable
print(CodonTable.unambiguous_rna_by_name['Standard'])


# Let's now identify all the polypeptides so basically separating at the stop codon, marked by * . Then let's remove any sequence less than 20 amino acids long, as this is the smallest known functional protein ([if curious](https://www.science20.com/princerain/blog/smallest_protein)). Note: In humans the smallest known functional protien is 44 amino acids long.

# In[ ]:


#Identify all the Proteins (chains of amino acids)
Proteins = Amino_Acid.split('*')
Proteins


# In[ ]:


#Remove chains smaller than 20 amino acids long
for i in Proteins[:]:
    if len(i) < 20:
        Proteins.remove(i)


# In[ ]:


Proteins


# In[ ]:


# Code should match proteins with online database, however kaggle is not able to connect to the URL
from Bio.Blast import NCBIWWW
result_handle = NCBIWWW.qblast("blastp", "nt", Proteins)


# **Edit 3/16/20**
# Using [PSI-BLAST](https://www.ebi.ac.uk/Tools/sss/psiblast/), **Replicase polyprotein 1ab** is the protein that matches this sequence, a well documented viral RNA replicator. 

# Lets do some interesting secondary analyses on the proteins, instead!
# 
# Documentation here (many more analyses can be done, bare bones in this notebook): [ProtParam](https://biopython.org/wiki/ProtParam)

# In[ ]:


from Bio.SeqUtils.ProtParam import ProteinAnalysis
MW = []
aromaticity =[]
AA_Freq = []
IsoElectric = []
for j in Proteins[:]:
    a = ProteinAnalysis(str(j))
    MW.append(a.molecular_weight())
    aromaticity.append(a.aromaticity())
    AA_Freq.append(a.count_amino_acids())
    IsoElectric.append(a.isoelectric_point())


# In[ ]:


MW = pd.DataFrame(data = MW,columns = ["Molecular Weights"] )
MW.head()


# In[ ]:


# Plot Molecular Weights Distribution
sns.set_style('whitegrid');
plt.figure(figsize=(10,6));
sns.distplot(MW,kde=False);
plt.title("SARS-CoV-2 Protein Molecular Weights Distribution");


# OH! The protein with molecular weight near 300000 seems to be something of interest and something worth investigating for potential therapuetics!

# In[ ]:


MW.idxmax()


# In[ ]:


print(Proteins[48])
len(Proteins[48])


# Lets continue our investigation of this guy and see all the data we can for scientists and pharmacuetical companies trying to combat this dreaded disease!

# In[ ]:


# Protein of Interest
POI = AA_Freq[48]


# In[ ]:


plt.figure(figsize=(10,6));
plt.bar(POI.keys(), list(POI.values()), align='center')


# Looks like the number of Lysines and Valines in this protein are high which indicates a good number of Alpha-Helices! Pharmacuetical companies should be excited.
# 
# Final 2 features aromaticity (ring-ness) and isoelectric point (gives good insight into the pH ranges where protein is functional, potential pharma target).

# In[ ]:


print('The aromaticity % is ',aromaticity[48])
print('The Isoelectric point is', IsoElectric[48])


# I hope you enjoyed this notebook, and I hope our data science community can continue to tackle data regarding the caronavirus now that we are all working from home. Next steps would be to look at the structure of **Replicase polyprotein 1ab** ,which is the viral RNA encoded ribosome protein to make more viral RNA (the one we found to be of interest), and identify unique elements that can be targeted by drugs.
# 
# ![CoronaVirus](https://medialib.aafp.org/content/dam/AAFP/images/ann/2020-january/coronavirus_update920.jpg)
