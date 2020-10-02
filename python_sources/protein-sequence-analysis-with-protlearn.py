#!/usr/bin/env python
# coding: utf-8

# ## About this Notebook

# Researchers and engineers all over the globe are scrambling to develop new vaccines and diagnostic tools for COVID-19. One aspect of research is the analysis and quantification of virus proteins. Thus, this notebook serves the purpose of engineering features from a protein sequence that can be used as inputs to a classifier or regressor. In this illustration, we will use the protein sequence of the **Human SARS coronavirus (SARS-CoV) spike glycoprotein**, which was obtained from UniProt.

# Disclaimer: I am the author of the `protlearn` package that is used throughout this notebook. The documentation of it can be found [here](https://github.com/tadorfer/ProtLearn).

# In[ ]:


# install protlearn
get_ipython().system('pip install protlearn')


# In[ ]:


import pandas as pd 
import protlearn as pl
import matplotlib.pyplot as plt


# ## Get protein sequence

# Get the amino acid sequence of the Human SARS coronavirus (SARS-CoV) spike glycoprotein from [UniProt](https://www.uniprot.org/uniprot/P59594.fasta). In this case, the sequence was copied from the UniProt website. Alternatively, the sequence can also be downloaded and imported into Python using the `biopython` package.

# In[ ]:


spike = "MFIFLLFLTLTSGSDLDRCTTFDDVQAPNYTQHTSSMRGVYYPDEIFRSDTLYLTQDLFL         PFYSNVTGFHTINHTFGNPVIPFKDGIYFAATEKSNVVRGWVFGSTMNNKSQSVIIINNS         TNVVIRACNFELCDNPFFAVSKPMGTQTHTMIFDNAFNCTFEYISDAFSLDVSEKSGNFK         HLREFVFKNKDGFLYVYKGYQPIDVVRDLPSGFNTLKPIFKLPLGINITNFRAILTAFSP         AQDIWGTSAAAYFVGYLKPTTFMLKYDENGTITDAVDCSQNPLAELKCSVKSFEIDKGIY         QTSNFRVVPSGDVVRFPNITNLCPFGEVFNATKFPSVYAWERKKISNCVADYSVLYNSTF         FSTFKCYGVSATKLNDLCFSNVYADSFVVKGDDVRQIAPGQTGVIADYNYKLPDDFMGCV         LAWNTRNIDATSTGNYNYKYRYLRHGKLRPFERDISNVPFSPDGKPCTPPALNCYWPLND         YGFYTTTGIGYQPYRVVVLSFELLNAPATVCGPKLSTDLIKNQCVNFNFNGLTGTGVLTP         SSKRFQPFQQFGRDVSDFTDSVRDPKTSEILDISPCSFGGVSVITPGTNASSEVAVLYQD         VNCTDVSTAIHADQLTPAWRIYSTGNNVFQTQAGCLIGAEHVDTSYECDIPIGAGICASY         HTVSLLRSTSQKSIVAYTMSLGADSSIAYSNNTIAIPTNFSISITTEVMPVSMAKTSVDC         NMYICGDSTECANLLLQYGSFCTQLNRALSGIAAEQDRNTREVFAQVKQMYKTPTLKYFG         GFNFSQILPDPLKPTKRSFIEDLLFNKVTLADAGFMKQYGECLGDINARDLICAQKFNGL         TVLPPLLTDDMIAAYTAALVSGTATAGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYE         NQKQIANQFNKAISQIQESLTTTSTALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLN         DILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSK         RVDFCGKGYHLMSFPQAAPHGVVFLHVTYVPSQERNFTTAPAICHEGKAYFPREGVFVFN         GTSWFITQRNFFSPQIITTDNTFVSGNCDVVIGIINNTVYDPLQPELDSFKEELDKYFKN         HTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYVWL         GFIAGLIAIVMVTILLCCMTSCCSCLKGACSCGSCCKFDEDDSEPVLKGVKLHYT"


# In[ ]:


# delete empty spaces
spike = spike.replace(" ", "")


# In[ ]:


len(spike)


# As `protlearn` was designed for datasets containing multiple sequences, it requires the input to be a Pandas DataFrame with column name 'Sequence'.

# In[ ]:


# build dataframe
spike = pd.DataFrame(data=[spike], columns=['Sequence'])


# ## Feature Engineering

# Information on the AAIndex used below can be accessed [here](https://www.genome.jp/aaindex/).

# #### Compute Features

# In[ ]:


# compute the frequency of each amino acid in the sequence
comp = pl.composition(spike)
comp


# In[ ]:


# compute the ngram compositions of the sequence (in this case, only the dipeptide composition)
ng = pl.ngram_composition(spike)
ng


# In[ ]:


# compute aaindex1
aa1 = pl.aaindex1(spike)
aa1


# In[ ]:


# compute aaindex1
aa2 = pl.aaindex2(spike)
aa2


# In[ ]:


# compute aaindex3
aa3 = pl.aaindex3(spike)
aa3


# #### Concatenate features

# In[ ]:


features = pd.concat([comp, ng, aa1, aa2, aa3], axis=1)
features


# In[ ]:


# the protlearn package automatically removes columns with NaNs 
features.isna().sum().sum()


# ## Visualizations

# #### Amino acid composition

# In[ ]:


pl.viz_composition(spike)


# #### N-gram composition

# In[ ]:


plt.figure(figsize=(20,12))
plt.subplot(2,1,1)
pl.viz_ngram(spike, method='relative', ngram=2, top=20, xtick_rotation=60)
plt.subplot(2,1,2)
pl.viz_ngram(spike, method='relative', ngram=3, top=5, xtick_rotation=60)


# The above plot shows the top 20% of the dipeptide composition and the top 5% of the tripeptide composition of the sequence.

# ## Conclusion

# This notebook provides some basic feature engineering techniques for protein sequences using the `protlearn` package. If you have any **criticism / feedback / or suggestions** for additional functions, I would highly welcome that :)
