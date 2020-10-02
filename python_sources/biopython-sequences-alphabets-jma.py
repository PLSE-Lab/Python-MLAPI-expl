#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
my_seq = Seq("AGTACACTGGT", IUPAC.unambiguous_dna)
my_seq


# In[ ]:


my_seq.alphabet


# In[ ]:


my_seq.count("A")


# In[ ]:


from Bio.SeqUtils import GC
GC(my_seq)


# In[ ]:


my_mRNA = my_seq.transcribe()
my_seq.translate()


# In[ ]:


str(my_seq)


# In[ ]:


my_seq.complement()


# In[ ]:


my_seq.reverse_complement()


# In[ ]:


from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
messenger_rna = Seq("AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG", IUPAC.unambiguous_rna)
messenger_rna
messenger_rna.translate()

