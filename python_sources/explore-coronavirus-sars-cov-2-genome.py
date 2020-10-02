#!/usr/bin/env python
# coding: utf-8

# # Explore Coronavirus (SARS-CoV-2) Genome
# * Using data from https://www.nature.com/articles/s41586-020-2008-3

# In[ ]:


import Bio.SeqIO
for sequence in Bio.SeqIO.parse('/kaggle/input/coronavirus-genome-sequence/MN908947.fna', "fasta"):
    print('Id: ' + sequence.id + '\nSize: ' + str(len(sequence))+' nucleotides')


# In[ ]:


sequence = '../input/coronavirus-genome-sequence/MN908947.txt'
with open(sequence) as text: 
    sequencestring = text.read(500)
    sequencestring = sequencestring[96:]
sequencestring = sequencestring.replace('\n','')

print(sequencestring)
cDNA = Seq(sequencestring)
RNA = cDNA.transcribe()
protein = RNA.translate()
print(protein)
print(len(protein))

