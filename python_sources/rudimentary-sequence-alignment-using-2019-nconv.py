#!/usr/bin/env python
# coding: utf-8

# # Rudimentary Sequence Alignment using Coronavirus Genome
# 
# [Link to the Original Post](https://labs.imaginea.com/novel-corona-virus-sequence-alignment/)
# 
# 
# 
# Pairwise sequence alignment is a useful tool in many fields of biology. For example, the similarity between sequences can used be in evolutionary analysis to find out what organisms share a common ancestor. Alternatively, the similarity between amino acid sequences can be used to predict the structure and functions of protein domains. Another use of pairwise sequence analysis is in genome sequencing assembly, where matches are used to find overlaps in the shorter pieces of DNA sequenced. (Source:Genome Assembly Page)
# 
# 
# > Now the question we will try to answer in the post: **How similar are two sequences?**
# 
# Let us know some terms before we jump into answering the above question with coding and visualizing the sequence.

# # What is Sequence Alignment ?
# 
# Sequence alignment is a technique for arranging sequences of DNA, RNA, or protein to identify sectors/regions of similarity. It is a part of Genome Assembly which you can read about at this post.
# 
# The similarity of the two given sequence may result of structural or evolutionary or functionality similarity.
# 
# ## How does a Genome Look?
# I will be using nucleotide sequencing of Coronavirus and try to compare it with a common existing virus like Human Rhinovirus A1.
# 
# Below is the example of Coronavirus Nucleotide:
# 
# > cacgcagtat aattaataac taattactgt cgttgacagg acacgagtaa ctcgtctatc
# 
# Here, cytosine (c) is followed by adenine (a) which is followed by a cytosine (c), which in turn is followed by a guanine (g), and so on. (T stands for thymine(t))
# 
# Here, I will be using Pairwise Similarity Alignment and see the scores.

# ## How does Pairwise Similarity Alignment work ?
# Let us see the below image and see different possible components of a sequence.
# 
# ![](https://labs.imaginea.com/content/images/2020/01/image.png)
# 
# Different Components (Source: compbio)
# 
# 
# There can be multiple ways to score the similarity from the genome sequence, like,
# 
# 
# ![](https://labs.imaginea.com/content/images/2020/01/image-1.png)
# 
# Different Scoring Methods(Source: compbio)
# 
# 
# We can chalk down 3 basic aspects for scoring.
# 
# 1. **Match Value Scoring** -Assign a value when there is an exact math
# 1. **Mismatch Value Scoring** -Genataly a negative value is assigned if there is a mismatch.
# 1. **Gap Penalty** - Gaps can be either Open or Extended gaps which is also assigned negative values.

# In[ ]:


pip install biopython


# In[ ]:


from Bio import SeqIO


import os
print(os.listdir("../input"))

count = 5
for record in SeqIO.parse("../input/2019ncov/2019-nCoV.fasta", "fasta"):
    print(record.description)
    print(repr(record.seq))
    nCoV2019 = record.seq
    print(len(record))
    print("\n")
    count = count - 1
    if count == 0:
        break


# ## I wont be using the entre file rather I will a small piece of it.

# # LOCALXX

# In[ ]:


# Import pairwise2 module
from Bio import pairwise2

# Import format_alignment method
from Bio.pairwise2 import format_alignment

# Define two sequences to be aligned
X = "attaaaggtt tataccttcc caggtaacaa accaaccaac tttcgatctc ttgtagatct"
Y = "ttaaaactgg gtgtgggttg ttcccaccca caccacccaa tgggtgttgt actctgttat"

alignments = pairwise2.align.localxx(X, Y)

# Use format_alignment method to format the alignments in the list
for a in alignments:
    print(format_alignment(*a))


# # LOCALMS

# In[ ]:


# Import pairwise2 module
from Bio import pairwise2

# Import format_alignment method
from Bio.pairwise2 import format_alignment

# Define two sequences to be aligned
X = "attaaaggtt tataccttcc caggtaacaa accaaccaac tttcgatctc ttgtagatct"
Y = "ttaaaactgg gtgtgggttg ttcccaccca caccacccaa tgggtgttgt actctgttat"


alignments = pairwise2.align.localms(X, Y,  2, -1, -0.5, -0.1)

# Use format_alignment method to format the alignments in the list
for a in alignments:
    print(format_alignment(*a))


# # GLOBALXX 

# In[ ]:


# Import pairwise2 module
from Bio import pairwise2

# Import format_alignment method
from Bio.pairwise2 import format_alignment

# Define two sequences to be aligned
X = "attaaaggtt tataccttcc caggtaacaa accaaccaac tttcgatctc ttgtagatct"
Y = "ttaaaactgg gtgtgggttg ttcccaccca caccacccaa tgggtgttgt actctgttat"

# Get a list of the local alignments between the two sequences ACGGGT and ACG
# No parameters. Identical characters have score of 1, else 0.
# No gap penalties.

alignments = pairwise2.align.globalxx(X, Y)


# Use format_alignment method to format the alignments in the list
for a in alignments:
    print(format_alignment(*a))


# # GLOBALMS

# In[ ]:


from Bio import pairwise2
from Bio.pairwise2 import format_alignment

# Define two sequences to be aligned (X==Y)
# Define two sequences to be aligned
X = "attaaaggtt tataccttcc caggtaacaa accaaccaac tttcgatctc ttgtagatct"
Y = "ttaaaactgg gtgtgggttg ttcccaccca caccacccaa tgggtgttgt actctgttat"


al = pairwise2.align.globalms(X, Y, 2, -1, -0.5, -0.1)


# Use format_alignment method to format the alignments in the list
for a in al:
    print(format_alignment(*a))


# # We can test with other virus RNA's and find the similarity score. That will help us to give more insight about Coronavirus.

# There are advanced algorithms which dynamic programming to obtain Global and Local Alignments.
# 
# ## Further Readings:
# 
# 1. Smith-Waterman Algorithm
# 2. Convex/Affine Gap Penalty
# 3. Needleman-Wunch Algorithm
# 
# 
# 
# ## References:
# * http://biopython.org/DIST/docs/tutorial/Tutorial.html
# * http://www.genomenewsnetwork.org/resources/whats_a_genome/Chp2_1.shtml
# * http://compbio.pbworks.com/w/page/16252912/Pairwise%20Sequence%20Alignment
# * https://www.ncbi.nlm.nih.gov/nuccore/MN908947?fbclid=IwAR1tyqtVCorwopIhxSwEkW3rVbWhAIcEbvT75K0p5WEQImflzolX6Yw3f8o
# * https://www.ncbi.nlm.nih.gov/nuccore/NC_038311.1
# * https://molbiol-tools.ca/Genomics.htm
# * https://en.wikipedia.org/wiki/Nucleic_acid_sequence
# * http://compbio.pbworks.com/w/page/16252894/Genome%20Sequencing%20and%20Assembly

# In[ ]:




