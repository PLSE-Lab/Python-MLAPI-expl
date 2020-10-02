#!/usr/bin/env python
# coding: utf-8

# **Preprocessed TorchText Data**

# The class skewed towards 84% percent approved, so I created a train/val/dev that has a 50/50 balance (all the no's and an equal but random selection of yes's).  Also removed junk (\n, \r \") from the text and made everything lowercase.  Feel free to use preprocessing script as needed.
# The majority of the current kernals focus on standard data science (what time was it posted, how many previous successful posts were there) rather than NLP.  If you're interested in training a NLP model, I set up a torchtext for the data.

# 1) Pull from git 
# https://github.com/DenisPeskov/DonorsChooseNLP
# 
# 2) Ensure paths match up for train.csv/val/dev
# 
# 3) All you have to do is:
# from DonorsChooseDataset import DCDataset
# train, val, dev = DCDataset.iters()

# If there's significant interest (as determined by upvotes), I'll post a couple of simple models and maybe a fancier attention-based one.  

# In[ ]:




