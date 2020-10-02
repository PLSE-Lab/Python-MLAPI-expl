#!/usr/bin/env python
# coding: utf-8

# # CCG Supertagging 
# 
#  This file provides the skeleton structure for your submission.
#  
#  Read the instructions carefully before proceeding.
#  
#  ## 1. The Problem
#  The problem formulation is very similar to POS tagging; again, you are tasked with designing and implementing an architecture that reads an english sentence and assigns each word a linguistic tag. The difference with the POS case is that the set of tags are now CCG categories, which are much more indicative of the sentence structure but also largely more numerous and sparse, hence harder to learn. CCG Supertagging is a popular ML/NLP task exactly due to its difficulty and linguistic significance; it is very likely that an architecture as simple as the one used for the previous part of the assignment will not take you very far..
# 
#  ## 2. Your task
#  Unlike previous assignments, this one is open-ended. You are allowed much more creative liberty in how you treat the problem, which means that you are also given less code to start from. Don't let this scare you; keep aware of the fact that the POS tagging and CCG supertagging only differ in scale and complexity, but are otherwise instances of the same abstraction (sequential classification). If at a loss, refer to the code of your last assignment and feel free to reuse parts of it.
#  
#  What is expected of you:
#  1. Write clean, readable and error-free code that implements:
#      * a neural sequence processing model 
#      * its training and validation functions
#      * all utilities necessary for training
#  2. Train your network as best as you can (performing model selection as needed)
#  3. Obtain your trained network's prediction for the provided test set
#  4. Submit your results using Kaggle's interface
#  
# ### Getting Started
#  Minimally, you should replicate the SRN used in the previous assignment, adapted to the current setting. To give you an idea of its expected performance, the score of a SRN trained for 20 epochs is provided in the leaderboard as benchmark.
# You are, however, encouraged to go a step further and try more advanced architectures in order to maximize your accuracy. Kaggle's leaderboard will allow you to compare your performance against that of other teams.
# The top team might be rewarded a cookie.
#  
#  
# ### Tips & Hints
# #### Model
# RNNs have limited expressive capacity that can be surpassed by more complicated variants such as [Gated Recurrent Units](https://en.wikipedia.org/wiki/Gated_recurrent_unit) and [Long Short-Term Memory Units](https://en.wikipedia.org/wiki/Long_short-term_memory); replacing your RNN with those could easily improve its performance.
# * [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
# * [Long Short-Term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)
#  
# #### Training 
# Remember that your goal is to train a model that is general and efficient enough to achieve high scores over the test set data. Take measures against overfitting: regularize with dropout and/or weight decay and keep track of the validation set performance to stop training early if needed.
#  
# #### Implementation
# Whatever your design choices are, keep the torch documentation close at all times! Do not reinvent the wheel, use existing abstractions whenever possible (but make sure you use them the right way!).
# 
# _Note_: A common trick in supertagging literature is to reduce the set of categories by removing all categories with very low frequency counts (i.e. < 10). This will inadvertedly remove your network's bility to predict them, but will also decrease the size of the softmax layer, enabling faster training. If you do choose to remove some categories, make sure that:
#  1. The ids of the kept categories are unchanged
#  2. All removed categories are mapped to the same id
# 
# For your convenience, the mapping from categories to unique ids is in descending order (in terms of occurrence counts).
#  
# #### Relevant Literature
#  1. [CCG Supertagging with a Recurrent Neural Network](https://aclweb.org/anthology/P15-2041)
#  2. [Supertagging with LSTMs](https://yonatanbisk.com/papers/2016-NAACLShort.pdf)
#  3. [LSTM CCG Parsing](https://homes.cs.washington.edu/~lsz/papers/llz-naacl16.pdf) (up to section 3)
#  
#  
#  ## 3. Kaggle: How-To
#  ### Fork this script
#  In the `notebooks` tab of the assignment, open this notebook and hit `Copy and Edit` (top right). This should create a copied version of it that you can edit and work with. While editing your script, change its name so that you can easily identify your own version.
#  ### Writing and running code
#  You can write and execute code in the exact same way as a local jupyter notebook.
#  ### Submission
#  To submit your work and receive your current score and position in the leaderboard, hit the `Commit` button (top-right corner). Wait for the code to compile and execute. Upon successful completion, press the `Open Version` dialog button. In the new window, find the `Output` tab, select your output file and hit `Submit to Competition`. After a couple of minutes you should receive your submission score and your ranking in the leaderboard. 
#  ### Scoring
#  Your score is the word-level accuracy, i.e. the overall percentage of words assigned their correct category.
#  ### Important Notes
#  * On the right sidebar, in the settings tab, set the `Docker` version to `kaggle/python from 2019-08-21` or `Original`
#  * In the `File` menu bar (top left), change the `Kernel Type` to `Script` before submission
#  * Running your script and commiting it are two different things. Remember that your output file is permanently stored and submitted for grading only after committing.
#  * If you encounter any trouble, ask away

# In[ ]:


from typing import List, Dict, Any, Tuple, Callable, Optional
from itertools import chain

import torch

import pickle

import pandas as pd


# In[ ]:


# List of training sentences
X_train: List[List[str]]
# ..their corresponding category sequences
Y_train: List[List[str]]
# Ditto for validation
X_val: List[List[str]]
Y_val: List[List[str]]
# .. and testing (but no output!)
X_test: List[List[str]]
# Mapping from categories to unique ids
cat_dict: Dict[str, int]

with open('../input/horror/data.p', 'rb') as f:
    X_train, Y_train, X_val, Y_val, X_test, cat_dict = pickle.load(f)


# In[ ]:


# Your vectorization goes here

# ..


# In[ ]:


# ..


# In[ ]:


# Your dataset and dataloader definitions go here;

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ..


# In[ ]:


# ..


# In[ ]:


# Your network goes here

class CCGSupertagger(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def predict_sentence(self, x) -> List[int]:
        """
            A function that accepts some input (a sentence and any other arguments)
            and produces a List of integers corresponding to the ids of the predicted
            categories.
            
            Note: Do NOT pad the input -- the output shape should be exactly
                equal to the length of the input sentence.
                
            Make sure this function respects the given type signature; wrong types
            may result in weird behavior, ruining your evaluation scores!
        """
        raise NotImplementedError
        
    def predict_many_sentences(self, inputs: Any) -> List[List[int]]:
        """
            A function that accepts an Iterable (a list, or perhaps a dataloader)
            of whatever `predict_sentence` expects, applies it iteratively and 
            returns the list containing the results.
        """
        return [self.predict_sentence(inp) for inp in inputs]
    
def convert_predictions_to_csv(predictions: List[List[int]]) -> None:
    """
        Takes your network's test set predictions, converts it into a 
        csv and stores it for grading.
    """
    predictions = list(chain.from_iterable(predictions))
    if len(predictions) != len(list(chain.from_iterable(X_test))):
        raise AssertionError('The number of predictions should match the number' 
                             'of words in the test set')
    df = pd.DataFrame({'id': list(range(len(predictions))), 'Category': predictions})
    df.to_csv('out.csv', index=False)


# In[ ]:


# Your utility functions go here

# ..


# In[ ]:


# Your training loop goes here, optimizer and loss_fn go here

# ..


# <div class="alert alert-block alert-warning">
# <b>Important!</b>
# The cell below should obtain your predictions over the test set in the format specified. 
# 
# **Remember**; running the cell will not automatically submit your results. You need to convert your file into a script, commit and manually submit your result file (refer to the first markdown cell for more detailed instructions)
# </div>

# In[ ]:


# Your predictions over the test set go here

predictions: List[List[int]] = NotImplemented
    
# Store your predictions into a csv for grading
convert_predictions_to_csv(predictions)

print('Done!')

