#!/usr/bin/env python
# coding: utf-8

# Forked from "Benchmark Kernel" 

# # A Thought Provoking Kernel for your Competition Consideration
# ## Introduction
# In April of 2019 legislators in the United States introduced a bill called the "Algorithmic Accountability Act of 2019" (AAA19). This Kernel seeks to understand the contents of that proposed new law in the context of this data set and goals, and possibly provide an advantage in winning the competition.
# 
# One of many news articles about the bill can be found here: https://techcrunch.com/2019/04/10/algorithmic-accountability-act/
# ## Details of the proposal
# (Disclaimer: These are my interpretations only, please read it for yourself). The proposal is based on the fact that there will be an Impact Assessment on every automated decision system that falls under the new law. Not every such system will fall under this new law.
# * Impact Assessment - ... a study evaluating an automated decision system and ... the development process, including the design and training data ... for impacts on accuracy, fairness, bias, discrimination, privacy, and security.
# * The above also includes a provisions for consumers to have access to the results ... and may correct or object to its results.
# * Covered by this law - Only systems that are run by a person or company making more than 50M US Dollars/year, or has more than 1M users, or is in the business of data analysis
# * High Risk systems are also targeted - where the definition of this includes 4 major areas that pose a significant risk to A, B, C, and D
# * High Risk A - privacy or security of personal information and/or contributing to inaccurate, unfair, biased, or discriminatory decisions.
# * High Risk B - extensive evaluation of peoples work performance, economic situation, health, preferences, behavior, location, or movements.
# * High Risk C - information about race, color, national origin, political opinions, religion, trade union membership, genetic data, biometric data, health, gender, gender identity, sexuality, sexual orientation, criminal convictions, or arrests.
# * High Risk D - uses systematic monitoring of a large publicly accessible physical place
# 
# Quite a bit of ground!
# ## This Competition
# This Kernel seeks to use Word Clouds to analyze the [Competition Evaluation](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation) in light of this proposed legislation. In addition, the Word Clouds provide insight into how the performance in the competition may be improved.

# # Environment Setup
# From the "Benchmark Kernel" provided as part of this competition.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/glove-global-vectors-for-word-representation"))
print(os.listdir("../input/jigsaw-unintended-bias-in-toxicity-classification"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import pandas as pd
import numpy as np
import pkg_resources
import seaborn as sns
import time
import scipy.stats as stats

from sklearn import metrics
from sklearn import model_selection

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.models import Model
from keras.models import load_model


# In[ ]:


# Plotly imports
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# Other imports
from collections import Counter
from scipy.misc import imread
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from wordcloud import WordCloud, STOPWORDS

import nltk
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm


# ## Load and pre-process the data set

# In[ ]:


train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
print('loaded %d records' % len(train))


# In[ ]:


# Make sure all comment_text values are strings
train['comment_text'] = train['comment_text'].astype(str) 

# List all identities
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

# Convert taget and identity columns to booleans
def convert_to_bool(df, col_name):
    df[col_name] = np.where(df[col_name] >= 0.5, True, False)
    
def convert_dataframe_to_bool(df):
    bool_df = df.copy()
    for col in ['target'] + identity_columns:
        convert_to_bool(bool_df, col)
    return bool_df

train = convert_dataframe_to_bool(train)


# In[ ]:


# Make a python dictionary with unique words and their respective word counts
def make_dict(words):
    counts = dict()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts


# In[ ]:


# Ignore these stopwords
stopwords = nltk.corpus.stopwords.words('english')
print("example stopwords", stopwords[0:10])
lemm = WordNetLemmatizer()
print("The lemmatized form of leaves is: {}".format(lemm.lemmatize("leaves")))


# In[ ]:


def delta_cond(cond1, cond2, cond1_name, cond2_name) :
    # make them all lowercase, and remove the stopwords
    scratch = train[cond1]["comment_text"].values
    scratch = " ".join(scratch)
    scratch = scratch.split()
    # print("remove stopwords and non-alphabetic then make everything lowercase and Lemmatize the results")
    cond1_words = [lemm.lemmatize(word.lower()) for word in scratch if (word.lower() not in stopwords) & word.isalpha()]
    del scratch
    # print("make a dictionary of unique ocurrances, and their counts")
    cond1_words = make_dict(cond1_words)
    
    # make them all lowercase, and remove the stopwords
    scratch = train[cond2]["comment_text"].values
    scratch = " ".join(scratch)
    scratch = scratch.split()
    # print("remove stopwords and non-alphabetic then make everything lowercase and Lemmatize the results")
    cond2_words = [lemm.lemmatize(word.lower()) for word in scratch if (word.lower() not in stopwords) & word.isalpha()]
    del scratch
    # print("make a dictionary of unique ocurrances, and their counts")
    cond2_words = make_dict(cond2_words)
    
    # create a python dictionary of the words in one dictionary but not the other
    cond1_not_cond2 = { k : cond1_words[k] for k in set(cond1_words) - set(cond2_words) }
    cond2_not_cond1 = { k : cond2_words[k] for k in set(cond2_words) - set(cond1_words) }
    print("",len(cond1_words),"Words found in", cond1_name,"\n", len(cond2_words), 
          "Words found in", cond2_name)
    print("",len(cond1_not_cond2),"Words found Only in",cond1_name,"and not in",cond2_name,"\n",
          len(cond2_not_cond1), "Words found Only in",cond2_name,"and not in",cond1_name,"\n")
    del cond1_words
    del cond2_words
    
    title = "Words found Only in " + cond1_name + " and not in " + cond2_name
    plt.figure(figsize=(16,13))
    wc = WordCloud(background_color="black", max_font_size= 40)
    wc.generate_from_frequencies(cond1_not_cond2)     
    plt.title(title, fontsize=20)
    plt.imshow(wc.recolor( colormap= 'Pastel2' , random_state=17), alpha=0.98)
    plt.axis('off')

    title = "Words found Only in " + cond2_name + " and not in " + cond1_name
    plt.figure(figsize=(16,13))
    wc = WordCloud(background_color="black", max_font_size= 40)
    wc.generate_from_frequencies(cond2_not_cond1)     
    plt.title(title, fontsize=20)
    plt.imshow(wc.recolor( colormap= 'Pastel2' , random_state=17), alpha=0.98)
    plt.axis('off')
    
    del cond1_not_cond2
    del cond2_not_cond1
    del title
    del wc
    
    return 


# # Word Cloud Visualizations
# The word cloud [(sometimes called a Tag Cloud or Weighted List)](https://en.wikipedia.org/wiki/Tag_cloud) is a popular data visualization mechanism that will be used below.
# 
# ## Preprocessing
# * In all of the cases below, the data is first subdivided by criteria (ex: Toxic is either True or False, based on the "target" column being >= 0.5 or not, respectively). 
# * After gathering the desired subset, we divide sentences into individual words.
# * Each word is then made into lower case, and "english" nltk stopwords are removed. Only lemmatized and alphabetic words are used.
# * Once a subset of words is so created, it is converted into a dictionary. A dictionary contains a list of words and their respective word counts.
# * Finally, two dictionaries are compared to find words that are present in one dictionary but not the other.
# 
# ## The First Wordclouds - Toxic versus Not Toxic
# The first two wordclouds below show all the words that come from "toxic" comment_texts, but that are not present in "non toxic" comment_texts; and vice versa.

# In[ ]:


c1 = ((train.target==True))
c1_name = "TOXIC"
c2 = ((train.target==False))
c2_name = "NOT TOXIC"
delta_cond(c1, c2, c1_name, c2_name)


# In[ ]:





# # Discussion of the above
# ## Take a close look at the two wordclouds above
# Here are some points to consider:
# ### Implications of AAA19
# * If we consider the data set used for training - does it really contain enough information such that words in the first wordcloud really represent words that are ONLY TYPICALLY used in Toxic Comment_Texts in general, or do they instead implicate a possible problem relating to situation "High Risk C" - information about race, color, national origin, political opinions, religion, trade union membership, genetic data, biometric data, health, gender, gender identity, sexuality, sexual orientation, criminal convictions, or arrests?
# * How might we scrub such information so that the first wordcloud only contains words typically used in toxic comment_texts, and that do not contain unnecessary words?
# * If this result were examined by a lawmaker, how might they perceive this particular data set with regard to impacts on accuracy, fairness, bias, discrimination, privacy, and security?
# 
# ### Implications on this Competition
# * With regard to the last statement above, we really care more about "Accuracy" from a perspective of the rules of the Competition Evaluation.
# * Specifically for this Kernel, we want to look at issues with the original "Benchmark Kernel" and think of how they might be improved.
# * Take a look at the total word counts also (printed as text above the wordclouds). Do we have enough slack to cull from the dictionaries and still classify with consistent accuracy? The answer may be yes, but as we examine the "other AUC's" below, the data gets considerably slimmer.
# 
# # Additional Wordcloud Pairs
# In the two wordclouds above, we talked about the overall accuracy of the toxic classifier (overall Area Under the Curve metric, or AUC). In the below software, we examine the issues with the BenchMark Kernel for the other AUC areas mentioned in the Competition Evaluation rules. Specifically, we will create WordCloud pairs for the three worst scores (from the original Benchmark Kernel) in the areas of "Subgroup AUC," "BPSN (Background Positive, Subgroup Negative) AUC," and "BNSP (Background Negative, Subgroup Positive) AUC."
# 

# ## What to look for in the below
# First, you should perform an overall sanity check. Look at the pair of wordclouds. Could you, as a human reader, pick which cloud was which from the pair of subdivided data? Many times the answer is "yes" but some pairs are more obvious than others.
# 
# Next, look at the specifics of the pairs:
# * Are the words in the wordclouds, especially the larger font words, really representative of words that you would expect to see only in one subset and not the other? Why or why not?
# * If you really don't like a bunch of the words, imagine that you (or machine learning) removes them from the list. Do the word counts justify enough remaining words to do the classification?
# * What other techniques might you use besides these "subtracted dictionaries" to subdivide the space more accurately? 
# * LSTM and other recurrent time varying models focus on the sequence of words, not just the individual words. Even a Hidden Markov Model could be used as a probability sequencer. How might you sanity check the "sequence dictionaries" just as the wordclouds helped you sanity check the word subsets.

# # Poorly scored AUC Subgroups
# ## Subgroup AUC: 
# Here, we restrict the data set to only the examples that mention the specific identity subgroup. A low value in this metric means the model does a poor job of distinguishing between toxic and non-toxic comments that mention the identity.
# 
# In a prior iteration of this Kernel, the AUC for specific subgroups scored the lowest, meaning that comment texts that were identified as having to do with those subgroups were poorly classified as being toxic or not toxic. Let's look at the word clouds for words only in the toxic and only in the not toxic comment texts relating to those subgroups. 
# 
# The three most poorly scored subgroups are black, white, and homosexual_gay_or_lesbian

# ## Subgroup AUC: - Black

# In[ ]:


c1 = ((train.target==True) & (train.black==True))
c1_name = "(toxic TRUE) + (black TRUE)"
c2 = ((train.target==False) & (train.black==True))
c2_name = "(toxic FALSE) + (black TRUE)"
delta_cond(c1, c2, c1_name, c2_name)


# ## Subgroup AUC: - White

# In[ ]:


c1 = ((train.target==True) & (train.white==True))
c1_name = "(toxic TRUE) + (white TRUE)"
c2 = ((train.target==False) & (train.white==True))
c2_name = "(toxic FALSE) + (white TRUE)"
delta_cond(c1, c2, c1_name, c2_name)


# ## Subgroup AUC: - Homosexual_gay_or_lesbian

# In[ ]:


c1 = ((train.target==True) & (train.homosexual_gay_or_lesbian==True))
c1_name = "(toxic TRUE) + (homosexual_gay_or_lesbian TRUE)"
c2 = ((train.target==False) & (train.homosexual_gay_or_lesbian==True))
c2_name = "(toxic FALSE) + (homosexual_gay_or_lesbian TRUE)"
delta_cond(c1, c2, c1_name, c2_name)


# # Poorly scored AUC Subgroups
# ## BPSN (Background Positive, Subgroup Negative) AUC: - homosexual_gay_or_lesbian
#  Here, we restrict the test set to the non-toxic examples that mention the identity and the toxic examples that do not. A low value in this metric means that the model confuses non-toxic examples that mention the identity with toxic examples that do not, likely meaning that the model predicts higher toxicity scores than it should for non-toxic examples mentioning the identity.
# 
# In a prior iteration of this Kernel, the UAC for specific subgroups scored the lowest, meaning that comment texts that were identified as having to do with those subgroups were poorly classified as being toxic or not toxic. Let's look at the word clouds for words only in the toxic and only in the not toxic comment texts relating to those subgroups. 
# 
# The three most poorly scored subgroups are black, homosexual_gay_or_lesbian, and white

# ## Poorly scored AUC Subgroup for BPSN
# ## homosexual_gay_or_lesbian

# In[ ]:


c1 = ((train.target==False) & (train.homosexual_gay_or_lesbian==True))
c1_name = "(toxic FALSE) + (homosexual_gay_or_lesbian TRUE)"
c2 = ((train.target==True) & (train.homosexual_gay_or_lesbian==False))
c2_name = "(toxic TRUE) + (homosexual_gay_or_lesbian FALSE)"
delta_cond(c1, c2, c1_name, c2_name)


# ## Poorly scored AUC Subgroup for BPSN
# ## black

# In[ ]:


c1 = ((train.target==False) & (train.black==True))
c1_name = "(toxic FALSE) + (black TRUE)"
c2 = ((train.target==True) & (train.black==False))
c2_name = "(toxic TRUE) + (black FALSE)"
delta_cond(c1, c2, c1_name, c2_name)


# ## Poorly scored AUC Subgroup for BPSN
# ## white

# In[ ]:


c1 = ((train.target==False) & (train.white==True))
c1_name = "(toxic FALSE) + (white TRUE)"
c2 = ((train.target==True) & (train.white==False))
c2_name = "(toxic TRUE) + (white FALSE)"
delta_cond(c1, c2, c1_name, c2_name)


# # Poorly scored AUC Subgroups
# ## BNSP (Background Negative, Subgroup Positive) AUC:  - Christian
# Here, we restrict the test set to the toxic examples that mention the identity and the non-toxic examples that do not. A low value here means that the model confuses toxic examples that mention the identity with non-toxic examples that do not, likely meaning that the model predicts lower toxicity scores than it should for toxic examples mentioning the identity.
# 
# In a prior iteration of this Kernel, the UAC for specific subgroups scored the lowest, meaning that comment texts that were identified as having to do with those subgroups were poorly classified as being toxic or not toxic. Let's look at the word clouds for words only in the toxic and only in the not toxic comment texts relating to those subgroups. 
# 
# The three most poorly scored subgroups are christian, female, and jewish

# ## BNSP: Christian

# In[ ]:


c1 = ((train.target==True) & (train.christian==True))
c1_name = "(toxic TRUE) + (christian TRUE)"
c2 = ((train.target==False) & (train.christian==False))
c2_name = "(toxic FALSE) + (christian FALSE)"
delta_cond(c1, c2, c1_name, c2_name)


# ## BNSP: Jewish

# In[ ]:


c1 = ((train.target==True) & (train.jewish==True))
c1_name = "(toxic TRUE) + (jewish TRUE)"
c2 = ((train.target==False) & (train.jewish==False))
c2_name = "(toxic FALSE) + (jewish FALSE)"
delta_cond(c1, c2, c1_name, c2_name)


# ## BNSP: Female

# In[ ]:


c1 = ((train.target==True) & (train.female==True))
c1_name = "(toxic TRUE) + (female TRUE)"
c2 = ((train.target==False) & (train.female==False))
c2_name = "(toxic FALSE) + (female FALSE)"
delta_cond(c1, c2, c1_name, c2_name)


# # Will not re-run "Benchmark Kernel" 
# ## Because, combined with word clouds, it overruns time limit
# ## but here is the results (apologies for the non-tabular view)
# 
# * ____bnsp_auc___bpsn_auc___subgroup						subgroup_auc___subgroup_size
# * 6___0.956725___0.766774___black___________________________0.801189_______2890
# * 2___0.950133___0.786869___homosexual_gay_or_lesbian_______0.803491_______2250
# * 5___0.947570___0.803467___muslim__________________________0.807359_______4210
# * 7___0.960967___0.769693___white___________________________0.816946_______4931
# * 4___0.922565___0.856830___jewish__________________________0.830503_______1554
# * 8___0.953574___0.838308___psychiatric_or_mental_illness___0.872289_______959
# * 1___0.939502___0.871237___female__________________________0.877651_______10710
# * 0___0.946130___0.862624___male____________________________0.881211_______8861
# * 3___0.915157___0.913734___christian_______________________0.892654_______8052
