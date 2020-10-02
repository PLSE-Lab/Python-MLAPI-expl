#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # A couple of words to start with

# Hello, everyone! I'm Ivan Lipatov, 3rd year student of Higher School of Economics from Russia. I'm deeply interested in ML and DL topics and recently I've started my way up to the top on Kaggle. At time of posting this notebook I'm taking part in Jigsaw Multilingual Toxic Comment Classification and already have reached quite high results. That is why I decided to post a series of notebooks that show a sequential process of investigation of the competition problem. The aim of my work is not to teach Kaggle grandmasters how to win competitions, but to help those, who are only at the beginning of their Kaggle way, because as I've learned the most difficult part is to start :) 
# 
# (don't think that next stages of Ml research will be free lunch for you - there will be much work to do, but as far as you plunge into competition spirit it gets much easier to make your way through)

#  # What data do we face

# The goal of competition is to classify toxic comments in non-english languages having only english training data

# In[ ]:


#define global variables
DATA_PATH = "../input/jigsaw-multilingual-toxic-comment-classification"
small_ds_path = "jigsaw-toxic-comment-train.csv"
large_ds_path = "jigsaw-unintended-bias-train.csv"
val_ds_path = "validation.csv"
test_ds_path = "test.csv"


# In[ ]:


#download the data
small_ds = pd.read_csv(os.path.join(DATA_PATH, small_ds_path), usecols=["id", "comment_text", "toxic"])
large_ds = pd.read_csv(os.path.join(DATA_PATH, large_ds_path), usecols=["id", "comment_text", "toxic"])
val_ds = pd.read_csv(os.path.join(DATA_PATH, val_ds_path))
test_ds = pd.read_csv(os.path.join(DATA_PATH, test_ds_path))


# We have 4 datasets provided:
# 
# 1) ~224k examples train dataset with english comment and binary target label (either 0 or 1)
# 
# 2) ~1,87M examples train datset with english commentsand  target label varying berween 0 and 1 - estimation of probability to be target
# 
# 3) 8k examples validation dataset with data in several non-english languages
# 
# 4) ~63k examples test data with data in several non-english languages
# 
# 
# Let's take a more detailed look to each piece of the data

# # Small train dataset

# In[ ]:


small_ds.head()


# In[ ]:


vals = small_ds.toxic.value_counts()
sns.barplot(vals.index, vals.values)
plt.title("Non-toxic vs toxic occurence in data")
plt.ylabel("Number exmaples")
plt.xlabel("Target value")


# So here is the first important insight - we face unbalanced classes problem - in following notebooks I will show to tackle it

# In[ ]:


toxic_examples = small_ds[small_ds["toxic"] == 1].sample(5, random_state=42)["comment_text"]
for comment in toxic_examples.values:
    print("Next comment:")
    print(comment)


# In[ ]:


#generate wordcloud to get more intuition about toxicity
from wordcloud import WordCloud
toxic_comments = " ".join(small_ds[small_ds["toxic"]==1]["comment_text"].values)
wc = WordCloud().generate(toxic_comments)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")


# Although it is not the most enjoyable actitivity, you can get more intuition about what means toxic by reading more comments. My own summary is that toxic means that a comment is written with offensive sense towards someone or something, often with profanity.

# In[ ]:


non_toxic_examples = small_ds[small_ds["toxic"] == 0].sample(5, random_state=42)["comment_text"]
for comment in non_toxic_examples.values:
    print("Next comment")
    print(comment)


# While non toxic comment is the ordinal speech about some topic - it seems quite understandable

# In[ ]:


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
small_ds["num_words"] = small_ds["comment_text"].str.split().apply(len)
temp_ds = small_ds[small_ds["num_words"] < 500]
sns.violinplot(x="toxic",y="num_words", data=temp_ds, ax=ax1)
ax1.set_title("Distributions of number of words/sentences in toxic/nontoxic comments")

small_ds["num_sents"] = small_ds["comment_text"].str.split(".").apply(len)
temp2_ds = small_ds[small_ds["num_sents"] < 100]
sns.violinplot(x="toxic",y="num_sents", data=temp2_ds, ax=ax2)
#ax2.set_title("Distribution of number of sentences in toxic/nontoxic comments")


# In[ ]:


print("Number of words descriptive stats")
print(small_ds["num_words"].describe())
print()
print("Number of sentences descriptive stats")
print(small_ds["num_sents"].describe())


# Also for training we have larger dataset with comments, though toxicity there is distributed between 0 and 1.

# In[ ]:


large_ds.head()


# In[ ]:


t = large_ds.toxic.round(1)
t.value_counts()


# In[ ]:


sns.barplot(t.value_counts().index, t.value_counts().values)

plt.ylabel("Num samples")
plt.xlabel("Probability of being toxic")


# So we see that there are lots of totaly non-toxic comments and a few probably toxic - approximately the same we've seen in the small datasets - so the problem of unbalanced classes occurs again

# From this data we need to decide which parts we can use as toxic, and which as non-toxic for training, as we need labels to be either 0 or 1

# In[ ]:


large_ds["rounded_toxic"] = large_ds.toxic.round(1)
maybe_toxic = large_ds[(large_ds.rounded_toxic == 0.5) | (large_ds.rounded_toxic == 0.6)].comment_text
probably_toxic = large_ds[(large_ds.rounded_toxic == 0.7) | (large_ds.rounded_toxic == 0.8)].comment_text
surely_toxic = large_ds[(large_ds.rounded_toxic == 0.9) | (large_ds.rounded_toxic == 1.0)].comment_text


# In[ ]:


# may be toxic examples

for comm in maybe_toxic.sample(3):
    print(comm)
    print()


# In[ ]:


maybe_toxic_comments = " ".join(maybe_toxic.values)
wc = WordCloud().generate(maybe_toxic_comments)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")


# In[ ]:


#probably toxic examples
for comm in probably_toxic.sample(3):
    print(comm)
    print()


# In[ ]:


probably_toxic_comments = " ".join(probably_toxic.values)
wc = WordCloud().generate(probably_toxic_comments)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")


# In[ ]:


#surely toxic examples
for comm in surely_toxic.sample(3):
    print(comm)
    print()


# In[ ]:


surely_toxic_comments = " ".join(surely_toxic.values)
wc = WordCloud().generate(surely_toxic_comments)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")


# We see that toxic == 0.5 is quite different from toxic == 1. Although, from examples it can be seen that toxic >= 0.5 are more likely toxic than non-toxic, from my point of view. Of course, it is still questionable which boarder is the best, but I decide to take 0.5. However, to give to the future model the information that originally not all the labels are straight zeros and ones we can apply technique called labels smoothing. The goal of this technique is to force the model to be less confident in its predictions - so to make resulting predicted probabilites closer to 0.5 than to 0 and 1. This can be reached by assigning downsizing labels == 1 and upsizing labels == 0 in training process due to some distribution for example uniform. 
# 
# Here is the nice guide to label smoothing https://towardsdatascience.com/label-smoothing-making-model-robust-to-incorrect-labels-2fae037ffbd0

# Test and Validation datasets

# In[ ]:


val_ds.head()


# In[ ]:


len(val_ds)


# In[ ]:


vs = val_ds.lang.value_counts()
sns.barplot(vs.index, vs.values)
plt.xlabel("language")
plt.ylabel("Number of samples")


# In[ ]:


ts = val_ds.toxic.value_counts()
sns.barplot(ts.index, ts.values)
plt.xlabel("Non-toxic vs Toxic")
plt.ylabel("Number of samples")


# In validation dataset we have data in 3 languages: turkish, spanish and italian - approximately the same number of samples for each
# 
# Class proportions are simillar to the training data we have, however train data differs from validation significantly because of the language. Nevertheless, it's okay as far as we want metrics on validation reflect the quality of the model on the test data. And as you can see from my following notebooks - results on validation and test are very close, which is very important for monitoring model quality while training

# In[ ]:


test_ds.head()


# In[ ]:


len(test_ds)


# In[ ]:


vc = test_ds.lang.value_counts()
sns.barplot(vc.index, vc.values)
plt.xlabel("language")
plt.ylabel("Number of samples")


# Overall, there are three main insights from the data, which can help to create the better model in the future.
# 
# 1. Data is multilingual that is why it's rational to prefer those models that can handle multilingual input data and can take out the semantics not depending on language as we train on english , test on non-english
# 2. Classes are unbalanced - which can cause a significant problem for model to converge. That is why, it is necessary to apply some methods to handle it
# 3. Validation dataset can be a good reflection of the test dataset, that is why we can rely on metrics computed on it

# These are my following notebooks:
# 
# 1. https://www.kaggle.com/vgodie/first-baseline - there I'm considering different methods to approach the problem and show to set up zero baseline with BERT architecture
# 2. https://www.kaggle.com/vgodie/class-balancing - there I share my ideas on how to fight unbalanced classes problem
# 3. https://www.kaggle.com/vgodie/data-encoding - there I show how to make custom preprocessing data for more sophisticated models such as XLM-Roberta, however this notebook can be used as a template for the preprocessing for any transformer model
# 4. https://www.kaggle.com/vgodie/xlm-roberta - and finally I build and train my best model with all data preparations and techniques discussed in the previous notebook

# In[ ]:




