#!/usr/bin/env python
# coding: utf-8

# <p style="font-family:Clarendon Bold; font-size:41px; text-align:center; color:#000000">
# <span style="color:#AA2200; font-size:53px"><b>Quora</b></span>
# <b>Insincere Questions Classification</b>
# </p>
# 
# In this fairly simple notebook, we will derive some insights from the training data. Consider this as a starter! :)
# 
# **Contents**
# 
# 1. Information about the dataset<br>
# 2. Feature extraction<br>
# >2.1  Number of Words<br>
# 2.2 Number of Stop Words<br>
# 2.3 Number of Digits<br>
# 2.4 Number of Uppercase Words<br>
# 3. Basic Prepocessing <br>
#  > 3.1 Conversion of words to Lowercase <br>
#     3.2 Removing Punctuations<br>
#     3.3 Removing Stop Words<br>
#     3.4 Removing Rare Words<br>
#     
# 

# In[ ]:


#Importing Libraries
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from IPython.display import display
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("darkgrid")
print(os.listdir("../input"))


# In[ ]:


#Loading training data
train = pd.read_csv("../input/train.csv")


# ## 1. Information about the dataset

# In[ ]:


#Training data preview
train.head(10)
#I won't preview the insincere questions, as some of those are very inappropriate


# In[ ]:


#Quick info
train.info()


# In[ ]:


#Visualizing the class label count
ax = plt.figure(figsize=(9,6))
plt.title("Class Label Frequency")
sns.countplot(x="target", data=train)
sincere = len(train[train["target"] == 0])
insincere = len(train[train["target"] == 1])
print("Number of Sincere questions:", sincere)
print("Number of Insincere questions:", insincere)
print("Percentage of Sincere questions:", (sincere/len(train["target"]))*100,"%")
print("Percentage of Sincere questions:", (insincere/len(train["target"]))*100,"%")


# In[ ]:


#Removing the "qid" column
train = train.drop(["qid"], axis=1)
train.head()


# ## 2. Feature Extraction

# ### 2.1 Number of Words

# In[ ]:


features_df = pd.DataFrame() #A new dataframe for extracted features
features_df = train[["question_text", "target"]]
features_df["Word Count"] = features_df["question_text"].apply(lambda x: len(str(x).split(" ")))
display(features_df.head())
print("Average number of words per question:", np.average(features_df["Word Count"]))


# ### 2.2 Number of Stop Words

# In[ ]:


stop = stopwords.words("english") #List of all the stop words
features_df["Stop Words"] = train["question_text"].apply(lambda x: len([x for x in x.split() if x in stop]))
display(features_df.head())
print("Average number of stopwords per question:", np.average(features_df["Stop Words"]))


# ### 2.3 Number of Digits

# In[ ]:


features_df["#Digits"] = train["question_text"].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
print("Count of #Digits")
print(features_df["#Digits"].value_counts())


# <p style="text-align:center">Most of the questions don't have a digit. Only 4% of the total questions seem to have atleast a single digit</p>

# ### 2.4 Number of Uppercase words

# In[ ]:


features_df["Uppercase Words"] = train["question_text"].apply(lambda x: len([x for x in x.split() if x.isupper()]))
print("Average Number of Uppercase words:", np.average(features_df["Uppercase Words"]))
print("\nCount of Number of Uppercase words:")
print(features_df["Uppercase Words"].value_counts())
#Boxplot
plt.figure(figsize=(20,4))
sns.boxplot(features_df["Uppercase Words"])
#Histogram
plt.figure(figsize=(19,5))
sns.distplot(features_df["Uppercase Words"], kde=False)


# <p style="text-align:center">About 20.7% of the questions have one uppercase letter.</p>

# ## 3. Basic Preprocessing

# ### 3.1 Conversion of words to Lowercase

# In[ ]:


train["question_text"] = train["question_text"].apply(lambda x: " ".join(x.lower() for x in x.split()))
print("Conversion Done!")


# ### 3.2 Removing Punctutations

# In[ ]:


train["question_text"] = train["question_text"].apply(lambda x: re.sub(r'[^\w\s]','',x))
print("Punctuations Removed!")


# ### 3.3 Removing Stop Words

# In[ ]:


train["question_text"] = train["question_text"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
print("Stop Words Removed!")


# ### 3.4 Removing Rare words

# In[ ]:


#Selecting last 200000 words to least occur
rare_words = pd.Series(" ".join(train["question_text"]).split()).value_counts()[-200000:]
rare_words.head()


# In[ ]:


train["question_text"] = train["question_text"].apply(lambda x: " ".join(x for x in x.split() if x not in rare_words))
print("Rare words removed!")

