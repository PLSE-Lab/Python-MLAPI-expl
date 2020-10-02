#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#imports

import numpy as np
import pandas as pd
import nltk
import collections as co
from io import StringIO
import matplotlib.pyplot as plt
import warnings
from IPython.display import display, HTML, Markdown, display

#constants
get_ipython().run_line_magic('matplotlib', 'inline')
def printmd(string):
    display(Markdown(string))
alphaLev = .5


# In[ ]:


#load in dataset
complaintFrame = pd.read_csv("../input/consumer_complaints.csv")


# # CFPB Consumer Complaints: Language Analysis
# 
# In this notebook, I will perform EDA and language analysis on the text-sensitive data found in the [CFPB Consumer Complaints](https://www.kaggle.com/cfpb/us-consumer-finance-complaints) dataset. You can find my analysis on the non-text-sensitve EDA within [this script](https://www.kaggle.com/mmrosenb/d/cfpb/us-consumer-finance-complaints/eda-on-consumer-complaints). As we saw in some summary statistics within that script, about $432499$ observations do not have text-sensitive data, which makes this section a general down-sizing our sample. That being said, this section potentially carries the most important aspects of the consumer complaint.
# 
# We will start by pre-processing our text data. Some of the code below is adapted from [Mike Chirico's EDA](https://www.kaggle.com/mchirico/d/cfpb/us-consumer-finance-complaints/analyzing-text-in-consumer-complaints).

# In[ ]:


#consider only narrative observations
complaintNarrativeFrame = complaintFrame[complaintFrame["consumer_complaint_narrative"].notnull()]
# build a fast way to get strings
# adapted from 
# https://www.kaggle.com/mchirico/d/cfpb/us-consumer-finance-complaints/analyzing-text-in-consumer-complaints
s = StringIO()
complaintNarrativeFrame["consumer_complaint_narrative"].apply(lambda x: s.write(x))
k=s.getvalue()
s.close()
k=k.lower()
k=k.split()


# In[ ]:


# Next only want valid strings
words = co.Counter(nltk.corpus.words.words())
stopWords =co.Counter( nltk.corpus.stopwords.words() )
k=[i for i in k if i in words and i not in stopWords]
c = co.Counter(k)
printmd("We see that we $" + str(len(k)) + "$ legal word tokens in our corpus. There are $" + str(
        len(list(c.most_common())))
       + "$ legal non-stopword types in our corpus.")


# As discussed on [Mike Chirico's EDA](https://www.kaggle.com/mchirico/d/cfpb/us-consumer-finance-complaints/analyzing-text-in-consumer-complaints), `k` represents the array of all legal words with stopwords removed for the sentences concatenated, and `c` represents  a per-word counter over the legal words within our dataset. Let us take a look at the rank-frequency graph of our vocabulary, the $15$ most common words, and the $15$ least common words.

# In[ ]:


wordFrequencyFrame = pd.DataFrame(c.most_common(len(c)),columns = ["Word","Frequency"])
#plot frequency on rank
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
#freq-rank
ax1.plot(wordFrequencyFrame.index,wordFrequencyFrame["Frequency"])
ax1.set_title("Frequency on Rank of Vocabulary")
ax1.set_xlabel("Rank")
ax1.set_ylabel("Frequency")
#freq-logRank
ax2.plot(np.log(wordFrequencyFrame.index + 1),np.log(wordFrequencyFrame["Frequency"]))
ax2.set_title("Log-Frequency on Log-Rank of Vocabulary")
ax2.set_xlabel("Log Rank")
ax2.set_ylabel("Frequency")
plt.show()
printmd("_Figure 1: Frequency-Rank Graphs of Our Vocabulary._")
#get 15 most common
top15FrequencyFrame = wordFrequencyFrame.iloc[0:15,:]
display(top15FrequencyFrame)
printmd("_Table 1: The $15$ most frequent words with their frequencies_")
#get 15 least common
bottom15FrequencyFrame = wordFrequencyFrame.iloc[(wordFrequencyFrame.shape[0]-15):wordFrequencyFrame.shape[0],:]
display(bottom15FrequencyFrame)
printmd("_Table 2: The $15$ least frequent words with their frequencies_")


# We see by the log-frequency on log-rank graph (Figure 1, right) that fitting a [Zipf Distribution](https://en.wikipedia.org/wiki/Zipf%27s_law) to this graph may potentially over-prediction the probability of less frequent words occuring, which suggests that our vocabulary is much more right-skewed than in a more ideal vocabulary. We see that the 15 most common words portray words that are very financially relevant, such as credit, account, loan, and bank.
# 
# Let us now view the token-type graph to study richness of the vocabulary in the corpus.

# In[ ]:


#get token-type list
typeSet = set([]) #we will add to this over time
typeTokenList = [] #we will add tuples to this
for i in range(len(k)):
    givenToken = k[i]
    if (givenToken not in typeSet): #we should get a new type count
        typeSet.add(givenToken)
    #then add information to type-token list
    typeTokenList.append((i+1,len(typeSet)))


# In[ ]:


#then plot
typeTokenFrame = pd.DataFrame(typeTokenList,columns = ["Token Count","Type Count"])
plt.plot(typeTokenFrame["Token Count"],typeTokenFrame["Type Count"])
plt.xlabel("Token Count")
plt.ylabel("Type Count")
plt.title("Token Count on Type Count")
plt.show()
printmd("_Figure 2: Type-Token Graph for full vocabulary._")


# We see that the growth of our vocabulary begins to slow after around $1000000$ tokens in our corpus, which is about $27\%$ of the way through our corpus. To me, this suggests that the vocabulary is not extremely diverse, although it is difficult to compare without studying the relationship with other corpora.
# 
# It would be very interesting to see if the richness of vocabulary changes based on the product being addressed. Let us take a look at the distribution of products over observations with complaint narratives and the type-token graphs for each product.

# In[ ]:


productCountFrame = complaintNarrativeFrame.groupby("product")["consumer_complaint_narrative"].count()
#from pylab import *
#val = 3+10*rand(5)    # the bar lengths
pos = np.arange(productCountFrame.shape[0])+.5    # the bar centers on the y axis

plt.barh(pos,productCountFrame, align='center')
plt.yticks(pos,productCountFrame.index)
plt.xlabel('Count')
plt.ylabel("Product Type")
plt.title('Distribution of Product Type')
plt.grid(True)
plt.show()
printmd("_Figure 3: Distribution of product types._")
printmd("The number of narratives of the product type 'Other financial service' is $" + str(
        productCountFrame["Other financial service"]) + "$.")


# We see that our distribution seems very uneven, as we have a large amount of Mortgage, Debt Collection, and credit reporting narratives, but relatively few observations in money transfer, other financial services, and payday loans. This may suggest that it would be difficult to predict some of these smaller groups if we are interested in a predictive model on this issue.

# In[ ]:


#declare functions before making type-token procedures
def makeTypeTokenFrame(tokenList):
    #helper that makes our type-token frame for a given token list
    typeSet = set([]) #we will add to this over time
    typeTokenList = [] #we will add tuples to this
    for i in range(len(tokenList)):
        givenToken = tokenList[i]
        if (givenToken not in typeSet): #we should get a new type count
            typeSet.add(givenToken)
        #then add information to type-token list
        typeTokenList.append((i+1,len(typeSet)))
    return pd.DataFrame(typeTokenList,columns = ["Token Count","Type Count"])

def makeTokenList(consumerComplaintFrame):
    #helper that makes token list from the given complaint frame
    s = StringIO()
    consumerComplaintFrame["consumer_complaint_narrative"].apply(lambda x: s.write(x))
    k = s.getvalue() #gets string of unprocessed words
    s.close()
    #get actual unprocessed words
    #k = k.lower()
    k = k.split()
    k = [i for i in k if i in words and i not in stopWords] #only consider legal words
    return k

def getTokenTypeFrameForProduct(consumerComplaintFrame,productName):
    #helper that gets our token-type frame for narratives of a given product name
    #get observations with this product name
    givenProductComplaintFrame = consumerComplaintFrame[consumerComplaintFrame["product"] == productName]
    #then get token list
    tokenList = makeTokenList(givenProductComplaintFrame)
    #then make type-token frame
    return makeTypeTokenFrame(tokenList)


# In[ ]:


#run through our observations
typeTokenFrameDict = {} #we will adds to this
for productName in productCountFrame.index:
    typeTokenFrameDict[productName] = getTokenTypeFrameForProduct(complaintNarrativeFrame,productName)


# In[ ]:


cmap = plt.get_cmap('Dark2')
colorList = [cmap(i) for i in np.linspace(0, 1, len(typeTokenFrameDict))]
for i in range(len(typeTokenFrameDict)):
    productName = list(typeTokenFrameDict)[i]
    givenProductTokenTypeFrame = typeTokenFrameDict[productName]
    plt.plot(givenProductTokenTypeFrame["Token Count"],
             givenProductTokenTypeFrame["Type Count"],label = productName,
            c = colorList[i])
plt.legend(bbox_to_anchor = (1.6,1))
plt.xlabel("Token Count")
plt.ylabel("Type Count")
plt.title("Token-Type Graph\nBy Product Name")
plt.show()
printmd("_Figure 4: Token-Type Graph By Product Name._")


# This graph makes one obvious thing apparent: that there are many more mortgage complaint observations than other categories. While it is difficult to define which of these products have the richest vocabulary due to the difference in line lengths, it is very obvious that debt collection and credit reporting seem to have less rich vocabularies than the other products. This may be essential for distinguishing the groups, although it is currently difficult to tell why this is. I am open to discussion on possible hypotheses for this.
# 
# What may also be useful is to consider if we could predict whether a customer would eventually dispute the final resolution of a claim based on the language in the complaint itself. While it is likely this will also be a function of the company's decision of the company's response to the consumer, it may be interesting to study if vagueness or complexity of a dispute would factor into whether a consumer disputes a resolution.

# In[ ]:


def getTokenTypeFrameForDispute(consumerComplaintFrame,disputeLev):
    #helper that gets our token-type frame for narratives of a dispute level
    #get observations with this product name
    givenDisputeComplaintFrame = consumerComplaintFrame[consumerComplaintFrame["consumer_disputed?"] == disputeLev]
    #then get token list
    tokenList = makeTokenList(givenDisputeComplaintFrame)
    #then make type-token frame
    return makeTypeTokenFrame(tokenList)


# In[ ]:


consumerDisputeDict = {} #we will add to this
for disputeLev in complaintNarrativeFrame["consumer_disputed?"].unique():
    consumerDisputeDict[disputeLev] = getTokenTypeFrameForDispute(complaintNarrativeFrame,disputeLev)


# In[ ]:


for disputeLev in consumerDisputeDict:
    DisputeTokenTypeFrame = consumerDisputeDict[disputeLev]
    plt.plot(DisputeTokenTypeFrame["Token Count"],
             DisputeTokenTypeFrame["Type Count"],label = disputeLev)
plt.legend(bbox_to_anchor = (1.3,1))
plt.xlabel("Token Count")
plt.ylabel("Type Count")
plt.title("Token-Type Graph\nBy Whether Consumer Disputed")
plt.show()
printmd("_Figure 5: Token-Type Graph By whether the consumer disputed._")


# While it looks as though the "Yes" observations have a slightly richer vocabulary than the "No" observations, seems to be a relatively small difference.
# 
# Nonetheless, I think this would be one of the interesting prediction problems to see whether or not the language of a given consumer complaint is as much of a reason for a dispute as the company response.
# 
# ## Predicting Consumer Disputes
# 
# We will initially model our language using bag-of-words with TF-IDF encodings. This is for the sake of simplicity: if there may be another possible language model that may better represent the language-generation process for prediction sake, I may re-model the language.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#make mappable for vocabulary
counterList = c.most_common()
vocabDict = {} #we will add to this
for i in range(len(counterList)):
    vocabWord = counterList[i][0]
    vocabDict[vocabWord] = i
#make array of tf-idf counts
vectorizer = TfidfVectorizer(min_df=1,stop_words = stopWords,vocabulary = vocabDict)
unigramArray = vectorizer.fit_transform(complaintNarrativeFrame["consumer_complaint_narrative"])


# In[ ]:


#generate our language matrix
languageFrame = pd.DataFrame(unigramArray.toarray(),columns = vectorizer.get_feature_names())
printmd("The number of features extracted is $" + str(languageFrame.shape[1]) + "$.")


# Given that this is an extremely high-dimensional model, it would be useful to consider a form of dimensionality reduction on this likely sparse vocabulary. We will consider a form of principal component analysis for this purpose.

# TODO:
# 
# * Fix Color Scheme For Figure 4
# 
# * Fix $x$-ticks on Figure 2
# 
# * Fix iPython Display Objects to adjust for Kaggle Notebook structure
# 
# * Run PCA

# In[ ]:




