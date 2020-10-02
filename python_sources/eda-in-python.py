#!/usr/bin/env python
# coding: utf-8

# # EDA of Personalised Medicine: Redefining Cancer Treatment

# ## 1.0. About

# **About**
# - There are 9 different genetic mutations that we can classify upon
# - GENE VALUES: The gene where a mutation is located
# - VARIARION: The amino acid change for the mutation
# - TEXT: The clinical evidance used to classify these mutations
# - In the given data `A Gene and Variation pair is unique`[refer here](https://www.kaggle.com/c/msk-redefining-cancer-treatment/discussion/35336#197792)
# - Once sequenced, a cancer tumor can have thousands of genetic mutations. But the challenge is distinguishing the mutations that contribute to tumor growth (drivers) from the neutral mutations (passengers).Currently this interpretation of genetic mutations is being done manually. This is a very time-consuming task where a clinical pathologist has to manually review and classify every single genetic mutation based on evidence from text-based clinical literature.
# - more about genetic mutations, [refer here](https://www.youtube.com/watch?v=qxXRKVompI8)

# **Workflow of classifying genetic mutations**
# -  molecular pathologist selects a list of genetic variations of interest that he/she want to analyze
# - The molecular pathologist searches for evidence in the medical literature that somehow are relevant to the genetic variations of interest
# - Finally this molecular pathologist spends a huge amount of time analyzing the evidence related to each of the variations to classify them
# 
# Our goal here is to replace step 3 by a machine learning model. The molecular pathologist will still have to decide which variations are of interest, and also collect the relevant evidence for them. But the last step, which is also the most time consuming, will be fully automated

# ### 1.1. All imports

# In[ ]:


# all imports
import numpy as np #linear algebra
import pandas as pd # data processing I/O

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

import re

import matplotlib.pyplot as plt #plots
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns#advanced plots

import warnings
warnings.filterwarnings("ignore")#ignore warnings

from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
import collections


# ### 1.4. Basic analysis

# In[ ]:


tr_var = pd.read_csv('../input/msk-redefining-cancer-treatment/training_variants', sep=',')
tr_var.head()


# In[ ]:


tr_text = pd.read_csv('../input/msk-redefining-cancer-treatment/training_text', sep='\|\|',engine="python",names=["ID","TEXT"],skiprows=1)
tr_text.head(1)


# ### 1.5. Summary of Dataset

# In[ ]:


tr_var.describe(include='all')


# - There are 264 unique Genes
# - There are 2996 unique Variations of mutations
# - Total 3321 rows

# In[ ]:


tr_text.describe(include='all')


# - there are roughly 1920 unique text fields
# - The most repeated one beeing repeated 53 times

# ### 1.6. Finding null values

# In[ ]:


print('Is there any null values in training variants:', tr_var.isnull().values.any())
print('Is there any null values in training text:', tr_text.isnull().values.any())


# In[ ]:


print("get indexes of null values in all training text")
print(tr_text.loc[tr_text.isnull().any(axis=1)])


# **Q: Why we are given empty values in text data?What is the best thing to do in such cenarios?**
# 
# **Answer:** The empty values might add robustness to the data, but it is given that analysing text is important for prediction of class and hence, for now I decide that, It is good idea to leave such rows with empty text data. This assumption might change in future, as we analyse more features.

# ## 2.0. Univariant Analysis

# ### 2.1. Number of Classes

# In[ ]:


plt.figure(figsize=(12,8))
plt.title("classes distribution: training Data")
#(1-9 the class this genetic mutation has been classified on)
sns.countplot(x="Class", data=tr_var)
plt.xlabel("Classes")
plt.show()


# - Given Train dataset is unbalanced
# - Class 7 is the most dominating one
# - Class 8, 9, 3 are relatively very small

# ### 2.2. Gene 

# - As we know from the summary we have total 264 different genes.
# 
# let's view the rough picture of frequency of these genes

# In[ ]:


plt.figure(figsize=(12,8))
plt.title("classes distribution: training Data")
#(1-9 the class this genetic mutation has been classified on)
sns.countplot(x="Gene", data=tr_var)
plt.xlabel("Classes")
plt.show()


# - we can observe that few classes are super dominant
# 
# let's try to understand things little better by viewing frequency distribution plot

# In[ ]:


plt.figure(figsize=(12,8))
plt.title('frequency dist of gene')
sns.distplot(sorted(tr_var["Gene"].value_counts().tolist(), reverse=True), hist = False, kde_kws=dict(cumulative=True))
plt.xlabel('Gene')
plt.grid()
plt.minorticks_on()
plt.grid(b=True, which='minor', color='r', linestyle='--')
plt.show()


# - 20 types of genes constitutes the 80% total counts
# - 50 types of genes constitute 87.5% of total counts
# 
# let's find out what are these top 20 genes

# In[ ]:


tr_var["Gene"].value_counts()[:20]


# - Top five genes are super dominant with BRCA1 leading the list

# ### 2.3. Variation of Mutation

# - From the summary of the data we can say that there are total 2996 unique variations
# - It was given that total 3321 rows exist and combination of gene and variation is unique
# - It means many variations only appeared once in the entire train dataset

# ## 3.0. Bivariant Analysis: Interaction between features

# ### 3.1. Gene vs Class

# **let's look at most frequent genes (top 20)vs Classes**

# In[ ]:


def selectTopGene(top):
  tmp_list = tr_var["Gene"].value_counts()[:top].index.tolist()
  tmp_df = tr_var[tr_var['Gene'].isin(tmp_list)]
  #print(tmp_df)
  return tmp_df


# In[ ]:


def drawClassVsTopGeneFacet(top):
  tmp_df = selectTopGene(top)
  fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(18,15))
  for i in range(3):
    for j in range(3):
      tmp_df_1 = tmp_df[tmp_df['Class'] == ((i*3+j)+1)].groupby('Gene').size().reset_index(name='counts')
      tmp_df_1 = tmp_df_1.sort_values('counts', ascending=False)
      tmp_df_top_7 = tmp_df_1[:]
      axs[i][j].set_title('for Class ' + str((i*3+j)+1))
      plt.sca(axs[i][j])
      plt.xticks(rotation=30)
      sns.barplot(x="Gene", y="counts", data=tmp_df_top_7, ax=axs[i][j])


# In[ ]:


drawClassVsTopGeneFacet(20)


# 
# - For Class 8 and 9 EGFR, ERBB2 and TP53 are the most dominating ones
# - EGFR is dominating in class 8, 7, 2
# - TP53 is most dominating in 1, 9
# - BRCA1 is dominating in class 3 and 5
# - PTEN is dominating in class 4
# - For Class 8 and 9 top 20 Genes appeared only once, it suggests class 8 and 9 are most dissimilar from others or from the total counts of class 8 and 9 are severly underrepresented

# ## 4.0. Text Analysis

# - It was given that text plays crusial role in predicting the class.
# 
# **Let's look at few text fields to get the abstract idea of how the text looks like**

# In[ ]:


tr_text.columns


# In[ ]:


print(tr_text['TEXT'].iloc[0])


# In[ ]:


print(tr_text['TEXT'].iloc[1])


# - There are multiple abstracts of different research papers involved in each text field.
# - Just like any other scientific papers, it involves references to other papers (ex: (1), (2)), references to figures, ex: (fig: 2), short forms and full forms in braces. I think the usefullness of such words or texts is limited, however my knowledge is limited.
# - There are words like CDK10, EBF (witho out paranthesis),which, I think are refering to a particular process or a particular thing in medical terminology, I think they are going to play crusial role in predicting Classes

# **Here the goal is to analyse the text and try to extract some features from them**

# ### 3.1. Word Cloud of All text docs:
#  to understand the frequent words

# In[ ]:


def plotWordCloud():
  combText = tr_text['TEXT'].agg(lambda x: ' '.join(x.dropna()))
  wordcloud = WordCloud().generate(combText)
  # Display the generated image:
  print("word cloud for text ")
  plt.figure(figsize=(12,8))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  plt.show()


# In[ ]:


plotWordCloud()


# - wild, type, cells, expressing, kinase, domain,breast, cancer, activity, supplementary, et, al, line, aminoe, acid, Figure, missense, mutation, tyrosine, gene etc are most frequent words.
# - Here, In this particular problem, we are trying predict the class and hence it is  common sensical to have our features to as much dissimilar as possible, and hence I think it good idea to remove such repeating words.
# 
# 
# let's do little preprocessing of text by removing some stop words and special characters

# ### 3.2. Text Preprocessing: remove stopwords

# In[ ]:


stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
stemmer = SnowballStemmer("english")


# In[ ]:


def removeStopWords(sentence):
  sentence = sentence.replace('\\r', ' ')
  sentence = sentence.replace('\\"', ' ')
  sentence = sentence.replace('\\n', ' ')
  sentence = re.sub('\(.*?\)', ' ', sentence)
  sentence = re.sub('[^A-Za-z0-9]+', ' ', sentence)
  list_of_words = [i.lower() for i in wordpunct_tokenize(sentence) if i.lower() not in stop_words]
  list_of_words = [stemmer.stem(i.lower()) for i in list_of_words if stemmer.stem(i.lower()) not in stop_words]
  sentence = ' '.join(list_of_words).lower().strip()

  return sentence


# In[ ]:


stop_words.update(['line', 'fig','figure', 'author','find',
                   'et', 'al', 'evaluate', 'show', 'demonstrate', 'conclusion', 'study', 'analysis', 'method'])


# In[ ]:


def preProcessText():
  tmp_sen = []
  for i in tqdm(tr_text['TEXT']):
    i = removeStopWords(i)
    tmp_sen.append(i)
  
  tr_text['TEXT'] = tmp_sen


# In[ ]:


tr_text = tr_text.replace(np.nan, '', regex=True)
preProcessText()
plotWordCloud()


# ### 3.3.Combine Text document with variant documet

# In[ ]:


df = tr_var.join(other=tr_text.set_index('ID'), on='ID')
df.head()


# ### 3.4. Jaccard Similarity between documents

# In[ ]:


def jaccard_similarity(document1, document2):
  intersection = set(document1).intersection(set(document2))
  union = set(document1).union(set(document2))
  return len(intersection)/len(set(union))


# In[ ]:


#https://stackoverflow.com/a/17841321
tmp_df = df.groupby('Class')['TEXT'].agg(lambda col: ' '.join(col)).reset_index()


# In[ ]:


tmp_df.head()


# In[ ]:


similarity = np.zeros((9, 9))
def calculateSimilrity():
  for i in tqdm(tmp_df["Class"].values):
    for j in tmp_df["Class"].values:
     if(i < j):
       sim = jaccard_similarity(tmp_df['TEXT'].iloc[i - 1].split(), tmp_df['TEXT'].iloc[j - 1].split())
       similarity[i - 1][j - 1] = sim
       similarity[j - 1][i - 1] = sim


# In[ ]:


calculateSimilrity()


# In[ ]:


#https://stackoverflow.com/a/58165593
#https://indianaiproduction.com/seaborn-heatmap/
plt.figure(figsize=(12, 8))
up_matrix = np.triu(similarity)
ax = sns.heatmap(similarity, xticklabels=range(1,10), yticklabels=range(1,10), annot=True, mask=up_matrix)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


# - 8 and 9 are most similar in terms of jaccard similarity
# - followed by (6,5),(5,3),(4,1),(7,2)

# ### 4.5. Let's view tf-idf of word importance per Class

# In[ ]:


def calculateTF(document):
  tmp_tfs = dict()
  words = document.split()
  len_doc = len(words)
  for word in words:
    if word in tmp_tfs:
      tmp_tfs[word] += 1
    else:
      tmp_tfs[word] = 1
  
  for word in tmp_tfs:
    tmp_tfs[word] = tmp_tfs[word] / len_doc
  return tmp_tfs


# In[ ]:


def calculateIDF(documents):
  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(documents)
  tmp_dct = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_ ))
  return tmp_dct


# In[ ]:


def calculateTfIdf(documents):
  tmp_tfIdf = dict()
  tmp_idfs = calculateIDF(documents)
  count = 0
  for i in documents:
    tmp_tfs = calculateTF(i)
    tmp_dict = dict()
    for j in set(i.split()):
      try:
        tmp_dict[j] = tmp_idfs[j] * tmp_tfs[j]
      except:
        continue
    tmp_tfIdf[count] = tmp_dict
    count += 1
  return tmp_tfIdf


# In[ ]:


tfIdfs = calculateTfIdf(df['TEXT'].values)


# In[ ]:


sorted(tfIdfs[0].items(), key=lambda kv: kv[1])[:2]


# In[ ]:


plt.figure(figsize=(12,15))
plt.title("word frequency for each class")
for i in tqdm(range(9)):
  tmp_arr = sorted(tfIdfs[i].items(), key=lambda kv: kv[1], reverse=True)[:10]
  plt.subplot(3, 3, (i + 1))
  plt.title('class ' + str(i + 1))
  tfidfseries = pd.Series(data=[p[1] for p in tmp_arr],name='TFIDF')
  names = pd.Series(data=[p[0] for p in tmp_arr], name='Words')
  frame = {'words': names, 'tfidfs': tfidfseries}
  tmp_df_tfidf = pd.DataFrame(frame)
  plt.xticks(rotation=30)
  ax = sns.barplot(x='words', y="tfidfs", data=tmp_df_tfidf)

plt.show()


# - TFIDF values are giving different picture than jaccard similarity
# - There are lot of common words between class 6 and 7 in top ifidf values and yer the jaccard similarity shows 6 and 7 are siginificantly different ones.

# **Question:** Is it good Idea to identify and remove most common words in documens, so that we have best dissimilar set of documents for each class?

# In[ ]:




