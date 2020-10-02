#!/usr/bin/env python
# coding: utf-8

# ### Models and open question on COVID19   
# Myrna MFL     
# 
# **Purpose**: Create summary tables that address models and open questions related to COVID-19 for the Open Research Dataset Challenge (CORD-19) Round #2.   
# 

# Specifically, what the literature reports about:   
# 
# 1. **Human immune response** to COVID-19
# 2. What is known about **mutations** of the virus?
# 3. Studies to monitor potential **adaptations**
# 4. Are there studies about **phenotypic change**?
# 5. Changes in COVID-19 as the virus **evolves**
# 6. What regional genetic variations (**mutations**) exist
# 7. What do **models** for transmission predict?
# 8. **Serial Interval** (for infector-infectee pair)
# 9. Efforts to develop **qualitative** assessment frameworks to systematically collect
# 
# 

# Data used:  
# Metadata provided in Kaggle's CORD19 research challenge. From the journals listed, I narrow the search to those articles with abstracts.   
#         

# In[ ]:


##Libraries
import numpy as np 
import pandas as pd 
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt


# In[ ]:


#Geting the Table of Studies
## a table of 18 columns and 63571 entries
articles=pd.read_csv('../input/CORD-19-research-challenge/metadata.csv') 
articles.shape


# In[ ]:


####Cleaned the TABLE for readability
Art1= articles[['title','publish_time','journal','url','abstract','doi','cord_uid']]
#Made a copy to work with
Articles1=Art1.copy()


# In[ ]:


#separate each word in the column: abstract, for browsing
Articles1['words'] = Articles1.abstract.str.strip().str.split('[\W_]+')

#separate words in the abstract column and create a new column
Articles1 = Articles1[Articles1.words.str.len() > 0]
Articles1.head(3)


# In[ ]:


# saving the Table (dataframe) above
Articles1.to_csv('Articles.csv') 


# #### Human immune response to COVID-19

# In[ ]:


##1 Human immune response to COVID-19
#looking through the abstracts  
## 
##TABLE OF abstracts related to COVID 
COVID=Articles1[Articles1['abstract'].str.contains('COVID')]
COVID.shape
#5443 entries with ABSTRACTS contain the term COVID.


# In[ ]:


# saving the dataframe above
COVID.to_csv('COVID_ArticleAbstracts.csv') 


# In[ ]:


##Looking among COVID articles for immune response

ImmResp=COVID[COVID['abstract'].str.contains('immune response')]
ImmResp.shape
##There are 124 COVID articles with abstracts 
##that include the phrase 'immune response'
# 'human immune response' search showed NO articles.


# In[ ]:


Human=ImmResp[ImmResp['abstract'].str.contains('human')]
Human.shape
#36 article abstracts found


# In[ ]:


Human


# In[ ]:


# saving the TABLE above as a 
#table-answer to task 1
Human.to_csv('COVID_ArticleAbstracts_Human_Immune_Response.csv') 


# #### 2 What is known about mutations of the virus?

# In[ ]:


#looking through the abstracts  
## 
##TABLE OF abstracts related to mutation 
Mutation=Articles1[Articles1['abstract'].str.contains('mutation')]
Mutation.shape
#2105 article abstracts found


# In[ ]:


coronaMut=Mutation[Mutation['abstract'].str.contains('corona')]
coronaMut.shape
##among the 2105, 580 include the term corona


# In[ ]:


COVIDMut=Mutation[Mutation['abstract'].str.contains('COVID')]
COVIDMut.shape
##among the 2105 mutation abstracts, 
#there are 95 article abstracts that include the term COVID


# In[ ]:


# saving the TABLE above as a 
#table-answer to task 2
COVIDMut.to_csv('COVID_ArticleAbstracts_COVID_mutation.csv') 


# #### Word Cloud: About COVID mutation

# In[ ]:


#to omit:
symbols1='!@#$%&*.,?"-'
ignoreThese=['background', 'abstract',
             'our','this','the',
             'objective','since', 'name',
            'word', 'words', 'and',
            'summary', 'study', 'dtype',
            'goal']

for char in symbols1:
        words1=COVIDMut['words'].replace(char,' ')
#lower case all words
words1=str(words1)
words1=words1.lower()

#ignore words
for item in ignoreThese:
        words1=words1.replace(item, ' ') 
        
wordcloud = WordCloud(
            width = 1000,
            height = 1000,
            background_color = 'black',
            stopwords = STOPWORDS).generate(words1)
fig = plt.figure(
    figsize = (20, 10),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# #### 3. Studies to monitor potential adaptations

# In[ ]:


##Using the COVID table of article abstracts
#to find studies about potential adaptation
Adapt=COVID[COVID['abstract'].str.contains('adaptation')]
Adapt.shape


# In[ ]:


Adapt


# In[ ]:


# saving the TABLE above as a 
#table-answer to task 3
Adapt.to_csv('COVID_ArticleAbstracts_COVID_adaptation.csv')


# #### 4 Are there studies about phenotypic change?

# In[ ]:


##4 Are there studies about phenotypic change?

#looking through the abstracts  
## 
##TABLE OF abstracts related to phenotypic change 
Pheno=Articles1[Articles1['abstract'].str.contains('phenotypic change')]
Pheno.shape
#There are 18 among all the article abstracts


# In[ ]:


#what about COVID-specific abstracts?
Pheno2=COVID[COVID['abstract'].str.contains('phenotypic change')]
Pheno2.shape
# None


# In[ ]:


# saving the TABLE Pheno 
#table-answer to task 4
Pheno.to_csv('ArticleAbstracts_studies_on_phenotypic_change.csv')


# #### 5 Changes in COVID-19 as the virus evolves.

# In[ ]:


#Looking through COVID-specific abstracts
Evo=COVID[COVID['abstract'].str.contains('evolve')]
Evo.shape


# In[ ]:


#searching through those article abstracts with COVID term
VirEvo=COVID[COVID['abstract'].str.contains('virus evolve')]
VirEvo.shape
#2 results


# In[ ]:


VirEvo


# In[ ]:


# saving the TABLE above 
#table-answer to task 5
VirEvo.to_csv('ArticleAbstracts_COVID_virus_evolve.csv')


# #### 6 Regional genetic variations (mutations) 

# In[ ]:


##In those COVID-related article abstracts..
Genetic=COVID[COVID['abstract'].str.contains('genetic')]
Genetic.shape
#..there are 146 abstracts with the term genetic


# In[ ]:


GeneticV=COVID[COVID['abstract'].str.contains('genetic variation')]
GeneticV.shape


# In[ ]:


# saving the TABLE above 
#table-answer to task 6
GeneticV.to_csv('ArticleAbstracts_COVID_genetic_variation.csv')


# #### 7 What do models for transmission predict?

# In[ ]:


trans=COVID[COVID['abstract'].str.contains('transmission')]
trans.shape
#957 article abstracts with the term transmission


# In[ ]:


##searching through the above dataframe
##for specifics
models=trans[trans['abstract'].str.contains('model')]
models.shape
##345 abstract articles about COVID transmission with the term model


# In[ ]:


##searching through the above 
##for PREDICTIONS
pred=models[models['abstract'].str.contains('predict')]
pred.shape


# In[ ]:


# saving the TABLE above 
#table-answer to task 7
pred.to_csv('ArticleAbstracts_COVID_transmission_model_predict.csv')


# #### Word Cloud: model predictions

# In[ ]:


#to omit:
#symbols1='!@#$%&*.,?"-'
#ignoreThese=['background', 'abstract',
            # 'our','this','the','objective','since', 'name',
            #'word', 'words', 'and','summary', 'study']

for char in symbols1:
        words2=pred['words'].replace(char,' ')
#lower case all words
words2=str(words2)
words2=words2.lower()

#ignore words
for item in ignoreThese:
        words2=words2.replace(item, ' ') 
        
wordcloud = WordCloud(
            width = 1000,
            height = 1000,
            background_color = 'black',
            stopwords = STOPWORDS).generate(words2)
fig = plt.figure(
    figsize = (20, 10),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# #### 8. Serial Interval (for infector-infectee pair)

# In[ ]:


#searching through those article abstracts with COVID term
SerialInt=COVID[COVID['abstract'].str.contains('virus evolve')]
SerialInt.shape
#found 2


# In[ ]:


SerialInt


# In[ ]:


# saving the TABLE above 
#table-answer to task 8
SerialInt.to_csv('ArticleAbstracts_COVID_Serial_Interval.csv')


# #### 9 Qualitative assessment frameworks 

# In[ ]:


#searching through ALL article abstracts for qualitative frameworks
Qual=Articles1[Articles1['abstract'].str.contains('qualitative')]
Qual.shape
#391 found


# In[ ]:


#searching through the above for: Framework
QFrame=Qual[Qual['abstract'].str.contains('framework')]
QFrame.shape
#23 article abstracts mention Qualitative Framework


# In[ ]:


# saving the TABLE above 
#table-answer to task 9
QFrame.to_csv('ArticleAbstracts_Qualitative_Framework.csv')

