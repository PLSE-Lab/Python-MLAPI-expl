#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import spacy
from spacy import displacy
import json


# In[ ]:


#Upload all articles
articles = []
for dirpath, subdirs, files in os.walk('/kaggle/input'):
    for x in files:
        if x.endswith(".json"):
            with open(os.path.join(dirpath, x)) as json_file:
                data = json.load(json_file)
                articles.append(data)
print (len(articles))
#29315


# In[ ]:


#Extract relevant paragraphs matching keyword
keyword = "risk factor" #lower case !!!

paragraphs = []

for a in articles:
    for p in a["body_text"]:
        if keyword in p["text"]:
            paragraphs.append(p["text"])
            
print(len(paragraphs))  
#9463


# In[ ]:


#Perform NLP processing
nlp = spacy.load("en_core_web_sm")
docs = list(nlp.pipe(paragraphs))


# In[ ]:


#Define function to extract relations.
#Check REF-01 for an analysis on the most dep for the token matching the keyword 

def extract_relations(doc,keyword):

    #Consolidate spans
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = spacy.util.filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)  
    
    relations = []
    
    #For each sentence in paragraph
    for sent in doc.sents:
        #For each token in the sentence containing the "keyword" i.e. hits       
        for t in sent:
            if keyword in t.text.lower():
                if t.dep_ == "attr":
                    n1 = [w for w in t.head.children if (w.dep_ == "nsubj")]
                    n2 = t
                    if len(n1) > 0:
                        n1 = n1[0]
                        n1_subtree_span = doc[n1.left_edge.i : n1.right_edge.i + 1]
                        n2_subtree_span = doc[n2.left_edge.i : n2.right_edge.i + 1]
                        relations.append([n1_subtree_span,n2_subtree_span,n1,n2,sent,"attr"])   
                if t.dep_ == "pobj" and t.head.dep_ == "prep":
                    n1 = [w for w in t.head.head.children if (w.dep_ == "nsubjpass")]
                    n2 = t
                    if len(n1) > 0:
                        n1 = n1[0]
                        n1_subtree_span = doc[n1.left_edge.i : n1.right_edge.i + 1]
                        n2_subtree_span = doc[n2.left_edge.i : n2.right_edge.i + 1]
                        relations.append([n1_subtree_span,n2_subtree_span,n1,n2,sent,"pobj"])  
    return relations


# In[ ]:


#Extract all risk factors -> cleaning needed
risk_factors_raw=[]

for doc in docs:
    hit = extract_relations(doc,keyword)
    if len(hit) > 0:
        for h in hit:
            risk_factors_raw.append(h)

print(len(risk_factors_raw))
#2654


# In[ ]:


#Build clean risk_factors list

risk_factors = []
exclusion_list = ["WDT","PRP","CD","DT","WP"] #Check REF-02 and REF-03 to understand exclusion list

for r in risk_factors_raw:
    if r[2].tag_ not in exclusion_list:
        risk_factors.append([r[0],r[4],r[5]])
        
print(len(risk_factors))
#count 2473


# In[ ]:


#Export to csv
outcome=pd.DataFrame(risk_factors,columns=["Risk_factor","Sentence","type"])
outcome.to_csv("Risk_factors.csv")


# In[ ]:


#Visualise selected type of deps -> att
plt=outcome[outcome["type"]=="attr"]["Sentence"].to_list()[1]
#displacy.serve(plt, style="dep")


# In[ ]:


#Visualise selected type of deps -> pobj
plt=outcome[outcome["type"]=="pobj"]["Sentence"].to_list()[1]
#displacy.serve(plt, style="dep")


# ### Appendix
# 
# **REF-01**
# Check most common dep for the tokens matching the keyword
# ```python
# s=[]
# for doc in docs:
#     for sent in doc.sents:
#         for t in sent:
#             if keyword in t.text:
#                 s.append({"t_dep":t.dep_,"t_head_dep":t.head.dep_,"t_head_head_dep":t.head.head.tag_,"sent":sent})
# 
# df = pd.DataFrame(s)  
# Check most common dep for the token matching the keyword
# 
# df.groupby(["t_dep"]).size().to_frame()
# ```
# Most common dep are "attr" and "pobj". In the relation extraction function we will address these types as 1st priority
# 
# **REF-02**
# Analyse tag associated to the n1 and exclude unwanted values. Case 1 -> "attr
# ```python
# for r in risk_factors_raw:
#     if r[5]== "attr" and r[2].tag_ not in ["NN","NNS","NNP","NNPS","VBG","VB"]:
#         print(r[2].tag_ + "--" + r[2].text)
# ```        
# TO EXCLUDE --> and r[2].tag_ not in ["WDT","PRP","CD","DT","WP"] 
# 
# **REF-03**
# Analyse tag associated to the n1 and exclude unwanted values. Case 1 -> "pobj
# ```python
# for r in risk_factors_raw:
#     if r[5]== "pobj" and r[2].tag_ not in ["NN","NNS","NNP","NNPS","VBG","VB"] :
#         print(r[2].tag_ + "--" + r[2].text)
# ```        
# TO EXCLUDE --> and r[2].tag_ not in ["WDT","PRP","CD","DT","WP"] 

# In[ ]:




