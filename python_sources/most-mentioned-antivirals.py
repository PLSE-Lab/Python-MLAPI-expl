#!/usr/bin/env python
# coding: utf-8

# As medical doctors are busy saving lives, data scientists can help them navigate through the mole of recent literature on COVID-19 to access quickly and efficiently new approaches and new therapies. This simple notebook presents the most mentioned names of antivirals in the CORD-19 database.

# # Data preparation

# In[ ]:


get_ipython().system(' pip install distance #Make sure you have Internet(ON) in the Kaggle notebook settings')


# In[ ]:


import pandas as pd
import numpy as np
import json
import os.path
import re
import distance
import matplotlib.pyplot as plt


# In[ ]:


metadata=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
metadata=metadata[metadata["sha"]==metadata["sha"]] #filters out entries having sha=NaN
metadata=metadata[metadata["has_full_text"]] #filters out entries for which the full text is not available


# In[ ]:


def path(shaid): #returns path of .json file
    for dirname, _, files in os.walk('/kaggle/input'):
        if shaid+'.json' in files:
            return os.path.join(dirname,shaid+".json")


# In[ ]:


metadata["Path"]=metadata.apply(lambda x: path(x["sha"]),axis=1) #this takes a while unfotunately
metadata=metadata[metadata["Path"]==metadata["Path"]]
metadata.shape


# # Looking for antivirals

# According to the standard drug nomenclature ( https://en.wikipedia.org/wiki/Drug_nomenclature ), the names for antivirals have the suffix *-vir*. 

# In[ ]:


STRING='vir ' #note the space at the end
KEY='corona' #filter for keywords such as corona, covid, sars, mers, etc.


# In[ ]:


Texts={} #dictionary: {id: "Text"}; adapted from cristian's notebook (https://www.kaggle.com/crprpr/vaccine-data-filter)
Abs_and_concl_w_punct={}
valid_id=[]
for shaid,file in zip(metadata["sha"],metadata["Path"]):
    with open(file, 'r') as f:
        doc=json.load(f)
    MainText=''
    A_C_w_p=''
    for item in doc["body_text"]:
        MainText=MainText+(re.sub('[^a-zA-Z0-9]', ' ', item["text"].lower()))
        if (item["section"]=="Discussion") or (item["section"]=="Abstract") or (item["section"]=="Conclusion"):
            A_C_w_p=A_C_w_p+item["text"].lower()
    if (STRING in MainText) and (KEY in MainText):
        Texts[shaid]=MainText
        Abs_and_concl_w_punct[shaid]=A_C_w_p
        valid_id.append(shaid)


# In[ ]:


metadata=metadata[metadata["sha"].isin(valid_id)] #filter only articles that contain names of antivirals


# In[ ]:


MIN_LENGTH=6 #most likely names of antivirals are longer than 4 letters + 2 spaces; shorter words are probably acronyms 
drugs=[]
for shaid in valid_id:
    iterator=re.finditer(STRING,Texts[shaid])
    for m in iterator:
        drugs.append(Texts[shaid][Texts[shaid].rfind(' ',0, m.end()-2):m.end()])
drugs=[i for i in drugs if len(i)>MIN_LENGTH]
drugs_set=list(set(drugs))
count=[]
for d in drugs_set:
    count.append(-drugs.count(d))
drugs_set=list(np.array(drugs_set)[np.array(count).argsort()])  # thanks Julian for the suggestion


# Now drugs_set contains the set of antiviral names.

# In[ ]:


len(drugs_set)


# # Looking for misspellings

# Curiously, some names look very similar to each other, and are most likely typos or different spellings. Here I look for the similarity between pairs of words via the Levenshtein distance ( https://en.wikipedia.org/wiki/Levenshtein_distance ). If the distance is smaller than THRESH, I regard the names as equivalent and retain the spelling with most entries.

# In[ ]:


THRESH=2 #Threshold for the Levenshtein distance
incorrects=[]
corrects=[]
from itertools import combinations
for str1,str2 in combinations(drugs_set,2):
    if (distance.levenshtein(str1, str2)<THRESH) and (drugs.count(str1)>10 or drugs.count(str2)>10):
            if drugs.count(str1)>=drugs.count(str2):
                incorrect=str2
                correct=str1
            else:
                incorrect=str1
                correct=str2
            print(str1, "(",drugs.count(str1),")", "and", str2, "(",drugs.count(str2),")", "look very similar.")
            if incorrect not in incorrects:
                print("I will substitute", incorrect, "with", correct,".")
                incorrects.append(incorrect)
                corrects.append(correct)


# In[ ]:


for item in incorrects:
    drugs_set.remove(item)


# In[ ]:


len(drugs_set)


# At least 214 different antivirals are mentioned in the CORD-19 database.

# This substitutes the correct spellings in the bodies of text:

# In[ ]:


for shaid in valid_id:
    for inc in range(0,len(incorrects)):
        Texts[shaid]=re.sub(incorrects[inc],corrects[inc], Texts[shaid])


# # Most mentioned antivirals

# Here I set up a dataframe ("antivirals") that contains useful information for each drug.

# In[ ]:


antivirals=pd.DataFrame(drugs_set,columns=["Drug"])

def count1(drug,druglist):
    return druglist.count(drug)

def count2(drug):
    n=0
    for shaid in valid_id:
        iterator=re.finditer(drug,Abs_and_concl_w_punct[shaid])
        for m in iterator:
            n+=1 
    return n
        
antivirals['Count_text'] = antivirals.apply(lambda x: count1(x["Drug"],drugs),axis=1) #counts occurences in the whole text
antivirals['Count_abs_concl'] = antivirals.apply(lambda x: count2(x["Drug"]),axis=1) #counts occurences in abstract + conclusions


# In[ ]:


MAXPLOT=20 #plot the MAXPLOT most mentioned antivirals
plt.figure(figsize=(20,5))
plt.bar(antivirals["Drug"][(-antivirals["Count_text"].to_numpy()).argsort()[:MAXPLOT]], antivirals["Count_text"][(-antivirals["Count_text"].to_numpy()).argsort()[:MAXPLOT]])
plt.xticks(rotation=90,fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("Counts",fontsize=15)
plt.show()


# In the scientific literature, "hot" topics and popular methods are often proper approaches, otherwise lot of experts (the scientists) would not bother writing about them. However, the contrary is not true: there may be very effective drugs taken into consideration only by few research groups. 

# # Sentiment analysis

# Here I apply sentiment analysis "off-the-shelf". I don't think this approach is optimal for scientific papers, which often involve neutral language, but it may be a good starting point.
# The function "Sentiment" looks for the name of the antiviral in the abstract and conclusions sections, cuts the sentence around the found word by looking for the nearest dots or commas, and then assigns a sentiment between -1 and +1.

# In[ ]:


from textblob import TextBlob
import nltk
histo=[]
nltk.download('punkt')


# In[ ]:


def Sentiment(drug): #looks for the drug name in the abstract or conclusions and measures the sentiment
    s=0
    smax=-1
    for shaid in valid_id:
            iterator=re.finditer(drug,Abs_and_concl_w_punct[shaid])
            for m in iterator:
                beg_comma=Abs_and_concl_w_punct[shaid].rfind(',',0, m.start())+1
                end_comma=Abs_and_concl_w_punct[shaid].find(',',m.end()-1,len(Texts[shaid]))
                beg_dot=Abs_and_concl_w_punct[shaid].rfind('.',0, m.start())+1
                end_dot=Abs_and_concl_w_punct[shaid].find('.',m.end()-1,len(Texts[shaid]))
                beg=max(beg_comma,beg_dot)
                end=min(end_comma,end_dot)
                blob = TextBlob(Abs_and_concl_w_punct[shaid][beg:end])
                s+=blob.sentiment.polarity
    return s

THRESH=0.3
def Sentence(drug): #records positive senctences, with the doi for reference
    nice_sentence=[]
    for shaid in valid_id:
            iterator=re.finditer(drug,Abs_and_concl_w_punct[shaid])
            for m in iterator:
                beg_comma=Abs_and_concl_w_punct[shaid].rfind(',',0, m.start())+1
                end_comma=Abs_and_concl_w_punct[shaid].find(',',m.end()-1,len(Texts[shaid]))
                beg_dot=Abs_and_concl_w_punct[shaid].rfind('.',0, m.start())+1
                end_dot=Abs_and_concl_w_punct[shaid].find('.',m.end()-1,len(Texts[shaid]))
                beg=max(beg_comma,beg_dot)
                end=min(end_comma,end_dot)
                blob = TextBlob(Abs_and_concl_w_punct[shaid][beg:end])
                if blob.sentiment.polarity > THRESH:
                    for doi in metadata[metadata["sha"]==shaid]["doi"]:
                        nice_sentence.append(str(blob)+" [ "+doi+" ]")
    if len(nice_sentence)==0:
        nice_sentence="Nothing found"
    return nice_sentence


# In[ ]:


antivirals['Sentiment'] = antivirals.apply(lambda x: Sentiment(x["Drug"]),axis=1)
antivirals["Nice_sentence"] = antivirals.apply(lambda x: Sentence(x["Drug"]),axis=1)
#antivirals['Sentiment_norm']=antivirals["Sentiment"]/antivirals["Count_abs_concl"]


# In[ ]:


MAXPLOT=20
plt.figure(figsize=(20,5))
plt.bar(antivirals["Drug"][(-antivirals["Sentiment"].to_numpy()).argsort()[:MAXPLOT]], antivirals["Sentiment"][(-antivirals["Sentiment"].to_numpy()).argsort()[:MAXPLOT]])
plt.xticks(rotation=90)
plt.xticks(rotation=90,fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("Total Sentiment",fontsize=15)
plt.show()


# Oseltamivir (Tamiflu) and Zanamivir (Relenza) were both recommended by the World Health Organization (WHO) against the 2009 H1N1 pandemics (swine flu), see https://www.who.int/csr/resources/publications/swineflu/h1n1_guidelines_pharmaceutical_mngt.pdf . Here are some sentences classified as "positive" by the sentiment analysis that involve Oseltamivir:

# In[ ]:


for sentences in antivirals[antivirals["Drug"]==" oseltamivir "]["Nice_sentence"]:
    for s in sentences:
        print(s)


# At the time of writing Favipiravir (Avigan) is regarded by a few research groups as a possible effective treatment against COVID-19 and experimental therapies have begun in Italy ( https://www.repubblica.it/salute/medicina-e-ricerca/2020/03/23/news/covid-19_speranza_aifa_procede_su_sperimentazione_avigan_-252090636/?refresh_ce , in Italian). However, the literature about this antiviral is scarce and the only positive sentence recorded by the sentiment analysis is referred to another virus (the Chikungunya virus):

# In[ ]:


for sentences in antivirals[antivirals["Drug"]==" favipiravir "]["Nice_sentence"]:
    for s in sentences:
        print(s)


# This demonstrates that the current literature on COVID-19 is still in its infancy.

# Even if antivirals are strongly virus-specific, recycling medications that are already in use to treat similar diseases is the most efficient way to face an emergency.

# In[ ]:




