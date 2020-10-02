#!/usr/bin/env python
# coding: utf-8

# ***CORD-19 is a resource of over 24,000 scholarly articles, including over 12,000 with full text, about COVID-19, and the coronavirus group.These are the entities from the Abstracts from the CORD-19 dataset.***

# # **R&D Question: **
# 
# 1. What do we know about vaccines and therapeutics?

# In[ ]:


get_ipython().system('pip install spacy==2.2.2')
get_ipython().system('pip install scispacy')
get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz')
get_ipython().system('pip install https://med7.s3.eu-west-2.amazonaws.com/en_core_med7_lg.tar.gz')
# !pip install googletrans


# # **Load the Data**
# 
# 
# Cite: Dataset Parsing Code | Kaggle, COVID EDA: Initial Exploration Tool
# 
# 

# # **Loading Metadata**

# In[ ]:


import os
import scispacy
import spacy
from spacy import displacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker


# nlp = spacy.load("en_core_sci_lg")
import en_core_sci_lg
nlp = en_core_sci_lg.load()
med7 = spacy.load("en_core_med7_lg")

linker = UmlsEntityLinker(resolve_abbreviations=True)
nlp.add_pipe(linker)


abs_link = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        #print(os.path.join(dirname, filename).split(".")[1])
        if os.path.join(dirname, filename).split(".")[1] == "json":
            abs_link.append(os.path.join(dirname, filename))

        
            
            # Any results you write to the current directory are saved as output.


# In[ ]:


# df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')
# df.shape


# In[ ]:


print(abs_link[0])


# In[ ]:


json_file_list = []
import json
#file_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13/noncomm_use_subset/noncomm_use_subset/252878458973ebf8c4a149447b2887f0e553e7b5.json'

for i in range(0,len(abs_link),1):
#     if i==6:
#         break
    with open(abs_link[i]) as json_file:
         json_file = json.load(json_file)
    json_file_list.append(json_file)


# # **scispacy and Med7 for find Entities**

# In[ ]:



for i in range(0,len(json_file_list),1):
    
#     if i==6:
#         break
    
    if len(json_file_list[i]['abstract']) != 0:
        print("\n")
        print("Title: " +str(json_file_list[i]['metadata']['title']))
        #print('\nAbstract: \n\n', json_file_list[i]['abstract'])
        print("\n")
        key, val = next(iter(json_file_list[i]['abstract'][0].items()))
        print("Abstract: "+str(val)) 
        print("\n")
        
        print("scispacy Abstract Entities: ")
        print("\n")
        
        doc = nlp(str(val))
        print(doc.ents)
        print("\n")
        abbreviation_pipe = AbbreviationDetector(nlp)

#         print("Abbreviation", "\t", "Definition")
#         for abrv in doc._.abbreviations:
#           print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")
        
        
        entity = doc.ents
        print("scispacy Entities: ")
        print("\n")
        # Each entity is linked to UMLS with a score
        # (currently just char-3gram matching).
#         for i in range(len(entity)):
            
#             for umls_ent in entity[i]._.umls_ents:
#                 print(linker.umls.cui_to_entity[umls_ent[0]])
                
                
        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)
#             print(str(ent)+" "+ str(ent.label_))
        
    
    


        # create distinct colours for labels
        col_dict = {}
        seven_colours = ['#e6194B', '#3cb44b', '#ffe119', '#ffd8b1', '#f58231', '#f032e6', '#42d4f4']
        for label, colour in zip(med7.pipe_labels['ner'], seven_colours):
            col_dict[label] = colour

        options = {'ents': med7.pipe_labels['ner'], 'colors':col_dict}

        doc_med7 = med7(val)

        print("\n")
        
        print("Med7 Entities: ")
        print("\n")
        
        #spacy.displacy.render(doc, style='ent', jupyter=True, options=options)
        #spacy.displacy.render(doc_med7, style='ent', jupyter=True, options=options)

        print([(ent.text, ent.label_) for ent in doc_med7.ents])


# # **Conclusion**
# 
# **Work in progress !! Leave an upvote if you like it, Thank you :)**
