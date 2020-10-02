#!/usr/bin/env python
# coding: utf-8

# # COVID-19: Open Research Dataset Challenge (CORD-19)
# # Task 01: What is known about transmission, incubation, and environmental stability?
# 
# 
# 
# ### Purpose:
# 
# - Our purpose is to extract relevant text sections related to the proposed questions from all the research dataset. 
# 
# ### Description of the methodology steps:
# 
# - Upload and structure all researches paper's sections separately.
# - P[](http://)rocess the sections' text using the nltk package (lowercase, remove punctuations, tokenization, Lemmatization).
# - To search for the relevant results about each task:   
#     - Select the keywords and their synonyms.   
#     - Search in the dataset those keywords (or most of them) on the lemmatized tokens of the sections.
# - While the obtained results are very large, a ranking of the resulted papers' sections is recommended using both: the freshness degree and the rank score of the papers' institutions, according to the following Formula: 
#     
#     ***Score = Affiliation Score . (0.2) + Publication Year . (0.8)***
#     
#     
# - The obtained results now are ranked, and ready to get used!
# 
# 

# In[ ]:


import pandas as pd


# In[ ]:


# Meta data :
meta = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")


# In[ ]:


meta.head(3)


# In[ ]:


meta.shape


# In[ ]:


meta.columns


# In[ ]:


meta.dtypes


# In[ ]:


# Charge all .Json files:
import glob
papers = glob.glob(f'/kaggle/input/CORD-19-research-challenge/**/*.json', recursive=True)


# In[ ]:


len(papers)


# In[ ]:


# import json
# papers_data = pd.DataFrame(columns=['PaperID','Title','Section','Text','Affilations'], index=range(len(papers)*50))

# # Remove duplicates in a list:
# def my_function(x):
#     return list(dict.fromkeys(x))

# i=0
# for j in range(len(papers)):
#     with open(papers[j]) as file:
#         content = json.load(file)
        
#         # ID and Title:
#         pap_id = content['paper_id']
#         title =  content['metadata']['title']
        
#         # Affiations:
#         affiliation = []
#         for sec in content['metadata']['authors']:
#             try:
#                 affiliation.append(sec['affiliation']['institution'])
#             except:
#                 pass
#         affiliation = my_function(affiliation)
        
#         # Abstract
#         for sec in content['abstract']:
#             papers_data.iloc[i, 0] = pap_id
#             papers_data.iloc[i, 1] = title
#             papers_data.iloc[i, 2] = sec['section']
#             papers_data.iloc[i, 3] = sec['text']
#             papers_data.iloc[i, 4] = affiliation
#             i = i + 1
            
#         # Body text
#         for sec in content['body_text']:
#             papers_data.iloc[i, 0] = pap_id
#             papers_data.iloc[i, 1] = title
#             papers_data.iloc[i, 2] = sec['section']
#             papers_data.iloc[i, 3] = sec['text']
#             papers_data.iloc[i, 4] = affiliation
#             i = i + 1

# papers_data.dropna(inplace=True)
# papers_data = papers_data.astype(str).drop_duplicates() 

# # Text processing:
# import nltk
# nltk.download('punkt')
# # Lowercase:
# for i in range(len(papers_data)):
#     try:
#         papers_data.iloc[i, 1] = papers_data.iloc[i, 1].lower()
#         papers_data.iloc[i, 2] = papers_data.iloc[i, 2].lower()
#         papers_data.iloc[i, 3] = papers_data.iloc[i, 3].lower()
#         papers_data.iloc[i, 4] = papers_data.iloc[i, 4].lower()
#     except:
#         pass
    
# # Tokenization:

# from nltk.tokenize import word_tokenize, sent_tokenize , RegexpTokenizer

# tokenizer = RegexpTokenizer(r'\w+') # remove punctuation
# papers_data["Title_Tokens_words"] = [list() for x in range(len(papers_data.index))]
# papers_data["Text_Tokens_words"] = [list() for x in range(len(papers_data.index))]

# for i in range(len(papers_data)):
#     try:
#         papers_data.iloc[i, 5] = tokenizer.tokenize(papers_data.iloc[i, 1])
#         papers_data.iloc[i, 6] = tokenizer.tokenize(papers_data.iloc[i, 3])
#     except:
#         pass
    
# # Remove stopwords:
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english')) 

# for i in range(len(papers_data)):
#     try:
#         papers_data.iloc[i, 5] = [w for w in papers_data.iloc[i, 5] if not w in stop_words] 
#         papers_data.iloc[i, 6] = [w for w in papers_data.iloc[i, 6] if not w in stop_words]
#     except:
#         pass
    
# # Words count:  
# papers_data["Words_count"] = 0

# # for i in range(len(papers_data)):
# #     try:
# #         papers_data.iloc[i, 7] = len(papers_data.iloc[i, 6])
# #     except:
# #         pass
    
# # Lemmatization :
# nltk.download('wordnet')

# from nltk.stem import WordNetLemmatizer

# wordnet_lemmatizer = WordNetLemmatizer()

# papers_data["Text_Lem_words"] = [list() for x in range(len(papers_data))]

# for i in range(len(papers_data)):
#     for j in range(len(papers_data.iloc[i, 6])):
#         papers_data.iloc[i, 8].append(wordnet_lemmatizer.lemmatize(papers_data.iloc[i, 6][j]))
        
# papers_data["Title_Lem_words"] = [list() for x in range(len(papers_data))]

# for i in range(len(papers_data)):
#     for j in range(len(papers_data.iloc[i, 5])):
#         papers_data.iloc[i, 9].append(wordnet_lemmatizer.lemmatize(papers_data.iloc[i, 5][j]))
        
# papers_data.to_csv("/kaggle/input/processed-researches-data/papers_data_final.csv")
print("Preprocessing done!")


# In[ ]:


papers_data = pd.read_csv("/kaggle/input/processed-researches-data/papers_data_final.csv")
del papers_data['Unnamed: 0']
papers_data.head()


# In[ ]:


import ast
papers_data['Affilations'] = papers_data['Affilations'].apply(lambda x: ast.literal_eval(x))
papers_data['Text_Lem_words'] = papers_data['Text_Lem_words'].apply(lambda x: ast.literal_eval(x))


# # Search for papers sections containing task's keywords:

# ### Task 01 : What is known about transmission, incubation, and environmental stability?
# 
# #### Task Details
# 
# - What is known about transmission, incubation, and environmental stability? What do we know about natural history, transmission, and diagnostics for the virus? What have we learned about infection prevention and control?
# 
# - Specifically, we want to know what the literature reports about:
#     
#     - Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.
#     
#     - Prevalence of asymptomatic shedding and transmission (e.g., particularly children).
#     - Seasonality of transmission.
#     - Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).
#     - Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).
#     - Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).
#     - Natural history of the virus and shedding of it from an infected person
#     - Implementation of diagnostics and products to improve clinical processes
#     - Disease models, including animal models for infection, disease and transmission 
#     - Tools and studies to monitor phenotypic change and potential adaptation of the virus 
#     - Immune response and immunity
#     - Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings
#     - Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings
#     - Role of the environment in transmission

# ### Task 1.1: ***Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.***

# In[ ]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

wordnet_lemmatizer = WordNetLemmatizer()
def my_function(x):
    return list(dict.fromkeys(x))

keywords =['incubation','disease', 'age', 'health','contagious','recovery','periods']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   


# In[ ]:


incubation =[]
for i in range(len(papers_data)):
    if "incubation" in papers_data.iloc[i, 8]:
        incubation.append(i)


# In[ ]:


len(incubation)


# In[ ]:


disease =[]
for i in range(len(papers_data)):
    if "disease" in papers_data.iloc[i, 8]:
        disease.append(i)


# In[ ]:


len(disease)


# In[ ]:


age =[]
for i in range(len(papers_data)):
    if "age" in papers_data.iloc[i, 8]:
        age.append(i)


# In[ ]:


len(age)


# In[ ]:


health =[]
for i in range(len(papers_data)):
    if "health" in papers_data.iloc[i, 8]:
        health.append(i)


# In[ ]:


len(health)


# In[ ]:


contagious =[]
for i in range(len(papers_data)):
    if "contagious" in papers_data.iloc[i, 8]:
        contagious.append(i)


# In[ ]:


len(contagious)


# In[ ]:


recovery =[]
for i in range(len(papers_data)):
    if "recovery" in papers_data.iloc[i, 8]:
        recovery.append(i)


# In[ ]:


len(recovery)


# In[ ]:


period =[]
for i in range(len(papers_data)):
    if "period" in papers_data.iloc[i, 8]:
        period.append(i)


# In[ ]:


len(period)


# In[ ]:


# At least 6 of words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 5: 
        return(True)  
    return(False)    
  
task1_1 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task1_1.append(i)
    
len(task1_1)


# In[ ]:


## Results for task 1.1 :
for i in task1_1:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")


# ### Ranking by most Recent:

# In[ ]:


# Task 1.1:
print(task1_1)


# In[ ]:


task1_1_rank = papers_data.iloc[task1_1, :]
task1_1_rank.reset_index(inplace=True,drop=True)

# Grab the publish year from the meta data file:
meta = meta.rename(columns={"title": "Title"})
for i in range(len(meta)):
    meta.iloc[i, 2] = str(meta.iloc[i, 2]).lower()
meta['Title'] = meta['Title'].astype(str) 
task1_1_rank['Title'] = task1_1_rank['Title'].astype(str) 

task1_1_rank = pd.merge(task1_1_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task1_1_rank.dropna(inplace=True)

# Extract the year from the string publish time
import dateutil.parser as parser
task1_1_rank['publish_time'] = task1_1_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))
task1_1_rank['publish_time'] = task1_1_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task1_1_rank['publish_time'] = pd.to_numeric(task1_1_rank['publish_time'])
task1_1_rank = task1_1_rank.sort_values(by='publish_time', ascending=False)
task1_1_rank.reset_index(inplace=True,drop=True)


# ### By affiliations Scores:
# 
# - Ranking using the : http://www.shanghairanking.com/arwu2019.html

# In[ ]:


rank = pd.read_csv("/kaggle/input/shanghai-ranking/rank-univ.csv")
del rank['Unnamed: 0']
rank.head(5)


# In[ ]:


# Extract the affiliations score to the task's results:
task1_1_rank['Aff_Score'] = 0
for i in range(len(task1_1_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task1_1_rank.iloc[i, 4]:
            task1_1_rank.iloc[i, 11] = rank.iloc[j, 3]


# In[ ]:


task1_1_rank


# In[ ]:


task1_1_rank["Ranking_Score"] = task1_1_rank["publish_time"]*0.8 + task1_1_rank["Aff_Score"]*0.2


# In[ ]:


task1_1_rank.head(5)


# In[ ]:


task1_1_rank = task1_1_rank.sort_values(by='Ranking_Score', ascending=False)
task1_1_rank.reset_index(inplace=True,drop=True)
task1_1_rank


# In[ ]:


## 20 - Ranked Results for task 1.1 :

for i in range(len(task1_1_rank)):
    print("\n")
    print("PaperID: ", task1_1_rank.iloc[i, 0])
    print("Title: ", task1_1_rank.iloc[i, 1])
    print("Section: ", task1_1_rank.iloc[i, 2])
    print("Text: ", task1_1_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break


# ### Task1.2: ***Prevalence of asymptomatic shedding and transmission (e.g., particularly children)***

# In[ ]:


keywords =['asymptomatic','prevalence', 'shedding', 'transmission','children','elderly','age']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   


# In[ ]:


# At least 8 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 7: 
        return(True)  
    return(False)    
  
task1_2 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task1_2.append(i)
    
len(task1_2)


# In[ ]:


## Results for task 1.2 :
for i in task1_2:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")


# ### Ranking by most Recent:

# In[ ]:


# Task 1.2:
print(task1_2)


# In[ ]:


task1_2_rank = papers_data.iloc[task1_2, :]
task1_2_rank.reset_index(inplace=True,drop=True)

# Grab the publish year from the meta data file:
meta = meta.rename(columns={"title": "Title"})
for i in range(len(meta)):
    meta.iloc[i, 2] = str(meta.iloc[i, 2]).lower()
meta['Title'] = meta['Title'].astype(str) 
task1_2_rank['Title'] = task1_2_rank['Title'].astype(str) 

task1_2_rank = pd.merge(task1_2_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task1_2_rank.dropna(inplace=True)

# Extract the year from the string publish time
import dateutil.parser as parser
task1_2_rank['publish_time'] = task1_2_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))
task1_2_rank['publish_time'] = task1_2_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task1_2_rank['publish_time'] = pd.to_numeric(task1_2_rank['publish_time'])
task1_2_rank = task1_2_rank.sort_values(by='publish_time', ascending=False)
task1_2_rank.reset_index(inplace=True,drop=True)


# ### By affiliations Scores:
# 

# In[ ]:


# Extract the affiliations score to the task's results:
task1_2_rank['Aff_Score'] = 0
for i in range(len(task1_2_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task1_2_rank.iloc[i, 4]:
            task1_2_rank.iloc[i, 11] = rank.iloc[j, 3]


# In[ ]:


task1_2_rank["Ranking_Score"] = task1_2_rank["publish_time"]*0.8 + task1_2_rank["Aff_Score"]*0.2
task1_2_rank = task1_2_rank.sort_values(by='Ranking_Score', ascending=False)
task1_2_rank.reset_index(inplace=True,drop=True)
task1_2_rank


# In[ ]:


## 20 - Ranked Results for task 1.2 :

for i in range(len(task1_2_rank)):
    print("\n")
    print("PaperID: ", task1_2_rank.iloc[i, 0])
    print("Title: ", task1_2_rank.iloc[i, 1])
    print("Section: ", task1_2_rank.iloc[i, 2])
    print("Text: ", task1_2_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break


# ### Task 1.3: ***Seasonality of transmission.***

# In[ ]:


keywords =['prevalence', 'seasonality','transmission']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   


# In[ ]:


# At least 4 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 3: 
        return(True)  
    return(False)    
  
task1_3 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task1_3.append(i)
    
len(task1_3)


# In[ ]:


## Results for task 1.3 :
for i in task1_3:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")


# ### Ranking by the most recent:

# In[ ]:


# Task 1.3:
print(task1_3)


# In[ ]:


task1_3_rank = papers_data.iloc[task1_3, :]
task1_3_rank.reset_index(inplace=True,drop=True)

# Grab the publish year from the meta data file:
meta = meta.rename(columns={"title": "Title"})
for i in range(len(meta)):
    meta.iloc[i, 2] = str(meta.iloc[i, 2]).lower()
meta['Title'] = meta['Title'].astype(str) 
task1_3_rank['Title'] = task1_3_rank['Title'].astype(str) 

task1_3_rank = pd.merge(task1_3_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task1_3_rank.dropna(inplace=True)

# Extract the year from the string publish time
import dateutil.parser as parser
task1_3_rank['publish_time'] = task1_3_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))
task1_3_rank['publish_time'] = task1_3_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task1_3_rank['publish_time'] = pd.to_numeric(task1_3_rank['publish_time'])
task1_3_rank = task1_3_rank.sort_values(by='publish_time', ascending=False)
task1_3_rank.reset_index(inplace=True,drop=True)


# ### By affiliations Scores:

# In[ ]:


# Extract the affiliations score to the task's results:
task1_3_rank['Aff_Score'] = 0
for i in range(len(task1_3_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task1_3_rank.iloc[i, 4]:
            task1_3_rank.iloc[i, 11] = rank.iloc[j, 3]


# In[ ]:


task1_3_rank["Ranking_Score"] = task1_3_rank["publish_time"]*0.8 + task1_3_rank["Aff_Score"]*0.2
task1_3_rank = task1_3_rank.sort_values(by='Ranking_Score', ascending=False)
task1_3_rank.reset_index(inplace=True,drop=True)
task1_3_rank


# In[ ]:


## 20 - Ranked Results for task 1.3 :

for i in range(len(task1_3_rank)):
    print("\n")
    print("PaperID: ", task1_3_rank.iloc[i, 0])
    print("Title: ", task1_3_rank.iloc[i, 1])
    print("Section: ", task1_3_rank.iloc[i, 2])
    print("Text: ", task1_3_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break


# ### Task 1.4: ***Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).***

# In[ ]:


keywords =['physical', 'coronavirus','charge','distribution','adhesion','hydrophilic','phobic','surfaces','environmental','survival','decontamination','affected','area','viral','shedding']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   


# In[ ]:


# At least 10 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 9: 
        return(True)  
    return(False)    
  
task1_4 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task1_4.append(i)
    
len(task1_4)


# In[ ]:


## Results for task 1.4 :
for i in task1_4:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")


# ### Ranking by teh most recent:

# In[ ]:


# Task 1.4:
print(task1_4)


# In[ ]:


task1_4_rank = papers_data.iloc[task1_4, :]
task1_4_rank.reset_index(inplace=True,drop=True)

# Grab the publish year from the meta data file:
meta = meta.rename(columns={"title": "Title"})
for i in range(len(meta)):
    meta.iloc[i, 2] = str(meta.iloc[i, 2]).lower()
meta['Title'] = meta['Title'].astype(str) 
task1_4_rank['Title'] = task1_4_rank['Title'].astype(str) 

task1_4_rank = pd.merge(task1_4_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task1_4_rank.dropna(inplace=True)

# Extract the year from the string publish time
import dateutil.parser as parser
task1_4_rank['publish_time'] = task1_4_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))
task1_4_rank['publish_time'] = task1_4_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task1_4_rank['publish_time'] = pd.to_numeric(task1_4_rank['publish_time'])
task1_4_rank = task1_4_rank.sort_values(by='publish_time', ascending=False)
task1_4_rank.reset_index(inplace=True,drop=True)


# ### By affiliations score:

# In[ ]:


# Extract the affiliations score to the task's results:
task1_4_rank['Aff_Score'] = 0
for i in range(len(task1_4_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task1_4_rank.iloc[i, 4]:
            task1_4_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task1_4_rank["Ranking_Score"] = task1_4_rank["publish_time"]*0.8 + task1_4_rank["Aff_Score"]*0.2
task1_4_rank = task1_4_rank.sort_values(by='Ranking_Score', ascending=False)
task1_4_rank.reset_index(inplace=True,drop=True)
task1_4_rank


# In[ ]:


## 20 - Ranked Results for task 1.4 :

for i in range(len(task1_4_rank)):
    print("\n")
    print("PaperID: ", task1_4_rank.iloc[i, 0])
    print("Title: ", task1_4_rank.iloc[i, 1])
    print("Section: ", task1_4_rank.iloc[i, 2])
    print("Text: ", task1_4_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break


# ### Task 1.5: ***Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).***

# In[ ]:


keywords =['persistence', 'stability','multitude','substrates','sources','nasal','discharge','sputum','urine','fecal','blood']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   


# In[ ]:


# At least 7 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 6: 
        return(True)  
    return(False)    
  
task1_5 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task1_5.append(i)
    
len(task1_5)


# In[ ]:


## Results for task 1.5 :
for i in task1_5:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")


# ### Ranking by the most recent:

# In[ ]:


# Task 1.5 
print(task1_5)


# In[ ]:


task1_5_rank = papers_data.iloc[task1_5, :]
task1_5_rank.reset_index(inplace=True,drop=True)

# Grab the publish year from the meta data file:
meta = meta.rename(columns={"title": "Title"})
for i in range(len(meta)):
    meta.iloc[i, 2] = str(meta.iloc[i, 2]).lower()
meta['Title'] = meta['Title'].astype(str) 
task1_5_rank['Title'] = task1_5_rank['Title'].astype(str) 

task1_5_rank = pd.merge(task1_5_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task1_5_rank.dropna(inplace=True)

# Extract the year from the string publish time
import dateutil.parser as parser
task1_5_rank['publish_time'] = task1_5_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))
task1_5_rank['publish_time'] = task1_5_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task1_5_rank['publish_time'] = pd.to_numeric(task1_5_rank['publish_time'])
task1_5_rank = task1_5_rank.sort_values(by='publish_time', ascending=False)
task1_5_rank.reset_index(inplace=True,drop=True)


# ### By affilitions Scores:

# In[ ]:


# Extract the affiliations score to the task's results:
task1_5_rank['Aff_Score'] = 0
for i in range(len(task1_5_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task1_5_rank.iloc[i, 4]:
            task1_5_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task1_5_rank["Ranking_Score"] = task1_5_rank["publish_time"]*0.8 + task1_5_rank["Aff_Score"]*0.2
task1_5_rank = task1_5_rank.sort_values(by='Ranking_Score', ascending=False)
task1_5_rank.reset_index(inplace=True,drop=True)
task1_5_rank


# In[ ]:


## 20 - Ranked Results for task 1.5 :

for i in range(len(task1_5_rank)):
    print("\n")
    print("PaperID: ", task1_5_rank.iloc[i, 0])
    print("Title: ", task1_5_rank.iloc[i, 1])
    print("Section: ", task1_5_rank.iloc[i, 2])
    print("Text: ", task1_5_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break


# ### Task 1.6: ***Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).***

# In[ ]:


keywords =['persistence', 'virus','surface','material','copper','steel','stainlesss','plastic']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   


# In[ ]:


# At least 6 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 5: 
        return(True)  
    return(False)    
  
task1_6 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task1_6.append(i)
    
len(task1_6)


# In[ ]:


## Results for task 1.6 :
for i in task1_6:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")


# ### Ranking by the most recent:

# In[ ]:


# Task 1.6:
print(task1_6)


# In[ ]:


task1_6_rank = papers_data.iloc[task1_6, :]
task1_6_rank.reset_index(inplace=True,drop=True)

task1_6_rank['Title'] = task1_6_rank['Title'].astype(str) 

task1_6_rank = pd.merge(task1_6_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task1_6_rank.dropna(inplace=True)

# Extract the year from the string publish time
import dateutil.parser as parser
task1_6_rank['publish_time'] = task1_6_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))
task1_6_rank['publish_time'] = task1_6_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task1_6_rank['publish_time'] = pd.to_numeric(task1_6_rank['publish_time'])
task1_6_rank = task1_6_rank.sort_values(by='publish_time', ascending=False)
task1_6_rank.reset_index(inplace=True,drop=True)


# ### By affiliations Score:

# In[ ]:


# Extract the affiliations score to the task's results:
task1_6_rank['Aff_Score'] = 0
for i in range(len(task1_6_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task1_6_rank.iloc[i, 4]:
            task1_6_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task1_6_rank["Ranking_Score"] = task1_6_rank["publish_time"]*0.8 + task1_6_rank["Aff_Score"]*0.2
task1_6_rank = task1_6_rank.sort_values(by='Ranking_Score', ascending=False)
task1_6_rank.reset_index(inplace=True,drop=True)
task1_6_rank


# In[ ]:


## 20 - Ranked Results for task 1.6 :

for i in range(len(task1_6_rank)):
    print("\n")
    print("PaperID: ", task1_6_rank.iloc[i, 0])
    print("Title: ", task1_6_rank.iloc[i, 1])
    print("Section: ", task1_6_rank.iloc[i, 2])
    print("Text: ", task1_6_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break


# ### Task 1.7: ***Natural history of the virus and shedding of it from an infected person***

# In[ ]:


keywords =['natural', 'history','virus','shedding','infected','person']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   


# In[ ]:


# At least 6 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 5: 
        return(True)  
    return(False)    
  
task1_7 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task1_7.append(i)
    
len(task1_7)


# In[ ]:


## Results for task 1.7 :
for i in task1_7:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")


# ### Ranking by the most recent:

# In[ ]:


task1_7_rank = papers_data.iloc[task1_7, :]
task1_7_rank.reset_index(inplace=True,drop=True)

task1_7_rank['Title'] = task1_7_rank['Title'].astype(str) 

task1_7_rank = pd.merge(task1_7_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task1_7_rank.dropna(inplace=True)

# Extract the year from the string publish time
import dateutil.parser as parser
task1_7_rank['publish_time'] = task1_7_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))
task1_7_rank['publish_time'] = task1_7_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task1_7_rank['publish_time'] = pd.to_numeric(task1_7_rank['publish_time'])
task1_7_rank = task1_7_rank.sort_values(by='publish_time', ascending=False)
task1_7_rank.reset_index(inplace=True,drop=True)


# ### By affiliations Scores:

# In[ ]:


# Extract the affiliations score to the task's results:
task1_7_rank['Aff_Score'] = 0
for i in range(len(task1_7_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task1_7_rank.iloc[i, 4]:
            task1_7_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task1_7_rank["Ranking_Score"] = task1_7_rank["publish_time"]*0.8 + task1_7_rank["Aff_Score"]*0.2
task1_7_rank = task1_7_rank.sort_values(by='Ranking_Score', ascending=False)
task1_7_rank.reset_index(inplace=True,drop=True)
task1_7_rank


# In[ ]:


## 20 - Ranked Results for task 1.7 :

for i in range(len(task1_7_rank)):
    print("\n")
    print("PaperID: ", task1_7_rank.iloc[i, 0])
    print("Title: ", task1_7_rank.iloc[i, 1])
    print("Section: ", task1_7_rank.iloc[i, 2])
    print("Text: ", task1_7_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break


# ### Task 1.8: ***Implementation of diagnostics and products to improve clinical processes***

# In[ ]:


keywords =['diagnostic', 'product','clinical','process']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   


# In[ ]:


# At least 5 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 4: 
        return(True)  
    return(False)    
  
task1_8 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task1_8.append(i)
    
len(task1_8)


# In[ ]:


## Results for task 1.8 :
for i in task1_8:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")


# ### Ranking by the most recent:

# In[ ]:


task1_8_rank = papers_data.iloc[task1_8, :]
task1_8_rank.reset_index(inplace=True,drop=True)

task1_8_rank['Title'] = task1_8_rank['Title'].astype(str) 

task1_8_rank = pd.merge(task1_8_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task1_8_rank.dropna(inplace=True)

# Extract the year from the string publish time
import dateutil.parser as parser
task1_8_rank['publish_time'] = task1_8_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))
task1_8_rank['publish_time'] = task1_8_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task1_8_rank['publish_time'] = pd.to_numeric(task1_8_rank['publish_time'])
task1_8_rank = task1_8_rank.sort_values(by='publish_time', ascending=False)
task1_8_rank.reset_index(inplace=True,drop=True)


# ### By affiliation scores:

# In[ ]:


# Extract the affiliations score to the task's results:
task1_8_rank['Aff_Score'] = 0
for i in range(len(task1_8_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task1_8_rank.iloc[i, 4]:
            task1_8_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task1_8_rank["Ranking_Score"] = task1_8_rank["publish_time"]*0.8 + task1_8_rank["Aff_Score"]*0.2
task1_8_rank = task1_8_rank.sort_values(by='Ranking_Score', ascending=False)
task1_8_rank.reset_index(inplace=True,drop=True)
task1_8_rank


# In[ ]:


## 20 - Ranked Results for task 1.8 :

for i in range(len(task1_8_rank)):
    print("\n")
    print("PaperID: ", task1_8_rank.iloc[i, 0])
    print("Title: ", task1_8_rank.iloc[i, 1])
    print("Section: ", task1_8_rank.iloc[i, 2])
    print("Text: ", task1_8_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break


# ### Task 1.9: ***Disease models, including animal models for infection, disease and transmission***

# In[ ]:


keywords =['disease', 'model','animal','infection','transmission']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   


# In[ ]:


# At least 8 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 7: 
        return(True)  
    return(False)    
  
task1_9 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task1_9.append(i)
    
len(task1_9)


# In[ ]:


## Results for task 1.9 :
for i in task1_9:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")


# ### Ranking by the most recent:

# In[ ]:


task1_9_rank = papers_data.iloc[task1_9, :]
task1_9_rank.reset_index(inplace=True,drop=True)

task1_9_rank['Title'] = task1_9_rank['Title'].astype(str) 

task1_9_rank = pd.merge(task1_9_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task1_9_rank.dropna(inplace=True)

# Extract the year from the string publish time
import dateutil.parser as parser
task1_9_rank['publish_time'] = task1_9_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))
task1_9_rank['publish_time'] = task1_9_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task1_9_rank['publish_time'] = pd.to_numeric(task1_9_rank['publish_time'])
task1_9_rank = task1_9_rank.sort_values(by='publish_time', ascending=False)
task1_9_rank.reset_index(inplace=True,drop=True)


# ### By affiliations scores:

# In[ ]:


# Extract the affiliations score to the task's results:
task1_9_rank['Aff_Score'] = 0
for i in range(len(task1_9_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task1_9_rank.iloc[i, 4]:
            task1_9_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task1_9_rank["Ranking_Score"] = task1_9_rank["publish_time"]*0.8 + task1_9_rank["Aff_Score"]*0.2
task1_9_rank = task1_9_rank.sort_values(by='Ranking_Score', ascending=False)
task1_9_rank.reset_index(inplace=True,drop=True)
task1_9_rank


# In[ ]:


## 20 - Ranked Results for task 1.9 :

for i in range(len(task1_9_rank)):
    print("\n")
    print("PaperID: ", task1_9_rank.iloc[i, 0])
    print("Title: ", task1_9_rank.iloc[i, 1])
    print("Section: ", task1_9_rank.iloc[i, 2])
    print("Text: ", task1_9_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break


# ### Task 1.10: ***Tools and studies to monitor phenotypic change and potential adaptation of the virus***

# In[ ]:


keywords =['tool', 'studies','monitor','phenotypic','potential','adaptation','virus']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   


# In[ ]:


# At least 8 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 7: 
        return(True)  
    return(False)    
  
task1_10 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task1_10.append(i)
    
len(task1_10)


# In[ ]:


## Results for task 1.10 :
for i in task1_10:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")


# ### Ranking by the most recent:

# In[ ]:


task1_10_rank = papers_data.iloc[task1_10, :]
task1_10_rank.reset_index(inplace=True,drop=True)

task1_10_rank['Title'] = task1_10_rank['Title'].astype(str) 

task1_10_rank = pd.merge(task1_10_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task1_10_rank.dropna(inplace=True)

# Extract the year from the string publish time
import dateutil.parser as parser
task1_10_rank['publish_time'] = task1_10_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))
task1_10_rank['publish_time'] = task1_10_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task1_10_rank['publish_time'] = pd.to_numeric(task1_10_rank['publish_time'])
task1_10_rank = task1_10_rank.sort_values(by='publish_time', ascending=False)
task1_10_rank.reset_index(inplace=True,drop=True)


# ### By afffiliation scores:

# In[ ]:


# Extract the affiliations score to the task's results:
task1_10_rank['Aff_Score'] = 0
for i in range(len(task1_10_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task1_10_rank.iloc[i, 4]:
            task1_10_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task1_10_rank["Ranking_Score"] = task1_10_rank["publish_time"]*0.8 + task1_10_rank["Aff_Score"]*0.2
task1_10_rank = task1_10_rank.sort_values(by='Ranking_Score', ascending=False)
task1_10_rank.reset_index(inplace=True,drop=True)
task1_10_rank


# In[ ]:


## 20 - Ranked Results for task 1.10 :

for i in range(len(task1_10_rank)):
    print("\n")
    print("PaperID: ", task1_10_rank.iloc[i, 0])
    print("Title: ", task1_10_rank.iloc[i, 1])
    print("Section: ", task1_10_rank.iloc[i, 2])
    print("Text: ", task1_10_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break


# ### Task 1.11: ***Immune response and immunity***

# In[ ]:


keywords =['Immune','immunity','coronavirus','corona','covid']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   


# In[ ]:


# At least 4 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 3: 
        return(True)  
    return(False)    
  
task1_11 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task1_11.append(i)
    
len(task1_11)


# In[ ]:


## Results for task 1.11 :
j = 0
for i in task1_11:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")
    j = j + 1
    if j==10:
        break


# ### Ranking by the most recent:

# In[ ]:


task1_11_rank = papers_data.iloc[task1_11, :]
task1_11_rank.reset_index(inplace=True,drop=True)

task1_11_rank['Title'] = task1_11_rank['Title'].astype(str) 

task1_11_rank = pd.merge(task1_11_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task1_11_rank.dropna(inplace=True)

# Extract the year from the string publish time
import dateutil.parser as parser
task1_11_rank['publish_time'] = task1_11_rank['publish_time'].apply(lambda x:  str(x).replace('Oct 28 Mar-Apr',''))
task1_11_rank['publish_time'] = task1_11_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task1_11_rank['publish_time'] = pd.to_numeric(task1_11_rank['publish_time'])
task1_11_rank = task1_11_rank.sort_values(by='publish_time', ascending=False)
task1_11_rank.reset_index(inplace=True,drop=True)


# ### By affiliation scores:

# In[ ]:


# Extract the affiliations score to the task's results:
task1_11_rank['Aff_Score'] = 0
for i in range(len(task1_11_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task1_11_rank.iloc[i, 4]:
            task1_11_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task1_11_rank["Ranking_Score"] = task1_11_rank["publish_time"]*0.8 + task1_11_rank["Aff_Score"]*0.2
task1_11_rank = task1_11_rank.sort_values(by='Ranking_Score', ascending=False)
task1_11_rank.reset_index(inplace=True,drop=True)
task1_11_rank


# In[ ]:


## 20 - Ranked Results for task 1.11 :

for i in range(len(task1_11_rank)):
    print("\n")
    print("PaperID: ", task1_11_rank.iloc[i, 0])
    print("Title: ", task1_11_rank.iloc[i, 1])
    print("Section: ", task1_11_rank.iloc[i, 2])
    print("Text: ", task1_11_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break


# ### Task 1.12: ***Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings***

# In[ ]:


keywords =['Effectiveness', 'movement','control','strategies','prevent','secondary','transmission','health','care','community','settings']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   


# In[ ]:


# At least 13 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 12: 
        return(True)  
    return(False)    
  
task1_12 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task1_12.append(i)
    
len(task1_12)


# In[ ]:


## Results for task 1.12 :
for i in task1_12:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")


# ### Ranking by the most recent:

# In[ ]:


task1_12_rank = papers_data.iloc[task1_12, :]
task1_12_rank.reset_index(inplace=True,drop=True)

task1_12_rank['Title'] = task1_12_rank['Title'].astype(str) 

task1_12_rank = pd.merge(task1_12_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task1_12_rank.dropna(inplace=True)

# Extract the year from the string publish time
import dateutil.parser as parser
task1_12_rank['publish_time'] = task1_12_rank['publish_time'].apply(lambda x:  str(x).replace('May 8 Summer',''))
task1_12_rank['publish_time'] = task1_12_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task1_12_rank['publish_time'] = pd.to_numeric(task1_12_rank['publish_time'])
task1_12_rank = task1_12_rank.sort_values(by='publish_time', ascending=False)
task1_12_rank.reset_index(inplace=True,drop=True)


# ### By affiliations scores:

# In[ ]:


# Extract the affiliations score to the task's results:
task1_12_rank['Aff_Score'] = 0
for i in range(len(task1_12_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task1_12_rank.iloc[i, 4]:
            task1_12_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task1_12_rank["Ranking_Score"] = task1_12_rank["publish_time"]*0.8 + task1_12_rank["Aff_Score"]*0.2
task1_12_rank = task1_12_rank.sort_values(by='Ranking_Score', ascending=False)
task1_12_rank.reset_index(inplace=True,drop=True)
task1_12_rank


# In[ ]:


## 20 - Ranked Results for task 1.12 :

for i in range(len(task1_12_rank)):
    print("\n")
    print("PaperID: ", task1_12_rank.iloc[i, 0])
    print("Title: ", task1_12_rank.iloc[i, 1])
    print("Section: ", task1_12_rank.iloc[i, 2])
    print("Text: ", task1_12_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break


# ### Task 1.13: ***Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings***

# In[ ]:


keywords =['Effectiveness', 'protective','ppe','equipment','usefulness','risk','reduce','health','care','community','settings','transmission']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   


# In[ ]:


# At least 12 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 11: 
        return(True)  
    return(False)    
  
task1_13 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task1_13.append(i)
    
len(task1_13)


# In[ ]:


## Results for task 1.13 :
for i in task1_13:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")


# ### Ranking by the most recent:

# In[ ]:


task1_13_rank = papers_data.iloc[task1_13, :]
task1_13_rank.reset_index(inplace=True,drop=True)

task1_13_rank['Title'] = task1_13_rank['Title'].astype(str) 

task1_13_rank = pd.merge(task1_13_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task1_13_rank.dropna(inplace=True)

# Extract the year from the string publish time
import dateutil.parser as parser
task1_13_rank['publish_time'] = task1_13_rank['publish_time'].apply(lambda x:  str(x).replace('May 8 Summer',''))
task1_13_rank['publish_time'] = task1_13_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task1_13_rank['publish_time'] = pd.to_numeric(task1_13_rank['publish_time'])
task1_13_rank = task1_13_rank.sort_values(by='publish_time', ascending=False)
task1_13_rank.reset_index(inplace=True,drop=True)


# ### By affiliations scores:

# In[ ]:


# Extract the affiliations score to the task's results:
task1_13_rank['Aff_Score'] = 0
for i in range(len(task1_13_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task1_13_rank.iloc[i, 4]:
            task1_13_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task1_13_rank["Ranking_Score"] = task1_13_rank["publish_time"]*0.8 + task1_13_rank["Aff_Score"]*0.2
task1_13_rank = task1_13_rank.sort_values(by='Ranking_Score', ascending=False)
task1_13_rank.reset_index(inplace=True,drop=True)
task1_13_rank


# In[ ]:


## 20 - Ranked Results for task 1.13 :

for i in range(len(task1_13_rank)):
    print("\n")
    print("PaperID: ", task1_13_rank.iloc[i, 0])
    print("Title: ", task1_13_rank.iloc[i, 1])
    print("Section: ", task1_13_rank.iloc[i, 2])
    print("Text: ", task1_13_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break


# ### Task 1.14: ***Role of the environment in transmission***

# In[ ]:


keywords =['transmission','role','environment']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   


# In[ ]:


# At least 6 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 5: 
        return(True)  
    return(False)    
  
task1_14 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task1_14.append(i)
    
len(task1_14)


# In[ ]:


## Results for task 1.14 :
for i in task1_14:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")
    


# ### Ranking by the most recent:

# In[ ]:


task1_14_rank = papers_data.iloc[task1_13, :]
task1_14_rank.reset_index(inplace=True,drop=True)

task1_14_rank['Title'] = task1_14_rank['Title'].astype(str) 

task1_14_rank = pd.merge(task1_14_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task1_14_rank.dropna(inplace=True)

# Extract the year from the string publish time
import dateutil.parser as parser
task1_14_rank['publish_time'] = task1_14_rank['publish_time'].apply(lambda x:  str(x).replace('May 8 Summer',''))
task1_14_rank['publish_time'] = task1_14_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task1_14_rank['publish_time'] = pd.to_numeric(task1_14_rank['publish_time'])
task1_14_rank = task1_14_rank.sort_values(by='publish_time', ascending=False)
task1_14_rank.reset_index(inplace=True,drop=True)


# ### By the effiliations scores:

# In[ ]:


# Extract the affiliations score to the task's results:
task1_14_rank['Aff_Score'] = 0
for i in range(len(task1_14_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task1_14_rank.iloc[i, 4]:
            task1_14_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task1_14_rank["Ranking_Score"] = task1_14_rank["publish_time"]*0.8 + task1_14_rank["Aff_Score"]*0.2
task1_14_rank = task1_14_rank.sort_values(by='Ranking_Score', ascending=False)
task1_14_rank.reset_index(inplace=True,drop=True)
task1_14_rank


# In[ ]:


## 20 - Ranked Results for task 1.14 :

for i in range(len(task1_14_rank)):
    print("\n")
    print("PaperID: ", task1_14_rank.iloc[i, 0])
    print("Title: ", task1_14_rank.iloc[i, 1])
    print("Section: ", task1_14_rank.iloc[i, 2])
    print("Text: ", task1_14_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break


# ### Save all sub-tasks results:

# In[ ]:


task1_1_rank.to_csv("task1_1_rank.csv")
task1_2_rank.to_csv("task1_2_rank.csv")
task1_3_rank.to_csv("task1_3_rank.csv")
task1_4_rank.to_csv("task1_4_rank.csv")
task1_5_rank.to_csv("task1_5_rank.csv")
task1_6_rank.to_csv("task1_6_rank.csv")
task1_7_rank.to_csv("task1_7_rank.csv")
task1_8_rank.to_csv("task1_8_rank.csv")
task1_9_rank.to_csv("task1_9_rank.csv")
task1_10_rank.to_csv("task1_10_rank.csv")
task1_11_rank.to_csv("task1_11_rank.csv")
task1_12_rank.to_csv("task1_12_rank.csv")
task1_13_rank.to_csv("task1_13_rank.csv")
task1_14_rank.to_csv("task1_14_rank.csv")

