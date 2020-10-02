#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Risk Factors Analysis
# 
# ## Aim of this Notebook
# To understand the risk factors surrounding COVID-19
# 
# **Task Details (Taken from [Task Home Page](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=558))**  
# What do we know about COVID-19 risk factors? What have we learned from epidemiological studies?
# 
# 1. Data on potential risks factors  
#     **a.** Smoking, pre-existing pulmonary disease  
#     **b.** Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other            co-morbidities  
#     **c.** Neonates and pregnant women  
#     **d.** Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.  
# 2. Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors
# 3. Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups
# 4. Susceptibility of populations
# 5. Public health mitigation measures that could be effective for control
# 
# ### Main Goal
# The main goal of this notebook will be to filter out answers to those questions asked in the [CORD-19 Task](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=558) as mentioned above. 
# 
# ## Acknowledgements
# 
# - I thank [xhlulu](https://www.kaggle.com/xhlulu) for cleaning up the original json files in the [CORD Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) and providing the data in a .csv format.  
# His kernel can be found [here](https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv).

# ## Methodology
# 
# ### I. Data Cleaning
#   
# ### II. Finding the Necessary Papers based on simple "Keyword" search
#   
# ### III. Analyzing papers that contain words like "pulmonary" or "smoking"
# 
# ### IV. Analyzing papers that contain words related to "pregnancy" and "newborns"

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re


# In[ ]:


# Load the datasets
bio = pd.read_csv("/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv")
noncomm = pd.read_csv("/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv")
comm = pd.read_csv("/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv")
pmc = pd.read_csv("/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv")


# ## I. Data Cleaning
# Making the data more usable for the analysis

# ### Missing Value Imputation
# 
# At this point, I am replacing all the NaN values with "Missing".  
# First, let's see how many missing values are even there.

# In[ ]:


# Missing Value Visualization

fig, axes = plt.subplots(2 ,2, figsize=(30,15))

title = "Biorxiv"
ax = sns.heatmap(bio.isnull(), cmap="Reds", cbar=False, ax=axes[0,0])
ax.vlines([1,2,3,4,5,6,7,8,9], *ax.get_ylim(), color="black")
axes[0,0].set_title(title, fontsize=15)

title = "Non-commercial Use"
ax = sns.heatmap(noncomm.isnull(), cmap="Reds", cbar=False, ax=axes[0,1])
ax.vlines([1,2,3,4,5,6,7,8,9], *ax.get_ylim(), color="black")
axes[0,1].set_title(title, fontsize=15)

title = "Commercial Use"
ax = sns.heatmap(comm.isnull(), cmap="Reds", cbar=False, ax=axes[1,0])
ax.vlines([1,2,3,4,5,6,7,8,9], *ax.get_ylim(), color="black")
axes[1,0].set_title(title, fontsize=15)

title = "PMC"
ax = sns.heatmap(pmc.isnull(), cmap="Reds", cbar=False, ax=axes[1,1])
ax.vlines([1,2,3,4,5,6,7,8,9], *ax.get_ylim(), color="black")
axes[1,1].set_title(title, fontsize=15)

fig.suptitle("Missing Value Heatmaps for all 4 datasets", fontsize=20)
plt.show()


# - The Dark Red horizontal lines indicate missing values
# - To get rid of the NaNs, all NaNs will be imputed with "Missing"

# In[ ]:


# Impute NaNs with "Missing"

bio = bio.fillna("Missing")
noncomm = noncomm.fillna("Missing")
comm = comm.fillna("Missing")
pmc = pmc.fillna("Missing")


# In[ ]:


# Concatenate all the dataframes together
papers = pd.concat([bio, comm, noncomm, pmc], ignore_index=True)


# The imputation has been performed.  
# 
# More steps for data cleaning will be added at a later stage if required.  
# 
# **NOTE :** The dataframe ***papers*** contains all the papers. This will be used for analysis.

# ### Cleaning the Textual Values
# 
# In this part, I am performing some rudimentary text cleaning. The following steps are taken :  
# - Remove newline characters
# - Remove citation numbers like [5]
# - Remove et. al
# - Remove Fig and Table citations like (Fig 5) or ( Table 6 )
# - Replace continuous spaces with a single space
# - Convert all alphabets to lower case

# In[ ]:


# Data Cleaning
def clean_up(t):
    """
    Cleans up the passed value
    """
    # Remove New Lines
    t = t.replace("\n"," ") # removes newlines

    # Remove citation numbers (Eg.: [4])
    t = re.sub("\[[0-9]+(, [0-9]+)*\]", "", t)

    # Remove et al.
    t = re.sub("et al.", "", t)

    # Remove Fig and Table
    t = re.sub("\( ?Fig [0-9]+ ?\)", "", t)
    t = re.sub("\( ?Table [0-9]+ ?\)", "", t)
    
    # Replace continuous spaces with a single space
    t = re.sub(' +', ' ', t)
    
    # Convert all to lowercase
    t = t.lower()
    
    return t

papers["abstract"] = papers["abstract"].apply(clean_up)
papers["text"] = papers["text"].apply(clean_up)


# ## II. Finding the Necessary Papers based on simple "Keyword" search

# The biggest dilemma is to select the right kind of papers to get the right information pertaining to a given task. This section deals with re-usable code that can help find the right papers based on specific keywords we are looking for.  
# 
# **How to use this section?**
# 
# - Step 1. Choose your dataframe (You can either choose one of the 4 initial dataframes loaded or select ***papers*** which consists of all papers (I will be using the latter in this notebook)
# - Step 2. Choose a specific keyword you are looking for (Eg. :***smoking***)
# - Step 3. Run the select_papers() function with necessary arguments
#     - The original dataframe will now have 3 more features, each feature depicting the presence of the given keyword in title, abstract or text respectively.
# 

# In[ ]:


# Utility Functions for this Section

def word_occurence(entry, word):
    """
    Identifies if a given word exists in a dataframe's entry
    or not
    
    Parameters
    ----------
    entry : The entry OR cell or value
    
    Returns
    -------
    0 if the word is not found
    1 if it the word is found
    """
    # convert to lower case for uniformity
    word=word.lower()
    
    if(word in entry.lower()):
        return 1
    else:
        return 0
    
def select_papers(dataframe, keyword):
    """
    Creates new features in a dataframe depicting
    the existence of the given keyword
    
    Parameters
    ----------
    dataframe : The dataframe in which you are searching
                for
    keyword : The keyword that you are searching for
    
    Returns
    -------
    The new dataframe with the newly created 
    feature/columns
    """
    # title
    feature_header = keyword+"_exists_title"
    dataframe[feature_header] = dataframe["title"].apply(word_occurence, word=keyword)
    
    # abstract
    feature_header = keyword+"_exists_abstract"
    dataframe[feature_header] = dataframe["abstract"].apply(word_occurence, word=keyword)
    
    # text
    feature_header = keyword+"_exists_text"
    dataframe[feature_header] = dataframe["text"].apply(word_occurence, word=keyword)
    
    return dataframe


# **NOTE :** Essentially what the select_papers() function is doing is **Introducing 3 new features everytime for every new word that is being searched for**. These features correspond to the presence of a given word in
# - Title of the Paper
# - Abstract of the Paper
# - Text of the Paper
# ***If the word exists, it is represented with a 1. Else, it's a 0.***
# 
# **Why am I doing this?**  
# An intuitive way to look at this is to maintain a kind of record within the dataframe itself to understand at a later stage what papers contain a certain word or not.

# ---
# ### Example of Working of the above Idea

# In[ ]:


# Example

bio = select_papers(bio, "risk")
bio.head()


# The **bio** dataframe has got 3 new features based on the existence of the given keyword. In this case, it is ***risk***.  
# 
# Based on these features, we can subset the dataframe into relevant papers and analyse.

# In[ ]:


# restore to normal
bio = bio.drop(["risk_exists_title", "risk_exists_abstract", "risk_exists_text"], axis=1)


# The old dataframe has been restored back to normal.  
# 
# ---

# ## III. Analyzing Papers that contain words like "pulmonary" or "smoking"

# In[ ]:


# Utility Functions for this Section

def display_papers(dataframe):
    """
    Displays all the papers in a 
    data subset obtained like 
    bio_pulmonary or bio_smoking
    
    Parameters
    ----------
    dataframe : The dataframe
    
    Returns
    -------
    Prints all paper titles and paper ids
    in a given dataframe
    """
    papers = ";".join(comment for comment in dataframe["title"])
    paper_ids = ";".join(comment for comment in dataframe["paper_id"])
    papers = papers.split(";")
    paper_ids = paper_ids.split(";")
    for p,p_id in zip(papers, paper_ids):
        print("-> ",p," ( Paper ID :", p_id,")")
    print("----------")
           
def gen_wordcloud(df, feature, remove_list=[]):
    """
    Generate word clouds for a given feature
    
    Parameters
    ----------
    df : Dataframe
    feature : Feature in the Dataframe
    remove_list : List of words that need to be removed
    
    Returns
    -------
    Displays the wordcloud
    """
    words = " ".join(comment for comment in df[feature])
    words = words.lower()
    for w in remove_list:
        words = words.replace(w,"")
    
    wordcloud = WordCloud(max_words=1000,background_color="white", width=800, height=800,
                     contour_width=3, contour_color='firebrick').generate(words)
    # Display the generated image:
    plt.figure(figsize=[15,15])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(feature)
    plt.axis("off")
    plt.show()
    
def choosing_sentences(text, word):
    """
    Function to choose sentences from a text
    passage/string based on the existence of a
    given word in these strings
    
    Parameters
    ----------
    text : The text passage
    word : Word to search for in the text
    
    Returns
    -------
    Sentences that contain the word
    """
    # Initializing empty list
    qualified = []
    
    text = text.split(".")
    text = [x.strip() for x in text]
    for i in text:
        if(word in i):
            qualified.append(i)
    return qualified

def extract_impt_sentences(dataframe, identifier, feature_list, keyword):
    """
    Select important sentences from textual features that
    contain a specific keyword
    
    Parameters
    ----------
    dataframe : The Dataframe
    identifier : The feature that has the ability
                 to uniquely identify each row
    feature_list : The features that are to be 
                   searched for the given keyword
    keyword : The keyword to search for; every sentence
              having this keyword should be stored in a
              list and returned
    
    Returns
    -------
    Dictionary with 
    Keys : Paper titles
    Values : Sentences that have the word in the given paper
    """
    # number of rows
    n_rows = dataframe.shape[0]
    documents = {}
    
    for f in feature_list:
        for i in range(dataframe.shape[0]):
            u_id = dataframe[identifier].iloc[i]
            qualifying_sentences = choosing_sentences(dataframe[f].iloc[i], keyword)
            # ignore papers where there are NO QUALIFYING SENTENCES
            if(len(qualifying_sentences) != 0):
                documents[u_id] = qualifying_sentences
            else:
                pass
    return (documents)

def choose_sentences_based_on_words(list_sentences, list_words):
    """
    Chooses all sentences in the list_sentences that consist of
    words from list_words
    """
    q = []
    for i in list_sentences:
        flag=0
        for j in list_words:
            if(not(j in i)):
                flag+=1
        if(flag==0):
            q.append(i)
    return q


# In[ ]:


# Selecting papers
papers = select_papers(papers, "smoking")
papers = select_papers(papers, "pulmonary")

# Subsetting
papers_smoking = papers[(papers["smoking_exists_title"]==1) | (papers["smoking_exists_abstract"]==1) | (papers["smoking_exists_text"]==1)]
papers_pulmonary = papers[(papers["pulmonary_exists_title"]==1) | (papers["pulmonary_exists_abstract"]==1) | (papers["pulmonary_exists_text"]==1)]


# > There are 1,191 papers with either "smoking" in the title, abstract or text.  
# > There are 5,012 papers with the word "pulmonary" in the title, abstract or text.  

# Let's have a look at 10 papers each of thoe that contain the word "smoking" or "pulmonary".

# In[ ]:


# Display papers in which "pulmonary" is mentioned atleast once in the title, abstract or text
print("10 Papers in which the word 'pulmonary' is mentioned :\n")
display_papers(papers_pulmonary.sample(10))

# Display papers in which "smoking" is mentioned atleast once in the title, abstract or text
print("\n10 Papers in which the word 'smoking' is mentioned :\n")
display_papers(papers_smoking.sample(10))


# In[ ]:


pulmonary_docs = extract_impt_sentences(papers_pulmonary, "title", ["abstract", "text"], "pulmonary")
smoking_docs = extract_impt_sentences(papers_smoking, "title", ["abstract", "text"], "smoking")


# ### Basic Wordcloud Analysis
# 
# **NOTE :** This section has been temporarily commented out as running it exceeds the memory capacity. I shall rectify that error and run this section again.

# In[ ]:


# Wordcloud Analysis on the "text" feature of papers_pulmonary
"""
print("papers_pulmonary")
gen_wordcloud(
    papers_pulmonary,
    "text",
    ["rights", "reserved", "supplementary", "fig", "holder", "preprint",
     "copyright", "peer", "reviewed", "et al", "reviewed", "permission",
    "using", "funder"])

# Wordcloud Analysis on the "text" feature of papers_smoking

print("papers_smoking")
gen_wordcloud(
    papers_smoking,
    "text",
    ["rights", "reserved", "supplementary", "fig", "holder", "preprint",
     "copyright", "peer", "reviewed", "et al", "reviewed", "permission",
    "using", "funder", "author", "license", "international", "made",
    "available", "medrxiv", "doi", "org"])
"""


# ---
# 
# ### Risks associated with Pulmonary Infections in the case of COVID-19
# 
# - The following sentence pools (i.e list of sentences) that are associated with a unique identifier(paper title) are those sentences where the word ***pulmonary and risk*** occur together
# - There are 439 such papers
# 
# Now, what is important for analysis is to understand the **Impact of pre-existing pulmonary disease on COVID-19**. For this purpose, I choose/select those sentences from the sentence pools in such fashion that only those sentencs which have either of the ***covid representation words*** will be chosen/selected.  
# 
# ***covid representation words*** : Just a term I have given to those words that represent the covid disease. From a preliminary observation, it has come to my notice that the following words/patterns are used to refer to the covid virus in multiple papers :
# - -cov
# - cov-
# - hcov
# - coronavirus
# 
# **NOTE :** More words that represent covid shall be added here when I find more. If you happen to find some that are not considered by me, do let me know. I shall add them to my code too.  
# 
# 
# After applying the above mentioned filtering, we have 7 papers with qualifying sentences.

# In[ ]:


# Sentences that contain both the words "pulmonary" and "risk" and any word of ("-cov", "cov-", "hcov", "coronavirus")

pulmonary_docs_risk = {}
for u_id,sent in pulmonary_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "-cov"])
    if(len(s)!=0):
        pulmonary_docs_risk[u_id] = s

for u_id,sent in pulmonary_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "cov-"])
    if(len(s)!=0):
        pulmonary_docs_risk[u_id] = s
        
for u_id,sent in pulmonary_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "hcov"])
    if(len(s)!=0):
        pulmonary_docs_risk[u_id] = s
        
for u_id,sent in pulmonary_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "coronavirus"])
    if(len(s)!=0):
        pulmonary_docs_risk[u_id] = s
        
for u_id,sent in pulmonary_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "covid"])
    if(len(s)!=0):
        pulmonary_docs_risk[u_id] = s

for u_id,sentence in pulmonary_docs_risk.items():
    print("Paper Title : "+u_id)
    print(sentence)
    print()


# There are some very key points that can be taken away from the sentences filtered out above like the following :
# - Risk of severe pulmonary disease in persons who fail to develop a neutralizing antibody response following exposure to mers-cov
# - In mers-cov infection, the important risk factors factors for death are old age (>50-65 years, depending on the study), underlying diseases (cardiac disease, chronic pulmonary disease, diabetes, chronic renal disease, etc
# 
# ---

# ### Risks associated with Smoking in the case of COVID-19
# 
# - The following sentence pools (i.e list of sentences) that are associated with a unique identifier(paper title) are those sentences where the word ***smoking and risk*** occur together
# - There are 322 such papers
# 
# After applying the filtering like in the previous section, I get 4 papers.

# In[ ]:


# Sentences that contain both the words "smoking" and "risk"

smoking_docs_risk = {}
for u_id,sent in smoking_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "cov-"])
    if(len(s)!=0):
        smoking_docs_risk[u_id] = s
        
for u_id,sent in smoking_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "-cov"])
    if(len(s)!=0):
        smoking_docs_risk[u_id] = s
        
for u_id,sent in smoking_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "hcov"])
    if(len(s)!=0):
        smoking_docs_risk[u_id] = s
        
for u_id,sent in smoking_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "coronavirus"])
    if(len(s)!=0):
        smoking_docs_risk[u_id] = s
        
for u_id,sent in smoking_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "covid"])
    if(len(s)!=0):
        smoking_docs_risk[u_id] = s

for u_id,sentence in smoking_docs_risk.items():
    print("Paper Title : "+u_id)
    print(sentence)
    print()


# A few of the associated key points are :
# - Other case-control and retrospective observational studies from both ksa and korea have suggested that **smoking and/or comorbid respiratory diseases** are **significant risk factors for mers-cov-related mortality**
# - In saudi arabia, the **probable risk factors for mers-cov** infection were older age , male sex , exposure to dromedary camels, comorbidities and **smoking**

# ---
# ## Analyzing papers that contain words related to "pregnancy" and "newborns"

# In[ ]:


# Selecting papers
papers = select_papers(papers, "women")
papers = select_papers(papers, "pregnancy")
papers = select_papers(papers, "pregnant")
papers = select_papers(papers, "newborn")
papers = select_papers(papers, "neonate")

# Subsetting
papers_preg_new = papers[(papers["women_exists_title"]==1) | (papers["women_exists_abstract"]==1) | (papers["women_exists_text"]==1) | (papers["pregnancy_exists_title"]==1) | (papers["pregnancy_exists_abstract"]==1) | (papers["pregnancy_exists_text"]==1) | (papers["pregnant_exists_title"]==1) | (papers["pregnant_exists_abstract"]==1) | (papers["pregnant_exists_text"]==1) | (papers["neonate_exists_title"]==1) | (papers["neonate_exists_abstract"]==1) | (papers["neonate_exists_text"]==1) | (papers["newborn_exists_title"]==1) | (papers["newborn_exists_abstract"]==1) | (papers["newborn_exists_text"]==1)]


# 10 papers in which "women", "pregnancy", "pregnant", "newborn" or "neonate" are mentioned atleast once

# In[ ]:


# Display papers in which "women", "pregnancy", "pregnant", "newborn" or "neonate" are mentioned atleast once
print("10 Papers in which \"women\", \"pregnancy\", \"pregnant\", \"newborn\" or \"neonate\" are mentioned atleast once:\n")
display_papers(papers_preg_new.sample(10))


# ---
# ### Risks associated with Pregnancy or Pregnant women in the case of COVID

# In[ ]:


preg_docs = extract_impt_sentences(papers_preg_new, "title", ["abstract", "text"], "pregnan") # to relate to any pregnan- word

# Sentences that contain both the words "pulmonary" and "risk" and any word of ("-cov", "cov-", "hcov", "coronavirus")

preg_docs_risk = {}
for u_id,sent in preg_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "-cov"])
    if(len(s)!=0):
        preg_docs_risk[u_id] = s

for u_id,sent in preg_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "cov-"])
    if(len(s)!=0):
        preg_docs_risk[u_id] = s
        
for u_id,sent in preg_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "hcov"])
    if(len(s)!=0):
        preg_docs_risk[u_id] = s
        
for u_id,sent in preg_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "coronavirus"])
    if(len(s)!=0):
        preg_docs_risk[u_id] = s
        
for u_id,sent in preg_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "covid"])
    if(len(s)!=0):
        preg_docs_risk[u_id] = s

for u_id,sentence in preg_docs_risk.items():
    print("Paper Title : "+u_id)
    print(sentence)
    print()


# A few of the associated key points are :
# 
# - Coronaviruses can result in maternal death in a small but significant number of cases, but the **specific risk factors for a fatal outcome during pregnancy have not been clarified**
# - Pregnant women are conventionally considered a high-risk group for the progression to severe disease or death, and a case was reported of stillbirth in the second trimester of pregnancy for a woman infected with mers-cov
# - According to current recommendations that are reviewed yearly, because of the high mortality rates associated with mers-cov infection, people with the following risk factors should postpone hajj or umrah for their own safety: individuals who are older than 65 years of age; individuals who have chronic diseases including diabetes, heart disease, kidney disease, respiratory disease, autoimmune disease, or immune defi ciency (congenital and acquired); people who are taking immunosuppressive drugs; individuals who have a malignant disease or a terminal illness; pregnant women; and children younger than 12 years
# - Given the maternal physiologic and immune function changes in pregnancy , pregnant individuals might face greater risk of getting infected by sars-cov-2 and might have more complicated clinical events
# - **Pregnant women are at an increased risk** to suffer from acute and chronic viral infections such as rhinovirus, severe acute respiratory syndrome, **coronavirus**, varicella zoster, hepatitis e/b, hiv, and cytomegalovirus
# 

# ---
# ### Points associated with Neonates and Newborns in the case of COVID

# In[ ]:


new_docs = extract_impt_sentences(papers_preg_new, "title", ["abstract", "text"], "newborn")
neo_docs = extract_impt_sentences(papers_preg_new, "title", ["abstract", "text"], "neonate") 

# Sentences that contain both the words "pulmonary" and "risk" and any word of ("-cov", "cov-", "hcov", "coronavirus")

new_docs_risk = {}
for u_id,sent in new_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "-cov"])
    if(len(s)!=0):
        new_docs_risk[u_id] = s

for u_id,sent in new_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "cov-"])
    if(len(s)!=0):
        new_docs_risk[u_id] = s
        
for u_id,sent in new_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "hcov"])
    if(len(s)!=0):
        new_docs_risk[u_id] = s
        
for u_id,sent in new_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "coronavirus"])
    if(len(s)!=0):
        new_docs_risk[u_id] = s
        
for u_id,sent in new_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "covid"])
    if(len(s)!=0):
        new_docs_risk[u_id] = s
    
for u_id,sent in neo_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "-cov"])
    if(len(s)!=0):
        new_docs_risk[u_id] = s

for u_id,sent in neo_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "cov-"])
    if(len(s)!=0):
        new_docs_risk[u_id] = s
        
for u_id,sent in neo_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "hcov"])
    if(len(s)!=0):
        new_docs_risk[u_id] = s
        
for u_id,sent in neo_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "coronavirus"])
    if(len(s)!=0):
        new_docs_risk[u_id] = s
        
for u_id,sent in neo_docs.items():
    s = choose_sentences_based_on_words(sent, ["risk", "covid"])
    if(len(s)!=0):
        new_docs_risk[u_id] = s

for u_id,sentence in new_docs_risk.items():
    print("Paper Title : "+u_id)
    print(sentence)
    print()


# A few of the associated key points are :
# 
# - the mother should not breastfeed until she has recovered from sars or is deemed not to have sars
# - to minimize neonatal transmission risk, the mother should be isolated from the neonate until she is no longer potentially infectious communications
# - neonates born to mothers with potential sars close contact are considered to be potentially infectious until 10 days postpartum
# - Pregnant women and newborn babies should be considered key atrisk populations in strategies focusing on prevention and management of covid19 infection

# ### My next steps would be to 
# - Fix the wordcloud
# - Deal with other points to be covered in the Tasks homepage for this task
