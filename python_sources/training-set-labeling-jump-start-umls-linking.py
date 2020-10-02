#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# ## Goal
# Provide a shortcut for subject matter expert labeling of the document dataset by providing a first draft of concept labels for each document. Expert-labeled data sets will be the basis for supervised learning in the future. *Feel free to take this data and make it better!*
# 
# ---
# ### Document Labeling Output Example
# **Document ID:** 23f5f98892b12255b20f4b0a0ec1f3e7b29d5d72
# 
# **Abstract:** Angiotensin-converting enzyme 2 (ACE2) shares some homology with angiotensin-converting enzyme (ACE) but is not inhibited by ACE inhibitors. The main role of ACE2 is the degradation of Ang II resulting in the formation of angiotensin 1-7 (Ang 1-7) which opposes the actions of Ang II. Increased Ang II levels are thought to upregulate ACE2 activity, and in ACE2 deficient mice Ang II levels are approximately double that of wild-type mice, whilst Ang 1-7 levels are almost undetectable. Thus, ACE2 plays a crucial role in the RAS because it opposes the actions of Ang II. Consequently, it has a beneficial role in many diseases such as hypertension, diabetes, and cardiovascular disease where its expression is decreased. Not surprisingly, current therapeutic strategies for ACE2 involve augmenting its expression using ACE2 adenoviruses, recombinant ACE2 or compounds in these diseases thereby affording some organ protection. ACE2 is a type 1 integral membrane glycoprotein [8] that is expressed and active in most tissues. The highest expression of ACE2 is observed in the kidney, the endothelium, the lungs, and in the heart [2, 8] . The extracellular domain of ACE2 enzyme contains a single catalytic metallopeptidase unit that shares 42% sequence identity and 61% sequence similarity with the catalytic domain of ACE [2].
# 
# **Ranked Labels**
# 1. ACE protein, human
#   * CUI: C1452534
#   * score: 2.068973
# 2. angiotensin converting enzyme 2
#   * CUI: C0960880
#   * score: 1.985163
# 3. Peptidyl-Dipeptidase A
#   * CUI: C0022709
#   * score: 1.798690
# 4. Angiotensin converting enzyme measurement
#   * CUI: C0201888
#   * score: 1.746387
# ---
# 
# ## How?
# First, I apply the [UMLS](https://en.wikipedia.org/wiki/Unified_Medical_Language_System) linking alpha feature available in SpaCy/scispacy to the abstracts in the dataset. Then, I compute the tf-idf scores for each term in each document. Lastly, I provide a ranked list of UMLS labels for each document, saved in a CSV.
# 
# ## Approaches Used
# The scispacy UMLS linking alpha feature as described [here](https://github.com/allenai/scispacy#umlsentitylinker-alpha-feature). The tf-idf approach is modeled after the scikit-learn TfidfTransformer class [here](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
# 
# ## Outcome
# For each document with an abstract, a set of ranked labels from the UMLS controlled vocabulary is provided. From these, experts can review which labels are correct, incorrect, or missing. Participants can also filter using the labels to create subsets of the data that may correspond to particular questions.
# 
# ## Data Created
# 
# Three files are created:
#  * `tfidf_named_{date}.csv` contains, for each document:
#    * A list of applied concepts from the UMLS
#    * The term frequency (raw count)
#    * The idf value of the concept within the dataset of abstracts
#    * The tf-idf score for that concept for that document
#  * `concepts_{date}.csv` contains, for each concept used in the project:
#    * The identifier (CUI)
#    * The definition (if available)
#    * Raw counts
#    * idf values
#    * Average score throughout the dataset of abstracts
#  * `covid_text_{date}.csv` contains, for each document:
#    * Document ID
#    * Title
#    * Abstract
#    * Full text
# 
# ## Next steps
# * **Engage subject matter experts to ...**
#   * **Review labels and verify their application**
#   * **Edit and improve the labels for the dataset**
#   * **Choose important labels that correspond to questions in the competition**
# * Engage machine learning expertise to take labeled data and begin model creation
# * Work on threshold/floor for tf-idf scores to automatically avoid misleading labels
# * Use linked data sources for related taxonomies to establish hierarchical relationships between concepts
# * Generate new scores as new documents arrive
# * Generate scores based off of full text
# 
# *Please let me know what you think, and don't be afraid to reach out. Thanks! - Brian*
# 
# ## Load abstracts into DataFrame
# 
# Iterating through the JSON files in each directory, a DataFrame is built that records the split abstract information as a single string.
# 

# In[ ]:


from glob import glob
import json
import pandas as pd
from tqdm.notebook import tqdm

dir_list = [
    '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv',
    '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset',
    '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license',
    '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset'
]

results_list = list()
for target_dir in dir_list:
    
    print(target_dir)
    
    for json_fp in tqdm(glob(target_dir + '/*.json')):

        with open(json_fp) as json_file:
            target_json = json.load(json_file)

        data_dict = dict()
        data_dict['doc_id'] = target_json['paper_id']
        data_dict['title'] = target_json['metadata']['title']

        abstract_section = str()
        for element in target_json['abstract']:
            abstract_section += element['text'] + ' '
        data_dict['abstract'] = abstract_section

        full_text_section = str()
        for element in target_json['body_text']:
            full_text_section += element['text'] + ' '
        data_dict['full_text'] = full_text_section
        
        results_list.append(data_dict)
    
df_results = pd.DataFrame(results_list)
df_results


# ## Save the DataFrame as CSV

# In[ ]:


df_results.to_csv('covid_text_20200322.csv', index=False)


# ## Install scispacy and the en_core_sci_sm modules

# In[ ]:


get_ipython().system('pip install scispacy')
get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz')


# ## Create scispacy pipeline
# 
# * Using small core science module
# * Adding abbreviations detector to pipe
# * Adding UMLS entity linker to pipe

# In[ ]:


import spacy
import scispacy
import en_core_sci_sm
from scispacy.umls_linking import UmlsEntityLinker

nlp = en_core_sci_sm.load()


# In[ ]:


from scispacy.abbreviation import AbbreviationDetector

linker = UmlsEntityLinker(resolve_abbreviations=True)
abbreviation_pipe = AbbreviationDetector(nlp)

nlp.add_pipe(abbreviation_pipe)
nlp.add_pipe(linker)


# ## Filter documents to only those with abstracts

# In[ ]:


df_abx = df_results.loc[df_results['abstract'] != '']
df_abx


# # Extracting concepts for each document
# 
# The UMLS linker will provide ~200 classifications for each abstract, with corresponding scores for each classification. A set of dictionaries track the classifications per document:
# 
# `umls_dict` stores UMLS concept IDs (CUIs) and their scores overall
# `umls_df` tracks how many documents have each term
# `classify_dict` tracks CUIs per document
# 

# In[ ]:


umls_dict = dict()  # Track CUIs and scores
umls_df = dict()  # Track document frequency
classify_dict = dict()  # Track indexing per document

for row in tqdm(df_abx.itertuples(), total=df_abx.shape[0]):
       
    if row.abstract is not None:
        
        doc = nlp(row.abstract)   

        umls_set = set()
        
        classify_dict[row.doc_id] = dict()
        
        for entity in doc.ents:
            
            for umls_entity in entity._.umls_ents:
                
                if umls_entity[0] not in umls_dict:
                    umls_dict[umls_entity[0]] = [umls_entity[1]]
                else:
                    umls_dict[umls_entity[0]].append(umls_entity[1])
                    
                umls_set.add(umls_entity[0])
                
                if umls_entity[0] not in classify_dict[row.doc_id]:
                    classify_dict[row.doc_id][umls_entity[0]] = [umls_entity[1]]
                else:
                    classify_dict[row.doc_id][umls_entity[0]].append(umls_entity[1])
                    
        for entity in umls_set:

            if entity not in umls_df:
                umls_df[entity] = 1
            else:
                umls_df[entity] += 1
    


# ## Count, average, and idf values per concept
# 
# Iterate through the `umls_dict` dictionary of CUIs, calculating the total count of applications, the average score from the linker, and the idf value to be used later.

# In[ ]:


from statistics import mean 
import math

umls_results = list()

umls_idf_lookup = dict()

for cui, scores in tqdm(umls_dict.items()):
    
    umls_idf_lookup[cui] = {
        'total_count': len(scores),
        'idf': math.log(df_abx.shape[0] / umls_df[cui]),  # No smoothing? Entire population known?
        'average_score': mean(scores)
    }


# ## Calculate tf-idf scores for each concept for each document
# 
# Using the data from above, take each document's concepts and calculate the tf-idf score.
# 
# Store the information in a DataFrame.

# In[ ]:


results_list_tfidf = list()
df_tfidf = pd.DataFrame()

for doc_id, classify_data in tqdm(classify_dict.items()):
    
    doc_list = list()
    radicand = 0
    
    if len(classify_data) == 0:
        continue
    
    for cui, scores in classify_data.items():
                
        doc_list.append({
            'cui': cui,
            'tf': len(scores),
            'idf': umls_idf_lookup[cui]['idf'],
            'doc_id': doc_id
        })
        
        radicand += len(scores) ** 2
        
    denominator = math.sqrt(radicand)
    
    for x in doc_list:
        x['tf_idf'] = (x['tf'] * x['idf']) / denominator
        results_list_tfidf.append(x)
    
df_tfidf = pd.DataFrame(results_list_tfidf)
df_tfidf


# In[ ]:


# Give each term its name

df_tfidf['canonical_name'] = df_tfidf['cui'].apply(lambda x: linker.umls.cui_to_entity[x].canonical_name)


# In[ ]:


# Sort values, save, and display in notebook

df_tfidf.sort_values(by=['doc_id', 'tf_idf'], ascending=False)
df_tfidf = df_tfidf[['doc_id', 'canonical_name', 'cui', 'tf_idf', 'tf', 'idf']]
df_tfidf.to_csv('tfidf_named_20200322.csv', index=False)
df_tfidf


# ## Random sample of the output
# 
# Choosing a random document from the set, showing its abstract, and showing the top concept terms provided by the process here.
# 
# **The hope is that the top concepts for each abstract, above a certain threshold, capture the "aboutness" of the document**

# In[ ]:


# Check our work

target_doc_id = df_tfidf['doc_id'].sample(n=1).values[0]
print(df_results.loc[df_results['doc_id'] == target_doc_id, 'abstract'].values[0])
df_tfidf.loc[df_tfidf['doc_id'] == target_doc_id].sort_values(by='tf_idf', ascending=False).head(10)


# ## Create a resource list of the concepts used

# In[ ]:


# Creating a resource list of the concepts used in this approach

df_concepts = df_tfidf.drop_duplicates(subset=['cui']).copy()
df_concepts = df_concepts[['cui', 'canonical_name', 'idf']]
df_concepts['definition'] = df_concepts['cui'].apply(lambda x: linker.umls.cui_to_entity[x].definition)
df_concepts['raw_count'] = df_concepts['cui'].apply(lambda x: umls_idf_lookup[x]['total_count'])
df_concepts['average_score'] = df_concepts['cui'].apply(lambda x: umls_idf_lookup[x]['average_score'])

df_concepts.to_csv('concepts_20200322.csv', index=False)
df_concepts.sort_values(by='average_score')


# # What can we do with this information?
# 
# ## Query to find documents given a particular term

# In[ ]:


df_tfidf.query("canonical_name == 'RNA Processing'").sort_values(by='tf', ascending=False)


# ## Search for concepts we want to target

# In[ ]:


df_concepts.loc[df_concepts['definition'].str.contains('coronav', case=False, na=False)]


# ## Visualize the data

# In[ ]:


import plotly_express as px

fig = px.histogram(df_tfidf, x='tf_idf', title='tf-idf scores')
fig.show()


# In[ ]:


fig = px.scatter(df_concepts, 
                 x='idf', 
                 y='average_score',
                 size='raw_count',
                 hover_name="canonical_name")
fig.show()


# ## Next steps
# 
# * **Engage subject matter experts to ...**
#   * **Review labels and verify their application**
#   * **Edit and improve the labels for the dataset**
#   * **Choose important labels that correspond to questions in the competition**
# * Engage machine learning expertise to take labeled data and begin model creation
# * Work on threshold/floor for tf-idf scores to automatically avoid misleading labels
# * Use linked data sources for related taxonomies to establish hierarchical relationships between concepts
# * Generate new scores as new documents arrive
# 
