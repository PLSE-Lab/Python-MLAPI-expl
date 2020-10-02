#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Open Research Dataset Challenge (CORD-19)
# An AI challenge with AI2, CZI, MSR, Georgetown, NIH & The White House
# 
# ### Task: What do we know about COVID-19 risk factors?

# ---

# ### About the data

# **Information** 
# 1. **Metadata** for papers from these sources are combined: CZI, PMC, BioRxiv/MedRxiv.
# (total records 29500)
#     - **CZI** 1236 records
#     - **PMC** 27337
#     - **bioRxiv** 566
#     - **medRxiv** 361
# 2. 17K of the paper records have PDFs and the hash of the PDFs are in 'sha'<br>
# 3. For PMC sourced papers, one paper's metadata can be associated with one or more PDFs/shas under that paper - a PDF/sha correponding to the main article, and possibly additional PDF/shas corresponding to supporting materials for the article.<br>
# 4. 13K of the PDFs were processed with fulltext ('has_full_text'=True)<br>
# 5. Various 'keys' are populated with the metadata:
#     - 'pmcid': populated for all PMC paper records (27337 non null)
# 	- 'doi': populated for all BioRxiv/MedRxiv paper records and most of the other records (26357 non null)
# 	- 'WHO #Covidence': populated for all CZI records and none of the other records (1236 non null)
# 	- 'pubmed_id': populated for some of the records
# 	- 'Microsoft Academic Paper ID': populated for some of the records
# ---
# Glossary:<br>
# **Chan Zuckerberg Initiative (CZI)**<br>
# **PubMed Central (PMC)** is a free digital repository that archives publicly accessible full-text scholarly articles that have been published within the biomedical and life sciences journal literature.<br>
# **BioRxiv** (pronounced "bio-archive") is an open access preprint repository for the biological sciences<br>
# **medRxiv. medRxiv** (pronounced med archive) is a preprint service for the medicine and health sciences and provides a free online platform for researchers to share, comment, and receive feedback on their work. Information among scientists spreads slowly, and often incompletely.
# 
# ---
# **The provided data are organized as followed**<br>
# - Commercial use subset (includes PMC content) -- 9118 full text (new: 128), 183Mb
# - Non-commercial use subset (includes PMC content) -- 2353 full text (new: 385), 41Mb
# - Custom license subset -- 16959 full text (new: 15533), 345Mb
# - bioRxiv/medRxiv subset (pre-prints that are not peer reviewed) -- 885 full text (new: 110), 14Mb
# - Metadata file -- 60Mb
# 
#   #### [More details on dataset](https://pages.semanticscholar.org/coronavirus-research)

# ---

# # The Appproach

# This is a straight forward approach that used Allen Institute For AI SciSpacy model.<br>
# - Step 1: transform the quoted factors (see. initial brief) in patterns via SciSpaCy. They will be called "Theme".<br>
# - Step 2: then, match the Themes among the 29500 articles (XIV) due to SpaCy model. Here, we retrieve THEME, KEYWORD, PAPER_ID.<br>
# - Step 3: from there TITLE, AUTHORS, SOURCE, ... are available if the data are available in the Metadata document.
# <br><br>
# 1. Pros:
# - provide quote, paper id, title, authors from article when the wanted 'Theme' is detected,
# - straight forward approach especially in the rush context,
# - use of models from Allen Institute of AI,
# - prune the volume of documents when searching a specific topic.
# <br><br>
# 2. Cons:
# - basic approach: does not provide text summarization nor sentiment analysis,
# - when matching/extraction actions are done, some manual actions are required.

# ## Patternizing the themes
# This is what we have to find among the 29 500 archives.

# Installing SciSpacy and its specific 'en_core_sci_md' model for later utilization.<br>
# <br>
# Many thanks to [Marek Grzenkowicz](https://www.kaggle.com/chopeen) for these two followed '!pip install --quiet' lines as even if my notebook was running perfectly within my Kaggle account environment, when committed it was not the case anymore. Now, everyone can read it smoothlessly.

# In[ ]:


get_ipython().system('pip install --quiet scispacy')


# In[ ]:


get_ipython().system('pip install --quiet https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz')


# ### Transform the required themes

# In[ ]:


import pandas as pd
from io import StringIO


# ### Initial form of the themes:
# > What do we know about COVID-19 risk factors? What have we learned from epidemiological studies?<br>
# <br>Specifically, we want to know what the literature reports about:
# 1. Data on potential risks factors
#     - Smoking, pre-existing pulmonary disease
#     - Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities
#     - Neonates and pregnant women
#     - Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.
# 2. Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors
# 3. Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups
# 4. Susceptibility of populations
# 5. Public health mitigation measures that could be effective for control

# ### Keeping track of record of the themes by exporting them as .csv

# In[ ]:


riskfac = StringIO("""Factor;Description
    Pulmonary;Smoking, preexisting pulmonary disease
    Infection;Coinfections determine whether coexisting respiratory or viral infections make the virus more transmissible or virulent and other comorbidities
    Birth;Neonates and pregnant women
    Socio-eco;Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences
    Transmission;Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors
    Severity;Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups
    Susceptibility;Susceptibility of populations
    Mitig-measures;Public health mitigation measures that could be effective for control
    """)

rf_base = pd.read_csv(riskfac, sep= ";")
rf_base


# In[ ]:


# exporting factors and description to save it. rf_base.to_csv(r'/2020-03-13/rf_base.csv', index = False).
rf_base.to_csv('rf_base.csv',index=False)


# ### Loading the themes' descriptions

# In[ ]:


data = pd.read_csv('../working/rf_base.csv', delimiter=',', header=None, skiprows=1, names=['Factor','Description'])

descp = data[:8][['Description']];
descp['index'] = descp.index
descp


# ### Loading the themes' designation

# In[ ]:


fact_name = data[:8][['Factor']];
fact_name['index'] = fact_name.index
fact_name


# ## Tokenization of the themes with [ScispaCy package](https://allenai.github.io/scispacy/)
# Preview of themes after preprocessing
# Sci-SpaCy are models for biomedical text processing made by Allen Institute AI2

# ### Defining Patterns

# In[ ]:


import scispacy
import spacy # needs to update spacy to load SciSpacy model.
from spacy import displacy

## need to install "en_core_sci_md" model https://allenai.github.io/scispacy/

nlp = spacy.load('en_core_sci_md') # "en_core_sci_md" larger biodmedical vocab. word vector

def patternizing(dataF):
    for i in range(8):
        theme_sample = dataF[dataF['index'] == i].values[0][0]
        
        text = theme_sample
        # print(theme_sample)

        doc = nlp(text)
       
        # print(list(doc.sents))
        # print(doc.ents)
        
        displacy.render(next(doc.sents), style='ent', jupyter=True)
patternizing(descp)


# ### Patterns into lists and then related to their name Theme.

# In[ ]:


import spacy
nlp = spacy.load("en_core_sci_md")

def pRnize(dataF, indice):
    
    mastlist = []
    """
    n_cov2 = ['covid19', 'covid-19',
              'Covid19', 'Covid-19',
              'COVID19', 'COVID-19',
              'Sars-Cov-2', 'Sars-CoV-2', 'Sars-COV-2', 'Sars-cov-2',
              'SARS-Cov-2', 'SARS-CoV-2', 'SARS-COV-2', 'SARS-cov-2',
              'sars-Cov-2', 'sars-CoV-2', 'sars-COV-2', 'sars-cov-2',
              'Sars Cov-2', 'Sars CoV-2', 'Sars COV-2', 'Sars cov-2',
              'SARS Cov-2', 'SARS CoV-2', 'SARS COV-2', 'SARS cov-2',
              'sars Cov-2', 'sars CoV-2', 'sars COV-2', 'sars cov-2',
              'Sars-Cov 2', 'Sars-CoV 2', 'Sars-COV 2', 'Sars-cov 2',
              'SARS-Cov 2', 'SARS-CoV 2', 'SARS-COV 2', 'SARS-cov 2',
              'sars-Cov 2', 'sars-CoV 2', 'sars-COV 2', 'sars-cov 2',
              'Sars Cov 2', 'Sars CoV 2', 'Sars COV 2', 'Sars cov 2',
              'SARS Cov 2', 'SARS CoV 2', 'SARS COV 2', 'SARS cov 2',
              'sars Cov 2', 'sars CoV 2', 'sars COV 2', 'sars cov 2',
              'Sars Cov2', 'Sars CoV2', 'Sars COV2', 'Sars cov2',
              'SARS Cov2', 'SARS CoV2', 'SARS COV2', 'SARS cov2',
              'sars Cov2', 'sars CoV2', 'sars COV2', 'sars cov2',]
    """
    for i in range(8):
        factor = []
        theme_sample = dataF[dataF['index'] == i].values[0][0]

        text = theme_sample

        doc = nlp(text) 
        
        for item in doc.ents:
            vocab = str(item).lower().strip('()')
            factor.append(vocab)
        
        #for name in n_cov2:
         #   factor.append(name)
        mastlist.append(factor)
    return mastlist[indice]

# To test unquote
#pRnize(descp, 0)


# **Here none of the COVID-19 designation are not been include in the patterns.<br>
# If needed, just unquote the list and the loop related to n_cov2.**

# ### Dictionnary {Theme: patterns} 

# In[ ]:


def key_per_theme(dataF, word):
    dic = {}
    for i in range(8):
        factor = dataF[dataF['index'] == i].values[0][0]
        wordy = pRnize(word, i)
    
        dic[factor.strip()] = wordy
    return dic


# #### Preview of Themes

# In[ ]:


key_per_theme(fact_name, descp)


# Patterns are ready!

# ---

# # Retrieve the text from articles (xiv)
# recall that the xiv are dispatched among different sources: Biomed, Commercial, ...

# In[ ]:


import os
import json
import glob


# In[ ]:


# json file access function
def data_access(path):
    d_acc = {}
    for i in glob.glob(path):
        # link = os.path.normpath(i)
        # print(link)
        
        # loading json file function
        with open(os.path.normpath(i)) as json_file:
            data = json.load(json_file)
            paper_id = data['paper_id']
            
            # text = [item['text'] for item in data['body_text']]
            for item in data['body_text']:
                text = (item['text'])
                
                d_acc[paper_id] = text
                
    return d_acc


# In[ ]:


# json files'path from each folder

# path if needed to check just one article.
biomed_path = "../input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/"  # bio and med archive
commu_path = "../input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/"
noncom_path = "../input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/"
pmc_path = "../input/CORD-19-research-challenge/pmc_custom_license/pmc_custom_license/"

# path if needed to check over all the folder.
biomed_fo = "../input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/*.json"  # bio and med archive
commu_fo = "../input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/*.json"
noncom_fo = "../input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/*.json"
pmc_fo = "../input/CORD-19-research-challenge/pmc_custom_license/pmc_custom_license/*.json"


# ### Data extraction from Metadata doc
# 
# This additional infos on article that are included into the Metadata doc.<br>
# sha / source_x / title / doi / pmcid / pubmed_id / authors / journal / Microsot A.P ID / Who # covidence

# In[ ]:


metadata = pd.read_csv("../input/CORD-19-research-challenge/metadata.csv")
metadata.head(5)


# In[ ]:


def met_xiv(metadata, sha):
    
    sha = str(sha)
        
    for i, tracksha in enumerate(metadata['sha']):
        if tracksha == sha:
            print("Title\n{} \n\nAuthors\n{} \n\nSource: {} \n\nPaper ID: {}\n\ndoi: {} \n\npmcid: {} - pubmed_id: {} \n\nJournal: {}\n\n-linked to-\n\nMicrosoft Academic Paper ID: {} \n\nWHO #Covidence: {}".format(
                                                                                                                                                                                                 metadata['title'][i],
                                                                                                                                                                                                 metadata['authors'][i],
                                                                                                                                                                                                 metadata['source_x'][i],
                                                                                                                                                                                                 metadata['sha'][i],
                                                                                                                                                                                                 metadata['doi'][i],
                                                                                                                                                                                                 metadata['pmcid'][i],
                                                                                                                                                                                                 metadata['pubmed_id'][i],
                                                                                                                                                                                                 
                                                                                                                                                                                                 metadata['journal'][i],
                                                                                                                                                                                                 
                                                                                                                                                                                                 metadata['Microsoft Academic Paper ID'][i],
                                                                                                                                                                                                 metadata['WHO #Covidence'][i]))
# met_xiv(metadata, '0015023cc06b5362d332b3baf348d11567ca2fbb')


# ---

# # Matching Part
# ### Match the theme within a folder of xiv's.
# display Theme / Keywords / Paper_id / Quote

# In[ ]:


# import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_sci_md")

def match_it(theme, xiv):
    
    phMatch = PhraseMatcher(nlp.vocab)
    
    article = data_access(xiv)
    
    
    p = key_per_theme(fact_name, descp)[theme]
    if len(p)==0:
        print('no patterns')
        
    patterns = [nlp(i) for i in p]

    phMatch.add(theme, None, *patterns)

    for num_id in article:
        paper_id = num_id
        
        doc = nlp(article[num_id])
        
        mat = phMatch(doc)
        # print(mat)
        
        for match_id, start, end in mat:
            string_id = nlp.vocab.strings[match_id]
            if len(string_id) == 0:
                print("No Result")
            else:
                span = doc[start:end]
                spant = doc[(start) : (end+20)]

                print("\nTHEME: \033[34m{}\033[00m - KEYWORDS: \033[32m{}\033[00m\n\nQUOTE: \033[0;37;40m{}\033[00m\n\nPAPER_ID:{}".format(string_id,
                                                                                                                     span.text, spant.text, paper_id))

            
                print()


# ### These are the exact spelling of the themes that has to be used. Each of them contains the keywords we had within the initial briefing.
# 
# - Pulmonary
# - Infection
# - Birth
# - Socio-eco
# - Transmission
# - Severity
# - Susceptibility
# - mitig-measures
# 
# ### These are the exact spelling of the Xiv folders path which contains the 29500 articles.
# 
# - biomed_fo: bioRxiv/medRxiv subset (pre-prints that are not peer reviewed)
# - commu_fo: Commercial use subset (includes PMC content)
# - noncom_fo: Non-commercial use subset (includes PMC content)
# - pmc_fo: Custom license subset

# ### Make the query of the desired theme over folder of xiv.

# In[ ]:


match_it('Susceptibility', biomed_fo)


# ### Additional info about the paper found.
# We can notice that the PAPER_ID: 564f8823050b52b5f5c36638ac1ae07557963f36 seems to match a lot with under this query.<br>
# Then, to obtain more info about the said article run the met_xiv function with the related PAPER_ID:

# In[ ]:


met_xiv(metadata,'564f8823050b52b5f5c36638ac1ae07557963f36')

