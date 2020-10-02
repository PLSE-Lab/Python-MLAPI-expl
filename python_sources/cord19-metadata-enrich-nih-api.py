#!/usr/bin/env python
# coding: utf-8

# # CORD-19 Metadata Enrichment [1/x]: Filling in Missing Keys/Identifiers (DOI/PMCID/PMID)

# # Goals and Motivation
# 
# Metadata serves as a critical feature in any data understanding effort. Across the tasks presented in the CORD-19 challenge, metadata has the chance to provide required context in order to best leverage the text. While the text of the various documents allows the researcher to group ideas, without accurate and precise metadata that understanding is limited. This will be a series of notebooks that seek to clean, augment, and enrich the provided metadata in order to help bolster the research efforts of all utilizing this dataset to address the tasks presented. 
# 
# ## Other Enrichment can be found here:
# #### [CORD19-Metadata Enrich: Microsoft Academic API](https://www.kaggle.com/dannellyz/cord19-metadata-enrich-microsoft-academic-api)
# #### [CORD-19 Metadata Enrich [2/x]: Altmetric API](https://www.kaggle.com/dannellyz/cord19-metadata-enrich-altmetric-api)
# 
# The below chart shows the data which is present in the metadata:

# In[ ]:


#Import pandas and the read in the metadata csv to a Dataframe
import pandas as pd
metadata = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")


# In[ ]:


#Utilizing matplotlib for display
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Show the percentage of each column that is present or not NULL
col_present_pct = metadata.notnull().sum() / len(metadata)

#Bar Plot
col_present_pct.sort_values().plot.bar()


# # Methodology
# 
# In order to fill in any missing data we must be able to reference the works by thier various IDs. DOI is the most prevelant and so the goal will be to turn all IDs into DOIs.
# 
# ## [DOI](https://www.doi.org/):
# A digital object identifier (DOI) is a persistent identifier or handle used to identify objects uniquely, standardized by the International Organization for Standardization (ISO). They can be thought of as a url or latter part of a url with resolutions coming from  https://doi.org/[DOI]
# 
# ## [PMCID & PMID](https://publicaccess.nih.gov/include-pmcid-citations.htm#Difference):
# The PubMed Central reference number (PMCID) is different from the PubMed reference number (PMID). PubMed Central is an index of full-text papers, while PubMed is an index of abstracts. The PMCID links to full-text papers in PubMed Central, while the PMID links to abstracts in PubMed. PMIDs have nothing to do with the NIH Public Access Policy.
# 
# The provided data had the following breakdown of DOI/PMCID/PMID identifiers for the articles.

# In[ ]:


id_col_list = ['doi','pmcid', 'pubmed_id']
def null_ids_graph(df):
    #Group by the various IDs and count their permutations
    id_types_present = df.notnull().groupby(id_col_list).size()
    #Plot with bar chart
    chart = id_types_present.plot.bar()
    for p in chart.patches:
        chart.annotate('{:,}'. format(p.get_height()), (p.get_x() * 1.00, p.get_height() * 1.01))
    return chart
null_ids_graph(metadata)


# #  PMCID -> DOI
# 
# Since PMCID is the most prevelant identifier type we will utilize it in order to try and fill in the other items. Converting these to PMID or DOI will allow them to be used later in looking up addition information and features. 
# 
# ## NCBI API
# The [National Center for Biotechnology Information(NCBI)](https://www.ncbi.nlm.nih.gov/) advances science and health by providing access to biomedical and genomic information. They offer an [API](https://www.ncbi.nlm.nih.gov/pmc/tools/id-converter-api/) to do conversions from PMCID to other journal identifiers.
# 
# ## Get vs Load
# The code below has two methods for getting the NCBI data. 
# 
# ### get_ncbi_results
# Does an API call to get the full list of NCBI results and saves them to a csv.
# 
# ### load_ncbi_results
# This loads the csv from the CORD-19 Metadata Enrichment Kaggle public dataset. Can be loaded by clicking "Add Data" on the right and using the url:
# https://www.kaggle.com/dannellyz/cord19-metadata-enrichment

# In[ ]:


#URL Lib to query API
from urllib.request import urlopen

#ElementTree to parse XML response
import xml.etree.ElementTree as ET

#Import Tracker
from tqdm.notebook import tqdm
    
def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    src: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def ncbi_api(pmcids):
    #Base string is the API end point
    api_base_string = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=my_tool&email=my_email@example.com&ids="
    #List of PMCIDS to send to API end point
    ids_string = ",".join(list(pmcids))
    #Call API with query
    api_query = api_base_string + ids_string

    #Get API response which is a list of dictionaries
    with urlopen(api_query) as response:
        response_content = response.read()
    root = ET.fromstring(response_content)
    
    #Return
    return [child.attrib for child in root[1:]]

def get_ncbi_results(file_name, pmcids):
    #Batch the results as API can only take 10 at a time
    batch_pmcids = chunks(pmcids, 10)
    batch_results = []

    #For each batch run against API
    for batch in tqdm(batch_pmcids):
        batch_results.extend(ncbi_api(batch))

    #Collect results into a Dataframe
    ncbi_results = pd.DataFrame(batch_results).drop("requested-id", axis=1)

    #Send dataframe to csv
    ncbi_results.to_csv(file_name)
    return ncbi_results

def load_ncbi_results(file_name):
    ncbi_results = pd.read_csv(file_name, usecols=["doi", "pmcid", "pubmed_id"])
    return ncbi_results

def get_pmcids(metadata):
    has_pmcid_no_doi = metadata[metadata.pmcids.notnull() & metadata.doi.isnull()]

#Get the NCBI Results
file_name = "ncbi_metadata.csv"
has_pmcid_no_doi = metadata[metadata.pmcid.notnull() & metadata.doi.isnull()]
pmcids_list = list(has_pmcid_no_doi.pmcid)

#Get from API
#You must also enable internet in the options to the right
ncbi_results = get_ncbi_results(file_name, pmcids_list)

#Load from Public Data Set
#ncbi_results = load_ncbi_results(file_name)


# # Results
# As the below dataframe and graph depict the query to NCBI has filled out an additional 10,339 results for the missing IDS.

# In[ ]:


#Update metadata with new values from NCBI results
metadata_v2 = metadata.copy()
metadata_v2.update(ncbi_results)

#Graph update
id_count_v1 = metadata.notnull().groupby(id_col_list).size()
id_count_v2 = metadata_v2.notnull().groupby(id_col_list).size()
updated_counts = id_count_v2 - id_count_v1
updated_counts


# ## Before and After

# In[ ]:


pd.concat([pd.DataFrame(id_count_v1, columns = ["Before"]).T,
           pd.DataFrame(updated_counts, columns = ["After"]).T]).T.plot.bar(stacked=True)


# ## Updated Null IDs Graph

# In[ ]:


null_ids_graph(metadata_v2)


# In[ ]:


#Output for future work
metadata_v2.to_csv("metadata_v2.csv")


# # Please upvote if this is useful!
# 
# Plan to make a couple more notebooks continuing the enrichment of the metadata and if you have any suggestions please leave a comment.
