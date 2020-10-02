#!/usr/bin/env python
# coding: utf-8

# # **Goal: ** 
# **Round 1 submission answers** updated on the following questions on **Covid-19** with respect to **Cord-19** literature and using **sentence embeddings** trained on the same:
# 
# 
# # Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019.
# 
# **Community Health Workers and Pandemic Preparedness: Current and Prospective Roles (3/26/2019):**
# > The guidance outlined in the IHR and JEE Tool provides a framework to promote global health security and pandemic preparedness where capacities already exist. Through their routine work, CHWs contribute to inherent resilience and pandemic preparedness by increasing access to health products and services, distributing health information, and reducing the burden felt by the formal healthcare system-all of which act to buffer against emergencies. Recognizing that CHWs already play a role in pandemic preparedness, the roles and responsibilities of CHWs in pandemic preparedness could be expanded to improve health security and communitylevel resilience. However, access to these capacities is not always a reality, and CHWs represent a proven strategy for improving access to healthcare. CHWs can also contribute to adaptive resilience by increasing social mobilization, completing surveillance activities, and by filling health systems gaps left in the wake of infectious disease outbreaks.
# 
# **Clinician Wellness During the COVID-19 Pandemic: Extraordinary Times and Unusual Challenges for the Allergist/Immunologist (4/4/2020):**
# > As we realize that we are in a seminal 453 moment, that future generations may refer to "pre-COVID" and "post-COVID"(61), we must 454 also pause to reflect, to breathe, and to care for ourselves and our loved ones. • Engage in non-professional related online endeavors, i.e. podcasts pertaining to areas of interest • Find social groups of similar interests to your own and engage or simply follow • Seek out groups and friends with positive, uplifting messaging • Implement social media to share resources, such as PPE access. • Use social media to promote kindness to others and connect with those that are socially distanced.
# 
# # Efforts to support sustained education, access, and capacity building in the area of ethics.
# 
# **Health Systems' "Surge Capacity": State of the Art and Priorities for Future Research (3/1/2013):**
# > Any general conceptual and/or analytical model will need to incorporate geographical, temporal, and social contingencies in the outcomes, so that surge and surge capacity scenarios derive from, and inform, real-world events. Despite the disproportionately high occurrence of surge-generating events in low-and middle-income countries, and their considerably heightened vulnerability to such events, the research on surge capacity to date has focused largely on high-income countries, principally the United States. Work is, however, needed to generate robust conceptual and analytical frameworks, along with innovations in data collection and methodological approaches.
# 
# **A Multi-Method Approach to Curriculum Development for In-Service Training in China’s Newly Established Health Emergency Response Offices (6/27/2014):**
# > It describes the multimethod approach used to identify training needs systematically, and to adopt international best practice in partnership with senior decision-makers and content experts from the government, academia and the military. The consultative process for developing the curriculum was designed to address the scale of the challenge for coordinating planning and training activities across jurisdictions that cover a population of over 1.3 billion people, and where most provinces have more than 60 million residents.
# 
# # Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.
# 
# **China's Belt and Road Initiative: Incorporating public health measures toward global economic growth and shared prosperity (6/30/2019):**
# > A strategy with a definition of health that goes beyond the absence of disease, involving partnership and community participation and that requires governments, organizations, communities, and individuals to work together to improve healthcare quality delivery would be most effective. More recently, it includes community and institutional development, early warning systems, surveillance and detection-such as developing platforms that would make operational and epidemiological information publicly available to decision-makers in the event of an outbreak or potential health emergency-and investment in strengthening countries' biosecurity and pandemic preparedness through a focus on national-scale immunization programs. It is also necessary to leverage the unique cross-disciplinary approach to understand the principles underlying research, policy, and practice in global health; to foster critical thinking; and to build transferable skills from a range of academic and professional backgrounds in the process of tackling local and global health issues and challenges. Behind It is crucial to promote greater cooperation and good practice among practitioners, policymakers, and researchers to ensure that quality evidence is translated into real and practical action, health education, and development implementation.
# 
# 
# # Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.
# 
# **Health Inequalities and Infectious Disease Epidemics: A Challenge for Global Health Security (9/1/2014):**
# > Policymakers and THE INEQUALITIES AND HEALTH SECURITY public health leaders must take these existing inequalities into account when planning for pandemics in order to prevent unnecessary suffering and the perpetuation of health and broader social inequities. Clearly, to achieve global health security and alleviate unnecessary suffering caused by social disadvantage and the social determinants of health requires complex collaboration across multiple sectors. We believe that neither the GHSA nor the PIP Framework gives true attention to inequalities within nations (which exist independent of national GDP) and their potential impact on disease transmission and resulting illness and death from influenza or other respiratory diseases with pandemic potential such as SARS or MERS-CoV. Therefore, it is our role as public health and healthcare professionals to work within our organizations, with communities, and with policymakers to decrease unnecessary exposure, minimize susceptibility (eg, by enhancing access to vaccines when available), and assure care after disease has developed. Our ability to accomplish that goal requires seeing pandemics for what they are: infectious diseases embedded in a social and political context-contexts defined by social determinants of health and unequal access to resources often resulting in behavioral and/or biological disparities between population subgroups.
# 
# **Emerging infectious diseases and outbreaks: implications for women’s reproductive health and rights in resource-poor settings (4/1/2020):**
# > In this time of global pandemic of Covid-19 and the new demand placed on the system to cope with the resulting new demands, Action Canada for Sexual Health and Rights has emphasized in its statement that there are concerns regarding increased wait times to access SRHR services, difficulties in accessing SRHR medications (including contraceptives, hormone therapy and HIV treatment and increased health risks), and increased health risks experienced by pregnant and immune-compromised people. All these global epidemics have been weakening the health care systems and increasing the barriers to access reproductive health services in the LMICs, especially by impacting the economic, social and personal decision-making of women. The report from African Development confirms the impossibility of building resilience to Ebola and future infectious disease shocks in households and communities without also addressing systemic gender inequality [16] , national development strategies for EVD response (or any EID), and gender-sensitive recovery that addresses the associated negative impacts on women and girls [17] .

# # **About Us:**
# We are a group of AI and NLP scientists with experience across NLP, image processing and computer vision. Covid-19 Kaggle challenge has provided us with a unique opportunity to help humanity fight the corona virus pandemic collectively by utilizing benefits of AI and NLP. We have focused on creating NLP solution to enable users to ask questions and get the most accurate results from the vast corpus of medical journals.
# 

# # **Approach:**
# 
# We have trained the sent2vec model on the most recent dump of CORD-19 corpus to generate sentence embeddings. This model is trained on 14658255 sentences and 864320 words. **Fasttext embeddings are huge in size so we could not generate this in the working directory of the kernel.** We trained our model on AWS and uploaded the same as additional dataset.

# ![sent2vec.PNG](attachment:sent2vec.PNG)

# ![image.png](attachment:image.png)

# # Benefits of this approach:
# 
# * Very simple and straightforward approach without complex indexing and other dependencies which need to be set up
# * Based on FastText Sent2Vec embeddings trained on CORD-19 corpus - embeddings are contextually relevant to the queries
# * Filtering helps remove lots of non-covid content
# * Text summarization helps to summarize longer paragraphs
# 
# # Further Improvements possible:
# * Embeddings are currently computed on entire paragraphs rather than at sentence level. Aggregating sentence level embeddings at paragraph level would be more relevant
# * Currently only looks for answers in papers which have abstracts available in the metadata.csv. Approach needs to be further scaled to entire literature
# * Disease names can be further extracted using spacy medical NER models that can add further structure to the response

# In[ ]:


get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz')
get_ipython().system('pip install scispacy')


# # Install and set-up sent2vec repository from GitHub

# In[ ]:


get_ipython().system('git clone https://github.com/epfml/sent2vec.git')


# In[ ]:


cd sent2vec


# In[ ]:


get_ipython().system('pip install .')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import spacy
import en_core_sci_sm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json
import pandas as pd
from tqdm import tqdm
import re
import spacy
import os

pd.options.mode.chained_assignment = None  # default='warn'


# # Load the Input JSON & metadata.csv from kaggle input directory

# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

root_path = '/kaggle/input/CORD-19-research-challenge/'
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
print(len(all_json))
metadata_path = f'{root_path}metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str,
    'doi': str
})
meta_df.head()


# # JSON file reader to load individual JSON files extracting their abstract and body sections

# In[ ]:


class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstracts = []
            self.body_texts = []
            d={}
            # Abstract
            if "abstract" in content:
                for idx, entry in enumerate(content['abstract']):
                    if not isinstance(entry['text'], str):
                        continue

                    d["idx"] = idx
                    d["section"] = entry['section']
                    d["para"] = entry['text']
                    self.abstracts.append(d)

            # Body text
            for idx,entry in enumerate(content['body_text']):
                if not isinstance(entry['text'], str):
                    continue
                d = {}
                d["idx"] = idx
                d["section"] = entry['section']
                d["para"] = entry['text']
                self.body_texts.append(d)
                
    def __repr__(self):
        return f'{self.paper_id}: {self.abstracts[:]}... {self.body_texts[:]}...'


# # Regex to filter out non-covid content:

# In[ ]:


# Keyword patterns to search for
keywords = [r"2019[\-\s]?n[\-\s]?cov", "2019 novel coronavirus", "coronavirus 2019", r"coronavirus disease (?:20)?19",
            r"covid(?:[\-\s]?19)?", r"n\s?cov[\-\s]?2019", r"sars-cov-?2", r"wuhan (?:coronavirus|cov|pneumonia)",
            r"rna (?:coronavirus|cov|pneumonia)", r"mers (?:coronavirus|cov|pneumonia)", r"influenza (?:coronavirus|cov|pneumonia)",
            r"sars (?:coronavirus|cov|pneumonia)", r"sars", r"mers", r"pandemic", r"pandemics"]

# Build regular expression for each keyword. Wrap term in word boundaries
regex = "|".join(["\\b%s\\b" % keyword.lower() for keyword in keywords])

def tags(text):
    if re.findall(regex, str(text).lower()):
        tags = "COVID-19"
    else:
        tags="NON COVID"
    return tags


# Iterate JSON directories to load all the paragraphs from abstracts and body of different papers. Abstracts if available in metadata.csv is also loaded separately.

# In[ ]:


dict_ = {'paper_id': [], 'section': [], 'sub_section': [], 'paragraph': [], 'authors': [], 'title': [],
         'journal': [],
         'source_x': [], 'publish_time': [],'diseases':[],'tags':[]}

for idx, entry in enumerate(all_json):
    if idx % (len(all_json) // 10) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    content = FileReader(entry)
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    # no metadata, skip this paper
    if len(meta_data) == 0:
         continue
    
    authors = meta_data['authors'].values[0]
    title = meta_data['title'].values[0]
    journal = meta_data['journal'].values[0]
    publish_time = meta_data['publish_time'].values[0]
    source_x = meta_data.source_x.values[0]
    
    metadata_abstract = meta_data.abstract.values[0]
    if isinstance(metadata_abstract, str):
        dict_['paper_id'].append(content.paper_id)
        dict_['section'].append('Metadata Abstract')
        dict_['sub_section'].append('Metadata Abstract')
        dict_['tags'].append(tags(metadata_abstract))
        dict_['paragraph'].append(metadata_abstract)
        dict_['publish_time'].append(publish_time)
        dict_['source_x'].append(source_x)
        dict_['authors'].append(authors)
        dict_['title'].append(title)
        dict_['journal'].append(journal)
          
    for items in content.abstracts:
        dict_['paper_id'].append(content.paper_id)
        dict_['section'].append('JSON Abstract')
        dict_['sub_section'].append(items['section'])
        #dict_['diseases'].append(items['diseases'])
        dict_['paragraph'].append(items['para'])
        dict_['publish_time'].append(publish_time)
        dict_['source_x'].append(source_x)
        dict_['authors'].append(authors)
        dict_['title'].append(title)
        dict_['journal'].append(journal)
        dict_['tags'].append(tags(str(items['para'])))
        
    for items in content.body_texts:
        dict_['paper_id'].append(content.paper_id)
        dict_['section'].append('Body')
        dict_['sub_section'].append(items['section'])
        #dict_['diseases'].append(items['diseases'])
        dict_['paragraph'].append(items['para'])
        dict_['publish_time'].append(publish_time)
        dict_['source_x'].append(source_x)
        dict_['authors'].append(authors)
        dict_['title'].append(title)
        dict_['journal'].append(journal)
        dict_['tags'].append(tags(str(items['para'])))


# # Filtering only COVID-19 related papers  - metadata abstracts & body paragraphs

# In[ ]:


import os.path
def extract_covid_abstracts_body(dict_):
    df_covid = pd.DataFrame(dict_,
                        columns=['paper_id', 'section', 'sub_section', 'paragraph', 'authors', 'title',
                                     'source_x', 'publish_time', 'journal','tags'])
    print(len(df_covid))
    
    # Extract only Covid articles
    df_covid_content = df_covid.loc[df_covid['tags'] == 'COVID-19']
    print("Total Covid19 paragraphs : ", len(df_covid_content))
    
    df_metadata_abstracts = df_covid_content.loc[df_covid_content['section']=="Metadata Abstract"]
    df_metadata_abstracts.to_csv('/kaggle/working/covid_abstracts.csv', index=False)
    
    df_body = df_covid_content.loc[df_covid_content['section']=="Body"]
    df_body.to_csv('/kaggle/working/covid_body.csv', index=False)
    return df_metadata_abstracts, df_body

df_abstracts, df_body = extract_covid_abstracts_body(dict_)
paper_bodies = df_body.groupby(['paper_id'])
print("Metadata uniques abstracts : ", df_abstracts['paper_id'].nunique())
print("# of paragraphs in body : ", len(df_body))
print("# of unique papers : ", df_body['paper_id'].nunique())


# In[ ]:


import gc
gc.collect()


# # Loading the custom trained sent2vec model:

# In[ ]:


import sent2vec
#model_path = "/kaggle/input/biosentvec/BioSentVec_CORD19-bigram_d700.bin"
model_path = "/kaggle/input/covid-sent2vec-ver2/BioSentVec_CORD19-bigram_d700_v2.bin"
model = sent2vec.Sent2vecModel()
try:
    model.load_model(model_path)
except Exception as e:
    print(e)
print("model successfully loaded")


# # Creating the abstract embeddings:

# In[ ]:


vector_dict = {}
abstracts_list = []
for sha, abstract in tqdm(df_abstracts[["paper_id","paragraph"]].values):
    if isinstance(abstract, str):
        vector_dict[sha] = model.embed_sentence(abstract)
        abstracts_list.append(abstract)

keys = list(vector_dict.keys())
vectors = np.array(list(vector_dict.values()))

nsamples, x, y = vectors.shape
values_array = vectors.reshape((nsamples,x*y))
print(values_array.shape)


# # Loading the task queries and summary text:

# In[ ]:


sub_tasks = [
    'What has been published concerning ethical considerations for research? What has been published concerning social sciences at the outbreak response?',
    'Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019.',
    'Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight.',
    'Efforts to support sustained education, access, and capacity building in the area of ethics.',
    'Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.',
    'Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)',
    'Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.',
    'Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media.'
]
sub_tasks


# # Run task sub-queries and generate summarized results in tabular format along with CSV files generated reporting response for each sub-task query containing more details:

# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from IPython.display import display, Markdown
no_of_docs = 20

medical = en_core_sci_sm.load()
for sub_task_id, sub_task in enumerate(sub_tasks):
    
    query_statement = sub_task
    query_vector = model.embed_sentence(query_statement)

    cosine_sim_matrix_query = cosine_similarity(values_array, query_vector.reshape(1,-1))
    query_sim_indexes = np.argsort(cosine_sim_matrix_query.reshape(1,-1)[0])[::-1][:no_of_docs]
    
    df_paragraphs = pd.DataFrame()
    body_para_list = []
    for index in query_sim_indexes:
        paper_id = keys[index]
        try:
            body = paper_bodies.get_group(paper_id)
        except:
            continue
        body["section_id"] = (pd.RangeIndex(stop=body.shape[0])+1)/len(body)*100
        df_paragraphs = df_paragraphs.append(body, ignore_index = True) 
        body_paras = body["paragraph"].values
        body_para_list.extend(body_paras)
    
    para_dict = {}
    para_list = []
    p = 0
    for para in body_para_list:
        para_dict[p] = model.embed_sentence(para)
        para_list.append(para)
        p += 1

    # Para level vectors
    p_vectors = np.array(list(para_dict.values()))
    nsamples, x, y = p_vectors.shape
    para_vectors = p_vectors.reshape((nsamples,x*y))

    para_matrix_query = cosine_similarity(para_vectors, query_vector.reshape(1,-1))
    para_similarities_array = para_matrix_query.reshape(1,-1)[0]
    df_paragraphs["cosine_similarity"] = para_similarities_array*100
    df_paragraphs = df_paragraphs.loc[df_paragraphs['cosine_similarity']>60]
    df_paragraphs["Relevance"] = (df_paragraphs['cosine_similarity']+df_paragraphs['section_id'])/2
    df_paragraphs = df_paragraphs.sort_values(by='Relevance', ascending=False)
    df_paragraphs = df_paragraphs.loc[df_paragraphs['Relevance']>=50]
    
    if len(df_paragraphs) > 10:
        df_paragraphs = df_paragraphs.head(10)
    df_paragraphs = df_paragraphs.drop(['section', 'tags', 'section_id'], axis = 1)  # 
    
    shortlisted_body_paras = df_paragraphs["paragraph"].values
    entities_list = []
    para_summ = []
    for para in shortlisted_body_paras:
        doc = medical(para)
        
        sent_dict = {}
        sent_list = []
        for sent in doc.sents:
            sent_dict[sent.text] = model.embed_sentence(sent.text)
            sent_list.append(sent.text)
            
        s_vectors = np.array(list(sent_dict.values()))
        nsamples, x, y = s_vectors.shape
        sent_vectors = s_vectors.reshape((nsamples,x*y))
        cosine_sim_matrix_sents = cosine_similarity(sent_vectors, query_vector.reshape(1,-1))
        
        if len(sent_list) > 30:
            no_of_sents_summ = int(len(sent_list) * 0.1)
        elif len(sent_list) > 20:
            no_of_sents_summ = int(len(sent_list) * 0.2)
        elif len(sent_list) > 10:
            no_of_sents_summ = int(len(sent_list) * 0.3)
        elif len(sent_list) > 5:
            no_of_sents_summ = int(len(sent_list) * 0.4)
        else:
            no_of_sents_summ = len(sent_list)
            
        sent_sim_indexes = np.argsort(cosine_sim_matrix_sents.reshape(1,-1)[0])[::-1][:no_of_sents_summ]
        sents_summ = [sent_list[j] for j in sent_sim_indexes]
        para_summ.append(" ".join(sents_summ))
        entities = [ent.text for ent in doc.ents]
        entities_list.append(", ".join(entities))
    
    df_paragraphs["para_summary"] = para_summ
    df_paragraphs["entities"] = entities_list
    df_paragraphs.to_csv("/kaggle/working/subtask_"+str(sub_task_id+1)+"_answers.csv", index=False)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', -1)
    
    display(Markdown("**Query : "+sub_task+"**"))
    
    if len(df_paragraphs) == 0:
        display(Markdown("**Answer : No answers.**"))
    else:
        display(Markdown("**Answer : **"))
        display(Markdown("**"+ df_paragraphs['para_summary'].values[0] +"**"))
        display(df_paragraphs[['paper_id', 'para_summary', 'paragraph', 'Relevance']].head(5))

