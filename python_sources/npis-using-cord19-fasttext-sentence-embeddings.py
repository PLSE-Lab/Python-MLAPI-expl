#!/usr/bin/env python
# coding: utf-8

# # **Goal: ** 
# **Round 1 submission answers** on the following questions on **Covid-19** with respect to **Cord-19** literature and using **sentence embeddings** trained on the same:
# 
# # What do we know about the effectiveness of non-pharmaceutical interventions? What is known about equity and barriers to compliance for non-pharmaceutical interventions?
# 
# **Building resilience against biological hazards and pandemics: COVID-19 and its implications for the Sendai Framework (3/30/2020):**
# > Core to our argument are strategies for resilience building against biological hazards and pandemic. In summary, we have examined current and unfolding responses to COVID-19 and their implications for the Sendai Framework. We reiterate our assertion that there is a lack of early and rapid actions from the DRR-related organisations, despite the SFDRR's call for building resilience including from biological hazards. The SFDRR's ultimate goal is a substantial reduction of risk and losses, coupled with laying the essential foundations for rapid and sustained recovery and sustainable development. We hope the evidence we have added shows the crisis of COVID-19 could be used to make 2020 a "super year" of great progress on these goals.
# 
# **Non-pharmaceutical public health interventions for pandemic influenza: an evaluation of the evidence base (8/15/2007):**
# > The demand for scientific evidence on non-pharmaceutical public health interventions for influenza is pervasive, and policy recommendations must rely heavily on expert judgment. Taken together, the literature and expert opinion reveal the kinds of explicit judgments required to translate existing knowledge into policy-relevant terms. In the absence of a definitive science base, our assessment of the evidence identified areas for further investigation as well as non-pharmaceutical public health interventions that experts believe are likely to be beneficial, feasible, and socially and politically acceptable in an influenza pandemic. These findings should be considered in forming national, state, local, and facility pandemic plans.
# 
# **From SARS to pandemic influenza: the framing of high-risk populations (2/22/2019):**
# > Use of person-first language and creation of opportunities to network and develop asset literacy are two practical solutions to open the door for active engagement in the coming years. Going forward, to maintain momentum towards inclusive engagement in the implementation of a wholeof-society approach to pandemic planning, there needs to be complementary application of an asset-oriented lens with institutional space for social participation in planning and governance.
# 
# # Guidance on ways to scale up NPIs in a more coordinated way (e.g., establish funding, infrastructure and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified) to give us time to enhance our health care delivery system capacity to respond to an increase in cases.
# 
# **Emergency Preparedness and Public Health Systems Lessons for Developing Countries (6/30/2008)**:
# > Instead, donor funds for emergency preparedness should be leveraged to strengthen health system fundamentals, such as health information systems, laboratories, human resources, and communication systems, to enable developing countries to better respond both to the current burden of disease and to future pandemics. An exclusive focus on bioterror and pandemic preparedness is inappropriate in developing countries, where underfunded ministries of health often strain to perform routine public health functions. Experience in the U.S. and other developed countries suggests that preparedness funding, when directed toward multiple-use investments, has strengthened core public health system functions. Concerns about bioterrorism and pandemic influenza have put a spotlight on public health systems across the globe.
# 
# # Rapid design and execution of experiments to examine and compare NPIs currently being implemented. DHS Centers for Excellence could potentially be leveraged to conduct these experiments.
# 
# **Health Systems' "Surge Capacity": State of the Art and Priorities for Future Research(3/1/2013):**
# > The incorporation of proximate and latent health care burdens can enable key sites of health system intervention to be identified. Any general conceptual and/or analytical model will need to incorporate geographical, temporal, and social contingencies in the outcomes, so that surge and surge capacity scenarios derive from, and inform, real-world events. While the development of general conceptual and analytical frameworks, together with improvements in data quality and methodological innovations for data analysis, can be of widespread applicability, there is a need to complement this with site-and scenario-specific findings.
# 
# # Rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches. 
# 
# **SARS Risk Perceptions in Healthcare Workers, Japan (3/10/2005):**
# > However, a collective assertion of 3 specific institutional measures stood out as the most important predictor for individual perception of risk, including avoidance of patient and acceptance of risk, as well as concept of general effectiveness of institutional measures. In view of the potential for future epidemics of SARS or other emerging infectious diseases, the planning and implementation of institutional measures should be given a high priority. We found that the level of anxiety among healthcare workers in Japan was relatively high and that the implementation of preventive measures at the institutional level was not perceived to be sufficient.
# 
# **COVID-19 - the role of mass gatherings (3/9/2020):**
# > Institutions with the mandates for outbreak monitoring and response should keep an of inventory of mass gatherings and provide advance warnings and recommendations about outbreaks to the organizers including information on event cancellation, crowd size limitations, or alternatives. Proactive utilization of current computing, epidemiological, laboratory capacities to fully understand the role of MGs, and that of their mitigation potential can usefully inform the future course of COVID-19.
# 
# 
# # Methods to control the spread in communities, barriers to compliance and how these vary among different populations.
# 
# **Risk and Outbreak Communication: Lessons from Taiwan's Experiences in the Post-SARS Era (4/1/2017):**
# > The Taiwan CDC will continue to maintain the strengths of its risk communication systems and resolve challenges as they emerge through active evaluation and monitoring of public opinion to advance Taiwan's capacity in outbreak communication and control. Moreover, the Taiwan CDC will continue to implement the IHR (2005) and to promote a global community working together to fight the shared risk and to reach the goal of ''One World, One Health.'' Many communication strategies, ranging from traditional media to social and new media strategies, have been implemented to improve transparency in public communication and promote civic engagement. According to the WHO, the ultimate purpose of risk communication is to enable people at risk to make informed decisions to protect themselves and their loved ones from harm. The government of Taiwan has demonstrated considerable improvement in its risk communication practices during public health emergencies since the SARS outbreak in 2003.
# 
# **Health Inequalities and Infectious Disease Epidemics: A Challenge for Global Health Security (9/1/2014):**
# > Policymakers and THE INEQUALITIES AND HEALTH SECURITY public health leaders must take these existing inequalities into account when planning for pandemics in order to prevent unnecessary suffering and the perpetuation of health and broader social inequities. Our ability to accomplish that goal requires seeing pandemics for what they are: infectious diseases embedded in a social and political context-contexts defined by social determinants of health and unequal access to resources often resulting in behavioral and/or biological disparities between population subgroups. We believe that neither the GHSA nor the PIP Framework gives true attention to inequalities within nations (which exist independent of national GDP) and their potential impact on disease transmission and resulting illness and death from influenza or other respiratory diseases with pandemic potential such as SARS or MERS-CoV. Therefore, it is our role as public health and healthcare professionals to work within our organizations, with communities, and with policymakers to decrease unnecessary exposure, minimize susceptibility (eg, by enhancing access to vaccines when available), and assure care after disease has developed. Clearly, to achieve global health security and alleviate unnecessary suffering caused by social disadvantage and the social determinants of health requires complex collaboration across multiple sectors.
# 
# **Pandemic Influenza Planning in the United States from a Health Disparities Perspective (5/10/2008):**
# > The framework used here-considering and proactively addressing social vulnerability in exposure to pathogens, susceptibility to disease once exposed, and consequences of illness-should be applicable across national and subnational settings. Countries in which large proportions of the population are impoverished or otherwise socially excluded and countries that have more limited resources and weaker public health and social welfare infrastructures will face the greatest challenges. We have focused here on the United States, but similar fundamental principles-the need for systematic and concrete planning to minimize the social disparities that can be expected to occur in the face of natural disasters such as an infl uenza pandemic-apply worldwide. Countries with universal fi nancial access to healthcare and strong social safety nets will be best positioned to minimize such disparities.
# 
# # Models of potential interventions to predict costs and benefits that take account of such factors as race, income, disability, age, geographic location, immigration status, housing status, employment status, and health insurance status.
# 
# **Initial Psychological Responses to Swine Flu (3/2/2010):**
# > Future work should attempt to replicate the current results with larger samples, further predictors and correlates of worry, and a greater range of demographic indicators (such as marital status and number of children). The emergence of a new pandemic influenza provides a major challenge to behavioral scientists, demanding consideration not only of the traditional cognitive estimates of risk but recognition too of the significant role of emotional worries in predicting behavioral outcomes. Such work should help facilitate intervention strategies that maximize the potential for the positive behavioral changes needed to confront such pandemic threats. This emotional concern is itself likely to be influenced by several individual and group factors.
# 
# **Health Inequalities and Infectious Disease Epidemics: A Challenge for Global Health Security (9/1/2014):**
# > Our ability to accomplish that goal requires seeing pandemics for what they are: infectious diseases embedded in a social and political context-contexts defined by social determinants of health and unequal access to resources often resulting in behavioral and/or biological disparities between population subgroups. Policymakers and THE INEQUALITIES AND HEALTH SECURITY public health leaders must take these existing inequalities into account when planning for pandemics in order to prevent unnecessary suffering and the perpetuation of health and broader social inequities. We believe that neither the GHSA nor the PIP Framework gives true attention to inequalities within nations (which exist independent of national GDP) and their potential impact on disease transmission and resulting illness and death from influenza or other respiratory diseases with pandemic potential such as SARS or MERS-CoV. Therefore, it is our role as public health and healthcare professionals to work within our organizations, with communities, and with policymakers to decrease unnecessary exposure, minimize susceptibility (eg, by enhancing access to vaccines when available), and assure care after disease has developed. Clearly, to achieve global health security and alleviate unnecessary suffering caused by social disadvantage and the social determinants of health requires complex collaboration across multiple sectors.
# 
# **Assessment of economic vulnerability to infectious disease crises (11/18/2016):**
# > Second, inclusion of economic vulnerability to infectious disease risk into Article IV consultations and other economic assessments would ensure that governments, donors, foreign investors, and bond markets pay greater attention to how these risks should be managed and mitigated. In this way, regular assessment of the risks that infectious disease crises pose to economic growth and stability would help reverse the neglect of this dimension of global security.
# 
# # Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with NPIs.
# 
# **An Intervention to Improve Compliance with Transmission Precautions for Influenza in the Emergency Department: Successes and Challenges (1/31/2012):**
# > Passive reminders to ED staff may help to improve compliance, however, more research studies should be undertaken to determine what interventions have the most significant impact, including active protocols, changes to hospital policies and procedures, or simple educational interventions. The electronic medical record may be a useful tool for improving compliance with transmission-based precautions, by implementing direct reminders on order sets and informational mailings to staff, and by tracking ED compliance. Given the possibility of a higher-volume influenza season at best or an upcoming influenza pandemic at worst, EDs will play a significant role in containment of index cases and prevention of transmission of influenza within the hospital and into the community, both by their infection control compliance and by staff and patient education.
# 
# **Public health measures during an anticipated influenza pandemic: Factors influencing willingness to comply (1/29/2009):**
# > This study collected data regarding anticipated responses to an infl uenza pandemic and although in the event of an actual pandemic The infl uence of age on willingness to comply with health protective behaviors. the overall level of compliance is likely to be infl uenced by a range of factors, it is probable that relative compliance levels within the data would be upheld and would be more robust for use in pandemic planning. Data from this study provide the fi rst Australian population baseline in this area against which future response can be tracked and pandemic modeling can be informed. Percentage shown is the proportion 'very'/'extremely' willing to comply.
# 
# # Research on why people fail to comply with public health advice, even if they want to do so (e.g., social or financial costs may be too high).
# 
# **Health Inequalities and Infectious Disease Epidemics: A Challenge for Global Health Security (9/1/2014):**
# > We believe that neither the GHSA nor the PIP Framework gives true attention to inequalities within nations (which exist independent of national GDP) and their potential impact on disease transmission and resulting illness and death from influenza or other respiratory diseases with pandemic potential such as SARS or MERS-CoV. Therefore, it is our role as public health and healthcare professionals to work within our organizations, with communities, and with policymakers to decrease unnecessary exposure, minimize susceptibility (eg, by enhancing access to vaccines when available), and assure care after disease has developed. Our ability to accomplish that goal requires seeing pandemics for what they are: infectious diseases embedded in a social and political context-contexts defined by social determinants of health and unequal access to resources often resulting in behavioral and/or biological disparities between population subgroups. Clearly, to achieve global health security and alleviate unnecessary suffering caused by social disadvantage and the social determinants of health requires complex collaboration across multiple sectors. Policymakers and THE INEQUALITIES AND HEALTH SECURITY public health leaders must take these existing inequalities into account when planning for pandemics in order to prevent unnecessary suffering and the perpetuation of health and broader social inequities.
# 
# **Alberta family physicians' willingness to work during an influenza pandemic: a cross-sectional study (6/26/2013):**
# > There may be differences according to country of origin, due perhaps to different cultural acceptance of risk or perception of duty. Our results suggest that during an outbreak more than half of Alberta physicians will be available and willing to work; however, in the midst of a severe pandemic, these numbers may drop due to their own illness, or unforeseen circumstances. Men appear to be more willing to continue working than women. These findings have implications for health care planning policy development.
# 
# # Research on the economic impact of this or any pandemic. This would include identifying policy and programmatic alternatives that lessen/mitigate risks to critical government services, food distribution and supplies, access to critical household supplies, and access to health diagnoses, treatment, and needed care, regardless of ability to pay.
# 
# **Health Inequalities and Infectious Disease Epidemics: A Challenge for Global Health Security (9/1/2014):**
# > We believe that neither the GHSA nor the PIP Framework gives true attention to inequalities within nations (which exist independent of national GDP) and their potential impact on disease transmission and resulting illness and death from influenza or other respiratory diseases with pandemic potential such as SARS or MERS-CoV. Therefore, it is our role as public health and healthcare professionals to work within our organizations, with communities, and with policymakers to decrease unnecessary exposure, minimize susceptibility (eg, by enhancing access to vaccines when available), and assure care after disease has developed. Policymakers and THE INEQUALITIES AND HEALTH SECURITY public health leaders must take these existing inequalities into account when planning for pandemics in order to prevent unnecessary suffering and the perpetuation of health and broader social inequities. Such planning for an influenza or other pandemic requires social interventions, policy initiatives, and enhancing access to care prior to the time of a pandemic. Our ability to accomplish that goal requires seeing pandemics for what they are: infectious diseases embedded in a social and political context-contexts defined by social determinants of health and unequal access to resources often resulting in behavioral and/or biological disparities between population subgroups.
# 
# **Pandemic Influenza Planning in the United States from a Health Disparities Perspective (5/10/2008):**
# > Countries in which large proportions of the population are impoverished or otherwise socially excluded and countries that have more limited resources and weaker public health and social welfare infrastructures will face the greatest challenges. The framework used here-considering and proactively addressing social vulnerability in exposure to pathogens, susceptibility to disease once exposed, and consequences of illness-should be applicable across national and subnational settings. Countries with universal fi nancial access to healthcare and strong social safety nets will be best positioned to minimize such disparities. We have focused here on the United States, but similar fundamental principles-the need for systematic and concrete planning to minimize the social disparities that can be expected to occur in the face of natural disasters such as an infl uenza pandemic-apply worldwide.

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
                    #doc = disease_ner(entry['section'])
                    #diseases = []
                    #for ent in doc.ents:
                     #   if ent.label_ == "DISEASE":
                      #      diseases.append(ent.text)
                    d["section"] = entry['section']
                    d["para"] = entry['text']
                    #d["diseases"] = ", ".join(diseases)
                    self.abstracts.append(d)

            # Body text
            for idx,entry in enumerate(content['body_text']):
                if not isinstance(entry['text'], str):
                    continue
                d = {}
                d["idx"] = idx
               # doc = disease_ner(entry['section'])
               # diseases = []
                #for ent in doc.ents:
                 #   if ent.label_ == "DISEASE":
                  #      diseases.append(ent.text)
                d["section"] = entry['section']
                d["para"] = entry['text']
                
                #d["diseases"] = ", ".join(diseases)
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
dict_.clear()
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
    "What do we know about the effectiveness of non-pharmaceutical interventions or community mitigation strategies for pandemics? What is known about equity and barriers to compliance for non-pharmaceutical interventions?",
    "Guidance on ways to scale up non-pharmaceutical interventions or community mitigation strategies in a more coordinated way. This could include how communities can establish funding, infrastructure setup and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified to give us time to enhance our health care delivery system capacity to respond to an increase in cases.",
    "Rapid design and execution of experiments to examine and compare non-pharmaceutical interventions or community mitigation strategies currently being implemented. Department of Homeland Security Centers for Excellence could potentially be leveraged to conduct these experiments.",
    "Rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches.",
    "Methods to control the spread in communities, barriers to compliance and how these vary among different populations and geographies.",
    "Examples of models of potential non-pharmaceutical interventions to predict costs and benefits that take account of such factors as race, income, disability, age, geographic location, immigration status, housing status, employment status, and health insurance status.",
    "Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with non-pharmaceutical interventions.",
    "Research on why people fail to comply with public health advice, even if they want to do so. Are people failing to comply with public health advice because of social obligations. Are the poorer segments of society failing to comply with containment measures because of losing jobs or not keeping up with financial costs.",
    "Research on the economic impact of COVID 19 or the novel coronavirus or any pandemic. This would include identifying policy and programmatic alternatives that lessen/mitigate risks to critical government services, food distribution and supplies, access to critical household supplies, and access to health diagnoses, treatment, and needed care, regardless of ability to pay."
]


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
    df_paragraphs["cosine_similarity"] = para_similarities_array
    
    df_paragraphs = df_paragraphs.loc[df_paragraphs['cosine_similarity']>=0.6]
    df_paragraphs = df_paragraphs.sort_values(['section_id','cosine_similarity'], ascending=(False, False))
    
    if len(df_paragraphs) > 10:
        df_paragraphs = df_paragraphs.head(10)
    df_paragraphs = df_paragraphs.drop(['section', 'tags', 'section_id'], axis = 1) 
    
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
    display(Markdown("**Answer : **"))
    display(Markdown("**"+ df_paragraphs['para_summary'].values[0] +"**"))
    display(df_paragraphs[['paper_id', 'para_summary', 'paragraph', 'cosine_similarity']])

