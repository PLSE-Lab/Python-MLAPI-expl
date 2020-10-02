#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json,csv,re,os
from zipfile import ZipFile
import re
a=[]
for k in os.walk("/kaggle/input/CORD-19-research-challenge/"):
    
    if 'pdf_json' in k[0] and len(k[2])>0:
        print(len(k[2]),k[0])
        for z in k[2]:
            if z.endswith(".json") :
                
                loaded_json=json.load((open(k[0]+'/'+z,"r")))
                body_string=[]
                
                title_string=re.sub('[^a-zA-Z0-9 \n\.]','',loaded_json['metadata']['title'])
                for x in loaded_json["body_text"]:
                    body_string.append(re.sub('[^a-zA-Z0-9 \n\.]','',x['text'])) 
                #print()
                if 'abstract' in loaded_json and len(loaded_json['abstract'])>0:
                   abstract_string=re.sub('[^a-zA-Z0-9 \n\.]','',loaded_json['abstract'][0]['text'] )
                else:
                   abstract_string=''
                a.append((title_string,abstract_string,body_string))
csv_read=open("/kaggle/working/read_csv_final1.csv","w")
for (i,j,k) in a:
    if i==a[0][0]:
      csv_read.write('"{}","{}","{}"'.format("title","abstract","paragraphs"))
      csv_read.write("\n")
    csv_read.write('"{}","{}","{}"'.format(i,j,k))
    csv_read.write("\n")
csv_read.close()


# In[ ]:


get_ipython().system('pip install cdqa')


# In[ ]:


import os
import pandas as pd
from ast import literal_eval

from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline import QAPipeline
from cdqa.utils.download import download_model, download_bnpp_data

download_bnpp_data(dir='./data/bnpp_newsroom_v1.1/')
download_model(model='bert-squad_1.1', dir='./models')


# In[ ]:


df = pd.read_csv('/kaggle/working/read_csv_final1.csv',converters={'paragraphs': literal_eval})
print(df.head())
df2 = filter_paragraphs(df)
print(df2.head())


# In[ ]:


# print(df.head())
cdqa_pipeline = QAPipeline(reader='./models/bert_qa.joblib')
cdqa_pipeline.fit_retriever(df=df2)


# QAPipeline(reader=BertQA(adam_epsilon=1e-08, bert_model='bert-base-uncased',
#                          do_lower_case=True, fp16=False,
#                          gradient_accumulation_steps=1, learning_rate=5e-05,
#                          local_rank=-1, loss_scale=0, max_answer_length=30,
#                          n_best_size=20, no_cuda=False,
#                          null_score_diff_threshold=0.0, num_train_epochs=3.0,
#                          output_dir=None, predict_batch_size=8, seed=42,
#                          server_ip='', server_po...size=8,
#                          verbose_logging=False, version_2_with_negative=False,
#                          warmup_proportion=0.1, warmup_steps=0),
#            retrieve_by_doc=False,
#            retriever=BM25Retriever(b=0.75, floor=None, k1=2.0, lowercase=True,
#                                    max_df=0.85, min_df=2, ngram_range=(1, 2),
#                                    preprocessor=None, stop_words='english',
#                                    token_pattern='(?u)\\b\\w\\w+\\b',
#                                    tokenizer=None, top_n=20, verbose=False,
#                                    vocabulary=None))

# In[ ]:


from IPython.display import display, Markdown, Latex, HTML

def show(y,x):
  # w=
  # dh(w)
  z="""<div><div class="question_title">{}</div><div class="single_answer">{}</div></div>""".format(y ,"<span class='answer'>" +x + "</span>")
  dh(z)
def layout_style():
    style = """
        div {
            color: black;
        }
        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }
        .answer{
            color: #dc7b15;
        }
        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }      
        div.output_scroll { 
            height: auto; 
        }
    """
    return "<style>" + style + "</style>"

def dh(z): display(HTML(layout_style() + z))


# In[ ]:



queries = [
    'What is known about transmission, incubation, and environmental stability?',
    'What do we know about COVID-19 risk factors?',
    'What do we know about virus genetics, origin, and evolution?',
    'What do we know about vaccines and therapeutics?',
    'What do we know about non-pharmaceutical interventions?',
    'What has been published about medical care?',
    'What do we know about diagnostics and surveillance?'
    'What has been published about information sharing and inter-sectoral collaboration?',
    'What has been published about ethical and social science considerations?'
]
for query in queries:
  prediction= cdqa_pipeline.predict(query, n_predictions=20,retriever_score_weight=0.6)
  print('Query: {}'.format(query))
  # for x,y,z in zip(prediction[0][:-1],prediction[1][:-1],prediction[2][:-1]):
  show('Answer',str(prediction[0][-2]))
  show('Title',str(prediction[1][-2]))
  show('Paragraph',str(prediction[2][-2]))
    # if x!=prediction[0][-2]:
      # print('-------------Next Prediction-------------')
  if query!=queries[-1]:
    print('---------------Next Query---------------------')

  


# Query: What is known about transmission, incubation, and environmental stability?
# 
# Answer
# Given the high environmental stability of the virus and the potential fomite transmission together with the long virus persistence in infected animals spontaneous disappearance from a mouse population e.g. by cessation of breeding is very unlikely. Eradication of infection is possible by elimination of infected animals and subsequent replacement with uninfected mice and the agent can be eliminated from breeding populations only by embryo transfer or by hysterectomy.
# Title
# Viruses do not replicate outside living cell but infectious virus may persist on contaminated environmental surfaces and the duration of persistence of viable virus is affected markedly by temperature and humidity. Contaminated surfaces are known to be significant vectors in the transmission of infections in the hospital setting as well as the community. The role of fomites in the transmission of RSV has been clearly demonstrated 20 . Survival of viruses on a variety of fomites has been studied for influenza viruses paramyxoviruses poxviruses and retroviruses 21 . The human coronavirus associated with the common cold was reported to remain viable only for 3 hours on environmental surfaces after drying although it remains viable for many days in liquid suspension 13 . Parainfluenza and RSV viruses were viable after drying on surfaces for 2 and 6 hours respectively 20 22 . In aerosolised form human coronavirus 229E is generally less stable in high humidity 12 . The environmental stability of SCoV was previously unknown and this information is clearly important for understanding the mechanisms of transmission of this virus in a hospital and community setting.
# Paragraph
# Most potential biologic weapons agents also are emerging or zoonotic diseases that can be found regularly under natural conditions. The CDC has compiled a list of select agents 90 whose propagation and possession are regulated by federal law based on their lethality and potential for use as biologic weapons 91 . Possession use and transfer of select agents and toxins that pose a severe threat to public health and safety are regulated by federal law to protect the public and laboratory workers. Many zoonotic diseases appear on the overlapping select agent list meaning they are regulated by the CDC under the Department of Health and Human Services or the Department of Agriculture based on human or agricultural risk respectively. The agents that cause VHFs and 1918 pandemic influenza virus are included along with many bacterial pathogens that offer better environmental stability for weaponization. The ideal weapons agents offer a low infectious dose high case fatality rate environmental stability and efficient humantohuman transmission allowing a small inoculum to infect a large population.
# 
# ---------------Next Query---------------------
# Query: What do we know about COVID-19 risk factors?
# 
# Answer
# Here we present details of all patients admitted to the two designated hospitals in WuhanJinyintan Hospital and Wuhan Pulmonary Hospitalwith laboratoryconfirmed COVID 19 outcome death or discharge as of Jan 31 2020. We aim to explore risk factors of inhospital death for patients and describe the clinical course of symptoms viral shedding and tem poral changes of laboratory findings during hospitalisation.
# Title
# Our study is the first comprehensive attempt to systematically assess the effect of a multitude of possible risk factors on severe ALRI in children aged less than five years. We identified in total 19 risk factors which had been reported to be associated with severe ALRI in the published literature. We observed a consistent significant association between 7 risk factors lowbirthweight undernutrition indoor air pollution incomplete immunization at one year HIV breastfeeding and crowding and severe ALRI definite risk factors. We also observed that 7 risk factors parental smoking lack of maternal education vitamin D deficiency male sex preterm births anemia and zinc deficiency had an inconsistent association with severe ALRI that was not significant likely risk factors. We further observed that 5 risk factors daycare birth interval birth order previous history of ALRI and vitamin A deficiency were sporadically reported to be associated with severe ALRI possible risk factors.
# Paragraph
# Evidence before this study We searched PubMed on Feb 23 2020 for articles that documented the risk factors of mortality and viral shedding in patients with coronavirus disease 2019 COVID 19 resulting from infection with severe acute respiratory syndrome coronavirus 2 SARSCoV2 using the search terms novel coronavirus OR SARSCoV2 OR COVID19 AND death OR mortality OR viral shedding with no language or time restrictions. Age comorbidities lymphocytopenia and elevated alanine aminotransferase ddimer creatine kinase highsensitivity cardiac troponin I prothrombin time and disease severity were reported to be associated with intensive care unit admission. However no published works were found about the risk factors of mortality for adult patients with COVID19. One study compared the sensitivity of SARSCoV2 RNA detection in throat and nasopharyngeal swab in 17 patients with COVID19.
# 
# ---------------Next Query---------------------
# Query: What do we know about virus genetics, origin, and evolution?
# 
# Answer
# We know much about virus replication and disease. However our understanding of the specifi c mechanisms of persistence is generally poor. Persistence is a generally silent and inscrutable state it does not lend itself to in vitro or cell culture experimental models. We are left with but a few examples from which to attempt to extrapolate the possible existence of general relationships. The study of virus evolution thus struggles to incorporate concepts of persistence.
# Title
# The 793B genotype was first isolated in China in 2003 the virus isolate was named Taian03 and the sequence of the S1 gene has been deposited in the GenBank database under the accession number AY837465. Since then some other strains of this virus also have been isolated and identified Xu et al. 2007 Han et al. 2009 . Interestingly we recently isolated a virulent nephropathogenic 793B type virus which was found to have emerged from a recombination event between the 491 vaccine strain and an LX4 type also known as QXlike virus Liu et al. 2013 . More recently we isolated another 793B genotype virus which also originated from a recombinant event that resulted in a serotype shift . These results indicate that the 793B type might be becoming a major concern to the poultry industry in China although the origin and evolution of the virus is not clear and little is known about the influence of the genotype on the ecology of other IBVs. In the present study 20 793B isolates were selected by screening 418 IBVs that were isolated from chickens during our continuous surveillance activities for IBV in China from 2009 to 2014. The molecular characteristics antigenicity and pathogenicity of the 20 viruses were investigated to further elucidate the origin and evolution of IBV isolates genotypically related to 793B viruses in China.
# Paragraph
# Several epidemiological surveys have been made to determine the PRCVseroprevalence in Spain Yus et al. 1989 Cubero et al. 1990 Lanza et al. 1990 but the origin and evolution of PRCVinfection in herds was unknown. In order to establish when PRCVinfection appeared in Spain and its evolution during the following years a retrospective survey was conducted. Furthermore a seroepidemiological study was made during 1991 to determine the TGEVand PRCVseroprevalence in the breeding farms of Catalunya.
# 
# ---------------Next Query---------------------
# Query: What do we know about vaccines and therapeutics?
# 
# Answer
# The Filoviridae family consists of the Ebolavirus Marburgvirus and Cuevavirus genera. Historically Ebola virus EBOV Zaire ebolavirus species has been the most common and deadly of the filoviruses. Therefore the research community has largely focused on the development of EBOV animal models tools vaccines and therapeutics and has been successful in producing several compounds that have reached the late stages of clinical trials 1 2 . In light of this success it is now possible to extend further research towards the discovery of panfilovirus vaccines and therapeutics. However animal models that are susceptible to all ebolaviruses species will need to be established first in order to directly evaluate whether panfilovirus vaccines and therapeutics provide crossprotection.
# Title
# With the possible expansion of 2019nCoV globally 8 and the declaration of the 2019nCoV outbreak as a Public Health Emergency of International Concern by the World Health Organization there is an urgent need for rapid diagnostics vaccines and therapeutics to detect prevent and contain 2019nCoV promptly. There is however currently a lack of understanding of what is available in the early phase of 2019nCoV outbreak. The systematic review describes and assesses the potential rapid diagnostics vaccines and therapeutics for 2019nCoV based in part on the developments for MERSCoV and SARSCoV.
# Paragraph
# There are a number of challenges that must be overcome to ensure adequate preparedness for future Ebola outbreaks including completing the remaining advanced development activities necessary for regulatory approval and subsequent stockpiling of these medical countermeasures for use during a public health emergency. BARDA remains committed to making available safe and effective FDAapproved vaccines and therapeutics for Ebola public health emergencies. Despite the advancement of the aforementioned vaccines and therapeutics against Ebola gaps remain in our overall preparedness posture against other filoviruses. As such BARDA will be pursuing the development of vaccines and therapeutics against Sudan ebolavirus and Marburg virus to address this gap. While we acknowledge that much work remains to prepare for future filovirus outbreaks the recently announced BARDA awards for vaccines and therapeutics against Ebola represent an important milestone in our preparedness and ongoing commitment to counter this health security threat.
# 
# ---------------Next Query---------------------
# Query: What do we know about non-pharmaceutical interventions?
# 
# Answer
# Pharmaceutical interventions alone cannot be relied upon to stem the tide of pandemic outbreaks. While influenza transmission can be halted with the use of antiviral medications mutations in the virus necessitate that a new vaccine be produced for each new flu strain. Vaccination production can take up to six months to complete with the burdens of delays likely shortages and virus mismatch reducing the potential impact of the vaccine. Furthermore pharmaceutical interventions often require consultation with a physician or in more severe cases hospitalization. These requirements reduce the potential impact of pharmaceutical interventions due to the fact that many people do not have access to health care or refuse to be seen by a health care provider. Additionally it is often impossible to satisfy this requirement during a pandemic influenza outbreak because the demand for staff facilities and equipment often exceeds the supply 5 . The limitations of pharmaceutical interventions during pandemic influenza outbreaks highlight the importance of also incorporating nonpharmaceutical interventions in public health campaigns aimed at limiting respiratory infectious disease spread.
# Title
# Finally individuals in certain demographic categories may be most receptive to pharmaceutical interventions. Young women from large households expressed the highest level of interests in pharmaceutical interventions and thus may be a potentially successful target of pharmaceutical intervention campaigns. Further study is needed to examine how perceptions and behavior change in response to intervention campaigns.
# Paragraph
# Our survey conducted at the initial stage of outbreak indicated that perceptions about the risks associated with 2009 H1N1 pandemic influenza as well as interest in pharmaceutical interventions and precautionary activities showed changes over time and variations over geography and demography. Although the perceived likelihood of H1N1 infection increased over time interest in preventive pharmaceutical interventions and engagement in information seeking activities declined. These declines were correlated with the decrease in media attention to H1N1 throughout May 2009. We did not observe the decline in engagement in quarantine measures partly because of the small number of respondents who reported the activities.
# 
# ---------------Next Query---------------------
# Query: What has been published about medical care?
# 
# Answer
# Clinical guidelines are meant to establish accepted standards of care and may have important economic implications. Medical Letter published by the Consumers Union is a longstanding and useful publication that reviews therapeutic issues of everyday medical practice and the relevant studies. It represents a balanced updated view of medical practice and summaries of current literature reviewed by respected experienced and competent medical authorities. Clinical practice guidelines are produced by hundreds of professional medical and governmental agencies in order to standardize and improve medical care.
# Title
# After 2005 the government took measures to increase the coverage of medical insurance. The government became aware of the importance of medical service to public welfare. They planned to raise the fiscal budget for the medical care system. However reform will never cease until people can access to more affordable accessible and equal medical service in this first largest populated country. And the milestone in this period is the publication of new scheme for medical care reform in 2009. The government promised to increase the fiscal subsidies rather than rely on market to sufficiently satisfy peoples needs for basic medical services. Meanwhile it would investment 850 million yuan in the next two years to deepen the reform including accelerating the establishment of basic medical care insurance system and essential drug system improving the communitylevel medical service system increasing the equalization in the public medical services and the pilot experiments on public hospital reforms.
# Paragraph
# The outbreak of the severe acute respiratory syndrome SARS epidemic that occurred during 2003 exposed serious deficiencies in Taiwan s medical care and public health care systems as well as its medical education system. The Department of Health Executive Yuan of Taiwan ROC has made efforts in promoting the Project of Reforming Taiwans Medical Care and Public Healthcare System since the spread of SARS was controlled. The reform of the medical care system aimed to provide better holistic medical treatment to people. The strategies and methods are strengthening the improvement of resident education and quality of medical care. A project titled Postgraduate General Medical Training Program was announced by the Department of Health in August 2003. Through this project each doctor in hisher first year of residency including internal medicine family medicine surgery pediatrics dermatologist ophthalmologist etc. is required to fulfill 3 months of an internal medical training course along with 36 hours of basic courses. In the past there was no such program in Taiwan to provide general medical training for medical students after graduation. Therefore the goal of this program is to ensure that all PGY 1 residents have acquired Accreditation Council for Graduate Medical Education ACGME core competence in internal medical care.
# 
# ---------------Next Query---------------------
# Query: What do we know about diagnostics and surveillance?What has been published about information sharing and inter-sectoral collaboration?
# 
# Answer
# The aim is to support cooperation and coordinated action of EUMS to improve their capacities at points of entry airports ports groundcrossings in preventing and combating crossborder health threats from the transport sector. The action activities include the following a facilitating EU MS evaluating and monitoring of core capacities at PoE b strengthening inter sectoral and cross sectoral collaboration through a communication network c producing catalogues of tested best practices guidelines and validated action plans d providing capacity buildingtraining on tested best practices guidelines validated action plans e facilitating EU MSs coordinating and executing hygiene inspections on conveyances f combatting all types of health threats focusing on infectious disease and vectors g supporting response to possible future public health emergencies of international concern. In future public health emergencies the action will move from interepidemic mode to emergency mode supporting coherent response as per Decision n10822013EU International Health Regulations and WHO temporary recommendations.
# Title
# One review suggests that there is little reason in theory why cooperation in information sharing could not continue but this depends on financing and investment in collaboration and the UK being given adequacy status under GDPR for sharing of personal information 27 . Many countries aspiring to be in the EU aspire to EU standards in health security management for example Turkey as an accession state 31 .
# Paragraph
# Since the outbreak of atypical pneumonia cases was detected in Wuhan at the end of December 2019 the Chinese Center for Disease Control and Prevention China CDC has launched a new surveillance system first in Wuhan then extended to the entire country to record information on COVID19 cases. Case definitions for suspected cases and laboratoryconfirmed cases and the description of the surveillance system have been published 1 and reported elsewhere 5 . Details are summarized in Appendix page 2.
# 
# ---------------Next Query---------------------
# Query: What has been published about ethical and social science considerations?
# 
# Answer
# In January 2006 the Council set up a working party to examine the ethical issues surrounding public health. This was chaired by Lord Krebs and included members with expertise in health economics law philosophy public health policy health promotion and social science. This article summarizes some of the conclusions and recommendations that were published in the report Public health ethical issues 1 in November 2007 and presented to the UK Public Health Association Annual Public Health Forum in April 2008.
# Title
# Research involving bioethics and social equity helps scientists incorporate ethical principles in the design and conduct studies involving human participants affected by public health emergencies 78 . Such studies are critically important for research examining the effectiveness of candidate vaccines and medicines understanding pathogen transmission and infection in natural settings and testing nonpharmaceutical interventions for disease prevention and mitigation. Although such studies have been conducted for years the U.S. National Academies of Science Engineering and Medicine highlighted research needs for preparedness and response to public health emergencies and associated bioethical considerations 79 . This focus on the bioethics of disaster research has prompted nongovernmental and governmental organizations alike to evaluate challenges and identify solutions to promote ethical practices in research during public health emergencies. Building on this and other social science research can promote the development and implementation of clinical and public health research that takes into account the culture society and benefits to and needs of research participants.
# Paragraph
# In the interest of comprehensiveness this study reviews various types of work journal articles book reviews published in both Englishand Chineselanguage journals that studied crisis communication in the three regions Mainland China Taiwan and Hong Kong from two academic perspectives public relations and communication. The criteria used for journal selection were high general reputation a strong influence on Chinese communication research and accreditation by way of inclusion in certain scholarly indices such as the Social Science Index SSCI the Chinese Social Science Index CSSCI and the Taiwanese Social Science Index TSSCI. We first looked at two major journals in the field of public relations the In order to represent academic journals from Hong Kong Communication and Society TSSCI was reviewed. 1 Titles of articles were queried with the key words crisiscrises and abstracts were queried with the key words ChinaChineseTaiwanTaiwaneseHong KongMacauMacanese. Only articles focusing on crisis management were included in the final sample. Articles selected for analysis were published in 15 academic journals between 1999 and 2014. These articles focused explicitly on Chinese crisis communication. Initial keyword searches were conducted in January 2012 further keyword searches were conducted in October 2014. This yielded a total of 93 articles for analysis 56 in January 2012 and 37 in October 2014. 2