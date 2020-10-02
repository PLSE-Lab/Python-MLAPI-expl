#!/usr/bin/env python
# coding: utf-8

# ### Summary Table of COVID-19 Population Studies
# Myrna M Figueroa Lopez   

# In[ ]:


##libraries
import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# #### Purpose:
# Summary of population studies for the Open Research Dataset Challenge (CORD-19) Round 2.    
#       

# Includes:   
# The tables and csv files show what the literature reports about:   
# 
# 1. **Ways of communicating** with target high-risk populations (elderly, health care workers).
# 2. **Management of patients who are underhoused** or otherwise lower socioeconomic status. 
# 3. What are ways to create hospital infrastructure to prevent **nosocomial outbreaks** and **protect uninfected patients**?
# 4. Methods to **control the spread** in communities, **barriers to compliance**.   
# 5. What are **recommendations** for combating/overcoming resource failures?  
# 
# Also, a summary table of COVID-19 population studies abstracts.   
# 

# **Introduction**:   
# In this notebook, I present a summary table on population studies on CORD-19.   
# 
# **Methodology**    
# Using Kaggle datafiles:   
#     metadata.csv   
#     list_of_tables_and_table_formats.csv   
# Also, links found by searching google for relevant sources.    
# 
# **Results**   
# Dataframes (TABLES) and CSV files saved to the output folder of this notebook.    
# These contain summary tables.    
# 

# ### TABLES
# The first table, tablesTable, is a directory of files saved in this notebook with article sources and online links.    
# Purpose: Ease of read.
# 

# In[ ]:


##Table format 
TABLEFORMAT=pd.read_csv('../input/CORD-19-research-challenge/Kaggle/list_of_tables_and_table_formats.csv')
t=TABLEFORMAT[0:5]


# In[ ]:


##Table of TABLES
#a directory of CSV files created
tablesTable=t[['Question']]


# In[ ]:


# Insert a row at the end 
tablesTable.loc[5] = 'Other relevant sources' 


# In[ ]:


data = {'From Abstacts file':['Q1_articles_Ways_of_communicating_with_highrisk_pop.csv', 'Q2_Mangmnt_underhoused_or_low_income_patients.csv', 'Q3_Articles_Ways_for_hospitals_prevent_nosocomial_outbreaks_and_protect_uninfected.csv', 'Q4_Articles_Methods_control_spread_in_communities_and_barriers_to_compliance.csv','Q5_Journal_Recommendations_for_overcoming_COVID19_resource_failures.csv','COVID19_Journal_Abrstracts.csv'], 
        'Online sources file':['Q1_online_Ways_of_communicating_with_highrisk_pop.csv', 'Q2_online_Managing_homeless_or_lowIncome_patients.csv','Q3_online_Ways_hospital_infrastr_prevent_nosocomial_outbreaks_and_protect_uninfected.csv', 'Q4_online_Methods_control_community_spread_and_barriers_to_compliance.csv','Q5_online_Recommendations_for_overcoming_COVID19_resource_failures.csv', 'COVID19_Population_Studies_online.csv']}
  
# table dataframe: DIRECTORY 
tbl2 = pd.DataFrame(data) 
tablesTable=tablesTable.join(tbl2)
tablesTable


# ### Table: ABSTRACTS   
# The (df) table below is a 'cleaned' (no missing entries, ease of read) table of articles and their abstracts in the KAGGLE DATASET for further review.

# In[ ]:


# METADATA FILE
## a table of 18 columns and 63571 entries
df1=pd.read_csv('../input/CORD-19-research-challenge/metadata.csv') 
df1.shape


# In[ ]:


#selecting specific columns
##these include the title of the article, its abstract, date, link to the article, and authors
journals= df1[['title', 'abstract', 'publish_time', 'url', 'authors']]

##ABSTRACTS
#separate each word in the ABSTRACT column
journals['words'] = journals.abstract.str.strip().str.split('[\W_]+')
journals['words'].head()


# In[ ]:


#separate words in the abstract column and create a new column
abstracts = journals[journals.words.str.len() > 0]

# saving the dataframe 
abstracts.to_csv('COVID19_Journal_Abrstracts.csv') 

#display dataframe
abstracts.head(3)


# In[ ]:


abstracts.shape
#There are 51,012 abstracts in the metadata csv file.


# ## BROWSING THE ABSTRACTS FOR ANSWERS
# 
# **Question 1**
# 1. Related to modes of communicating with target high-risk populations (elderly, health care workers).

# In[ ]:


#looking for abstracts with specific terms among publications
#in the dataset provided by Kaggle

Q1A=abstracts[abstracts['abstract'].str.contains('communicating')]


# In[ ]:


Q1B=abstracts[abstracts['abstract'].str.contains('with high-risk')]


# In[ ]:


Q1C=abstracts[abstracts['abstract'].str.contains('contacting')]


# In[ ]:


Q1D=abstracts[abstracts['abstract'].str.contains('elderly')]


# In[ ]:


Q1F=abstracts[abstracts['abstract'].str.contains('health care workers')]


# In[ ]:


Q1E=abstracts[abstracts['abstract'].str.contains('risk population')]


# In[ ]:


# Concatenating the dataframes into one table per question
Question1= pd.concat([Q1A, Q1B, Q1C, Q1D, Q1E, Q1F])
# dropping null value columns to avoid errors 
Question1.dropna(inplace = True) 

Question1.shape


# In[ ]:


#Relevant articles in the dataset to Q1
Question1.head(3)


# In[ ]:


#686 articles identified for QUESTION 1
## Saved as CSV file
Question1.to_csv('Q1_articles_Ways_of_communicating_with_highrisk_pop.csv')


# 2. Management of patients who are underhoused or otherwise lower socioeconomic status.

# In[ ]:


##Term: homeless
Q2a=abstracts[abstracts['abstract'].str.contains('homeless')]


# In[ ]:


Q2b=abstracts[abstracts['abstract'].str.contains('low income')]


# In[ ]:


Q2c=abstracts[abstracts['abstract'].str.contains('poverty')]


# In[ ]:


Q2d=abstracts[abstracts['abstract'].str.contains('housing')]


# In[ ]:


# Concatenating the dataframes into one table per question
Question2= pd.concat([Q2a,Q2b,Q2c, Q2d])
# dropping null value columns to avoid errors 
Question2.dropna(inplace = True) 

Question2.shape


# In[ ]:


# saving the dataframe of ARTICLES related to Q2
Question2.to_csv('Q2_Mangmnt_underhoused_or_low_income_patients.csv')


# 3. What are ways to create hospital infrastructure to prevent nosocomial outbreaks and protect uninfected patients?

# In[ ]:


##Term: nosocomial
q3a=abstracts[abstracts['abstract'].str.contains('nosocomial')]


# In[ ]:


q3b=abstracts[abstracts['abstract'].str.contains('hospital spread')]


# In[ ]:


q3c=abstracts[abstracts['abstract'].str.contains('hospital patients')]


# In[ ]:


q3d=abstracts[abstracts['abstract'].str.contains('nosocomial outbreak')]


# In[ ]:


# Concatenating the dataframes into one table per question
Q3= pd.concat([q3a,q3b,q3c, q3d])
# dropping null value columns to avoid errors 
Q3.dropna(inplace = True) 

Q3.shape


# In[ ]:


Q3.to_csv('Q3_Articles_Ways_for_hospitals_prevent_nosocomial_outbreaks_and_protect_uninfected.csv')


# 4. What are methods to control the spread in communities, barriers to compliance?

# In[ ]:


##Term: compliance
q4A=abstracts[abstracts['abstract'].str.contains('compliance')]


# In[ ]:


q4B=abstracts[abstracts['abstract'].str.contains('community spread')]


# In[ ]:


q4C=abstracts[abstracts['abstract'].str.contains('prevent spread')]


# In[ ]:


q4D=abstracts[abstracts['abstract'].str.contains('methods to prevent')]


# In[ ]:


# Concatenating the dataframes into one table per question
Question4= pd.concat([q4A,q4B,q4C, q4D])
# dropping null value columns to avoid errors 
Question4.dropna(inplace = True) 

Question4.shape


# In[ ]:


Question4.to_csv('Q4_Articles_Methods_control_spread_in_communities_and_barriers_to_compliance.csv')


# 5. What are recommendations for combating/overcoming resource failures
# 

# In[ ]:


q5a=abstracts[abstracts['abstract'].str.contains('not reach')]


# In[ ]:


q5b=abstracts[abstracts['abstract'].str.contains('improve access')]


# In[ ]:


q5c=abstracts[abstracts['abstract'].str.contains('access to resource')]


# In[ ]:


q5d=abstracts[abstracts['abstract'].str.contains('outreach')]


# In[ ]:


q5e=abstracts[abstracts['abstract'].str.contains('faulty')]


# In[ ]:


q5f=abstracts[abstracts['abstract'].str.contains('meet demand')]


# In[ ]:


q5g=abstracts[abstracts['abstract'].str.contains('waste')]


# In[ ]:


# Concatenating the dataframes into one table per question
Question5= pd.concat([q5a,q5b,q5c, q5d, q5e, q5f, q5g])
# dropping null value columns to avoid errors 
Question5.dropna(inplace = True) 

Question5.shape


# #### ONLINE: Using Google search to find relevant links to questions

# In[ ]:


pip install beautifulsoup4


# In[ ]:


pip install google


# In[ ]:


try: 
    from googlesearch import search 
except ImportError:  
    print("Error/Not found") 
### browsing for: COVID 19 population studies  
query = "COVID 19 population studies"
##limit to the first 10 relevant links:
for j in search(query, tld="co.in", num=10, stop=10, pause=2): 
    print(j)  


# In[ ]:


StudiesDictionary={'url': ['https://www.cdc.gov/coronavirus/2019-ncov/covid-data/serology-surveillance/index.html',
                   'https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30854-0/fulltext',
                   'https://iussp.org/fr/node/11297', 'https://www.nejm.org/doi/full/10.1056/NEJMp2006761',
                   'https://www.vox.com/2020/4/24/21229415/coronavirus-antibody-testing-covid-19-california-survey',
                    'https://www.statnews.com/2020/04/04/cdc-launches-studies-to-get-more-precise-count-of-undetected-covid-19-cases/',
                    'https://ourworldindata.org/coronavirus', 
                    'https://www.360dx.com/infectious-disease/new-york-california-serology-studies-give-early-estimates-covid-19-prevalence',
                    'https://www.popcouncil.org/research/responding-to-the-COVID-19-pandemic', 
                    'https://www.nature.com/articles/s41591-020-0883-7']}


# In[ ]:


StudiesDF=pd.DataFrame.from_dict(StudiesDictionary)
StudiesDF


# In[ ]:


# saving the dataframe of google search results 
StudiesDF.to_csv('COVID19_Population_Studies_online.csv') 


# Q1 online search:

# In[ ]:


try: 
    from googlesearch import search 
except ImportError:  
    print("Error/Not found") 
### browsing for: COVID 19 population studies  
search1 = "ways to reach COVID 19 vulnerable population"
##limit to the first 10 relevant links:
for a in search(search1, tld="co.in", num=10, stop=10, pause=2): 
    print(a)


# In[ ]:


try: 
    from googlesearch import search 
except ImportError:  
    print("Error/Not found") 
### browsing for: COVID 19 population studies  
search2 = "communicating with COVID 19 high risk population"
##limit to the first 10 relevant links:
for a2 in search(search2, tld="co.in", num=10, stop=10, pause=2): 
    print(a2)


# In[ ]:


try: 
    from googlesearch import search 
except ImportError:  
    print("Error/Not found") 
### browsing for: COVID 19 population studies  
search3 = "telehealth and COVID19"
##limit to the first 10 relevant links:
for a3 in search(search3, tld="co.in", num=10, stop=10, pause=2): 
    print(a3)


# In[ ]:


try: 
    from googlesearch import search 
except ImportError:  
    print("Error/Not found") 
### browsing for: COVID 19 population studies  
search4 = "communicating with the elderly about COVID19"
##limit to the first 10 relevant links:
for a4 in search(search4, tld="co.in", num=10, stop=10, pause=2): 
    print(a4)


# In[ ]:


##Not all links included
###Only those most relevant are included
#Chosen manually
Q1CommDictionary={'url': ['https://www.ncoa.org/blog/managing-the-covid-19-crisis-for-vulnerable-populations/',
    'https://reliefweb.int/sites/reliefweb.int/files/resources/COVID-19_CommunityEngagement_130320.pdf',
    'https://reliefweb.int/report/world/covid-19-how-include-marginalized-and-vulnerable-people-risk-communication-and',
    'https://www.commonwealthfund.org/blog/2020/how-cities-can-provide-rapid-relief-vulnerable-people-during-covid-19-crisis',
    'https://www.cdc.gov/coronavirus/2019-ncov/php/public-health-communicators-get-your-community-ready.html',
    'https://www.weforum.org/agenda/2020/03/covid19-minimize-impact-on-vulnerable-communities/',
    'https://www.un.org/en/un-coronavirus-communications-team/un-working-ensure-vulnerable-groups-not-left-behind-covid-19',
    'https://www.healthaffairs.org/do/10.1377/hblog20200319.757883/full/',    
    'https://www.cdc.gov/coronavirus/2019-ncov/communication/public-service-announcements.html',
    'https://www.cdc.gov/coronavirus/2019-ncov/communication/videos.html',
    'https://www.cdc.gov/coronavirus/2019-ncov/communication/social-media-toolkit.html',
    'https://www.cdc.gov/coronavirus/2019-ncov/travelers/communication-resources.html',
    'https://www.hhs.gov/sites/default/files/telehealth-faqs-508.pdf',
    'https://www.fcc.gov/covid-19-telehealth-program',
    'https://www.nejm.org/doi/full/10.1056/NEJMp2003539',
    'https://www.statnews.com/2020/04/19/telehealth-silver-lining-discovered-covid-19-crisis/',
    'https://www.aafp.org/patient-care/emergency/2019-coronavirus/telehealth.html',
    'https://lernercenter.syr.edu/2020/04/06/tips-for-communicating-with-older-adults-about-covid-19/',
    'https://www.eraliving.com/blog/4-ways-seniors-can-connect-with-others-during-the-coronavirus-outbreak/',
    'https://www.pnj.com/story/opinion/2020/04/15/communication-elderly-vital-during-covid-19-guestview/5136966002/',
    'https://www.hopkinsmedicine.org/health/conditions-and-diseases/coronavirus/coronavirus-caregiving-for-the-elderly',
    'https://www.munsonhealthcare.org/blog/5-tips-for-talking-to-older-adults-about-covid-19',
    'https://www.commercialappeal.com/story/news/2020/04/24/seniors-and-families-communication-key-during-covid-19-pandemic/3008125001/']}


# In[ ]:


Q1Comm=pd.DataFrame.from_dict(Q1CommDictionary)
Q1Comm


# In[ ]:


# saving the dataframe of google search results related to Q1
Q1Comm.to_csv('Q1_online_Ways_of_communicating_with_highrisk_pop.csv') 


# Question 2

# In[ ]:


##Question2
try: 
    from googlesearch import search 
except ImportError:  
    print("Error/Not found") 
### browsing for: COVID 19 population studies  
browse1 = "homeless COVID 19 patients"
##limit to the first 10 relevant links:
for b1 in search(browse1, tld="co.in", num=10, stop=10, pause=2): 
    print(b1)


# In[ ]:


try: 
    from googlesearch import search 
except ImportError:  
    print("Error/Not found") 
### browsing for: COVID 19 population studies  
browse2 = "low income COVID 19 patients"
##limit to the first 10 relevant links:
for b2 in search(browse2, tld="co.in", num=10, stop=10, pause=2): 
    print(b2)


# In[ ]:


Q2HomeDictionary= {'url':['https://www.statnews.com/2020/04/11/coronavirus-san-francisco-homeless-doctors-difficult-choices/',
    'https://www.cdc.gov/coronavirus/2019-ncov/community/homeless-shelters/unsheltered-homelessness.html',
    'https://www.cdc.gov/coronavirus/2019-ncov/community/homeless-shelters/plan-prepare-respond.html',
    'https://www.cdc.gov/mmwr/volumes/69/wr/mm6917e2.htm',
    'https://www.beckershospitalreview.com/public-health/homeless-covid-19-patients-put-strain-on-san-francisco-hospitals.html',
    'https://www.boston.com/news/local-news/2020/04/16/covid-19-homeless-patient-challenges',
    'https://www.sciencedaily.com/releases/2020/04/200430191258.htm',
    'https://www.nejm.org/doi/full/10.1056/NEJMp2005638',
    'https://www.beckershospitalreview.com/public-health/low-income-groups-people-of-color-at-higher-risk-of-serious-covid-19-illness.html',
    'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7173081/',
    'https://www.thelancet.com/journals/lanres/article/PIIS2213-2600(20)30114-4/fulltext',
    'https://www.bellinghamherald.com/news/coronavirus/article242745386.html',
    'https://nlihc.org/coronavirus-and-housing-homelessness']}


# In[ ]:


Q2Home=pd.DataFrame.from_dict(Q2HomeDictionary)
Q2Home


# In[ ]:


# saving the dataframe of google search results related to Q1
Q2Home.to_csv('Q2_online_Managing_homeless_or_lowIncome_patients.csv')


# Q3 online search

# In[ ]:


try: 
    from googlesearch import search 
except ImportError:  
    print("Error/Not found") 
### browsing for: COVID 19 population studies  
online1 = "prevent contamination of uninfected patients"
##limit to the first 10 relevant links:
for o1 in search(online1, tld="co.in", num=10, stop=10, pause=2): 
    print(o1)


# In[ ]:


try: 
    from googlesearch import search 
except ImportError:  
    print("Error/Not found") 
### browsing for: COVID 19 population studies  
online2 = "hospital spread prevention"
##limit to the first 10 relevant links:
for o2 in search(online1, tld="co.in", num=10, stop=10, pause=2): 
    print(o2)


# In[ ]:


try: 
    from googlesearch import search 
except ImportError:  
    print("Error/Not found") 
### browsing for: COVID 19 population studies  
online3 = "prevent nosocomial outbreaks in hospitals"
##limit to the first 10 relevant links:
for o3 in search(online1, tld="co.in", num=10, stop=10, pause=2): 
    print(o3)


# In[ ]:


Q3HospitalOutBDictionary={'url': ['https://www.ncbi.nlm.nih.gov/books/NBK2683/',
    'https://www.cdc.gov/infectioncontrol/guidelines/isolation/prevention.html',
    'https://www.who.int/water_sanitation_health/medicalwaste/148to158.pdf',
    'http://www.infectioncontroltoday.com/transmission-prevention/patient-colonization-implications-and-possible-solutions-contamination',
    'https://lms.rn.com/getpdf.php/2208.pdf',
    'https://www.health.ny.gov/professionals/diseases/reporting/communicable/infection/95-14_vre_control_guidelines.htm',
    'https://www.healthline.com/health/cross-infection',
    'https://www1.health.gov.au/internet/publications/publishing.nsf/Content/cda-cdna-norovirus.htm-l~cda-cdna-norovirus.htm-l-8',
    'http://www.bccdc.ca/resource-gallery/Documents/Guidelines%20and%20Forms/Guidelines%20and%20Manuals/Epid/CD%20Manual/Chapter%203%20-%20IC/InfectionControl_GF_IC_In_Physician_Office.pdf',
    'https://www.cochrane.org/CD011621/protective-clothes-and-equipment-healthcare-workers-prevent-them-catching-coronavirus-and-other']}


# In[ ]:


Q3HospitalOutB=pd.DataFrame.from_dict(Q3HospitalOutBDictionary)
Q3HospitalOutB


# In[ ]:


# saving the dataframe of google search results related to Q1
Q3HospitalOutB.to_csv('Q3_online_Ways_hospital_infrastr_prevent_nosocomial_outbreaks_and_protect_uninfected.csv')


# Question 4 online

# In[ ]:


try: 
    from googlesearch import search 
except ImportError:  
    print("Error/Not found") 
### browsing for: COVID 19 population studies  
online3 = "prevent nosocomial outbreaks in hospitals"
##limit to the first 10 relevant links:
for o3 in search(online1, tld="co.in", num=10, stop=10, pause=2): 
    print(o3)


# In[ ]:


try: 
    from googlesearch import search 
except ImportError:  
    print("Error/Not found") 
### browsing for: COVID 19 population studies  
seek1 = "prevent COVID community spread"
##limit to the first 10 relevant links:
for s1 in search(seek1, tld="co.in", num=10, stop=10, pause=2): 
    print(s1)


# In[ ]:


try: 
    from googlesearch import search 
except ImportError:  
    print("Error/Not found") 
### browsing for: COVID 19 population studies  
seek2 = "COVID 19 compliance"
##limit to the first 10 relevant links:
for s2 in search(seek2, tld="co.in", num=10, stop=10, pause=2): 
    print(s2)


# In[ ]:


Q4SpreadDictionary= {'url':['https://www.cdc.gov/coronavirus/2019-ncov/community/index.html',
    'https://www.cdc.gov/coronavirus/2019-ncov/community/community-mitigation.html',
    'https://www.cdc.gov/coronavirus/2019-ncov/community/large-events/index.html',
    'https://www.cdc.gov/coronavirus/2019-ncov/community/schools-childcare/index.html',
    'https://www.cdc.gov/coronavirus/2019-ncov/prevent-getting-sick/prevention.html',
    'https://www.cdc.gov/coronavirus/2019-ncov/prevent-getting-sick/index.html',
    'https://www.cdc.gov/coronavirus/2019-ncov/prevent-getting-sick/disinfecting-your-home.html',
    'https://www.cdc.gov/coronavirus/2019-ncov/prevent-getting-sick/social-distancing.html',
    'https://espanol.cdc.gov/coronavirus/2019-ncov/prevent-getting-sick/index.html',
    'http://dph.illinois.gov/topics-services/diseases-and-conditions/diseases-a-z-list/coronavirus/preventing-spread-communities',
    'https://www.osha.gov/SLTC/covid-19/standards.html',
    'https://www.epa.gov/enforcement/covid-19-enforcement-and-compliance-resources',
    'https://www.complianceweek.com/covid-19/7208.tag',
    'https://www.nj.gov/dep/covid19regulatorycompliance/',
    'https://www2.deloitte.com/us/en/pages/regulatory/articles/regulatory-response-to-covid-19.html',
    'https://www.acacompliancegroup.com/tags/covid-19-resources',
    'http://www.dli.mn.gov/business/workplace-safety-and-health/mnosha-compliance-novel-coronavirus-covid-19']}


# In[ ]:


Q4Comply=pd.DataFrame.from_dict(Q4SpreadDictionary)
Q4Comply


# In[ ]:


# saving the dataframe of google search results related to Q5
Q4Comply.to_csv('Q4_online_Methods_control_community_spread_and_barriers_to_compliance.csv') 


# Online search for Q5

# In[ ]:


try: 
    from googlesearch import search 
except ImportError:  
    print("Error/Not found") 
# browsing for: COVID 19 resources failure  
query2 = "recommendations for COVID 19 resources limits"
 
for j2 in search(query2, tld="co.in", num=10, stop=10, pause=2): 
    print(j2) 


# In[ ]:


try: 
    from googlesearch import search 
except ImportError:  
    print("Error/Not found") 
# browsing for: COVID 19 resources failure  
query3 = "recommendations for COVID 19 testing problems"
 
for j3 in search(query3, tld="co.in", num=10, stop=10, pause=2): 
    print(j3) 


# In[ ]:


try: 
    from googlesearch import search 
except ImportError:  
    print("Error/Not found") 
# browsing for: COVID 19 resources failure  
query4 = "recommendations for PPE problems"
 
for j4 in search(query4, tld="co.in", num=10, stop=10, pause=2): 
    print(j4) 


# In[ ]:


try: 
    from googlesearch import search 
except ImportError:  
    print("Error/Not found") 
# browsing for: COVID 19 resources failure  
query5 = "recommendations for improving access to COVID 19 resources"
 
for j5 in search(query5, tld="co.in", num=10, stop=10, pause=2): 
    print(j5) 


# In[ ]:


Q5RecomDictionary={'url': ['https://www.ama-assn.org/delivering-care/public-health/covid-19-policy-recommendations-oud-pain-harm-reduction',
                           'https://www.statnews.com/2020/03/31/covid-19-overcoming-testing-challenges/', 
                           'https://apps.who.int/iris/bitstream/handle/10665/331509/WHO-COVID-19-lab_testing-2020.1-eng.pdf',
                           'https://www.modernhealthcare.com/technology/covid-19-testing-problems-started-early-us-still-playing-behind',
                            'https://www.modernhealthcare.com/technology/labs-face-challenges-creating-diagnosis-testing-covid-19',
                            'https://www.ama-assn.org/delivering-care/public-health/covid-19-frequently-asked-questions',
                            'https://www.vox.com/recode/2020/4/24/21229774/coronavirus-covid-19-testing-social-distancing',
                            'https://www.vdh.virginia.gov/coronavirus/health-professionals/vdh-updated-guidance-on-testing-for-covid-19/',
                            'https://www.mckinsey.com/industries/healthcare-systems-and-services/our-insights/major-challenges-remain-in-covid-19-testing',
                            'https://www.fda.gov/medical-devices/emergency-situations-medical-devices/faqs-testing-sars-cov-2',
                            'https://www.jointcommission.org/resources/news-and-multimedia/news/2020/03/statement-on-shortages-of-personal-protective-equipment-amid-covid-19-pandemic/',
                            'https://jamanetwork.com/journals/jama/fullarticle/2764238','https://www.ncbi.nlm.nih.gov/books/NBK209587/',
                            'https://www.cdc.gov/coronavirus/2019-ncov/hcp/ppe-strategy/index.html',
                            'https://www.cdc.gov/coronavirus/2019-ncov/hcp/ppe-strategy/burn-calculator.html',
                            'https://www.cdc.gov/coronavirus/2019-ncov/hcp/ppe-strategy/face-masks.html',
                            'https://www.cdc.gov/coronavirus/2019-ncov/hcp/respirators-strategy/index.html',
                            'https://www.cdc.gov/coronavirus/2019-ncov/hcp/ppe-strategy/eye-protection.html',
                            'http://www.infectioncontroltoday.com/personal-protective-equipment/addressing-challenges-ppe-non-compliance',
                            'https://www.healio.com/gastroenterology/practice-management/news/online/%7B331d768c-91dd-4094-a2fd-6c9b0c07627d%7D/aga-issues-covid-19-recommendations-for-ppe-use-during-gi-procedures',
                            'https://www.facs.org/covid-19/ppe/additional' ]}


# In[ ]:


Q5Recomm=pd.DataFrame.from_dict(Q5RecomDictionary)
Q5Recomm


# In[ ]:


# saving the dataframe of ARTICLES related to Q5
Question5.to_csv('Q5_Journal_Recommendations_for_overcoming_COVID19_resource_failures.csv') 

# saving the dataframe of google search results related to Q5
Q5Recomm.to_csv('Q5_online_Recommendations_for_overcoming_COVID19_resource_failures.csv') 

