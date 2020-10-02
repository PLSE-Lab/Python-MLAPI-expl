#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


#multi_choice = pd.read_csv('C:\\Users\\goldie.sahni\\Downloads\\KaggleSurvey\\multiple_choice_responses.csv')


# In[ ]:


multi_choice = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")


# In[ ]:


multi_choice.shape


# In[ ]:


listcols = []

for x in multi_choice.columns:
    
    if '_OTHER_TEXT' in x:
        
        listcols.append(x)


# In[ ]:


multi_choice.drop(listcols,axis=1,inplace=True)


# In[ ]:


#Qs = multi_choice.loc[0,:]


# In[ ]:


multi_choice.drop([0],inplace=True)


# In[ ]:


listnum = ['Q9','Q12','Q13','Q16','Q17','Q18','Q20','Q21','Q24','Q25','Q26','Q27','Q28','Q29','Q30','Q31','Q32','Q33','Q34']


# In[ ]:


for n in listnum:
    
    print(n)
    
    listcols1 = []
    
    na_string = 'No_Value'    
       
    for x in multi_choice.columns:
        
        if n in x:
            
            listcols1.append(x)
            
    df1 = multi_choice.loc[:,listcols1]
    
    df1[n + '_whole'] = ''
    
    for c in listcols1:
        
        new = df1[c].copy()
        
        df1[n + '_whole']= df1[n + '_whole'].str.cat(new, sep =",", na_rep = na_string)
        
    df1[n + '_whole'] = df1[n + '_whole'].apply(lambda x : x[1: ])
    
    df1[n + '_whole'] = df1[n + '_whole'].apply(lambda x : x.replace(',No_Value',''))
    
    #df1[n + '_whole'] = df1[n + '_whole'].apply(lambda x : x.replace('No_Value,',''))
    
    df1[n + '_whole'] = df1[n + '_whole'].apply(lambda x : x.replace(',None',''))
    
    df1[n + '_whole'] = df1[n + '_whole'].apply(lambda x : x.replace('No_Value,',''))
    
    multi_choice[n + '_whole'] = df1[n + '_whole'].copy()

    print(df1.loc[13,n + '_whole'])   
           


# In[ ]:


part_cols = []

for x in multi_choice.columns:
    
    if '_Part_' in x:
        
        part_cols.append(x)


# In[ ]:


multi_choice.drop(part_cols,axis=1,inplace=True)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


#  <font size='10' color='green'>DATA SCIENCE INSIGHTS FOR INDIA</font>

#  <font size='7' color='BLUE'>---------------PEOPLE------------------</font>

#  <font size='6' color='RED'>HIGHEST % OF RESPONDENTS - INDIA</font>

# # Highest percentage of respondents are from India.Data Science is hot right now in India.
# 

# This means that Data Science has a very bright future in India. Data Science is poised for a robust growth as a career option in India. Indians will take over many of Data Science positions in the whole world.

# In[ ]:


ax = (multi_choice['Q3'].value_counts()[0:5]*100/multi_choice.shape[0]).plot(kind='bar',figsize=(20,10))
ax.set_xlabel("country",fontsize=20)
ax.set_ylabel("%",fontsize = 20)
ax.tick_params(labelsize=20)


#  <font size='6' color='RED'>MOST PREVALENT GENDER - MALE</font>

# # Indian Males constitute highest % of respondents. Indian males perceive Data Science as a great & hip career option.

# In[ ]:


ax1 = (multi_choice.loc[multi_choice['Q3']=='India','Q2'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q2'].shape[0]).plot(kind='bar',figsize=(20,10))
ax1.set_xlabel("gender",fontsize=20)
ax1.set_ylabel("%",fontsize = 20)
ax1.tick_params(labelsize=20)


#  <font size='6' color='RED'>MOST PREVALENT AGE GROUP - 18 TO 21 YEARS</font>

# # 18-21 age group in India constitutes highest percentage of respondents. Youngsters form major group which looks at Data Science as good career option promising high growth and good salary.

# In[ ]:


ax2 = (multi_choice.loc[multi_choice['Q3']=='India','Q1'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q1'].shape[0]).plot(kind='bar',figsize=(20,10))
ax2.set_xlabel("age_group",fontsize=20)
ax2.set_ylabel("%",fontsize = 20)
ax2.tick_params(labelsize=20)


#  <font size='6' color='RED'>MOST PREVALENT STUDY PROGRAM - BACHELORS PROGRAM</font>

# # Most Indians are hoping to enter Data Science field after studying Bachelor's degree. Highest % of respondents from India are doing bachelor degree.

# In[ ]:


ax3 = (multi_choice.loc[multi_choice['Q3']=='India','Q4'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q4'].shape[0]).plot(kind='bar',figsize=(20,10))
ax3.set_xlabel("degree",fontsize=20)
ax3.set_ylabel("%",fontsize = 20)
ax3.tick_params(labelsize=20)


#  <font size='6' color='RED'>MOST PREVALENT OCCUPATION - STUDENTS</font>

# # Most respondents from India are students hoping to become Data Scientists.

# In[ ]:


ax4 = (multi_choice.loc[multi_choice['Q3']=='India','Q5'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q5'].shape[0]).plot(kind='bar',figsize=(20,10))
ax4.set_xlabel("occupation",fontsize=20)
ax4.set_ylabel("%",fontsize = 20)
ax4.tick_params(labelsize=20)


#  <font size='7' color='BLUE'>---------------BUSINESS-------------------</font>

#  <font size='6' color='RED'>MOST PREVALENT COMPANY SIZE - > 10000 EMPLOYEES</font>

# # Most Respondents from India are working in big companies. Data Science teams are big with workload shared between 20 or more people. Data Science is getting big in India.

# In[ ]:


ax5 = (multi_choice.loc[multi_choice['Q3']=='India','Q6'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q6'].shape[0]).plot(kind='bar',figsize=(20,10))
ax5.set_xlabel("no._of_employees",fontsize=20)
ax5.set_ylabel("%",fontsize = 20)
ax5.tick_params(labelsize=20)


#  <font size='6' color='RED'>MOST PREVALENT DATA SCIENCE GROUP SIZE - 20+</font>

# In[ ]:


ax6 = (multi_choice.loc[multi_choice['Q3']=='India','Q7'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q7'].shape[0]).plot(kind='bar',figsize=(20,10))
ax6.set_xlabel("people_sharing_workload",fontsize=20)
ax6.set_ylabel("%",fontsize = 20)
ax6.tick_params(labelsize=20)


#  <font size='6' color='RED'>MACHINE LEARNING OBJECTIVE - MODEL TO PRODUCTION</font>

# # Machine Learning methods are gaining foothold in Indian companies. Most of respondents are going to put Machine Learning methods into production phase.

# In[ ]:


ax7 = (multi_choice.loc[multi_choice['Q3']=='India','Q8'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q8'].shape[0]).plot(kind='bar',figsize=(20,10))
ax7.set_xlabel("state_of_ML_in_company",fontsize=20)
ax7.set_ylabel("%",fontsize = 20)
ax7.tick_params(labelsize=20)


#  <font size='6' color='RED'>DATA SCIENCE AS DECISION MAKER - IN INFANCY</font>

# # Data Science is still in nascent stage as regards important business decision making science in India

# In[ ]:


ax8 = (multi_choice.loc[multi_choice['Q3']=='India','Q9_whole'].value_counts()[0:3]*100/multi_choice.loc[multi_choice['Q3']=='India','Q9_whole'].shape[0]).plot(kind='bar',figsize=(20,10))
ax8.set_xlabel("gender",fontsize=20)
ax8.set_ylabel("%",fontsize = 20)
ax8.tick_params(labelsize=20)


#  <font size='6' color='RED'>MOST PREVALENT ANNUAL SALARY - $0-999</font>

# In[ ]:


ax9 = (multi_choice.loc[multi_choice['Q3']=='India','Q10'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q10'].shape[0]).plot(kind='bar',figsize=(20,10))
ax9.set_xlabel("annual_salary",fontsize=20)
ax9.set_ylabel("%",fontsize = 20)
ax9.tick_params(labelsize=20)


#  <font size='6' color='RED'>ML OR CLOUD COMPUTING SPENDING - NULL</font>

# # Most companies in India have spent almost nothing on Machine Learning or Cloud Computing products which shows that AI penetration is low in India.

# In[ ]:


ax10 = (multi_choice.loc[multi_choice['Q3']=='India','Q11'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q11'].shape[0]).plot(kind='bar',figsize=(20,10))
ax10.set_xlabel("investment",fontsize=20)
ax10.set_ylabel("%",fontsize = 20)
ax10.tick_params(labelsize=20)


#  <font size='8' color='BLUE'>-----DATA SCIENCE PRACTICE----</font>

#  <font size='6' color='RED'>MOST PREVALENT DATA SCIENCE FORUM - KAGGLE</font>

# In[ ]:


ax11 = (multi_choice.loc[multi_choice['Q3']=='India','Q12_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q12_whole'].shape[0]).plot(kind='bar',figsize=(20,10))
ax11.set_xlabel("forum",fontsize=20)
ax11.set_ylabel("%",fontsize = 20)
ax11.tick_params(labelsize=20)


#  <font size='6' color='RED'>MOOCs PENETRATION - VERY LOW</font>

# # Most Indian Respondents are doing formal Undergrad or Master courses to get into Data Science. MOOCs learning penetration is very low in India.

# In[ ]:


ax12 = (multi_choice.loc[multi_choice['Q3']=='India','Q13_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q13_whole'].shape[0]).plot(kind='bar',figsize=(20,10))
ax12.set_xlabel("course_platform",fontsize=20)
ax12.set_ylabel("%",fontsize = 20)
ax12.tick_params(labelsize=20)


#  <font size='6' color='RED'>MOST PREVALENT ENVIRONMENTS - JUPYTERLAB/RSTUDIO</font>

# # Most Indian Data Science practitioners are using local development environments such as RStudio or Jupyterlab. Cloud based  platforms adoption is low in India. Data Science has not become very important tool for making business decisions yet.

# In[ ]:


ax13 = (multi_choice.loc[multi_choice['Q3']=='India','Q14'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q14'].shape[0]).plot(kind='bar',figsize=(20,10))
ax13.set_xlabel("development_environment",fontsize=20)
ax13.set_ylabel("%",fontsize = 20)
ax13.tick_params(labelsize=20)


#  <font size='6' color='RED'>MOST PREVALENT DATA SCIENCE EXPERIENCE - 1 TO 2 YEARS</font>

# # Most Data Science Practitioners in India have been in this field for only 1 to 2 years. Data Science is still young in India.

# In[ ]:


ax14 = (multi_choice.loc[multi_choice['Q3']=='India','Q15'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q15'].shape[0]).plot(kind='bar',figsize=(20,10))
ax14.set_xlabel("experience",fontsize=20)
ax14.set_ylabel("%",fontsize = 20)
ax14.tick_params(labelsize=20)


#  <font size='6' color='RED'>MOST PREVALENT IDE - JUPYTER</font>

# # India uses Jupyter environment for doing Data Science

# In[ ]:


ax15 = (multi_choice.loc[multi_choice['Q3']=='India','Q16_whole'].value_counts()[0:4]*100/multi_choice.loc[multi_choice['Q3']=='India','Q16_whole'].shape[0]).plot(kind='bar',figsize=(20,10))
ax15.set_xlabel("environment",fontsize=20)
ax15.set_ylabel("%",fontsize = 20)
ax15.tick_params(labelsize=20)


#  <font size='6' color='RED'>MOST PREVALENT NOTEBOOK PRODUCT - KAGGLE NOTEBOOK</font>

# # Kaggle Notebooks is the most used hosted notebook in India for Data Science.

# In[ ]:


ax16 = (multi_choice.loc[multi_choice['Q3']=='India','Q17_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q17_whole'].shape[0]).plot(kind='bar',figsize=(20,10))
ax16.set_xlabel("notebook_type",fontsize=20)
ax16.set_ylabel("%",fontsize = 20)
ax16.tick_params(labelsize=20)


#  <font size='6' color='RED'>MOST PREVALENT TOOL - PYTHON</font>

# # India does Data Science in Python. Most People are learning Python to become Data Scientist

# In[ ]:


ax17 = (multi_choice.loc[multi_choice['Q3']=='India','Q18_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q18_whole'].shape[0]).plot(kind='bar',figsize=(20,10))
ax17.set_xlabel("tools",fontsize=20)
ax17.set_ylabel("%",fontsize = 20)
ax17.tick_params(labelsize=20)


#  <font size='6' color='RED'>MOST PREVALENT COMPUTER LANGUAGE - PYTHON</font>

# In[ ]:


ax18 = (multi_choice.loc[multi_choice['Q3']=='India','Q19'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q19'].shape[0]).plot(kind='bar',figsize=(20,10))
ax18.set_xlabel("computer_language",fontsize=20)
ax18.set_ylabel("%",fontsize = 20)
ax18.tick_params(labelsize=20)


#  <font size='6' color='RED'>MOST PREVALENT VISUALISATION TOOL - MATPLOTLIB, SEABORN</font>

# # Matplotlib & Seaborn are the most used libraries for visualizations

# In[ ]:


ax19 = (multi_choice.loc[multi_choice['Q3']=='India','Q20_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q20_whole'].shape[0]).plot(kind='bar',figsize=(20,10))
ax19.set_xlabel("visualization_tools",fontsize=20)
ax19.set_ylabel("%",fontsize = 20)
ax19.tick_params(labelsize=20)


#  <font size='6' color='RED'>MOST PREVALENT PROCESSOR TYPES - CPUs, GPUs</font>

# # CPUs & GPUs are equally used for doing Data Science in India

# In[ ]:


ax20 = (multi_choice.loc[multi_choice['Q3']=='India','Q21_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q21_whole'].shape[0]).plot(kind='bar',figsize=(20,10))
ax20.set_xlabel("processing_unit_type",fontsize=20)
ax20.set_ylabel("%",fontsize = 20)
ax20.tick_params(labelsize=20)


#  <font size='6' color='RED'>TENSOR PROCESSING UNIT EXPOSURE - LOW</font>

# # Indian Data Science Practitioners have very low exposure to TPU use

# In[ ]:


ax21 = (multi_choice.loc[multi_choice['Q3']=='India','Q22'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q22'].shape[0]).plot(kind='bar',figsize=(20,10))
ax21.set_xlabel("TPU_use_frequency",fontsize=20)
ax21.set_ylabel("%",fontsize = 20)
ax21.tick_params(labelsize=20)


#  <font size='8' color='BLUE'>ML,CLOUD COMPUTING</font>

#  <font size='6' color='RED'>MACHINE LEARNING ADOPTION - LOW</font>

# # Machine Learning practice has just started in India. Machine Learning adoption has still long way to go in India

# In[ ]:


ax22 = (multi_choice.loc[multi_choice['Q3']=='India','Q23'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q23'].shape[0]).plot(kind='bar',figsize=(20,10))
ax22.set_xlabel("machine_learning_use",fontsize=20)
ax22.set_ylabel("%",fontsize = 20)
ax22.tick_params(labelsize=20)


#  <font size='6' color='RED'>MOST PREVALENT ALGORITHMS - LINEAR/LOGISTIC REGRESSION,DECISION TREE/RANDOM FOREST</font>

# # Most Indian Data Science Practitioners are still using only simple Machine Learning algorithms. Machine Learning still has a long way to go in India.

# In[ ]:


ax23 = (multi_choice.loc[multi_choice['Q3']=='India','Q24_whole'].value_counts()[0:3]*100/multi_choice.loc[multi_choice['Q3']=='India','Q24_whole'].shape[0]).plot(kind='bar',figsize=(20,10))
ax23.set_xlabel("machine_learning_algorithms",fontsize=20)
ax23.set_ylabel("%",fontsize = 20)
ax23.tick_params(labelsize=20)


#  <font size='6' color='RED'>AUTOMATIC MODEL SELECTION TOOL ADOPTION - LOW</font>

# # Automated model selection tools adoption is low in India.

# In[ ]:


ax24 = (multi_choice.loc[multi_choice['Q3']=='India','Q25_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q25_whole'].shape[0]).plot(kind='bar',figsize=(20,10))
ax24.set_xlabel("automated_tool_selection",fontsize=20)
ax24.set_ylabel("%",fontsize = 20)
ax24.tick_params(labelsize=20)


#  <font size='6' color='RED'>MOST PREVALENT IMAGE PROCESSING TOOL - NONE</font>

# # Image processing tools are not used by majority of Indian Data Scientists.

# In[ ]:


ax25 = (multi_choice.loc[multi_choice['Q3']=='India','Q26_whole'].value_counts()[0:2]*100/multi_choice.loc[multi_choice['Q3']=='India','Q26_whole'].shape[0]).plot(kind='bar',figsize=(20,10))
ax25.set_xlabel("image_processing_tools",fontsize=20)
ax25.set_ylabel("%",fontsize = 20)
ax25.tick_params(labelsize=20)


#  <font size='6' color='RED'>MOST PREVALENT NATURAL LANGUAGE PROCESSING TOOL - NONE</font>

# # NLP is not used much by Indian Data Scientists. Text Analytics has still not become mainstream in India.

# In[ ]:


ax26 = (multi_choice.loc[multi_choice['Q3']=='India','Q27_whole'].value_counts()[0:2]*100/multi_choice.loc[multi_choice['Q3']=='India','Q27_whole'].shape[0]).plot(kind='bar',figsize=(20,10))
ax26.set_xlabel("nlp_tools",fontsize=20)
ax26.set_ylabel("%",fontsize = 20)
ax26.tick_params(labelsize=20)


#  <font size='6' color='RED'>MACHINE LEARNING FRAMEWORK ADOPTED - SCIKIT_LEARN</font>

# # Scikit_learn is gaining adoption as top Machine Learning framework in India.

# In[ ]:


ax27 = (multi_choice.loc[multi_choice['Q3']=='India','Q28_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q28_whole'].shape[0]).plot(kind='bar',figsize=(20,10))
ax27.set_xlabel("machine_learning_frameworks",fontsize=20)
ax27.set_ylabel("%",fontsize = 20)
ax27.tick_params(labelsize=20)


#  <font size='6' color='RED'>MOST PREVALENT CLOUD COMPUTING PLATFORMS - AWS, GCP</font>

# # AWS & GCP are gaining adoption in cloud computing platform field in India.

# In[ ]:


ax28 = (multi_choice.loc[multi_choice['Q3']=='India','Q29_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q29_whole'].shape[0]).plot(kind='bar',figsize=(20,10))
ax28.set_xlabel("cloud_computing_platform",fontsize=20)
ax28.set_ylabel("%",fontsize = 20)
ax28.tick_params(labelsize=20)


#  <font size='6' color='RED'>TOP CLOUD COMPUTING PRODUCT - AWS EC2</font>

# # AWS EC2 is top cloud computing product used in India.

# In[ ]:


ax29 = (multi_choice.loc[multi_choice['Q3']=='India','Q30_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q30_whole'].shape[0]).plot(kind='bar',figsize=(20,10))
ax29.set_xlabel("cloud_computing_products",fontsize=20)
ax29.set_ylabel("%",fontsize = 20)
ax29.tick_params(labelsize=20)


#  <font size='6' color='RED'>TOP DATA ANALYSIS PRODUCT - GOOGLE BIGQUERY, DATABRICKS</font>

# # Indian Data Science Practitioners have started using Google BigQuery and DataBricks.

# In[ ]:


ax30 = (multi_choice.loc[multi_choice['Q3']=='India','Q31_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q31_whole'].shape[0]).plot(kind='bar',figsize=(20,10))
ax30.set_xlabel("data_analysis_products",fontsize=20)
ax30.set_ylabel("%",fontsize = 20)
ax30.tick_params(labelsize=20)


#  <font size='6' color='RED'>TOP MACHINE LEARNING PRODUCTS - AMAZON SAGEMAKER, AZURE ML STUDIO</font>

# # Amazon Sagemaker & Azure Machine Learning Studio are two top machine learning products used in India.

# In[ ]:


ax31 = (multi_choice.loc[multi_choice['Q3']=='India','Q32_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q32_whole'].shape[0]).plot(kind='bar',figsize=(20,10))
ax31.set_xlabel("machine_learning_products",fontsize=20)
ax31.set_ylabel("%",fontsize = 20)
ax31.tick_params(labelsize=20)


#  <font size='6' color='RED'>TOP AUTOMATED MACHINE LEARNING PLATFORMS - AUTO-SKLEARN, GOOGLE AUTOML</font>

# # Indians have started using Auto-Sklearn & Google AutoML.

# In[ ]:


ax32 = (multi_choice.loc[multi_choice['Q3']=='India','Q33_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q33_whole'].shape[0]).plot(kind='bar',figsize=(20,10))
ax32.set_xlabel("automated_platforms",fontsize=20)
ax32.set_ylabel("%",fontsize = 20)
ax32.tick_params(labelsize=20)


#  <font size='6' color='RED'>TOP RELATIONAL DATABASE PRODUCTS - MYSQL</font>

# # MySQL is the top relational DB product used in India.

# In[ ]:


ax33 = (multi_choice.loc[multi_choice['Q3']=='India','Q34_whole'].value_counts()[0:5]*100/multi_choice.loc[multi_choice['Q3']=='India','Q34_whole'].shape[0]).plot(kind='bar',figsize=(20,10))
ax33.set_xlabel("relational_database_products",fontsize=20)
ax33.set_ylabel("%",fontsize = 20)
ax33.tick_params(labelsize=20)


# In[ ]:




