#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import re
import glob
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # Import data
# Check the competition page for the details of the data: https://www.kaggle.com/c/data-science-for-good-city-of-los-angeles

# In[ ]:


# Read all txt files into a panda dataframe
def importData():
   
    path='../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/*.txt'
    files=glob.glob(path)
    jobs_list=[]
    file_names=[]
    for file in files:
        with open(file,'r',errors='replace') as f:
            jobs_list.append(f.read())
        match = re.search('Bulletins/(.*\.txt)',file)
        file_names.append(match.group(1))
    jobs_df = pd.DataFrame({"File_Name":file_names,"job_info":jobs_list})
    
    return jobs_df


# # Extract data using regex

# In[ ]:


# Define functions for pandas dataframe apply

# Get job title (JOB_CLASS_TITLE)
def get_job_title(text):
    title = text.split('\n',1)[0]
    return title.strip()

# Get class code (JOB_CLASS_NO)
def get_class_code(text):
    match = re.search(r'Class\s*Code:\s*(\d+)',text)
    if match:
        return match.group(1)
    else:
        return None

    
# Get open date (OPEN_DATE)
def get_open_date(text):
    match = re.search(r'Open\s*Date:\s*(\d+-\d+-\d+)',text)
    if match:
        return match.group(1)
    else:
        return None

# Get exam type
def get_exam_type(text):
    match = re.search(r'(?s)Open\s*Date(.*?)ANNUAL\s*SALARY',text)
    if match:
        result = match.group(1)
        op=re.search(r'open',result,flags=re.IGNORECASE)
        open_int_prom=re.search(r'(open.*comp.*?tive)|(comp.*?tive.*open)',result,flags=re.IGNORECASE)
        int_dept_prom=re.search(r'inter.*?mental',result,flags=re.IGNORECASE)
        dept_prom=re.search(r'dep.*?mental',result,flags=re.IGNORECASE)
        if int_dept_prom:
            return 'INT_DEPT_PROM'
        elif dept_prom:
            return 'DEPT_PROM'
        elif open_int_prom:
            return 'OPEN_INT_PROM'
        elif op:
            return 'OPEN'
        else:
            return None
    else:
        return None
    
# Get general salary (ENTRY_SALARY_GEN)
def get_salary_gen(text):
    match = re.search(r'(?s)ANNUAL\s*SALARY(.*?)DUTIES.*',text)
    if match:
        result = match.group(1)
        sal=re.search(r'(\d+,\d+.*?to.*?\d+,\d+|\d+,\d+).*?Water.*Power',text,flags=re.IGNORECASE)                if get_salary_dwp(text) else re.search(r'(\d+,\d+.*?to.*?\d+,\d+|\d+,\d+)',text,flags=re.IGNORECASE)
        if sal:
            return sal.group(1).replace('to','-').replace('$','')
        else:
            return None
    else:
        return None


# Get DWP salary (ENTRY_SALARY_DWP)
def get_salary_dwp(text):
    match = re.search(r'(?s)ANNUAL\s*SALARY(.*?)DUTIES.*',text)
    if match:
        result = match.group(1)
        sal=re.search(r'Water.*Power.*?(\d+,\d+.*?to.*?\d+,\d+|\d+,\d+)',text)
        if sal:
            return sal.group(1).replace('$','').replace('to','-')
        else:
            return None 
    else:
        return None
    
# Get driver license req (DRIVERS_LICENSE_REQ)
def get_dl_req(text):
    match= re.search(r"(.*?)driver\'s license",text)    
    if match:
        result = match.group(1).lower().split()
        if 'may' in result:
            return 'P'
        else:
            return 'R'
    else:
        return None
    
# Get driver license type (DRIV_LIC_TYPE)
def get_dl_type(text):
    dl_types=[]
    match= re.search(r"(?s)(valid California Class|valid Class|valid California Commercial Class)(.*?)(California driver\'s license|driver\'s license)",text)
    if match:
        dl=match.group(2)
        if 'A' in dl:
            dl_types.append('A')
        if 'B' in dl:
            dl_types.append('B') 
        if 'C' in dl:
            dl_types.append('C')  
        if 'I' in dl:
            dl_types.append('I')   
        return ','.join(dl_types)
    else:
        return None

# Get duties (JOB_DUTIES)
def get_duties(text):
    match= re.search(r"(?s)DUTIES(.*?)(REQ.*?MENT|MINI.*?REQ)", text)
    if match:
        return match.group(1).strip()
    else:
        return None

# Get requirements section
def get_req_section(text):  
    match= re.search(r"(?s)(QUAL.*?TIONS*|REQ.*?MENTS*).*?\n(.*?)(PROCESS NOTES|NOTES|WHERE TO APPLY|HOW TO APPLY)", text)
    if match:
        return match.group(2).strip()
    else:
        return None

# Split requirements to rows
def split_req(df):
    req_list = df['Req_section'].apply(lambda x: re.split(r'or\s*\n+(?=\d\.)', x))
    df = pd.DataFrame({col:np.repeat(df[col].values, req_list.str.len())                   for col in df.columns})
    df['req_list'] = np.concatenate(req_list.values)

    return df
    
    
# Extract all by regex
def extract_df(df):    
    # JOB_CLASS_TITLE
    df['JOB_CLASS_TITLE'] = df['job_info'].apply(lambda x: get_job_title(x))
    # JOB_CLASS_NO
    df['JOB_CLASS_NO'] = df['job_info'].apply(lambda x: get_class_code(x))
    # OPEN_DATE
    df['OPEN_DATE'] = df['job_info'].apply(lambda x: get_open_date(x))
    # EXAM_TYPE
    df['EXAM_TYPE'] = df['job_info'].apply(lambda x: get_exam_type(x))
    # ENTRY_SALARY_GEN
    df['ENTRY_SALARY_GEN'] = df['job_info'].apply(lambda x: get_salary_gen(x))
    # ENTRY_SALARY_DWP
    df['ENTRY_SALARY_DWP'] = df['job_info'].apply(lambda x: get_salary_dwp(x))
    # JOB_DUTIES
    df['JOB_DUTIES'] = df['job_info'].apply(lambda x: get_duties(x))
    # DRIVERS_LICENSE_REQ
    df['DRIVERS_LICENSE_REQ'] = df['job_info'].apply(lambda x: get_dl_req(x))
    # DRIV_LIC_TYPE
    df['DRIV_LIC_TYPE'] = df['job_info'].apply(lambda x: get_dl_type(x))
    # Requirements section
    df['Req_section'] = df['job_info'].apply(lambda x: get_req_section(x))
    # Split reqs
    df = split_req(df)
    # Add req_set_id
    df['REQUIREMENT_SET_ID'] = df.groupby('JOB_CLASS_NO').cumcount() + 1
    
    return df


# In[ ]:


# Perform the extraction
df = importData()
df = extract_df(df)
df.head(10)


# # Extract data using NER in spaCy

# ## Generate spacy training data

# In[ ]:


import logging
import json
# Convert json to spacy training data structure
def json_to_spacy(filePath):
    try:
        training_data = []
        lines=[]
        with open(filePath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                #only a single point in text annotation.
                point = annotation['points'][0]
                labels = annotation['label']
                # handle both list of labels or a single label.
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    #dataturks indices are both inclusive [start, end] but spacy is not [start, end)
                    entities.append((point['start'], point['end'] + 1 ,label))


            training_data.append((text, {"entities" : entities}))

        return training_data
    except Exception as e:
        logging.exception("Unable to process " + filePath + "\n" + "error = " + str(e))
        return None
    
# Create spacy training data
train_data = json_to_spacy('../input/ner-annotation-of-city-of-la-jobs/city_la_jobs_ner_labeling.json')


# ## Train NER model

# In[ ]:


import random
import spacy

# Model training function
def ner_model(train_data, n_iter=50):
    
    # Create Ner model and add labels
    nlp = spacy.blank("en")
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner, last=True)
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    
    # Begin training
    nlp.begin_training()
    
    for itn in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        batches = spacy.util.minibatch(train_data, size=spacy.util.compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts,annotations,drop=0.5,losses=losses)
        if itn%10==0:
            print(losses)
    return nlp

nlp = ner_model(train_data)
nlp.to_disk('./')


# In[ ]:


# Test the model with predictions
for text, _ in train_data:
    doc = nlp(text)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


# ## Extract using NER model

# In[ ]:


# Extract data by ner model
def ner_extract(nlp,text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Extract requirements 
df['temp_req'] = df['req_list'].apply(lambda x: ner_extract(nlp,x))


# In[ ]:


df.head(5)


# # Create CSV

# ##  Flatten the temp_req column

# In[ ]:


# Extract one column from list
def extract_col(l,col):
    result = [x[0] for x in l if x[1]==col]
    return None if result==[] else result[0]

# Extract colums from temp_req column
def extract_col_df(df,col):
    df[col]=df['temp_req'].apply(lambda x: extract_col(x,col))
    return df


# In[ ]:


# Extract coulums in df
cols_to_extract = ['EDUCATION_YEARS', 'SCHOOL_TYPE' ,'EDUCATION_MAJOR', 'EXPERIENCE_LENGTH' ,                   'FULL_TIME_PART_TIME','EXP_JOB_CLASS_TITLE', 'EXP_JOB_CLASS_ALT_RESP',                   'EXP_JOB_CLASS_FUNCTION', 'EXP_JOB_CLASS_ADDITIONAL_FUNCTION','COURSE_COUNT',                   'COURSE_LENGTH', 'COURSE_SUBJECT', 'MISC_COURSE_DETAILS', 'EXP_JOB_COMPANY',                   'DEGREE NAME', 'EXP_JOB_CLASS_ALT_JOB_TITLE', 'REQUIRED_CERTIFICATE',                   'CERTIFICATE_ISSUED_BY','COURSE_TITLE', 'REQUIRED_EXAM_PASS', 'EXPERIENCE_EXTRA_DETAILS']

for col in cols_to_extract:
    df = extract_col_df(df,col)


# In[ ]:


col_orders = ['File_Name','JOB_CLASS_TITLE', 'JOB_CLASS_NO', 
                 'REQUIREMENT_SET_ID',
                 'JOB_DUTIES','EDUCATION_YEARS', 'SCHOOL_TYPE',
                 'EDUCATION_MAJOR', 'DEGREE NAME','EXPERIENCE_LENGTH',
                 'FULL_TIME_PART_TIME',
                 'EXP_JOB_CLASS_TITLE', 'EXP_JOB_CLASS_FUNCTION',
                 'EXP_JOB_COMPANY','EXP_JOB_CLASS_ALT_JOB_TITLE',
                 'EXP_JOB_CLASS_ALT_RESP', 'COURSE_COUNT',
                 'COURSE_LENGTH', 'COURSE_SUBJECT',
                 'REQUIRED_CERTIFICATE','CERTIFICATE_ISSUED_BY',
                 'DRIVERS_LICENSE_REQ', 'DRIV_LIC_TYPE',
                 'EXAM_TYPE','ENTRY_SALARY_GEN','ENTRY_SALARY_DWP','OPEN_DATE']

df[col_orders].to_csv('./jobs.csv', index=None)


# In[ ]:




