#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Literature Review Data Load
# 
# These are the CSV files behind [the Kaggle community's AI-powered literature review](https://www.kaggle.com/covid-19-contributions). The tables have been curated by a large team of domain experts. They can be used as "target variables" for those trying to extract the relevant information from [CORD-19 research papers](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge).
# 
# This Notebook shows how to preprocess the CSVs to get useful target variables.

# In[ ]:


import pandas as pd
import os

pd.set_option('display.max_columns', None)


# In[ ]:


BASE_DIR = '../input/aipowered-literature-review-csvs/kaggle/working/'
dir_list = os.listdir(BASE_DIR)
dir_list


# # Data structure
# 
# **TIE** (tranmission, incubation and environmental stability) contains CSVs in a range of different formats.
# 
# **Risk Factors** contains CSVs in a consistent format which can be cleaned and merged.

# In[ ]:


for d in dir_list:
    print(d)
    file_list = os.listdir(os.path.join(BASE_DIR, d))
    for f in file_list:
        print(f'\t{f}')
        df = pd.read_csv(os.path.join(BASE_DIR, d, f), index_col=0)
        for c in df.columns:
            print(f'\t\t- {c}')
        print()


# # Load Risk Factors

# In[ ]:


risk_dir = os.path.join(BASE_DIR, 'Risk Factors')
df_list = []
for f in os.listdir(risk_dir):
    df = pd.read_csv(os.path.join(risk_dir, f), index_col=0)
    df['csv_source'] = f
    df_list.append(df)


# In[ ]:


df = pd.concat(df_list).reset_index(drop=True)
df['Date'] = pd.to_datetime(df.Date)


# In[ ]:


# Check what metrics have been used
df.Severe.str.split(' ', expand=True)[0].value_counts()


# In[ ]:


df.Fatality.str.split(' ', expand=True)[0].value_counts()


# In[ ]:


# Extract the various metrics used
for col in ['Severe', 'Fatality']:
    for metric in ['OR', 'AOR', 'HR', 'AHR', 'RR']:
        capture_string = metric + r'(?:\s|=)(\d+.\d+)'
        df[f'{col.lower()}_{metric.lower()}'] = df[col].str.extract(capture_string)


# In[ ]:


# Extract the upper and lower confidence intervals
lower_capture_string = r'95% CI: (\d+.\d+)'
upper_capture_string = r'95% CI: \d+.\d+\s?-\s?(\d+.\d+)'
for col in ['Severe', 'Fatality']:
    df[f'{col.lower()}_ci_lower'] = df[col].str.extract(lower_capture_string)
    df[f'{col.lower()}_ci_upper'] = df[col].str.extract(upper_capture_string)


# In[ ]:


# Extract the p values
p_value_capture_string = r'p=(0.\d+)'
for col in ['Severe', 'Fatality']:
    df[f'{col.lower()}_p_value'] = df[col].str.extract(p_value_capture_string)


# In[ ]:


df.head(20)


# In[ ]:


# Quick tidy to clarify the subject
df['risk_factor'] = df.csv_source.str.slice(0, -4)


# In[ ]:


df.to_csv('risk_factors_training_data.csv', index=False)

