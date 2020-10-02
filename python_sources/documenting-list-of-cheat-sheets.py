#!/usr/bin/env python
# coding: utf-8

# # Cheatsheet list
# 
# Given below are the list of documents with cheatsheet in their title.
# 
# The dataframe gives most of the location of the documents along with the file size in mb and the number of pages contained(if it is a pdf).
# 
# This can be useful to  directly get the cheatsheets in a glance.

# In[ ]:


get_ipython().system('pip install PyPDF2')
import numpy as np
import pandas as pd
import os
from PyPDF2 import PdfFileReader

def get_pages(file_path):
    try:
        if file_path[-4:] == '.pdf':
            pdf = PdfFileReader(open(file_path,'rb'),strict=False)
            return(int(pdf.getNumPages()))
    except:
        pass
    return np.nan

def add_files():
    file_paths = []
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            current_path = os.path.join(dirname, filename)
            file_paths.append(current_path)
    return sorted(file_paths)

def get_df():
    df = pd.DataFrame()
    df['file_paths'] = add_files()
    df['modified_paths'] = df['file_paths'].map(lambda x: x.replace('/kaggle/input/data-science-cheat-sheets/', ''))
    df['extensions']  = df['file_paths'].map(lambda x: x.split('.')[-1])
    df = df[df.extensions.map(lambda x: x not in ['md', 'ini', 'tex'])]
    df['folders'] =  df['modified_paths'].map(lambda x: x.split('/')[0] if len(x.split('/'))>1 else '' )
    df['MB size'] = df['file_paths'].map(lambda x: round(os.stat(x).st_size/1024**2,2))
    df['pages'] =df['file_paths'].apply(get_pages)
    return df

def show_cheat_sheet(df):
    df = df.copy()
    df['cheat'] = df['modified_paths'].map(lambda x: 'cheat' in x.lower().split('/')[-1])
    df['sheet'] = df['modified_paths'].map(lambda x: 'sheet' in x.lower().split('/')[-1])
    return df[df['cheat'] & df['sheet']].drop(columns=['cheat', 'sheet']).reset_index(drop=True)


# In[ ]:


original_file_list = get_df()
df= show_cheat_sheet(original_file_list)
df.drop(columns=['file_paths'])


# # Cheat Sheet Topics
# 
# Here are the list of topics which contain cheatsheets. 

# In[ ]:


list(df['folders'].unique())


# # Previewing cheatsheets
# Using wand package can enable to preview what a cheatsheet looks like. 
# It is useful to view pdfs and other documents.

# In[ ]:


from wand.image import Image as WImage
def display_cheatsheet(index, df=df):
    data = df[df.index == index].to_dict()
    path = list(data['file_paths'].values())[0].replace('/kaggle/input/', '../input/')
    return WImage(filename=path)


# In[ ]:


display_cheatsheet(11)


# # Preview all files
# 
# To preview all files list, one can use the below command. 
# 
# This is generalised for all documents, not just cheatsheets.

# In[ ]:


original_file_list.drop(columns=['file_paths']).head()


# In[ ]:


list(original_file_list['folders'].unique())


# Here is an NLP example where we can see the files present in NLP folder. 
# 
# Some of the extensions like tex and ini are excluded from the files displayed.

# In[ ]:


def get_files_in_folder(folder_name):
    return original_file_list[original_file_list['folders']==folder_name].drop('file_paths', axis=1)

get_files_in_folder('NLP')


# One can substitute the index below to display the file necessary.

# In[ ]:


display_cheatsheet(179, original_file_list)

