#!/usr/bin/env python
# coding: utf-8

# **<h2><ins>Naive Lookup With Document Save</ins></h2><br>**
# 
# To begin answering the questions in task 2 I wanted to get an understanding of how many documents covered the required topics. To quickly gain that understanding I constructed a naive document look up, simply using (relatively) standard libraries to find documents based on a keyword. 
# 
# Once found, the index of these documents within the corona dataframe are saved to a list. The progam then simply navigates to the relevant directory and copies the file to a newly created folder (names with date and time). 
# 
# I plan to use this simple lookup table to assign naive labels to the data (`smoking`, `neonates`, `high_risk`, `transmission_dynamics` and `co_infection`). Once labelled I will then use unsupervised learning (K-Means clustering) to cluster the data (using a set of vector features engineered using the doc2vec model) and compare the contents of each cluster with my manually assigned labels. Thsi clustered data will be further used to engineer features for a Convolutional Neural Net.  
# 
# Using this approach I aim to provide a `proof of concept` for using vectorised text to assign clustered features, with a view to using a CNN to classify unseen documents as containing information surrounding the outlined tasks. 
# 
# -------------------------------------------
# 

# **<ins>Import Libraries</ins><br>**
# 
# As mentioned above, nothing facy here, just good old pandas, numpy and the standard library (with a dash of glob for good measure).
# 
# --------------------------

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import time
import shutil
import os
import stat, sys 


# **<ins>Custom Functions</ins><br>**
# 
# Below I created a series of custom functions for finding documents, creating a folder and copying the json files. 
# 
# ----------------------------

# In[ ]:


def return_doc_index(subject, w2v_model):
    
    documents = np.where(corona_data['text_body'].str.contains(subject) ,corona_data['text_body'].index, 0)
    index = np.where(documents != 0)
    return index



def print_document_title(list_of_index):
    
    for index in list_of_index:
        for i in index:
            print('-----------\n')
            report = corona_data.iloc[i]
            print(report['title'])
            print(f"Paper Index {i}")
            print('-------------\n')
    
    



def print_document_body(list_of_index):
    
    for index in list_of_index:
        for i in index:
            print('-----------\n')
            report = corona_data.iloc[i]
            print(report['text_body'])
            print(f"Paper Index {i}")
            print('-------------\n')



def create_document_directory():
    
    current_directory = os.getcwd()
    current_time = time.strftime("%Y-%m-%d @ %H:%M:%S")
    final_directory = os.path.join(current_directory, r'Saved Documents: ' + current_time)
    if not os.path.exists(final_directory):
       os.makedirs(final_directory)
    folder_name = f"Saved Documents: {current_time}"
    
    return folder_name


def save_documents(list_of_index, corona_data, new_image_folder):
    
    dir_ = "../input/CORD-19-research-challenge/2020-03-13"
    output_dir = "../output/kaggle/working"
    
    for i in list_of_index:
        for ind in i:
            paper = corona_data.iloc[ind]
            filename = paper['doc_id']
            source = paper['source']
            ext = ".json"
            
            source_folder = f"/{source}/{source}/"
            output_dir = f"/{new_image_folder}"
            
            save_doc = shutil.copy2(os.path.join(dir_ + source_folder, filename + ext),
                                    os.path.join(output_dir, filename + ext))
            


# **<ins>Get The Data</ins><br>**
# 
# Can't do anything without it! This data is loaded from the .csv file created in a previous kernel. 
# 
# ---------------------------------

# In[ ]:


corona_data = pd.read_csv("../input/kaggle-covid19/kaggle_covid-19_open_csv_format.csv")
corona_data = corona_data.drop(columns=['abstract'])
corona_data = corona_data.fillna("Unknown")


# **<ins>A Place To Hold All Those Delightful Documents</ins><br>**
# 
# What better place than a python dictionary. 
# 
# Using the `return_doc_index` function I can query the corpus with keywords and return the document index in a list. 
# 
# ----------------------------

# In[ ]:


doc_folder = {"risk": return_doc_index("risk", corona_data),

              "preg": return_doc_index("pregnant", corona_data),

               "smoking": return_doc_index("smoking", corona_data),

               "co_infection": return_doc_index("co infection", corona_data),

                "neonates": return_doc_index("neonates", corona_data),

               "transmission": return_doc_index("transmission dynamics", corona_data),

                "high_risk": return_doc_index("high-risk patient", corona_data)
             }


# **<ins>Take A Peek Inside The Folder</ins><br>**
# 
# For sanity let's check that it actually found some documents. 
# 
# --------------------------

# In[ ]:


print(f"Number of Documents that Mention Risk: {len(doc_folder['risk'][0])}")

print(f"Number of Documents that Mention Pregnancy: {len(doc_folder['preg'][0])}")

print(f"Number of Documents that Mention Smoking: {len(doc_folder['smoking'][0])}")

print(f"Number of Documents that Mention Neonates: {len(doc_folder['neonates'][0])}")

print(f"Number of Documents that Mention Transmission Dynamics: {len(doc_folder['transmission'][0])}")

print(f"Number of Documents that Mention High Risk Patients: {len(doc_folder['high_risk'][0])}")


# **<ins>Save Files And We're Done!</ins><br>**
# 
# With the index of any relevant documents saved I use the `create_document_folder` and `save_documents` function to...well....create a folder and save the documents. 
# 
# ---------------------------------------
# 
# <i>Please note that I have hashed out the final line of code as it accesses the original dataset of json files, please ensure you are running this script from a root directory that contains the project subfolders.</i>
# 
# ------------------------------------------
# 
# <i>Also note that if running in a local environment you will need to change the `directory path` inside the `save_documents` function.</i>
# 

# In[ ]:


new_image_folder = create_document_directory()

# save_docs = save_documents(doc_folder['high_risk'], corona_data, new_image_folder)


# 
# 
# 
# -----------------------------------------
