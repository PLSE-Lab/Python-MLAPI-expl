#!/usr/bin/env python
# coding: utf-8

# ## Covid-19 Analytics

# In[ ]:


# Import Libraries 
import numpy as np
import json 
import pandas as pd
import glob 
import os


# <p>
# The aim of this notebook is to bring together the provided dataset into a single dataframe. 
# The text are recovered from the relevant folders. The section title are included in the dataframe text column. 
# The conclusion and title were  assigned  separate columns in the new dataframe. The preprocessed data were augmented with the information contained in the metadata. 
# </p>
# <p>
# 
#     The final data output has been uploaded: so you don't need to repeat this process. Check the bottom of this notebook to have a feel of the merged data.
# Feel free to use.
# </p>

# ### Directory set up

# <p>
# I had two folders called 'computations' and 'dataset' in a directory named 'covid'. The dataset
# was downloaded and extracted in the folder 'dataset'. This notebook was placed in
# 'computations' folder. Below I try to set up the folder path to organize my work.
#     
# </p>

# In[ ]:


# set the base directory.
BASE_DIR = os.path.join( os.path.dirname(path), '' )
# data set folder 
DATASET_DIR  =  BASE_DIR + 'dataset/CORD-19-research-challenge'
# data directories list
DATA_PATH_LIST = [DATASET_DIR +str("/")+ i for i in os.listdir(DATASET_DIR) ]
# get the name of the important folder from the folder names
DATA_FOCUS = [i.split('/')[-1] for i in DATA_PATH_LIST ] 
# deletee the unimportant subset
del DATA_FOCUS[2:5], DATA_FOCUS[3]
DATA_FOCUS 


# <p>
# In the cell below, I try to extract the full path to the json files in the folders named above
# </p>

# In[ ]:


# collect data set
folder_contents = {}
for i in DATA_PATH_LIST: #  DATA_PATH_LIST contains all the folder  paths containing the dataset
    temp = i.split('/')[-1] # get the last elemnt of the path name
    if temp in  DATA_FOCUS: # DATA FOCUS has the names of the folders one is interested in
        filename = i+'/'+temp+'/pdf_json'  # json files reside in the subfolder pdf_json
        jsonfiles = glob.glob(os.path.join(filename, "*.json")) # collect all json file path names in the folder
        if temp != 'biorxiv_medrxiv': # exclude biorxiv_medrxiv ; it has no pmc_json subfolder
            filename2 = i+'/'+temp+'/pmc_json' # some dataset are in the folder pmc_json
            jsonfiles2 = glob.glob(os.path.join(filename2, "*.json"))# collect json file path in the folder
            folder_contents[temp+'_pmc']  = jsonfiles # save in dictionary
        folder_contents[temp]  = jsonfiles   # save the file path names in dictionary
    


# <p>
# In the cell below, the keys to the dictionary are listed. The values of the keys correspond to the full path names of all
# json files contained in that folder</p>

# In[ ]:


# keys in the dictionary
folder_contents.keys()


# In[ ]:


fulldata=[] # list for storing dataset

for i in  folder_contents.keys(): # for each key(=folder name) in the dictionary folder_contents
    print(i) # print the source folder name /key
   
    #if i == 'biorxiv_medrxiv':
    for j in folder_contents[i]: # for each json file path in the value (folder name)
        paper_id = ''; # to store paper id
        temp = ''; # string to store text
        with open(j) as json_data: # open json file. j==json file path
            data = json.load(json_data)  # load the data
            paper_id = data['paper_id']  # get paper id
            temp+= ' ' + data['metadata']['title']  # get the title
            temp += ' :'
            conclusions = ''
            for text in data['body_text']:  # the 'element' body_text contains the text information of interest
                if text['section'] !=  'CONCLUSIONS': # check that  'element' section in the text is not conclusion
                    temp += 'section: ' + text['section'] + ':' # get the section name
                    temp += text['text'] # get the correspnsing section text
                elif text['section'] ==  'CONCLUSIONS': # if the text is a conclusion
                    conclusions +=  text['text'] # add to the connclusion string
                else:
                    temp += text['text'] # else add text
                
            datadict = {'paper_id': paper_id  , 'text':temp , 'conclusions':conclusions,  'source':i} # collect data
            #title_doc[paper_id] = data['metadata']['title']  
        
            fulldata.append(datadict) # add to list
            
    


# <p>
# The 'fulldata' wich is a list of dictionaries would be made into a dataframe in the cell below.
# </p>

# In[ ]:


# make data into data frame
covid = pd.DataFrame(fulldata)
covid.head()


# In[ ]:


# describe the columns with non empty conclusions
# 1189 articles with conclusions
covid.loc[covid['conclusions'] != '' ]['conclusions'].describe()


# In[ ]:


# save the dataset
covid.to_parquet('cleaned_covid.parquet')


# ### Meta data load

# <p>
# In this section the metadata is loaded. Metadata would later be combined  with the covid dataframe above.
# </p>

# In[ ]:


# load metadata
meta_data = pd.read_csv(DATASET_DIR+'/metadata.csv')


# In[ ]:


# check metadata
meta_data.head()[1:2]


# In[ ]:


# check number of free Nans in sha
len(meta_data) - meta_data['sha'].count()


# In[ ]:


# change the sha id to paper id
# useful for joining the dataset
meta_data.rename(columns={'sha':'paper_id'}, inplace=True)


# ### Merge Data

# <p>
# Here the metadata os merged with thw covid data. This becomes the full data frame that will be stored.
# </p>

# In[ ]:


# merge covid data with meta data
merged_data = covid.merge(meta_data, on='paper_id', how='inner', suffixes=('_1', '_2'))


# In[ ]:


# save as feather file
merged_data.to_parquet('merged_data.parquet.gzip',compression='gzip')


# **Final Merged Data **

# In[ ]:


ls


# In[ ]:


#Load data
merged_data = pd.read_parquet("merged_data.parquet.gzip")


# In[ ]:


merged_data.head()


# In[ ]:


# shape of the final data 
merged_data.shape


# In[ ]:


# columns in the data 
merged_data.columns


# In[ ]:


# describe the datatset
merged_data.info()


# In[ ]:


# articles with Non emoty  conclusions
merged_data[merged_data['conclusions']!='']['conclusions']


# In[ ]:


# articles with null title
merged_data[merged_data['title'].isnull()]['title']


# In[ ]:


merged_data['abstract'][7]


# ### Conclusion
# 
# <p>
# Enjoy and stay safe .
# 
# </p>

# In[ ]:




