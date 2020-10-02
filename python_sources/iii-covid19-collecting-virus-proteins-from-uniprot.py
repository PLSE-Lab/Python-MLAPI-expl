#!/usr/bin/env python
# coding: utf-8

# ## Collecting Corona Virus Proteins from Uniprot Database
# 
# **What is Uniprot Database?**
# 
# UniProt is a freely accessible database of protein sequence and functional information, many entries being derived from genome sequencing projects. It contains a large amount of information about the biological function of proteins derived from the research literature.[source: Wikipedia](https://en.wikipedia.org/wiki/UniProt)
# 
# **Why is this protein list important to current COVID-19 Dataset?**
# 
# Inorder to gather the information about the biomolecular mechanism from the scientific literature (COVID-19 Dataset), one need to have the list of associated Proteins, Genes, Pathways, Drugs etc. This notebook presents the steps to gather Corona Virus associated proteins, Gene names and associated Pathways from Uniprot database. These lits could be useful to look at the textual documents for further NLP processing and to present the entity relationship.

# ### 1. Getting Data

# #### Step -I
# Gp to Uniprot Database (https://www.uniprot.org/) and select UniprotKB in search bar. Then inter corona virus into the search bar.
# 

# ![](https://github.com/Vasuji/COVID19/blob/master/img/uniprot-search.png?raw=true)

# #### Step -II:
# After you hit search operation, you will get a table like disply of the result. It is multi page table. 
# 
# ![](https://github.com/Vasuji/COVID19/blob/master/img/uniprot-table.png?raw=true)

# #### Step-III:
# Look at the right most task bar of this table. You can see pen like icon through which you get next window. You can make a selection of the information you want to gather (e.g., Name, Gene, Pathways).
# 
# ![](https://github.com/Vasuji/COVID19/blob/master/img/pen.png?raw=true)

# #### Step - IV
# 
# Once you are done with selection of information, you can go back to previous table and hit download button. You can select the format of the data. Excel file download is one option.
# 
# ![](https://github.com/Vasuji/COVID19/blob/master/img/uniprot-download.png?raw=true)
# 

# ### 2. Data Wrangling

# #### What After getting Protein Data?
# 
# Lets play around with this data

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
file_path = "../input/corona-virus-proteins-from-uniprot-database/corona.csv"
df = pd.read_csv(file_path)


# In[ ]:


df.head(5)


# **There are total 21,876 proteins from different sources**

# In[ ]:


df.shape


# #### Q: What are the different Organisms? Can you find the top 50 Organisms?

# In[ ]:


df_organism = pd.DataFrame(df.groupby("Organism").count()['Entry'])
df_organism = df_organism.sort_values(by = "Entry", ascending = False)
df_organism[0:20].plot.barh(figsize = [15,10], fontsize =20)
plt.gca().invert_yaxis()


# In[ ]:


df_organism[0:20]


# #### Q: What are the different Viral hosts? Can you find top Virus hosts?

# In[ ]:


df['Virus hosts'] = df['Virus hosts'].apply(lambda x: str(x)[0:50] )
df_host = pd.DataFrame(df.groupby("Virus hosts").count()['Entry'])
df_host = df_host.sort_values(by = "Entry", ascending = False)
df_host[1:20].plot.barh(figsize = [15,10], fontsize =20)
plt.gca().invert_yaxis()


# In[ ]:


df_host[1:20]


# ### 3. Cleaning Protein Names, Synonyms and abbreviations

# In[ ]:


def filter(line):
    proteins = set()
    line = str(line)
    line = line.lower()
    
    '''for lines without () or [] terms'''
    if "(" not in line or "[" not in line:
        proteins.add(line.strip().replace(' ', '_'))
        
        
    '''for line including () terms'''    
    if '(' in line:
        start = 0
        open_in = line.find('(')
        tmp = line[start:open_in].strip().replace(' ', '_')
        proteins.add(tmp)
        while open_in >=0:
            start = open_in+1
            end = line.find(')', start)
            proteins.add(line[start:end].strip().replace(' ', '_'))
            open_in = line.find('(', end)
     
    '''for lines including [] trems'''
    if '[' in line:
        raw = line[line.find('['):line.find(']')]
        #print("THIS IS RAW:", raw[15:-1])
        raw = raw[15:-1]
        lraw = raw.split("; ")
        for item in lraw:
            #print(item)
            if '(' in item:
                start = 0
                open_in = item.find('(')
                tmp = item[start:open_in].strip().replace(' ', '_')
                proteins.add(tmp)
            else:
                proteins.add(item.strip().replace(' ', '_'))
    return proteins


# In[ ]:


allProteins = []
i = 0
for u,p in zip(df['Entry'],df['Protein names']):
    print(u,"|",p)
    print("------------")
    print(u,"|",filter(p))
    print("===================================================")
    i += 1
    if i>4:
        break


# In[ ]:


allProteins = []
for u,p in zip(df['Entry'],df['Protein names']):
    allProteins.append({"id":u, "names":list(filter(p))})


# In[ ]:


allProteins[0:5]


# In[ ]:


import json
with open("virus-proteins.json", 'w') as fn:
    json.dump(allProteins,fn)


# ***Note: Bringing more detail about how to create data indexing and search of Proteins, Genes etc from COVID19 dataset in the next notebook***
