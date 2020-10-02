#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Human body is a complex and fascinating machine. According to a [2016 research](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002533), there are 30 trillion cells in a 70kg man and and along with these cells, we are inhabited with almost the same number of bacterial cells. Other than bacteria we are also colonised by fungi, protists and viruses. These microbes govern a range of functions in health, disease and our behaviour. So, how about taking a journey and analyse this amazing world of human microbiota through a lens of data science?
# In this kernel, we will explore a data set that is available on Amazon Web Services (AWS). The data has been collected from 300 individuals and various genetic analysis have been done to analyse the microbes that inhabit various regions of the human body.

# ### Objective 
# Here, we will try to answer some interesting questions, such as -
# i) Human body site showing most microbial diversity 
# ii) The most common genus (taxonomic rank that comes above species) of microorganism found in the human body
# iii) Most ubiquitous microbe in the human body
# 

# ### Import the necessary modules 
# Import the necessary modules and read the data. 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

micro=pd.read_csv('../input/project_catalog.csv')
pd.set_option('display.max_rows', None, 'display.max_columns', None)


# We then check how the data looks like by checking its the shape, its columns and the amount of missing data.

# In[ ]:


micro.shape


# In[ ]:


micro.columns 


# In[ ]:


micro.head(2)


# In[ ]:


micro.info()


# We then check the statistics for Gene count (it has no missing data). From this, it is evident that minimum gene count is 0. How?

# In[ ]:


micro['Gene Count'].describe()


# Digging further it is evident that 1331 Bacteria have 0 gene count followed by 8 eukaryotes and 6 viruses. 

# In[ ]:


micro_gene_count=micro[micro['Gene Count']==0]
micro_gene_count['NCBI Superkingdom'].value_counts()


# I was curious to check if the reason of 0 gene count is correlated with the project status. It appears that the reason behid 0 gene count  for 1284bacteria, 4 eukaryotes, 1virus could be that the pojects are still in progress (when they released the data).

# In[ ]:


micro_no_gene_progress= micro[(micro['Gene Count']==0) & (micro['Project Status']=='In Progress')]
micro_no_gene_progress['NCBI Superkingdom'].value_counts()


# But for 47 bacteria, 5 viruses, 4 eukaryotes the project has been completed. Thus, the reason could be a reporting error.

# In[ ]:


micro_no_gene_complete=micro[(micro['Gene Count']==0) & (micro['Project Status']=='Complete')]
micro_no_gene_complete['NCBI Superkingdom'].value_counts()


# Next, I found out that the microorganism with the highest gene count is *Streptomyces* sp. HGB0020, with 8490 genes and which lives in the gastrointestinal tract.

# In[ ]:


micro[micro['Gene Count']==8490]


# Next, I found that the research found 16 human body sites, with most diversity shown in the gastrointestinal tract.

# In[ ]:


micro['HMP Isolation Body Site'].nunique()


# In[ ]:


micro['HMP Isolation Body Site'].value_counts()


# In[ ]:


micro['HMP Isolation Body Site'].value_counts().plot(kind='bar')
plt.title('Distribution of microorganisms in various body sites')
plt.ylabel('Number of different microbes')
plt.xlabel('Human body sites')
plt.title('Diversity of microorganisms at different body sites')


# Next, I was interested in finding the most common genus. A scientific name is made up of 2 parts - genus name followed by scientific name.

# In[ ]:


micro['Genus']= micro['Organism Name'].str.split(' ').str[0]
micro['species']=micro['Organism Name'].str.split(' ').str[1]
micro[['Genus','species']].head()


# There are 242 genera found, with *Streptococcus* being the most common genus in the human body.

# In[ ]:


micro['Genus'].nunique()


# In[ ]:


micro['Genus'].value_counts().head(10)


# Before proceeding further, I checked for the unique values in 'NCBI Superkingdom' column. There are 3 observations labeled 'Error!!!' for this column.

# In[ ]:


micro.groupby('NCBI Superkingdom').count()


# As Streptococcus species ( which are bacteria) are the ones with 'Error!!!' in NCBI superkingdom, I have replaced this 'Error!!!' with 'Bacteria'.

# In[ ]:


micro[micro['NCBI Superkingdom']=='Error!!!']


# In[ ]:


micro['NCBI Superkingdom'].replace('Error!!!', 'Bacteria', inplace=True)


# To proceed further, I wanted to fill the missing values in Domain and NCBI superkingdom columns. One can infer the result of one column if the other column's value is known. 

# In[ ]:


micro[['Domain','NCBI Superkingdom']].isnull().sum()


# But we cannot fill the missing values in either of these columns, if both values of the 2 columns are missing. For that I checked how many such observations are present. Then I removed those observations.

# In[ ]:


len(micro.loc[micro['Domain'].isnull()& micro['NCBI Superkingdom'].isnull()])


# In[ ]:


micro=micro.drop(micro[(micro['Domain'].isnull()) & (micro['NCBI Superkingdom'].isnull())].index)
micro.shape


# **Bug** - As many of the same rows have missing values for NCBI Superkingdom and Domain columns. The following groupby followed by transform steps were not working and giving an error message. To overcome that I first replaced missing value with the string 'NaN' in NCBI Superkingdom and then proceeded to the next step of filling missing value in Domain column ( groupby 'NCBI Superkingdom' and then transform it).

# In[ ]:


micro['NCBI Superkingdom'].fillna('NaN', inplace=True)


# In[ ]:


print(micro.shape)
micro['Domain'] =micro.groupby('NCBI Superkingdom')['Domain'].transform(lambda x: x.fillna(x.mode().max()))
micro['Domain'].isnull().sum()


# Next, I replaced the string 'NaN' using groupby and transform methods.

# In[ ]:


micro['NCBI Superkingdom']= micro.groupby('Domain')['NCBI Superkingdom'].transform(lambda x: x.replace('NaN', x.mode().max()))
micro.loc[micro['NCBI Superkingdom']=='NaN']


# After getting rid of the missing values in 'NCBI Superkingdom' and 'Domain' columns, I was keen in checking where all the different types of microbes such as Bacteria, eukaryotes, viruses and archeae are located in the human body. From the given probing, it is clear that bacteria are located in all 16 studied human body sites followed by eukaryotes in 5 body sites, followed by viruses and archaea. 

# In[ ]:


micro.groupby('NCBI Superkingdom')['HMP Isolation Body Site'].nunique().sort_values(ascending=False)


# As bacteria are more ubiquitous in the human body. The diversity is most vast in gastrointentestinal tract.

# In[ ]:


bac=micro.loc[micro['Domain']=='BACTERIAL']
bac['HMP Isolation Body Site'].unique()


# In[ ]:


bac['HMP Isolation Body Site'].value_counts(ascending=False).plot(kind='bar')
plt.ylabel('Number of different bacteria')
plt.xlabel('Human body sites')
plt.title('Diversity of bacteria at different body sites')


# More Eukaryotic diversity exist in the blood, followed by skin, airways, , wound, unknown.  

# In[ ]:


euk=micro.loc[micro['Domain']=='EUKARYAL']
euk['HMP Isolation Body Site'].unique()


# In[ ]:


euk['HMP Isolation Body Site'].value_counts(ascending=False).plot(kind='bar')
plt.ylabel('Number of different eukaryotes')
plt.xlabel('Human body sites')
plt.title('Diversity of eukaryotes at different body sites')


# The study didn't find any precise location for viruses. Although, some previous studies have found virueses in blood, skin.

# In[ ]:


vir=micro.loc[micro['Domain']=='VIRUS']
vir['HMP Isolation Body Site'].unique()


# Archaea is found primarily in the gastrointestinal tract.

# In[ ]:


arc=micro.loc[micro['Domain']=='ARCHAEAL']
arc['HMP Isolation Body Site'].unique()


# Next question that comes to mind is which is the most ubiquitous organism found in this analysis.  Staphylococcus is the most ubiquitous among all, with 11 habitats- urogenital_tract, skin,  airways, unknown, gastrointestinal_tract, nose, blood, bone, eye, ear, other.

# In[ ]:


z=micro.groupby('Genus')['HMP Isolation Body Site'].nunique().sort_values(ascending=False)
y=pd.DataFrame(z)
w=y[y['HMP Isolation Body Site']>4]
print(w)
w.plot(kind='bar')
plt.ylabel('Number of different body sites')
plt.title('Number of habitats for different microorganisms')


# In[ ]:


staph=micro.loc[micro['Genus']=='Staphylococcus']
staph['HMP Isolation Body Site'].unique()


# Just to mention that this study, includes 2882 Bacteria, 8 eukaryotes, 6 viruses and 2 archaea.

# In[ ]:


micro['NCBI Superkingdom'].value_counts()


# Following is the list of names of viruses, eukaryotes and archaea.

# In[ ]:


viruses= micro[micro['NCBI Superkingdom'] =='Viruses']
viruses['Organism Name']


# In[ ]:


eukaryotes= micro[micro['NCBI Superkingdom']=='Eukaryota']
eukaryotes['Organism Name']


# In[ ]:


archaea= micro[micro['NCBI Superkingdom']=='Archaea']
archaea['Organism Name']


# ### Conclusion
# In this kernel we have performed EDA of the Human Microbiome dataset. Found- i) Gastrointestine shows most diversity of microbes, ii) *Streptomyces* sp. HGB0020 shows the maximum gene count in human, iii) *Streptococcus* is most common genus while *Staphylococcus* is most ubiquitous in humans.
# 
