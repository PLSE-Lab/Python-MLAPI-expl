#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip install cmapPy')
from cmapPy.pandasGEXpress.parse import parse


# # Introduction
# This notebook is a part of Assignment provided by Elucidata for 6 months internship in data science.
# 
# The motive of this notebook is to explore and analyse the **cancers patient** based on their **genes structure**.
# 
# ### Cancers
# 1. Cancer arises from the transformation of normal cells into tumour cells in a multistage process that generally progresses from a pre-cancerous lesion to a malignant tumour. 
# 
# 1. Cancer is a leading cause of death worldwide, accounting for an estimated 9.6 million deaths in 2018. The most common cancers are:
#  * Lung (2.09 million cases)
#  * Breast (2.09 million cases)
#  * Colorectal (1.80 million cases)
#  * Prostate (1.28 million cases)
#  * Skin cancer (non-melanoma) (1.04 million cases)
#  * Stomach (1.03 million cases)
# 1. **Pancreatic Adenocarcinoma (PAAD)** is the third most common cause of death from cancer, with an overall 5-year survival rate of less than 5%, and is predicted to become the second leading cause of cancer mortality in the United States by 2030.
# 1. Cancer mortality can be reduced if cases are detected and treated early. In this assignment we are going to analyse the previous data of **genes structure** to detect the cancer patient and provide early treatment.
# 1. **Genes Structures**: **Ribonucleic acid (RNA)** is a polymeric molecule essential in various biological roles in coding, decoding, regulation and expression of genes. RNA and DNA are nucleic acids, and, along with lipids, proteins and carbohydrates, constitute the four major macromolecules essential for all known forms of life. **RNA-Seq (RNA sequencing)**, is a sequencing technique to detect the quantity of RNA in a biological sample at a given moment.
# 

# # Data Exploration
# 
# * Here we have a dataset of normalized RNA Sequencing reads for pancreatic cancer tumors. The measurement consists of ~20,000 genes for 183 pancreatic cancer tumors. The **file format is GCT** , a tab-delimited file used for sharing gene expression data and metadata (details for each sample) for samples.
# * **GCT file:** The GCT file format, a tab-delimited text-based format pairing matrix expression values with row and column metadata, allowing comparison of both transcriptional and contextual differences across samples. A schematic of a sample GCT file is pictured below.
# ![gct data format](https://github.com/cmap/cmapPy/raw/f3fdf016095bb08d9402ec9b6d3ebf6e603d20a1/tutorials/GCT_mockup.png)
# Let's dive into data, and explore it.

# ### Load Dataset

# In[ ]:


data = parse('/kaggle/input/data-analyst-intern/data_analyst_intern/PAAD.gct')
print(type(data))


# Here we parse our dataset into GCToo pandas dataframe instance which contains 3 component dataframes (row_metadata_df, column_metadata_df, and data_df). Lets see individual dataframe and explore it. 

# In[ ]:


# Read all portion of data
col_meta_data = data.col_metadata_df
row_meta_data = data.row_metadata_df
my_data = data.data_df


# In[ ]:


# Columns meta data
col_meta_data.head()


# In[ ]:


# Shape of columns metadata
col_meta_data.shape


# In[ ]:


# Number of missing values in each column of col_meta_data
missing_val_count_by_column = (col_meta_data.isnull().sum())
missing_val_count_by_col = missing_val_count_by_column[missing_val_count_by_column > 0].sort_values(ascending=False)
print(f"Total {len(missing_val_count_by_col)} columns have missing value, and",
      f"Total {col_meta_data.isnull().sum().sum()} missing values in dataset")
plt.figure(figsize=(10, 7))
sns.countplot(data=missing_val_count_by_col, y= missing_val_count_by_col )


# In[ ]:


# Explore each columns of col_meatadata
for col in col_meta_data.columns:
    print(f"{col}:   {len(col_meta_data[col].unique())} unique values:")


# In[ ]:


# drop columns which have constant value
for col in col_meta_data.columns:
    if len(col_meta_data[col].unique())<2:
        col_meta_data.drop(columns=col, inplace=True)


# In[ ]:


col_meta_data.shape


# Columns metadata explain about behaviour of patient. Lets see the description of some variables of this data.
# * **sample_type** is our **target variable** which tells that which patient have cancer tumor or not.
# * **participant_id** unique value for each patient.
# * **mRNAseq_cluster**  estimates of the levels of individual isoforms within the cell of RNA sequence.
# * **bcr_patient_barcode** and **bcr_patient_uuid** representing the metadata of the participants and their samples.
# * **vital_status** Current status of patient(live/death). (may pose data leakage)
# * **adenocarcinoma_invasion** Tells about cancer grows and spreads to near cell or not.
# * **maximum_tumor_dimension** Size of Tumor, tumor size is strongly related to chances for survival.
# * **pathologic_stage** Stage of cancer (amount or spread of cancer in the body)
# * **pathologic_m** The M refers to whether the cancer has metastasized. This means that the cancer has spread from the primary tumor to other parts of the body
# * **pathologic_n** The N refers to the the number of nearby lymph nodes that have cancer.
# * **pathologic_t** The T refers to the size and extent of the main tumor. The main tumor is usually called the primary tumor.
# 
# Let's explore some variables:

# In[ ]:


col_meta_data.adenocarcinoma_invasion.value_counts()


# In[ ]:


conts_mRNA = col_meta_data.mRNAseq_cluster.value_counts()
plt.figure(figsize=(10,4))
sns.barplot(x=conts_mRNA.index,y=conts_mRNA.values,)
plt.ylabel('Number of Patient')
plt.xlabel('RNA sequence cluster Types')
plt.title('levels of individual isoforms')
plt.legend()
for i, index in enumerate(conts_mRNA.index):
    val = conts_mRNA.values[i]
    val_pcn = round(((val)/sum(conts_mRNA.values))*100, 2)
    print(f"RNA sequence cluster:{index} types patient is {val_pcn}% of total ")


# In[ ]:


mRNA_seq_group = col_meta_data.groupby(['mRNAseq_cluster', 'vital_status'])['vital_status'].count().reset_index(name='counts')
plt.figure(figsize=(10, 5))
sns.barplot(x='mRNAseq_cluster', y='counts', hue='vital_status', data=mRNA_seq_group)
plt.ylabel('Number of Patient')
plt.xlabel('RNA sequence cluster Types')
plt.title('Vital status of patient by mRNA sequence cluster types ')


# In[ ]:


for cluster_type in mRNA_seq_group.mRNAseq_cluster.unique():
    data = mRNA_seq_group[mRNA_seq_group.mRNAseq_cluster==cluster_type]
    live_pcn = round((data[data['vital_status']=='alive']['counts']
                      /sum(data.counts.values))*100, 2)
    dead_pcn = round((data[data['vital_status']=='dead']['counts']/
                     sum(data.counts.values))*100, 2)
    print(f"mRNAseq_cluster type {cluster_type} patient dead {dead_pcn.values[0]}% times")
    print(f"mRNAseq_cluster type {cluster_type} patient alive {live_pcn.values[0]}% times")
    print("\n")
    
    


# Clearly 65% of time patient are dead if mRNAseq_cluster type is 1 and 3.
# mRNAseq_cluster 1, 3, 4 are risky types then 2, 5 

# In[ ]:


col_meta_data.sample_type.value_counts()


# In[ ]:


sample_type_group = col_meta_data.groupby(['sample_type', 'vital_status'])['vital_status'].count().reset_index(name='counts')
plt.figure(figsize=(8, 3))
sns.barplot(x='sample_type', y='counts', hue='vital_status', data=sample_type_group)
plt.ylabel('Number of Patient')
plt.xlabel('Types of cancer')
plt.title('Vital status vs cancer type ')


# Types of tumer not play role in current vital status of patient. There is 50-50 chance of patient are currently alive or dead

# In[ ]:


col_meta_data.adenocarcinoma_invasion.value_counts()


# Approx every patient cancer grows and spreads into body cells.

# In[ ]:


sns.catplot(x='vital_status', y='maximum_tumor_dimension', hue='sample_type', data=col_meta_data)


# In[ ]:


sns.boxplot(x='vital_status', y='maximum_tumor_dimension', data=col_meta_data)


# on average which has higher tumor dimension are most likely to dead.

# In[ ]:


col_meta_data.pathologic_stage.value_counts()


# In[ ]:


pathologic_stage_group = col_meta_data.groupby(['pathologic_stage', 'vital_status'])['vital_status'].count().reset_index(name='counts')
plt.figure(figsize=(10, 5))
sns.barplot(x='pathologic_stage', y='counts', hue='vital_status', data=pathologic_stage_group)


# **pathologic stages are as follow:**
# * **stage 0:** Abnormal cells are present but have not spread to nearby tissue.Data set have not this stage because approx all **adenocarcinoma_invasion ** value is True.
# * **stage i**, **stage ii**,**stage iii** Cancer is present. The higher the number, the **larger the cancer tumor** and the more it has spread into nearby tissues.
# * **stage iv** The cancer has spread to distant parts of the body.
# 

# In[ ]:



pathologic_m_group = col_meta_data.groupby(['pathologic_m', 'vital_status'])['vital_status'].count().reset_index(name='counts')
plt.figure(figsize=(10, 5))
sns.barplot(x='pathologic_m', y='counts', hue='vital_status', data=pathologic_m_group)


# In[ ]:



pathologic_n_group = col_meta_data.groupby(['pathologic_n', 'vital_status'])['vital_status'].count().reset_index(name='counts')
plt.figure(figsize=(10, 5))
sns.barplot(x='pathologic_n', y='counts', hue='vital_status', data=pathologic_n_group)


# In[ ]:


pathologic_t_group = col_meta_data.groupby(['pathologic_t', 'vital_status'])['vital_status'].count().reset_index(name='counts')
plt.figure(figsize=(10, 5))
sns.barplot(x='pathologic_t', y='counts', hue='vital_status', data=pathologic_t_group)


# * **The T refers** to the size and extent of the main tumor. The main tumor is usually called the primary tumor.
# * **The N refers** to the the number of nearby lymph nodes that have cancer.
# * **The M refers** to whether the cancer has metastasized. This means that the cancer has spread from the primary tumor to other parts of the body.
# * majority of patient have **pathologic_t3** and **pathologic_n1** types stages.

# In[ ]:


print(row_meta_data.index[:5])
print(row_meta_data.shape)
row_meta_data.head()


# this rid show the types of RNA sequence(18465 different types of RNA seq).

# In[ ]:


my_data.head()


# In[ ]:


# Number of missing values in each column of my data
missing_val_count_by_column = (my_data.isnull().sum())
missing_val_count_by_col = missing_val_count_by_column[missing_val_count_by_column > 0].sort_values(ascending=False)
print(f"Total {len(missing_val_count_by_col)} columns have missing value, and",
      f"Total {my_data.isnull().sum().sum()} missing values in dataset")
plt.figure(figsize=(10, 7))
sns.countplot(data=missing_val_count_by_col, y= missing_val_count_by_col )


# In[ ]:


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_my_data = pd.DataFrame(my_imputer.fit_transform(my_data))

# Fill in the lines below: imputation removed column names; put them back
imputed_my_data.columns = my_data.columns
imputed_my_data.index = my_data.index

imputed_my_data.head()


# In[ ]:


print(len(my_data.columns.unique()))
print(len(col_meta_data.index.unique()))
# Lets combine col_meta_data and my_data


# In[ ]:


col_to_merge = ['sample_type', 'mRNAseq_cluster','adenocarcinoma_invasion', 'vital_status', 'maximum_tumor_dimension', 'pathologic_stage', 'pathologic_m', 'pathologic_n', 'pathologic_t']
participant_id = col_meta_data.participant_id
target_variable = col_meta_data.sample_type
col_meta_data = col_meta_data.set_index('participant_id')
imputed_my_data = imputed_my_data.rename(columns=participant_id).T
#imputed_my_data = imputed_my_data.join(col_meta_data[col_to_merge])
imputed_my_data.head()


# In[ ]:


for col in col_to_merge:
    imputed_my_data[col] = col_meta_data[col]
imputed_my_data.head(2)


# In[ ]:


def cat_to_numeric(x):
    if x=='Solid Tissue Normal':
        return 0
    elif x=='Primary solid Tumor':
        return 1
    else:
        return 2
target_variable = target_variable.map(lambda x: cat_to_numeric(x))
target_variable.value_counts().plot(kind='bar')


# In[ ]:


def remove_nulls(df):

    rows = df.shape[0]
    columns = df.shape[1]
    null_cols = 0
    list_of_nulls_cols = []
    for col in list(df.columns):
        null_values_rows = df[col].isnull().sum()
        null_rows_pcn = round(((null_values_rows)/rows)*100, 2)
        col_type = df[col].dtype
        if null_values_rows > 0:
            print("The column {} has {} null values. It is {}% of total rows.".format(col, null_values_rows, null_rows_pcn))
            print("The column {} is of type {}.\n".format(col, col_type))
            null_cols += 1
            list_of_nulls_cols.append(col)
            df[[col]] = df[[col]].apply(lambda x: x.fillna(method='backfill'))
            print(f"The column {col} has removed {null_values_rows} null values")
    null_cols_pcn = round((null_cols/columns)*100, 2)
    print("The DataFrame has {} columns with null values. It is {}% of total columns.".format(null_cols, null_cols_pcn))
    return df

my_data = remove_nulls(imputed_my_data.select_dtypes(exclude='object'))


# In[ ]:


imputed_my_data.isnull().sum().sum()


# In[ ]:



from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(imputed_my_data.drop(columns=col_to_merge))

from sklearn.decomposition import PCA as sklearnPCA
n_components = 100
sklearn_pca = sklearnPCA(n_components=n_components)
Y_sklearn = sklearn_pca.fit_transform(X_std)


# ### What can be said about the variance of the PCA?

# In[ ]:


cum_sum = sklearn_pca.explained_variance_ratio_.cumsum()

explained_var = round(sklearn_pca.explained_variance_ratio_.sum()*100, 2)

cum_sum = cum_sum*100

fig, ax = plt.subplots(figsize=(8,8))
plt.bar(range(n_components), cum_sum, label='Cumulative _Sum_of_Explained _Varaince', color = 'b',alpha=0.5)
plt.title(f"Around {explained_var}% of variance is explained by the Fisrt {n_components} principle component ");


# ### Visualize the data whole data using PCA.

# In[ ]:


from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=3)
X_reduced  = sklearn_pca.fit_transform(X_std)
Y=target_variable
from mpl_toolkits.mplot3d import Axes3D
plt.clf()
fig = plt.figure(1, figsize=(10,6 ))
ax = Axes3D(fig, elev=-150, azim=110,)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],cmap=plt.cm.Paired,linewidths=10)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])


# In[ ]:


fig = plt.figure(1, figsize=(10,6))
sns.scatterplot(X_reduced[:, 0],  X_reduced[:, 1],hue=Y)
plt.title("This The 2D Transformation of above graph ")


# the neuroendocrine tumors clearly not separable from the adenocarcinoma tumors?
# # 2
# Lets Remove the neuroendocrine tumors from the dataset so that it contains only the adenocarcinoma tumor samples. The histology for the different tumor samples is contained in the my_data.

# In[ ]:


imputed_my_data = imputed_my_data[imputed_my_data['adenocarcinoma_invasion']=='yes']
imputed_my_data.shape


# **Interferons (IFNs)** are a group of signaling proteins made and released by host cells in response to the presence of several pathogens, such as viruses, bacteria, parasites, and also tumor cells. Type I interferons (IFNs) are a large subgroup of interferon proteins that help regulate the activity of the immune system. The genes responsible for type 1 Interferons is called Type 1 IFN signature and consists a set of 25 genes in homo sapiens.
# 
# Let's read these 25 IFN signature

# In[ ]:


ifn_sig = pd.read_csv('/kaggle/input/data-analyst-intern/data_analyst_intern/type1_IFN.txt', header=None)
ifn_sig.columns = ['member']
ifn_sig.head()


# In[ ]:


ifn_sig_data = imputed_my_data[ifn_sig['member'].to_list()]
ifn_sig_data.shape


# In[ ]:


X_std = StandardScaler().fit_transform(ifn_sig_data)
n_components = 2
sklearn_pca = sklearnPCA(n_components=n_components)
X_reduced  = sklearn_pca.fit_transform(X_std)
fig = plt.figure(1, figsize=(10,6))
sns.scatterplot(X_reduced[:, 0],  X_reduced[:, 1],hue=imputed_my_data['sample_type'])


# In[ ]:


get_ipython().system('pip install gsva')
from GSVA import gsva
from plotnine import *
from sklearn.manifold import TSNE


# In[ ]:


XV = TSNE(n_components=2).    fit_transform(ifn_sig_data.T)
df = pd.DataFrame(XV).rename(columns={0:'x',1:'y'})
(ggplot(df,aes(x='x',y='y'))
 + geom_point(alpha=0.2)
)


# In[ ]:




