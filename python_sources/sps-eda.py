#!/usr/bin/env python
# coding: utf-8

# # Introduction
# ![](https://image.slidesharecdn.com/proteinstructureprediction1-190207163342/95/protein-structure-levels-1-638.jpg?cb=1549788361)
# ## Aim:
# prediction of classification of molecules according to data values.
# 
# ## Context:
# 
# This is a protein data set retrieved from Research Collaboratory for Structural Bioinformatics (RCSB) Protein Data Bank (PDB).
# 
# The PDB archive is a repository of atomic coordinates and other information describing proteins and other important biological macromolecules. Structural biologists use methods such as X-ray crystallography, NMR spectroscopy, and cryo-electron microscopy to determine the location of each atom relative to each other in the molecule. They then deposit this information, which is then annotated and publicly released into the archive by the wwPDB.
# 
# ## Content:
# 0. [Load and Check Data](#0)
# 1. [Variable Description](#1)
# 1. [Data Visualization](#2)
#     * [Correlation](#3)
#     * [densityMatthews-densityPercentSol](#4)
#     * [MacromoleculeType](#5)
#     * [experimentalTechnique](#6)
#     * [Classification](#7)
#     * [phValue](#8)
#     * [publicationYear](#9)
# 1. [Outliers Detection](#10)
# 1. [Missing Value Imputation](#11)
#     * [Missing Value Visualization](#12)
#     * [Fill Missing Value](#13)
#         * [Mode Method](#14)
#         * [Mean and Std Method](#15)
#         * [KNN Method](#16)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualization
import matplotlib.pyplot as plt # visualization
import missingno as msno # visualizatin for missing values

import warnings
warnings.filterwarnings("ignore") # ignore warnings

from sklearn.model_selection import train_test_split # train and test split

from sklearn.impute import KNNImputer # filling missing data with KNN method

from sklearn.preprocessing import LabelEncoder # filling missing categorical values with label encoder method
import category_encoders as ce

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <a id="0"></a>
# # 0) Load Data

# In[ ]:


# load file 
df = pd.read_csv("/kaggle/input/protein-data-set/pdb_data_no_dups.csv")


# * first look at the dataset

# In[ ]:


df.head(3)


# <a id="1"></a>
# # 1) Variable Description
# 1. **structureId**: identity of the structure
# 1. **classification**: classification type
# 1. **experimentalTechnique**: technique of experiment
# 1. **macromoleculeType**: type of macromolecule
# 1. **residueCount**: number of residue
# 1. **resolution**: amount of resolution
# 1. **structureMolecularWeight**: molecular weight
# 1. **crystallizationMethod**: method of crystallization
# 1. **crystallizationTempK**: crystallization temperature in Kelvin 
# 1. **densityMatthews**: crystalline density 
# 1. **densityPercentSol**: resolution ratio by density
# 1. **pdbxDetails**: detail about row
# 1. **phValue**: PH value 
# 1. **publicationYear**: published year

# In[ ]:


df.info()


# * float64(7): resolution, structureMolecularWeight, crystallizationTempK, densityMatthews, densityPercentSol, phValue, publicationYear
# * int64(1): residueCount
# * object(6): structureId, classification, experimentalTechnique, macromoleculeType, crystallizationMethod, pdbxDetails
# * rows: 141401
# * columns: 14   

# In[ ]:


df.describe().T


# * when we look this describe table phValue must be 0-14 but there is max value 724.
# * publicationYear has a value of 201, which is unlikely.
# * we can examine that proteins are denatured at 50-60 celsius degrees or above 10 pH and below 10-15 celsius degrees or 4 pH.

# In[ ]:


df.publicationYear[df.publicationYear==201.0]=np.nan


# In[ ]:


df.phValue[df.phValue>14]=np.nan


# * removing unnecessary column.(pdbxDetails)

# In[ ]:


df.drop("pdbxDetails", axis=1, inplace = True)


# * Examination of the unique value categorical variables

# In[ ]:


df.classification.nunique()


# In[ ]:


df.experimentalTechnique.nunique()


# In[ ]:


df.structureId.nunique()


# In[ ]:


df.macromoleculeType.nunique()


# In[ ]:


df.crystallizationMethod.nunique()


# Some values in the structurId column have multiple values so some values have same structureId:
# 1. 1UJQ    4
# 1. 3NO0    4
# 1. 2FYM    4
# 1. 4KJ7    3
# 1. 3K6U    3
# 1. ...
# 
# We may need to review these values.

# In[ ]:


df.structureId.value_counts(ascending=False).head(10)


# In[ ]:


# Missing Value Table
def missing_value_table(df):
    missing_value = df.isna().sum().sort_values(ascending=False)
    missing_value_percent = 100 * df.isna().sum()//len(df)
    missing_value_table = pd.concat([missing_value, missing_value_percent], axis=1)
    missing_value_table_return = missing_value_table.rename(columns = {0 : 'Missing Values', 1 : '% Value'})
    cm = sns.light_palette("lightgreen", as_cmap=True)
    missing_value_table_return = missing_value_table_return.style.background_gradient(cmap=cm)
    return missing_value_table_return
  
missing_value_table(df)


# In missing value table three most missing values 
# 
# 1. crystallizationMethod	45159
# 1. crystallizationTempK	44362
# 1. phValue	36291

# <a id="2"></a>
# # 2) Data Visualization

# In[ ]:


sns.pairplot(df)
sns.set(style="ticks", color_codes=True)


# * pairplot to compare each column

# <a id="3"></a>
# ## Correlation

# In[ ]:


corr = df.corr()
plt.figure(figsize=(12,5))
sns.heatmap(corr, annot=True)


# * Looking at the correlation between numerical values. Light colors are values with higher correlations
# * there is a good correlation between densityMatthews and densityPercentSol.(0.84)
# * another correlation that can be taken into consideration is structureMolecularWeight-residueCount(0.55)

# <a id="4"></a>
# ## densityMatthews-densityPercentSol

# In[ ]:


sns.lmplot(x = "densityPercentSol", y = "densityMatthews", line_kws={'color': 'red'}, data = df);
plt.title("densityMatthews-densityPercentSol",color = 'darkblue',fontsize=15)
plt.show()


# * The link between density and density dissolution rate

# In[ ]:


density_data = df[['densityMatthews','densityPercentSol']] 
sns.pairplot(density_data);
plt.title("densityMatthews-densityPercentSol",color = 'darkblue',fontsize=15)
plt.show()


# <a id="5"></a>
# ## MacromoleculeType

# In[ ]:


plt.figure(figsize=(20,18))
ex = df.macromoleculeType.value_counts(ascending=False)[:5]
figureObject, axesObject = plt.subplots() 
explode = (0.2, 0.5, 0.5, 0.5, 0.5)
plt.title("Macro Molecule Type",color = 'darkblue',fontsize=15)

axesObject.pie(ex.values,
               labels   = ex.index,
               shadow   = True,                       
               explode  = explode,
               autopct  = '%.1f%%',
               wedgeprops = { 'linewidth' : 3,'edgecolor' : "orange" })                              
             
axesObject.axis('equal') 

plt.show() 


# * As seen from the chart, the most common type of molecule is protein.

# <a id="6"></a>
# ## experimentalTechnique

# In[ ]:


experimentalTechnique=df["experimentalTechnique"].value_counts(ascending=False)[:5]
plt.figure(figsize=[10,5])
plt.plot(experimentalTechnique, color="#588da8", linestyle="--", linewidth=3, label = "experimentalTechnique")
plt.title("Experimental Technique-Frequency",color = 'darkblue',fontsize=15)
plt.xlabel("Experimental Technique")
plt.ylabel("Frequency")
plt.show()


# * Looking at the data, it is clear that x-ray diffraction is the most used technique.

# <a id="7"></a>
# ## Classification

# In[ ]:


classification = df.classification.value_counts()[:10]
plt.figure(figsize=(12,5))
sns.barplot(x=classification.index, y=classification.values, palette="dark")
plt.xticks(rotation='vertical')
plt.ylabel('Number of Classification')
plt.xlabel('Classification Types')
plt.title('Top 10 Classification',color = 'darkblue',fontsize=15);


# * Looking at the classification, the sum of the top 3 data clearly provides the majority.

# <a id="8"></a>
# ## phValue

# In[ ]:


def ph(ph):
    if ph < 7 :
        ph = 'Acidic'
    elif ph > 7:
        ph = 'Base'
    else:
        ph = 'Neutral'
    return ph


# In[ ]:


df_ph = df.dropna(subset=["phValue"])
df_ph['pH'] = df_ph['phValue'].apply(ph)
labels = df_ph['pH'].value_counts().index
values = df_ph['pH'].value_counts().values


# In[ ]:


fig1, ax1 = plt.subplots()
ax1.pie(values, labels=labels, autopct='%1.1f%%')
ax1.axis('equal')
plt.title("Acid-Base-Neutral Balance",color = 'darkblue',fontsize=15)
plt.show()


# * Acidic: %47.5
# * Base: %41.0
# * Neutral: %11.5

# <a id="9"></a>
# ## publicationYear

# In[ ]:


plt.figure(figsize=(12,5))
sns.scatterplot(x=df.publicationYear.value_counts().sort_index().index, y=df.publicationYear.value_counts().sort_index().values)
plt.xticks(rotation='vertical')
plt.ylabel('Frequency')
plt.xlabel('Years')
plt.title('Publication Distribution by Years',color = 'darkblue',fontsize=15);


# * 
# It is seen that it has started a great rise after 1992.

# <a id="10"></a>
# # 3) Outliers Detection

# In statistics, an outlier is a data point that differs significantly from other observations.
# 
# * Outlier is smaller than Q1-1.5(Q3-Q1) and higher than Q3+1.5(Q3-Q1) .
# 
#     * (Q3-Q1) = IQR (INTER QUARTILE RANGE)
#     * Q3 = Third Quartile(%75)
#     * Q1 = First Quartile(%25)

# * train and test separation

# In[ ]:


y = df[["classification"]] # target
x = df.drop("classification", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42, test_size=0.2)


# * keeping the necessary columns and removing unnecessary

# In[ ]:


columns = x_train.select_dtypes(["int","float64","int64"])

del columns["phValue"] # delete unneccessary value
del columns["publicationYear"] # delete unneccessary value


# * outlier detection for x_train values

# In[ ]:


lower_and_upper = {} # storage
x_train_copy = x_train.copy() # train copy 

for col in columns.columns: # outlier detect
    q1 = x_train[col].describe()[4] # Q1 = Quartile 1 median 25 
    q3 = x_train[col].describe()[6] # Q3 = Quartile 3 median 75 
    iqr = q3-q1  #IQR Q3 -Q1
    
    lower_bound = q1-(1.5*iqr)
    upper_bound = q3+(1.5*iqr)
    
    lower_and_upper[col] = (lower_bound, upper_bound)
    x_train_copy.loc[(x_train_copy.loc[:,col]<lower_bound),col]=lower_bound*0.75
    x_train_copy.loc[(x_train_copy.loc[:,col]>upper_bound),col]=upper_bound*1.25
    
lower_and_upper


# * outlier detection for x_test values

# In[ ]:


x_test_copy = x_test.copy() # train copy   

for col in columns.columns:
    x_test_copy.loc[(x_test_copy.loc[:,col]<lower_and_upper[col][0]),col]=lower_and_upper[col][0]*0.75
    x_test_copy.loc[(x_test_copy.loc[:,col]>lower_and_upper[col][1]),col]=lower_and_upper[col][1]*1.25


# <a id="11"></a>
# # 4) Missing Value

# <a id="12"></a>
# # Missing Value Visualization

# In[ ]:


msno.bar(df, figsize=(15,8), sort='descending');


# In[ ]:


msno.matrix(df)
plt.title("Missing Value",color = 'darkblue',fontsize=15)
plt.show()


# * Looking at the 8 values, it is seen that the white lines are missing value rows.
# * We can think of a link about the crystallizationMethod and crystallizationTempK because the white lines are almost same.
# * 5 lines are empty at the same time and 13 lines are filled at the same time.

# In[ ]:


msno.heatmap(df)
plt.title("Missing Value Correlation HeatMap",color = 'darkblue',fontsize=15)
plt.show()


# * crystallizationMethod missing value depends on crystallizationTempK missing value, but crystallizationTempK missing value is not depends on crystallizationMethod missing value.
# * there are the highest correlation between densityMatthews and densityPercentSol. If one variable changes, there will be another

# <a id="13"></a>
# # Fill Missing Value

# <a id="14"></a>
# ## a) Mode Method

# * Filling train macromoleculeType with mode.

# In[ ]:


x_train_copy['macromoleculeType'].fillna(x_train_copy['macromoleculeType'].mode()[0], inplace=True) # fill missing data with mode


# * Filling test macromoleculeType with mode.

# In[ ]:


x_test_copy['macromoleculeType'].fillna(x_test_copy['macromoleculeType'].mode()[0], inplace=True) # fill missing data with mode


# * Filling train resolution with the number of rows as random as the mean subtracted from standard deviation.

# <a id="15"></a>
# ## b) Mean and Std Method

# In[ ]:


x_train_resol_std, x_train_resol_mean = x_train_copy.resolution.std(), x_train_copy.resolution.mean() # mean and standard deviation
random = np.random.uniform(x_train_resol_std, x_train_resol_mean, 113120) # 113120 numbers of rows 
x_train.resolution = x_train.resolution.mask(x_train.resolution.isnull(), random)


# * Filling test resolution with the number of rows as random as the mean subtracted from standard deviation.

# In[ ]:


x_test_resol_std, x_test_resol_mean = x_test_copy.resolution.std(), x_test_copy.resolution.mean() # mean and standard deviation
random = np.random.uniform(x_test_resol_std, x_test_resol_mean, 28281) # 28281 numbers of rows 
x_test.resolution = x_test.resolution.mask(x_test.resolution.isnull(), random)


# <a id="16"></a>
# ## c) KNN Method

# ## x_train KNN
# * KNN imputation for each columns without for the columns in the upper rows(macromoleculeType-resolution)

# In[ ]:


imputer = KNNImputer(n_neighbors=5)
imputer.fit_transform([x_train.crystallizationTempK])


# In[ ]:


imputer = KNNImputer(n_neighbors=5)
imputer.fit_transform([x_train.densityMatthews])


# In[ ]:


imputer = KNNImputer(n_neighbors=5)
imputer.fit_transform([x_train.densityPercentSol])


# In[ ]:


imputer = KNNImputer(n_neighbors=5)
imputer.fit_transform([x_train.phValue])


# In[ ]:


imputer = KNNImputer(n_neighbors=5)
imputer.fit_transform([x_train.resolution])


# In[ ]:


imputer = KNNImputer(n_neighbors=5)
imputer.fit_transform([x_train.publicationYear])


# ## x_test KNN
# * KNN imputation for each columns without for the columns in the upper rows(macromoleculeType-resolution)

# In[ ]:


imputer = KNNImputer(n_neighbors=5)
imputer.fit_transform([x_test.crystallizationTempK])


# In[ ]:


imputer = KNNImputer(n_neighbors=5)
imputer.fit_transform([x_test.densityMatthews])


# In[ ]:


imputer = KNNImputer(n_neighbors=5)
imputer.fit_transform([x_test.densityPercentSol])


# In[ ]:


imputer = KNNImputer(n_neighbors=5)
imputer.fit_transform([x_test.phValue])


# In[ ]:


imputer = KNNImputer(n_neighbors=5)
imputer.fit_transform([x_test.resolution])


# In[ ]:


imputer = KNNImputer(n_neighbors=5)
imputer.fit_transform([x_test.publicationYear])

