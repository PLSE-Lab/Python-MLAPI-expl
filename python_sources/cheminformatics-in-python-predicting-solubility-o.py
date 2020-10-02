#!/usr/bin/env python
# coding: utf-8

# # **Cheminformatics in Python: Predicting Solubility of Molecules | End-to-End Data Science Project** 
# 
# 
# In this Jupyter notebook, we will dive into the world of Cheminformatics which lies at the interface of Informatics and Chemistry. We will be reproducing a research article (by John S. Delaney$^1$) by applying Linear Regression to predict the solubility of molecules (i.e. solubility of drugs is an important physicochemical property in Drug discovery, design and development).
# 
# This idea for this notebook was inspired by the excellent blog post by Pat Walters$^2$ where he reproduced the linear regression model with similar degree of performance as that of Delaney. This example is also briefly described in the book ***Deep Learning for the Life Sciences: Applying Deep Learning to Genomics, Microscopy, Drug Discovery, and More***.$^3$

# ## **1. Install rdkit**

# In[ ]:


get_ipython().system(' wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh')
get_ipython().system(' chmod +x Miniconda3-py37_4.8.2-Linux-x86_64.sh')
get_ipython().system(' bash ./Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -f -p /usr/local')
get_ipython().system(' conda install -c rdkit rdkit -y')
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')


# ## **2. Delaney's solubility dataset**
# 
# The original [Delaney's dataset](https://pubs.acs.org/doi/10.1021/ci034243x) available as a [Supplementary file](https://pubs.acs.org/doi/10.1021/ci034243x)$^4$. The full paper is entitled [ESOL:? Estimating Aqueous Solubility Directly from Molecular Structure](https://pubs.acs.org/doi/10.1021/ci034243x).$^1$

# ### **2.1. Download the dataset**

# In[ ]:


get_ipython().system(' wget https://pubs.acs.org/doi/suppl/10.1021/ci034243x/suppl_file/ci034243xsi20040112_053635.txt')


# In[ ]:


get_ipython().system(' wget https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv')


# ### **2.2. Read in the dataset**

# In[ ]:


import pandas as pd


# In[ ]:


sol = pd.read_csv('delaney.csv')
sol


# ### **2.3. Examining the SMILES data**

# Chemical structures are encoded by a string of text known as **SMILES** which is an acronym for **Simplified Molecular-Input Line-Entry System**.

# In[ ]:


sol.SMILES


# The first element from the **SMILES** column of the **sol** dataframe.

# In[ ]:


sol.SMILES[0]


# ### **2.4. Convert a molecule from the SMILES string to an rdkit object**

# In[ ]:


from rdkit import Chem


# In[ ]:


Chem.MolFromSmiles(sol.SMILES[0])


# In[ ]:


Chem.MolFromSmiles('ClCC(Cl)(Cl)Cl')


# ### **2.5. Working with rdkit object**

# In[ ]:


m = Chem.MolFromSmiles('ClCC(Cl)(Cl)Cl')


# In[ ]:


m.GetNumAtoms()


# ## **3. Calculate molecular descriptors in rdkit**

# ### **3.1. Convert list of molecules to rdkit object**

# In[ ]:


from rdkit import Chem


# #### **3.1.1. Method 1**

# In[ ]:


mol_list= []
for element in sol.SMILES:
  mol = Chem.MolFromSmiles(element)
  mol_list.append(mol)


# In[ ]:


len(mol_list)


# In[ ]:


mol_list[:5]


# #### **3.1.2. Method 2**

# In[ ]:


mol_list2 = [Chem.MolFromSmiles(element) for element in sol.SMILES]


# In[ ]:


len(mol_list2)


# In[ ]:


mol_list2[:5]


# ### **3.2. Calculate molecular descriptors**
# 
# To predict **LogS** (log of the aqueous solubility), the study by Delaney makes use of 4 molecular descriptors:
# 1. **cLogP** *(Octanol-water partition coefficient)*
# 2. **MW** *(Molecular weight)*
# 3. **RB** *(Number of rotatable bonds)*
# 4. **AP** *(Aromatic proportion = number of aromatic atoms / total number of heavy atoms)*
# 
# Unfortunately, rdkit readily computes the first 3. As for the AP descriptor, we will calculate this by manually computing the ratio of the *number of aromatic atoms* to the *total number of heavy atoms* which rdkit can compute.

# #### **3.2.1. LogP, MW and RB**

# In[ ]:


import numpy as np
from rdkit.Chem import Descriptors


# In[ ]:


# Inspired by: https://codeocean.com/explore/capsules?query=tag:data-curation

def generate(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem) 
        moldata.append(mol)
       
    baseData= np.arange(1,1)
    i=0  
    for mol in moldata:        
       
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
           
        row = np.array([desc_MolLogP,
                        desc_MolWt,
                        desc_NumRotatableBonds])   
    
        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1      
    
    columnNames=["MolLogP","MolWt","NumRotatableBonds"]   
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)
    
    return descriptors


# In[ ]:


df = generate(sol.SMILES)
df


# #### **3.2.2. Aromatic proportion**

# ##### 3.2.1.1. Number of aromatic atoms
# 
# Here, we will create a custom function to calculate the **Number of aromatic atoms**. With this descriptor we can use it to subsequently calculate the AP descriptor.

# Computing for a single molecule.

# In[ ]:


m = Chem.MolFromSmiles('COc1cccc2cc(C(=O)NCCCCN3CCN(c4cccc5nccnc54)CC3)oc21')


# In[ ]:


aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
aromatic_atoms


# In[ ]:


def AromaticAtoms(m):
  aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
  aa_count = []
  for i in aromatic_atoms:
    if i==True:
      aa_count.append(1)
  sum_aa_count = sum(aa_count)
  return sum_aa_count


# In[ ]:


AromaticAtoms(m)


# Computing for molecules in the entire dataset.

# In[ ]:


desc_AromaticAtoms = [AromaticAtoms(element) for element in mol_list]
desc_AromaticAtoms


# ##### 3.2.1.2. **Number of heavy atoms**
# 
# Here, we will use an existing function for calculating the Number of heavy atoms.

# Computing for a single molecule.

# In[ ]:


m = Chem.MolFromSmiles('COc1cccc2cc(C(=O)NCCCCN3CCN(c4cccc5nccnc54)CC3)oc21')
Descriptors.HeavyAtomCount(m)


# Computing for molecules in the entire dataset.

# In[ ]:


desc_HeavyAtomCount = [Descriptors.HeavyAtomCount(element) for element in mol_list]
desc_HeavyAtomCount


# ##### **3.2.1.3. Computing the Aromatic Proportion (AP) descriptor**

# Computing for a single molecule.

# In[ ]:


m = Chem.MolFromSmiles('COc1cccc2cc(C(=O)NCCCCN3CCN(c4cccc5nccnc54)CC3)oc21')
AromaticAtoms(m)/Descriptors.HeavyAtomCount(m)


# Computing for molecules in the entire dataset.

# In[ ]:


desc_AromaticProportion = [AromaticAtoms(element)/Descriptors.HeavyAtomCount(element) for element in mol_list]
desc_AromaticProportion


# In[ ]:


df_desc_AromaticProportion = pd.DataFrame(desc_AromaticProportion, columns=['AromaticProportion'])
df_desc_AromaticProportion


# ### **3.3. X matrix (Combining all computed descriptors into 1 dataframe)**

# In[ ]:


df


# In[ ]:


df_desc_AromaticProportion


# Let's combine the 2 dataframes to produce the **X** matrix

# In[ ]:


X = pd.concat([df,df_desc_AromaticProportion], axis=1)
X


# ### **3.4. Y matrix**

# In[ ]:


sol.head()


# Assigning the second column (index 1) to the Y matrix

# In[ ]:


Y = sol.iloc[:,1]
Y


# ---

# ## **Data split**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# ## **Linear Regression Model**

# In[ ]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:


model = linear_model.LinearRegression()
model.fit(X_train, Y_train)


# ### **Predicts the X_train**

# In[ ]:


Y_pred_train = model.predict(X_train)


# In[ ]:


print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_train, Y_pred_train))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_train, Y_pred_train))


# ### **Predicts the X_test**

# In[ ]:


Y_pred_test = model.predict(X_test)


# In[ ]:


print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred_test))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred_test))


# ### **Linear Regression Equation**

# The work of Delaney$^1$ provided the following linear regression equation:
# 
# > LogS = 0.16 -  0.63 cLogP - 0.0062 MW + 0.066 RB - 0.74 AP
# 
# The reproduction by Pat Walters$^2$ provided the following:
# 
# > LogS = 0.26 -  0.74 LogP - 0.0066 MW + 0.0034 RB - 0.42 AP
# 
# This notebook's reproduction gave the following equation:
# 
# * Based on the Train set
# > LogS = 0.30 -0.75 LogP - .0066 MW -0.0041 RB - 0.36 AP
# 
# * Based on the Full dataset
# > LogS =  0.26 -0.74 LogP - 0.0066 + MW 0.0032 RB - 0.42 AP

# #### **Our linear regression equation**

# In[ ]:


print('LogS = %.2f %.2f LogP %.4f MW %.4f RB %.2f AP' % (model.intercept_, model.coef_[0], model.coef_[1], model.coef_[2], model.coef_[3] ) )


# The same equation can also be produced with the following code (which breaks up the previous one-line code into several comprehensible lines.

# In[ ]:


yintercept = '%.2f' % model.intercept_
LogP = '%.2f LogP' % model.coef_[0]
MW = '%.4f MW' % model.coef_[1]
RB = '%.4f RB' % model.coef_[2]
AP = '%.2f AP' % model.coef_[3]


# In[ ]:


print('LogS = ' + 
      ' ' + 
      yintercept + 
      ' ' + 
      LogP + 
      ' ' + 
      MW + 
      ' ' + 
      RB + 
      ' ' + 
      AP)


# #### **Use entire dataset for model training (For Comparison)**

# In[ ]:


full = linear_model.LinearRegression()
full.fit(X, Y)


# In[ ]:


full_pred = model.predict(X)


# In[ ]:


print('Coefficients:', full.coef_)
print('Intercept:', full.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y, full_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y, full_pred))


# In[ ]:


full_yintercept = '%.2f' % full.intercept_
full_LogP = '%.2f LogP' % full.coef_[0]
full_MW = '%.4f MW' % full.coef_[1]
full_RB = '+ %.4f RB' % full.coef_[2]
full_AP = '%.2f AP' % full.coef_[3]


# In[ ]:


print('LogS = ' + 
      ' ' + 
      full_yintercept + 
      ' ' + 
      full_LogP + 
      ' ' + 
      full_MW + 
      ' ' + 
      full_RB + 
      ' ' + 
      full_AP)


# ## **Scatter plot of experimental vs. predicted LogS**

# In[ ]:


import matplotlib.pyplot as plt


# ### **Quick check of the variable dimensions of Train and Test sets**

# In[ ]:


Y_train.shape, Y_pred_train.shape


# In[ ]:


Y_test.shape, Y_pred_test.shape


# ### **Vertical plot**

# In[ ]:


plt.figure(figsize=(5,11))

# 2 row, 1 column, plot 1
plt.subplot(2, 1, 1)
plt.scatter(x=Y_train, y=Y_pred_train, c="#7CAE00", alpha=0.3)

# Add trendline
# https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs
z = np.polyfit(Y_train, Y_pred_train, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')


# 2 row, 1 column, plot 2
plt.subplot(2, 1, 2)
plt.scatter(x=Y_test, y=Y_pred_test, c="#619CFF", alpha=0.3)

z = np.polyfit(Y_test, Y_pred_test, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')

plt.savefig('plot_vertical_logS.png')
plt.savefig('plot_vertical_logS.pdf')
plt.show()


# ### **Horizontal plot**

# In[ ]:


plt.figure(figsize=(11,5))

# 1 row, 2 column, plot 1
plt.subplot(1, 2, 1)
plt.scatter(x=Y_train, y=Y_pred_train, c="#7CAE00", alpha=0.3)

z = np.polyfit(Y_train, Y_pred_train, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')

# 1 row, 2 column, plot 2
plt.subplot(1, 2, 2)
plt.scatter(x=Y_test, y=Y_pred_test, c="#619CFF", alpha=0.3)

z = np.polyfit(Y_test, Y_pred_test, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.xlabel('Experimental LogS')

plt.savefig('plot_horizontal_logS.png')
plt.savefig('plot_horizontal_logS.pdf')
plt.show()


# In[ ]:


get_ipython().system(' ls -l')


# ---

# ## **Reference**
# 
# 1. John S. Delaney. [ESOL:? Estimating Aqueous Solubility Directly from Molecular Structure](https://pubs.acs.org/doi/10.1021/ci034243x). ***J. Chem. Inf. Comput. Sci.*** 2004, 44, 3, 1000-1005.
# 
# 2. Pat Walters. [Predicting Aqueous Solubility - It's Harder Than It Looks](http://practicalcheminformatics.blogspot.com/2018/09/predicting-aqueous-solubility-its.html). ***Practical Cheminformatics Blog***
# 
# 3. Bharath Ramsundar, Peter Eastman, Patrick Walters, and Vijay Pande. [Deep Learning for the Life Sciences: Applying Deep Learning to Genomics, Microscopy, Drug Discovery, and More](https://learning.oreilly.com/library/view/deep-learning-for/9781492039822/), O'Reilly, 2019.
# 
# 4. [Supplementary file](https://pubs.acs.org/doi/10.1021/ci034243x) from Delaney's ESOL:? Estimating Aqueous Solubility Directly from Molecular Structure.
