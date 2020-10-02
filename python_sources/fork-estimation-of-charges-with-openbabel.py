#!/usr/bin/env python
# coding: utf-8

# # Synopsis
# 
# ![OpenBabel](http://openbabel.org/babel130.png)
# 
# In this kernel I will explore an alternative way of calculating local `mulliken charges`. 
# You may be interrested only if you wish to incorportate these features in your model.  
# => The charges calculated with Open Babel are available as an autonomous dataset in [Open Babel Atom Charges](https://www.kaggle.com/asauve/open-babel-atom-charges)
# 
# The [Open Babel](http://openbabel.org/wiki/Main_Page) Package allow to
# * Read .xyz files (the ones provided in the structures directory)
# * Build bonding scheme (groovy baby!)
# * Compute partial charges, with several builtin methods
# 
# Humm this last one is really interresting as it provides a quick way of computing atoms local charge with a method based on linear algebra. Hence it is fast and in the range of the allowed options for the competition.
# 
# The most promising method is Electronegativity Equalization Method  (EEM) which is describeed in this paper:  
# [High-quality and universal empirical atomic charges for chemoinformatics applications](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4667495/)
# 
# 
# 

# # Definition of partial charge
# 
# There is no such thing as a space localized electron, they are rather something like a wave all around atomic nuclei. But there are some models for getting an approximation of the average charge at the location of atoms. This is this what is computed in this kernel under the name partial charge. 
# 
# Open babel offers several methods to compute these values (you have to dig into the code to get them) which are all provided under the column with the matching name.

# # Load Data

# In[ ]:


import numpy as np # linear algebra
from scipy.stats.stats import pearsonr
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm_notebook as tqdm
import seaborn as sns 
import matplotlib.pyplot as plt
sns.set()
import os

from sklearn.linear_model import LinearRegression,ElasticNetCV, RidgeCV
from sklearn.model_selection import cross_val_predict, cross_val_score, GroupKFold


# Load CSV data

# In[ ]:


def load_dir_csv(directory, csv_files=None):
    if csv_files is None:
        csv_files = sorted( [ f for f in os.listdir(directory) if f.endswith(".csv") ])    
    csv_vars  = [ filename[:-4] for filename in csv_files ]
    gdict = globals()
    for filename, var in zip( csv_files, csv_vars ):
        print(f"{var:32s} = pd.read_csv({directory}/{filename})")
        gdict[var] = pd.read_csv( f"{directory}/{filename}" )
        print(f"{'nb of cols ':32s} = " + str(len(gdict[var])))
        display(gdict[var].head())

load_dir_csv("../input/", 
             ["train.csv", "test.csv", "structures.csv", "mulliken_charges.csv"])


# # Install OpenBabel
# 
# The installation method was  demonstrated in: [Molecule with OpenBabel](https://www.kaggle.com/jmtest/molecule-with-openbabel)

# In[ ]:


get_ipython().system('conda install -y -c openbabel openbabel ')
import openbabel as ob


# # Load Mulliken Charges from train set

# In[ ]:


# differentiate train and test set
train_molecules = train.molecule_name.unique()
test_molecules  = test.molecule_name.unique()


mulliken   = []
mulliken_charges_idx = mulliken_charges.set_index(['molecule_name'])
# ensure mulliken charges are in same order as for partial charges
for molecule_name in train_molecules:
    mc  = mulliken_charges_idx.loc[molecule_name].sort_index()
    mulliken.extend(mc.mulliken_charge.values)


# In[ ]:


print(len(mulliken))
mulliken[0:2]


# # Compute atom partial charges with Open Babel 

# In[ ]:


##
## Build molecules from files.xyz
##

obConversion = ob.OBConversion()
#def read_ob_molecule(molecule_name, datadir="../input/champs-scalar-coupling/structures"):
def read_ob_molecule(molecule_name, datadir="../input/structures"):
    mol = ob.OBMol()
    path = f"{datadir}/{molecule_name}.xyz"
    if not obConversion.ReadFile(mol, path):
        raise FileNotFoundError(f"Could not read molecule {path}")
    return mol
    

ob_methods = [ "eem", "mmff94", "gasteiger", "qeq", "qtpie", 
               "eem2015ha", "eem2015hm", "eem2015hn", "eem2015ba", "eem2015bm", "eem2015bn" ]

structures_idx = structures.set_index( ["molecule_name"] )
def get_charges_df(molecule_names):
    ob_methods_charges = [ [] for _ in ob_methods]
    ob_molecule_name = []  # container for output  DF
    ob_atom_index    = []  # container for output  DF
    ob_error         = []
    for molecule_name in molecule_names:
        # fill data for output DF
        ms = structures_idx.loc[molecule_name].sort_index()
        natoms = len(ms)
        ob_molecule_name.extend( [molecule_name] * natoms )
        ob_atom_index.extend(    ms.atom_index.values )

        # calculate open babel charge for each method
        mol = read_ob_molecule(molecule_name)
        assert( mol.NumAtoms() == natoms ) # consistency
        error = 0
        for method, charges in zip(ob_methods, ob_methods_charges):
            ob_charge_model = ob.OBChargeModel.FindType(method)
            if not ob_charge_model.ComputeCharges(mol):
                error = 1
            charges.extend( ob_charge_model.GetPartialCharges() )
        ob_error.extend([error] * natoms)
            
    ob_charges = pd.DataFrame({
        'molecule_name' : ob_molecule_name,
        'atom_index'    : ob_atom_index}
    )
    for method, charges in zip(ob_methods, ob_methods_charges):
        ob_charges[method] = charges
    ob_charges["error"] = ob_error
    display(ob_charges.head())
    return ob_charges


# In[ ]:


get_ipython().run_line_magic('time', 'train_ob_charges = get_charges_df(train_molecules)')


# In[ ]:


get_ipython().run_line_magic('time', 'test_ob_charges = get_charges_df(test_molecules)')


# # Compare Mulliken charges to Open Babel EEM

# In[ ]:


train_molecules[0:1]


# In[ ]:


# correlation plots
corrs = []
for method in ob_methods:
    charges = train_ob_charges[method].values
    fig = plt.figure()
    ax = sns.scatterplot(mulliken, charges)
    corr, pval = pearsonr(mulliken, charges)
    corrs.append(corr)
    title = f"method = {method:10s}  corr = {corr:7.4f}"
    print(title)
    plt.title(title)
    plt.xlabel("Mulliken charge")
    plt.ylabel(f"Open Babel {method}")


# In[ ]:


fig = plt.figure(figsize=(12,6))
data = pd.DataFrame( {'method':ob_methods, 'corr':[abs(c) for c in corrs]}).sort_values('corr', ascending=False)
ax = sns.barplot(data=data, x='corr', y='method', orient="h", dodge=False)


# ## Added step: cross validated predictions of the target variable
# * Could just use single best variable, but using multiple allows us to take advantage of any non colinear properties. i.e if they're not correlated, then combining them may improve prediction. 
# * Likely bettered by: Adding in the OHE of the atome , and molecule base d properties (index number, surrounding atoms..) 
# 
# * Linear regression default score is R^2 : higher is better
# 
# * Conclusion: using multiple methods in gives a better score than just the top method(s) alone \[makes sense!]

# In[ ]:


from sklearn.preprocessing import LabelEncoder
# split by molecule for CV
le = LabelEncoder()


# In[ ]:


train_ob_charges["mulliken_actual"] = mulliken
train_ob_charges.tail()


# In[ ]:


## Let's look at correlation between the methods

train_ob_charges.drop(["error"],axis=1).corr()


# In[ ]:


X = train_ob_charges.drop(["mulliken_actual"],axis=1).copy()
X["molecule_name"] = le.fit_transform(X["molecule_name"])

y = mulliken

print(X.shape)
# print()
X.head()


# In[ ]:


# LinearRegression,ElasticNetCV, RidgeCV
clf = LinearRegression()

print(cross_val_score(clf,X,y,groups=X["molecule_name"],cv=GroupKFold(4)).mean())


# In[ ]:


print(cross_val_score(clf,X.drop(["atom_index"],axis=1),y,groups=X["molecule_name"],cv=GroupKFold(4)).mean())


# In[ ]:


print(cross_val_score(clf,X[["atom_index","eem"]],y,groups=X["molecule_name"],cv=GroupKFold(4)).mean())


# In[ ]:


# train_ob_charges.isna().sum() # no missing values
# sns.heatmap(train_ob_charges.drop(["atom_index","error"],axis=1)) # gives error about ufunc 'isnan' ? 


# # Write Output dataframes

# In[ ]:


train_ob_charges.to_csv("train_ob_charges.csv.gzip",index=False,compression="gzip")
test_ob_charges.to_csv("test_ob_charges.csv.gzip",index=False,compression="gzip")


# # Conclusion

# Open Babel local charge estimation is worth a try in our models!
# 
# Best match is method **"eem"** and some of its variants!
# 
# Note that some outliers in the scatter plots come from different convention for how ionization charge is located inside the molecule. For example COO- can have -1 charge set on one O, or let it be shared by both oxygens. Hence the fact that some models differ from the data in the table `mulliken_charges` does not mean that it is systematically worse for the end purpose to predict coupling.
# 
