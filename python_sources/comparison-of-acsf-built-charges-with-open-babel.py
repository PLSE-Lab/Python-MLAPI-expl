#!/usr/bin/env python
# coding: utf-8

# # Synopsis
# 
# 
# This kernel objective is to check the robustness of Mulliken charges estimated in this kernel : [Predicting Mulliken Charges With ACSF Descriptors](https://www.kaggle.com/borisdee/predicting-mulliken-charges-with-acsf-descriptors)
# 
# 
# # Context
# 
# Boris has built a clever model to estimate Mulliken charges of atoms for the test set in the kernel linked above. These local properties are a significant feature in any serious model for predicting the `scalar_coupling_constant`.
# 
# As we do not have a comparison basis excepted the cross validation of the prediction model itself, I wanted  to check how this data compare against the Open Babel estimated values. The idea here is to check if the correlation of the competiton provided Mulliken charges and Open Babel estimates behave the same when using the Boris set.
# 
# # Method
# 
# I will use here the same code as in my previous kernel [V7 Estimation of Mulliken Charges with Open Babel](https://www.kaggle.com/asauve/v7-estimation-of-mulliken-charges-with-open-babel). But the data used for correlation will be
# * Boris Mulliken charges estimates for test set
# * Open babel Partial charges estimates for test set -- for each of the 11 method available --
# 
# After that, the 11 correlation coefficients of the test set will be compared againts the 11 ones obtained with the train set.
# 
# 

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

load_dir_csv("../input/champs-scalar-coupling/", 
             ["test.csv"])
load_dir_csv("../input/boris-mulliken-test", 
             ["mulliken_charges_test_set.csv"])
load_dir_csv("../input/open-babel-atom-charges/",
            ["test_ob_charges.csv"])


# # Load Mulliken Charges from Boris test set

# In[ ]:


# differentiate train and test set
test_molecules  = test.molecule_name.unique()

mulliken_test_boris = []
mulliken_charges_test_set_idx = mulliken_charges_test_set.set_index(['molecule_name'])
# ensure mulliken charges are in same order as for Open Babel
for molecule_name in test_molecules:
    mc  = mulliken_charges_test_set_idx.loc[molecule_name].sort_index()
    mulliken_test_boris.extend(mc.mulliken_charge.values)


# # Load partial charges from Open Babel set
# 
# These values have been computed in a previous kernel :   [V7 Estimation of Mulliken Charges with Open Babel](https://www.kaggle.com/asauve/v7-estimation-of-mulliken-charges-with-open-babel)

# In[ ]:



ob_methods = [ "eem", "mmff94", "gasteiger", "qeq", "qtpie", 
               "eem2015ha", "eem2015hm", "eem2015hn", "eem2015ba", "eem2015bm", "eem2015bn" ]

mulliken_test_ob = [ [] for _ in ob_methods ]

test_ob_charges_idx = test_ob_charges.set_index(['molecule_name','atom_index'])
for molecule_name in test_molecules:
    mc  = test_ob_charges_idx.loc[molecule_name].sort_index()
    for i, method in enumerate(ob_methods):
        mulliken_test_ob[i].extend(mc[method].values)


# # Compare ACSF built charges to Open Babel charges
# 
# This section produce the same plots as in [V7 Estimation of Mulliken Charges with Open Babel](https://www.kaggle.com/asauve/v7-estimation-of-mulliken-charges-with-open-babel) but with Boris test data.

# In[ ]:


# correlation plots
test_corrs = []
for method in ob_methods:
    charges = test_ob_charges[method].values
    fig = plt.figure()
    ax = sns.scatterplot(mulliken_test_boris, charges)
    corr, pval = pearsonr(mulliken_test_boris, charges)
    test_corrs.append(corr)
    title = f"method = {method:10s}  test_corr = {corr:7.4f}"
    print(title)
    plt.title(title)
    plt.xlabel("ACSF built charge")
    plt.ylabel(f"Open Babel {method}")


# In[ ]:


fig = plt.figure(figsize=(12,6))
data = pd.DataFrame( {'method':ob_methods, 'corr':[abs(c) for c in test_corrs]}).sort_values('corr', ascending=False)
ax = sns.barplot(data=data, x='corr', y='method', orient="h", dodge=False)
plt.title("Correlation coefficients obtained for Boris test set vs Open Babel")


# Visually the correlations values are very similar to those obtained in the train set case.
# 
# Let's compare both!

# # Compare correlation coefs of train set vs predicted test set

# In[ ]:


# These values are cut'n paste of previous kernel
train_corrs = np.asarray([0.9320270774536107, 0.5796413944230157, 0.6898804803363003, -0.78940127765262, 0.7562651088260747, 0.5997842861246153, 0.9078154604927864, 0.9265845452071686, 0.6473353501533773, 0.8564406550261474, 0.9377723092343451])

print("std(test_corrs - train_corrs) = ", np.std(test_corrs - train_corrs))
print("pearsonr(train_corrs-test_corrs) = ", pearsonr(train_corrs, test_corrs))

lines = plt.plot(test_corrs - train_corrs)
lines = plt.plot([0, 10], [0, 0], color="black")
txt = plt.title("test_corrs - train_corrs")
txt = plt.xlabel("Open Babel method index")


# Excellent agreement!
# 
# It is interresting that the Open Babel charges estimates are on average more correlated with the Boris predictions than with the QM competition data. Maybe an effect of both sets using simplified models?

# # Conclusion
# 
# There is an excellent agreement between the test charges produced by Boris with ACSF topological indices and the competion Mulliken charges *when compared to Open Babel estimations*.
# 
# While this is not a definitive validation of the dataset it is a strong argument in favor of its quality. I would definitively recommend using it :-)

# 
