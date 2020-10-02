#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_excel('/kaggle/input/convid/Mpro full XChem screen - experiment summary - ver-2020-03-25-annotated.xlsx')
df = df[df["RefinementOutcome"].isin(["7 - Analysed & Rejected","6 - Deposited"])]
df = df[df['CompoundSMILES'].notnull()]
df.RefinementOutcome.replace("7 - Analysed & Rejected",False,inplace=True)
df.RefinementOutcome.replace("6 - Deposited",True,inplace=True)
df.RefinementOutcome.value_counts()


# In[ ]:


get_ipython().system('conda install -y -c rdkit rdkit;')
# !pip install pandas==0.23.0


# # Mol Feature Engineering

# In[ ]:


#Importing Chem module
from rdkit import Chem 

#Method transforms smiles strings to mol rdkit object
df['mol'] = df['CompoundSMILES'].apply(lambda x: Chem.MolFromSmiles(x)) 
df['mol'] = df['mol'].apply(lambda x: Chem.AddHs(x))
df['num_of_atoms'] = df['mol'].apply(lambda x: x.GetNumAtoms())
df['num_of_heavy_atoms'] = df['mol'].apply(lambda x: x.GetNumHeavyAtoms())

def number_of_atoms(atom_list, df):
    for i in atom_list:
        df['num_of_{}_atoms'.format(i)] = df['mol'].apply(lambda x: len(x.GetSubstructMatches(Chem.MolFromSmiles(i))))

number_of_atoms(['C','O', 'N', 'Cl'], df)        

from rdkit.Chem import Descriptors
df['tpsa'] = df['mol'].apply(lambda x: Descriptors.TPSA(x))
df['mol_w'] = df['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
df['num_valence_electrons'] = df['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))
df['num_heteroatoms'] = df['mol'].apply(lambda x: Descriptors.NumHeteroatoms(x))


# In[ ]:


df.head()


# In[ ]:


df.columns.values


# In[ ]:


get_ipython().run_cell_magic('time', '', 'import lightgbm as lgb\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import KFold,StratifiedKFold,GroupKFold, train_test_split\nfrom sklearn.metrics import roc_auc_score, roc_curve\n\n\ndef lgb_kfold(train_df,test_df, train_target,  test_target,features,target,cat_features,folds,params,classification=False):\n    oof_preds = np.zeros(train_df.shape[0])\n    sub_preds = np.zeros(test_df.shape[0])\n\n    cv_list = []\n    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_target)):\n        print (\'FOLD:\' + str(n_fold+1))\n        \n        train_x, train_y = train_df.iloc[train_idx], train_target.iloc[train_idx]\n        valid_x, valid_y = train_df.iloc[valid_idx], train_target.iloc[valid_idx]\n        \n        print (\'train_x shape:\',train_x.shape)\n        print (\'valid_x shape:\',valid_x.shape)\n        \n        dtrain = lgb.Dataset(train_x, label=train_y,categorical_feature=cat_features)\n        dval = lgb.Dataset(valid_x, label=valid_y, reference=dtrain,categorical_feature=cat_features) \n        bst = lgb.train(params, dtrain, num_boost_round=10000,\n            valid_sets=[dval,dtrain], verbose_eval=100,early_stopping_rounds=100, ) #feval = evalerror\n        new_list = sorted(zip(features, bst.feature_importance(\'gain\')),key=lambda x: x[1], reverse=True)[:]\n        for item in new_list:\n            print (item) \n              \n        oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)\n        sub_preds += bst.predict(test_df[features], num_iteration=bst.best_iteration) / folds.n_splits\n        \n        oof_cv = roc_auc_score(valid_y,  oof_preds[valid_idx])\n        cv_list.append(oof_cv)\n        print (cv_list)\n \n    auc = roc_auc_score(train_target,  oof_preds)\n    print(\'Full OOF AUC %.6f\' % auc)  \n    auc = roc_auc_score(test_target,  sub_preds)\n    print(\'Holdout OOF AUC %.6f\' % auc) \n    train_df[\'prediction\'] = oof_preds\n    test_df[\'prediction\'] = sub_preds\n    \n    return train_df,test_df,auc\n\nparams = {\n        "nthread": -1,\n        "boosting_type": "gbdt",\n        "objective": "binary",\n        "metric": "auc",\n        "min_data_in_leaf": 70, \n        "min_gain_to_split": 0.1,\n        "min_child_weight": 0.001,\n        "reg_alpha": 0.1, \n        "reg_lambda": 1, \n        "max_depth" : -1,\n        "num_leaves" : 31, \n        "max_bin" : 256, \n        "is_unbalanced" : True, \n        "learning_rate" :0.01,\n        "bagging_fraction" : 0.9,\n        "bagging_freq" : 1,\n        "bagging_seed" : 4590,\n        "feature_fraction" : 0.9,\n        "verbosity": -1,\n        "boost_from_average": False,\n}\n\n    \ndrop_features = [\'CrystalName\', \'CompoundCode\', \'CompoundSMILES\', \'MountingResult\',\n       \'DataCollectionOutcome\', \'DataProcessingResolutionHigh\',\n       \'RefinementOutcome\', \'Deposition_PDB_ID\', \'mol\', \n                 ]\ntarget = \'RefinementOutcome\'\ncat_features = []\nfeatures = [f for f in df.columns if f not in drop_features]\nprint (\'features:\', len(features), features)\n\ntrain_x,test_x, train_y,  test_y  = train_test_split(df[features], df[target], test_size=0.2, random_state=223,stratify=df[target]) \n\nfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=223)\ntrain_lgb,test_lgb,auc = lgb_kfold(train_x,test_x, train_y,  test_y,features,target,cat_features,folds,params,classification=True)')


# # Mol2vec

# In[ ]:


get_ipython().system('pip install git+https://github.com/samoturk/mol2vec;')


# In[ ]:


df.shape


# In[ ]:


from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
model = word2vec.Word2Vec.load('/kaggle/input/mol2vec/model_300dim.pkl')

#Constructing sentences
df['sentence'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)

#Extracting embeddings to a numpy.array
#Note that we always should mark unseen='UNK' in sentence2vec() so that model is taught how to handle unknown substructures
df['mol2vec'] = [DfVec(x) for x in sentences2vec(df['sentence'], model, unseen='UNK')]
X = np.array([x.vec for x in df['mol2vec']])
X.shape


# In[ ]:


df = df.reset_index(drop=True)
mdf = pd.DataFrame(X)
new_df = pd.concat([mdf, df], axis=1)


# In[ ]:


new_df.head()


# In[ ]:


drop_features = ['CrystalName', 'CompoundCode', 'CompoundSMILES', 'MountingResult',
       'DataCollectionOutcome', 'DataProcessingResolutionHigh',
       'RefinementOutcome', 'Deposition_PDB_ID', 'mol', 'sentence', 'mol2vec'
                 ]
target = 'RefinementOutcome'
cat_features = []
features = [f for f in new_df.columns if f not in drop_features]
print ('features:', len(features), features)
train_x,test_x, train_y,  test_y  = train_test_split(new_df[features], new_df[target], test_size=0.2, random_state=223,stratify=new_df[target]) 

train_lgb,test_lgb,auc = lgb_kfold(train_x,test_x, train_y,  test_y,features,target,cat_features,folds,params,classification=True)


# In[ ]:




