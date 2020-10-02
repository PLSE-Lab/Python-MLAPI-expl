#!/usr/bin/env python
# coding: utf-8

# # Chemical Shielding
# 
# CORY M. WIDDIFIELD, ROBERT W. SCHURKO.
# "Understanding Chemical Shielding Tensors Using Group Theory, MO Analysis, and Modern Density- Functional Theory"
# *Concepts in Magnetic Resonance Part A (Bridging Education and Research). (2009)
# https://onlinelibrary.wiley.com/doi/pdf/10.1002/cmr.a.20136
# 
# The chemical shielding Hamiltonian:
# 
# ![Hamiltonian](https://user-images.githubusercontent.com/11532812/60358602-78018d80-9a11-11e9-894f-60b8122e8a4c.png)
# 
# where gamma is the gyromagnetic ratio, I^ is the nuclear spin operator, and sigma is the magnetic shielding tensor.

# In[ ]:


get_ipython().system('pip install ase==3.17 schnetpack')


# We need `ASE 3.17` for `SchNetPack 0.2.1`.

# In[ ]:


import numpy as np
import pandas as pd
molecules = pd.read_csv('../input/structures.csv')
molecules = molecules.groupby('molecule_name')
magnetic_shielding_tensors = pd.read_csv('../input/magnetic_shielding_tensors.csv')


# In[ ]:


magnetic_shielding_tensors.head(10)


# The magnetic shielding tensor:
# ![Magnetic shielding tensor](https://user-images.githubusercontent.com/11532812/60358839-3ae9cb00-9a12-11e9-8f3a-0236f7bc072d.png)
# 
# The symmetric portion of the shielding tensor can be diagonalized into its own principal axis system (PAS):
# 
# ![PAS](https://user-images.githubusercontent.com/11532812/60359035-c95e4c80-9a12-11e9-897d-2e422e0f2e83.png)

# In[ ]:


x = magnetic_shielding_tensors.columns.values[2:]
x = magnetic_shielding_tensors[x].values
x = x.reshape(-1,3,3)
x = x + np.transpose(x,(0,2,1))
x = 0.5 * x
w, v = np.linalg.eigh(x)


# The isotropic shielding value is defined as:
# 
# ![Isotropic shielding tensor](https://user-images.githubusercontent.com/11532812/60358900-75536800-9a12-11e9-850c-654dc81fc258.png)

# In[ ]:


sigma_iso = np.sum(w, axis=1)/3 


# ![span](https://user-images.githubusercontent.com/11532812/60359189-45589480-9a13-11e9-9016-42f8a90c1ddd.png)
# 
# Ths span (Omega) describes the magnitude of the shielding anisotropy.

# In[ ]:


omega = w[:,2] - w[:,0]


# ![skew](https://user-images.githubusercontent.com/11532812/60359338-c57efa00-9a13-11e9-9619-17ebd4f44ac8.png)
# 
# The skew (kappa) describes degree of axial symmetry of the shielding tensor.

# In[ ]:


kappa = 3 * (sigma_iso - w[:,1])/omega


# In[ ]:


magnetic_shielding_parameters = magnetic_shielding_tensors[magnetic_shielding_tensors.columns.values[:2]]
magnetic_shielding_parameters = pd.DataFrame(magnetic_shielding_parameters)
magnetic_shielding_parameters["sigma_iso"] = sigma_iso
magnetic_shielding_parameters["omega"] = omega
magnetic_shielding_parameters["kappa"] = kappa

magnetic_shielding_parameters.head(10)


# In[ ]:


_ = magnetic_shielding_parameters.sigma_iso.hist(bins=100)


# In[ ]:


_ = magnetic_shielding_parameters.omega.hist(bins=100)


# In[ ]:


_ = magnetic_shielding_parameters.kappa.hist(bins=100)


# Please see LibreTexts [Chemical Shift][1] if you want to understand the meaning of the parameters.
# 
# [1]: https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Spectroscopy/Magnetic_Resonance_Spectroscopies/Nuclear_Magnetic_Resonance/NMR%3A_Theory/NMR_Interactions/Chemical_Shift_(Shielding)

# # ASE Database

# In[ ]:


train = pd.read_csv('../input/train.csv')
train_molecule_names = train.molecule_name.unique()

msp = magnetic_shielding_parameters.groupby('molecule_name')


# In[ ]:


from ase import Atoms
from ase.db import connect

def create_db(db_path, molecule_names):
    with connect(db_path) as db:
        for name in molecule_names:
            mol = molecules.get_group(name)
            atoms = Atoms(symbols=mol.atom.values,
                          positions=[(row.x,row.y,row.z) for row in mol.itertuples()])
            try:
                mol_msp = msp.get_group(name)
                sigma_iso = mol_msp['sigma_iso'].values.reshape(-1,1)
                omega = mol_msp['omega'].values.reshape(-1,1)
                kappa = mol_msp['kappa'].values.reshape(-1,1)
            except KeyError:
                sigma_iso, omega, kappa = [None] * 3
            db.write(atoms, name=name,
                     data=dict(sigma_iso=sigma_iso, omega=omega, kappa=kappa)
                    )


# In[ ]:


champs_path = 'CHAMPS_train.db'
dataset_size = len(train_molecule_names) # 40000
dataset_molecule_names = train_molecule_names[:dataset_size]
create_db(db_path=champs_path, molecule_names=dataset_molecule_names)


# In[ ]:


with connect(champs_path) as db:
    print(len(db))


# In[ ]:


import schnetpack
dataset = schnetpack.data.AtomsData(champs_path,
                properties=['sigma_iso', 'omega', 'kappa'])


# In[ ]:


len(dataset)


# # Shielding Parameter Prediction
# 
# ## SchNet Model

# In[ ]:


import pandas as pd

import torch
import torch.nn.functional as F
from torch.optim import Adam

import schnetpack as spk
import schnetpack.atomistic as atm
import schnetpack.representation as rep
from schnetpack.datasets import *

device = torch.device("cuda")


# In[ ]:


# This function comes from the following script:
# https://github.com/atomistic-machine-learning/schnetpack/blob/v0.2.1/src/scripts/schnetpack_qm9.py
def evaluate_dataset(metrics, model, loader, device):
    for metric in metrics:
        metric.reset()

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }
            result = model(batch)

            for metric in metrics:
                metric.add_batch(batch, result)

    results = [
        metric.aggregate() for metric in metrics
    ]
    return results


# In[ ]:


from schnetpack.data import Structure

class MagneticShielding(atm.Atomwise):
    def __init__(self, property):
        super(MagneticShielding, self).__init__(return_contributions=True)
        self.property = property
        
    def forward(self, inputs):
        result = super().forward(inputs)
        
        atom_mask = inputs[Structure.atom_mask].byte()
        
        yi = inputs[self.property]
        yi = torch.masked_select(yi.squeeze(dim=2), atom_mask)
        inputs[self.property+'_true'] = yi
        
        yi = result['yi']
        yi = torch.masked_select(yi.squeeze(dim=2), atom_mask)
        result[self.property+'_pred'] = yi
        
        return result


# In[ ]:


def schnet_model(property):
    reps = rep.SchNet(n_interactions=6)
    output = MagneticShielding(property=property)
    model = atm.AtomisticModel(reps, output)
    model = model.to(device)
    return model


# In[ ]:


def train_model(property, max_epochs=500):
    # split in train and val
    n_dataset = len(dataset)
    n_val = n_dataset // 10
    train_data, val_data, test_data = dataset.create_splits(n_dataset-n_val*2, n_val)
    train_loader = spk.data.AtomsLoader(train_data, batch_size=128, num_workers=2)
    val_loader = spk.data.AtomsLoader(val_data, batch_size=256, num_workers=2)

    # create model
    model = schnet_model(property)

    # create trainer
    target_key = property+'_true'
    output_key = property+'_pred'
    opt = Adam(model.parameters(), lr=1e-4)
    loss = lambda b, p: F.mse_loss(p[output_key], b[target_key])
    metrics = [
        spk.metrics.MeanAbsoluteError(target_key, output_key, name='MAE_'+property),
        spk.metrics.RootMeanSquaredError(target_key, output_key, name='RMSE_'+property),
    ]
    hooks = [
        spk.train.MaxEpochHook(max_epochs),
        spk.train.CSVHook(property+'/log', metrics, every_n_epochs=1),
    ]
    trainer = spk.train.Trainer(property+'/output', model, loss,
                            opt, train_loader, val_loader, hooks=hooks)

    # start training
    trainer.train(device)
    
    # evaluation
    model.load_state_dict(torch.load(property+'/output/best_model'))
    test_loader = spk.data.AtomsLoader(test_data, batch_size=256, num_workers=2)
    model.eval()

    df = pd.DataFrame()
    df['metric'] = ['MAE', 'RMSE']
    df['training'] = evaluate_dataset(metrics, model, train_loader, device)
    df['validation'] = evaluate_dataset(metrics, model, val_loader, device)
    df['test'] = evaluate_dataset(metrics, model, test_loader, device)
    display(df)
    
    return test_data


# In[ ]:


def show_history(property):
    df = pd.read_csv(property+'/log/log.csv')
    display(df.tail())
    max_value = None # df['RMSE_'+property].min()*5
    _ = df[['MAE_'+property,'RMSE_'+property]].plot(ylim=(0,max_value))


# In[ ]:


def test_prediction(dataset, property):
    # create model
    model = schnet_model(property)
    
    # load best parameters
    model.load_state_dict(torch.load(property+'/output/best_model'))
    loader = spk.data.AtomsLoader(dataset, batch_size=256, num_workers=2)
    model.eval()
    
    # predict shielding parameters
    targets = []
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }
            result = model(batch)
            targets += batch[property+'_true'].tolist()
            predictions += result[property+'_pred'].tolist()
    return targets, predictions


# In[ ]:


def show_predictions(dataset, property):
    targets, predictions = test_prediction(dataset, property)
    df_pred = pd.DataFrame()
    df_pred['Target'] = targets
    df_pred['Prediction'] = predictions
    df_pred.plot.scatter(x='Target', y='Prediction', title=property)


# ## Results

# In[ ]:


used_test_data = dict()
for p in ['sigma_iso', 'omega', 'kappa']:
    print(p)
    used_test_data[p] = train_model(p, max_epochs=50)
    show_history(p)


# In[ ]:


for p in ['sigma_iso', 'omega', 'kappa']:
    show_predictions(used_test_data[p], p)


# In[ ]:


get_ipython().system('mv kappa/log/log.csv log_kappa.csv')
get_ipython().system('mv omega/log/log.csv log_omega.csv')
get_ipython().system('mv sigma_iso/log/log.csv log_sigma_iso.csv')
get_ipython().system('mv kappa/output/best_model best_model_kappa')
get_ipython().system('mv omega/output/best_model best_model_omega')
get_ipython().system('mv sigma_iso/output/best_model best_model_sigma_iso')
get_ipython().system('rm -r kappa')
get_ipython().system('rm -r omega')
get_ipython().system('rm -r sigma_iso')


# In[ ]:


get_ipython().system('ls')

