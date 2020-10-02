#!/usr/bin/env python
# coding: utf-8

# This kernel presents how to build an ASE database from CHAMPS data and create a dataset, and then uses it for molecular property predictions.
# 
# First, please see an SchNet tutorial on QM9:
# 
# [SchNet QM9](https://www.kaggle.com/tonyyy/schnet-qm9)
# 
# which contains references of the SchNet architecture.
# 
# This kernel uses only potential energies and dipole moments in CHAMPS data. Mulliken charges and magnetic shielding tensors are used in the following kernels:
# * [SchNet Mulliken Charges](https://www.kaggle.com/tonyyy/schnet-mulliken-charges)
# * [SchNet Magnetic Shielding](https://www.kaggle.com/tonyyy/schnet-magnetic-shielding)

# In[ ]:


get_ipython().system('pip install ase==3.17 schnetpack')


# We need `ASE 3.17` for `SchNetPack 0.2.1`.

# In[ ]:


get_ipython().system('ls ../input')


# # CHAMPS data

# In[ ]:


import numpy as np
import pandas as pd
molecules = pd.read_csv('../input/structures.csv')
molecules = molecules.groupby('molecule_name')
energies = pd.read_csv('../input/potential_energy.csv')
dipoles = pd.read_csv('../input/dipole_moments.csv')
dipoles['scalar'] = np.sqrt(np.square(dipoles[['X', 'Y', 'Z']]).sum(axis=1))


# The total number of molecules:

# In[ ]:


molecules.ngroups


# ### Potential Energy

# In[ ]:


energies.head()


# In[ ]:


len(energies)


# In[ ]:


energy_series = pd.Series(energies.set_index('molecule_name')['potential_energy'])
energy_series.describe()


# In[ ]:


ax = energy_series.hist(bins=50)
_ = ax.set_xlabel("Potential Energy")


# ### Dipole Moment

# In[ ]:


dipoles.head()


# In[ ]:


len(dipoles)


# In[ ]:


dipole_series = pd.Series(dipoles.set_index('molecule_name')['scalar'])
dipole_series.describe()


# In[ ]:


ax = dipole_series.hist(bins=50)
_ = ax.set_xlabel("Dipole Moment")


# # ASE Database
# 
# A dataset object used in SchNetPack is built on an ASE database. Please see the document of the ASE database:
# 
# https://wiki.fysik.dtu.dk/ase/ase/db/db.html

# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


train_molecule_names = train.molecule_name.unique()
len(train_molecule_names)


# In[ ]:


from ase import Atoms
from ase.db import connect

def create_db(db_path, molecule_names):
    with connect(db_path) as db:
        for name in molecule_names:
            mol = molecules.get_group(name)
            atoms = Atoms(symbols=mol.atom.values,
                          positions=[(row.x,row.y,row.z) for row in mol.itertuples()])
            db.write(atoms, name=name,
                     potential_energy=energy_series.get(name, default=float('nan')),
                     scalar_dipole=dipole_series.get(name, default=float('nan'))
                    )


# In[ ]:


champs_path = 'CHAMPS_train.db'
dataset_size = len(train_molecule_names) # 20000
dataset_molecule_names = train_molecule_names[:dataset_size]
create_db(db_path=champs_path, molecule_names=dataset_molecule_names)


# In[ ]:


with connect(champs_path) as db:
    print(len(db))


# In[ ]:


import schnetpack

properties=['potential_energy', 'scalar_dipole']

dataset = dict()
for p in properties:
    dataset[p] = schnetpack.data.AtomsData(champs_path, properties=[p])


# In[ ]:


for p in properties:
    print(p, len(dataset[p]))


# # SchNet Model
# 
# Please see the SchNetPack API document:
# 
# https://schnetpack.readthedocs.io/en/stable/modules/index.html

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


def schnet_model(property):
    reps = rep.SchNet(n_interactions=6)
    if 'dipole' in property:
        print('use dipole moment')
        output = atm.DipoleMoment(n_in=128, predict_magnitude=True)
    else:
        output = atm.Atomwise()
    model = atm.AtomisticModel(reps, output)
    model = model.to(device)
    
    return model


# In[ ]:


def train_model(property, max_epochs=500):
    # split in train and val
    n_dataset = len(dataset[property])
    n_val = n_dataset // 10
    train_data, val_data, test_data = dataset[property].create_splits(n_dataset-n_val*2, n_val)
    train_loader = spk.data.AtomsLoader(train_data, batch_size=128, num_workers=2)
    val_loader = spk.data.AtomsLoader(val_data, batch_size=256, num_workers=2)

    # create model
    model = schnet_model(property)

    # create trainer
    opt = Adam(model.parameters(), lr=2e-4, weight_decay=1e-6)
    loss = lambda b, p: F.mse_loss(p["y"], b[property])
    metrics = [
        spk.metrics.MeanAbsoluteError(property, "y"),
        spk.metrics.RootMeanSquaredError(property, "y"),
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
    
    # load the best parameters
    model.load_state_dict(torch.load(property+'/output/best_model'))
    loader = spk.data.AtomsLoader(dataset, batch_size=256, num_workers=2)
    model.eval()
    
    # predict molecular properties
    targets = []
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }
            result = model(batch)
            targets += batch[property].squeeze().tolist()
            predictions += result['y'].squeeze().tolist()
    return targets, predictions


# In[ ]:


def show_predictions(dataset, property):
    targets, predictions = test_prediction(dataset, property)
    df_pred = pd.DataFrame()
    df_pred['Target'] = targets
    df_pred['Prediction'] = predictions
    df_pred.plot.scatter(x='Target', y='Prediction', title=property)


# # Results

# In[ ]:


used_test_data = dict()
for p in properties:
    print(p)
    used_test_data[p] = train_model(p, max_epochs=100)
    show_history(p)


# In[ ]:


for p in properties:
    show_predictions(used_test_data[p], p)


# In[ ]:


get_ipython().system('mv potential_energy/log/log.csv log_potential.csv')
get_ipython().system('mv scalar_dipole/log/log.csv log_dipole.csv')
get_ipython().system('mv potential_energy/output/best_model best_model_potential')
get_ipython().system('mv scalar_dipole/output/best_model best_model_dipole')
get_ipython().system('rm -r potential_energy')
get_ipython().system('rm -r scalar_dipole')


# In[ ]:


get_ipython().system('ls')

