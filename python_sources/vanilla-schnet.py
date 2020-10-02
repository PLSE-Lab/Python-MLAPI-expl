#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install ase==3.17 schnetpack==0.2.1')


# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


import numpy as np
import pandas as pd
molecules = pd.read_csv('../input/champs-scalar-coupling/structures.csv')
molecules = molecules.groupby('molecule_name')
train = pd.read_csv('../input/champs-scalar-coupling/train.csv')
test = pd.read_csv('../input/champs-scalar-coupling/test.csv')
test['scalar_coupling_constant'] = -1

coupling_type = '1JHN'

train = train[train.type == coupling_type]
test = test[test.type == coupling_type]


# In[ ]:


len(train)


# In[ ]:


train.head()


# In[ ]:


len(test)


# In[ ]:


test.head()


# In[ ]:


train_scalar_couplings = train.groupby('molecule_name')
test_scalar_couplings = test.groupby('molecule_name')


# # ASE Database

# In[ ]:


from ase import Atoms
from ase.db import connect

def create_db(db_path, scalar_couplings, molecule_names):
    with connect(db_path) as db:
        for name in molecule_names:
            mol = molecules.get_group(name)
            atoms = Atoms(symbols=mol.atom.values,
                          positions=[(row.x,row.y,row.z) for row in mol.itertuples()])
            numbers = atoms.get_atomic_numbers()
            group = scalar_couplings.get_group(name)
            ai0 = group.atom_index_0.values
            ai1 = group.atom_index_1.values
            scc = group.scalar_coupling_constant.values
            ids = group.id.values
            for i, j, v, w in zip(ai0, ai1, scc, ids):
                new_numbers = numbers.copy()
                new_numbers[i] = 100 - new_numbers[i]
                new_numbers[j] = 100 - new_numbers[j]
                atoms.set_atomic_numbers(new_numbers)
                data = dict(scc=v)
                data[coupling_type+'_id'] = w
                db.write(atoms, name=name+'_H{}_C{}'.format(i,j), data=data)
                


# In[ ]:


properties=['scc', coupling_type+'_id']


# In[ ]:


import schnetpack

import sys
INT_MAX = sys.maxsize

dataset_size = INT_MAX

dataset_molecule_names = train.molecule_name.unique()
champs_path = 'CHAMPS_train.db' 
molecule_names = dataset_molecule_names[:dataset_size]
create_db(db_path=champs_path,
          scalar_couplings=train_scalar_couplings,
          molecule_names=molecule_names)
dataset = schnetpack.data.AtomsData(champs_path, properties=properties)


# In[ ]:


#dataset[30]


# In[ ]:


len(dataset)


# In[ ]:


dataset_molecule_names = test.molecule_name.unique()
test_champs_path = 'CHAMPS_test.db' 
test_molecule_names = dataset_molecule_names[:dataset_size]
create_db(db_path=test_champs_path,
          scalar_couplings=test_scalar_couplings,
          molecule_names=test_molecule_names)
test_dataset = schnetpack.data.AtomsData(test_champs_path, properties=properties)


# In[ ]:


#test_dataset[0]


# In[ ]:


len(test_dataset)


# # SchNet Model

# In[ ]:


import pandas as pd

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

import schnetpack as spk
import schnetpack.atomistic as atm
import schnetpack.representation as rep
from schnetpack.datasets import *

device = torch.device("cuda")
#device = torch.device("cpu")
torch.manual_seed(12345)
np.random.seed(12345)


# In[ ]:


# The original function comes from the following script:
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


import torch.nn as nn
from schnetpack.data import Structure

class MolecularOutput(atm.OutputModule):
    def __init__(self, property_name, n_in=128, n_out=1, aggregation_mode='sum',
                 n_layers=2, n_neurons=None,
                 activation=schnetpack.nn.activations.shifted_softplus,
                 outnet=None):
        super(MolecularOutput, self).__init__(n_in, n_out)
        self.property_name = property_name
        self.n_layers = n_layers
        self.create_graph = False
        
        if outnet is None:
            self.out_net = nn.Sequential(
                schnetpack.nn.base.GetItem('representation'),
                schnetpack.nn.blocks.MLP(n_in, n_out, n_neurons, n_layers, activation)
            )
        else:
            self.out_net = outnet
        
        if aggregation_mode == 'sum':
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=False)
        elif aggregation_mode == 'avg':
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=True)
            
    def forward(self, inputs):
        r"""
        predicts molecular property
        """
        atom_mask = inputs[Structure.atom_mask]

        yi = self.out_net(inputs)
        y = self.atom_pool(yi, atom_mask)

        result = {self.property_name: y}

        return result


# In[ ]:


def schnet_model():
    reps = rep.SchNet(n_atom_basis=128, n_filters=128, n_interactions=6)
    output = MolecularOutput('scc')
    model = atm.AtomisticModel(reps, output)
    model = model.to(device)
    return model


# In[ ]:


def train_model(max_epochs=500):
    # print configuration
    print('max_epochs:', max_epochs)
    
    # split in train and val
    n_dataset = len(dataset)
    n_val = n_dataset // 10
    train_data, val_data, test_data = dataset.create_splits(n_dataset-n_val*2, n_val, 'split')
    train_loader = spk.data.AtomsLoader(train_data, batch_size=128, num_workers=4, shuffle=True)
    val_loader = spk.data.AtomsLoader(val_data, batch_size=128, num_workers=4)

    # create model
    model = schnet_model()

    # create trainer
    output_key = "scc"
    target_key = "scc"
    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = MultiStepLR(opt, milestones=[15, 320], gamma=0.2)
    def loss(b, p): 
        return F.mse_loss(p[output_key], b[target_key])
    
    metrics = [
        spk.metrics.MeanAbsoluteError(target_key, output_key, name='MAE_scc'),
        spk.metrics.RootMeanSquaredError(target_key, output_key, name='RMSE_scc'),
    ]
    hooks = [
        spk.train.MaxEpochHook(max_epochs),
        spk.train.CSVHook('log', metrics, every_n_epochs=1),
        spk.train.LRScheduleHook(scheduler),
    ]
    trainer = spk.train.Trainer('output', model, loss,
                                opt, train_loader, val_loader, hooks=hooks)

    # start training
    trainer.train(device)
    
    # evaluation
    model.load_state_dict(torch.load('output/best_model'))
    test_loader = spk.data.AtomsLoader(test_data, batch_size=128, num_workers=4)
    model.eval()

    df = pd.DataFrame()
    df['metric'] = [
        'MAE_scc', 'RMSE_scc',
    ]
    df['training'] = evaluate_dataset(metrics, model, train_loader, device)
    df['validation'] = evaluate_dataset(metrics, model, val_loader, device)
    df['test'] = evaluate_dataset(metrics, model, test_loader, device)
    df.to_csv('output/evaluation.csv', index=False)
    display(df)
    
    return test_data


# In[ ]:


def show_history():
    df = pd.read_csv('log/log.csv')
    display(df.tail())
    
    _ = display(df[['MAE_scc', 'RMSE_scc']].plot())


# In[ ]:


def test_prediction(dataset):
    # create model
    model = schnet_model()
    
    # load best parameters
    model.load_state_dict(torch.load('output/best_model'))
    loader = spk.data.AtomsLoader(dataset, batch_size=128, num_workers=4)
    model.eval()
    
    # predict scalar coupling constants
    entry_id = []
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }
            result = model(batch)
            entry_id += batch[coupling_type+'_id'].long().view(-1).tolist()
            predictions += result['scc'].view(-1).tolist()
    return entry_id, predictions


# In[ ]:


def show_predictions(dataset, train_df):
    scc_id, scc = test_prediction(dataset)
    df_pred = pd.DataFrame()
    df_pred['Prediction'] = scc
    df_pred['id'] = scc_id
    df_pred = train_df.merge(df_pred, on='id', how='inner')
    display(df_pred.head())
    df_pred['Target'] = df_pred['scalar_coupling_constant']
    display(df_pred.plot.scatter(x='Target', y='Prediction', title=coupling_type, figsize=(5,5)))
    df_pred[['id', 'Target', 'Prediction']].to_csv('test_predictoins_{}.csv'.format(coupling_type), index=False)
    
    diff = (df_pred['Prediction'].values-df_pred['Target'].values)
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    df_eval = pd.DataFrame()
    df_eval['RMSE'] = [rmse]
    df_eval['MAE'] = [mae]
    df_eval['log(MAE)'] = [np.log(mae)]
    _ = display(df_eval)


# # Results

# In[ ]:


used_test_data = train_model(max_epochs=400)


# In[ ]:


show_history()


# In[ ]:


split = np.load('split.npz')


# In[ ]:


list(split.keys())


# In[ ]:


used_test_data =  dataset.create_subset(split['test_idx'])


# In[ ]:


show_predictions(used_test_data, train)


# In[ ]:


def make_submission():
    scc_id, scc = test_prediction(test_dataset)
    submission = pd.DataFrame()
    submission['id'] = scc_id
    submission['scalar_coupling_constant'] = scc
    submission.to_csv('submission_{}.csv'.format(coupling_type), index=False)


# In[ ]:


make_submission()

