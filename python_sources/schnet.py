#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
__print__ = print
def print(string, end = '', flush = True):
    os.system(f'echo \"{string}\"')
    __print__(string, end = end, flush = flush)


# In[ ]:


print('Hi')


# In[ ]:


get_ipython().system('pip install ase==3.17 schnetpack')


# In[ ]:


get_ipython().system('pip install  --no-cache-dir torch-scatter')


# In[ ]:


import numpy as np
import pandas as pd

import schnetpack as spk
import torch

from ase import Atoms
from ase.db import connect

import time

from torch_scatter import *


# In[ ]:


df = pd.read_csv('/kaggle/input/champs-scalar-coupling/train.csv')
df = df[df['type'] == '1JHC']
mol_names = df['molecule_name'].unique()
molecules = pd.read_csv('/kaggle/input/champs-scalar-coupling/structures.csv')
molecules = molecules.set_index('molecule_name')
molecules = molecules.loc[mol_names]
molecules = molecules.reset_index()
molecules = molecules.groupby('molecule_name')
df = df.groupby('molecule_name')


# In[ ]:


def one_hot_encoding(x, set):
    one_hot = [int(x == s) for s in set]
    if 0:
        if sum(one_hot)==0: print('one_hot_encoding() return NULL!', x, set)
    return one_hot


# In[ ]:


def create_db(db_path, molecule_names):
    with connect(db_path) as db:
        for name in molecule_names:
            mol = molecules.get_group(name)
            atoms = Atoms(symbols=mol.atom.values,
                          positions=[[row.x,row.y,row.z] for row in mol.itertuples()])
            #charges = mulliken_charges.get_group(name).mulliken_charge.values.reshape(-1,1)
            df_target = df.get_group(name)
            
            indexes = []
            targets = []
            types = []
            types_to_num = {'1JHC':0, '2JHH':1, '1JHN':2, '2JHN':3, '2JHC':4, '3JHH':5, '3JHC':6, '3JHN':7}
            for row in df_target.itertuples():
                indexes.append([row.atom_index_0 ,row.atom_index_1])
                targets.append(row.scalar_coupling_constant)
                types.append(types_to_num[row.type])
            db.write(atoms, name=name,
                     data={'indexes': indexes, 'targets': targets, 'types':types})


# In[ ]:


len(mol_names)


# In[ ]:


train_molecule_names = mol_names
champs_path = 'CHAMPS_train.db'
dataset_size =  len(mol_names)
dataset_molecule_names = train_molecule_names[:dataset_size]
create_db(db_path=champs_path, molecule_names=dataset_molecule_names)


# In[ ]:


import schnetpack
database = schnetpack.data.AtomsData(champs_path,
                properties=['indexes', 'targets', 'types'])


# In[ ]:


import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch import nn

import schnetpack as spk
import schnetpack.atomistic as atm
import schnetpack.representation as rep
from schnetpack.datasets import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from schnetpack.data import Structure
from schnetpack.atomistic import OutputModule

from torch.optim.lr_scheduler import ReduceLROnPlateau


# In[ ]:


# Batched index_select
def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy) # b x e x f
    return out


# In[ ]:


class Set2Set(torch.nn.Module):

    def softmax(self, x, index, num=None):
        x = x -  scatter_max(x, index, dim=0, dim_size=num)[0][index]
        x = x.exp()
        x = x / (scatter_add(x, index, dim=0, dim_size=num)[index] + 1e-16)
        return x

    def __init__(self, in_channel, processing_step=1):
        super(Set2Set, self).__init__()
        num_layer = 1
        out_channel = 2 * in_channel

        self.processing_step = processing_step
        self.in_channel  = in_channel
        self.out_channel = out_channel
        self.num_layer   = num_layer
        self.lstm = torch.nn.LSTM(out_channel, in_channel, num_layer)
        self.lstm.reset_parameters()

    def forward(self, x, batch_index):
        batch_size = batch_index.max().item() + 1

        h = (x.new_zeros((self.num_layer, batch_size, self.in_channel)),
             x.new_zeros((self.num_layer, batch_size, self.in_channel)))

        q_star = x.new_zeros(batch_size, self.out_channel)
        for i in range(self.processing_step):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, -1)

            e = (x * q[batch_index]).sum(dim=-1, keepdim=True) #shape = num_node x 1
            a = self.softmax(e, batch_index, num=batch_size)   #shape = num_node x 1
            r = scatter_add(a * x, batch_index, dim=0, dim_size=batch_size) #apply attention #shape = batch_size x ...
            q_star = torch.cat([q, r], dim=-1)

        return q_star


# In[ ]:


from schnetpack.data import Structure

class Scalar_Coupling(atm.Atomwise):
    def __init__(self):
        super(Scalar_Coupling, self).__init__(return_contributions=True, n_out = 128)
        
        #self.att = Attention(128)
        self.num_s2s = 6
        self.red = nn.Linear(256, 128)
        self.set2set = Set2Set(128, processing_step=self.num_s2s)
        
        self.predict = nn.Sequential(
            nn.Linear(512, 256),  #node_hidden_dim
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 8),
        )
        
    def forward(self, inputs):
        result = super().forward(inputs)
        #print(inputs['targets'].shape)
        embeds = result['yi']
        indexes = inputs['indexes'].long()
        targets = inputs['targets']
        mask = (targets != 0).reshape(indexes.shape[0] * indexes.shape[1])
        
        embed_index = torch.tensor([embeds.shape[1]*[i] for i in range(embeds.shape[0])]).reshape(embeds.shape[0]*embeds.shape[1]).to(device)
        
        pool = self.set2set(embeds.reshape((embeds.shape[0]*embeds.shape[1],embeds.shape[2])), embed_index)
        
        x1 = batched_index_select(embeds, 1,indexes[:,:,0])
        x2 = batched_index_select(embeds, 1,indexes[:,:,1])
        #print(embeds.shape)
        x = torch.cat((x1, x2), dim = 2)
        
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
        pool = pool.repeat(1, indexes.shape[1]).reshape(x.shape[0], -1)
        x = torch.cat((x, pool), dim = 1)
        '''
        query = self.red(query)
        query = query.unsqueeze(dim = 1)
        context = embeds.repeat(x.shape[1], 1, 1)
        pool, _ = self.att(query, context)
        '''
        #x = torch.cat()
        #x = torch.cat((query, pool), dim = 2).squeeze(dim = 1)
        #x = x[mask]
        #print(targets.shape)
        targets = targets.reshape(targets.shape[0]*targets.shape[1])
        targets = targets[mask]
        #print(targets.shape)
        #print(targets.shape)
        inputs['scalar_true'] = targets.unsqueeze(dim = 1)
        x = self.predict(x)
        
        types = inputs['types'].long()
        types = types.reshape(types.shape[0]*types.shape[1]).unsqueeze(dim = 1)
        #print(types)
        #print(types.shape)
        #print(x.shape)
        x = torch.gather(x, dim = 1, index = types)
        
        x = x[mask]
        
       # x = x.reshape((targets.shape[0], targets.shape[1]))
        #x = x.reshape((indexes.shape[0], indexes.shape[1]))
        result['scalar_pred'] = x
        #print(result['scalar_pred'].shape)
        #print(inputs['scalar_true'].shape)
        #print(inputs['scalar_true'].shape)
        #print(result['scalar_pred'].shape)
        
        return result


# In[ ]:


import os
import torch
import numpy as np


class Trainer:
    def __init__(self, model_path, model, loss_fn, optimizer,
                 train_loader, validation_loader, keep_n_checkpoints=3,
                 checkpoint_interval=10, validation_interval=1, hooks=[], loss_is_normalized=True):
        self.model_path = model_path
        self.checkpoint_path = os.path.join(self.model_path, 'checkpoints')
        self.best_model = os.path.join(self.model_path, 'best_model')
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.validation_interval = validation_interval
        self.keep_n_checkpoints = keep_n_checkpoints
        self.hooks = hooks
        self.loss_is_normalized = loss_is_normalized

        self._model = model
        self._stop = False
        self.checkpoint_interval = checkpoint_interval

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        #self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', min_lr = 0.00001)

        if os.path.exists(self.checkpoint_path):
            self.restore_checkpoint()
        else:
            os.makedirs(self.checkpoint_path)
            self.epoch = 0
            self.step = 0
            self.best_loss = float('inf')
            self.store_checkpoint()

    def _check_is_parallel(self):
        return True if isinstance(self._model, torch.nn.DataParallel) else False

    def _load_model_state_dict(self, state_dict):
        if self._check_is_parallel():
            self._model.module.load_state_dict(state_dict)
        else:
            self._model.load_state_dict(state_dict)

    @property
    def state_dict(self):
        state_dict = {
            'epoch': self.epoch,
            'step': self.step,
            'model': self._model.state_dict() if not self._check_is_parallel() else self._model.module.state_dict(),
            'best_loss': self.best_loss,
            'optimizer': self.optimizer.state_dict(),
            'hooks': [h.state_dict for h in self.hooks]
        }
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.step = state_dict['step']
        self.best_loss = state_dict['best_loss']
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self._load_model_state_dict(state_dict['model'])

        for h, s in zip(self.hooks, self.state_dict['hooks']):
            h.state_dict = s

    def store_checkpoint(self):
        chkpt = os.path.join(self.checkpoint_path,
                             'checkpoint-' + str(self.epoch) + '.pth.tar')
        torch.save(self.state_dict, chkpt)

        chpts = [f for f in os.listdir(self.checkpoint_path)
                 if f.endswith('.pth.tar')]
        if len(chpts) > self.keep_n_checkpoints:
            chpt_epochs = [int(f.split('.')[0].split('-')[-1]) for f in chpts]
            sidx = np.argsort(chpt_epochs)
            for i in sidx[:-self.keep_n_checkpoints]:
                os.remove(os.path.join(self.checkpoint_path, chpts[i]))

    def restore_checkpoint(self, epoch=None):
        if epoch is None:
            epoch = max([int(f.split('.')[0].split('-')[-1]) for f in os.listdir(self.checkpoint_path)])
        epoch = str(epoch)

        chkpt = os.path.join(self.checkpoint_path,
                             'checkpoint-' + str(epoch) + '.pth.tar')
        self.state_dict = torch.load(chkpt)

    def train(self, device):
        r"""
        Starts training of model on a specified device.

        Args:
            device (torch.torch.Device): device on which training takes place

        """
        self._stop = False

        for h in self.hooks:
            h.on_train_begin(self)

        try:
            while True:
                self.epoch += 1

                for h in self.hooks:
                    h.on_epoch_begin(self)

                if self._stop:
                    break

                # perform training epoch
                for train_batch in self.train_loader:
                    self.optimizer.zero_grad()

                    for h in self.hooks:
                        h.on_batch_begin(self, train_batch)

                    # move input to gpu, if needed
                    train_batch = {
                        k: v.to(device)
                        for k, v in train_batch.items()
                    }

                    result = self._model(train_batch)
                    loss = self.loss_fn(train_batch, result)

                    loss.backward()
                    self.optimizer.step()
                    self.step += 1

                    for h in self.hooks:
                        h.on_batch_end(self, train_batch, result, loss)

                    if self._stop:
                        break

                if self.epoch % self.checkpoint_interval == 0:
                    self.store_checkpoint()

                # validation
                if self.epoch % self.validation_interval == 0 or self._stop:
                    for h in self.hooks:
                        h.on_validation_begin(self)

                    val_loss = 0.
                    n_val = 0
                    for val_batch in self.validation_loader:
                        # append batch_size
                        vsize = list(val_batch.values())[0].size(0)
                        n_val += vsize

                        for h in self.hooks:
                            h.on_validation_batch_begin(self)

                        # move input to gpu, if needed
                        val_batch = {
                            k: v.to(device)
                            for k, v in val_batch.items()
                        }

                        val_result = self._model(val_batch)
                        val_batch_loss = self.loss_fn(val_batch, val_result).data.cpu().numpy()
                        if self.loss_is_normalized:
                            val_loss += val_batch_loss * vsize
                        else:
                            val_loss += val_batch_loss

                        for h in self.hooks:
                            h.on_validation_batch_end(self, val_batch, val_result)

                    # weighted average over batches
                    if self.loss_is_normalized:
                        val_loss /= n_val

                    if self.best_loss > val_loss:
                        self.best_loss = val_loss
                        state_dict = self._model.state_dict() if not self._check_is_parallel() else self._model.module.state_dict()
                        torch.save(state_dict, self.best_model)

                    for h in self.hooks:
                        h.on_validation_end(self, val_loss)
                
                #self.scheduler.step(val_loss)
                
                for h in self.hooks:
                    h.on_epoch_end(self)

                if self._stop:
                    break

            for h in self.hooks:
                h.on_train_ends(self)

        except Exception as e:
            for h in self.hooks:
                h.on_train_failed(self)

            raise e


# In[ ]:


def schnet_model():
    reps = rep.SchNet(n_interactions=6, cutoff=10.0, n_gaussians=100)
    #reps = rep.SchNet(n_interactions=6)
    output = Scalar_Coupling()
    model = atm.AtomisticModel(reps, output)
    model = model.to(device)
    return model


# In[ ]:


class MyHook(spk.train.CSVHook):
    """ Hook for logging to csv files.

            This class provides an interface to write logging information about the training process to csv files.

            Args:
                log_path (str): path to directory to which log files will be written.
                metrics (list): list containing all metrics to be logged. Metrics have to be subclass of spk.Metric.
                log_train_loss (bool): enable logging of training loss (default: True)
                log_validation_loss (bool): enable logging of validation loss (default: True)
                log_learning_rate (bool): enable logging of current learning rate (default: True)
                every_n_epochs (int): interval after which logging takes place (default: 1)
        """

    def __init__(self, log_path, metrics, log_train_loss=True,
                 log_validation_loss=True, log_learning_rate=True, every_n_epochs=1):
        log_path = os.path.join(log_path, 'log.csv')
        super(MyHook, self).__init__(log_path, metrics, log_train_loss,
                                      log_validation_loss, log_learning_rate)
        self._offset = 0
        self._restart = False
        self.every_n_epochs = every_n_epochs

    def on_train_begin(self, trainer):

        if os.path.exists(self.log_path):
            remove_file = False
            with open(self.log_path, 'r') as f:
                # Ensure there is one entry apart from header
                lines = f.readlines()
                if len(lines) > 1:
                    self._offset = float(lines[-1].split(',')[0]) - time.time()
                    self._restart = True
                else:
                    remove_file = True

            # Empty up to header, remove to avoid adding header twice
            if remove_file:
                os.remove(self.log_path)
        else:
            self._offset = -time.time()
            # Create the log dir if it does not exists, since write cannot create a full path
            log_dir = os.path.dirname(self.log_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

        if not self._restart:
            log = ''
            log += 'Time'

            if self.log_learning_rate:
                log += ',Learning rate'

            if self.log_train_loss:
                log += ',Train loss'

            if self.log_validation_loss:
                log += ',Validation loss'

            if len(self.metrics) > 0:
                log += ','

            for i, metric in enumerate(self.metrics):
                log += str(metric.name)
                if i < len(self.metrics) - 1:
                    log += ','

            with open(self.log_path, 'a+') as f:
                f.write(log + os.linesep)

    def on_validation_end(self, trainer, val_loss):
        if trainer.epoch % self.every_n_epochs == 0:
            ctime = time.time() + self._offset
            log = str(ctime)

            if self.log_learning_rate:
                log += ',' + str(trainer.optimizer.param_groups[0]['lr'])

            if self.log_train_loss:
                if hasattr(self._train_loss, "__iter__"):
                    train_string = ','.join([str(k) for k in self._train_loss])
                    log += ',' + train_string
                else:
                    log += ',' + str(self._train_loss)

            if self.log_validation_loss:
                if hasattr(val_loss, "__iter__"):
                    valid_string = ','.join([str(k) for k in val_loss])
                    log += ',' + valid_string
                else:
                    log += ',' + str(val_loss)

            if len(self.metrics) > 0:
                log += ','

            for i, metric in enumerate(self.metrics):
                m = metric.aggregate()
                if hasattr(m, "__iter__"):
                    log += ','.join([str(j) for j in m])
                else:
                    log += str(m)
                if i < len(self.metrics) - 1:
                    log += ','

            with open(self.log_path, 'a') as f:
                f.write(log + os.linesep)
                print(log + os.linesep)


# In[ ]:


def criterion(predict, truth):
    predict = predict.view(-1)
    truth   = truth.view(-1)
    assert(predict.shape==truth.shape)

    loss = torch.abs(predict-truth)
    loss = loss.mean()
    loss = torch.log(loss)
    return loss


# In[ ]:


def train_model(max_epochs=500):
    n_dataset = len(database)
    n_val = n_dataset // 20
    train_data, val_data, test_data = database.create_splits(n_dataset - 2*n_val, n_val, 'split')
    train_loader = spk.data.AtomsLoader(train_data, batch_size=64, num_workers=2,)
    val_loader = spk.data.AtomsLoader(val_data + test_data , batch_size=256, num_workers=2)
    
    model = schnet_model()

    # create trainer
    true_key = 'scalar_true'
    pred_key = 'scalar_pred'
    opt = Adam(model.parameters(), lr=0.0002)
    loss = lambda b, p: criterion(p[pred_key], b[true_key])
    metrics = [
        spk.metrics.MeanAbsoluteError(true_key, pred_key, name='MAE_scalar'),
        spk.metrics.RootMeanSquaredError(true_key, pred_key, name='RMSE_scalar'),
    ]
    hooks = [
        spk.train.MaxEpochHook(max_epochs),
        MyHook('log', metrics, every_n_epochs=1),
        #spk.train.CSVHook('log', metrics, every_n_epochs=1),
    ]
    #trainer = spk.train.Trainer('output', model, loss, opt, train_loader, val_loader, hooks=hooks)
    trainer = Trainer('output', model, loss, opt, train_loader, val_loader, hooks=hooks)

    # start training
    trainer.train(device)
    
    # evaluation
    model.load_state_dict(torch.load('best_model'))
    test_loader = spk.data.AtomsLoader(test_data, batch_size=1, num_workers=2)
    model.eval()
    
    return test_data


# In[ ]:


used_test_data = train_model(max_epochs=10000000)

