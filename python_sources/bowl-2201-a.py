#!/usr/bin/env python
# coding: utf-8

# **limerobot's part**

# In[ ]:


import os
#print(list(os.walk('/kaggle/input')))
import sys
import torch
import json
import gc
import time
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from numba import jit 
from functools import partial
from scipy import optimize
from torch.utils.data import DataLoader
from pytorch_transformers.modeling_bert import BertConfig, BertEncoder

import warnings
warnings.filterwarnings(action='ignore')


TARGET = ['accuracy_group', 'num_correct', 'num_incorrect']
GAME_TARGET = ['accuracy_group_game', 'num_correct_game', 'num_incorrect_game']
#TARGET = ['accuracy_group']

from torch.utils.data import Dataset


class BowlDataset(Dataset):
    def __init__(self, cfg, df, sample_indices, aug=0.0, aug_p=0.5, padding_front=True, use_tta=False):
        self.cfg = cfg
        self.df = df.copy()    
        self.sample_indices = sample_indices
        self.seq_len = self.cfg.seq_len
        self.aug = aug
        self.aug_p = aug_p
        self.use_tta = use_tta
        self.padding_front = padding_front
         
        self.cate_cols = self.cfg.cate_cols
        self.cont_cols = self.cfg.cont_cols
        
        self.cate_df = self.df[self.cate_cols]
        self.cont_df = np.log1p(self.df[self.cont_cols])                
        if 'accuracy_group' in self.df:
            self.df['num_incorrect'][self.df['num_incorrect']==1] = 0.5
            self.df['num_incorrect'][self.df['num_incorrect']>1] = 1.0            
            self.df['num_correct'][self.df['num_correct']>1] = 1.0
            self.target_df = self.df[TARGET]
        else:
            self.target_df = None
            
        if 'accuracy_group_game' in self.df:
            self.df['num_incorrect_game'][self.df['num_incorrect_game']==1] = 0.5
            self.df['num_incorrect_game'][self.df['num_incorrect_game']>1] = 1.0            
            self.df['num_correct_game'][self.df['num_correct_game']>1] = 1.0
            self.target_game_df = self.df[GAME_TARGET]
        else:
            self.target_game_df = None
        
    def __getitem__(self, idx):
        indices = self.sample_indices[idx]
        
        seq_len = min(self.seq_len, len(indices))
        
        if self.aug > 0:
            if len(indices)>30:
                if np.random.binomial(1, self.aug_p) == 1:
                    cut_ratio = random.random()
                    if cut_ratio > self.aug:
                        cut_ratio = self.aug
                    #cut_ratio = self.aug
                    start_idx = max(int(len(indices)*cut_ratio), 30)
                    indices = indices[start_idx:]
                    seq_len = min(self.seq_len, len(indices))
        
        tmp_cate_x = torch.LongTensor(self.cate_df.iloc[indices].values)
        cate_x = torch.LongTensor(self.seq_len, len(self.cate_cols)).zero_()
        if self.padding_front:
            cate_x[-seq_len:] = tmp_cate_x[-seq_len:]
        else:
            cate_x[:seq_len] = tmp_cate_x[-seq_len:]
        
        tmp_cont_x = torch.FloatTensor(self.cont_df.iloc[indices].values)
        tmp_cont_x[-1] = 0
        cont_x = torch.FloatTensor(self.seq_len, len(self.cont_cols)).zero_()
        if self.padding_front:            
            cont_x[-seq_len:] = tmp_cont_x[-seq_len:]
        else:
            cont_x[:seq_len] = tmp_cont_x[-seq_len:]
        
        mask = torch.ByteTensor(self.seq_len).zero_()
        if self.padding_front:
            mask[-seq_len:] = 1
        else:
            mask[:seq_len] = 1
        
        if self.target_df is not None:
            target = torch.FloatTensor(self.target_df.iloc[indices[-1]].values)
            if target.sum() == 0:                
                target = torch.FloatTensor(self.target_game_df.iloc[indices[-1]].values)            
        else:
            target = 0
        
        return cate_x, cont_x, mask, target

    def __len__(self):
        return len(self.sample_indices)


class TransfomerModel(nn.Module):
    def __init__(self, cfg):
        super(TransfomerModel, self).__init__()
        self.cfg = cfg
        cate_col_size = len(cfg.cate_cols)
        cont_col_size = len(cfg.cont_cols)
        self.cate_emb = nn.Embedding(cfg.total_cate_size, cfg.emb_size, padding_idx=0)
        self.cate_proj = nn.Sequential(
            nn.Linear(cfg.emb_size*cate_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )        
        self.cont_emb = nn.Sequential(                
            nn.Linear(cont_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )
        
        self.config = BertConfig( 
            3, # not used
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.nlayers,
            num_attention_heads=cfg.nheads,
            intermediate_size=cfg.hidden_size,
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
        )
        self.encoder = BertEncoder(self.config)        
        
        def get_reg():
            return nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.target_size),            
        )        
        self.reg_layer = get_reg()
        
    def forward(self, cate_x, cont_x, mask):        
        batch_size = cate_x.size(0)
        
        cate_emb = self.cate_emb(cate_x).view(batch_size, self.cfg.seq_len, -1)
        cate_emb = self.cate_proj(cate_emb)     
        cont_emb = self.cont_emb(cont_x)
        
        seq_emb = torch.cat([cate_emb, cont_emb], 2)
        
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.config.num_hidden_layers
        
        encoded_layers = self.encoder(seq_emb, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]
        sequence_output = sequence_output[:, -1]
        
        pred_y = self.reg_layer(sequence_output)
        return pred_y

    
class LSTMATTNModel(nn.Module):
    def __init__(self, cfg):
        super(LSTMATTNModel, self).__init__()
        self.cfg = cfg
        cate_col_size = len(cfg.cate_cols)
        cont_col_size = len(cfg.cont_cols)
        self.cate_emb = nn.Embedding(cfg.total_cate_size, cfg.emb_size, padding_idx=0)        
        self.cate_proj = nn.Sequential(
            nn.Linear(cfg.emb_size*cate_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )        
        self.cont_emb = nn.Sequential(                
            nn.Linear(cont_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )
        
        self.encoder = nn.LSTM(cfg.hidden_size, 
                            cfg.hidden_size, 1, dropout=cfg.dropout, batch_first=True)
        
        self.config = BertConfig( 
            3, # not used
            hidden_size=cfg.hidden_size,
            num_hidden_layers=1,
            num_attention_heads=cfg.nheads,
            intermediate_size=cfg.hidden_size,
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
        )
        self.attn = BertEncoder(self.config)                 
        
        def get_reg():
            return nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.target_size),            
        )           
        self.reg_layer = get_reg()
        
    def forward(self, cate_x, cont_x, mask):        
        batch_size = cate_x.size(0)
        
        cate_emb = self.cate_emb(cate_x).view(batch_size, self.cfg.seq_len, -1)
        cate_emb = self.cate_proj(cate_emb) 
        cont_emb = self.cont_emb(cont_x)
        
        seq_emb = torch.cat([cate_emb, cont_emb], 2)        
        
        output, _ = self.encoder(seq_emb)
        
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.config.num_hidden_layers
        
        encoded_layers = self.attn(output, extended_attention_mask, head_mask=head_mask)        
        sequence_output = encoded_layers[-1]
        sequence_output = sequence_output[:, -1]
        pred_y = self.reg_layer(sequence_output)
        return pred_y

    
ENCODERS = {    
    'TRANSFORMER':TransfomerModel,
    'LSTMATTN':LSTMATTNModel,
}


def replace_4110_4100(df):
    rep_code4110_bool = (df['title']=='Bird Measurer (Assessment)')&(df['event_code']==4110)
    rep_code4100_bool = (df['title']=='Bird Measurer (Assessment)')&(df['event_code']==4100)
    df['event_code'][rep_code4110_bool] = 4100
    df['event_code'][rep_code4100_bool] = 5110


def get_agged_session(df):    
    event_code = pd.crosstab(df['game_session'], df['event_code'])
    event_id = pd.crosstab(df['game_session'], df['event_id'])
    event_num_correct = pd.pivot_table(df[(~df['correct'].isna())], index='game_session', columns='event_code', values='num_correct', aggfunc='sum')
    event_num_incorrect = pd.pivot_table(df[(~df['correct'].isna())], index='game_session', columns='event_code', values='num_incorrect', aggfunc='sum')
    event_accuracy = event_num_correct/(event_num_correct+event_num_incorrect[event_num_correct.columns])
    event_accuracy = event_accuracy.add_prefix('accuray_')    
    del event_num_correct, event_num_incorrect    
    
    event_round = pd.pivot_table(df[~df['correct'].isna()], index='game_session', columns='event_code', values='round', aggfunc='max')
    event_round = event_round.add_prefix('round_')
    
    print('max_game_time')    
    df['elapsed_time'] = df[['game_session', 'game_time']].groupby('game_session')['game_time'].diff()
    game_time = df.groupby('game_session', as_index=False)['elapsed_time'].agg(['mean', 'max']).reset_index()
    game_time.columns = ['game_session', 'mean_game_time', 'max_game_time']    
    df = df.merge(game_time, on='game_session', how='left')    
    event_max_game_time = pd.pivot_table(df, index='game_session', columns='event_code', values='elapsed_time', aggfunc='max')
    event_max_game_time = event_max_game_time.add_prefix('max_game_time_')
    del df['elapsed_time'] 
    
    print('session_extra_df')
    session_extra_df = pd.concat([event_code, event_id, event_accuracy, event_round], 1)
    session_extra_df.index.name = 'game_session'
    session_extra_df.reset_index(inplace=True)
    del event_code, event_id, event_accuracy, event_round
    
    print('session_df')
    session_df = df.drop_duplicates('game_session', keep='last').reset_index(drop=True)
    session_df['row_id'] = session_df.index
    session_df = session_df.merge(session_extra_df, how='left', on='game_session')
    return session_df

def gen_label(df):
    num_corrects = []
    for inst_id, one_df in df.groupby('installation_id'):
        one_df = one_df[(one_df['type']=='Assessment')&(one_df['event_code']==4100)]
        for game_session, title_df in one_df.groupby('game_session'):            
            num_correct = title_df['event_data'].str.contains('"correct":true').sum()
            num_incorrect = title_df['event_data'].str.contains('"correct":false').sum()            
            num_corrects.append([inst_id, game_session, num_correct, num_incorrect])
    label_df = pd.DataFrame(num_corrects, columns=['installation_id', 'game_session', 'num_correct', 'num_incorrect'])
    label_df['accuracy'] = label_df['num_correct'] / (label_df['num_correct']+label_df['num_incorrect'])
    label_df['accuracy_group'] = 3
    label_df['accuracy_group'][label_df['accuracy']==0.5] = 2    
    label_df['accuracy_group'][label_df['accuracy']<0.5] = 1
    label_df['accuracy_group'][label_df['accuracy']==0] = 0    
    return label_df


def extract_data_from_event_code(df, columns=['correct', 'round']):
    for col in columns:
        col_bool = df['event_data'].str.contains(col)
        df[col] = np.nan
        df[col][col_bool] = df['event_data'][col_bool].apply(lambda x: json.loads(x).get(col)).astype(float)

        
def get_train_sample_indices(df):
    sample_indices = []
    inst_indiecs = []    
    df_groups = df.groupby('installation_id').groups
    for inst_idx, indices in enumerate(df_groups.values()):
        one_df = df.iloc[indices].reset_index(drop=True)
        assessment_start_indices = one_df[(one_df['type']=='Assessment')&
                                          (one_df['accuracy_group']>=0)
                                         ].index
        for num, start_index in enumerate(assessment_start_indices):
            sample_indices.append( one_df.iloc[:start_index+1]['row_id'].tolist() )
            inst_indiecs.append(inst_idx)            
    return sample_indices, inst_indiecs

def choose_one(train_samples, train_groups, random_state):    
    random.seed(random_state)    
    group_dict = {}
    for row_id, group in zip(train_samples, train_groups):
        if group not in group_dict:
            group_dict[group] = []
        group_dict[group].append(row_id)
    new_train_samples = []    
    for v in group_dict.values():        
        new_train_samples.append(random.choice(v))         
    
    return np.array(new_train_samples)

def preprocessing(df, train_columns, mappers_dict, cate_offset, cate_cols, cont_cols, extra_cont_cls):
    print('preprocessing ... ')
    replace_4110_4100(df)
    
    print('generating label ...')
    label_df = gen_label(df)
    
    print('extract_data_from_event_code ...')
    extract_data_from_event_code(df)
    df['num_incorrect'] = np.where(df['correct']==0, 1, np.nan)
    df['num_correct'] = np.where(df['correct']==1, 1, np.nan)
    
    df['game_time'] = df['game_time'] // 1000
    
    df = get_agged_session(df)
    df = df.drop(['correct', 'round', 'num_correct', 'num_incorrect'], axis=1)
    
    df = df.merge(label_df, on=['game_session', 'installation_id'], how='left')
    
    samples, groups = get_train_sample_indices(df)
    
    df = df.append(pd.DataFrame(columns=train_columns))[train_columns]
    df = df.fillna(0)
    
    for col in cate_cols:
        df[col] = df[col].map(mappers_dict[col]).fillna(0).astype(int)
    
    print('preprocessing ... done')        
    return df, samples, groups

@jit
def qwk3(a1, a2, max_rat=3):
    assert(len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / (e+1e-08)


class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])

        return -qwk3(y, X_p)

    def fit(self, X, y):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])

    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']

def get_optimized_kappa_score(predictions, groundtruth):
    optR = OptimizedRounder()
    optR.fit(predictions, groundtruth)
    coefficients = optR.coefficients()
    #print(coefficients)
    temp_predictions = predictions.copy()
    temp_predictions[temp_predictions < coefficients[0]] = 0
    temp_predictions[(coefficients[0]<=temp_predictions)&(temp_predictions< coefficients[1])] = 1
    temp_predictions[(coefficients[1]<=temp_predictions)&(temp_predictions< coefficients[2])] = 2
    temp_predictions[(coefficients[2]<=temp_predictions)] = 3

    kappa_score = qwk3(temp_predictions, groundtruth)
    return kappa_score, coefficients 

class CFG:
    learning_rate=1.0e-4
    batch_size=64
    num_workers=4
    print_freq=100
    test_freq=1
    start_epoch=0
    num_train_epochs=1
    warmup_steps=30
    max_grad_norm=1000
    gradient_accumulation_steps=1
    weight_decay=0.01    
    dropout=0.2
    emb_size=100
    hidden_size=500
    nlayers=2
    nheads=8    
    device='cpu'
    seed=7
    ntta = [0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6] # TEST KAPPA_SCORE:0.5990772768904306
    wtta = [0.8]
CFG.wtta += [ (1-CFG.wtta[0])/(len(CFG.ntta)-1) for _ in range(len(CFG.ntta)-1)]


def compute_th_acc_gp(temp, coef):
    temp[temp < coef[0]] = 0
    temp[(coef[0]<=temp)&(temp< coef[1])] = 1
    temp[(coef[1]<=temp)&(temp< coef[2])] = 2
    temp[(coef[2]<=temp)] = 3    

def compute_acc_gp(pred):
    #batch_size = pred.size(0)
    pred = (3*pred[:, 0] - 2*pred[:, 1])    
    pred[pred < 0] = 0    
    return pred


def validate_function(valid_loader, model):
    model.eval()    
    
    predictions = []
    groundtruths = []
    for step, (cate_x, cont_x, mask, y) in enumerate(valid_loader):
        
        cate_x, cont_x, mask = cate_x.to(CFG.device), cont_x.to(CFG.device), mask.to(CFG.device)        
        
        k = 0.5
        with torch.no_grad():        
            pred = model(cate_x, cont_x, mask)
          
        # record accuracy
        pred_y = (1-k)*pred[:, 0] + (k)*compute_acc_gp(pred[:, 1:])
        predictions.append(pred_y.detach().cpu())        
        groundtruths.append(y[:, 0])

    predictions = torch.cat(predictions).numpy()
    groundtruths = torch.cat(groundtruths).numpy()
    
    return predictions, groundtruths


def test_function(valid_loader, model):
    model.eval()    
    
    predictions = []
    for step, (cate_x, cont_x, mask, _) in enumerate(valid_loader):
        
        cate_x, cont_x, mask = cate_x.to(CFG.device), cont_x.to(CFG.device), mask.to(CFG.device)        
        
        k = 0.5
        with torch.no_grad():        
            pred = model(cate_x, cont_x, mask)
          
        # record accuracy
        pred_y = (1-k)*pred[:, 0] + (k)*compute_acc_gp(pred[:, 1:])
        predictions.append(pred_y.detach().cpu())        

    predictions = torch.cat(predictions).numpy()
    
    return predictions


# MAIN #

os.environ['PYTHONHASHSEED'] = str(CFG.seed)
random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)    
torch.cuda.manual_seed(CFG.seed)
torch.backends.cudnn.deterministic = True       

test_df = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

[train_columns, mappers_dict, cate_offset, 
 cate_cols, cont_cols, extra_cont_cls] = torch.load('/kaggle/input/dsb2019-models/bowl_info_v70.pt')
test_df, test_samples, test_groups = preprocessing(test_df, train_columns, mappers_dict, cate_offset, 
                        cate_cols, cont_cols, extra_cont_cls)    

CFG.target_size = 3
CFG.total_cate_size = cate_offset
print(CFG.__dict__)
CFG.cate_cols = cate_cols
CFG.cont_cols = cont_cols+extra_cont_cls    

base_model_path_list = [
    ['bowl_v62.pt', [
        [1.0, f'/kaggle/input/dsb2019-models/v64/b-32_a-TRANSFORMER_e-100_h-500_d-0.2_l-2_hd-10_s-7_len-100_aug-0.5_da-bowl_v62.pt_k-0.pt'],            
    ]],
]

################################################
# find the coefficients
################################################
rand_seed_list = [7, 77, 777, 1, 2]
#rand_seed_list = [110798, 497274, 885651, 673327, 599183, 272713, 582394, 180043, 855725, 932850]    
sum_coefficients = 0
sum_cnt = 0
for _, base_model_paths in base_model_path_list:        
    for model_w, base_model_path in base_model_paths:        
        path = base_model_path.split('/')[-1]
        path = path.replace('bowl_', '')
        cfg_dict = dict([tok.split('-') for tok in path.split('_')])
        CFG.encoder = cfg_dict['a']
        CFG.seq_len = int(cfg_dict['len'])
        CFG.emb_size = int(cfg_dict['e'])
        CFG.hidden_size = int(cfg_dict['h'])
        CFG.nlayers = int(cfg_dict['l'])
        CFG.nheads = int(cfg_dict['hd'])
        CFG.seed = int(cfg_dict['s'])
        CFG.data_seed = int(cfg_dict['s'])

        for k in range(5):
            model = ENCODERS[CFG.encoder](CFG)
            model_path = base_model_path.replace('k-0', f'k-{k}')

            checkpoint = torch.load(model_path, map_location=CFG.device)        
            model.load_state_dict(checkpoint['state_dict'])
            model.to(CFG.device)
            print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))            

            for rand_seed in rand_seed_list:
                chosen_samples = choose_one(test_samples, test_groups, random_state=rand_seed)
                predictions = 0    
                for w, tta in zip(CFG.wtta, CFG.ntta):
                    padding_front = False if CFG.encoder=='LSTM' else True
                    valid_db = BowlDataset(CFG, test_df, chosen_samples, aug=tta, aug_p=1.0, 
                                           padding_front=padding_front, use_tta=True)
                    valid_loader = DataLoader(
                            valid_db, batch_size=CFG.batch_size, shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True)                
                    prediction, groundtruths = validate_function(valid_loader, model)
                    predictions += w*prediction                                            
                try:
                    valid_kappa, valid_coefficients = get_optimized_kappa_score(predictions, groundtruths)
                    print(f'k[{k}]-s2[{rand_seed}]: valid_kappa:{valid_kappa} - {valid_coefficients}') 
                    sum_coefficients += np.array(valid_coefficients)
                    sum_cnt += 1
                except Exception as e:
                    print(e)
                    print(f'k[{k}]-s2[{rand_seed}]: valid_kappa: Failed!')
                    pass
            del model
################################################
test_samples = list(test_df.groupby(['installation_id']).groups.values())    

coefficients = 0.5*sum_coefficients/sum_cnt + 0.5*np.array([0.53060865, 1.66266655, 2.31145611])       
print('=======================================')
print(f'coefficients - {coefficients}')
print('=======================================')

random.seed(CFG.seed)

submission_df = test_df.groupby('installation_id').tail(1)[['installation_id']]
submission_df['accuracy_group'] = 0

for _, base_model_paths in base_model_path_list:
    for model_w, base_model_path in base_model_paths:        
        path = base_model_path.split('/')[-1]
        path = path.replace('bowl_', '')
        cfg_dict = dict([tok.split('-') for tok in path.split('_')])
        CFG.encoder = cfg_dict['a']
        CFG.seq_len = int(cfg_dict['len'])
        CFG.emb_size = int(cfg_dict['e'])
        CFG.hidden_size = int(cfg_dict['h'])
        CFG.nlayers = int(cfg_dict['l'])
        CFG.nheads = int(cfg_dict['hd'])
        CFG.seed = int(cfg_dict['s'])
        CFG.data_seed = int(cfg_dict['s'])

        for k in range(5):
            model = ENCODERS[CFG.encoder](CFG)
            model_path = base_model_path.replace('k-0', f'k-{k}')

            checkpoint = torch.load(model_path, map_location=CFG.device)        
            model.load_state_dict(checkpoint['state_dict'])
            model.to(CFG.device)
            print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))            

            for w, tta in zip(CFG.wtta, CFG.ntta):
                padding_front = False if CFG.encoder=='LSTM' else True
                valid_db = BowlDataset(CFG, test_df, test_samples, aug=tta, aug_p=1.0, 
                                       padding_front=padding_front, use_tta=True)
                valid_loader = DataLoader(
                        valid_db, batch_size=CFG.batch_size, shuffle=False,
                        num_workers=CFG.num_workers, pin_memory=True)                
                predictions = test_function(valid_loader, model)
                submission_df['accuracy_group'] += w*predictions*model_w*(1/5)
            del model

submission_df['accuracy_group'] /= len(base_model_path_list)

limerobot_raw = submission_df['accuracy_group'].values
compute_th_acc_gp(submission_df['accuracy_group'], coefficients) 
submission_df['accuracy_group'] = submission_df['accuracy_group'].astype(int)
limerobot_int = submission_df['accuracy_group'].values
submission_df.to_csv('submission.csv', index=False)
print('done')


# In[ ]:





# **  KHA AND AGNIS'S PART**

# In[ ]:





# In[ ]:


# Version 1: memory error
# Version 2: change Kha's part to 0.553 (submitted on 19/01), resolve memory error
# Version 3: resolve not submission csv error. Revert back to 0.566 LB (Kha's part), using new dataset kha_tr_2001
# Version 4: change Kha's part with using true pseudo from test

import  datetime, itertools, zipfile, os, time, psutil, gc, copy, sys, warnings, json
from datetime import date, timedelta
import lightgbm as lgb, matplotlib.pyplot as plt, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error as mse
from numba import jit
from tqdm import tqdm_notebook as tqdm
from collections import Counter, OrderedDict
from sklearn.linear_model import Ridge
import gc
pd.set_option('display.max_colwidth', -1)
pd.set_option('max_columns', 100)
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
import scipy as sp
from functools import partial
from sklearn import metrics
from collections import Counter
os.listdir('/kaggle/input/')


# In[ ]:


INPUT_PATH = '/kaggle/input/data-science-bowl-2019/'
KHA_PATH = '/kaggle/input/bowl-kha-data/'

# train = pd.read_csv(INPUT_PATH+'train.csv')
test = pd.read_csv(INPUT_PATH+'test.csv')
train_labels = pd.read_csv(INPUT_PATH+'train_labels.csv')
specs = pd.read_csv(INPUT_PATH+'specs.csv')
sample_submission = pd.read_csv(INPUT_PATH+'sample_submission.csv')

RUN_MODE = 'public' if len(sample_submission)==1000 else 'private'


clip_len = {'Welcome to Lost Lagoon!': 19, 'Tree Top City - Level 1': 17, 'Ordering Spheres': 61,
 'Costume Box': 61, '12 Monkeys': 109, 'Tree Top City - Level 2': 25, "Pirate's Tale": 80, 'Treasure Map': 156,
 'Tree Top City - Level 3': 26, 'Rulers': 126, 'Magma Peak - Level 1': 20, 'Slop Problem': 60, 'Magma Peak - Level 2': 22,
 'Crystal Caves - Level 1': 18, 'Balancing Act': 72, 'Lifting Heavy Things': 118, 'Crystal Caves - Level 2': 24,
 'Honey Cake': 142, 'Crystal Caves - Level 3': 19, 'Heavy, Heavier, Heaviest': 61}

WORLD = {0: ['Welcome to Lost Lagoon!', 'Tree Top City - Level 1', 'Ordering Spheres', 'All Star Sorting',
 'Costume Box', 'Fireworks (Activity)', '12 Monkeys', 'Tree Top City - Level 2', 'Flower Waterer (Activity)',
 "Pirate's Tale", 'Mushroom Sorter (Assessment)', 'Air Show', 'Treasure Map', 'Tree Top City - Level 3',
 'Crystals Rule', 'Rulers', 'Bug Measurer (Activity)', 'Bird Measurer (Assessment)'], 
         1: ['Magma Peak - Level 1', 'Sandcastle Builder (Activity)', 'Slop Problem', 'Scrub-A-Dub', 
             'Watering Hole (Activity)', 'Magma Peak - Level 2', 'Dino Drink', 'Bubble Bath',
             'Bottle Filler (Activity)', 'Dino Dive', 'Cauldron Filler (Assessment)'], 
         2: ['Crystal Caves - Level 1', 'Chow Time', 'Balancing Act', 'Chicken Balancer (Activity)','Lifting Heavy Things', 
             'Crystal Caves - Level 2', 'Honey Cake', 'Happy Camel', 'Cart Balancer (Assessment)', 'Leaf Leader', 'Crystal Caves - Level 3', 
             'Heavy, Heavier, Heaviest', 'Pan Balance', 'Egg Dropper (Activity)', 'Chest Sorter (Assessment)']
        }

INFER_ASSE = {'Mushroom Sorter (Assessment)': 'Mushroom Sorter (Assessment)',
              'Bird Measurer (Assessment)': 'Bird Measurer (Assessment)',
              'Cauldron Filler (Assessment)': 'Cauldron Filler (Assessment)',
              'Cart Balancer (Assessment)' :'Cart Balancer (Assessment)',
              'Chest Sorter (Assessment)': 'Chest Sorter (Assessment)' , 
              'Welcome to Lost Lagoon!': 'Mushroom Sorter (Assessment)',
              'Tree Top City - Level 1': 'Mushroom Sorter (Assessment)',
              'Ordering Spheres': 'Mushroom Sorter (Assessment)',
              'All Star Sorting': 'Mushroom Sorter (Assessment)',
              'Costume Box': 'Mushroom Sorter (Assessment)',
              'Fireworks (Activity)': 'Mushroom Sorter (Assessment)',
              '12 Monkeys': 'Mushroom Sorter (Assessment)',
              'Tree Top City - Level 2': 'Mushroom Sorter (Assessment)',
              'Flower Waterer (Activity)': 'Mushroom Sorter (Assessment)',
              "Pirate's Tale": 'Mushroom Sorter (Assessment)',
              'Air Show': 'Bird Measurer (Assessment)',
              'Treasure Map': 'Bird Measurer (Assessment)',
              'Tree Top City - Level 3': 'Bird Measurer (Assessment)',
              'Crystals Rule': 'Bird Measurer (Assessment)',
              'Rulers': 'Bird Measurer (Assessment)',
              'Bug Measurer (Activity)': 'Bird Measurer (Assessment)',
              'Magma Peak - Level 1': 'Cauldron Filler (Assessment)',
              'Sandcastle Builder (Activity)': 'Cauldron Filler (Assessment)',
              'Slop Problem': 'Cauldron Filler (Assessment)',
              'Scrub-A-Dub': 'Cauldron Filler (Assessment)',
              'Watering Hole (Activity)': 'Cauldron Filler (Assessment)',
              'Magma Peak - Level 2': 'Cauldron Filler (Assessment)',
              'Dino Drink': 'Cauldron Filler (Assessment)',
              'Bubble Bath': 'Cauldron Filler (Assessment)',
              'Bottle Filler (Activity)': 'Cauldron Filler (Assessment)',
              'Dino Dive': 'Cauldron Filler (Assessment)',
              'Crystal Caves - Level 1': 'Cart Balancer (Assessment)',
              'Chow Time': 'Cart Balancer (Assessment)',
              'Balancing Act': 'Cart Balancer (Assessment)',
              'Chicken Balancer (Activity)': 'Cart Balancer (Assessment)',
              'Lifting Heavy Things': 'Cart Balancer (Assessment)',
              'Crystal Caves - Level 2': 'Cart Balancer (Assessment)',
              'Honey Cake': 'Cart Balancer (Assessment)',
              'Happy Camel': 'Cart Balancer (Assessment)',
              'Leaf Leader': 'Chest Sorter (Assessment)', 
              'Crystal Caves - Level 3' : 'Chest Sorter (Assessment)',
              'Heavy, Heavier, Heaviest': 'Chest Sorter (Assessment)', 
              'Pan Balance': 'Chest Sorter (Assessment)',
              'Egg Dropper (Activity)': 'Chest Sorter (Assessment)', 
        }

def title_to_code(t):
    return list(INFER_ASSE.keys()).index(t)

def title_to_world(t):
    if t in WORLD[0]: return 0
    elif t in WORLD[1]: return 1
    elif t in WORLD[2]: return 2
    else: return -1

#train['timestamp'] = pd.to_datetime(train['timestamp'])
test['timestamp'] = pd.to_datetime(test['timestamp'])   

clip_titles = test[test.type=='Clip'].title.unique().tolist()
game_titles = test[test.type=='Game'].title.unique().tolist()
activity_titles = test[test.type=='Activity'].title.unique().tolist()
assessment_titles = test[test.type=='Assessment'].title.unique().tolist()
all_titles = assessment_titles + game_titles + activity_titles + clip_titles


# In[ ]:


@jit
def qwk(a1, a2, max_rat=3):
    assert(len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)
    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))
    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)
    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)
    e = e / a1.shape[0]
    return 1 - o / e

def eval_qwk_lgb(y_true, y_pred):
    """
    Fast cappa eval function for lgb.
    """

    y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
    return 'cappa', qwk(y_true, y_pred), True


def preds_to_int(prin, t1=1.12232214, t2=1.73925866, t3=2.22506454):
    pr1 = np.copy(prin)
    pr1[pr1 <= t1] = 0
    pr1[np.where(np.logical_and(pr1 > t1, pr1 <= t2))] = 1
    pr1[np.where(np.logical_and(pr1 > t2, pr1 <= t3))] = 2
    pr1[pr1 > t3] = 3
    return pr1


def eval_qwk_lgb_regr(y_true, y_pred_in):
    """
    Fast cappa eval function for lgb.
    """
    return 'cappa', qwk(y_true, preds_to_int(y_pred_in)), True

def sample_data(data):
    new_data, indices = [], []
    for ins_id, sub_df in data.groupby('id', sort=False):
        nb_asse = len(sub_df)
        s = np.random.choice(list(range(nb_asse)))
        new_data.append(sub_df.iloc[s].values.tolist())
        indices.append(sub_df.index[s])
    new_data = pd.DataFrame(columns=data.columns, data=new_data)
    return new_data, np.array(indices)

def count(_ses, value, field='event_code'):
    if type(value)==int: return _ses.query(field + '== ' + str(value)).shape[0]
    else:
        cnt = 0
        for v in value: cnt += _ses.query(field + '== ' + str(v)).shape[0]
        return cnt

def get_game_stats(_ses, title=None): 
    all_attempts = _ses.query('event_code == 2030') #round end event
    if all_attempts.shape[0] > 0:
        gmisses = all_attempts['event_data'].apply(lambda x: parsefields(x, '"misses":')).values
        glevel = all_attempts['event_data'].apply(lambda x: parsefields(x, '"level":')).values
        ground = all_attempts['event_data'].apply(lambda x: parsefields(x, '"round":')).values
        nb_attempts = len(gmisses)
        nb_0_miss, nb_1_3_miss = len(gmisses[gmisses==0]), len(gmisses[(gmisses>=1)&(gmisses<=3)])
        nb_3_10_miss, nb_more_10_miss = len(gmisses[(gmisses>=3)&(gmisses<10)]), len(gmisses[gmisses>=10])
        ratio_0_miss, ratio_1_3_miss = nb_0_miss/nb_attempts, nb_1_3_miss/nb_attempts
        ratio_3_10_miss, ratio_more_10_miss = nb_3_10_miss/nb_attempts, nb_more_10_miss/nb_attempts
        stats = [ratio_0_miss, ratio_1_3_miss, ratio_3_10_miss, ratio_more_10_miss, 
               nb_0_miss, nb_1_3_miss, nb_3_10_miss, nb_more_10_miss]
    else: stats = [np.NaN for i in range(8)]
    names = ['game_ratio_0_miss', 'game_ratio_1_3_miss', 'game_ratio_3_10_miss', 'game_ratio_more_10_miss',
            'game_nb_0_miss', 'game_nb_1_3_miss', 'game_nb_3_10_miss', 'game_nb_more_10_miss']
    modes = ['avg', 'avg', 'avg', 'avg', 'sum', 'sum', 'sum', 'sum']
    return stats, names, modes

    
def search_attempts(_ses, event_code_value):
    subses = _ses.query('event_code == '+str(event_code_value))
    true_attempts = subses['event_data'].str.contains('"correct":true').sum()
    false_attempts = subses['event_data'].str.contains('"correct":false').sum()
    return true_attempts, false_attempts

def acc_to_accgrp(acc, array_mode=False): 
    if array_mode:
        acc_grp = np.ones(len(acc))
        acc_grp[acc==0] = 0
        acc_grp[acc==1] = 3
        acc_grp[(acc>=0.5)&(acc<1)] = 2
        return acc_grp
        
    else:
        if acc is np.NaN: return np.NaN
        elif acc == 0: return 0
        elif acc == 1: return 3
        elif acc >= 0.5: return 2
        else: return 1
    

def mediatype_to_prefix(t):
    if t=='Assessment': return 'asse_'
    if t=='Game': return 'game_'
    if t=='Clip': return 'clip_'
    if t=='Activity': return 'acti_'
    
SPECIFIC_TITLES = assessment_titles + activity_titles + game_titles

    
class OptimizedRounder(object):
    def __init__(self, mode='accuracy_group'):
        self.coef_ = 0
        self.mode=mode

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:  X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]: X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]: X_p[i] = 2
            else: X_p[i] = 3
        ll = qwk(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [1.12, 1.72, 2.24] if self.mode=='accuracy_group' else [0.32, 0.48, 0.7]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='Nelder-Mead',options={'maxiter':100000})

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]: X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]: X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:  X_p[i] = 2
            else:  X_p[i] = 3
        return X_p

    def coefficients(self):
        return self.coef_['x']

    
def get_labels_sampled_distribution(y_true, group_idx, sample=True):
    distr = np.zeros(4)
    if sample:
        N = 200
        for t in range(N):
            idx_to_score = []
            for ins_idx in group_idx: idx_to_score.append(np.random.choice(ins_idx))
            y_sampled = y_true[idx_to_score]
            distr += np.histogram(y_sampled, bins=4)[0]/N
        
    else: distr = np.histogram(y_true, bins=4)[0]
    distr = distr/np.sum(distr)
    print('Distribution:', distr)
    
def get_resample_score(y_preds, y_true, group_idx, times=1000, metric='qwk'):
    scores = []
    for t in range(times):
        idx_to_score = []
        for ins_idx in group_idx: idx_to_score.append(np.random.choice(ins_idx))
        if metric=='qwk': scores.append(qwk(y_preds[idx_to_score], y_true[idx_to_score]))
        elif metric=='rmse': scores.append(mse(y_preds[idx_to_score], y_true[idx_to_score]))
    return np.mean(scores), np.std(scores)

def get_idx_group(df):
    group_idx = []
    for ins_id, sub_df in df.groupby('id', sort=False):
        idx = sub_df.index.values.tolist()
        group_idx.append(idx)
    return np.array(group_idx)

def string_to_preds(string):
    preds = []
    for i, s in enumerate(string): preds.append(int(s))
    return np.array(preds)

def preds_to_string(preds):
    pred_string = ''
    for p in preds: pred_string += str(int(p))
    return pred_string

def regr_to_accgrp(x, thres=[0.32, 0.48, 0.7]):
    acc_grp = np.zeros(len(x))
    acc_grp[x<=thres[0]] = 0
    acc_grp[(x>thres[0])&(x<=thres[1])] = 1
    acc_grp[(x>thres[1])&(x<thres[2])] = 2
    acc_grp[x>=thres[2]] = 3
    return acc_grp

def parsefields(inp, fld):
    try:
        beg = inp.index(fld)
        end = inp.index(',', beg + len(fld))
        return int(inp[beg + len(fld):end])
    except:
        return 0
    
def get_mediatype(title):
    if title in assessment_titles: return 'Assessment'
    elif title in game_titles: return 'Game'
    elif title in clip_titles: return 'Clip'
    elif title in activity_titles: return 'Activity'
    
def get_train_weights(x, weights_coef=[1.6533593 , 1.09794629, 0.87330317, 0.77491961, 0.57421875, 0.47301587]):
    if np.isnan(x): x=0
    return weights_coef[int(min(x, len(weights_coef)-1))]


def sum_count(dct, keys): # sum values of all keys in dict dct
    return np.sum([dct[k] for k in keys])

def encode_title(t):
    if t == 'Mushroom Sorter (Assessment)': return 2 
    if t == 'Bird Measurer (Assessment)': return 0
    if t == 'Cauldron Filler (Assessment)': return 3
    if t == 'Cart Balancer (Assessment)': return 4
    if t == 'Chest Sorter (Assessment)': return 1
    
def encode_title_old(t):
    if t == 'Mushroom Sorter (Assessment)': return 4
    if t == 'Bird Measurer (Assessment)': return 0
    if t == 'Cauldron Filler (Assessment)': return 2
    if t == 'Cart Balancer (Assessment)': return 1
    if t == 'Chest Sorter (Assessment)': return 3
    
SPECIFIC_COUNT = {"Air Show": [3020,3021,3121],
                  "All Star Sorting": [2030, 3020, 4070],
                  "Bird Measurer (Assessment)": [3121, 4040],
                  "Bottle Filler (Activity)": [2020],
                  "Bubble Bath": [2080, 4220], 
                  "Bug Measurer (Activity)": [4070],
                  "Cauldron Filler (Assessment)": [3120, 4040],
                  "Chest Sorter (Assessment)": [3020, 3021, 3120, 4020, 4025],
                  "Chicken Balancer (Activity)": [4020, 4022],
                  "Chow Time": [3121],
                  "Crystals Rule": [3110, 3120],
                  "Dino Dive": [2020],
                  "Dino Drink": [2030],
                  "Fireworks (Activity)": [4030, 4070],
                  "Happy Camel": [2030,3110,3120,4035,4095],
                  "Leaf Leader": [2020, 3021, 4070],
                  "Pan Balance": [2030, 3020, 3120],
                  "Sandcastle Builder (Activity)": [4020],
                  "Scrub-A-Dub": [2000, 2050, 2083, 4010, 4020]
                 }

def get_train_weights(id_arr, weight_coefs=np.array([1.6533593 , 1.09794629, 0.87330317, 0.77491961, 0.57421875, 0.47301587])):
    if weight_coefs=='manual':
        distr = []
        for ins_id in np.unique(id_arr):
            nb_asse = len(id_arr[id_arr==ins_id])
            distr.append(nb_asse-1)
        distr = np.array(distr)
        nb_prior_train = [distr[distr==k].shape[0]/distr.shape[0] for k in range(5)]
        nb_prior_train.append(distr[distr>=5].shape[0]/distr.shape[0])

        nb_prior_private = [0.44, 0.192, 0.11, 0.066, 0.04, 0.12]
        weight_coefs = np.array(nb_prior_private) / np.array(nb_prior_train)
    print('WEIGHTS COEF:', weight_coefs)
    
    prev_row = id_arr[0]
    x = 0
    weights = []
    for row in id_arr:
        if row != prev_row: 
            prev_row = row
            x = 0
        weights.append(weight_coefs[x])
        x += 1
        if x > 5: x = 5
            
    return weights


def get_target_col(title):
    t = title[:4]
    if t == 'Bird': return 'Bird_Measurer__Assessment_asse_manual_accuracy_grpLast'
    if t == 'Ches': return 'Chest_Sorter__Assessment_asse_manual_accuracy_grpLast'
    if t == 'Caul': return 'Cauldron_Filler__Assessment_asse_manual_accuracy_grpLast'
    if t == 'Mush': return 'Mushroom_Sorter__Assessment_asse_manual_accuracy_grpLast'
    if t == 'Cart': return 'Cart_Balancer__Assessment_asse_manual_accuracy_grpLast'
    

class Feature_Buffer:
    
    def __init__(self):
        self.buf, self.last, self.count = {}, {}, {}
        self.basic_features, self.specific_features = [], []
        self.specific_titles = SPECIFIC_TITLES
        self.previous_title = {0: np.NaN, 1: np.NaN, 2:np.NaN, 3:np.NaN, 4: np.NaN}
        self.previous_acc = {0: np.NaN, 1: np.NaN, 2:np.NaN, 3:np.NaN, 4: np.NaN}
        self.previous_misclickratio = {0: np.NaN, 1: np.NaN, 2:np.NaN, 3:np.NaN, 4: np.NaN}
        self.previous_timespent = {0: np.NaN, 1: np.NaN, 2:np.NaN, 3:np.NaN, 4: np.NaN}
        self.last_asse_timestamp = np.NaN
        self.lastaccgroup = {'Mushroom Sorter (Assessment)': np.NaN,   
                           'Bird Measurer (Assessment)':np.NaN,   
                           'Cauldron Filler (Assessment)': np.NaN,
                           'Cart Balancer (Assessment)': np.NaN,
                           'Chest Sorter (Assessment)': np.NaN }
        self.maxgamedur = 0

   
    def init_new_feat(self, f, category='basic', initial=np.NaN):
        self.buf[f], self.last[f], self.count[f] = initial, initial, initial
        if category=='basic': self.basic_features.append(f)
        elif category=='specific': self.specific_features.append(f)
                
    def get_mediatype(self, title):
        if title in assessment_titles: return 'Assessment'
        elif title in game_titles: return 'Game'
        elif title in clip_titles: return 'Clip'
        elif title in activity_titles: return 'Activity'
        
    def clear(self, feature_name):
        self.buf[feature_name], self.last[feature_name], self.count[feature_name] = np.NaN, np.NaN, np.NaN

    def update(self, title, feature_name, value, mode='avg',specific=True,initial=np.NaN):
        world_prefix = 'w' + str(title_to_world(title)) + '_'
        feats = [feature_name, world_prefix+feature_name]
        for f in feats:
            if f not in self.basic_features: self.init_new_feat(f, initial=initial)
        
        if specific:
            if title in self.specific_titles: 
                f2 = title + feature_name
                if f2 not in self.specific_features: self.init_new_feat(f2, category='specific',initial=initial)
                feats.append(f2)
        
        for f in feats:
            if value is not np.NaN:
                if mode=='avg': 
                    self.buf[f] = (self.buf[f]*self.count[f] + value)/(self.count[f]+1) if self.buf[f] is not np.NaN else value
                if mode=='sum':  
                    self.buf[f] = self.buf[f]+value if self.count[f] is not np.NaN else value
                self.count[f] = self.count[f]+1 if self.count[f] is not np.NaN else 1      
            self.last[f] = value
            
    def update_count_simple(self, f, value):
        if f not in self.specific_features: self.init_new_feat(f, category='specific',initial=0)
        self.buf[f] = self.buf[f]+value if self.count[f] is not np.NaN else value
        self.last[f] = value
        
            
    def make_features(self, title, begin_timestamp):
        world_prefix = 'w' + str(title_to_world(title)) + '_'
        feature_values, feature_names = [], []
        # Basic features
        for f in self.basic_features: 
            feature_values += [self.buf[f]]
            feature_names += [f]
            if world_prefix in f:
                feature_values += [self.buf[f]]
                feature_names += ['sameworld_'+f.split(world_prefix)[-1]]
            
        # Specific features to each session title
        for f in self.specific_features:
            if 'count_dif' in f: continue
            feature_values += [self.buf[f], self.last[f]]
            feature_names += [f, f+'Last']
            if title in f:
                feature_values += [self.buf[f], self.last[f]]
                feature_names += ['sameasse_'+f.split(title)[-1], 'sameasse_'+f.split(title)[-1]+'Last']
                
        # Extra features
        feature_values += [self.buf['count_4020']/self.buf['count_allevents'] if 'count_4020' in self.buf.keys() else 0, # C
                           self.buf['count_4220']/self.buf['count_allevents'] if 'count_4220' in self.buf.keys() else 0,
                           self.buf['game_count']/self.buf['asse_count'] if 'asse_count' in self.buf.keys() and 'game_count' in self.buf.keys() else np.NaN, # Games divide Assessments
                           title_to_world(title), 
                           1 if self.last_asse_timestamp is np.NaN else (begin_timestamp - self.last_asse_timestamp).total_seconds(),
                           self.maxgamedur,
                           encode_title(title),
                           encode_title_old(title)
                          ]
        
        feature_names += ['ratio_4020', 'ratio_4220', 'ratio_gamecount_asseacount', 'world_enc', 
                          'time_from_prev_asse','maxgamedur', 'title_enc', 'title_enc_old'
                         ]
        
        
        return feature_values, feature_names
    
class RNN_Feature:
    def __init__(self):
        self.feats, self.titles, self.targets, self.ses_id = [], [], [], []
        self.cur_feat = {}
    
    def register(self, title, target, ses_id):
        self.titles.append(title_to_code(title))
        self.feats.append(self.cur_feat)
        self.targets.append(target)
        self.ses_id.append(ses_id)
        self.cur_feat = {}
        
    def add(self, value, field):
        self.cur_feat[field] = value


# In[ ]:


train_ins_with_labels = train_labels.installation_id.unique()
train_game_sessions_with_labels = train_labels.game_session.unique()

if RUN_MODE == 'private':

    for df, mode in zip([test], ['test']):
        print('Building', mode, 'features...')

        feat_df =  pd.DataFrame(columns=['id', 'ses_id', 'title', 'acc', 'acc_g'])
        feat_rnn = {}

        for i, (ins_id, sub_df) in tqdm(enumerate(df.groupby('installation_id', sort=False))):
            features = Feature_Buffer()
    #         R = RNN_Feature()
            all_idx = sub_df.index.values
            nb_sessions = len(sub_df.groupby('game_session', sort=False))

            for i_ses, (ses_id, ses) in enumerate(sub_df.groupby('game_session', sort=False)):

                ses_type, ses_title = ses['type'].iloc[0], ses['title'].iloc[0]
                dur = (ses.iloc[-1].timestamp - ses.iloc[0].timestamp).seconds
                dur = min(dur, 1800)

                # Make new sample
                acc, acc_g = np.NaN, np.NaN
                if ses_type == 'Assessment' or i_ses==nb_sessions-1: 
                    if ses_id in train_game_sessions_with_labels:
                        acc = train_labels[train_labels.game_session==ses_id].accuracy.values[0]
                        acc_g = train_labels[train_labels.game_session==ses_id].accuracy_group.values[0]
                        label_type = 'train_real'
                    elif ins_id in train_ins_with_labels:
                        acc, acc_g = np.NaN, np.NaN
                        label_type = 'train_interpolated'
                    else: 
                        acc, acc_g = np.NaN, np.NaN
                        if mode=='train': label_type = 'train_pseudo' 
                        if mode=='test':
                            if i_ses==nb_sessions-1: label_type = 'test_final'
                            else: label_type = 'test_middle'

                    tt = INFER_ASSE[ses_title]
                    feat_list, feat_names = features.make_features(title=tt, begin_timestamp=ses.iloc[0].timestamp)
                    feats = feat_list + [ins_id, ses_id, tt, acc, acc_g, label_type]
                    col_names = feat_names + ['id', 'ses_id', 'title', 'acc', 'acc_g', 'label_type']
                    for c in col_names: 
                        if c not in feat_df.columns: feat_df[c] = np.NaN
                    feat_df.loc[len(feat_df), col_names] = feats

                    # Clear dif buffer
                    for f in features.basic_features:
                        if 'count_dif' in f: features.clear(f)

                # Feature crafting
                if ses_type == 'Assessment':                   
                    all_codes = [2000,2025,2030,2035,3020,3021,3120,3121,4020,4025,4030,4035,4040,4070,4090,4100,3110]
                    cc = {} # code count
                    for c in all_codes: 
                        cc[c] = count(ses, c)
                        features.update(ses_title, 'count_'+str(c), cc[c], mode='sum', specific=False)
                        features.update(ses_title, 'count_dif_'+str(c), cc[c], mode='sum', specific=False)
    #                     R.add(min(cc[c]/200, 1), str(c))

                    win_code = 4100 if ses_title!='Bird Measurer (Assessment)' else 4110
                    true_att, false_att = search_attempts(ses, win_code)
                    true_placement, false_placement = search_attempts(ses, 4020)

                    codes_ = [2025, 4025, 4030, 4040, 4020]
                    cnt_actions = sum_count(cc, codes_)
                    misclick = sum_count(cc, [4070, 4035]) 
                    features.update(ses_title, 'asse_misclickdrag_ratio', misclick/ (cnt_actions + 1))
    #                 R.add(misclick/ (cnt_actions + 1), 'misclickdrag_ratio')
                    features.update(ses_title, 'asse_misclickdrag', misclick)
                    features.update(ses_title, 'asse_seekhelp' , cc[4090] /  (cnt_actions + 1))
                    features.update(ses_title, 'asse_finish_round' ,  cc[2030] /  (cnt_actions + 1))
    #                 R.add(cc[2030] /  (cnt_actions + 1) , 'finish_round')
                    features.update(ses_title, 'asse_action_freq' ,  cnt_actions / (dur + 0.01)   )
    #                 R.add(cnt_actions / (dur + 0.01), 'action_freq')
                    features.update(ses_title, 'asse_cnt_actions', cnt_actions)
    #                 R.add(cnt_actions , 'cnt_actions')
                    cnt_3021, cnt_3020 = cc[3021], cc[3020]
                    features.update(ses_title, 'asse_success_ratio' ,  cnt_3021 / (cnt_3021+cnt_3020) if cnt_3021+cnt_3020>0 else 0)
    #                 R.add(cnt_3021 / (cnt_3021+cnt_3020) if cnt_3021+cnt_3020>0 else 0 , 'success_ratio')
                    features.update(ses_title, 'asse_true_att_freq' , true_att/(cnt_actions+1))
                    features.update(ses_title, 'asse_true_att' , true_att, mode='sum')
    #                 R.add(true_att , 'true_att')
    #                 R.add(false_att , 'false_att')
                    features.update(ses_title, 'asse_false_att' , false_att, mode='sum')
                    features.update(ses_title, 'asse_false_att_freq' , false_att/(cnt_actions+1))
                    features.update(ses_title, 'asse_true_att_freq_2' , true_att/(dur + 0.01))
                    manual_accuracy = true_att/(true_att + false_att) if true_att + false_att!=0 else 0
                    features.update(ses_title, 'asse_correctplacement_ratio', true_placement/(true_placement+false_placement+1))
                    features.update(ses_title, 'asse_correctplacement_ratio2', true_placement/(true_placement+cnt_actions+1))
                    features.update(ses_title, 'asse_manual_accuracy' , manual_accuracy)
                    features.update(ses_title, 'asse_manual_accuracy_sum' , manual_accuracy, mode='sum')
    #                 R.add(manual_accuracy , 'accuracy')
                    acc_grp = acc_to_accgrp(manual_accuracy)
                    if true_att + false_att!=0: 
                        features.lastaccgroup[ses_title] = acc_grp
                        features.update(ses_title, 'asse_accgrp_'+str(acc_grp), 1, mode='sum',initial=0)
                    features.update(ses_title, 'asse_manual_accuracy_grp' , features.lastaccgroup[ses_title] )
                    features.update(ses_title, 'asse_manual_accuracy_grp_old' , acc_grp )
                    features.update(ses_title, 'asse_accgrp_'+str(acc_grp) +'_old', 1, mode='sum',initial=0)
                    features.update(ses_title, 'asse_timespent', dur, mode='sum')
                    features.update(ses_title, 'asse_count', 1, mode='sum')
                    features.update(ses_title, 'asse_timespent_avg', dur)

                    if len(ses) > 4 or true_att+false_att > 0: features.last_asse_timestamp = ses.iloc[-1].timestamp


                if ses_type == "Game":
                    all_codes = [2000,2020,2025,2030,2035,2040,2050,2060,2075,2080,2081,2083,3020,3021,3120,4050,4110,
                                 3121,4010,4020,4025,4030,4031,4035,4040,4045,4070,4090,4095,4100,4220,4235,4230,3110]
                    cc = {} # code count
                    for c in all_codes: 
                        cc[c] = count(ses, c)
                        features.update(ses_title, 'count_'+str(c), cc[c], mode='sum', specific=False)
                        features.update(ses_title, 'count_dif_'+str(c), cc[c], mode='sum', specific=False)
    #                     R.add(min(cc[c]/200, 1), str(c))
                    for c in [2020, 2030, 2040]: features.update(ses_title, 'game_count_'+str(c), cc[c], mode='sum', specific=False)
                    features.update(ses_title, 'game_skip_tut', cc[2075]/(1 + cc[2060]))
                    cnt_3021, cnt_3020 = cc[3021], cc[3020]
                    features.update(ses_title, 'game_success_ratio' , cnt_3021 / (cnt_3021+cnt_3020) if cnt_3021+cnt_3020>0 else 0)
    #                 R.add(cnt_3021 / (cnt_3021+cnt_3020) if cnt_3021+cnt_3020>0 else 0 , 'success_ratio')
                    features.update(ses_title, 'game_skip_intro',cc[2081] / (1 + cc[2080]))
                    codes_ = [4020, 4030, 4031, 4045, 4040, 2035, 4025, 4235, 4050, 4230]
                    cnt_actions = sum_count(cc, codes_)
    #                 R.add(cnt_actions , 'cnt_actions')
                    features.update(ses_title, 'game_action_freq', cnt_actions / (dur + 0.01))
    #                 R.add(cnt_actions / (dur + 0.01), 'action_freq')
                    misclick = sum_count(cc, [4070, 4035])
                    features.update(ses_title, 'game_misclickdrag_ratio', misclick / (cnt_actions + 1))
                    features.update(ses_title, 'game_misclickdrag', misclick)
    #                 R.add(misclick/ (cnt_actions + 1), 'misclickdrag_ratio')

                    levelbeaten, roundbeaten = cc[2050], cc[2030]
                    features.update(ses_title, 'game_levelbeaten', levelbeaten)
                    features.update(ses_title, 'game_roundbeaten', roundbeaten)
    #                 R.add(cc[2030] /  (cnt_actions + 1) , 'finish_round')
                    features.update(ses_title, 'game_levelbeaten_freq', levelbeaten/(dur + 0.01))
                    features.update(ses_title, 'game_roundbeaten_freq', roundbeaten/(dur + 0.01))                
                    features.update(ses_title, 'game_levelbeaten_freq2', levelbeaten/(cnt_actions + 1))
                    features.update(ses_title, 'game_roundbeaten_freq2', roundbeaten/(cnt_actions + 1))
                    features.update(ses_title, 'game_seekhelp' , sum_count(cc, [4110, 4090, 4100]))
                    true_att, false_att = search_attempts(ses, 4020)
    #                 R.add(true_att , 'true_att')
    #                 R.add(false_att , 'false_att')
                    gameacc =  true_att/(true_att+false_att) if true_att+false_att!=0 else 0
    #                 R.add(gameacc , 'accuracy')
                    features.update(ses_title, 'game_manual_gameacc', gameacc)
                    features.update(ses_title, 'game_true_att_freq' , true_att/(cnt_actions+1))
                    features.update(ses_title, 'game_false_att_freq' , false_att/(cnt_actions+1))
                    features.update(ses_title, 'game_true_att' , true_att, mode='sum')
                    features.update(ses_title, 'game_false_att' , false_att, mode='sum')
                    features.update(ses_title, 'game_true_att_freq_2' , true_att/(dur + 0.01))
                    features.update(ses_title, 'game_count', 1, mode='sum')
                    features.update(ses_title, 'game_timespent', dur, mode='sum')
                    features.update(ses_title, 'game_timespent_avg', dur)
                    if dur > features.maxgamedur: features.maxgamedur = dur
                    gamestats, gamestats_names, gamestats_modes = get_game_stats(ses)
                    for s, n, m in zip(gamestats, gamestats_names, gamestats_modes): features.update(ses_title, n, s, m)


                if ses_type == "Activity":
                    all_codes = [2000, 2030, 4020, 4021, 4025, 4030, 4035, 4070, 4090, 4022,3110]
                    cc = {} # current count
                    for c in all_codes: 
                        cc[c] = count(ses, c)
    #                     R.add(min(cc[c]/200, 1), str(c))
                        features.update(ses_title, 'count_'+str(c), cc[c], mode='sum', specific=False)
                        features.update(ses_title, 'count_dif_'+str(c), cc[c], mode='sum', specific=False)
                    cnt_actions, cnt_good_actions = sum_count(cc, [4020,4021,4022,4025,4030,4070,4090,4035]), sum_count(cc, [4021,4020,4022,4025]),
    #                 R.add(cnt_actions , 'cnt_actions')
                    misclick = sum_count(cc, [4070, 4035]) 
    #                 R.add(misclick/ (cnt_actions + 1), 'misclickdrag_ratio')
                    features.update(ses_title, 'acti_misclickdrag_ratio' , misclick / (cnt_actions + 1))
                    features.update(ses_title, 'acti_misclickdrag' , misclick)
                    acti_acc = cnt_good_actions / (cnt_good_actions+cnt_actions) if cnt_good_actions+cnt_actions>0 else 0 
                    features.update(ses_title, 'acti_success_ratio', acti_acc)
    #                 R.add(acti_acc , 'success_ratio')
                    features.update(ses_title, 'acti_action_freq' ,  cnt_actions/(dur+0.01))
    #                 R.add(cnt_actions / (dur + 0.01), 'action_freq')
                    features.update(ses_title, 'acti_count', 1, mode='sum')
                    features.update(ses_title, 'acti_timespent', dur, mode='sum')
                    features.update(ses_title, 'acti_timespent_avg', dur)
                    features.update(ses_title, 'acti_countevents', len(ses), mode='sum', specific=False)
                    if len(ses) > 4: features.update(ses_title, 'acti_countevents2', len(ses), mode='sum',specific=False)

                if ses_type == "Clip":
                    clip_dur = clip_len[ses.iloc[0].title]
                    next_idx =  ses.index[-1] + 1
                    if next_idx in all_idx: 
                        dur = (sub_df.loc[next_idx].timestamp - ses.iloc[0].timestamp).seconds
                        dur = min(dur, clip_dur)                
                    else: dur = clip_dur
                    features.update(ses_title, 'clip_timespent', dur, mode='sum')
                    features.update(ses_title, 'clip_viewratio', min(1, dur/clip_dur))
                    features.update(ses_title, 'clip_clipsviewed', 1, mode='sum')

                # count some specific events for some titles
                if ses_title in SPECIFIC_COUNT.keys():
                    for c in SPECIFIC_COUNT[ses_title]: features.update_count_simple(ses_title+'_count_'+str(c), count(ses, c))

                # count all events
                features.update_count_simple('count_allevents', len(ses))

                # count all events for some titles
                if ses_title in ['Cart Balancer (Assessment)', 'Cauldron Filler (Assessment)', 'Scrub-A-Dub', 'Crystal Caves - Level 1']:
                    features.update_count_simple(ses_title+'_countallevents', len(ses))


                last_idx =  ses.index[0] - 1
                gap = (ses.iloc[0].timestamp - sub_df.loc[last_idx].timestamp).seconds if sub_df.index[0]<last_idx else 0
    #             R.add(min(gap/1000, 1), 'gap')    
    #             R.register(ses_title, acc_g, ses_id)

    #         feat_rnn[ins_id] = R

        if mode == 'train': 
            tr_feat = feat_df
    #         tr_rnn_feat = feat_rnn
        elif mode == 'test': 
            te_feat = feat_df
    #         te_rnn_feat = feat_rnn


# In[ ]:


tr_feat = pd.read_csv(KHA_PATH+'kha_tr_2001.csv')
if RUN_MODE=='public': te_feat = pd.read_csv(KHA_PATH+'kha_te_2001.csv')

tr_feat.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in tr_feat.columns]
te_feat.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in te_feat.columns]

def get_old_feat(f):
    oldfeatdict = {'asse_manual_accuracy_grp': 'asse_manual_accuracy_grp_old', 
                   'w0_asse_manual_accuracy_grp': 'w0_asse_manual_accuracy_grp_old', 
                   'sameworld_asse_manual_accuracy_grp': 'sameworld_asse_manual_accuracy_grp_old', 
                   'Mushroom_Sorter__Assessment_asse_manual_accuracy_grp': 'Mushroom_Sorter__Assessment_asse_manual_accuracy_grp_old', 
                   'Mushroom_Sorter__Assessment_asse_manual_accuracy_grpLast': 'Mushroom_Sorter__Assessment_asse_manual_accuracy_grp_oldLast',
                   'asse_accgrp_0': 'asse_accgrp_0_old', 
                   'w0_asse_accgrp_0': 'w0_asse_accgrp_0_old', 
                   'sameworld_asse_accgrp_0': 'sameworld_asse_accgrp_0_old', 
                   'sameasse_asse_manual_accuracy_grp': 'sameasse_asse_manual_accuracy_grp_old', 
                   'sameasse_asse_manual_accuracy_grpLast': 'sameasse_asse_manual_accuracy_grp_oldLast', 
                   'Mushroom_Sorter__Assessment_asse_accgrp_0': 'Mushroom_Sorter__Assessment_asse_accgrp_0_old', 
                   'Mushroom_Sorter__Assessment_asse_accgrp_0Last': 'Mushroom_Sorter__Assessment_asse_accgrp_0_oldLast', 
                   'sameasse_asse_accgrp_0': 'sameasse_asse_accgrp_0_old', 
                   'sameasse_asse_accgrp_0Last': 'sameasse_asse_accgrp_0_oldLast', 
                   'w1_asse_manual_accuracy_grp': 'w1_asse_manual_accuracy_grp_old', 
                   'Cauldron_Filler__Assessment_asse_manual_accuracy_grp': 'Cauldron_Filler__Assessment_asse_manual_accuracy_grp_old', 
                   'Cauldron_Filler__Assessment_asse_manual_accuracy_grpLast': 'Cauldron_Filler__Assessment_asse_manual_accuracy_grp_oldLast', 
                   'w1_asse_accgrp_0': 'w1_asse_accgrp_0_old', 
                   'Cauldron_Filler__Assessment_asse_accgrp_0': 'Cauldron_Filler__Assessment_asse_accgrp_0_old', 
                   'Cauldron_Filler__Assessment_asse_accgrp_0Last':'Cauldron_Filler__Assessment_asse_accgrp_0_oldLast', 
                   'w2_asse_manual_accuracy_grp':'w2_asse_manual_accuracy_grp_old', 
                   'Cart_Balancer__Assessment_asse_manual_accuracy_grp':'Cart_Balancer__Assessment_asse_manual_accuracy_grp_old', 
                   'Cart_Balancer__Assessment_asse_manual_accuracy_grpLast':'Cart_Balancer__Assessment_asse_manual_accuracy_grp_oldLast', 
                   'w2_asse_accgrp_0':'w2_asse_accgrp_0_old', 
                   'Cart_Balancer__Assessment_asse_accgrp_0':'Cart_Balancer__Assessment_asse_accgrp_0_old', 
                   'Cart_Balancer__Assessment_asse_accgrp_0Last':'Cart_Balancer__Assessment_asse_accgrp_0_oldLast', 
                   'Chest_Sorter__Assessment_asse_accgrp_0':'Chest_Sorter__Assessment_asse_accgrp_0_old', 
                   'title_enc': 'title_enc_old'
    }
    
    if f in oldfeatdict.keys(): return oldfeatdict[f]
    else: return f
    
top195 = ['title_enc', 'game_success_ratio', 'acti_misclickdrag_ratio', 'game_manual_gameacc', 'sameworld_game_success_ratio', 'acti_success_ratio', 'clip_viewratio', 'sameasse_asse_success_ratio', 'asse_manual_accuracy_grp', 'asse_false_att_freq', 'count_dif_4070', 'game_roundbeaten_freq', 'asse_misclickdrag_ratio', 'game_false_att_freq', 'Chow_Timegame_misclickdrag_ratio', 'sameworld_acti_misclickdrag_ratio', 'sameworld_game_roundbeaten_freq', 'game_misclickdrag_ratio', 'w2_clip_timespent', 'sameworld_acti_timespent_avg', 'game_ratio_0_miss', 'acti_action_freq', 'sameworld_acti_action_freq', 'w0_clip_timespent', 'Sandcastle_Builder__Activity_acti_success_ratioLast', 'clip_timespent', 'clip_clipsviewed', 'asse_success_ratio', 'acti_timespent_avg', 'w2_clip_viewratio', 'game_true_att_freq_2', 'w0_clip_viewratio', 'sameasse_asse_true_att_freq_2', 'sameworld_count_4030', 'acti_timespent', 'sameasse_asse_false_att_freqLast', 'w1_clip_timespent', 'sameworld_clip_viewratio', 'sameworld_count_dif_4070', 'sameworld_asse_correctplacement_ratio2', 'w2_game_misclickdrag_ratio', 'sameasse_asse_true_att_freq', 'sameworld_clip_timespent', 'count_4070', 'sameworld_asse_correctplacement_ratio', 'w0_acti_success_ratio', 'sameasse_asse_true_att_freq_2Last', 'sameworld_game_timespent_avg', 'asse_true_att_freq', 'w1_acti_success_ratio', 'asse_manual_accuracy', 'sameworld_game_roundbeaten_freq2', 'w2_game_roundbeaten_freq', 'sameworld_count_4020', 'Happy_Camelgame_false_att_freq', 'sameworld_game_misclickdrag_ratio', 'Chow_Timegame_misclickdrag_ratioLast', 'asse_finish_round', 'w2_game_success_ratio', 'game_action_freq', 'sameworld_acti_success_ratio', 'sameasse_asse_manual_accuracyLast', 'Sandcastle_Builder__Activity_acti_misclickdrag_ratioLast', 'sameworld_game_manual_gameacc', 'w1_acti_action_freq', 'sameasse_asse_finish_round', 'sameworld_count_dif_4030', 'count_4020', 'sameworld_game_false_att_freq', 'count_dif_4031', 'w0_asse_finish_round', 'sameworld_count_4070', 'Sandcastle_Builder__Activity_acti_success_ratio', 'sameworld_acti_timespent', 'w1_acti_misclickdrag_ratio', 'sameworld_game_ratio_0_miss', 'count_dif_4020', 'asse_true_att_freq_2', 'Fireworks__Activity_acti_timespentLast', 'asse_correctplacement_ratio', 'game_true_att_freq', 'w1_game_misclickdrag_ratio', 'w0_clip_clipsviewed', 'w1_clip_viewratio', 'Sandcastle_Builder__Activity_acti_timespentLast', 'Sandcastle_Builder__Activity_acti_timespent', 'Sandcastle_Builder__Activity_acti_timespent_avg', 'game_timespent_avg', 'w1_game_success_ratio', 'sameworld_clip_clipsviewed', 'w2_game_roundbeaten_freq2', 'sameworld_count_dif_4020', 'sameworld_asse_false_att_freq', 'All_Star_Sortinggame_timespentLast', 'sameasse_asse_false_att_freq', 'asse_action_freq', 'count_4035', 'sameworld_count_dif_2030', 'w0_acti_timespent_avg', 'w1_acti_timespent', 'game_ratio_1_3_miss', 'asse_timespent_avg', 'count_4030', 'Scrub_A_Dubgame_true_att_freq', 'sameworld_game_action_freq', 'sameworld_game_true_att_freq', 'game_skip_tut', 'sameasse_asse_success_ratioLast', 'w0_asse_true_att_freq_2', 'Pan_Balancegame_success_ratio', 'sameworld_game_true_att_freq_2', 'w0_game_ratio_0_miss', 'sameworld_count_dif_4090', 'Flower_Waterer__Activity_acti_action_freq', 'w2_game_false_att_freq', 'Chow_Timegame_roundbeaten_freq', 'Chow_Timegame_timespentLast', 'w1_game_timespent_avg', 'sameworld_count_dif_3020', 'count_dif_4035', 'w2_game_timespent_avg', 'Sandcastle_Builder__Activity_acti_misclickdrag_ratio', 'w1_acti_timespent_avg', 'count_dif_4030', 'Chow_Timegame_roundbeaten_freqLast', 'asse_cnt_actions', 'Flower_Waterer__Activity_acti_success_ratio', 'Dino_Drinkgame_timespentLast', 'w1_asse_cnt_actions', 'count_dif_4022', 'w0_game_true_att_freq_2', 'w2_acti_timespent_avg', 'w2_game_manual_gameacc', 'sameworld_count_dif_4031', 'sameworld_asse_misclickdrag_ratio', 'sameworld_count_dif_3021', 'Flower_Waterer__Activity_acti_action_freqLast', 'count_dif_4010', 'Flower_Waterer__Activity_acti_timespent', 'Fireworks__Activity_acti_success_ratio', 'Dino_Divegame_roundbeaten_freqLast', 'w0_acti_action_freq', 'sameworld_count_4035', 'asse_timespent', 'sameworld_game_false_att', 'Flower_Waterer__Activity_acti_timespentLast', 'Chow_Timegame_ratio_1_3_miss', 'count_dif_3020', 'w1_game_false_att_freq', 'sameworld_count_3020', 'game_timespent', 'game_false_att', 'Chow_Timegame_success_ratio', 'w2_clip_clipsviewed', 'sameworld_asse_manual_accuracy', 'w0_asse_manual_accuracy', 'Fireworks__Activity_acti_success_ratioLast', 'count_3120', 'All_Star_Sortinggame_timespent_avg', 'Chow_Timegame_timespent', 'game_roundbeaten_freq2', 'sameasse_asse_true_att_freqLast', 'Bottle_Filler__Activity_acti_misclickdrag_ratioLast', 'sameworld_game_roundbeaten', 'count_dif_2030', 'Fireworks__Activity_acti_timespent_avg', 'Sandcastle_Builder__Activity_acti_action_freqLast', 'sameworld_count_dif_3121', 'w0_acti_timespent', 'w2_acti_misclickdrag_ratio', 'Flower_Waterer__Activity_acti_timespent_avg', 'All_Star_Sortinggame_true_att_freq_2Last', 'sameasse_asse_misclickdrag_ratio', 'w1_asse_true_att_freq', 'sameworld_count_dif_4025', 'count_dif_4021', 'sameworld_asse_manual_accuracy_grp', 'w0_acti_misclickdrag_ratio', 'sameasse_asse_correctplacement_ratio2', 'w0_game_misclickdrag_ratio', 'sameworld_game_ratio_3_10_miss', 'sameworld_game_timespent', 'sameworld_count_2030', 'All_Star_Sortinggame_roundbeaten_freqLast', 'sameworld_count_4040', 'Happy_Camelgame_misclickdrag_ratioLast', 'asse_correctplacement_ratio2', 'sameworld_asse_finish_round', 'game_true_att', 'w1_game_action_freq', 'game_skip_intro', 'w2_game_true_att_freq_2', 'Bubble_Bathgame_misclickdrag_ratioLast', 'sameworld_count_4021', 'w2_game_action_freq']
top195 = [get_old_feat(f) for f in top195]
features_1700 = ['clip_timespent', 'w0_clip_timespent', 'sameworld_clip_timespent', 'clip_viewratio', 'w0_clip_viewratio', 'sameworld_clip_viewratio', 'clip_clipsviewed', 'w0_clip_clipsviewed', 'sameworld_clip_clipsviewed', 'w1_clip_timespent', 'w1_clip_viewratio', 'w1_clip_clipsviewed', 'count_2000', 'w1_count_2000', 'count_dif_2000', 'w1_count_dif_2000', 'count_2030', 'w1_count_2030', 'count_dif_2030', 'w1_count_dif_2030', 'count_4020', 'w1_count_4020', 'count_dif_4020', 'w1_count_dif_4020', 'count_4021', 'w1_count_4021', 'count_dif_4021', 'w1_count_dif_4021', 'count_4025', 'w1_count_4025', 'count_dif_4025', 'w1_count_dif_4025', 'count_4030', 'w1_count_4030', 'count_dif_4030', 'w1_count_dif_4030', 'count_4035', 'w1_count_4035', 'count_dif_4035', 'w1_count_dif_4035', 'count_4070', 'w1_count_4070', 'count_dif_4070', 'w1_count_dif_4070', 'count_4090', 'w1_count_4090', 'count_dif_4090', 'w1_count_dif_4090', 'count_4022', 'w1_count_4022', 'count_dif_4022', 'w1_count_dif_4022', 'acti_misclickdrag_ratio', 'w1_acti_misclickdrag_ratio', 'acti_misclickdrag', 'w1_acti_misclickdrag', 'acti_success_ratio', 'w1_acti_success_ratio', 'acti_action_freq', 'w1_acti_action_freq', 'acti_count', 'w1_acti_count', 'acti_timespent', 'w1_acti_timespent', 'acti_timespent_avg', 'w1_acti_timespent_avg', 'count_2025', 'w1_count_2025', 'count_dif_2025', 'w1_count_dif_2025', 'count_2035', 'w1_count_2035', 'count_dif_2035', 'w1_count_dif_2035', 'count_2040', 'w1_count_2040', 'count_dif_2040', 'w1_count_dif_2040', 'count_2050', 'w1_count_2050', 'count_dif_2050', 'w1_count_dif_2050', 'count_2060', 'w1_count_2060', 'count_dif_2060', 'w1_count_dif_2060', 'count_2075', 'w1_count_2075', 'count_dif_2075', 'w1_count_dif_2075', 'count_2080', 'w1_count_2080', 'count_dif_2080', 'w1_count_dif_2080', 'count_2081', 'w1_count_2081', 'count_dif_2081', 'w1_count_dif_2081', 'count_2083', 'w1_count_2083', 'count_dif_2083', 'w1_count_dif_2083', 'count_3020', 'w1_count_3020', 'count_dif_3020', 'w1_count_dif_3020', 'count_3021', 'w1_count_3021', 'count_dif_3021', 'w1_count_dif_3021', 'count_3120', 'w1_count_3120', 'count_dif_3120', 'w1_count_dif_3120', 'count_4050', 'w1_count_4050', 'count_dif_4050', 'w1_count_dif_4050', 'count_4110', 'w1_count_4110', 'count_dif_4110', 'w1_count_dif_4110', 'count_3121', 'w1_count_3121', 'count_dif_3121', 'w1_count_dif_3121', 'count_4010', 'w1_count_4010', 'count_dif_4010', 'w1_count_dif_4010', 'count_4031', 'w1_count_4031', 'count_dif_4031', 'w1_count_dif_4031', 'count_4040', 'w1_count_4040', 'count_dif_4040', 'w1_count_dif_4040', 'count_4045', 'w1_count_4045', 'count_dif_4045', 'w1_count_dif_4045', 'count_4095', 'w1_count_4095', 'count_dif_4095', 'w1_count_dif_4095', 'count_4100', 'w1_count_4100', 'count_dif_4100', 'w1_count_dif_4100', 'count_4220', 'w1_count_4220', 'count_dif_4220', 'w1_count_dif_4220', 'count_4235', 'w1_count_4235', 'count_dif_4235', 'w1_count_dif_4235', 'count_4230', 'w1_count_4230', 'count_dif_4230', 'w1_count_dif_4230', 'game_skip_tut', 'w1_game_skip_tut', 'game_success_ratio', 'w1_game_success_ratio', 'game_skip_intro', 'w1_game_skip_intro', 'game_action_freq', 'w1_game_action_freq', 'game_misclickdrag_ratio', 'w1_game_misclickdrag_ratio', 'game_misclickdrag', 'w1_game_misclickdrag', 'game_levelbeaten', 'w1_game_levelbeaten', 'game_roundbeaten', 'w1_game_roundbeaten', 'game_levelbeaten_freq', 'w1_game_levelbeaten_freq', 'game_roundbeaten_freq', 'w1_game_roundbeaten_freq', 'game_levelbeaten_freq2', 'w1_game_levelbeaten_freq2', 'game_roundbeaten_freq2', 'w1_game_roundbeaten_freq2', 'game_seekhelp', 'w1_game_seekhelp', 'game_manual_gameacc', 'w1_game_manual_gameacc', 'game_true_att_freq', 'w1_game_true_att_freq', 'game_false_att_freq', 'w1_game_false_att_freq', 'game_true_att', 'w1_game_true_att', 'game_false_att', 'w1_game_false_att', 'game_true_att_freq_2', 'w1_game_true_att_freq_2', 'game_count', 'w1_game_count', 'game_timespent', 'w1_game_timespent', 'game_timespent_avg', 'w1_game_timespent_avg', 'game_ratio_0_miss', 'w1_game_ratio_0_miss', 'game_ratio_1_3_miss', 'w1_game_ratio_1_3_miss', 'game_ratio_3_10_miss', 'w1_game_ratio_3_10_miss', 'game_ratio_more_10_miss', 'w1_game_ratio_more_10_miss', 'game_nb_0_miss', 'w1_game_nb_0_miss', 'game_nb_1_3_miss', 'w1_game_nb_1_3_miss', 'game_nb_3_10_miss', 'w1_game_nb_3_10_miss', 'game_nb_more_10_miss', 'w1_game_nb_more_10_miss', 'Sandcastle_Builder__Activity_acti_misclickdrag_ratio', 'Sandcastle_Builder__Activity_acti_misclickdrag_ratioLast', 'Sandcastle_Builder__Activity_acti_misclickdrag', 'Sandcastle_Builder__Activity_acti_misclickdragLast', 'Sandcastle_Builder__Activity_acti_success_ratio', 'Sandcastle_Builder__Activity_acti_success_ratioLast', 'Sandcastle_Builder__Activity_acti_action_freq', 'Sandcastle_Builder__Activity_acti_action_freqLast', 'Sandcastle_Builder__Activity_acti_count', 'Sandcastle_Builder__Activity_acti_countLast', 'Sandcastle_Builder__Activity_acti_timespent', 'Sandcastle_Builder__Activity_acti_timespentLast', 'Sandcastle_Builder__Activity_acti_timespent_avg', 'Sandcastle_Builder__Activity_acti_timespent_avgLast', 'Scrub_A_Dubgame_skip_tut', 'Scrub_A_Dubgame_skip_tutLast', 'Scrub_A_Dubgame_success_ratio', 'Scrub_A_Dubgame_success_ratioLast', 'Scrub_A_Dubgame_skip_intro', 'Scrub_A_Dubgame_skip_introLast', 'Scrub_A_Dubgame_action_freq', 'Scrub_A_Dubgame_action_freqLast', 'Scrub_A_Dubgame_misclickdrag_ratio', 'Scrub_A_Dubgame_misclickdrag_ratioLast', 'Scrub_A_Dubgame_misclickdrag', 'Scrub_A_Dubgame_misclickdragLast', 'Scrub_A_Dubgame_levelbeaten', 'Scrub_A_Dubgame_levelbeatenLast', 'Scrub_A_Dubgame_roundbeaten', 'Scrub_A_Dubgame_roundbeatenLast', 'Scrub_A_Dubgame_levelbeaten_freq', 'Scrub_A_Dubgame_levelbeaten_freqLast', 'Scrub_A_Dubgame_roundbeaten_freq', 'Scrub_A_Dubgame_roundbeaten_freqLast', 'Scrub_A_Dubgame_levelbeaten_freq2', 'Scrub_A_Dubgame_levelbeaten_freq2Last', 'Scrub_A_Dubgame_roundbeaten_freq2', 'Scrub_A_Dubgame_roundbeaten_freq2Last', 'Scrub_A_Dubgame_seekhelp', 'Scrub_A_Dubgame_seekhelpLast', 'Scrub_A_Dubgame_manual_gameacc', 'Scrub_A_Dubgame_manual_gameaccLast', 'Scrub_A_Dubgame_true_att_freq', 'Scrub_A_Dubgame_true_att_freqLast', 'Scrub_A_Dubgame_false_att_freq', 'Scrub_A_Dubgame_false_att_freqLast', 'Scrub_A_Dubgame_true_att', 'Scrub_A_Dubgame_true_attLast', 'Scrub_A_Dubgame_false_att', 'Scrub_A_Dubgame_false_attLast', 'Scrub_A_Dubgame_true_att_freq_2', 'Scrub_A_Dubgame_true_att_freq_2Last', 'Scrub_A_Dubgame_count', 'Scrub_A_Dubgame_countLast', 'Scrub_A_Dubgame_timespent', 'Scrub_A_Dubgame_timespentLast', 'Scrub_A_Dubgame_timespent_avg', 'Scrub_A_Dubgame_timespent_avgLast', 'Scrub_A_Dubgame_ratio_0_miss', 'Scrub_A_Dubgame_ratio_0_missLast', 'Scrub_A_Dubgame_ratio_1_3_miss', 'Scrub_A_Dubgame_ratio_1_3_missLast', 'Scrub_A_Dubgame_ratio_3_10_miss', 'Scrub_A_Dubgame_ratio_3_10_missLast', 'Scrub_A_Dubgame_ratio_more_10_miss', 'Scrub_A_Dubgame_ratio_more_10_missLast', 'Scrub_A_Dubgame_nb_0_miss', 'Scrub_A_Dubgame_nb_0_missLast', 'Scrub_A_Dubgame_nb_1_3_miss', 'Scrub_A_Dubgame_nb_1_3_missLast', 'Scrub_A_Dubgame_nb_3_10_miss', 'Scrub_A_Dubgame_nb_3_10_missLast', 'Scrub_A_Dubgame_nb_more_10_miss', 'Scrub_A_Dubgame_nb_more_10_missLast', 'Dino_Drinkgame_skip_tut', 'Dino_Drinkgame_skip_tutLast', 'Dino_Drinkgame_success_ratio', 'Dino_Drinkgame_success_ratioLast', 'Dino_Drinkgame_skip_intro', 'Dino_Drinkgame_skip_introLast', 'Dino_Drinkgame_action_freq', 'Dino_Drinkgame_action_freqLast', 'Dino_Drinkgame_misclickdrag_ratio', 'Dino_Drinkgame_misclickdrag_ratioLast', 'Dino_Drinkgame_misclickdrag', 'Dino_Drinkgame_misclickdragLast', 'Dino_Drinkgame_levelbeaten', 'Dino_Drinkgame_levelbeatenLast', 'Dino_Drinkgame_roundbeaten', 'Dino_Drinkgame_roundbeatenLast', 'Dino_Drinkgame_levelbeaten_freq', 'Dino_Drinkgame_levelbeaten_freqLast', 'Dino_Drinkgame_roundbeaten_freq', 'Dino_Drinkgame_roundbeaten_freqLast', 'Dino_Drinkgame_levelbeaten_freq2', 'Dino_Drinkgame_levelbeaten_freq2Last', 'Dino_Drinkgame_roundbeaten_freq2', 'Dino_Drinkgame_roundbeaten_freq2Last', 'Dino_Drinkgame_seekhelp', 'Dino_Drinkgame_seekhelpLast', 'Dino_Drinkgame_manual_gameacc', 'Dino_Drinkgame_manual_gameaccLast', 'Dino_Drinkgame_true_att_freq', 'Dino_Drinkgame_true_att_freqLast', 'Dino_Drinkgame_false_att_freq', 'Dino_Drinkgame_false_att_freqLast', 'Dino_Drinkgame_true_att', 'Dino_Drinkgame_true_attLast', 'Dino_Drinkgame_false_att', 'Dino_Drinkgame_false_attLast', 'Dino_Drinkgame_true_att_freq_2', 'Dino_Drinkgame_true_att_freq_2Last', 'Dino_Drinkgame_count', 'Dino_Drinkgame_countLast', 'Dino_Drinkgame_timespent', 'Dino_Drinkgame_timespentLast', 'Dino_Drinkgame_timespent_avg', 'Dino_Drinkgame_timespent_avgLast', 'Dino_Drinkgame_ratio_0_miss', 'Dino_Drinkgame_ratio_0_missLast', 'Dino_Drinkgame_ratio_1_3_miss', 'Dino_Drinkgame_ratio_1_3_missLast', 'Dino_Drinkgame_ratio_3_10_miss', 'Dino_Drinkgame_ratio_3_10_missLast', 'Dino_Drinkgame_ratio_more_10_miss', 'Dino_Drinkgame_ratio_more_10_missLast', 'Dino_Drinkgame_nb_0_miss', 'Dino_Drinkgame_nb_0_missLast', 'Dino_Drinkgame_nb_1_3_miss', 'Dino_Drinkgame_nb_1_3_missLast', 'Dino_Drinkgame_nb_3_10_miss', 'Dino_Drinkgame_nb_3_10_missLast', 'Dino_Drinkgame_nb_more_10_miss', 'Dino_Drinkgame_nb_more_10_missLast', 'sameworld_count_2000', 'sameworld_count_dif_2000', 'sameworld_count_2030', 'sameworld_count_dif_2030', 'sameworld_count_4020', 'sameworld_count_dif_4020', 'sameworld_count_4021', 'sameworld_count_dif_4021', 'sameworld_count_4025', 'sameworld_count_dif_4025', 'sameworld_count_4030', 'sameworld_count_dif_4030', 'sameworld_count_4035', 'sameworld_count_dif_4035', 'sameworld_count_4070', 'sameworld_count_dif_4070', 'sameworld_count_4090', 'sameworld_count_dif_4090', 'sameworld_count_4022', 'sameworld_count_dif_4022', 'sameworld_acti_misclickdrag_ratio', 'sameworld_acti_misclickdrag', 'sameworld_acti_success_ratio', 'sameworld_acti_action_freq', 'sameworld_acti_count', 'sameworld_acti_timespent', 'sameworld_acti_timespent_avg', 'Watering_Hole__Activity_acti_misclickdrag_ratio', 'Watering_Hole__Activity_acti_misclickdrag_ratioLast', 'Watering_Hole__Activity_acti_misclickdrag', 'Watering_Hole__Activity_acti_misclickdragLast', 'Watering_Hole__Activity_acti_success_ratio', 'Watering_Hole__Activity_acti_success_ratioLast', 'Watering_Hole__Activity_acti_action_freq', 'Watering_Hole__Activity_acti_action_freqLast', 'Watering_Hole__Activity_acti_count', 'Watering_Hole__Activity_acti_countLast', 'Watering_Hole__Activity_acti_timespent', 'Watering_Hole__Activity_acti_timespentLast', 'Watering_Hole__Activity_acti_timespent_avg', 'Watering_Hole__Activity_acti_timespent_avgLast', 'w0_count_2000', 'w0_count_dif_2000', 'w0_count_2025', 'sameworld_count_2025', 'w0_count_dif_2025', 'sameworld_count_dif_2025', 'w0_count_2030', 'w0_count_dif_2030', 'w0_count_2035', 'sameworld_count_2035', 'w0_count_dif_2035', 'sameworld_count_dif_2035', 'w0_count_2040', 'sameworld_count_2040', 'w0_count_dif_2040', 'sameworld_count_dif_2040', 'w0_count_2050', 'sameworld_count_2050', 'w0_count_dif_2050', 'sameworld_count_dif_2050', 'w0_count_2060', 'sameworld_count_2060', 'w0_count_dif_2060', 'sameworld_count_dif_2060', 'w0_count_2075', 'sameworld_count_2075', 'w0_count_dif_2075', 'sameworld_count_dif_2075', 'w0_count_2080', 'sameworld_count_2080', 'w0_count_dif_2080', 'sameworld_count_dif_2080', 'w0_count_2081', 'sameworld_count_2081', 'w0_count_dif_2081', 'sameworld_count_dif_2081', 'w0_count_2083', 'sameworld_count_2083', 'w0_count_dif_2083', 'sameworld_count_dif_2083', 'w0_count_3020', 'sameworld_count_3020', 'w0_count_dif_3020', 'sameworld_count_dif_3020', 'w0_count_3021', 'sameworld_count_3021', 'w0_count_dif_3021', 'sameworld_count_dif_3021', 'w0_count_3120', 'sameworld_count_3120', 'w0_count_dif_3120', 'sameworld_count_dif_3120', 'w0_count_4050', 'sameworld_count_4050', 'w0_count_dif_4050', 'sameworld_count_dif_4050', 'w0_count_4110', 'sameworld_count_4110', 'w0_count_dif_4110', 'sameworld_count_dif_4110', 'w0_count_3121', 'sameworld_count_3121', 'w0_count_dif_3121', 'sameworld_count_dif_3121', 'w0_count_4010', 'sameworld_count_4010', 'w0_count_dif_4010', 'sameworld_count_dif_4010', 'w0_count_4020', 'w0_count_dif_4020', 'w0_count_4025', 'w0_count_dif_4025', 'w0_count_4030', 'w0_count_dif_4030', 'w0_count_4031', 'sameworld_count_4031', 'w0_count_dif_4031', 'sameworld_count_dif_4031', 'w0_count_4035', 'w0_count_dif_4035', 'w0_count_4040', 'sameworld_count_4040', 'w0_count_dif_4040', 'sameworld_count_dif_4040', 'w0_count_4045', 'sameworld_count_4045', 'w0_count_dif_4045', 'sameworld_count_dif_4045', 'w0_count_4070', 'w0_count_dif_4070', 'w0_count_4090', 'w0_count_dif_4090', 'w0_count_4095', 'sameworld_count_4095', 'w0_count_dif_4095', 'sameworld_count_dif_4095', 'w0_count_4100', 'sameworld_count_4100', 'w0_count_dif_4100', 'sameworld_count_dif_4100', 'w0_count_4220', 'sameworld_count_4220', 'w0_count_dif_4220', 'sameworld_count_dif_4220', 'w0_count_4235', 'sameworld_count_4235', 'w0_count_dif_4235', 'sameworld_count_dif_4235', 'w0_count_4230', 'sameworld_count_4230', 'w0_count_dif_4230', 'sameworld_count_dif_4230', 'w0_game_skip_tut', 'sameworld_game_skip_tut', 'w0_game_success_ratio', 'sameworld_game_success_ratio', 'w0_game_skip_intro', 'sameworld_game_skip_intro', 'w0_game_action_freq', 'sameworld_game_action_freq', 'w0_game_misclickdrag_ratio', 'sameworld_game_misclickdrag_ratio', 'w0_game_misclickdrag', 'sameworld_game_misclickdrag', 'w0_game_levelbeaten', 'sameworld_game_levelbeaten', 'w0_game_roundbeaten', 'sameworld_game_roundbeaten', 'w0_game_levelbeaten_freq', 'sameworld_game_levelbeaten_freq', 'w0_game_roundbeaten_freq', 'sameworld_game_roundbeaten_freq', 'w0_game_levelbeaten_freq2', 'sameworld_game_levelbeaten_freq2', 'w0_game_roundbeaten_freq2', 'sameworld_game_roundbeaten_freq2', 'w0_game_seekhelp', 'sameworld_game_seekhelp', 'w0_game_manual_gameacc', 'sameworld_game_manual_gameacc', 'w0_game_true_att_freq', 'sameworld_game_true_att_freq', 'w0_game_false_att_freq', 'sameworld_game_false_att_freq', 'w0_game_true_att', 'sameworld_game_true_att', 'w0_game_false_att', 'sameworld_game_false_att', 'w0_game_true_att_freq_2', 'sameworld_game_true_att_freq_2', 'w0_game_count', 'sameworld_game_count', 'w0_game_timespent', 'sameworld_game_timespent', 'w0_game_timespent_avg', 'sameworld_game_timespent_avg', 'w0_game_ratio_0_miss', 'sameworld_game_ratio_0_miss', 'w0_game_ratio_1_3_miss', 'sameworld_game_ratio_1_3_miss', 'w0_game_ratio_3_10_miss', 'sameworld_game_ratio_3_10_miss', 'w0_game_ratio_more_10_miss', 'sameworld_game_ratio_more_10_miss', 'w0_game_nb_0_miss', 'sameworld_game_nb_0_miss', 'w0_game_nb_1_3_miss', 'sameworld_game_nb_1_3_miss', 'w0_game_nb_3_10_miss', 'sameworld_game_nb_3_10_miss', 'w0_game_nb_more_10_miss', 'sameworld_game_nb_more_10_miss', 'w0_count_4021', 'w0_count_dif_4021', 'w0_count_4022', 'w0_count_dif_4022', 'w0_acti_misclickdrag_ratio', 'w0_acti_misclickdrag', 'w0_acti_success_ratio', 'w0_acti_action_freq', 'w0_acti_count', 'w0_acti_timespent', 'w0_acti_timespent_avg', 'All_Star_Sortinggame_skip_tut', 'All_Star_Sortinggame_skip_tutLast', 'All_Star_Sortinggame_success_ratio', 'All_Star_Sortinggame_success_ratioLast', 'All_Star_Sortinggame_skip_intro', 'All_Star_Sortinggame_skip_introLast', 'All_Star_Sortinggame_action_freq', 'All_Star_Sortinggame_action_freqLast', 'All_Star_Sortinggame_misclickdrag_ratio', 'All_Star_Sortinggame_misclickdrag_ratioLast', 'All_Star_Sortinggame_misclickdrag', 'All_Star_Sortinggame_misclickdragLast', 'All_Star_Sortinggame_levelbeaten', 'All_Star_Sortinggame_levelbeatenLast', 'All_Star_Sortinggame_roundbeaten', 'All_Star_Sortinggame_roundbeatenLast', 'All_Star_Sortinggame_levelbeaten_freq', 'All_Star_Sortinggame_levelbeaten_freqLast', 'All_Star_Sortinggame_roundbeaten_freq', 'All_Star_Sortinggame_roundbeaten_freqLast', 'All_Star_Sortinggame_levelbeaten_freq2', 'All_Star_Sortinggame_levelbeaten_freq2Last', 'All_Star_Sortinggame_roundbeaten_freq2', 'All_Star_Sortinggame_roundbeaten_freq2Last', 'All_Star_Sortinggame_seekhelp', 'All_Star_Sortinggame_seekhelpLast', 'All_Star_Sortinggame_manual_gameacc', 'All_Star_Sortinggame_manual_gameaccLast', 'All_Star_Sortinggame_true_att_freq', 'All_Star_Sortinggame_true_att_freqLast', 'All_Star_Sortinggame_false_att_freq', 'All_Star_Sortinggame_false_att_freqLast', 'All_Star_Sortinggame_true_att', 'All_Star_Sortinggame_true_attLast', 'All_Star_Sortinggame_false_att', 'All_Star_Sortinggame_false_attLast', 'All_Star_Sortinggame_true_att_freq_2', 'All_Star_Sortinggame_true_att_freq_2Last', 'All_Star_Sortinggame_count', 'All_Star_Sortinggame_countLast', 'All_Star_Sortinggame_timespent', 'All_Star_Sortinggame_timespentLast', 'All_Star_Sortinggame_timespent_avg', 'All_Star_Sortinggame_timespent_avgLast', 'All_Star_Sortinggame_ratio_0_miss', 'All_Star_Sortinggame_ratio_0_missLast', 'All_Star_Sortinggame_ratio_1_3_miss', 'All_Star_Sortinggame_ratio_1_3_missLast', 'All_Star_Sortinggame_ratio_3_10_miss', 'All_Star_Sortinggame_ratio_3_10_missLast', 'All_Star_Sortinggame_ratio_more_10_miss', 'All_Star_Sortinggame_ratio_more_10_missLast', 'All_Star_Sortinggame_nb_0_miss', 'All_Star_Sortinggame_nb_0_missLast', 'All_Star_Sortinggame_nb_1_3_miss', 'All_Star_Sortinggame_nb_1_3_missLast', 'All_Star_Sortinggame_nb_3_10_miss', 'All_Star_Sortinggame_nb_3_10_missLast', 'All_Star_Sortinggame_nb_more_10_miss', 'All_Star_Sortinggame_nb_more_10_missLast', 'Fireworks__Activity_acti_misclickdrag_ratio', 'Fireworks__Activity_acti_misclickdrag_ratioLast', 'Fireworks__Activity_acti_misclickdrag', 'Fireworks__Activity_acti_misclickdragLast', 'Fireworks__Activity_acti_success_ratio', 'Fireworks__Activity_acti_success_ratioLast', 'Fireworks__Activity_acti_action_freq', 'Fireworks__Activity_acti_action_freqLast', 'Fireworks__Activity_acti_count', 'Fireworks__Activity_acti_countLast', 'Fireworks__Activity_acti_timespent', 'Fireworks__Activity_acti_timespentLast', 'Fireworks__Activity_acti_timespent_avg', 'Fireworks__Activity_acti_timespent_avgLast', 'Flower_Waterer__Activity_acti_misclickdrag_ratio', 'Flower_Waterer__Activity_acti_misclickdrag_ratioLast', 'Flower_Waterer__Activity_acti_misclickdrag', 'Flower_Waterer__Activity_acti_misclickdragLast', 'Flower_Waterer__Activity_acti_success_ratio', 'Flower_Waterer__Activity_acti_success_ratioLast', 'Flower_Waterer__Activity_acti_action_freq', 'Flower_Waterer__Activity_acti_action_freqLast', 'Flower_Waterer__Activity_acti_count', 'Flower_Waterer__Activity_acti_countLast', 'Flower_Waterer__Activity_acti_timespent', 'Flower_Waterer__Activity_acti_timespentLast', 'Flower_Waterer__Activity_acti_timespent_avg', 'Flower_Waterer__Activity_acti_timespent_avgLast', 'asse_misclickdrag_ratio', 'w0_asse_misclickdrag_ratio', 'sameworld_asse_misclickdrag_ratio', 'asse_misclickdrag', 'w0_asse_misclickdrag', 'sameworld_asse_misclickdrag', 'asse_seekhelp', 'w0_asse_seekhelp', 'sameworld_asse_seekhelp', 'asse_finish_round', 'w0_asse_finish_round', 'sameworld_asse_finish_round', 'asse_action_freq', 'w0_asse_action_freq', 'sameworld_asse_action_freq', 'asse_cnt_actions', 'w0_asse_cnt_actions', 'sameworld_asse_cnt_actions', 'asse_success_ratio', 'w0_asse_success_ratio', 'sameworld_asse_success_ratio', 'asse_true_att_freq', 'w0_asse_true_att_freq', 'sameworld_asse_true_att_freq', 'asse_true_att', 'w0_asse_true_att', 'sameworld_asse_true_att', 'asse_false_att', 'w0_asse_false_att', 'sameworld_asse_false_att', 'asse_false_att_freq', 'w0_asse_false_att_freq', 'sameworld_asse_false_att_freq', 'asse_true_att_freq_2', 'w0_asse_true_att_freq_2', 'sameworld_asse_true_att_freq_2', 'asse_correctplacement_ratio', 'w0_asse_correctplacement_ratio', 'sameworld_asse_correctplacement_ratio', 'asse_correctplacement_ratio2', 'w0_asse_correctplacement_ratio2', 'sameworld_asse_correctplacement_ratio2', 'asse_manual_accuracy', 'w0_asse_manual_accuracy', 'sameworld_asse_manual_accuracy', 'asse_manual_accuracy_grp', 'w0_asse_manual_accuracy_grp', 'sameworld_asse_manual_accuracy_grp', 'asse_accgrp_3', 'w0_asse_accgrp_3', 'sameworld_asse_accgrp_3', 'asse_timespent', 'w0_asse_timespent', 'sameworld_asse_timespent', 'asse_count', 'w0_asse_count', 'sameworld_asse_count', 'asse_timespent_avg', 'w0_asse_timespent_avg', 'sameworld_asse_timespent_avg', 'Mushroom_Sorter__Assessment_asse_misclickdrag_ratio', 'Mushroom_Sorter__Assessment_asse_misclickdrag_ratioLast', 'Mushroom_Sorter__Assessment_asse_misclickdrag', 'Mushroom_Sorter__Assessment_asse_misclickdragLast', 'Mushroom_Sorter__Assessment_asse_seekhelp', 'Mushroom_Sorter__Assessment_asse_seekhelpLast', 'Mushroom_Sorter__Assessment_asse_finish_round', 'Mushroom_Sorter__Assessment_asse_finish_roundLast', 'Mushroom_Sorter__Assessment_asse_action_freq', 'Mushroom_Sorter__Assessment_asse_action_freqLast', 'Mushroom_Sorter__Assessment_asse_cnt_actions', 'Mushroom_Sorter__Assessment_asse_cnt_actionsLast', 'Mushroom_Sorter__Assessment_asse_success_ratio', 'Mushroom_Sorter__Assessment_asse_success_ratioLast', 'Mushroom_Sorter__Assessment_asse_true_att_freq', 'Mushroom_Sorter__Assessment_asse_true_att_freqLast', 'Mushroom_Sorter__Assessment_asse_true_att', 'Mushroom_Sorter__Assessment_asse_true_attLast', 'Mushroom_Sorter__Assessment_asse_false_att', 'Mushroom_Sorter__Assessment_asse_false_attLast', 'Mushroom_Sorter__Assessment_asse_false_att_freq', 'Mushroom_Sorter__Assessment_asse_false_att_freqLast', 'Mushroom_Sorter__Assessment_asse_true_att_freq_2', 'Mushroom_Sorter__Assessment_asse_true_att_freq_2Last', 'Mushroom_Sorter__Assessment_asse_correctplacement_ratio', 'Mushroom_Sorter__Assessment_asse_correctplacement_ratioLast', 'Mushroom_Sorter__Assessment_asse_correctplacement_ratio2', 'Mushroom_Sorter__Assessment_asse_correctplacement_ratio2Last', 'Mushroom_Sorter__Assessment_asse_manual_accuracy', 'Mushroom_Sorter__Assessment_asse_manual_accuracyLast', 'Mushroom_Sorter__Assessment_asse_manual_accuracy_grp', 'Mushroom_Sorter__Assessment_asse_manual_accuracy_grpLast', 'Mushroom_Sorter__Assessment_asse_accgrp_3', 'Mushroom_Sorter__Assessment_asse_accgrp_3Last', 'Mushroom_Sorter__Assessment_asse_timespent', 'Mushroom_Sorter__Assessment_asse_timespentLast', 'Mushroom_Sorter__Assessment_asse_count', 'Mushroom_Sorter__Assessment_asse_countLast', 'Mushroom_Sorter__Assessment_asse_timespent_avg', 'Mushroom_Sorter__Assessment_asse_timespent_avgLast', 'Air_Showgame_skip_tut', 'Air_Showgame_skip_tutLast', 'Air_Showgame_success_ratio', 'Air_Showgame_success_ratioLast', 'Air_Showgame_skip_intro', 'Air_Showgame_skip_introLast', 'Air_Showgame_action_freq', 'Air_Showgame_action_freqLast', 'Air_Showgame_misclickdrag_ratio', 'Air_Showgame_misclickdrag_ratioLast', 'Air_Showgame_misclickdrag', 'Air_Showgame_misclickdragLast', 'Air_Showgame_levelbeaten', 'Air_Showgame_levelbeatenLast', 'Air_Showgame_roundbeaten', 'Air_Showgame_roundbeatenLast', 'Air_Showgame_levelbeaten_freq', 'Air_Showgame_levelbeaten_freqLast', 'Air_Showgame_roundbeaten_freq', 'Air_Showgame_roundbeaten_freqLast', 'Air_Showgame_levelbeaten_freq2', 'Air_Showgame_levelbeaten_freq2Last', 'Air_Showgame_roundbeaten_freq2', 'Air_Showgame_roundbeaten_freq2Last', 'Air_Showgame_seekhelp', 'Air_Showgame_seekhelpLast', 'Air_Showgame_manual_gameacc', 'Air_Showgame_manual_gameaccLast', 'Air_Showgame_true_att_freq', 'Air_Showgame_true_att_freqLast', 'Air_Showgame_false_att_freq', 'Air_Showgame_false_att_freqLast', 'Air_Showgame_true_att', 'Air_Showgame_true_attLast', 'Air_Showgame_false_att', 'Air_Showgame_false_attLast', 'Air_Showgame_true_att_freq_2', 'Air_Showgame_true_att_freq_2Last', 'Air_Showgame_count', 'Air_Showgame_countLast', 'Air_Showgame_timespent', 'Air_Showgame_timespentLast', 'Air_Showgame_timespent_avg', 'Air_Showgame_timespent_avgLast', 'Air_Showgame_ratio_0_miss', 'Air_Showgame_ratio_0_missLast', 'Air_Showgame_ratio_1_3_miss', 'Air_Showgame_ratio_1_3_missLast', 'Air_Showgame_ratio_3_10_miss', 'Air_Showgame_ratio_3_10_missLast', 'Air_Showgame_ratio_more_10_miss', 'Air_Showgame_ratio_more_10_missLast', 'Air_Showgame_nb_0_miss', 'Air_Showgame_nb_0_missLast', 'Air_Showgame_nb_1_3_miss', 'Air_Showgame_nb_1_3_missLast', 'Air_Showgame_nb_3_10_miss', 'Air_Showgame_nb_3_10_missLast', 'Air_Showgame_nb_more_10_miss', 'Air_Showgame_nb_more_10_missLast', 'Crystals_Rulegame_skip_tut', 'Crystals_Rulegame_skip_tutLast', 'Crystals_Rulegame_success_ratio', 'Crystals_Rulegame_success_ratioLast', 'Crystals_Rulegame_skip_intro', 'Crystals_Rulegame_skip_introLast', 'Crystals_Rulegame_action_freq', 'Crystals_Rulegame_action_freqLast', 'Crystals_Rulegame_misclickdrag_ratio', 'Crystals_Rulegame_misclickdrag_ratioLast', 'Crystals_Rulegame_misclickdrag', 'Crystals_Rulegame_misclickdragLast', 'Crystals_Rulegame_levelbeaten', 'Crystals_Rulegame_levelbeatenLast', 'Crystals_Rulegame_roundbeaten', 'Crystals_Rulegame_roundbeatenLast', 'Crystals_Rulegame_levelbeaten_freq', 'Crystals_Rulegame_levelbeaten_freqLast', 'Crystals_Rulegame_roundbeaten_freq', 'Crystals_Rulegame_roundbeaten_freqLast', 'Crystals_Rulegame_levelbeaten_freq2', 'Crystals_Rulegame_levelbeaten_freq2Last', 'Crystals_Rulegame_roundbeaten_freq2', 'Crystals_Rulegame_roundbeaten_freq2Last', 'Crystals_Rulegame_seekhelp', 'Crystals_Rulegame_seekhelpLast', 'Crystals_Rulegame_manual_gameacc', 'Crystals_Rulegame_manual_gameaccLast', 'Crystals_Rulegame_true_att_freq', 'Crystals_Rulegame_true_att_freqLast', 'Crystals_Rulegame_false_att_freq', 'Crystals_Rulegame_false_att_freqLast', 'Crystals_Rulegame_true_att', 'Crystals_Rulegame_true_attLast', 'Crystals_Rulegame_false_att', 'Crystals_Rulegame_false_attLast', 'Crystals_Rulegame_true_att_freq_2', 'Crystals_Rulegame_true_att_freq_2Last', 'Crystals_Rulegame_count', 'Crystals_Rulegame_countLast', 'Crystals_Rulegame_timespent', 'Crystals_Rulegame_timespentLast', 'Crystals_Rulegame_timespent_avg', 'Crystals_Rulegame_timespent_avgLast', 'Crystals_Rulegame_ratio_0_miss', 'Crystals_Rulegame_ratio_0_missLast', 'Crystals_Rulegame_ratio_1_3_miss', 'Crystals_Rulegame_ratio_1_3_missLast', 'Crystals_Rulegame_ratio_3_10_miss', 'Crystals_Rulegame_ratio_3_10_missLast', 'Crystals_Rulegame_ratio_more_10_miss', 'Crystals_Rulegame_ratio_more_10_missLast', 'Crystals_Rulegame_nb_0_miss', 'Crystals_Rulegame_nb_0_missLast', 'Crystals_Rulegame_nb_1_3_miss', 'Crystals_Rulegame_nb_1_3_missLast', 'Crystals_Rulegame_nb_3_10_miss', 'Crystals_Rulegame_nb_3_10_missLast', 'Crystals_Rulegame_nb_more_10_miss', 'Crystals_Rulegame_nb_more_10_missLast', 'Bug_Measurer__Activity_acti_misclickdrag_ratio', 'Bug_Measurer__Activity_acti_misclickdrag_ratioLast', 'Bug_Measurer__Activity_acti_misclickdrag', 'Bug_Measurer__Activity_acti_misclickdragLast', 'Bug_Measurer__Activity_acti_success_ratio', 'Bug_Measurer__Activity_acti_success_ratioLast', 'Bug_Measurer__Activity_acti_action_freq', 'Bug_Measurer__Activity_acti_action_freqLast', 'Bug_Measurer__Activity_acti_count', 'Bug_Measurer__Activity_acti_countLast', 'Bug_Measurer__Activity_acti_timespent', 'Bug_Measurer__Activity_acti_timespentLast', 'Bug_Measurer__Activity_acti_timespent_avg', 'Bug_Measurer__Activity_acti_timespent_avgLast', 'asse_accgrp_0', 'w0_asse_accgrp_0', 'sameworld_asse_accgrp_0', 'sameasse_asse_misclickdrag_ratio', 'sameasse_asse_misclickdrag_ratioLast', 'sameasse_asse_misclickdrag', 'sameasse_asse_misclickdragLast', 'sameasse_asse_seekhelp', 'sameasse_asse_seekhelpLast', 'sameasse_asse_finish_round', 'sameasse_asse_finish_roundLast', 'sameasse_asse_action_freq', 'sameasse_asse_action_freqLast', 'sameasse_asse_cnt_actions', 'sameasse_asse_cnt_actionsLast', 'sameasse_asse_success_ratio', 'sameasse_asse_success_ratioLast', 'sameasse_asse_true_att_freq', 'sameasse_asse_true_att_freqLast', 'sameasse_asse_true_att', 'sameasse_asse_true_attLast', 'sameasse_asse_false_att', 'sameasse_asse_false_attLast', 'sameasse_asse_false_att_freq', 'sameasse_asse_false_att_freqLast', 'sameasse_asse_true_att_freq_2', 'sameasse_asse_true_att_freq_2Last', 'sameasse_asse_correctplacement_ratio', 'sameasse_asse_correctplacement_ratioLast', 'sameasse_asse_correctplacement_ratio2', 'sameasse_asse_correctplacement_ratio2Last', 'sameasse_asse_manual_accuracy', 'sameasse_asse_manual_accuracyLast', 'sameasse_asse_manual_accuracy_grp', 'sameasse_asse_manual_accuracy_grpLast', 'sameasse_asse_accgrp_3', 'sameasse_asse_accgrp_3Last', 'sameasse_asse_timespent', 'sameasse_asse_timespentLast', 'sameasse_asse_count', 'sameasse_asse_countLast', 'sameasse_asse_timespent_avg', 'sameasse_asse_timespent_avgLast', 'Bird_Measurer__Assessment_asse_misclickdrag_ratio', 'Bird_Measurer__Assessment_asse_misclickdrag_ratioLast', 'Bird_Measurer__Assessment_asse_misclickdrag', 'Bird_Measurer__Assessment_asse_misclickdragLast', 'Bird_Measurer__Assessment_asse_seekhelp', 'Bird_Measurer__Assessment_asse_seekhelpLast', 'Bird_Measurer__Assessment_asse_finish_round', 'Bird_Measurer__Assessment_asse_finish_roundLast', 'Bird_Measurer__Assessment_asse_action_freq', 'Bird_Measurer__Assessment_asse_action_freqLast', 'Bird_Measurer__Assessment_asse_cnt_actions', 'Bird_Measurer__Assessment_asse_cnt_actionsLast', 'Bird_Measurer__Assessment_asse_success_ratio', 'Bird_Measurer__Assessment_asse_success_ratioLast', 'Bird_Measurer__Assessment_asse_true_att_freq', 'Bird_Measurer__Assessment_asse_true_att_freqLast', 'Bird_Measurer__Assessment_asse_true_att', 'Bird_Measurer__Assessment_asse_true_attLast', 'Bird_Measurer__Assessment_asse_false_att', 'Bird_Measurer__Assessment_asse_false_attLast', 'Bird_Measurer__Assessment_asse_false_att_freq', 'Bird_Measurer__Assessment_asse_false_att_freqLast', 'Bird_Measurer__Assessment_asse_true_att_freq_2', 'Bird_Measurer__Assessment_asse_true_att_freq_2Last', 'Bird_Measurer__Assessment_asse_correctplacement_ratio', 'Bird_Measurer__Assessment_asse_correctplacement_ratioLast', 'Bird_Measurer__Assessment_asse_correctplacement_ratio2', 'Bird_Measurer__Assessment_asse_correctplacement_ratio2Last', 'Bird_Measurer__Assessment_asse_manual_accuracy', 'Bird_Measurer__Assessment_asse_manual_accuracyLast', 'Bird_Measurer__Assessment_asse_manual_accuracy_grp', 'Bird_Measurer__Assessment_asse_manual_accuracy_grpLast', 'Bird_Measurer__Assessment_asse_accgrp_0', 'Bird_Measurer__Assessment_asse_accgrp_0Last', 'Bird_Measurer__Assessment_asse_timespent', 'Bird_Measurer__Assessment_asse_timespentLast', 'Bird_Measurer__Assessment_asse_count', 'Bird_Measurer__Assessment_asse_countLast', 'Bird_Measurer__Assessment_asse_timespent_avg', 'Bird_Measurer__Assessment_asse_timespent_avgLast', 'Bubble_Bathgame_skip_tut', 'Bubble_Bathgame_skip_tutLast', 'Bubble_Bathgame_success_ratio', 'Bubble_Bathgame_success_ratioLast', 'Bubble_Bathgame_skip_intro', 'Bubble_Bathgame_skip_introLast', 'Bubble_Bathgame_action_freq', 'Bubble_Bathgame_action_freqLast', 'Bubble_Bathgame_misclickdrag_ratio', 'Bubble_Bathgame_misclickdrag_ratioLast', 'Bubble_Bathgame_misclickdrag', 'Bubble_Bathgame_misclickdragLast', 'Bubble_Bathgame_levelbeaten', 'Bubble_Bathgame_levelbeatenLast', 'Bubble_Bathgame_roundbeaten', 'Bubble_Bathgame_roundbeatenLast', 'Bubble_Bathgame_levelbeaten_freq', 'Bubble_Bathgame_levelbeaten_freqLast', 'Bubble_Bathgame_roundbeaten_freq', 'Bubble_Bathgame_roundbeaten_freqLast', 'Bubble_Bathgame_levelbeaten_freq2', 'Bubble_Bathgame_levelbeaten_freq2Last', 'Bubble_Bathgame_roundbeaten_freq2', 'Bubble_Bathgame_roundbeaten_freq2Last', 'Bubble_Bathgame_seekhelp', 'Bubble_Bathgame_seekhelpLast', 'Bubble_Bathgame_manual_gameacc', 'Bubble_Bathgame_manual_gameaccLast', 'Bubble_Bathgame_true_att_freq', 'Bubble_Bathgame_true_att_freqLast', 'Bubble_Bathgame_false_att_freq', 'Bubble_Bathgame_false_att_freqLast', 'Bubble_Bathgame_true_att', 'Bubble_Bathgame_true_attLast', 'Bubble_Bathgame_false_att', 'Bubble_Bathgame_false_attLast', 'Bubble_Bathgame_true_att_freq_2', 'Bubble_Bathgame_true_att_freq_2Last', 'Bubble_Bathgame_count', 'Bubble_Bathgame_countLast', 'Bubble_Bathgame_timespent', 'Bubble_Bathgame_timespentLast', 'Bubble_Bathgame_timespent_avg', 'Bubble_Bathgame_timespent_avgLast', 'Bubble_Bathgame_ratio_0_miss', 'Bubble_Bathgame_ratio_0_missLast', 'Bubble_Bathgame_ratio_1_3_miss', 'Bubble_Bathgame_ratio_1_3_missLast', 'Bubble_Bathgame_ratio_3_10_miss', 'Bubble_Bathgame_ratio_3_10_missLast', 'Bubble_Bathgame_ratio_more_10_miss', 'Bubble_Bathgame_ratio_more_10_missLast', 'Bubble_Bathgame_nb_0_miss', 'Bubble_Bathgame_nb_0_missLast', 'Bubble_Bathgame_nb_1_3_miss', 'Bubble_Bathgame_nb_1_3_missLast', 'Bubble_Bathgame_nb_3_10_miss', 'Bubble_Bathgame_nb_3_10_missLast', 'Bubble_Bathgame_nb_more_10_miss', 'Bubble_Bathgame_nb_more_10_missLast', 'Bottle_Filler__Activity_acti_misclickdrag_ratio', 'Bottle_Filler__Activity_acti_misclickdrag_ratioLast', 'Bottle_Filler__Activity_acti_misclickdrag', 'Bottle_Filler__Activity_acti_misclickdragLast', 'Bottle_Filler__Activity_acti_success_ratio', 'Bottle_Filler__Activity_acti_success_ratioLast', 'Bottle_Filler__Activity_acti_action_freq', 'Bottle_Filler__Activity_acti_action_freqLast', 'Bottle_Filler__Activity_acti_count', 'Bottle_Filler__Activity_acti_countLast', 'Bottle_Filler__Activity_acti_timespent', 'Bottle_Filler__Activity_acti_timespentLast', 'Bottle_Filler__Activity_acti_timespent_avg', 'Bottle_Filler__Activity_acti_timespent_avgLast', 'Mushroom_Sorter__Assessment_asse_accgrp_0', 'Mushroom_Sorter__Assessment_asse_accgrp_0Last', 'sameasse_asse_accgrp_0', 'sameasse_asse_accgrp_0Last', 'asse_accgrp_2', 'w0_asse_accgrp_2', 'sameworld_asse_accgrp_2', 'Mushroom_Sorter__Assessment_asse_accgrp_2', 'Mushroom_Sorter__Assessment_asse_accgrp_2Last', 'Bird_Measurer__Assessment_asse_accgrp_3', 'Bird_Measurer__Assessment_asse_accgrp_3Last', 'Dino_Divegame_skip_tut', 'Dino_Divegame_skip_tutLast', 'Dino_Divegame_success_ratio', 'Dino_Divegame_success_ratioLast', 'Dino_Divegame_skip_intro', 'Dino_Divegame_skip_introLast', 'Dino_Divegame_action_freq', 'Dino_Divegame_action_freqLast', 'Dino_Divegame_misclickdrag_ratio', 'Dino_Divegame_misclickdrag_ratioLast', 'Dino_Divegame_misclickdrag', 'Dino_Divegame_misclickdragLast', 'Dino_Divegame_levelbeaten', 'Dino_Divegame_levelbeatenLast', 'Dino_Divegame_roundbeaten', 'Dino_Divegame_roundbeatenLast', 'Dino_Divegame_levelbeaten_freq', 'Dino_Divegame_levelbeaten_freqLast', 'Dino_Divegame_roundbeaten_freq', 'Dino_Divegame_roundbeaten_freqLast', 'Dino_Divegame_levelbeaten_freq2', 'Dino_Divegame_levelbeaten_freq2Last', 'Dino_Divegame_roundbeaten_freq2', 'Dino_Divegame_roundbeaten_freq2Last', 'Dino_Divegame_seekhelp', 'Dino_Divegame_seekhelpLast', 'Dino_Divegame_manual_gameacc', 'Dino_Divegame_manual_gameaccLast', 'Dino_Divegame_true_att_freq', 'Dino_Divegame_true_att_freqLast', 'Dino_Divegame_false_att_freq', 'Dino_Divegame_false_att_freqLast', 'Dino_Divegame_true_att', 'Dino_Divegame_true_attLast', 'Dino_Divegame_false_att', 'Dino_Divegame_false_attLast', 'Dino_Divegame_true_att_freq_2', 'Dino_Divegame_true_att_freq_2Last', 'Dino_Divegame_count', 'Dino_Divegame_countLast', 'Dino_Divegame_timespent', 'Dino_Divegame_timespentLast', 'Dino_Divegame_timespent_avg', 'Dino_Divegame_timespent_avgLast', 'Dino_Divegame_ratio_0_miss', 'Dino_Divegame_ratio_0_missLast', 'Dino_Divegame_ratio_1_3_miss', 'Dino_Divegame_ratio_1_3_missLast', 'Dino_Divegame_ratio_3_10_miss', 'Dino_Divegame_ratio_3_10_missLast', 'Dino_Divegame_ratio_more_10_miss', 'Dino_Divegame_ratio_more_10_missLast', 'Dino_Divegame_nb_0_miss', 'Dino_Divegame_nb_0_missLast', 'Dino_Divegame_nb_1_3_miss', 'Dino_Divegame_nb_1_3_missLast', 'Dino_Divegame_nb_3_10_miss', 'Dino_Divegame_nb_3_10_missLast', 'Dino_Divegame_nb_more_10_miss', 'Dino_Divegame_nb_more_10_missLast', 'w2_clip_timespent', 'w2_clip_viewratio', 'w2_clip_clipsviewed', 'w2_count_2000', 'w2_count_dif_2000', 'w2_count_2025', 'w2_count_dif_2025', 'w2_count_2030', 'w2_count_dif_2030', 'w2_count_2035', 'w2_count_dif_2035', 'w2_count_2040', 'w2_count_dif_2040', 'w2_count_2050', 'w2_count_dif_2050', 'w2_count_2060', 'w2_count_dif_2060', 'w2_count_2075', 'w2_count_dif_2075', 'w2_count_2080', 'w2_count_dif_2080', 'w2_count_2081', 'w2_count_dif_2081', 'w2_count_2083', 'w2_count_dif_2083', 'w2_count_3020', 'w2_count_dif_3020', 'w2_count_3021', 'w2_count_dif_3021', 'w2_count_3120', 'w2_count_dif_3120', 'w2_count_4050', 'w2_count_dif_4050', 'w2_count_4110', 'w2_count_dif_4110', 'w2_count_3121', 'w2_count_dif_3121', 'w2_count_4010', 'w2_count_dif_4010', 'w2_count_4020', 'w2_count_dif_4020', 'w2_count_4025', 'w2_count_dif_4025', 'w2_count_4030', 'w2_count_dif_4030', 'w2_count_4031', 'w2_count_dif_4031', 'w2_count_4035', 'w2_count_dif_4035', 'w2_count_4040', 'w2_count_dif_4040', 'w2_count_4045', 'w2_count_dif_4045', 'w2_count_4070', 'w2_count_dif_4070', 'w2_count_4090', 'w2_count_dif_4090', 'w2_count_4095', 'w2_count_dif_4095', 'w2_count_4100', 'w2_count_dif_4100', 'w2_count_4220', 'w2_count_dif_4220', 'w2_count_4235', 'w2_count_dif_4235', 'w2_count_4230', 'w2_count_dif_4230', 'w2_game_skip_tut', 'w2_game_success_ratio', 'w2_game_skip_intro', 'w2_game_action_freq', 'w2_game_misclickdrag_ratio', 'w2_game_misclickdrag', 'w2_game_levelbeaten', 'w2_game_roundbeaten', 'w2_game_levelbeaten_freq', 'w2_game_roundbeaten_freq', 'w2_game_levelbeaten_freq2', 'w2_game_roundbeaten_freq2', 'w2_game_seekhelp', 'w2_game_manual_gameacc', 'w2_game_true_att_freq', 'w2_game_false_att_freq', 'w2_game_true_att', 'w2_game_false_att', 'w2_game_true_att_freq_2', 'w2_game_count', 'w2_game_timespent', 'w2_game_timespent_avg', 'w2_game_ratio_0_miss', 'w2_game_ratio_1_3_miss', 'w2_game_ratio_3_10_miss', 'w2_game_ratio_more_10_miss', 'w2_game_nb_0_miss', 'w2_game_nb_1_3_miss', 'w2_game_nb_3_10_miss', 'w2_game_nb_more_10_miss', 'Chow_Timegame_skip_tut', 'Chow_Timegame_skip_tutLast', 'Chow_Timegame_success_ratio', 'Chow_Timegame_success_ratioLast', 'Chow_Timegame_skip_intro', 'Chow_Timegame_skip_introLast', 'Chow_Timegame_action_freq', 'Chow_Timegame_action_freqLast', 'Chow_Timegame_misclickdrag_ratio', 'Chow_Timegame_misclickdrag_ratioLast', 'Chow_Timegame_misclickdrag', 'Chow_Timegame_misclickdragLast', 'Chow_Timegame_levelbeaten', 'Chow_Timegame_levelbeatenLast', 'Chow_Timegame_roundbeaten', 'Chow_Timegame_roundbeatenLast', 'Chow_Timegame_levelbeaten_freq', 'Chow_Timegame_levelbeaten_freqLast', 'Chow_Timegame_roundbeaten_freq', 'Chow_Timegame_roundbeaten_freqLast', 'Chow_Timegame_levelbeaten_freq2', 'Chow_Timegame_levelbeaten_freq2Last', 'Chow_Timegame_roundbeaten_freq2', 'Chow_Timegame_roundbeaten_freq2Last', 'Chow_Timegame_seekhelp', 'Chow_Timegame_seekhelpLast', 'Chow_Timegame_manual_gameacc', 'Chow_Timegame_manual_gameaccLast', 'Chow_Timegame_true_att_freq', 'Chow_Timegame_true_att_freqLast', 'Chow_Timegame_false_att_freq', 'Chow_Timegame_false_att_freqLast', 'Chow_Timegame_true_att', 'Chow_Timegame_true_attLast', 'Chow_Timegame_false_att', 'Chow_Timegame_false_attLast', 'Chow_Timegame_true_att_freq_2', 'Chow_Timegame_true_att_freq_2Last', 'Chow_Timegame_count', 'Chow_Timegame_countLast', 'Chow_Timegame_timespent', 'Chow_Timegame_timespentLast', 'Chow_Timegame_timespent_avg', 'Chow_Timegame_timespent_avgLast', 'Chow_Timegame_ratio_0_miss', 'Chow_Timegame_ratio_0_missLast', 'Chow_Timegame_ratio_1_3_miss', 'Chow_Timegame_ratio_1_3_missLast', 'Chow_Timegame_ratio_3_10_miss', 'Chow_Timegame_ratio_3_10_missLast', 'Chow_Timegame_ratio_more_10_miss', 'Chow_Timegame_ratio_more_10_missLast', 'Chow_Timegame_nb_0_miss', 'Chow_Timegame_nb_0_missLast', 'Chow_Timegame_nb_1_3_miss', 'Chow_Timegame_nb_1_3_missLast', 'Chow_Timegame_nb_3_10_miss', 'Chow_Timegame_nb_3_10_missLast', 'Chow_Timegame_nb_more_10_miss', 'Chow_Timegame_nb_more_10_missLast', 'w1_asse_misclickdrag_ratio', 'w1_asse_misclickdrag', 'w1_asse_seekhelp', 'w1_asse_finish_round', 'w1_asse_action_freq', 'w1_asse_cnt_actions', 'w1_asse_success_ratio', 'w1_asse_true_att_freq', 'w1_asse_true_att', 'w1_asse_false_att', 'w1_asse_false_att_freq', 'w1_asse_true_att_freq_2', 'w1_asse_correctplacement_ratio', 'w1_asse_correctplacement_ratio2', 'w1_asse_manual_accuracy', 'w1_asse_manual_accuracy_grp', 'w1_asse_accgrp_3', 'w1_asse_timespent', 'w1_asse_count', 'w1_asse_timespent_avg', 'Cauldron_Filler__Assessment_asse_misclickdrag_ratio', 'Cauldron_Filler__Assessment_asse_misclickdrag_ratioLast', 'Cauldron_Filler__Assessment_asse_misclickdrag', 'Cauldron_Filler__Assessment_asse_misclickdragLast', 'Cauldron_Filler__Assessment_asse_seekhelp', 'Cauldron_Filler__Assessment_asse_seekhelpLast', 'Cauldron_Filler__Assessment_asse_finish_round', 'Cauldron_Filler__Assessment_asse_finish_roundLast', 'Cauldron_Filler__Assessment_asse_action_freq', 'Cauldron_Filler__Assessment_asse_action_freqLast', 'Cauldron_Filler__Assessment_asse_cnt_actions', 'Cauldron_Filler__Assessment_asse_cnt_actionsLast', 'Cauldron_Filler__Assessment_asse_success_ratio', 'Cauldron_Filler__Assessment_asse_success_ratioLast', 'Cauldron_Filler__Assessment_asse_true_att_freq', 'Cauldron_Filler__Assessment_asse_true_att_freqLast', 'Cauldron_Filler__Assessment_asse_true_att', 'Cauldron_Filler__Assessment_asse_true_attLast', 'Cauldron_Filler__Assessment_asse_false_att', 'Cauldron_Filler__Assessment_asse_false_attLast', 'Cauldron_Filler__Assessment_asse_false_att_freq', 'Cauldron_Filler__Assessment_asse_false_att_freqLast', 'Cauldron_Filler__Assessment_asse_true_att_freq_2', 'Cauldron_Filler__Assessment_asse_true_att_freq_2Last', 'Cauldron_Filler__Assessment_asse_correctplacement_ratio', 'Cauldron_Filler__Assessment_asse_correctplacement_ratioLast', 'Cauldron_Filler__Assessment_asse_correctplacement_ratio2', 'Cauldron_Filler__Assessment_asse_correctplacement_ratio2Last', 'Cauldron_Filler__Assessment_asse_manual_accuracy', 'Cauldron_Filler__Assessment_asse_manual_accuracyLast', 'Cauldron_Filler__Assessment_asse_manual_accuracy_grp', 'Cauldron_Filler__Assessment_asse_manual_accuracy_grpLast', 'Cauldron_Filler__Assessment_asse_accgrp_3', 'Cauldron_Filler__Assessment_asse_accgrp_3Last', 'Cauldron_Filler__Assessment_asse_timespent', 'Cauldron_Filler__Assessment_asse_timespentLast', 'Cauldron_Filler__Assessment_asse_count', 'Cauldron_Filler__Assessment_asse_countLast', 'Cauldron_Filler__Assessment_asse_timespent_avg', 'Cauldron_Filler__Assessment_asse_timespent_avgLast', 'w1_asse_accgrp_0', 'w2_count_4021', 'w2_count_dif_4021', 'w2_count_4022', 'w2_count_dif_4022', 'w2_acti_misclickdrag_ratio', 'w2_acti_misclickdrag', 'w2_acti_success_ratio', 'w2_acti_action_freq', 'w2_acti_count', 'w2_acti_timespent', 'w2_acti_timespent_avg', 'Cauldron_Filler__Assessment_asse_accgrp_0', 'Cauldron_Filler__Assessment_asse_accgrp_0Last', 'Chicken_Balancer__Activity_acti_misclickdrag_ratio', 'Chicken_Balancer__Activity_acti_misclickdrag_ratioLast', 'Chicken_Balancer__Activity_acti_misclickdrag', 'Chicken_Balancer__Activity_acti_misclickdragLast', 'Chicken_Balancer__Activity_acti_success_ratio', 'Chicken_Balancer__Activity_acti_success_ratioLast', 'Chicken_Balancer__Activity_acti_action_freq', 'Chicken_Balancer__Activity_acti_action_freqLast', 'Chicken_Balancer__Activity_acti_count', 'Chicken_Balancer__Activity_acti_countLast', 'Chicken_Balancer__Activity_acti_timespent', 'Chicken_Balancer__Activity_acti_timespentLast', 'Chicken_Balancer__Activity_acti_timespent_avg', 'Chicken_Balancer__Activity_acti_timespent_avgLast', 'Pan_Balancegame_skip_tut', 'Pan_Balancegame_skip_tutLast', 'Pan_Balancegame_success_ratio', 'Pan_Balancegame_success_ratioLast', 'Pan_Balancegame_skip_intro', 'Pan_Balancegame_skip_introLast', 'Pan_Balancegame_action_freq', 'Pan_Balancegame_action_freqLast', 'Pan_Balancegame_misclickdrag_ratio', 'Pan_Balancegame_misclickdrag_ratioLast', 'Pan_Balancegame_misclickdrag', 'Pan_Balancegame_misclickdragLast', 'Pan_Balancegame_levelbeaten', 'Pan_Balancegame_levelbeatenLast', 'Pan_Balancegame_roundbeaten', 'Pan_Balancegame_roundbeatenLast', 'Pan_Balancegame_levelbeaten_freq', 'Pan_Balancegame_levelbeaten_freqLast', 'Pan_Balancegame_roundbeaten_freq', 'Pan_Balancegame_roundbeaten_freqLast', 'Pan_Balancegame_levelbeaten_freq2', 'Pan_Balancegame_levelbeaten_freq2Last', 'Pan_Balancegame_roundbeaten_freq2', 'Pan_Balancegame_roundbeaten_freq2Last', 'Pan_Balancegame_seekhelp', 'Pan_Balancegame_seekhelpLast', 'Pan_Balancegame_manual_gameacc', 'Pan_Balancegame_manual_gameaccLast', 'Pan_Balancegame_true_att_freq', 'Pan_Balancegame_true_att_freqLast', 'Pan_Balancegame_false_att_freq', 'Pan_Balancegame_false_att_freqLast', 'Pan_Balancegame_true_att', 'Pan_Balancegame_true_attLast', 'Pan_Balancegame_false_att', 'Pan_Balancegame_false_attLast', 'Pan_Balancegame_true_att_freq_2', 'Pan_Balancegame_true_att_freq_2Last', 'Pan_Balancegame_count', 'Pan_Balancegame_countLast', 'Pan_Balancegame_timespent', 'Pan_Balancegame_timespentLast', 'Pan_Balancegame_timespent_avg', 'Pan_Balancegame_timespent_avgLast', 'Pan_Balancegame_ratio_0_miss', 'Pan_Balancegame_ratio_0_missLast', 'Pan_Balancegame_ratio_1_3_miss', 'Pan_Balancegame_ratio_1_3_missLast', 'Pan_Balancegame_ratio_3_10_miss', 'Pan_Balancegame_ratio_3_10_missLast', 'Pan_Balancegame_ratio_more_10_miss', 'Pan_Balancegame_ratio_more_10_missLast', 'Pan_Balancegame_nb_0_miss', 'Pan_Balancegame_nb_0_missLast', 'Pan_Balancegame_nb_1_3_miss', 'Pan_Balancegame_nb_1_3_missLast', 'Pan_Balancegame_nb_3_10_miss', 'Pan_Balancegame_nb_3_10_missLast', 'Pan_Balancegame_nb_more_10_miss', 'Pan_Balancegame_nb_more_10_missLast', 'Bird_Measurer__Assessment_asse_accgrp_2', 'Bird_Measurer__Assessment_asse_accgrp_2Last', 'sameasse_asse_accgrp_2', 'sameasse_asse_accgrp_2Last', 'Happy_Camelgame_skip_tut', 'Happy_Camelgame_skip_tutLast', 'Happy_Camelgame_success_ratio', 'Happy_Camelgame_success_ratioLast', 'Happy_Camelgame_skip_intro', 'Happy_Camelgame_skip_introLast', 'Happy_Camelgame_action_freq', 'Happy_Camelgame_action_freqLast', 'Happy_Camelgame_misclickdrag_ratio', 'Happy_Camelgame_misclickdrag_ratioLast', 'Happy_Camelgame_misclickdrag', 'Happy_Camelgame_misclickdragLast', 'Happy_Camelgame_levelbeaten', 'Happy_Camelgame_levelbeatenLast', 'Happy_Camelgame_roundbeaten', 'Happy_Camelgame_roundbeatenLast', 'Happy_Camelgame_levelbeaten_freq', 'Happy_Camelgame_levelbeaten_freqLast', 'Happy_Camelgame_roundbeaten_freq', 'Happy_Camelgame_roundbeaten_freqLast', 'Happy_Camelgame_levelbeaten_freq2', 'Happy_Camelgame_levelbeaten_freq2Last', 'Happy_Camelgame_roundbeaten_freq2', 'Happy_Camelgame_roundbeaten_freq2Last', 'Happy_Camelgame_seekhelp', 'Happy_Camelgame_seekhelpLast', 'Happy_Camelgame_manual_gameacc', 'Happy_Camelgame_manual_gameaccLast', 'Happy_Camelgame_true_att_freq', 'Happy_Camelgame_true_att_freqLast', 'Happy_Camelgame_false_att_freq', 'Happy_Camelgame_false_att_freqLast', 'Happy_Camelgame_true_att', 'Happy_Camelgame_true_attLast', 'Happy_Camelgame_false_att', 'Happy_Camelgame_false_attLast', 'Happy_Camelgame_true_att_freq_2', 'Happy_Camelgame_true_att_freq_2Last', 'Happy_Camelgame_count', 'Happy_Camelgame_countLast', 'Happy_Camelgame_timespent', 'Happy_Camelgame_timespentLast', 'Happy_Camelgame_timespent_avg', 'Happy_Camelgame_timespent_avgLast', 'Happy_Camelgame_ratio_0_miss', 'Happy_Camelgame_ratio_0_missLast', 'Happy_Camelgame_ratio_1_3_miss', 'Happy_Camelgame_ratio_1_3_missLast', 'Happy_Camelgame_ratio_3_10_miss', 'Happy_Camelgame_ratio_3_10_missLast', 'Happy_Camelgame_ratio_more_10_miss', 'Happy_Camelgame_ratio_more_10_missLast', 'Happy_Camelgame_nb_0_miss', 'Happy_Camelgame_nb_0_missLast', 'Happy_Camelgame_nb_1_3_miss', 'Happy_Camelgame_nb_1_3_missLast', 'Happy_Camelgame_nb_3_10_miss', 'Happy_Camelgame_nb_3_10_missLast', 'Happy_Camelgame_nb_more_10_miss', 'Happy_Camelgame_nb_more_10_missLast', 'w2_asse_misclickdrag_ratio', 'w2_asse_misclickdrag', 'w2_asse_seekhelp', 'w2_asse_finish_round', 'w2_asse_action_freq', 'w2_asse_cnt_actions', 'w2_asse_success_ratio', 'w2_asse_true_att_freq', 'w2_asse_true_att', 'w2_asse_false_att', 'w2_asse_false_att_freq', 'w2_asse_true_att_freq_2', 'w2_asse_correctplacement_ratio', 'w2_asse_correctplacement_ratio2', 'w2_asse_manual_accuracy', 'w2_asse_manual_accuracy_grp', 'w2_asse_accgrp_3', 'w2_asse_timespent', 'w2_asse_count', 'w2_asse_timespent_avg', 'Cart_Balancer__Assessment_asse_misclickdrag_ratio', 'Cart_Balancer__Assessment_asse_misclickdrag_ratioLast', 'Cart_Balancer__Assessment_asse_misclickdrag', 'Cart_Balancer__Assessment_asse_misclickdragLast', 'Cart_Balancer__Assessment_asse_seekhelp', 'Cart_Balancer__Assessment_asse_seekhelpLast', 'Cart_Balancer__Assessment_asse_finish_round', 'Cart_Balancer__Assessment_asse_finish_roundLast', 'Cart_Balancer__Assessment_asse_action_freq', 'Cart_Balancer__Assessment_asse_action_freqLast', 'Cart_Balancer__Assessment_asse_cnt_actions', 'Cart_Balancer__Assessment_asse_cnt_actionsLast', 'Cart_Balancer__Assessment_asse_success_ratio', 'Cart_Balancer__Assessment_asse_success_ratioLast', 'Cart_Balancer__Assessment_asse_true_att_freq', 'Cart_Balancer__Assessment_asse_true_att_freqLast', 'Cart_Balancer__Assessment_asse_true_att', 'Cart_Balancer__Assessment_asse_true_attLast', 'Cart_Balancer__Assessment_asse_false_att', 'Cart_Balancer__Assessment_asse_false_attLast', 'Cart_Balancer__Assessment_asse_false_att_freq', 'Cart_Balancer__Assessment_asse_false_att_freqLast', 'Cart_Balancer__Assessment_asse_true_att_freq_2', 'Cart_Balancer__Assessment_asse_true_att_freq_2Last', 'Cart_Balancer__Assessment_asse_correctplacement_ratio', 'Cart_Balancer__Assessment_asse_correctplacement_ratioLast', 'Cart_Balancer__Assessment_asse_correctplacement_ratio2', 'Cart_Balancer__Assessment_asse_correctplacement_ratio2Last', 'Cart_Balancer__Assessment_asse_manual_accuracy', 'Cart_Balancer__Assessment_asse_manual_accuracyLast', 'Cart_Balancer__Assessment_asse_manual_accuracy_grp', 'Cart_Balancer__Assessment_asse_manual_accuracy_grpLast', 'Cart_Balancer__Assessment_asse_accgrp_3', 'Cart_Balancer__Assessment_asse_accgrp_3Last', 'Cart_Balancer__Assessment_asse_timespent', 'Cart_Balancer__Assessment_asse_timespentLast', 'Cart_Balancer__Assessment_asse_count', 'Cart_Balancer__Assessment_asse_countLast', 'Cart_Balancer__Assessment_asse_timespent_avg', 'Cart_Balancer__Assessment_asse_timespent_avgLast', 'w2_asse_accgrp_0', 'Cart_Balancer__Assessment_asse_accgrp_0', 'Cart_Balancer__Assessment_asse_accgrp_0Last', 'Egg_Dropper__Activity_acti_misclickdrag_ratio', 'Egg_Dropper__Activity_acti_misclickdrag_ratioLast', 'Egg_Dropper__Activity_acti_misclickdrag', 'Egg_Dropper__Activity_acti_misclickdragLast', 'Egg_Dropper__Activity_acti_success_ratio', 'Egg_Dropper__Activity_acti_success_ratioLast', 'Egg_Dropper__Activity_acti_action_freq', 'Egg_Dropper__Activity_acti_action_freqLast', 'Egg_Dropper__Activity_acti_count', 'Egg_Dropper__Activity_acti_countLast', 'Egg_Dropper__Activity_acti_timespent', 'Egg_Dropper__Activity_acti_timespentLast', 'Egg_Dropper__Activity_acti_timespent_avg', 'Egg_Dropper__Activity_acti_timespent_avgLast', 'Chest_Sorter__Assessment_asse_misclickdrag_ratio', 'Chest_Sorter__Assessment_asse_misclickdrag_ratioLast', 'Chest_Sorter__Assessment_asse_misclickdrag', 'Chest_Sorter__Assessment_asse_misclickdragLast', 'Chest_Sorter__Assessment_asse_seekhelp', 'Chest_Sorter__Assessment_asse_seekhelpLast', 'Chest_Sorter__Assessment_asse_finish_round', 'Chest_Sorter__Assessment_asse_finish_roundLast', 'Chest_Sorter__Assessment_asse_action_freq', 'Chest_Sorter__Assessment_asse_action_freqLast', 'Chest_Sorter__Assessment_asse_cnt_actions', 'Chest_Sorter__Assessment_asse_cnt_actionsLast', 'Chest_Sorter__Assessment_asse_success_ratio', 'Chest_Sorter__Assessment_asse_success_ratioLast', 'Chest_Sorter__Assessment_asse_true_att_freq', 'Chest_Sorter__Assessment_asse_true_att_freqLast', 'Chest_Sorter__Assessment_asse_true_att', 'Chest_Sorter__Assessment_asse_true_attLast', 'Chest_Sorter__Assessment_asse_false_att', 'Chest_Sorter__Assessment_asse_false_attLast', 'Chest_Sorter__Assessment_asse_false_att_freq', 'Chest_Sorter__Assessment_asse_false_att_freqLast', 'Chest_Sorter__Assessment_asse_true_att_freq_2', 'Chest_Sorter__Assessment_asse_true_att_freq_2Last', 'Chest_Sorter__Assessment_asse_correctplacement_ratio', 'Chest_Sorter__Assessment_asse_correctplacement_ratioLast', 'Chest_Sorter__Assessment_asse_correctplacement_ratio2', 'Chest_Sorter__Assessment_asse_correctplacement_ratio2Last', 'Chest_Sorter__Assessment_asse_manual_accuracy', 'Chest_Sorter__Assessment_asse_manual_accuracyLast', 'Chest_Sorter__Assessment_asse_manual_accuracy_grp', 'Chest_Sorter__Assessment_asse_manual_accuracy_grpLast', 'Chest_Sorter__Assessment_asse_accgrp_0', 'Chest_Sorter__Assessment_asse_accgrp_0Last', 'Chest_Sorter__Assessment_asse_timespent', 'Chest_Sorter__Assessment_asse_timespentLast', 'Chest_Sorter__Assessment_asse_count', 'Chest_Sorter__Assessment_asse_countLast', 'Chest_Sorter__Assessment_asse_timespent_avg', 'Chest_Sorter__Assessment_asse_timespent_avgLast', 'Leaf_Leadergame_skip_tut', 'Leaf_Leadergame_skip_tutLast', 'Leaf_Leadergame_success_ratio', 'Leaf_Leadergame_success_ratioLast', 'Leaf_Leadergame_skip_intro', 'Leaf_Leadergame_skip_introLast', 'Leaf_Leadergame_action_freq', 'Leaf_Leadergame_action_freqLast', 'Leaf_Leadergame_misclickdrag_ratio', 'Leaf_Leadergame_misclickdrag_ratioLast', 'Leaf_Leadergame_misclickdrag', 'Leaf_Leadergame_misclickdragLast', 'Leaf_Leadergame_levelbeaten', 'Leaf_Leadergame_levelbeatenLast', 'Leaf_Leadergame_roundbeaten', 'Leaf_Leadergame_roundbeatenLast', 'Leaf_Leadergame_levelbeaten_freq', 'Leaf_Leadergame_levelbeaten_freqLast', 'Leaf_Leadergame_roundbeaten_freq', 'Leaf_Leadergame_roundbeaten_freqLast', 'Leaf_Leadergame_levelbeaten_freq2', 'Leaf_Leadergame_levelbeaten_freq2Last', 'Leaf_Leadergame_roundbeaten_freq2', 'Leaf_Leadergame_roundbeaten_freq2Last', 'Leaf_Leadergame_seekhelp', 'Leaf_Leadergame_seekhelpLast', 'Leaf_Leadergame_manual_gameacc', 'Leaf_Leadergame_manual_gameaccLast', 'Leaf_Leadergame_true_att_freq', 'Leaf_Leadergame_true_att_freqLast', 'Leaf_Leadergame_false_att_freq', 'Leaf_Leadergame_false_att_freqLast', 'Leaf_Leadergame_true_att', 'Leaf_Leadergame_true_attLast', 'Leaf_Leadergame_false_att', 'Leaf_Leadergame_false_attLast', 'Leaf_Leadergame_true_att_freq_2', 'Leaf_Leadergame_true_att_freq_2Last', 'Leaf_Leadergame_count', 'Leaf_Leadergame_countLast', 'Leaf_Leadergame_timespent', 'Leaf_Leadergame_timespentLast', 'Leaf_Leadergame_timespent_avg', 'Leaf_Leadergame_timespent_avgLast', 'Leaf_Leadergame_ratio_0_miss', 'Leaf_Leadergame_ratio_0_missLast', 'Leaf_Leadergame_ratio_1_3_miss', 'Leaf_Leadergame_ratio_1_3_missLast', 'Leaf_Leadergame_ratio_3_10_miss', 'Leaf_Leadergame_ratio_3_10_missLast', 'Leaf_Leadergame_ratio_more_10_miss', 'Leaf_Leadergame_ratio_more_10_missLast', 'Leaf_Leadergame_nb_0_miss', 'Leaf_Leadergame_nb_0_missLast', 'Leaf_Leadergame_nb_1_3_miss', 'Leaf_Leadergame_nb_1_3_missLast', 'Leaf_Leadergame_nb_3_10_miss', 'Leaf_Leadergame_nb_3_10_missLast', 'Leaf_Leadergame_nb_more_10_miss', 'Leaf_Leadergame_nb_more_10_missLast', 'asse_accgrp_1', 'w0_asse_accgrp_1', 'Bird_Measurer__Assessment_asse_accgrp_1', 'Bird_Measurer__Assessment_asse_accgrp_1Last', 'sameworld_asse_accgrp_1', 'sameasse_asse_accgrp_1', 'sameasse_asse_accgrp_1Last', 'w1_asse_accgrp_1', 'Cauldron_Filler__Assessment_asse_accgrp_1', 'Cauldron_Filler__Assessment_asse_accgrp_1Last', 'w2_asse_accgrp_2', 'Cart_Balancer__Assessment_asse_accgrp_2', 'Cart_Balancer__Assessment_asse_accgrp_2Last', 'Chest_Sorter__Assessment_asse_accgrp_2', 'Chest_Sorter__Assessment_asse_accgrp_2Last', 'Chest_Sorter__Assessment_asse_accgrp_3', 'Chest_Sorter__Assessment_asse_accgrp_3Last', 'w1_asse_accgrp_2', 'Cauldron_Filler__Assessment_asse_accgrp_2', 'Cauldron_Filler__Assessment_asse_accgrp_2Last', 'w2_asse_accgrp_1', 'Cart_Balancer__Assessment_asse_accgrp_1', 'Cart_Balancer__Assessment_asse_accgrp_1Last', 'Chest_Sorter__Assessment_asse_accgrp_1', 'Chest_Sorter__Assessment_asse_accgrp_1Last', 'Mushroom_Sorter__Assessment_asse_accgrp_1', 'Mushroom_Sorter__Assessment_asse_accgrp_1Last', 'title_enc']
features_1700 = [get_old_feat(f) for f in features_1700]


# In[ ]:





# In[ ]:


good_test_pseudo_samples = []
good_test_pseudo_targets = []
for ins_id, subdf in te_feat.groupby('id', sort=False):
    #if len(subdf) <= 1: continue
    for j in range(len(subdf)-1): # in [0]:
        current_title = subdf.iloc[j].title
        target_col = get_target_col(current_title)
        target = subdf.iloc[j+1][target_col]
        if target not in [0,1,2,3]: continue
        good_test_pseudo_samples.append(subdf.iloc[j].ses_id)
        good_test_pseudo_targets.append(target)

pseudo_test = te_feat[te_feat.ses_id.isin(good_test_pseudo_samples)]
pseudo_test['acc_g'] = good_test_pseudo_targets


join_tr = tr_feat[(tr_feat.label_type=='train_real')].reset_index(drop=True)
join_te = te_feat[te_feat.label_type=='test_final'].reset_index(drop=True)
pseudo_train = tr_feat[tr_feat.label_type.isin(['train_pseudo'])].reset_index(drop=True)


# In[ ]:





# In[ ]:


# FULL MODEL WITH PSEUDO
USE_PSEUDO = True

train_data =  join_tr.copy()
len_ori = len(train_data)
tr_group_idx_ori = get_idx_group(train_data)

test_data = join_te.copy()

if USE_PSEUDO: train_data = pd.concat([train_data, pseudo_test], axis=0).reset_index(drop=True)
tr_group_idx = get_idx_group(train_data)

NFOLDS = 5
SEEDS = [66, 67, 68] 

lgb_metric = eval_qwk_lgb_regr
LGB_PARAMS = {'boosting_type': 'gbdt', 'objective': 'regression',  'subsample': 0.75, 'subsample_freq': 1,
        'learning_rate': 0.04, 'feature_fraction': 1, 'num_leaves': 25, 'max_depth': 15,
        'lambda_l1': 1, 'lambda_l2': 1, 'verbose': 100,  'seed':1, 'bagging_seed':1, 'feature_fraction_seed':1
        }

# FEATURES USED
features_used_all = [f for f in train_data.columns if f not in ['id', 'ses_id', 'acc', 'acc_g', 'title', 'session_title','label_type']]
features_used = top195
cat_feats = [c for c in features_used if '_enc' in c]
print('Number of features:', len(features_used), 'Categorical features:', cat_feats)

# Initialize predictions array
y_full = train_data['acc_g'].values
X_te = test_data[features_used]
X_ps = pseudo_train[features_used]
preds_oof = np.zeros(len(train_data))
preds_test = np.zeros(len(X_te))
preds_pseudo = np.zeros(len(pseudo_train))

train_weights = np.array(get_train_weights(train_data['id'].values, weight_coefs='manual'))

# Begin training
for seed_id, seed in enumerate(SEEDS):
    ipt = np.zeros(len(features_used))
    kfold = KFold(n_splits=NFOLDS, random_state=seed, shuffle=True) 
    for fold_i, (tr_idx, va_idx) in enumerate(kfold.split(tr_group_idx)): 
 
        print('\n\nTraining fold', fold_i, 'seed', seed, '........')
        
        tr_idx = [j for sublist in tr_group_idx[tr_idx] for j in sublist]
        va_idx = [j for sublist in tr_group_idx[va_idx] for j in sublist]

        X_tr = train_data[features_used].iloc[tr_idx]
        y_tr = y_full[tr_idx]
        
        X_va = train_data[features_used].iloc[va_idx]
        y_va = y_full[va_idx]
        
        print('Xtr', X_tr.shape, 'Xva', X_va.shape)

        # LGB
        mdl = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=5000, n_jobs = -1)
        mdl.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], eval_metric=lgb_metric, verbose=-1,
               early_stopping_rounds=50, categorical_feature=cat_feats,
               sample_weight=train_weights[tr_idx])
        ipt += mdl.feature_importances_  
        preds_oof[va_idx] += mdl.predict(X_va, num_iteration = mdl.best_iteration_)/len(SEEDS)
        preds_test += mdl.predict(X_te, num_iteration = mdl.best_iteration_)/NFOLDS/len(SEEDS)
        preds_pseudo += mdl.predict(X_ps, num_iteration = mdl.best_iteration_)/NFOLDS/len(SEEDS)
        ######

ipt = pd.DataFrame(data={'feature':features_used, 'ipt':ipt}).sort_values('ipt',ascending=False)

print('rmse oof:', np.sqrt(mse(y_full[:len_ori], preds_oof[:len_ori])))
optR = OptimizedRounder(mode='accuracy_group')
optR.fit(preds_oof, y_full)
coefficients = optR.coefficients()
optim_oof = optR.predict(preds_oof[:len_ori], coefficients)
print('Seeds Avarage, Optim:', qwk(optim_oof, y_full[:len_ori]))
optim_test = optR.predict(preds_test, coefficients)
optim_pseudo = optR.predict(preds_pseudo, coefficients)
s, v = get_resample_score(optim_oof, y_full[:len_ori], tr_group_idx_ori)
print('Seeds Avarage, Optim, Resampled:', s, 'Variance', v)

kha_oof_1 = preds_oof[:len_ori].copy()
kha_oof_1_int = optim_oof.copy()
kha_test_1 = preds_test.copy()
kha_test_1_int = optim_test.copy()
kha_pseudo_1 = preds_pseudo.copy()
kha_pseudo_1_int = optim_pseudo.copy()
print(preds_to_string(kha_test_1_int))


# In[ ]:





# In[ ]:


# 5SUBMODELS WITH PSEUDO
NFOLDS = 5
SEEDS = [12, 13, 14] 
USE_PSEUDO = True

train_data =  join_tr.copy()
len_ori = len(train_data)
tr_group_idx_ori = get_idx_group(train_data)

test_data = join_te.copy()

if USE_PSEUDO: train_data = pd.concat([train_data, pseudo_test], axis=0).reset_index(drop=True)
tr_group_idx = get_idx_group(train_data)

LGB_PARAMS = {'boosting_type': 'gbdt', 'objective': 'regression',  'subsample': 0.75, 'subsample_freq': 1,
        'learning_rate': 0.04, 'feature_fraction': 0.8, 'num_leaves': 25, 'max_depth': 15,
        'lambda_l1': 1, 'lambda_l2': 1, 'verbose': 100,  'seed':1, 'bagging_seed':1, 'feature_fraction_seed':1
        }

train_weights = np.array(get_train_weights(train_data['id'].values,  weight_coefs='manual'))
y_full = train_data['acc_g'].values
preds_test2, preds_oof2 = np.zeros(len(test_data)), np.zeros(len(train_data))
tr_group_idx = get_idx_group(train_data)
features_used = features_1700

titles = list(train_data['title_enc_old'].unique())
for title in titles:
    print('Training title: ',title)
    sel_train = (train_data.title_enc_old==title)
    sel_test = (test_data.title_enc_old==title)
    
    train_data_t = train_data[sel_train].reset_index(drop=True)
    test_data_t = test_data[sel_test].reset_index(drop=True)
    X_te_t = test_data[sel_test].reset_index(drop=True)
    preds_oof_va = np.zeros(len(train_data_t))
    y_used_t = y_full[sel_train]
    
    train_weights_t = train_weights[sel_train]
    
    tr_group_idx_t = get_idx_group(train_data_t)
    te_group_idx_t = get_idx_group(test_data_t)
    print(train_data_t.shape[0],test_data_t.shape[0],X_te_t.shape[0])

    for seed_id, seed in enumerate(SEEDS):
        kfold = KFold(n_splits=NFOLDS, random_state=seed, shuffle=True)   
        for fold_i, (tr_idx, va_idx) in enumerate(kfold.split(tr_group_idx_t)): 

            print('\n\nTraining fold', fold_i, 'seed', seed, '........')

            tr_idx = [j for sublist in tr_group_idx_t[tr_idx] for j in sublist]
            va_idx = [j for sublist in tr_group_idx_t[va_idx] for j in sublist]

            X_tr, X_va = train_data_t[features_used].iloc[tr_idx], train_data_t[features_used].iloc[va_idx]
            y_tr, y_va = y_used_t[tr_idx],  y_used_t[va_idx]
            print('Xtr', X_tr.shape, 'Xva', X_va.shape)

            # LGB
            mdl = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=5000, n_jobs = -1)
            mdl.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], eval_metric=lgb_metric, verbose=-1,
                   early_stopping_rounds=50, sample_weight=train_weights_t[tr_idx])
            preds_va = mdl.predict(X_va, num_iteration = mdl.best_iteration_)
            preds_te = mdl.predict(X_te_t[features_used], num_iteration = mdl.best_iteration_)
            preds_oof_va[va_idx] += preds_va/len(SEEDS)
            preds_test2[sel_test] += preds_te/len(SEEDS)/NFOLDS
    preds_oof2[sel_train] = preds_oof_va
            ######

print('rmse oof full:', np.sqrt(mse(y_full[:len_ori], preds_oof2[:len_ori])))

# Optimize thresholds
optR = OptimizedRounder(mode='accuracy_group')
optR.fit(preds_oof2, y_full)
coefficients = optR.coefficients()
optim_oof = optR.predict(preds_oof2[:len_ori], coefficients)
optim_test = optR.predict(preds_test2, coefficients)
print(coefficients)
resample_score, resample_score_var = get_resample_score(optim_oof, y_full[:len_ori], tr_group_idx_ori)
print('Optimized Resampled Score', resample_score, 'Variance', resample_score_var)

print('5-submodel Test preds string:', preds_to_string(optim_test))

kha_oof_2 = preds_oof2[:len_ori].copy()
kha_oof_2_int = optim_oof.copy()
kha_test_2 = preds_test2.copy()
kha_test_2_int = optim_test.copy()
kha_pseudo_2 = preds_pseudo.copy()
kha_pseudo_2_int = optim_pseudo.copy()
print(preds_to_string(kha_test_2_int))


# In[ ]:





# In[ ]:





# In[ ]:


# CLASSIFIERS - NO PSEUDO
train_data =  join_tr.copy()
test_data = join_te.copy()
tr_group_idx = get_idx_group(train_data)

y_full = train_data['acc_g'].values
y_used = train_data['acc_g'].values
train_weights = np.array(get_train_weights(train_data['id'].values, weight_coefs='manual'))

NFOLDS = 7
lgb_metric = eval_qwk_lgb_regr #'rmse' # eval_qwk_lgb_regr
LGB_PARAMS = {'boosting_type': 'gbdt', 'objective': 'binary',  'metric':'auc', 'subsample': 0.75, 'subsample_freq': 1,
        'learning_rate': 0.04, 'feature_fraction': 0.9, 'num_leaves': 30, 'max_depth': 15,
        'lambda_l1': 1, 'lambda_l2': 1, 'verbose': 100,  'seed':1, 'bagging_seed':1, 'feature_fraction_seed':1
        }
features_used = features_1700
X_te = test_data[features_used]

preds_test3, preds_oof3 = np.zeros((len(X_te), 3)), np.zeros((len(train_data), 3))
SEEDS = [33, 34 ,35] 
for b1 in [0, 1, 2]:
    b2 = b1+1
    print('Training binary between:',b1,b2)
    sel_train = (y_used==b1) | (y_used==b2)
    sel_train_inv = (~sel_train)
    
    train_data_t = train_data[sel_train].reset_index(drop=True)
    test_data_t = test_data.copy()
    X_va_inv = train_data[sel_train_inv].reset_index(drop=True)
    X_te_t = test_data_t[features_used]
    preds_oof_va = np.zeros(len(train_data_t))
    preds_va_inv = np.zeros(len(X_va_inv))
    
    train_weights_t = train_weights[sel_train]

    y_used_t = y_used[sel_train]
    y_used_t[y_used[sel_train] == b1] = 0
    y_used_t[y_used[sel_train] == b2] = 1
    print(pd.Series(y_used_t).value_counts())

    tr_group_idx_t = get_idx_group(train_data_t)
    te_group_idx_t = get_idx_group(test_data_t)
    print(train_data_t.shape[0],test_data_t.shape[0],X_te_t.shape[0])
    for seed in SEEDS:
        kfold = KFold(n_splits=NFOLDS, random_state=seed, shuffle=True) 
        LGB_PARAMS['seed'] = seed
        ipt = np.zeros(len(features_used))
        for fold_i, (tr_idx, va_idx) in enumerate(kfold.split(tr_group_idx_t)): 
            print('\n\nTraining fold', fold_i, 'seed', seed, '........')

            tr_idx = [j for sublist in tr_group_idx_t[tr_idx] for j in sublist]
            va_idx = [j for sublist in tr_group_idx_t[va_idx] for j in sublist]

            X_tr, X_va = train_data_t[features_used].iloc[tr_idx], train_data_t[features_used].iloc[va_idx]
            y_tr, y_va = y_used_t[tr_idx],  y_used_t[va_idx]
            print('Xtr', X_tr.shape)
            # LGB
            mdl = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=5000, n_jobs = -1)
            mdl.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], eval_metric='auc', verbose=100,
                   early_stopping_rounds=50, categorical_feature=cat_feats,
                   sample_weight=train_weights_t[tr_idx])
            ipt += mdl.feature_importances_
            preds_va = mdl.predict(X_va, num_iteration = mdl.best_iteration_)
            preds_te = mdl.predict(X_te_t[features_used], num_iteration = mdl.best_iteration_)
            preds_va_inv += mdl.predict(X_va_inv[features_used], num_iteration = mdl.best_iteration_)/len(SEEDS)/NFOLDS
            preds_oof_va[va_idx] += preds_va/len(SEEDS)
            preds_test3[:, b1] += preds_te/len(SEEDS)/NFOLDS
    preds_oof3[sel_train, b1] = preds_oof_va
    preds_oof3[sel_train_inv, b1] = preds_va_inv
            ######

print(preds_oof3.shape, preds_test3.shape)
preds_oof31 = preds_oof3.copy()
preds_test31 = preds_test3.copy()


# In[ ]:





# In[ ]:


# CLASSIFIERS - PSEUDO
train_data =  join_tr.copy()
test_data = join_te.copy()
train_data = pd.concat([train_data, pseudo_test], axis=0).reset_index(drop=True)

tr_group_idx = get_idx_group(train_data)

y_full = train_data['acc_g'].values
y_used = train_data['acc_g'].values
train_weights = np.array(get_train_weights(train_data['id'].values, weight_coefs='manual'))

NFOLDS = 5
lgb_metric = eval_qwk_lgb_regr #'rmse' # eval_qwk_lgb_regr
LGB_PARAMS = {'boosting_type': 'gbdt', 'objective': 'binary',  'metric':'auc', 'subsample': 0.75, 'subsample_freq': 1,
        'learning_rate': 0.04, 'feature_fraction': 0.8, 'num_leaves': 40, 'max_depth': 15,
        'lambda_l1': 1, 'lambda_l2': 1, 'verbose': 100,  'seed':1, 'bagging_seed':1, 'feature_fraction_seed':1
        }
features_used = features_1700
X_te = test_data[features_used]

preds_test3, preds_oof3 = np.zeros((len(X_te), 3)), np.zeros((len(train_data), 3))
SEEDS = [4, 5, 6] 
for b1 in [0, 1, 2]:
    b2 = b1+1
    print('Training binary between:',b1,b2)
    sel_train = (y_used==b1) | (y_used==b2)
    sel_train_inv = (~sel_train)
    
    train_data_t = train_data[sel_train].reset_index(drop=True)
    test_data_t = test_data.copy()
    X_va_inv = train_data[sel_train_inv].reset_index(drop=True)
    X_te_t = test_data_t[features_used]
    preds_oof_va = np.zeros(len(train_data_t))
    preds_va_inv = np.zeros(len(X_va_inv))
    
    train_weights_t = train_weights[sel_train]

    y_used_t = y_used[sel_train]
    y_used_t[y_used[sel_train] == b1] = 0
    y_used_t[y_used[sel_train] == b2] = 1
    print(pd.Series(y_used_t).value_counts())

    tr_group_idx_t = get_idx_group(train_data_t)
    te_group_idx_t = get_idx_group(test_data_t)
    print(train_data_t.shape[0],test_data_t.shape[0],X_te_t.shape[0])
    for seed in SEEDS:
        kfold = KFold(n_splits=NFOLDS, random_state=seed, shuffle=True) 
        LGB_PARAMS['seed'] = seed
        ipt = np.zeros(len(features_used))
        for fold_i, (tr_idx, va_idx) in enumerate(kfold.split(tr_group_idx_t)): 
            print('\n\nTraining fold', fold_i, 'seed', seed, '........')

            tr_idx = [j for sublist in tr_group_idx_t[tr_idx] for j in sublist]
            va_idx = [j for sublist in tr_group_idx_t[va_idx] for j in sublist]

            X_tr, X_va = train_data_t[features_used].iloc[tr_idx], train_data_t[features_used].iloc[va_idx]
            y_tr, y_va = y_used_t[tr_idx],  y_used_t[va_idx]
            print('Xtr', X_tr.shape)
            # LGB
            mdl = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=5000, n_jobs = -1)
            mdl.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], eval_metric='auc', verbose=100,
                   early_stopping_rounds=50, categorical_feature=cat_feats,
                   sample_weight=train_weights_t[tr_idx])
            ipt += mdl.feature_importances_
            preds_va = mdl.predict(X_va, num_iteration = mdl.best_iteration_)
            preds_te = mdl.predict(X_te_t[features_used], num_iteration = mdl.best_iteration_)
            preds_va_inv += mdl.predict(X_va_inv[features_used], num_iteration = mdl.best_iteration_)/len(SEEDS)/NFOLDS
            preds_oof_va[va_idx] += preds_va/len(SEEDS)
            preds_test3[:, b1] += preds_te/len(SEEDS)/NFOLDS
    preds_oof3[sel_train, b1] = preds_oof_va
    preds_oof3[sel_train_inv, b1] = preds_va_inv
            ######

print(preds_oof3.shape, preds_test3.shape)

preds_oof32 = preds_oof3.copy()
preds_test32 = preds_test3.copy()


# In[ ]:





# In[ ]:


preds_oof3 = (preds_oof31 + preds_oof32[:len_ori])/2
preds_test3 = (preds_test31 + preds_test32)/2


# In[ ]:


# NOT PSEUDO
USE_PSEUDO = False

train_data =  join_tr.copy()
test_data = join_te.copy()

tr_group_idx = get_idx_group(train_data)

NFOLDS = 5
SEEDS = [44, 45 , 46] 

lgb_metric = eval_qwk_lgb_regr
LGB_PARAMS = {'boosting_type': 'gbdt', 'objective': 'regression',  'subsample': 0.75, 'subsample_freq': 1,
        'learning_rate': 0.04, 'feature_fraction': 1, 'num_leaves': 25, 'max_depth': 15,
        'lambda_l1': 1, 'lambda_l2': 1, 'verbose': 100,  'seed':1, 'bagging_seed':1, 'feature_fraction_seed':1
        }

# FEATURES USED
features_used_all = [f for f in train_data.columns if f not in ['id', 'ses_id', 'acc', 'acc_g', 'title', 'session_title','label_type']]
features_used = top195
cat_feats = [c for c in features_used if '_enc' in c]
print('Number of features:', len(features_used), 'Categorical features:', cat_feats)

# Initialize predictions array
y_full = train_data['acc_g'].values
X_te = test_data[features_used]
X_ps = pseudo_train[features_used]
preds_oof = np.zeros(len(train_data))
preds_test = np.zeros(len(X_te))
preds_pseudo = np.zeros(len(pseudo_train))

# Begin training
for seed_id, seed in enumerate(SEEDS):
    ipt = np.zeros(len(features_used))
    kfold = KFold(n_splits=NFOLDS, random_state=seed, shuffle=True) 
    for fold_i, (tr_idx, va_idx) in enumerate(kfold.split(tr_group_idx)): 
 
        print('\n\nTraining fold', fold_i, 'seed', seed, '........')
        
        tr_idx = [j for sublist in tr_group_idx[tr_idx] for j in sublist]
        va_idx = [j for sublist in tr_group_idx[va_idx] for j in sublist]
        
        if USE_PSEUDO: 
            X_tr = pd.concat([train_data[features_used].iloc[tr_idx], pseudo_test[features_used]], axis=0)
            y_tr = np.concatenate([y_full[tr_idx], pseudo_test['acc_g'].values])
            w = get_train_weights(np.concatenate([train_data['id'].iloc[tr_idx].values, pseudo_test['id'].values]))
        else: 
            X_tr = train_data[features_used].iloc[tr_idx]
            y_tr = y_full[tr_idx]
            w = get_train_weights(train_data['id'].iloc[tr_idx].values)
        
        X_va = train_data[features_used].iloc[va_idx]
        y_va = y_full[va_idx]
        
        print('Xtr', X_tr.shape, 'Xva', X_va.shape)

        # LGB
        mdl = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=5000, n_jobs = -1)
        mdl.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], eval_metric=lgb_metric, verbose=-1,
               early_stopping_rounds=50, categorical_feature=cat_feats,
               sample_weight=w)
        ipt += mdl.feature_importances_  
        preds_oof[va_idx] += mdl.predict(X_va, num_iteration = mdl.best_iteration_)/len(SEEDS)
        preds_test += mdl.predict(X_te, num_iteration = mdl.best_iteration_)/NFOLDS/len(SEEDS)
        preds_pseudo += mdl.predict(X_ps, num_iteration = mdl.best_iteration_)/NFOLDS/len(SEEDS)
        ######

ipt = pd.DataFrame(data={'feature':features_used, 'ipt':ipt}).sort_values('ipt',ascending=False)

print('rmse oof:', np.sqrt(mse(y_full, preds_oof)))
optR = OptimizedRounder(mode='accuracy_group')
optR.fit(preds_oof, y_full)
coefficients = optR.coefficients()
optim_oof = optR.predict(preds_oof, coefficients)
print('Seeds Avarage, Optim:', qwk(optim_oof, y_full))
optim_test = optR.predict(preds_test, coefficients)
optim_pseudo = optR.predict(preds_pseudo, coefficients)
s, v = get_resample_score(optim_oof, y_full, tr_group_idx)
print('Seeds Avarage, Optim, Resampled:', s, 'Variance', v)

kha_oof_3 = preds_oof.copy()
kha_oof_3_int = optim_oof.copy()
kha_test_3 = preds_test.copy()
kha_test_3_int = optim_test.copy()
kha_pseudo_3 = preds_pseudo.copy()
kha_pseudo_3_int = optim_pseudo.copy()
print(preds_to_string(kha_test_3_int))


# In[ ]:





# In[ ]:


# 5-SUBMODELS, NOT PSEUDO
NFOLDS = 5
SEEDS = [91, 92, 93] 
LGB_PARAMS = {'boosting_type': 'gbdt', 'objective': 'regression',  'subsample': 0.75, 'subsample_freq': 1,
        'learning_rate': 0.04, 'feature_fraction': 0.8, 'num_leaves': 25, 'max_depth': 15,
        'lambda_l1': 1, 'lambda_l2': 1, 'verbose': 100,  'seed':1, 'bagging_seed':1, 'feature_fraction_seed':1
        }
train_data =  join_tr.copy()
test_data = join_te.copy()
tr_group_idx = get_idx_group(train_data)

weights_coef = np.array([1.6533593 , 1.09794629, 0.87330317, 0.77491961, 0.57421875, 0.47301587])
# Assigning weights, old method
nb_prior = []
for g in tr_group_idx: nb_prior += list(range(len(g)))
fixed_weights = np.array([weights_coef[nb] if nb<=len(weights_coef)-1 else weights_coef[-1] for nb in nb_prior])

preds_test2, preds_oof2 = np.zeros(len(test_data)), np.zeros(len(train_data))
tr_group_idx = get_idx_group(train_data)
features_used = features_1700

titles = list(train_data['title_enc_old'].unique())
for title in titles:
    print('Training title: ',title)
    sel_train = (train_data.title_enc_old==title)
    sel_test = (test_data.title_enc_old==title)
    
    train_data_t = train_data[sel_train].reset_index(drop=True)
    test_data_t = test_data[sel_test].reset_index(drop=True)
    X_te_t = test_data[sel_test].reset_index(drop=True)
    preds_oof_va = np.zeros(len(train_data_t))
    y_used_t = y_full[sel_train]
    
    tr_group_idx_t = get_idx_group(train_data_t)
    te_group_idx_t = get_idx_group(test_data_t)
    print(train_data_t.shape[0],test_data_t.shape[0],X_te_t.shape[0])

    for seed_id, seed in enumerate(SEEDS):
        kfold = KFold(n_splits=NFOLDS, random_state=seed, shuffle=True)   
        for fold_i, (tr_idx, va_idx) in enumerate(kfold.split(tr_group_idx_t)): 

            print('\n\nTraining fold', fold_i, 'seed', seed, '........')

            tr_idx = [j for sublist in tr_group_idx_t[tr_idx] for j in sublist]
            va_idx = [j for sublist in tr_group_idx_t[va_idx] for j in sublist]

            X_tr, X_va = train_data_t[features_used].iloc[tr_idx], train_data_t[features_used].iloc[va_idx]
            y_tr, y_va = y_used_t[tr_idx],  y_used_t[va_idx]
            print('Xtr', X_tr.shape, 'Xva', X_va.shape)

            # LGB
            mdl = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=5000, n_jobs = -1)
            mdl.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], eval_metric=lgb_metric, verbose=-1,
                   early_stopping_rounds=50, sample_weight=fixed_weights[tr_idx])
            preds_va = mdl.predict(X_va, num_iteration = mdl.best_iteration_)
            preds_te = mdl.predict(X_te_t[features_used], num_iteration = mdl.best_iteration_)
            preds_oof_va[va_idx] += preds_va/len(SEEDS)
            preds_test2[sel_test] += preds_te/len(SEEDS)/NFOLDS
    preds_oof2[sel_train] = preds_oof_va
            ######

print('rmse oof full:', np.sqrt(mse(y_full, preds_oof2)))

# Optimize thresholds
optR = OptimizedRounder(mode='accuracy_group')
optR.fit(preds_oof2, y_full)
coefficients = optR.coefficients()
optim_oof = optR.predict(preds_oof2, coefficients)
optim_test = optR.predict(preds_test2, coefficients)
print(coefficients)
resample_score, resample_score_var = get_resample_score(optim_oof, y_full, tr_group_idx)
print('Optimized Resampled Score', resample_score, 'Variance', resample_score_var)

print('5-submodel Test preds string:', preds_to_string(optim_test))

kha_oof_4 = preds_oof2.copy()
kha_oof_4_int = optim_oof.copy()
kha_test_4 = preds_test2.copy()
kha_test_4_int = optim_test.copy()
kha_pseudo_4 = preds_pseudo.copy()
kha_pseudo_4_int = optim_pseudo.copy()
print(preds_to_string(kha_test_4_int))


# In[ ]:





# In[ ]:


# Merge full_model with full_model_pseudo, and 5sub with 5sub_pseudo
p_full_oof = (kha_oof_1 + kha_oof_3)/2
p_full_test = (kha_test_1 + kha_test_3)/2
optR = OptimizedRounder(mode='accuracy_group')
optR.fit(p_full_oof, y_full)
coefficients = optR.coefficients()
optim_oof = optR.predict(p_full_oof, coefficients)
optim_test = optR.predict(p_full_test, coefficients)
print(coefficients)
resample_score, resample_score_var = get_resample_score(optim_oof, y_full, tr_group_idx)
print('Optimized Resampled Score', resample_score, 'Variance', resample_score_var)
kha_merged_oof_1_int = optim_oof.copy()
kha_merged_test_1_int = optim_test.copy()

p_5sub_oof = (kha_oof_2 + kha_oof_4)/2
p_5sub_test = (kha_test_2 + kha_test_4)/2
optR = OptimizedRounder(mode='accuracy_group')
optR.fit(p_5sub_oof, y_full)
coefficients = optR.coefficients()
optim_oof = optR.predict(p_5sub_oof, coefficients)
optim_test = optR.predict(p_5sub_test, coefficients)
print(coefficients)
resample_score, resample_score_var = get_resample_score(optim_oof, y_full, tr_group_idx)
print('Optimized Resampled Score', resample_score, 'Variance', resample_score_var)
kha_merged_oof_2_int = optim_oof.copy()
kha_merged_test_2_int = optim_test.copy()


# In[ ]:





# In[ ]:


# ADVANCED CLASSIFIER
df = pd.DataFrame(preds_oof3)
y_full_original = join_tr['acc_g'].values
df['truth'] = y_full_original
df.head()

oofs = pd.DataFrame({'oof1':kha_merged_oof_1_int, 'oof2':kha_merged_oof_2_int, 'id':train_data['id']})
oofs['diff'] = np.abs(oofs['oof1'] - oofs['oof2'])
oofs = pd.concat([oofs, df], axis=1)
print(oofs['diff'].value_counts())

oofs['oof'] = -1
oofs.loc[oofs['diff'] == 0, 'oof'] = oofs.loc[oofs['diff'] == 0, 'oof1']
print(oofs['oof'].value_counts())

sel = (oofs['diff'] == 1) & ((oofs['oof1'] == 0) | (oofs['oof2'] == 0)) & (oofs[0] >= 0.2)
print('to 1:',sel.astype(int).sum())
oofs.loc[sel, 'oof'] = 1    

sel = (oofs['diff'] == 1) & ((oofs['oof1'] == 0) | (oofs['oof2'] == 0)) & (oofs[0] < 0.2)
print('to 0:',sel.astype(int).sum())
oofs.loc[sel, 'oof'] = 0   

sel = (oofs['diff'] == 1) & (((oofs['oof1'] == 1) & (oofs['oof2'] == 2)) | ((oofs['oof1'] == 2) & (oofs['oof2'] == 1))) & (oofs[1] >= 0.5)
print('to 2:',sel.astype(int).sum())
oofs.loc[sel, 'oof'] = 2    

sel = (oofs['diff'] == 1) & (((oofs['oof1'] == 1) & (oofs['oof2'] == 2)) | ((oofs['oof1'] == 2) & (oofs['oof2'] == 1))) & (oofs[1] < 0.5)
print('to 1:',sel.astype(int).sum())
oofs.loc[sel, 'oof'] = 1    

sel = (oofs['diff'] == 1) & ((oofs['oof1'] == 3) | (oofs['oof2'] == 3)) & (oofs[2] >= 0.85)
print('to 3:',sel.astype(int).sum())
oofs.loc[sel, 'oof'] = 3    

sel = (oofs['diff'] == 1) & ((oofs['oof1'] == 3) | (oofs['oof2'] == 3)) & (oofs[2] < 0.85)
print('to 2:',sel.astype(int).sum())
oofs.loc[sel, 'oof'] = 2   

oofs.loc[(oofs['diff'] > 1), 'oof'] = np.round(oofs.loc[(oofs['diff'] > 1), 'oof1'] *0.5 + oofs.loc[(oofs['diff'] > 1), 'oof2']*0.5).astype(int)

print('drop',oofs[oofs['oof'] == -1].shape)
oofs = oofs[oofs['oof'] > -1]

print('simple score:',eval_qwk_lgb_regr(oofs['truth'], oofs['oof']))
resample_score, resample_score_var = get_resample_score(oofs['oof'], oofs['truth'], get_idx_group(oofs))
print('Optimized Resampled Score', resample_score, 'Variance', resample_score_var)
oofs.to_csv('oofs3.csv', index=False)
kha_final_oof_preds_optim = oofs['oof'].copy()
    
print('\n\nTEST:\n')

oofs = pd.DataFrame({'oof1':kha_merged_test_1_int, 'oof2':kha_merged_test_2_int})
oofs['diff'] = np.abs(oofs['oof1'] - oofs['oof2'])
oofs = pd.concat([oofs, pd.DataFrame(preds_test3)], axis=1)
print(oofs['diff'].value_counts())

oofs['oof'] = -1
oofs.loc[oofs['diff'] == 0, 'oof'] = oofs.loc[oofs['diff'] == 0, 'oof1']
print(oofs['oof'].value_counts())

sel = (oofs['diff'] == 1) & ((oofs['oof1'] == 0) | (oofs['oof2'] == 0)) & (oofs[0] >= 0.2)
print('to 1:',sel.astype(int).sum())
oofs.loc[sel, 'oof'] = 1    

sel = (oofs['diff'] == 1) & ((oofs['oof1'] == 0) | (oofs['oof2'] == 0)) & (oofs[0] < 0.2)
print('to 0:',sel.astype(int).sum())
oofs.loc[sel, 'oof'] = 0   

sel = (oofs['diff'] == 1) & (((oofs['oof1'] == 1) & (oofs['oof2'] == 2)) | ((oofs['oof1'] == 2) & (oofs['oof2'] == 1))) & (oofs[1] >= 0.5)
print('to 2:',sel.astype(int).sum())
oofs.loc[sel, 'oof'] = 2    

sel = (oofs['diff'] == 1) & (((oofs['oof1'] == 1) & (oofs['oof2'] == 2)) | ((oofs['oof1'] == 2) & (oofs['oof2'] == 1))) & (oofs[1] < 0.5)
print('to 1:',sel.astype(int).sum())
oofs.loc[sel, 'oof'] = 1    

sel = (oofs['diff'] == 1) & ((oofs['oof1'] == 3) | (oofs['oof2'] == 3)) & (oofs[2] >= 0.85)
print('to 3:',sel.astype(int).sum())
oofs.loc[sel, 'oof'] = 3    

sel = (oofs['diff'] == 1) & ((oofs['oof1'] == 3) | (oofs['oof2'] == 3)) & (oofs[2] < 0.85)
print('to 2:',sel.astype(int).sum())
oofs.loc[sel, 'oof'] = 2   

oofs.loc[(oofs['diff'] > 1), 'oof'] = np.round(oofs.loc[(oofs['diff'] > 1), 'oof1'] *0.5 + oofs.loc[(oofs['diff'] > 1), 'oof2']*0.5).astype(int)
print(oofs['oof'].value_counts())

kha_final_test_preds_optim = oofs['oof'].copy()

oofs.head()
print('kha_final_test_preds_optim:', preds_to_string(kha_final_test_preds_optim))


# In[ ]:


np.save('kha_oof_1.npy', kha_oof_1)
np.save('kha_oof_1_int.npy', kha_oof_1_int) 
np.save('kha_oof_2.npy', kha_oof_2) 
np.save('kha_oof_2_int.npy', kha_oof_2_int) 
np.save('kha_test_1.npy', kha_test_1) 
np.save('kha_test_1_int.npy', kha_test_1_int) 
np.save('kha_test_2.npy', kha_test_2) 
np.save('kha_test_2_int.npy', kha_test_2_int) 
np.save('kha_oof_3.npy', kha_oof_3)
np.save('kha_oof_3_int.npy', kha_oof_3_int) 
np.save('kha_oof_4.npy', kha_oof_4) 
np.save('kha_oof_4_int.npy', kha_oof_4_int) 
np.save('kha_test_3.npy', kha_test_3) 
np.save('kha_test_3_int.npy', kha_test_3_int) 
np.save('kha_test_4.npy', kha_test_4) 
np.save('kha_test_4_int.npy', kha_test_4_int) 

np.save('kha_merged_oof_1_int.npy', kha_merged_oof_1_int) 
np.save('kha_merged_test_1_int.npy', kha_merged_test_1_int) 
np.save('kha_merged_oof_2_int.npy', kha_merged_oof_2_int) 
np.save('kha_merged_test_2_int.npy', kha_merged_test_2_int) 

np.save('kha_final_oof_preds_optim.npy', kha_final_oof_preds_optim)  
np.save('kha_final_test_preds_optim.npy', kha_final_test_preds_optim)  


# In[ ]:


sample_submission['accuracy_group'] = kha_final_test_preds_optim
sample_submission['accuracy_group'] = sample_submission['accuracy_group'].astype(int)
sample_submission.to_csv('submission_kha_agnis.csv', index=False)


# In[ ]:


test = None
tr_feat = None
del test
del tr_feat
gc.collect()


# ** EVGENY 'S PART**

# In[ ]:


import xgboost as xgb
import catboost as cb

from sklearn.metrics import log_loss,  mean_squared_error
from collections import Counter

from catboost import CatBoostClassifier, Pool, CatBoostRegressor

pd.options.display.max_columns = 300
pd.options.display.max_rows = 1000
pd.options.display.max_colwidth = 2000


# In[ ]:


path_kaggle_data = '/kaggle/input/data-science-bowl-2019/'
path_stored_data = '/kaggle/input/bowl-kha-data/'
path_stored_data2 = '/kaggle/input/dsb2019-v2/'
path_models = '/kaggle/input/dsb19-models-v1/'


# In[ ]:


def get_data():
    if (RUN_MODE=='private'):
        test = te_feat # here path to prepared dataset
    else:
        test = pd.read_csv(path_stored_data + 'kha_te_1901.csv')
        
    oof = pd.read_csv(path_stored_data2 + 'train_test4kaggle.csv')
    train_mask = oof['tr']==1
    oof = oof[train_mask].copy()

    return test, oof


# In[ ]:


def process_data(df0):
    
    def convertNA2to0(x):
        if np.isnan(x): 
            return 0
        else: 
            return x
    
    def convertNA2toM1(x):
        if np.isnan(x): 
            return -1
        else: 
            return x

    def calc_3120(x, y):
        res = round(x/y,6)
        res[(x>0) & (y==0)] = 5
        return res
        
    NA2to0_list = ["count_2000","count_2030","count_2040","count_3110","count_3120","count_3121","count_4020","count_4070","game_count_2020","game_count_2030","game_count_2040","sameworld_count_2030","Air_Show_count_3020","Air_Show_count_3021","Air_Show_count_3121","Leaf_Leader_count_2020","Leaf_Leader_count_3021","Leaf_Leader_count_4070","Crystals_Rule_count_3110","Crystals_Rule_count_3120","Scrub_A_Dub_count_2000","Scrub_A_Dub_count_2050","Scrub_A_Dub_count_2083","Scrub_A_Dub_count_4010","Scrub_A_Dub_count_4020","Pan_Balance_count_2030","Pan_Balance_count_3020","Pan_Balance_count_3120","All_Star_Sorting_count_2030","All_Star_Sorting_count_3020","All_Star_Sorting_count_4070","Dino_Dive_count_2020","Dino_Drink_count_2030","Happy_Camel_count_2030","Happy_Camel_count_3110","Happy_Camel_count_3120","Happy_Camel_count_4035","Happy_Camel_count_4095","Bubble_Bath_count_4220","Bubble_Bath_count_2080","Chow_Time_count_3121","Cauldron_Filler__Assessment__count_3120","Cauldron_Filler__Assessment__count_4040","Bird_Measurer__Assessment__count_3121","Bird_Measurer__Assessment__count_4040","Chest_Sorter__Assessment__count_3020","Chest_Sorter__Assessment__count_3021","Chest_Sorter__Assessment__count_3120","Chest_Sorter__Assessment__count_4020","Chest_Sorter__Assessment__count_4025","Cart_Balancer__Assessment__countallevents","Bottle_Filler__Activity__count_2020","Fireworks__Activity__count_4030","Bug_Measurer__Activity__count_4070","Sandcastle_Builder__Activity__count_4020","Chicken_Balancer__Activity__count_4020","Chicken_Balancer__Activity__count_4022","Cauldron_Filler__Assessment_asse_count","clip_clipsviewed","Scrub_A_Dub_countallevents","Crystal_Caves___Level_1_countallevents","Cauldron_Filler__Assessment__countallevents","sameworld_acti_countevents","asse_false_att","Cauldron_Filler__Assessment_asse_manual_accuracy_sum","Mushroom_Sorter__Assessment_asse_manual_accuracy_sum","asse_manual_accuracy_grp","asse_manual_accuracy",]
    
    NA2toM1_list =["Bird_Measurer__Assessment_asse_manual_accuracyLast","Cart_Balancer__Assessment_asse_manual_accuracyLast","Cauldron_Filler__Assessment_asse_manual_accuracyLast","Mushroom_Sorter__Assessment_asse_manual_accuracyLast","Cart_Balancer__Assessment_asse_manual_accuracy","Chest_Sorter__Assessment_asse_manual_accuracy","Bird_Measurer__Assessment_asse_manual_accuracy","Mushroom_Sorter__Assessment_asse_manual_accuracy"]
    df = df0.copy()
    
    for f in NA2to0_list:
        df[f] = df[f].map(convertNA2to0)
    
    for f in NA2toM1_list:
        df[f] = df[f].map(convertNA2toM1)

    df['ratio_4220'] = df['ratio_4220'] * 100
    df['ratio_4020'] = df['ratio_4020'] * 100

    df['event_3120_share2'] = calc_3120(df['count_3120'], df['count_3121'])
    df['count_2000_mod'] = df['count_2000'] + df['clip_clipsviewed']

    print(df.shape)
    return df


# In[ ]:


def make_test_predict_E(dt0, addproc=11, learn_mode=2, trainseeds=[5849,3990,4209]):
        
    model_dir = path_models + f"model{addproc}_{learn_mode}"
    features = list(pd.read_csv(model_dir+'/train_features.txt').columns)
    features = [i.replace('yy_', '')  for i in features]
    features = [i.replace('yy1_', '')  for i in features]
    total_models = 5 * len(trainseeds)
    x_to_pred = dt0[features]
    y_predicted = np.zeros(shape=(x_to_pred.shape[0],))
    for seed in trainseeds:
        for fold in range(1,6):
            fname = model_dir + f"/mod{addproc}_{learn_mode}_f{fold}_seed{seed}.model"
            
            if learn_mode==2:
                model_loaded = lgb.Booster(model_file=fname)
                tmp_pred = model_loaded.predict(x_to_pred.round(6), num_iteration=-1) 
            elif learn_mode==1:
                best_iter = pd.read_csv(fname + '.iter')['best_iter'][0]
                model_loaded = xgb.Booster(model_file=fname)
                tmp_pred = model_loaded.predict(xgb.DMatrix(x_to_pred.round(6)), ntree_limit=best_iter) 
            elif learn_mode==401:
                model_loaded = CatBoostRegressor()
                model_loaded.load_model(fname)
                tmp_pred = model_loaded.predict(x_to_pred.round(6), ntree_end=0)
                
            tmp_pred[tmp_pred<0] = 0
            y_predicted += tmp_pred / total_models
                
    print(  f'{addproc}  {learn_mode} ok')
    return y_predicted


# In[ ]:


def call_all_models_E(df0):
    df = df0.copy()
    df['prl'] = make_test_predict_E(test1, addproc=11, learn_mode=2, trainseeds=[5849,3990,4209])
    df['prc'] = make_test_predict_E(test1, addproc=11, learn_mode=401, trainseeds=[5849,3990,4209])
    df['prx'] = make_test_predict_E(test1, addproc=11, learn_mode=1, trainseeds=[5849,3990,4209])
    df['pr3'] = make_test_predict_E(test1, addproc=12, learn_mode=2, trainseeds=[5849,3990,4209])
    df['pr0'] = make_test_predict_E(test1, addproc=13, learn_mode=2, trainseeds=[5849,3990,4209])
    df['pr1'] = make_test_predict_E(test1, addproc=14, learn_mode=2, trainseeds=[5849,3990,4209])
    return df


# In[ ]:


def combine_predictions_E(df0, oof):
    
    def get_rank(x , x1):
        y = np.concatenate((np.array(x), np.array(x1)),axis=0)
        order = y.argsort()
        ranks = order.argsort()
        return ranks[:len(x)], ranks[len(x):]
    
    def get_bounds(targ, pred): 
        dist = Counter(targ)
        for k in dist:
            dist[k] /= len(targ)

        acum = 0
        bound = {}
        for i in range(3):
            acum += dist[i]
            bound[i] = np.percentile(pred, acum * 100)
        print(bound)
        return bound
    
    def classify(x):
        if x <= bound[0]:
            return 0
        elif x <= bound[1]:
            return 1
        elif x <= bound[2]:
            return 2
        else:
            return 3

    
    def normalize_prob(x0,x1,x3):
        y3 = np.array(x3)
        y0 = np.array(x0)
        y1 = np.array(x1)

        sum_pred013 = np.array(x0 + x1 + x3)
        sum_pred03 = np.array(x0 + x3)

        mask1 = (sum_pred013 < 1)
        y3[mask1] = x3[mask1]/sum_pred013[mask1]
        y0[mask1] = x0[mask1]/sum_pred013[mask1]
        y1[mask1] = x1[mask1]/sum_pred013[mask1]

        mask2 = (sum_pred013 > 1) & (sum_pred03 <= 1)
        y1[mask2] = 1 - sum_pred03[mask2]

        mask3 = (sum_pred013 > 1) & (sum_pred03 > 1)
        y3[mask3] = x3[mask3]/sum_pred03[mask3]
        y0[mask3] = x0[mask3]/sum_pred03[mask3]
        y1[mask3] = 0
        
        return y0, y1, y3

    df = df0.copy()
    

    df['pr0s'],df['pr1s'], df['pr3s'] = normalize_prob(df['pr0'],df['pr1'], df['pr3'])
    df['pr_cl'] = df['pr3s']*3.0 + df['pr1s']*1.8
    
    # combined ranks for test-train
    te_rl, tr_rl, = get_rank(df['prl'], oof['prl']) 
    te_rx, tr_rx, = get_rank(df['prx'], oof['prx']) 
    te_rc, tr_rc, = get_rank(df['prc'], oof['prc'])
    te_rcl, tr_rcl, = get_rank(df['pr_cl'], oof['pr_cl'])

    # predict for regression
    df['pr_reg_comb'] = te_rl + te_rx + te_rc
    df['pr_reg_comb'] = df['pr_reg_comb']/df['pr_reg_comb'].max() * 3
    oof['pr_reg_comb']  = tr_rl + tr_rx + tr_rc
    oof['pr_reg_comb'] = oof['pr_reg_comb']/oof['pr_reg_comb'].max() * 3

    # combo predict  classifier - regression  with weigthts 1*2
    df['pr_comb'] = te_rcl + (te_rl + te_rx + te_rc) * 2
    df['pr_comb'] = df['pr_comb']/df['pr_comb'].max() * 3
    oof['pr_comb'] = tr_rcl + (tr_rl + tr_rx + tr_rc) * 2
    oof['pr_comb'] = oof['pr_comb']/oof['pr_comb'].max() * 3

    # combo predict  classifier - regression  with weigthts 1*1
    df['pr_comb2'] = te_rcl + (te_rl + te_rx + te_rc) * 1
    df['pr_comb2'] = df['pr_comb2']/df['pr_comb2'].max() * 3
    oof['pr_comb2'] = tr_rcl + (tr_rl + tr_rx + tr_rc) * 1
    oof['pr_comb2'] = oof['pr_comb2']/oof['pr_comb2'].max() * 3


    #find bounds for each predict
    bound = get_bounds(oof['target'], oof['pr_cl'])
    df['pr2_cl'] = np.array(list(map(classify, df['pr_cl'])))

    bound = get_bounds(oof['target'], oof['pr_reg_comb'])
    df['pr2_reg_combk'] = np.array(list(map(classify, df['pr_reg_comb'])))

    bound = get_bounds(oof['target'], oof['pr_comb'])
    df['pr2_comb'] = np.array(list(map(classify, df['pr_comb'])))

    bound = get_bounds(oof['target'], oof['pr_comb2'])
    df['pr2_comb2'] = np.array(list(map(classify, df['pr_comb2'])))

    return df


# In[ ]:


def save_submission_E(df, pred='pr2_comb2'):
        
    test_mask=(df['label_type']=='test_final')
    submission = df[test_mask][['id', pred]].copy()
    submission = submission.rename({'id':'installation_id', pred:'accuracy_group'}, axis=1)  
    submission.to_csv('submission_evgeny.csv', index=False)
    print(submission['accuracy_group'].value_counts(normalize=True))


# In[ ]:


test, oof_E = get_data()
print(test.shape)
print(oof_E.shape)

test1 = process_data(test)

res = test1[['id','label_type']] 
print(res.shape)
res1 = call_all_models_E(res)
res2 = combine_predictions_E(res1, oof_E)


oof_E.to_csv('oof_E.csv', index=False)
save_submission_E(res2, pred='pr2_comb2')
print('Done')


# In[ ]:


# Evgeny's test (notice: should filter where label_type=='test_final' only)
print(res2.shape)
print(res2[res2.label_type=='test_final'].shape)
res2.head(10)


# In[ ]:


# Evgeny's oof
oof_E


# In[ ]:


del test
del te_feat
del test1
del res
del res1
gc.collect()


# **  MARIOS PART**

# ** MERGE**

# In[ ]:





# In[ ]:


def preds_to_string(preds):
    pred_string = ''
    for p in preds: pred_string += str(int(p))
    return pred_string

def qwk_from_raw(y_true, y_pred_in):
    """
    Fast cappa eval function for lgb.
    """
    return qwk(y_true, preds_to_int(y_pred_in))

def get_resample_score(y_preds, y_true, group_idx, times=1000, metric='qwk'):
    scores = []
    for t in range(times):
        idx_to_score = []
        for ins_idx in group_idx: idx_to_score.append(np.random.choice(ins_idx))
        if metric=='qwk': scores.append(qwk(y_preds[idx_to_score], y_true[idx_to_score]))
        elif metric=='rmse': scores.append(mse(y_preds[idx_to_score], y_true[idx_to_score]))
    return np.mean(scores), np.std(scores)

def get_idx_group(df, idcol='id'):
    group_idx = []
    for ins_id, sub_df in df.groupby(idcol, sort=False):
        idx = sub_df.index.values.tolist()
        group_idx.append(idx)
    return np.array(group_idx)

def qwk_resample(y_preds, y_true, df):
    return get_resample_score(y_preds, y_true, get_idx_group(df))

#o3 = joblib.load('/kaggle/input/bowl-data-1/submission_marios_trainv2.pkl')
# oof_E = oof_E.reset_index(drop=True)
# oofs = pd.DataFrame({'oof1':kha_final_oof_preds_optim.astype(int), \
#                      'oof1a_raw':kha_oof_1,
#                      'oof1b_raw':kha_oof_2,
#                      'oof1c_raw':kha_oof_3,
#                      'oof1d_raw':kha_oof_4,
#                      'oof1a':kha_oof_1_int,
#                      'oof1b':kha_oof_2_int,
#                      'oof2':oof_E['pr2_comb2'], \
#                      'oof2_raw':oof_E['pr_comb2'], \
#                      #'oof3':o3['accuracy_group'], \
#                      #'oof3_raw':joblib.load('/kaggle/input/bowl-data-1/marios_predictions_trainv2.pkl'), \
#                      'truth':oof_E['target'].astype(int)})
# oofs['oof1a_rank'] = oofs['oof1a_raw'].rank()
# oofs['oof1b_rank'] = oofs['oof1b_raw'].rank()
# oofs['oof2_rank'] = oofs['oof2_raw'].rank()
#oofs['oof3_rank'] = oofs['oof3_raw'].rank()
#oofs['rankAverage'] = oofs[['oof1a_rank','oof1b_rank','oof2_rank','oof3_rank']].mean(axis=1)
# print('OOFs preds loaded:',oofs.shape)
# print(oofs.head())

res2 = res2[res2['label_type'] == 'test_final'].reset_index(drop=True)
test = pd.DataFrame({'t1':kha_final_test_preds_optim.astype(int),                      't1a_raw':kha_test_1,
                     't1b_raw':kha_test_2,
                     't1c_raw':kha_test_3,
                     't1d_raw':kha_test_4,
                     't1a':kha_test_1_int,
                     't1b':kha_test_2_int,
                     't2':res2['pr2_comb2'], \
                     't2_raw':res2['pr_comb2'], \
                     #'t3':submission_marios_test['accuracy_group'], \
                     #'t3_raw':marios_predictions_test, \
                     't4': limerobot_int, \
                     't4_raw': limerobot_raw})
test['t1a_rank'] = test['t1a_raw'].rank()
test['t1b_rank'] = test['t1b_raw'].rank()
test['t2_rank'] = test['t2_raw'].rank()
test['t4_rank'] = test['t4_raw'].rank()
test['rankAverage'] = test[['t1a_rank','t1b_rank','t2_rank', 't4_rank']].mean(axis=1)
print('Test preds loaded:',test.shape)
print(test.head())

def string_to_preds(string):
    preds = []
    for i, s in enumerate(string): preds.append(int(s))
    return preds

current_best = string_to_preds('2333332301333301323311331031230201330120210133213301333103102213033320231133132021332330333221033203303203233313222223313323303233310233301333003303221010203023303302313022333313003000213232330023102333003030213103322200233003303103333311212032333300333232123313312222020130030132330113203333212323330300312323333003320333332332232003233110312103301330333233031323233333311322313333303031332113313303331221233222313103002032033312303032310230330003310233323303103333130333332323103133213303130333330332132033320223013333333201032131033123203133330320233031033131313133330323333333030310220323323332233001110332233133330030311300023330311233013132303311303330332002022013333313303011122201310031310133033323323333132332333311333303120003233333013323203333113323231331223202313023130033203333033223320302130230130033333223232203031303312303333120232303323323133131331132333130010220003333233312011333122133013223003320331322200333303312311013032230321233300320310230313212000103033131330303131300230212')
current_distr = {}
for i in range(4):
    current_distr[i] = sum([1 for x in current_best if x == i])
print('Current distribution:', current_distr)

def force_distribution(lst, distr):
    l = len(lst)
    newPreds = []
    for i in range(4):
        newPreds += [i for x in range(int(np.round(distr[i] * l / 1000)))]
    newPreds += [3]
    newPreds = newPreds[:l]
    tmp = pd.DataFrame({'preds':lst, 'origsort':np.arange(l)}).sort_values('preds')
    tmp['newPreds'] = newPreds
    tmp = tmp.sort_values('origsort')
    return tmp['newPreds'].tolist()
    
#oofs['hist'] = force_distribution(oofs['rankAverage'].tolist(), current_distr)  
test['hist'] = force_distribution(test['rankAverage'].tolist(), current_distr)  
    
#Option A - replaces labels 0->1 and 3->2 for rows where 2 model predictions differs
def merge2_conservative(df, in1, in2):
    oofs = df.copy()
    oofs['diff'] = np.abs(oofs[in1] - oofs[in2])
    oofs['oof'] = oofs[in1]
    print('Differences:')
    print(oofs['diff'].value_counts())

    cond = (oofs['diff'] > 1)
    oofs.loc[cond, 'oof'] = np.round(oofs.loc[cond, in1] *0.5 + oofs.loc[cond, in2]*0.5).astype(int)
    oofs.loc[(oofs['diff'] == 1) & (oofs['oof'] == 0), 'oof'] = 1
    oofs.loc[(oofs['diff'] == 1) & (oofs['oof'] == 3), 'oof'] = 2
    return oofs['oof']

print('\nMerging 2 and 3')
#oofs['merge_A_23'] = merge2_conservative(oofs, 'oof2', 'oof3')
test['merge_A_23'] = merge2_conservative(test, 't2', 't4')

#Option C - replaces labels 1->0 and 2->3 for rows where 2 model predictions differs
def merge2_aggressive(df, in1, in2):
    oofs = df.copy()
    oofs['diff'] = np.abs(oofs[in1] - oofs[in2])
    oofs['oof'] = oofs[in1]
    print('Differences:')
    print(oofs['diff'].value_counts())

    cond = (oofs['diff'] > 1)
    oofs.loc[cond, 'oof'] = np.round(oofs.loc[cond, in1] *0.5 + oofs.loc[cond, in2]*0.5).astype(int)
    oofs.loc[(oofs['diff'] == 1) & (oofs[in2] == 0), 'oof'] = 0
    oofs.loc[(oofs['diff'] == 1) & (oofs[in2] == 3), 'oof'] = 3
    return oofs['oof']

print('\nMerging 1 and A_23')
#oofs['merge_C'] = merge2_aggressive(oofs, 'merge_A_23', 'oof1')
test['merge_C'] = merge2_aggressive(test, 'merge_A_23', 't1')

print('\nVoting average')
#oofs['votingAvg'] = oofs[['oof1','oof2','oof3']].mean(axis=1).apply(np.round).astype(int)
test['votingAvg'] = test[['t1','t2','t4']].mean(axis=1).apply(np.round).astype(int)

print('\nSaving submission')
test['installation_id'] = sample_submission['installation_id']
test['accuracy_group'] = test['merge_C']
test['accuracy_group'] = test['accuracy_group'].astype(int)
test[['installation_id','accuracy_group']].to_csv('submission.csv', index=False)
print(preds_to_string(test['accuracy_group']))

#oofs.to_csv('merge_oofs.csv', index=False)
test.to_csv('merge_test.csv', index=False)

print('Done.')


# ** STACKING**

# In[ ]:


# #Collect train
# stack_train = pd.DataFrame(joblib.load( "/kaggle/input/bowl-data-1/x_mariosv2.pkl"))
# stack_train.columns = ['m{}'.format(x) for x in stack_train.columns]
# stack_train = pd.concat([stack_train, oofs[['oof1a_raw','oof1b_raw','oof1c_raw','oof1d_raw']], oof_E[['target','installation_id','prl','pr3','pr0','pr1','prx','prc','pr_cl']]], axis=1)
# stack_train.to_csv('stack_train.csv',index=False)
# stack_train


# In[ ]:


# #Collect test
# stack_test = pd.DataFrame(xtest_marios)
# stack_test.columns = ['m{}'.format(x) for x in stack_test.columns]
# stack_test = pd.concat([stack_test, test[['t1a_raw','t1b_raw','t1c_raw','t1d_raw']].rename({'t1a_raw':'oof1a_raw','t1b_raw':'oof1b_raw','t1c_raw':'oof1c_raw','t1d_raw':'oof1d_raw'}, axis=1), res2[['id','prl','pr3','pr0','pr1','prx','prc','pr_cl']]], axis=1)
# stack_test.to_csv('stack_test.csv',index=False)
# stack_test


# In[ ]:


# train_data =  stack_train.copy().rename({'installation_id':'id'}, axis=1)
# test_data = stack_test.copy()

# tr_group_idx = get_idx_group(train_data)

# NFOLDS = 7
# SEEDS = [66, 67, 68] 

# lgb_metric = eval_qwk_lgb_regr
# LGB_PARAMS = {'boosting_type': 'gbdt', 'objective': 'regression',  'subsample': 0.75, 'subsample_freq': 1,
#         'learning_rate': 0.04, 'feature_fraction': 0.8, 'num_leaves': 25, 'max_depth': 15,
#         'lambda_l1': 1, 'lambda_l2': 1, 'verbose': 100,  'seed':1, 'bagging_seed':1, 'feature_fraction_seed':1
#         }

# # FEATURES USED
# features_used = [f for f in train_data.columns if f not in ['id', 'installation_id', 'target']]

# # Initialize predictions array
# y_full = train_data['target'].values
# X_te = test_data[features_used]
# preds_oof = np.zeros(len(train_data))
# preds_test = np.zeros(len(X_te))

# # Begin training
# for seed_id, seed in enumerate(SEEDS):
#     ipt = np.zeros(len(features_used))
#     kfold = KFold(n_splits=NFOLDS, random_state=seed, shuffle=True) 
#     for fold_i, (tr_idx, va_idx) in enumerate(kfold.split(tr_group_idx)): 
 
#         print('\n\nTraining fold', fold_i, 'seed', seed, '........')
        
#         tr_idx = [j for sublist in tr_group_idx[tr_idx] for j in sublist]
#         va_idx = [j for sublist in tr_group_idx[va_idx] for j in sublist]
        
#         X_tr = train_data[features_used].iloc[tr_idx]
#         y_tr = y_full[tr_idx]
#         w = get_train_weights(train_data['id'].iloc[tr_idx].values)
        
#         X_va = train_data[features_used].iloc[va_idx]
#         y_va = y_full[va_idx]
        
#         print('Xtr', X_tr.shape, 'Xva', X_va.shape)

#         # LGB
#         mdl = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=5000, n_jobs = -1)
#         mdl.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], eval_metric=lgb_metric, verbose=-1,
#                early_stopping_rounds=50, #categorical_feature=['m8'],
#                sample_weight=w)
#         ipt += mdl.feature_importances_  
#         preds_oof[va_idx] += mdl.predict(X_va, num_iteration = mdl.best_iteration_)/len(SEEDS)
#         preds_test += mdl.predict(X_te, num_iteration = mdl.best_iteration_)/NFOLDS/len(SEEDS)
#         ######

# print('rmse oof:', np.sqrt(mse(y_full, preds_oof)))
# optR = OptimizedRounder(mode='accuracy_group')
# optR.fit(preds_oof, y_full)
# coefficients = optR.coefficients()
# optim_oof = optR.predict(preds_oof, coefficients)
# print(pd.Series(optim_oof).value_counts(normalize=True))
# print('Seeds Avarage, Optim:', qwk(optim_oof, y_full))
# optim_test = optR.predict(preds_test, coefficients)
# print(pd.Series(optim_test).value_counts(normalize=True))
# if RUN_MODE == 'public':
#     s, v = get_resample_score(optim_oof, y_full, tr_group_idx)
#     print('Seeds Avarage, Optim, Resampled:', s, 'Variance', v)

# #rmse oof: 0.9753928773569734
# #Seeds Avarage, Optim: 0.6216156849666088
# #Seeds Avarage, Optim, Resampled: 0.5846447888585715 Variance 0.00852699677658543

# stack_lgb_oof = np.copy(preds_oof)
# stack_lgb_test = np.copy(preds_test)
# stack_lgb_oof_int = np.copy(optim_oof)
# stack_lgb_test_int = np.copy(optim_test)

# ipt = pd.DataFrame(data={'feature':features_used, 'ipt':ipt}).sort_values('ipt',ascending=False)
# ipt.head(30)


# In[ ]:


# train_data =  stack_train.copy().rename({'installation_id':'id'}, axis=1)
# test_data = stack_test.copy()

# tr_group_idx = get_idx_group(train_data)

# NFOLDS = 7
# SEEDS = [166, 167, 168] 

# lgb_metric = 'rmse' #eval_qwk_lgb_regr
# LGB_PARAMS = {'boosting_type': 'gbdt', 'objective': 'regression',  'subsample': 0.75, 'subsample_freq': 1,
#         'learning_rate': 0.04, 'feature_fraction': 0.8, 'num_leaves': 25, 'max_depth': 15,
#         'lambda_l1': 1, 'lambda_l2': 1, 'verbose': 100,  'seed':1, 'bagging_seed':1, 'feature_fraction_seed':1
#         }

# # FEATURES USED
# features_used = [f for f in train_data.columns if f not in ['id', 'installation_id', 'target']]

# # Initialize predictions array
# y_full = train_data['target'].values
# X_te = test_data[features_used]
# preds_oof = np.zeros(len(train_data))
# preds_test = np.zeros(len(X_te))

# # Begin training
# for seed_id, seed in enumerate(SEEDS):
#     ipt = np.zeros(len(features_used))
#     kfold = KFold(n_splits=NFOLDS, random_state=seed, shuffle=True) 
#     for fold_i, (tr_idx, va_idx) in enumerate(kfold.split(tr_group_idx)): 
 
#         print('\n\nTraining fold', fold_i, 'seed', seed, '........')
        
#         tr_idx = [j for sublist in tr_group_idx[tr_idx] for j in sublist]
#         va_idx = [j for sublist in tr_group_idx[va_idx] for j in sublist]
        
#         X_tr = train_data[features_used].iloc[tr_idx]
#         y_tr = y_full[tr_idx]
#         w = get_train_weights(train_data['id'].iloc[tr_idx].values)
        
#         X_va = train_data[features_used].iloc[va_idx]
#         y_va = y_full[va_idx]
        
#         print('Xtr', X_tr.shape, 'Xva', X_va.shape)

#         # LGB
#         mdl = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=5000, n_jobs = -1)
#         mdl.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], eval_metric=lgb_metric, verbose=-1,
#                early_stopping_rounds=50, categorical_feature=['m8'],
#                sample_weight=w)
#         ipt += mdl.feature_importances_  
#         preds_oof[va_idx] += mdl.predict(X_va, num_iteration = mdl.best_iteration_)/len(SEEDS)
#         preds_test += mdl.predict(X_te, num_iteration = mdl.best_iteration_)/NFOLDS/len(SEEDS)
#         ######

# print('rmse oof:', np.sqrt(mse(y_full, preds_oof)))
# optR = OptimizedRounder(mode='accuracy_group')
# optR.fit(preds_oof, y_full)
# coefficients = optR.coefficients()
# optim_oof = optR.predict(preds_oof, coefficients)
# print(pd.Series(optim_oof).value_counts(normalize=True))
# print('Seeds Avarage, Optim:', qwk(optim_oof, y_full))
# optim_test = optR.predict(preds_test, coefficients)
# print(pd.Series(optim_test).value_counts(normalize=True))
# if RUN_MODE == 'public':
#     s, v = get_resample_score(optim_oof, y_full, tr_group_idx)
#     print('Seeds Avarage, Optim, Resampled:', s, 'Variance', v)

# #rmse oof: 0.9753928773569734
# #Seeds Avarage, Optim: 0.6216156849666088
# #Seeds Avarage, Optim, Resampled: 0.5846447888585715 Variance 0.00852699677658543

# stack_lgb2_oof = np.copy(preds_oof)
# stack_lgb2_test = np.copy(preds_test)
# stack_lgb2_oof_int = np.copy(optim_oof)
# stack_lgb2_test_int = np.copy(optim_test)

# ipt = pd.DataFrame(data={'feature':features_used, 'ipt':ipt}).sort_values('ipt',ascending=False)
# ipt.head(30)


# In[ ]:


# from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.preprocessing import StandardScaler

# train_data =  stack_train.copy().rename({'installation_id':'id'}, axis=1)
# test_data = stack_test.copy()

# tr_group_idx = get_idx_group(train_data)

# NFOLDS = 7
# SEEDS = [266, 267, 268] 

# if 1:
#     from sklearn.preprocessing import OneHotEncoder
#     enc = OneHotEncoder(handle_unknown='ignore')
#     enc.fit(pd.concat([train_data[['m8']], test_data[['m8']]]).reset_index(drop=True))
#     m8_tr = pd.DataFrame(enc.transform(train_data[['m8']]).toarray())
#     m8_tr.columns = ['m8_{}'.format(x) for x in m8_tr.columns.values]
#     m8_te = pd.DataFrame(enc.transform(test_data[['m8']]).toarray())
#     m8_te.columns = ['m8_{}'.format(x) for x in m8_te.columns.values]
#     print('before:',train_data.shape,test_data.shape,m8_tr.shape,m8_te.shape)
#     train_data = pd.concat([train_data.drop('m8', axis=1), m8_tr], axis=1)
#     test_data = pd.concat([test_data.drop('m8', axis=1), m8_te], axis=1)
#     print('after:',train_data.shape,test_data.shape)
    
# # FEATURES USED
# features_used = [f for f in train_data.columns if f not in ['id', 'installation_id', 'target']]

# scaler = StandardScaler()
# scaler.fit(pd.concat([train_data[features_used], test_data[features_used]]).reset_index(drop=True))
# SCALE = True

# # Initialize predictions array
# y_full = train_data['target'].values
# X_te = test_data[features_used]
# if SCALE:
#     X_te = scaler.transform(X_te)
# preds_oof = np.zeros(len(train_data))
# preds_test = np.zeros(len(X_te))

# # Begin training
# for seed_id, seed in enumerate(SEEDS):
#     ipt = np.zeros(len(features_used))
#     kfold = KFold(n_splits=NFOLDS, random_state=seed, shuffle=True) 
#     for fold_i, (tr_idx, va_idx) in enumerate(kfold.split(tr_group_idx)): 
 
#         print('\n\nTraining fold', fold_i, 'seed', seed, '........')
        
#         tr_idx = [j for sublist in tr_group_idx[tr_idx] for j in sublist]
#         va_idx = [j for sublist in tr_group_idx[va_idx] for j in sublist]
        
#         X_tr = train_data[features_used].iloc[tr_idx]
#         y_tr = y_full[tr_idx]
#         w = get_train_weights(train_data['id'].iloc[tr_idx].values)
        
#         X_va = train_data[features_used].iloc[va_idx]
#         y_va = y_full[va_idx]
        
#         if SCALE:
#             X_tr = scaler.transform(X_tr)
#             X_va = scaler.transform(X_va)
#         print('Xtr', X_tr.shape, 'Xva', X_va.shape)

#         # LGB
#         mdl =  ExtraTreesRegressor(n_estimators=1200, random_state=seed, max_depth=6, n_jobs=-1, \
#                                    min_samples_leaf=6, max_features=0.6, min_impurity_decrease=0.00001, \
#                                    bootstrap=False)
#         mdl.fit(X_tr, y_tr)
#         ipt += mdl.feature_importances_  
#         preds_oof[va_idx] += mdl.predict(X_va)/len(SEEDS)
#         preds_test += mdl.predict(X_te)/NFOLDS/len(SEEDS)
#         ######

# print('rmse oof:', np.sqrt(mse(y_full, preds_oof)))
# optR = OptimizedRounder(mode='accuracy_group')
# optR.fit(preds_oof, y_full)
# coefficients = optR.coefficients()
# optim_oof = optR.predict(preds_oof, coefficients)
# print(pd.Series(optim_oof).value_counts(normalize=True))
# print('Seeds Avarage, Optim:', qwk(optim_oof, y_full))
# optim_test = optR.predict(preds_test, coefficients)
# print(pd.Series(optim_test).value_counts(normalize=True))
# if RUN_MODE == 'public':
#     s, v = get_resample_score(optim_oof, y_full, tr_group_idx)
#     print('Seeds Avarage, Optim, Resampled:', s, 'Variance', v)

# stack_et_oof = np.copy(preds_oof)
# stack_et_test = np.copy(preds_test)
# stack_et_oof_int = np.copy(optim_oof)
# stack_et_test_int = np.copy(optim_test)

# #rmse oof: 0.964753320568696
# #Seeds Avarage, Optim: 0.6198868751782004
# #Seeds Avarage, Optim, Resampled: 0.5792202185015074 Variance 0.009207662896997357

# #Seeds Avarage, Optim, Resampled: 0.5812488767739824 Variance 0.0087763887848003
# #Seeds Avarage, Optim, Resampled: 0.581437109548584 Variance 0.008877791188508193


# ipt = pd.DataFrame(data={'feature':features_used, 'ipt':ipt}).sort_values('ipt',ascending=False)
# ipt.head(30)


# In[ ]:


# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler

# train_data =  stack_train.copy().rename({'installation_id':'id'}, axis=1)
# test_data = stack_test.copy()

# tr_group_idx = get_idx_group(train_data)

# NFOLDS = 7
# SEEDS = [366, 367, 368] 


# if 1:
#     from sklearn.preprocessing import OneHotEncoder
#     enc = OneHotEncoder(handle_unknown='ignore')
#     enc.fit(pd.concat([train_data[['m8']], test_data[['m8']]]).reset_index(drop=True))
#     m8_tr = pd.DataFrame(enc.transform(train_data[['m8']]).toarray())
#     m8_tr.columns = ['m8_{}'.format(x) for x in m8_tr.columns.values]
#     m8_te = pd.DataFrame(enc.transform(test_data[['m8']]).toarray())
#     m8_te.columns = ['m8_{}'.format(x) for x in m8_te.columns.values]
#     print('before:',train_data.shape,test_data.shape,m8_tr.shape,m8_te.shape)
#     train_data = pd.concat([train_data.drop('m8', axis=1), m8_tr], axis=1)
#     test_data = pd.concat([test_data.drop('m8', axis=1), m8_te], axis=1)
#     print('after:',train_data.shape,test_data.shape)

# # FEATURES USED
# features_used = [f for f in train_data.columns if f not in ['id', 'installation_id', 'target']]

# scaler = StandardScaler()
# scaler.fit(pd.concat([train_data[features_used], test_data[features_used]]).reset_index(drop=True))
# SCALE = True

# # Initialize predictions array
# y_full = train_data['target'].values
# X_te = test_data[features_used]
# if SCALE:
#     X_te = scaler.transform(X_te)
# preds_oof = np.zeros(len(train_data))
# preds_test = np.zeros(len(X_te))

# # Begin training
# for seed_id, seed in enumerate(SEEDS):
#     ipt = np.zeros(len(features_used))
#     kfold = KFold(n_splits=NFOLDS, random_state=seed, shuffle=True) 
#     for fold_i, (tr_idx, va_idx) in enumerate(kfold.split(tr_group_idx)): 
 
#         print('\n\nTraining fold', fold_i, 'seed', seed, '........')
        
#         tr_idx = [j for sublist in tr_group_idx[tr_idx] for j in sublist]
#         va_idx = [j for sublist in tr_group_idx[va_idx] for j in sublist]
        
#         X_tr = train_data[features_used].iloc[tr_idx]
#         y_tr = y_full[tr_idx]
#         w = get_train_weights(train_data['id'].iloc[tr_idx].values)
        
#         X_va = train_data[features_used].iloc[va_idx]
#         y_va = y_full[va_idx]
        
#         if SCALE:
#             X_tr = scaler.transform(X_tr)
#             X_va = scaler.transform(X_va)
#         print('Xtr', X_tr.shape, 'Xva', X_va.shape)

#         # LGB
#         mdl =  LinearRegression(n_jobs=-1)
#         mdl.fit(X_tr, y_tr)
#         preds_oof[va_idx] += mdl.predict(X_va)/len(SEEDS)
#         preds_test += mdl.predict(X_te)/NFOLDS/len(SEEDS)
#         ######

# print('rmse oof:', np.sqrt(mse(y_full, preds_oof)))
# optR = OptimizedRounder(mode='accuracy_group')
# optR.fit(preds_oof, y_full)
# coefficients = optR.coefficients()
# optim_oof = optR.predict(preds_oof, coefficients)
# print(pd.Series(optim_oof).value_counts(normalize=True))
# print('Seeds Avarage, Optim:', qwk(optim_oof, y_full))
# optim_test = optR.predict(preds_test, coefficients)
# print(pd.Series(optim_test).value_counts(normalize=True))
# if RUN_MODE == 'public':
#     s, v = get_resample_score(optim_oof, y_full, tr_group_idx)
#     print('Seeds Avarage, Optim, Resampled:', s, 'Variance', v)

# stack_lnr_oof = np.copy(preds_oof)
# stack_lnr_test = np.copy(preds_test)
# stack_lnr_oof_int = np.copy(optim_oof)
# stack_lnr_test_int = np.copy(optim_test)


# In[ ]:


# blend_of_stack_oof = (stack_lgb_oof + stack_lgb2_oof + stack_et_oof + stack_lnr_oof) / 4
# blend_of_stack_test = (stack_lgb_test + stack_lgb2_test + stack_et_test + stack_lnr_test) / 4
# #blend_of_stack_oof = (stack_lgb_oof + stack_lgb2_oof + stack_et_oof) / 3
# #blend_of_stack_test = (stack_lgb_test + stack_lgb2_test + stack_et_test) / 3

# print('rmse oof:', np.sqrt(mse(y_full, blend_of_stack_oof)))
# optR = OptimizedRounder(mode='accuracy_group')
# optR.fit(blend_of_stack_oof, y_full)
# coefficients = optR.coefficients()
# optim_oof = optR.predict(blend_of_stack_oof, coefficients)
# print(pd.Series(optim_oof).value_counts(normalize=True))
# print('Seeds Avarage, Optim:', qwk(optim_oof, y_full))
# optim_test = optR.predict(blend_of_stack_test, coefficients)
# print(pd.Series(optim_test).value_counts(normalize=True))
# if RUN_MODE == 'public':
#     s, v = get_resample_score(optim_oof, y_full, tr_group_idx)
#     print('Seeds Avarage, Optim, Resampled:', s, 'Variance', v)
# #Seeds Avarage, Optim, Resampled: 0.580669010411863 Variance 0.008937232060557696
# #Seeds Avarage, Optim, Resampled: 0.5820111868211523 Variance 0.008726096802174342


# print('\nSaving submission')
# test['accuracy_group'] = optim_test
# test[['installation_id','accuracy_group']].to_csv('submission.csv', index=False)
# print(preds_to_string(test['accuracy_group']))

