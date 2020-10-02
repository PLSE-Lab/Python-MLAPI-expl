from tqdm import tqdm
from typing import Union
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
from datetime import timedelta
import sklearn
import datetime

from datetime import timedelta

timestring = lambda : datetime.datetime.now().strftime("%H_%M_%S")

pred_steps=28

class WRMSSEEvaluator(object):
    
    group_ids = ( 'all_id', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id',
        ['state_id', 'cat_id'],  ['state_id', 'dept_id'], ['store_id', 'cat_id'],
        ['store_id', 'dept_id'], ['item_id', 'state_id'], ['item_id', 'store_id'])

    def __init__(self, 
                 train_df: pd.DataFrame, 
                 valid_df: pd.DataFrame, 
                 calendar: pd.DataFrame, 
                 prices: pd.DataFrame):
        '''
        intialize and calculate weights
        '''
        self.calendar = calendar
        self.prices = prices
        self.train_df = train_df.copy()
        self.valid_df = valid_df
        self.train_target_columns = [i for i in self.train_df.columns if i.startswith('d_')]
        self.weight_columns = self.train_df.iloc[:, -pred_steps:].columns.tolist()

        self.train_df['all_id'] = "all"

        self.id_columns = [i for i in self.train_df.columns if not i.startswith('d_')]
        self.valid_target_columns = [i for i in self.valid_df.columns if i.startswith('d_')]

        if not all([c in self.valid_df.columns for c in self.id_columns]):
            self.valid_df = pd.concat([self.train_df[self.id_columns], self.valid_df],
                                      axis=1, 
                                      sort=False)
        self.train_series = self.trans_30490_to_42840(self.train_df, 
                                                      self.train_target_columns, 
                                                      self.group_ids)
        self.valid_series = self.trans_30490_to_42840(self.valid_df, 
                                                      self.valid_target_columns, 
                                                      self.group_ids)
        self.weights = self.get_weight_df()
        self.scale = self.get_scale()
        # self.train_series = None
        # self.train_df = None
        # self.prices = None
        # self.calendar = None

    def get_scale(self):
        '''
        scaling factor for each series ignoring starting zeros
        '''
        scales = []
        for i in tqdm(range(len(self.train_series))):
            series = self.train_series.iloc[i].values
            series = series[np.argmax(series!=0):]
            scale = ((series[1:] - series[:-1]) ** 2).mean()
            scales.append(scale)
        return np.array(scales)
    
    def get_name(self, i):
        '''
        convert a str or list of strings to unique string 
        used for naming each of 42840 series
        '''
        if type(i) == str or type(i) == int:
            return str(i)
        else:
            return "_".join(i)
    
    def get_weight_df(self) -> pd.DataFrame:
        """
        returns weights for each of 42840 series in a dataFrame
        """
        day_to_week = self.calendar.set_index("d")["wm_yr_wk"].to_dict()
        weight_df = self.train_df[["item_id", "store_id"] + self.weight_columns].set_index(
            ["item_id", "store_id"]
        )
        weight_df = (
            weight_df.stack().reset_index().rename(columns={"level_2": "d", 0: "value"})
        )
        weight_df["wm_yr_wk"] = weight_df["d"].map(day_to_week)
        weight_df = weight_df.merge(
            self.prices, how="left", on=["item_id", "store_id", "wm_yr_wk"]
        )
        weight_df["value"] = weight_df["value"] * weight_df["sell_price"]
        weight_df = weight_df.set_index(["item_id", "store_id", "d"]).unstack(level=2)[
            "value"
        ]
        weight_df = weight_df.loc[
            zip(self.train_df.item_id, self.train_df.store_id), :
        ].reset_index(drop=True)
        weight_df = pd.concat(
            [self.train_df[self.id_columns], weight_df], axis=1, sort=False
        )
        weights_map = {}
        for i, group_id in enumerate(tqdm(self.group_ids, leave=False)):
            lv_weight = weight_df.groupby(group_id)[self.weight_columns].sum().sum(axis=1)
            lv_weight = lv_weight / lv_weight.sum()
            for i in range(len(lv_weight)):
                weights_map[self.get_name(lv_weight.index[i])] = np.array(
                    [lv_weight.iloc[i]]
                )
        weights = pd.DataFrame(weights_map).T / len(self.group_ids)

        return weights

    def trans_30490_to_42840(self, df, cols, group_ids, dis=False):
        '''
        transform 30490 sries to all 42840 series
        '''
        series_map = {}
        for i, group_id in enumerate(tqdm(self.group_ids, leave=False, disable=dis)):
            tr = df.groupby(group_id)[cols].sum()
            for i in range(len(tr)):
                series_map[self.get_name(tr.index[i])] = tr.iloc[i].values
        return pd.DataFrame(series_map).T
    
    def get_rmsse(self, valid_preds) -> pd.Series:
        '''
        returns rmsse scores for all 42840 series
        '''
        score = ((self.valid_series - valid_preds) ** 2).mean(axis=1)
        rmsse = (score / self.scale).map(np.sqrt)
        return rmsse

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds],
                                axis=1, 
                                sort=False)
        valid_preds = self.trans_30490_to_42840(valid_preds, 
                                                self.valid_target_columns, 
                                                self.group_ids, 
                                                True)
        self.rmsse = self.get_rmsse(valid_preds)
        self.contributors = pd.concat([self.weights, self.rmsse], 
                                      axis=1, 
                                      sort=False).prod(axis=1)
        return np.sum(self.contributors)

def get_time_block_series(series_array, date_to_index, start_date, end_date):
    
    inds = date_to_index[start_date:end_date]
    return series_array[:,inds]

def transform_series_encode(series_array):
    series_array = np.log1p(series_array) 
    series_mean = series_array.mean(axis=1).reshape(-1,1) 
    series_array = series_array - series_mean
    series_array = series_array.reshape((series_array.shape[0],series_array.shape[1], 1))
    return series_array, series_mean

def transform_series_decode(series_array, encode_series_mean):
    series_array = np.log1p(series_array) 
    series_array = series_array - encode_series_mean
    series_array = series_array.reshape((series_array.shape[0],series_array.shape[1], 1))
    return series_array

def untransform_series_decode(series_array, encode_series_mean):
    series_array = series_array.reshape((series_array.shape[0],series_array.shape[1]))
    series_array = series_array + encode_series_mean
    series_array = np.expm1(series_array) 
    return series_array#.astype(int)

def get_all_data(date_to_index, series_array, enc_start_date, enc_end_date, pred_start_date, pred_end_date, shuffle=True, n_samples=100000):

    # sample of series from train_enc_start to train_enc_end  
    encoder_input_data = get_time_block_series(series_array, date_to_index, enc_start_date, enc_end_date)[:n_samples]
    encoder_input_data, encode_series_mean = transform_series_encode(encoder_input_data)

    # sample of series from train_pred_start to train_pred_end 
    decoder_target_data = get_time_block_series(series_array, date_to_index, pred_start_date, pred_end_date)[:n_samples]
    decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean)

    # lagged target series for teacher forcing
    decoder_input_data = np.zeros(decoder_target_data.shape)
    decoder_input_data[:,1:,0] = decoder_target_data[:,:-1,0]
    decoder_input_data[:,0,0] = encoder_input_data[:,-1,0]
    
    if shuffle:
        encoder_input_data = sklearn.utils.shuffle(encoder_input_data, random_state=42)
        decoder_input_data = sklearn.utils.shuffle(decoder_input_data, random_state=42)
        decoder_target_data = sklearn.utils.shuffle(decoder_target_data, random_state=42)
        
    return encoder_input_data, encode_series_mean, decoder_input_data, decoder_target_data

class score_callback(tf.keras.callbacks.Callback):
    
    def __init__(self, e, ids, val_encoder_input, val_encode_series_mean):
        
        self.e = e
        self.ids = ids
        self.val_encoder_input = val_encoder_input
        self.encode_series_mean = val_encode_series_mean
        
    def on_epoch_end(self, epoch, logs):
        
        predictions = self.model.predict(self.val_encoder_input)
        predictions = untransform_series_decode(predictions, self.encode_series_mean)
        predictions = pd.DataFrame(predictions)
        predictions.columns = ['d_' + str(x+1886) for x in np.arange(pred_steps)]
        predictions['id'] = self.ids
        predictions = predictions.merge(self.e.valid_df[['id']], on='id', how='right')
        predictions = predictions[['id']+list(predictions.columns[:-1])]
        predictions = predictions.iloc[:,1:]
        logs['score'] = self.e.score(predictions)
        print("")
        print("Current score is: " + str(logs['score']))
        
def make_callbacks(model_name, run_name, do_cp=False, do_tb=True):
    
    callbacks = []
    if do_cp:
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
        callbacks.append(checkpoint_cb)
    if do_tb:
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{run_name}_'+timestring(), histogram_freq=0, write_graph=False, write_images=False, update_freq='epoch', profile_batch=0, embeddings_freq=0)
        callbacks.append(tensorboard_cb)
    
    return callbacks

def make_training_plot(history):
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Squared Error Loss')
    plot1 = ax1.plot(history.history['loss'], label='Train', color='red')
    ax1.set_title('Loss Over Time')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    plot2 = ax2.plot(history.history['val_loss'], label='Valid',  color='blue')

    lns = plot1+plot2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    
def decode_sequence(enc_model, dec_model, input_seq):
    
    # Encode the input as state vectors.
    states_value = enc_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, 1))
    
    # Populate the first target sequence with end of encoding series pageviews
    target_seq[0, 0, 0] = input_seq[0, -1, 0]

    # Sampling loop for a batch of sequences - we will fill decoded_seq with predictions
    # (to simplify, here we assume a batch of size 1).

    decoded_seq = np.zeros((1,pred_steps,1))
    
    for i in range(pred_steps):
        
        output, h, c = dec_model.predict([target_seq] + states_value)
        
        decoded_seq[0,i,0] = output[0,0,0]

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, 1))
        target_seq[0, 0, 0] = output[0,0,0]

        # Update states
        states_value = [h, c]

    return decoded_seq

def predict_sequence(model, input_sequence):

    history_sequence = input_sequence.copy()
    pred_sequence = np.zeros((input_sequence.shape[0],pred_steps,1)) # initialize output (pred_steps time steps)  
    
    for i in range(pred_steps):
        # record next time step prediction (last time step of model output) 
        last_step_pred = model.predict(history_sequence)[:,-1,0]
        pred_sequence[:,i,0] = last_step_pred
        
        # add the next time step prediction to the history sequence
        history_sequence = np.concatenate([history_sequence, last_step_pred.reshape(-1,1,1)], axis=1)

    return pred_sequence

def predict_and_plot(model, encoder_input_data, decoder_target_data, encode_series_mean, sample_ind, enc_tail_len=250, lstm=False, dec_model=None, nbeats=False,):

    encode_series = encoder_input_data[sample_ind:sample_ind+1,:,:]
    target_series = decoder_target_data[sample_ind,:,:1]
    encode_series_mean = encode_series_mean[sample_ind:sample_ind+1,:]
    
    if lstm:
        pred_series = decode_sequence(model, dec_model, encode_series)
    elif nbeats:
        pred_series = model.predict(encode_series)
    else:
        pred_series = predict_sequence(model, encode_series)
    
    pred_series = pred_series.reshape(-1,1)
    
    encode_series = untransform_series_decode(encode_series, encode_series_mean)
    pred_series = untransform_series_decode(pred_series, encode_series_mean[-pred_steps:,:])
    target_series = untransform_series_decode(target_series, encode_series_mean[-pred_steps:,:])
    
    encode_series = encode_series.reshape(-1,1)
    pred_series = pred_series.reshape(-1,1)
    target_series = target_series.reshape(-1,1) 
    
    encode_series_tail = np.concatenate([encode_series[-enc_tail_len:],target_series[:1]])
    x_encode = encode_series_tail.shape[0]
    
    plt.figure(figsize=(10,6))   
    
    plt.plot(range(1,x_encode+1),encode_series_tail)
    plt.plot(range(x_encode,x_encode+pred_steps),target_series,color='orange')
    plt.plot(range(x_encode,x_encode+pred_steps),pred_series,color='teal',linestyle='--')
    
    plt.title('Encoder Series Tail of Length %d, Target Series, and Predictions' % enc_tail_len)
    plt.legend(['Encoding Series','Target Series','Predictions'])