from typing import Union
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
from datetime import timedelta

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate
from tensorflow.keras.optimizers import Adam

pred_steps=28

def create_enc_dec(learning_rate, hidden_size = 32, dropout = 0.20):
    enc_model, encoder_inputs, encoder_states, decoder_inputs = create_lstm_enc(learning_rate, hidden_size, dropout)
    dec_model = create_lstm_dec(hidden_size, dropout, encoder_inputs, encoder_states, decoder_inputs)
    return enc_model, dec_model, encoder_inputs, encoder_states,

def create_lstm_enc(learning_rate, hidden_size, dropout):
    # LSTM hidden units
    
    # Define an input series and encode it with an LSTM. 
    encoder_inputs = Input(shape=(None, 1)) 
    encoder = LSTM(hidden_size, dropout=dropout, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    
    # We discard `encoder_outputs` and only keep the final states. These represent the "context"
    # vector that we use as the basis for decoding.
    encoder_states = [state_h, state_c]
    
    # Set up the decoder, using `encoder_states` as initial state.
    # This is where teacher forcing inputs are fed in.
    decoder_inputs = Input(shape=(None, 1)) 
    
    # We set up our decoder using `encoder_states` as initial state.  
    # We return full output sequences and return internal states as well. 
    # We don't use the return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(hidden_size, dropout=dropout, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    
    decoder_dense = Dense(1) # 1 continuous output at each timestep
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    model.compile(Adam(learning_rate=learning_rate), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model, encoder_inputs, encoder_states, decoder_inputs

def create_lstm_dec(hidden_size, dropout, encoder_inputs, encoder_states, decoder_inputs):
    # from our previous model - mapping encoder sequence to state vectors
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_lstm = LSTM(hidden_size, dropout=dropout, return_sequences=True, return_state=True)
    decoder_dense = Dense(1)
    # A modified version of the decoding stage that takes in predicted target inputs
    # and encoded state vectors, returning predicted target outputs and decoder state vectors.
    # We need to hang onto these state vectors to run the next step of the inference loop.
    decoder_state_input_h = Input(shape=(hidden_size,))
    decoder_state_input_c = Input(shape=(hidden_size,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)
    return decoder_model


def create_simple_wave(learning_rate):
    
    # convolutional layer parameters
    n_filters = 32 
    filter_width = 2
    dilation_rates = [2**i for i in range(8)] 
    
    # define an input history series and pass it through a stack of dilated causal convolutions. 
    history_seq = Input(shape=(None, 1))
    x = history_seq
    
    for dilation_rate in dilation_rates:
        x = Conv1D(filters=n_filters,
                   kernel_size=filter_width, 
                   padding='causal',
                   dilation_rate=dilation_rate)(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(.2)(x)
    x = Dense(1)(x)
    
    # extract the last 28 time steps as the training target
    def slice(x, seq_length):
        return x[:,-seq_length:,:]
    
    pred_seq_train = Lambda(slice, arguments={'seq_length':pred_steps})(x)
    
    model = Model(history_seq, pred_seq_train)
    model.compile(Adam(learning_rate=learning_rate), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

def create_full_wave(learning_rate):
    # convolutional operation parameters
    n_filters = 64 # 32 
    filter_width = 2
    dilation_rates = [2**i for i in range(8)] * 2 
    
    # define an input history series and pass it through a stack of dilated causal convolution blocks. 
    history_seq = Input(shape=(None, 1))
    x = history_seq
    
    skips = []
    for dilation_rate in dilation_rates:
        
        # preprocessing - equivalent to time-distributed dense
        x = Conv1D(32, 1, padding='same', activation='relu')(x) 
        
        # filter convolution
        x_f = Conv1D(filters=n_filters,
                     kernel_size=filter_width, 
                     padding='causal',
                     dilation_rate=dilation_rate)(x)
        
        # gating convolution
        x_g = Conv1D(filters=n_filters,
                     kernel_size=filter_width, 
                     padding='causal',
                     dilation_rate=dilation_rate)(x)
        
        # multiply filter and gating branches
        z = Multiply()([Activation('tanh')(x_f),
                        Activation('sigmoid')(x_g)])
        
        # postprocessing - equivalent to time-distributed dense
        z = Conv1D(32, 1, padding='same', activation='relu')(z)
        
        # residual connection
        x = Add()([x, z])    
        
        # collect skip connections
        skips.append(z)
    
    # add all skip connection outputs 
    out = Activation('relu')(Add()(skips))
    
    # final time-distributed dense layers 
    out = Conv1D(128, 1, padding='same')(out)
    out = Activation('relu')(out)
    out = Dropout(.2)(out)
    out = Conv1D(1, 1, padding='same')(out)
    
    # extract the last 28 time steps as the training target
    def slice(x, seq_length):
        return x[:,-seq_length:,:]
    
    pred_seq_train = Lambda(slice, arguments={'seq_length':pred_steps})(out)
    
    model = Model(history_seq, pred_seq_train)
    model.compile(Adam(learning_rate=learning_rate), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

