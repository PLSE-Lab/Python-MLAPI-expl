#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math, re, os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from kaggle_datasets import KaggleDatasets
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE


# In[ ]:


get_ipython().system('wget aibrian.com/checkpoint')

import os
import pickle
import logging

import numpy as np
np.random.seed(6788)

import tensorflow as tf
try:
    tf.set_random_seed(6788)
except:
    pass

from tensorflow.keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, SimpleRNN, Activation, dot, concatenate, Bidirectional, GRU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint

# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)
# Placeholder for max lengths of input and output which are user configruable constants
max_input_length = None
max_output_length = None

char_start_encoding = 1
char_padding_encoding = 0

def build_sequence_encode_decode_dicts(input_data):
    """
    Builds encoding and decoding dictionaries for given list of strings.

    Parameters:

    input_data (list): List of strings.

    Returns:

    encoding_dict (dict): Dictionary with chars as keys and corresponding integer encodings as values.

    decoding_dict (dict): Reverse of above dictionary.

    len(encoding_dict) + 2: +2 because of special start (1) and padding (0) chars we add to each sequence.

    """
    encoding_dict = {}
    decoding_dict = {}
    for line in input_data:
        for char in line:
            if char not in encoding_dict:
                # Using 2 + because our sequence start encoding is 1 and padding encoding is 0
                encoding_dict[char] = 2 + len(encoding_dict)
                decoding_dict[2 + len(decoding_dict)] = char
    
    return encoding_dict, decoding_dict, len(encoding_dict) + 2

def encode_sequences(encoding_dict, sequences, max_length):
    """
    Encodes given input strings into numpy arrays based on the encoding dicts supplied.

    Parameters:

    encoding_dict (dict): Dictionary with chars as keys and corresponding integer encodings as values.

    sequences (list): List of sequences (strings) to be encoded.

    max_length (int): Max length of input sequences. All sequences will be padded to this length to support batching.

    Returns:
    
    encoded_data (numpy array): Reverse of above dictionary.

    """
    encoded_data = np.zeros(shape=(len(sequences), max_length))
    for i in range(len(sequences)):
        for j in range(min(len(sequences[i]), max_length)):
            encoded_data[i][j] = encoding_dict[sequences[i][j]]
    return encoded_data


def decode_sequence(decoding_dict, sequence):
    """
    Decodes integers into string based on the decoding dict.

    Parameters:

    decoding_dict (dict): Dictionary with chars as values and corresponding integer encodings as keys.

    Returns:

    sequence (str): Decoded string.

    """
    text = ''
    for i in sequence:
        if i == 0:
            break
        text += decoding_dict[i]
    return text


def generate(texts, input_encoding_dict, model, max_input_length, max_output_length, beam_size, max_beams, min_cut_off_len, cut_off_ratio):
    """
    Main function for generating encoded output(s) for encoded input(s)

    Parameters:

    texts (list/str): List of input strings or single input string.

    input_encoding_dict (dict): Encoding dictionary generated from the input strings.

    model (kerasl model): Loaded keras model.

    max_input_length (int): Max length of input sequences. All sequences will be padded to this length to support batching.

    max_output_length (int): Max output length. Need to know when to stop decoding if no padding character comes.

    beam_size (int): Beam size at each prediction.

    max_beams (int): Maximum number of beams to be kept in memory.

    min_cut_off_len (int): Used in deciding when to stop decoding.

    cut_off_ratio (float): Used in deciding when to stop decoding.

        # min_cut_off_len = max(min_cut_off_len, cut_off_ratio*len(max(texts, key=len)))
        # min_cut_off_len = min(min_cut_off_len, max_output_length)

    Returns:
    
    all_completed_beams (list): List of completed beams.

    """
    if not isinstance(texts, list):
        texts = [texts]

    min_cut_off_len = max(min_cut_off_len, cut_off_ratio*len(max(texts, key=len)))
    min_cut_off_len = min(min_cut_off_len, max_output_length)

    all_completed_beams = {i:[] for i in range(len(texts))}
    all_running_beams = {}
    for i, text in enumerate(texts):
        all_running_beams[i] = [[np.zeros(shape=(len(text), max_output_length)), [1]]]
        all_running_beams[i][0][0][:,0] = char_start_encoding

    
    while len(all_running_beams) != 0:
        for i in all_running_beams:
            all_running_beams[i] = sorted(all_running_beams[i], key=lambda tup:np.prod(tup[1]), reverse=True)
            all_running_beams[i] = all_running_beams[i][:max_beams]
        
        in_out_map = {}
        batch_encoder_input = []
        batch_decoder_input = []
        t_c = 0
        for text_i in all_running_beams:
            if text_i not in in_out_map:
                in_out_map[text_i] = []
            for running_beam in all_running_beams[text_i]:
                in_out_map[text_i].append(t_c)
                t_c+=1
                batch_encoder_input.append(texts[text_i])
                batch_decoder_input.append(running_beam[0][0])


        batch_encoder_input = encode_sequences(input_encoding_dict, batch_encoder_input, max_input_length)
        batch_decoder_input = np.asarray(batch_decoder_input)
        batch_predictions = model.predict([batch_encoder_input, batch_decoder_input])

        t_c = 0
        for text_i, t_cs in in_out_map.items():
            temp_running_beams = []
            for running_beam, probs in all_running_beams[text_i]:
                if len(probs) >= min_cut_off_len:
                    all_completed_beams[text_i].append([running_beam[:,1:], probs])
                else:
                    prediction = batch_predictions[t_c]
                    sorted_args = prediction.argsort()
                    sorted_probs = np.sort(prediction)

                    for i in range(1, beam_size+1):
                        temp_running_beam = np.copy(running_beam)
                        i = -1 * i
                        ith_arg = sorted_args[:, i][len(probs)]
                        ith_prob = sorted_probs[:, i][len(probs)]
                        
                        temp_running_beam[:, len(probs)] = ith_arg
                        temp_running_beams.append([temp_running_beam, probs + [ith_prob]])

                t_c+=1

            all_running_beams[text_i] = [b for b in temp_running_beams]
        
        to_del = []
        for i, v in all_running_beams.items():
            if not v:
                to_del.append(i)
        
        for i in to_del:
            del all_running_beams[i]

    return all_completed_beams

def infer(texts, model, params, beam_size=3, max_beams=3, min_cut_off_len=10, cut_off_ratio=1.5):
    """
    Main function for generating output(s) for given input(s)

    Parameters:

    texts (list/str): List of input strings or single input string.

    model (kerasl model): Loaded keras model.

    params (dict): Loaded params generated by build_params

    beam_size (int): Beam size at each prediction.

    max_beams (int): Maximum number of beams to be kept in memory.

    min_cut_off_len (int): Used in deciding when to stop decoding.

    cut_off_ratio (float): Used in deciding when to stop decoding.

        # min_cut_off_len = max(min_cut_off_len, cut_off_ratio*len(max(texts, key=len)))
        # min_cut_off_len = min(min_cut_off_len, max_output_length)

    Returns:
    
    outputs (list of dicts): Each dict has the sequence and probability.

    """
    if not isinstance(texts, list):
        texts = [texts]

    input_encoding_dict = params['input_encoding']
    output_decoding_dict = params['output_decoding']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']

    all_decoder_outputs = generate(texts, input_encoding_dict, model, max_input_length, max_output_length, beam_size, max_beams, min_cut_off_len, cut_off_ratio)
    outputs = []

    for i, decoder_outputs in all_decoder_outputs.items():
        outputs.append([])
        for decoder_output, probs in decoder_outputs:
            outputs[-1].append({'sequence': decode_sequence(output_decoding_dict, decoder_output[0]), 'prob': np.prod(probs)})

    return outputs

def generate_greedy(texts, input_encoding_dict, model, max_input_length, max_output_length):
    """
    Main function for generating output(s) for given input(s)

    Parameters:

    texts (list/str): List of input strings or single input string.

    input_encoding_dict (dict): Encoding dictionary generated from the input strings.

    model (kerasl model): Loaded keras model.

    max_input_length (int): Max length of input sequences. All sequences will be padded to this length to support batching.

    max_output_length (int): Max output length. Need to know when to stop decoding if no padding character comes.

    Returns:
    
    outputs (list): Generated outputs.

    """
    if not isinstance(texts, list):
        texts = [texts]

    encoder_input = encode_sequences(input_encoding_dict, texts, max_input_length)
    decoder_input = np.zeros(shape=(len(encoder_input), max_output_length))
    decoder_input[:,0] = char_start_encoding
    for i in range(1, max_output_length):
        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:,i] = output[:,i]
        
        if np.all(decoder_input[:,i] == char_padding_encoding):
            return decoder_input[:,1:]

    return decoder_input[:,1:]

def infer_greedy(texts, model, params):
    """
    Main function for generating output(s) for given input(s)

    Parameters:

    texts (list/str): List of input strings or single input string.

    model (kerasl model): Loaded keras model.

    params (dict): Loaded params generated by build_params

    Returns:
    
    outputs (list): Generated outputs.

    """
    return_string = False
    if not isinstance(texts, list):
        return_string = True
        texts = [texts]

    input_encoding_dict = params['input_encoding']
    output_decoding_dict = params['output_decoding']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']

    decoder_output = generate_greedy(texts, input_encoding_dict, model, max_input_length, max_output_length)
    if return_string:
        return decode_sequence(output_decoding_dict, decoder_output[0])

    return [decode_sequence(output_decoding_dict, i) for i in decoder_output]


def build_params(input_data = [], output_data = [], params_path = 'test_params', max_lenghts = (5,5)):
    """
    Build the params and save them as a pickle. (If not already present.)

    Parameters:

    input_data (list): List of input strings.

    output_data (list): List of output strings.

    params_path (str): Path for saving the params.

    max_lenghts (tuple): (max_input_length, max_output_length)

    Returns:
    
    params (dict): Generated params (encoding, decoding dicts ..).

    """
    if os.path.exists(params_path):
        print('Loading the params file')
        params = pickle.load(open(params_path, 'rb'))
        return params
    
    print('Creating params file')
    input_encoding, input_decoding, input_dict_size = build_sequence_encode_decode_dicts(input_data)
    output_encoding, output_decoding, output_dict_size = build_sequence_encode_decode_dicts(output_data)
    params = {}
    params['input_encoding'] = input_encoding
    params['input_decoding'] = input_decoding
    params['input_dict_size'] = input_dict_size
    params['output_encoding'] = output_encoding
    params['output_decoding'] = output_decoding
    params['output_dict_size'] = output_dict_size
    params['max_input_length'] = max_lenghts[0]
    params['max_output_length'] = max_lenghts[1]

    pickle.dump(params, open(params_path, 'wb'))
    return params

def convert_training_data(input_data, output_data, params):
    """
    Encode training data.

    Parameters:

    input_data (list): List of input strings.

    output_data (list): List of output strings.

    params (dict): Generated params (encoding, decoding dicts ..).

    Returns:
    
    x, y: encoded inputs, outputs.

    """
    input_encoding = params['input_encoding']
    input_decoding = params['input_decoding']
    input_dict_size = params['input_dict_size']
    output_encoding = params['output_encoding']
    output_decoding = params['output_decoding']
    output_dict_size = params['output_dict_size']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']

    encoded_training_input = encode_sequences(input_encoding, input_data, max_input_length)
    encoded_training_output = encode_sequences(output_encoding, output_data, max_output_length)
    training_encoder_input = encoded_training_input
    training_decoder_input = np.zeros_like(encoded_training_output)
    training_decoder_input[:, 1:] = encoded_training_output[:,:-1]
    training_decoder_input[:, 0] = char_start_encoding
    training_decoder_output = np.eye(output_dict_size)[encoded_training_output.astype('int')]
    x=[training_encoder_input, training_decoder_input]
    y=[training_decoder_output]
    return x, y

def build_model(params_path = 'test/params', enc_lstm_units = 128, unroll = True, use_gru=False, optimizer='adam', display_summary=True):
    """
    Build keras model

    Parameters:

    params_path (str): Path for saving/loading the params.

    enc_lstm_units (int): Positive integer, dimensionality of the output space.

    unroll (bool): Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.

    use_gru (bool): GRU will be used instead of LSTM

    optimizer (str): optimizer to be used

    display_summary (bool): Set to true for verbose information.


    Returns:

    model (keras model): built model object.
    
    params (dict): Generated params (encoding, decoding dicts ..).

    """
    # generateing the encoding, decoding dicts
    params = build_params(params_path = params_path)

    input_encoding = params['input_encoding']
    input_decoding = params['input_decoding']
    input_dict_size = params['input_dict_size']
    output_encoding = params['output_encoding']
    output_decoding = params['output_decoding']
    output_dict_size = params['output_dict_size']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']


    if display_summary:
        print('Input encoding', input_encoding)
        print('Input decoding', input_decoding)
        print('Output encoding', output_encoding)
        print('Output decoding', output_decoding)


    # We need to define the max input lengths and max output lengths before training the model.
    # We pad the inputs and outputs to these max lengths
    encoder_input = Input(shape=(max_input_length,))
    decoder_input = Input(shape=(max_output_length,))

    # Need to make the number of hidden units configurable
    encoder = Embedding(input_dict_size, enc_lstm_units, input_length=max_input_length, mask_zero=True)(encoder_input)
    # using concat merge mode since in my experiments it g ave the best results same with unroll
    if not use_gru:
        encoder = Bidirectional(LSTM(enc_lstm_units, return_sequences=True, return_state=True, unroll=unroll), merge_mode='concat')(encoder)
        encoder_outs, forward_h, forward_c, backward_h, backward_c = encoder
        encoder_h = concatenate([forward_h, backward_h])
        encoder_c = concatenate([forward_c, backward_c])
    
    else:
        encoder = Bidirectional(GRU(enc_lstm_units, return_sequences=True, return_state=True, unroll=unroll), merge_mode='concat')(encoder)        
        encoder_outs, forward_h, backward_h= encoder
        encoder_h = concatenate([forward_h, backward_h])
    

    # using 2* enc_lstm_units because we are using concat merge mode
    # cannot use bidirectionals lstm for decoding (obviously!)
    
    decoder = Embedding(output_dict_size, 2 * enc_lstm_units, input_length=max_output_length, mask_zero=True)(decoder_input)

    if not use_gru:
        decoder = LSTM(2 * enc_lstm_units, return_sequences=True, unroll=unroll)(decoder, initial_state=[encoder_h, encoder_c])
    else:
        decoder = GRU(2 * enc_lstm_units, return_sequences=True, unroll=unroll)(decoder, initial_state=encoder_h)


    # luong attention
    attention = dot([decoder, encoder_outs], axes=[2, 2])
    attention = Activation('softmax', name='attention')(attention)

    context = dot([attention, encoder_outs], axes=[2,1])

    decoder_combined_context = concatenate([context, decoder])

    output = TimeDistributed(Dense(enc_lstm_units, activation="tanh"))(decoder_combined_context)
    output = TimeDistributed(Dense(output_dict_size, activation="softmax"))(output)

    model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
    if display_summary:
      model.summary()
    
    return model, params
    
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver) 
input_data = ['123', '213', '312', '321', '132', '231']
output_data = ['123', '123', '123', '123', '123', '123']
build_params(input_data = input_data, output_data = output_data, params_path = 'params', max_lenghts=(10, 10))

with strategy.scope():
  #model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
  model, params = build_model(params_path='params')
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

input_data, output_data = convert_training_data(input_data, output_data, params)

checkpoint = ModelCheckpoint('checkpoint', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(input_data, output_data, validation_data=(input_data, output_data), batch_size=2, epochs=40, callbacks=callbacks_list)
#model.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=get_validation_dataset())
 input_data = ['123', '213', '312', '321', '132', '231']
    output_data = ['123', '123', '123', '123', '123', '123']
    build_params(input_data = input_data, output_data = output_data, params_path = 'params', max_lenghts=(10, 10))
    
    model, params = build_model(params_path='params')

    input_data, output_data = convert_training_data(input_data, output_data, params)
    
    checkpoint = ModelCheckpoint('checkpoint', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(input_data, output_data, validation_data=(input_data, output_data), batch_size=2, epochs=40, callbacks=callbacks_list)
         model, params = build_model(params_path='params', enc_lstm_units=256)
    model.load_weights('checkpoint')
    start = time.time()
    print(infer("i will be there for you", model, params))
    end = time.time()
    print(end - start)

    start = time.time()
    print(infer(["i will be there for you","these is not that gre at","i dotlike this","i work at reckonsys"], model, params))
    end = time.time()
    print(end - start)
    start = time.time()
    print(infer("i will be there for you", model, params))
    end = time.time()
    print(end - start )


# # TPU or GPU detection

# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)


# # Competition data access
# TPUs read data directly from Google Cloud Storage (GCS). This Kaggle utility will copy the dataset to a GCS bucket co-located with the TPU. If you have multiple datasets attached to the notebook, you can pass the name of a specific dataset to the get_gcs_path function. The name of the dataset is the name of the directory it is mounted in. Use `!ls /kaggle/input/` to list attached datasets.

# In[ ]:


GCS_DS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_DS_PATH"


# # Configuration

# In[ ]:


IMAGE_SIZE = [512, 512] # at this size, a GPU will run out of memory. Use the TPU
EPOCHS = 12
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

GCS_PATH_SELECT = { # available image sizes
    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',
    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',
    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',
    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'
}
GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')
VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for the competition

CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102


# ## Visualization utilities
# data -> pixels, nothing of much interest for the machine learning practitioner in this section.

# In[ ]:


# numpy and matplotlib defaults
np.set_printoptions(threshold=15, linewidth=80)

def batch_to_numpy_images_and_labels(data):
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if numpy_labels.dtype == object: # binary string in this case, these are image ID strings
        numpy_labels = [None for _ in enumerate(numpy_images)]
    # If no labels, only image IDs, return None for labels (this is the case for test data)
    return numpy_images, numpy_labels

def title_from_label_and_target(label, correct_label):
    if correct_label is None:
        return CLASSES[label], True
    correct = (label == correct_label)
    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',
                                CLASSES[correct_label] if not correct else ''), correct

def display_one_flower(image, title, subplot, red=False, titlesize=16):
    plt.subplot(*subplot)
    plt.axis('off')
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)
    
def display_batch_of_images(databatch, predictions=None):
    """This will work with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data
    images, labels = batch_to_numpy_images_and_labels(databatch)
    if labels is None:
        labels = [None for _ in enumerate(images)]
        
    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle
    rows = int(math.sqrt(len(images)))
    cols = len(images)//rows
        
    # size and spacing
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols,1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))
    
    # display
    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):
        title = '' if label is None else CLASSES[label]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images
        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)
    
    #layout
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()

def display_confusion_matrix(cmat, score, precision, recall):
    plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.matshow(cmat, cmap='Reds')
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    titlestring = ""
    if score is not None:
        titlestring += 'f1 = {:.3f} '.format(score)
    if precision is not None:
        titlestring += '\nprecision = {:.3f} '.format(precision)
    if recall is not None:
        titlestring += '\nrecall = {:.3f} '.format(recall)
    if len(titlestring) > 0:
        ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})
    plt.show()
    
def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])


# # Datasets

# In[ ]:


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label # returns a dataset of (image, label) pairs

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
    return image, idnum # returns a dataset of image(s)

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def data_augment(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_saturation(image, 0, 2)
    return image, label   

def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))


# # Dataset visualizations

# In[ ]:


# data dump
print("Training data shapes:")
for image, label in get_training_dataset().take(3):
    print(image.numpy().shape, label.numpy().shape)
print("Training data label examples:", label.numpy())
print("Validation data shapes:")
for image, label in get_validation_dataset().take(3):
    print(image.numpy().shape, label.numpy().shape)
print("Validation data label examples:", label.numpy())
print("Test data shapes:")
for image, idnum in get_test_dataset().take(3):
    print(image.numpy().shape, idnum.numpy().shape)
print("Test data IDs:", idnum.numpy().astype('U')) # U=unicode string


# In[ ]:


# Peek at training data
training_dataset = get_training_dataset()
training_dataset = training_dataset.unbatch().batch(20)
train_batch = iter(training_dataset)


# In[ ]:


# run this cell again for next set of images
display_batch_of_images(next(train_batch))


# In[ ]:


# peer at test data
test_dataset = get_test_dataset()
test_dataset = test_dataset.unbatch().batch(20)
test_batch = iter(test_dataset)


# In[ ]:


# run this cell again for next set of images
display_batch_of_images(next(test_batch))


# # Model
# Not the best but it converges ...

# In[ ]:


with strategy.scope():
    pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
    pretrained_model.trainable = False # tramsfer learning
    
    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
        
model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)
model.summary()


# # Training

# In[ ]:


import os
import pickle
import logging

import numpy as np
np.random.seed(6788)

import tensorflow as tf
try:
    tf.set_random_seed(6788)
except:
    pass

from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, SimpleRNN, Activation, dot, concatenate, Bidirectional, GRU
from keras.models import Model, load_model

from keras.callbacks import ModelCheckpoint


# Placeholder for max lengths of input and output which are user configruable constants
max_input_length = None
max_output_length = None

char_start_encoding = 1
char_padding_encoding = 0

def build_sequence_encode_decode_dicts(input_data):
    """
    Builds encoding and decoding dictionaries for given list of strings.

    Parameters:

    input_data (list): List of strings.

    Returns:

    encoding_dict (dict): Dictionary with chars as keys and corresponding integer encodings as values.

    decoding_dict (dict): Reverse of above dictionary.

    len(encoding_dict) + 2: +2 because of special start (1) and padding (0) chars we add to each sequence.

    """
    encoding_dict = {}
    decoding_dict = {}
    for line in input_data:
        for char in line:
            if char not in encoding_dict:
                # Using 2 + because our sequence start encoding is 1 and padding encoding is 0
                encoding_dict[char] = 2 + len(encoding_dict)
                decoding_dict[2 + len(decoding_dict)] = char
    
    return encoding_dict, decoding_dict, len(encoding_dict) + 2

def encode_sequences(encoding_dict, sequences, max_length):
    """
    Encodes given input strings into numpy arrays based on the encoding dicts supplied.

    Parameters:

    encoding_dict (dict): Dictionary with chars as keys and corresponding integer encodings as values.

    sequences (list): List of sequences (strings) to be encoded.

    max_length (int): Max length of input sequences. All sequences will be padded to this length to support batching.

    Returns:
    
    encoded_data (numpy array): Reverse of above dictionary.

    """
    encoded_data = np.zeros(shape=(len(sequences), max_length))
    for i in range(len(sequences)):
        for j in range(min(len(sequences[i]), max_length)):
            encoded_data[i][j] = encoding_dict[sequences[i][j]]
    return encoded_data


def decode_sequence(decoding_dict, sequence):
    """
    Decodes integers into string based on the decoding dict.

    Parameters:

    decoding_dict (dict): Dictionary with chars as values and corresponding integer encodings as keys.

    Returns:

    sequence (str): Decoded string.

    """
    text = ''
    for i in sequence:
        if i == 0:
            break
        text += decoding_dict[i]
    return text


def generate(texts, input_encoding_dict, model, max_input_length, max_output_length, beam_size, max_beams, min_cut_off_len, cut_off_ratio):
    """
    Main function for generating encoded output(s) for encoded input(s)

    Parameters:

    texts (list/str): List of input strings or single input string.

    input_encoding_dict (dict): Encoding dictionary generated from the input strings.

    model (kerasl model): Loaded keras model.

    max_input_length (int): Max length of input sequences. All sequences will be padded to this length to support batching.

    max_output_length (int): Max output length. Need to know when to stop decoding if no padding character comes.

    beam_size (int): Beam size at each prediction.

    max_beams (int): Maximum number of beams to be kept in memory.

    min_cut_off_len (int): Used in deciding when to stop decoding.

    cut_off_ratio (float): Used in deciding when to stop decoding.

        # min_cut_off_len = max(min_cut_off_len, cut_off_ratio*len(max(texts, key=len)))
        # min_cut_off_len = min(min_cut_off_len, max_output_length)

    Returns:
    
    all_completed_beams (list): List of completed beams.

    """
    if not isinstance(texts, list):
        texts = [texts]

    min_cut_off_len = max(min_cut_off_len, cut_off_ratio*len(max(texts, key=len)))
    min_cut_off_len = min(min_cut_off_len, max_output_length)

    all_completed_beams = {i:[] for i in range(len(texts))}
    all_running_beams = {}
    for i, text in enumerate(texts):
        all_running_beams[i] = [[np.zeros(shape=(len(text), max_output_length)), [1]]]
        all_running_beams[i][0][0][:,0] = char_start_encoding

    
    while len(all_running_beams) != 0:
        for i in all_running_beams:
            all_running_beams[i] = sorted(all_running_beams[i], key=lambda tup:np.prod(tup[1]), reverse=True)
            all_running_beams[i] = all_running_beams[i][:max_beams]
        
        in_out_map = {}
        batch_encoder_input = []
        batch_decoder_input = []
        t_c = 0
        for text_i in all_running_beams:
            if text_i not in in_out_map:
                in_out_map[text_i] = []
            for running_beam in all_running_beams[text_i]:
                in_out_map[text_i].append(t_c)
                t_c+=1
                batch_encoder_input.append(texts[text_i])
                batch_decoder_input.append(running_beam[0][0])


        batch_encoder_input = encode_sequences(input_encoding_dict, batch_encoder_input, max_input_length)
        batch_decoder_input = np.asarray(batch_decoder_input)
        batch_predictions = model.predict([batch_encoder_input, batch_decoder_input])

        t_c = 0
        for text_i, t_cs in in_out_map.items():
            temp_running_beams = []
            for running_beam, probs in all_running_beams[text_i]:
                if len(probs) >= min_cut_off_len:
                    all_completed_beams[text_i].append([running_beam[:,1:], probs])
                else:
                    prediction = batch_predictions[t_c]
                    sorted_args = prediction.argsort()
                    sorted_probs = np.sort(prediction)

                    for i in range(1, beam_size+1):
                        temp_running_beam = np.copy(running_beam)
                        i = -1 * i
                        ith_arg = sorted_args[:, i][len(probs)]
                        ith_prob = sorted_probs[:, i][len(probs)]
                        
                        temp_running_beam[:, len(probs)] = ith_arg
                        temp_running_beams.append([temp_running_beam, probs + [ith_prob]])

                t_c+=1

            all_running_beams[text_i] = [b for b in temp_running_beams]
        
        to_del = []
        for i, v in all_running_beams.items():
            if not v:
                to_del.append(i)
        
        for i in to_del:
            del all_running_beams[i]

    return all_completed_beams

def infer(texts, model, params, beam_size=3, max_beams=3, min_cut_off_len=10, cut_off_ratio=1.5):
    """
    Main function for generating output(s) for given input(s)

    Parameters:

    texts (list/str): List of input strings or single input string.

    model (kerasl model): Loaded keras model.

    params (dict): Loaded params generated by build_params

    beam_size (int): Beam size at each prediction.

    max_beams (int): Maximum number of beams to be kept in memory.

    min_cut_off_len (int): Used in deciding when to stop decoding.

    cut_off_ratio (float): Used in deciding when to stop decoding.

        # min_cut_off_len = max(min_cut_off_len, cut_off_ratio*len(max(texts, key=len)))
        # min_cut_off_len = min(min_cut_off_len, max_output_length)

    Returns:
    
    outputs (list of dicts): Each dict has the sequence and probability.

    """
    if not isinstance(texts, list):
        texts = [texts]

    input_encoding_dict = params['input_encoding']
    output_decoding_dict = params['output_decoding']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']

    all_decoder_outputs = generate(texts, input_encoding_dict, model, max_input_length, max_output_length, beam_size, max_beams, min_cut_off_len, cut_off_ratio)
    outputs = []

    for i, decoder_outputs in all_decoder_outputs.items():
        outputs.append([])
        for decoder_output, probs in decoder_outputs:
            outputs[-1].append({'sequence': decode_sequence(output_decoding_dict, decoder_output[0]), 'prob': np.prod(probs)})

    return outputs

def generate_greedy(texts, input_encoding_dict, model, max_input_length, max_output_length):
    """
    Main function for generating output(s) for given input(s)

    Parameters:

    texts (list/str): List of input strings or single input string.

    input_encoding_dict (dict): Encoding dictionary generated from the input strings.

    model (kerasl model): Loaded keras model.

    max_input_length (int): Max length of input sequences. All sequences will be padded to this length to support batching.

    max_output_length (int): Max output length. Need to know when to stop decoding if no padding character comes.

    Returns:
    
    outputs (list): Generated outputs.

    """
    if not isinstance(texts, list):
        texts = [texts]

    encoder_input = encode_sequences(input_encoding_dict, texts, max_input_length)
    decoder_input = np.zeros(shape=(len(encoder_input), max_output_length))
    decoder_input[:,0] = char_start_encoding
    for i in range(1, max_output_length):
        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:,i] = output[:,i]
        
        if np.all(decoder_input[:,i] == char_padding_encoding):
            return decoder_input[:,1:]

    return decoder_input[:,1:]

def infer_greedy(texts, model, params):
    """
    Main function for generating output(s) for given input(s)

    Parameters:

    texts (list/str): List of input strings or single input string.

    model (kerasl model): Loaded keras model.

    params (dict): Loaded params generated by build_params

    Returns:
    
    outputs (list): Generated outputs.

    """
    return_string = False
    if not isinstance(texts, list):
        return_string = True
        texts = [texts]

    input_encoding_dict = params['input_encoding']
    output_decoding_dict = params['output_decoding']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']

    decoder_output = generate_greedy(texts, input_encoding_dict, model, max_input_length, max_output_length)
    if return_string:
        return decode_sequence(output_decoding_dict, decoder_output[0])

    return [decode_sequence(output_decoding_dict, i) for i in decoder_output]


def build_params(input_data = [], output_data = [], params_path = 'test_params', max_lenghts = (5,5)):
    """
    Build the params and save them as a pickle. (If not already present.)

    Parameters:

    input_data (list): List of input strings.

    output_data (list): List of output strings.

    params_path (str): Path for saving the params.

    max_lenghts (tuple): (max_input_length, max_output_length)

    Returns:
    
    params (dict): Generated params (encoding, decoding dicts ..).

    """
    if os.path.exists(params_path):
        print('Loading the params file')
        params = pickle.load(open(params_path, 'rb'))
        return params
    
    print('Creating params file')
    input_encoding, input_decoding, input_dict_size = build_sequence_encode_decode_dicts(input_data)
    output_encoding, output_decoding, output_dict_size = build_sequence_encode_decode_dicts(output_data)
    params = {}
    params['input_encoding'] = input_encoding
    params['input_decoding'] = input_decoding
    params['input_dict_size'] = input_dict_size
    params['output_encoding'] = output_encoding
    params['output_decoding'] = output_decoding
    params['output_dict_size'] = output_dict_size
    params['max_input_length'] = max_lenghts[0]
    params['max_output_length'] = max_lenghts[1]

    pickle.dump(params, open(params_path, 'wb'))
    return params

def convert_training_data(input_data, output_data, params):
    """
    Encode training data.

    Parameters:

    input_data (list): List of input strings.

    output_data (list): List of output strings.

    params (dict): Generated params (encoding, decoding dicts ..).

    Returns:
    
    x, y: encoded inputs, outputs.

    """
    input_encoding = params['input_encoding']
    input_decoding = params['input_decoding']
    input_dict_size = params['input_dict_size']
    output_encoding = params['output_encoding']
    output_decoding = params['output_decoding']
    output_dict_size = params['output_dict_size']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']

    encoded_training_input = encode_sequences(input_encoding, input_data, max_input_length)
    encoded_training_output = encode_sequences(output_encoding, output_data, max_output_length)
    training_encoder_input = encoded_training_input
    training_decoder_input = np.zeros_like(encoded_training_output)
    training_decoder_input[:, 1:] = encoded_training_output[:,:-1]
    training_decoder_input[:, 0] = char_start_encoding
    training_decoder_output = np.eye(output_dict_size)[encoded_training_output.astype('int')]
    x=[training_encoder_input, training_decoder_input]
    y=[training_decoder_output]
    return x, y

def build_model(params_path = 'test/params', enc_lstm_units = 128, unroll = True, use_gru=False, optimizer='adam', display_summary=True):
    """
    Build keras model

    Parameters:

    params_path (str): Path for saving/loading the params.

    enc_lstm_units (int): Positive integer, dimensionality of the output space.

    unroll (bool): Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.

    use_gru (bool): GRU will be used instead of LSTM

    optimizer (str): optimizer to be used

    display_summary (bool): Set to true for verbose information.


    Returns:

    model (keras model): built model object.
    
    params (dict): Generated params (encoding, decoding dicts ..).

    """
    # generateing the encoding, decoding dicts
    params = build_params(params_path = params_path)

    input_encoding = params['input_encoding']
    input_decoding = params['input_decoding']
    input_dict_size = params['input_dict_size']
    output_encoding = params['output_encoding']
    output_decoding = params['output_decoding']
    output_dict_size = params['output_dict_size']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']


    if display_summary:
        print('Input encoding', input_encoding)
        print('Input decoding', input_decoding)
        print('Output encoding', output_encoding)
        print('Output decoding', output_decoding)


    # We need to define the max input lengths and max output lengths before training the model.
    # We pad the inputs and outputs to these max lengths
    encoder_input = Input(shape=(max_input_length,))
    decoder_input = Input(shape=(max_output_length,))

    # Need to make the number of hidden units configurable
    encoder = Embedding(input_dict_size, enc_lstm_units, input_length=max_input_length, mask_zero=True)(encoder_input)
    # using concat merge mode since in my experiments it gave the best results same with unroll
    if not use_gru:
        encoder = Bidirectional(LSTM(enc_lstm_units, return_sequences=True, return_state=True, unroll=unroll), merge_mode='concat')(encoder)
        encoder_outs, forward_h, forward_c, backward_h, backward_c = encoder
        encoder_h = concatenate([forward_h, backward_h])
        encoder_c = concatenate([forward_c, backward_c])
    
    else:
        encoder = Bidirectional(GRU(enc_lstm_units, return_sequences=True, return_state=True, unroll=unroll), merge_mode='concat')(encoder)        
        encoder_outs, forward_h, backward_h= encoder
        encoder_h = concatenate([forward_h, backward_h])
    

    # using 2* enc_lstm_units because we are using concat merge mode
    # cannot use bidirectionals lstm for decoding (obviously!)
    
    decoder = Embedding(output_dict_size, 2 * enc_lstm_units, input_length=max_output_length, mask_zero=True)(decoder_input)

    if not use_gru:
        decoder = LSTM(2 * enc_lstm_units, return_sequences=True, unroll=unroll)(decoder, initial_state=[encoder_h, encoder_c])
    else:
        decoder = GRU(2 * enc_lstm_units, return_sequences=True, unroll=unroll)(decoder, initial_state=encoder_h)


    # luong attention
    attention = dot([decoder, encoder_outs], axes=[2, 2])
    attention = Activation('softmax', name='attention')(attention)

    context = dot([attention, encoder_outs], axes=[2,1])

    decoder_combined_context = concatenate([context, decoder])

    output = TimeDistributed(Dense(enc_lstm_units, activation="tanh"))(decoder_combined_context)
    output = TimeDistributed(Dense(output_dict_size, activation="softmax"))(output)

    model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    if display_summary:
        model.summary()
    
    return model, params



if __name__ == '__main__':
    input_data = ['123', '213', '312', '321', '132', '231']
    output_data = ['123', '123', '123', '123', '123', '123']
    build_params(input_data = input_data, output_data = output_data, params_path = 'params', max_lenghts=(10, 10))
    
    model, params = build_model(params_path='params')

    input_data, output_data = convert_training_data(input_data, output_data, params)
    
    checkpoint = ModelCheckpoint('checkpoint', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(input_data, output_data, validation_data=(input_data, output_data), batch_size=2, epochs=40, callbacks=callbacks_list)
         model, params = build_model(params_path='params', enc_lstm_units=256)
    model.load_weights('checkpoint')
    start = time.time()
    print(infer("i will be there for you", model, params))
    end = time.time()
    print(end - start)

    start = time.time()
    print(infer(["i will be there for you","these is not that gre at","i dotlike this","i work at reckonsys"], model, params))
    end = time.time()
    print(end - start)
    start = time.time()
    print(infer("i will be there for you", model, params))
    end = time.time()
    print(end - start )


# In[ ]:


history = model.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=get_validation_dataset())


# In[ ]:


display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)
display_training_curves(history.history['sparse_categorical_accuracy'], history.history['val_sparse_categorical_accuracy'], 'accuracy', 212)


# # Confusion matrix

# In[ ]:



#!wget aibrian.com/checkpoint
#!wget aibrian.com/params

import tensorflow as tf
# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)

import time
import os
import pickle
import logging

import numpy as np
np.random.seed(6788)

import tensorflow as tf
try:
    tf.set_random_seed(6788)
except:
    pass

from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, SimpleRNN, Activation, dot, concatenate, Bidirectional, GRU
from keras.models import Model, load_model

from keras.callbacks import ModelCheckpoint


# Placeholder for max lengths of input and output which are user configruable constants
max_input_length = None
max_output_length = None

char_start_encoding = 1
char_padding_encoding = 0

def build_sequence_encode_decode_dicts(input_data):
    """
    Builds encoding and decoding dictionaries for given list of strings.

    Parameters:

    input_data (list): List of strings.

    Returns:

    encoding_dict (dict): Dictionary with chars as keys and corresponding integer encodings as values.

    decoding_dict (dict): Reverse of above dictionary.

    len(encoding_dict) + 2: +2 because of special start (1) and padding (0) chars we add to each sequence.

    """
    encoding_dict = {}
    decoding_dict = {}
    for line in input_data:
        for char in line:
            if char not in encoding_dict:
                # Using 2 + because our sequence start encoding is 1 and padding encoding is 0
                encoding_dict[char] = 2 + len(encoding_dict)
                decoding_dict[2 + len(decoding_dict)] = char
    
    return encoding_dict, decoding_dict, len(encoding_dict) + 2

def encode_sequences(encoding_dict, sequences, max_length):
    """
    Encodes given input strings into numpy arrays based on the encoding dicts supplied.

    Parameters:

    encoding_dict (dict): Dictionary with chars as keys and corresponding integer encodings as values.

    sequences (list): List of sequences (strings) to be encoded.

    max_length (int): Max length of input sequences. All sequences will be padded to this length to support batching.

    Returns:
    
    encoded_data (numpy array): Reverse of above dictionary.

    """
    encoded_data = np.zeros(shape=(len(sequences), max_length))
    for i in range(len(sequences)):
        for j in range(min(len(sequences[i]), max_length)):
            encoded_data[i][j] = encoding_dict[sequences[i][j]]
    return encoded_data


def decode_sequence(decoding_dict, sequence):
    """
    Decodes integers into string based on the decoding dict.

    Parameters:

    decoding_dict (dict): Dictionary with chars as values and corresponding integer encodings as keys.

    Returns:

    sequence (str): Decoded string.

    """
    text = ''
    for i in sequence:
        if i == 0:
            break
        text += decoding_dict[i]
    return text


def generate(texts, input_encoding_dict, model, max_input_length, max_output_length, beam_size, max_beams, min_cut_off_len, cut_off_ratio):
    """
    Main function for generating encoded output(s) for encoded input(s)

    Parameters:

    texts (list/str): List of input strings or single input string.

    input_encoding_dict (dict): Encoding dictionary generated from the input strings.

    model (kerasl model): Loaded keras model.

    max_input_length (int): Max length of input sequences. All sequences will be padded to this length to support batching.

    max_output_length (int): Max output length. Need to know when to stop decoding if no padding character comes.

    beam_size (int): Beam size at each prediction.

    max_beams (int): Maximum number of beams to be kept in memory.

    min_cut_off_len (int): Used in deciding when to stop decoding.

    cut_off_ratio (float): Used in deciding when to stop decoding.

        # min_cut_off_len = max(min_cut_off_len, cut_off_ratio*len(max(texts, key=len)))
        # min_cut_off_len = min(min_cut_off_len, max_output_length)

    Returns:
    
    all_completed_beams (list): List of completed beams.

    """
    if not isinstance(texts, list):
        texts = [texts]

    min_cut_off_len = max(min_cut_off_len, cut_off_ratio*len(max(texts, key=len)))
    min_cut_off_len = min(min_cut_off_len, max_output_length)

    all_completed_beams = {i:[] for i in range(len(texts))}
    all_running_beams = {}
    for i, text in enumerate(texts):
        all_running_beams[i] = [[np.zeros(shape=(len(text), max_output_length)), [1]]]
        all_running_beams[i][0][0][:,0] = char_start_encoding

    
    while len(all_running_beams) != 0:
        for i in all_running_beams:
            all_running_beams[i] = sorted(all_running_beams[i], key=lambda tup:np.prod(tup[1]), reverse=True)
            all_running_beams[i] = all_running_beams[i][:max_beams]
        
        in_out_map = {}
        batch_encoder_input = []
        batch_decoder_input = []
        t_c = 0
        for text_i in all_running_beams:
            if text_i not in in_out_map:
                in_out_map[text_i] = []
            for running_beam in all_running_beams[text_i]:
                in_out_map[text_i].append(t_c)
                t_c+=1
                batch_encoder_input.append(texts[text_i])
                batch_decoder_input.append(running_beam[0][0])

        with strategy.scope():
            batch_encoder_input = encode_sequences(input_encoding_dict, batch_encoder_input, max_input_length)
            batch_decoder_input = np.asarray(batch_decoder_input)
            batch_predictions = model.predict([batch_encoder_input, batch_decoder_input])

        t_c = 0
        for text_i, t_cs in in_out_map.items():
            temp_running_beams = []
            for running_beam, probs in all_running_beams[text_i]:
                if len(probs) >= min_cut_off_len:
                    all_completed_beams[text_i].append([running_beam[:,1:], probs])
                else:
                    prediction = batch_predictions[t_c]
                    sorted_args = prediction.argsort()
                    sorted_probs = np.sort(prediction)

                    for i in range(1, beam_size+1):
                        temp_running_beam = np.copy(running_beam)
                        i = -1 * i
                        ith_arg = sorted_args[:, i][len(probs)]
                        ith_prob = sorted_probs[:, i][len(probs)]
                        
                        temp_running_beam[:, len(probs)] = ith_arg
                        temp_running_beams.append([temp_running_beam, probs + [ith_prob]])

                t_c+=1

            all_running_beams[text_i] = [b for b in temp_running_beams]
        
        to_del = []
        for i, v in all_running_beams.items():
            if not v:
                to_del.append(i)
        
        for i in to_del:
            del all_running_beams[i]

    return all_completed_beams

def infer(texts, model, params, beam_size=3, max_beams=3, min_cut_off_len=10, cut_off_ratio=1.5):
    """
    Main function for generating output(s) for given input(s)

    Parameters:

    texts (list/str): List of input strings or single input string.

    model (kerasl model): Loaded keras model.

    params (dict): Loaded params generated by build_params

    beam_size (int): Beam size at each prediction.

    max_beams (int): Maximum number of beams to be kept in memory.

    min_cut_off_len (int): Used in deciding when to stop decoding.

    cut_off_ratio (float): Used in deciding when to stop decoding.

        # min_cut_off_len = max(min_cut_off_len, cut_off_ratio*len(max(texts, key=len)))
        # min_cut_off_len = min(min_cut_off_len, max_output_length)

    Returns:
    
    outputs (list of dicts): Each dict has the sequence and probability.

    """
    if not isinstance(texts, list):
        texts = [texts]

    input_encoding_dict = params['input_encoding']
    output_decoding_dict = params['output_decoding']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']

    all_decoder_outputs = generate(texts, input_encoding_dict, model, max_input_length, max_output_length, beam_size, max_beams, min_cut_off_len, cut_off_ratio)
    outputs = []

    for i, decoder_outputs in all_decoder_outputs.items():
        outputs.append([])
        for decoder_output, probs in decoder_outputs:
            outputs[-1].append({'sequence': decode_sequence(output_decoding_dict, decoder_output[0]), 'prob': np.prod(probs)})

    return outputs

def generate_greedy(texts, input_encoding_dict, model, max_input_length, max_output_length):
    """
    Main function for generating output(s) for given input(s)

    Parameters:

    texts (list/str): List of input strings or single input string.

    input_encoding_dict (dict): Encoding dictionary generated from the input strings.

    model (kerasl model): Loaded keras model.

    max_input_length (int): Max length of input sequences. All sequences will be padded to this length to support batching.

    max_output_length (int): Max output length. Need to know when to stop decoding if no padding character comes.

    Returns:
    
    outputs (list): Generated outputs.

    """
    if not isinstance(texts, list):
        texts = [texts]

    encoder_input = encode_sequences(input_encoding_dict, texts, max_input_length)
    decoder_input = np.zeros(shape=(len(encoder_input), max_output_length))
    decoder_input[:,0] = char_start_encoding
    for i in range(1, max_output_length):
        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:,i] = output[:,i]
        
        if np.all(decoder_input[:,i] == char_padding_encoding):
            return decoder_input[:,1:]

    return decoder_input[:,1:]

def infer_greedy(texts, model, params):
    """
    Main function for generating output(s) for given input(s)

    Parameters:

    texts (list/str): List of input strings or single input string.

    model (kerasl model): Loaded keras model.

    params (dict): Loaded params generated by build_params

    Returns:
    
    outputs (list): Generated outputs.

    """
    return_string = False
    if not isinstance(texts, list):
        return_string = True
        texts = [texts]

    input_encoding_dict = params['input_encoding']
    output_decoding_dict = params['output_decoding']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']

    decoder_output = generate_greedy(texts, input_encoding_dict, model, max_input_length, max_output_length)
    if return_string:
        return decode_sequence(output_decoding_dict, decoder_output[0])

    return [decode_sequence(output_decoding_dict, i) for i in decoder_output]


def build_params(input_data = [], output_data = [], params_path = 'test_params', max_lenghts = (5,5)):
    """
    Build the params and save them as a pickle. (If not already present.)

    Parameters:

    input_data (list): List of input strings.

    output_data (list): List of output strings.

    params_path (str): Path for saving the params.

    max_lenghts (tuple): (max_input_length, max_output_length)

    Returns:
    
    params (dict): Generated params (encoding, decoding dicts ..).

    """
    if os.path.exists(params_path):
        print('Loading the params file')
        params = pickle.load(open(params_path, 'rb'))
        return params
    
    print('Creating params file')
    input_encoding, input_decoding, input_dict_size = build_sequence_encode_decode_dicts(input_data)
    output_encoding, output_decoding, output_dict_size = build_sequence_encode_decode_dicts(output_data)
    params = {}
    params['input_encoding'] = input_encoding
    params['input_decoding'] = input_decoding
    params['input_dict_size'] = input_dict_size
    params['output_encoding'] = output_encoding
    params['output_decoding'] = output_decoding
    params['output_dict_size'] = output_dict_size
    params['max_input_length'] = max_lenghts[0]
    params['max_output_length'] = max_lenghts[1]

    pickle.dump(params, open(params_path, 'wb'))
    return params

def convert_training_data(input_data, output_data, params):
    """
    Encode training data.

    Parameters:

    input_data (list): List of input strings.

    output_data (list): List of output strings.

    params (dict): Generated params (encoding, decoding dicts ..).

    Returns:
    
    x, y: encoded inputs, outputs.

    """
    input_encoding = params['input_encoding']
    input_decoding = params['input_decoding']
    input_dict_size = params['input_dict_size']
    output_encoding = params['output_encoding']
    output_decoding = params['output_decoding']
    output_dict_size = params['output_dict_size']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']

    encoded_training_input = encode_sequences(input_encoding, input_data, max_input_length)
    encoded_training_output = encode_sequences(output_encoding, output_data, max_output_length)
    training_encoder_input = encoded_training_input
    training_decoder_input = np.zeros_like(encoded_training_output)
    training_decoder_input[:, 1:] = encoded_training_output[:,:-1]
    training_decoder_input[:, 0] = char_start_encoding
    training_decoder_output = np.eye(output_dict_size)[encoded_training_output.astype('int')]
    x=[training_encoder_input, training_decoder_input]
    y=[training_decoder_output]
    return x, y

def build_model(params_path = 'test/params', enc_lstm_units = 128, unroll = True, use_gru=False, optimizer='adam', display_summary=True):
    """
    Build keras model

    Parameters:

    params_path (str): Path for saving/loading the params.

    enc_lstm_units (int): Positive integer, dimensionality of the output space.

    unroll (bool): Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.

    use_gru (bool): GRU will be used instead of LSTM

    optimizer (str): optimizer to be used

    display_summary (bool): Set to true for verbose information.


    Returns:

    model (keras model): built model object.
    
    params (dict): Generated params (encoding, decoding dicts ..).

    """
    # generateing the encoding, decoding dicts
    params = build_params(params_path = params_path)

    input_encoding = params['input_encoding']
    input_decoding = params['input_decoding']
    input_dict_size = params['input_dict_size']
    output_encoding = params['output_encoding']
    output_decoding = params['output_decoding']
    output_dict_size = params['output_dict_size']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']


    if display_summary:
        print('Input encoding', input_encoding)
        print('Input decoding', input_decoding)
        print('Output encoding', output_encoding)
        print('Output decoding', output_decoding)


    # We need to define the max input lengths and max output lengths before training the model.
    # We pad the inputs and outputs to these max lengths
    encoder_input = Input(shape=(max_input_length,))
    decoder_input = Input(shape=(max_output_length,))

    # Need to make the number of hidden units configurable
    encoder = Embedding(input_dict_size, enc_lstm_units, input_length=max_input_length, mask_zero=True)(encoder_input)
    # using concat merge mode since in my experiments it gave the best results same with unroll
    if not use_gru:
        encoder = Bidirectional(LSTM(enc_lstm_units, return_sequences=True, return_state=True, unroll=unroll), merge_mode='concat')(encoder)
        encoder_outs, forward_h, forward_c, backward_h, backward_c = encoder
        encoder_h = concatenate([forward_h, backward_h])
        encoder_c = concatenate([forward_c, backward_c])
    
    else:
        encoder = Bidirectional(GRU(enc_lstm_units, return_sequences=True, return_state=True, unroll=unroll), merge_mode='concat')(encoder)        
        encoder_outs, forward_h, backward_h= encoder
        encoder_h = concatenate([forward_h, backward_h])
    

    # using 2* enc_lstm_units because we are using concat merge mode
    # cannot use bidirectionals lstm for decoding (obviously!)
    
    decoder = Embedding(output_dict_size, 2 * enc_lstm_units, input_length=max_output_length, mask_zero=True)(decoder_input)

    if not use_gru:
        decoder = LSTM(2 * enc_lstm_units, return_sequences=True, unroll=unroll)(decoder, initial_state=[encoder_h, encoder_c])
    else:
        decoder = GRU(2 * enc_lstm_units, return_sequences=True, unroll=unroll)(decoder, initial_state=encoder_h)


    # luong attention
    attention = dot([decoder, encoder_outs], axes=[2, 2])
    attention = Activation('softmax', name='attention')(attention)

    context = dot([attention, encoder_outs], axes=[2,1])

    decoder_combined_context = concatenate([context, decoder])

    output = TimeDistributed(Dense(enc_lstm_units, activation="tanh"))(decoder_combined_context)
    output = TimeDistributed(Dense(output_dict_size, activation="softmax"))(output)
    with strategy.scope():
        model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    if display_summary:
        model.summary()
    
    return model, params



if __name__ == '__main__':
    with strategy.scope():
        model, params = build_model(params_path='params', enc_lstm_units=256)
        model.load_weights('checkpoint')
        start = time.time()
        print(infer("i will be there for you", model, params))
        end = time.time()
        print(end - start)

        start = time.time()
        print(infer(["i will be there for you","these is not that gre at","i dotlike this","i work at reckonsys"], model, params))
        end = time.time()
        print(end - start)
        start = time.time()
        print(infer("i will be there for you", model, params))
        end = time.time()
        print(end - start )


# In[ ]:


cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))
score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
#cmat = (cmat.T / cmat.sum(axis=1)).T # normalized
display_confusion_matrix(cmat, score, precision, recall)
print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))


# # Predictions

# In[ ]:


test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.

print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds)
predictions = np.argmax(probabilities, axis=-1)
print(predictions)

print('Generating submission.csv file...')
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
get_ipython().system('head submission.csv')


# # Visual validation

# In[ ]:


dataset = get_validation_dataset()
dataset = dataset.unbatch().batch(20)
batch = iter(dataset)


# In[ ]:


# run this cell again for next set of images
images, labels = next(batch)
probabilities = model.predict(images)
predictions = np.argmax(probabilities, axis=-1)
display_batch_of_images((images, labels), predictions)

