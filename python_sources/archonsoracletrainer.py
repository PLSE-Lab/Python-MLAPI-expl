# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# https://github.com/TannerGilbert/Tutorials/blob/master/Keras-Tutorials/4.%20LSTM%20Text%20Generation/Keras%20LSTM%20Text%20Generation.ipynb
# https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
# https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/
# https://medium.com/@thomas_dehaene/an-nlp-view-on-holiday-movies-part-ii-text-generation-using-lstms-in-keras-36dc1ff8a6d2
# https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
# https://machinelearningmastery.com/use-dropout-lstm-networks-time-series-forecasting/

import re
import numpy as np  # linear algebra
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import random
import sys


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(corpus_txt) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = corpus_txt[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            # sys.stdout.write(next_char)
            # sys.stdout.flush()
            print(next_char, end='')
        print()

if __name__ == '__main__':
    # Input data files are available in the "../input/" directory.
    with open('../input/lasthope-v2/training.txt', 'r') as corpus:
        corpus_lines = [re.sub(r'\s\s+', ' ', line.replace(u'\xa0', u' ')) for line in corpus.read().splitlines()]
    
    corpus_txt = ' '.join(corpus_lines)
    print(corpus_txt[:300])
    
    chars = sorted(list(set(corpus_txt)))
    print(chars)
    print('total unique chars: ', len(chars))
    
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # average number of characters per english word is 5-ish
    # a good average number of words per sentence is 15 - 20
    maxlen = 29
    words = []
    next_chars = []
    # build our sequences
    for line in corpus_lines:
        if len(line) > maxlen:
            for i in range(0, len(line) - maxlen):
                words.append(line[i: i + maxlen])  # our word or context
                next_chars.append(line[i + maxlen])  # the next character in that word or context
    # so essentially the machine should learn how to spell and learn some basic grammar
    print('total sequences:', len(words))
    print(words[:3])
    print(next_chars[:3])
    
    x = np.zeros((len(words), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(words), len(chars)), dtype=np.bool)
    # one-hot-encode the the input and output variables
    # this allows the machine to do math with our characters
    for i, word in enumerate(words):
        for t, char in enumerate(word):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    # print(x[:3])
    # print(y[:3])
    
    layers = 2
    neurons = 256  # more neurons = more learning but longer epochs
    dropout = 0.3  # percentage of randomly selected neurons ignored during training.
    model = Sequential()
    # LSTM is Long-Short-Term-Memory
    # this means the machine will understand our input as a time-step sequence, 
    # recognizing that the order of the sequence matters and that the output is the next time-step
    #
    # use recurrent_dropout when dealing with RNN’s, don’t add a simple Dropout layer in between. 
    # this droput is applied to the recurrent input signal on the LSTM units.
    for i in range(layers - 1):
        # set return_sequences to True when stacking LSTM layers, for each but the final LSTM layer.
        model.add(LSTM(neurons, input_shape=(maxlen, len(chars)), recurrent_dropout=dropout, return_sequences=True))
    model.add(LSTM(neurons, recurrent_dropout=dropout))
    model.add(Dense(len(chars), activation='softmax'))
    optimizer = RMSprop(lr=0.01)  # usually a good choice for recurrent neural networks
    model.compile(loss='categorical_crossentropy', optimizer='adam')  # optimizer='adam' is another option for speed
    model.summary()
    
    model.load_weights('../input/lasthope-weights/weights.hdf5')
    
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    filepath = "weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, 
                                 monitor='loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='loss', 
                                  factor=0.2,
                                  patience=1, 
                                  min_lr=0.001)
    early_stop = EarlyStopping(monitor='loss',
                               verbose=1,
                               min_delta=0.001,
                               patience=5,
                               mode='min')
    callbacks = [print_callback, checkpoint, reduce_lr, early_stop]
    # Keras shuffles the training dataset before each training epoch. 
    # To ensure the training data patterns remain sequential, we can disable this shuffling.
    model.fit(x, y, batch_size=128, epochs=50, callbacks=callbacks, shuffle=False, verbose=2) 
    # verbose=1 => animated progress bar
    # verbose=2 => one line per epoch
# Any results you write to the current directory are saved as output.