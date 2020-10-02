#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# 
# ## The problem
# 
# The Porto Seguro data set is heavily unbalanced, with positive examples around 3% of the total data set. This makes it hard for conventional neural network loss functions to work with. We can use class weights to correct this, but then we will tend to overfit to the positive examples. That might be ok if we, say, had only 3% pictures of dogs in a sample of pictures of cats and dogs, but the problem is that a driver doesn't *always* claim in the same way that a dog is *always* a dog; the dataset is inherently noisy and probablistic by it's nature. This is where a ranking based metric like Gini or AUC can be useful.
# 
# However, this is a problem for neural nets, as AUC (and hence Gini which is (2AUC-1) is not differentiable. Using conventional gradient descent metrics such as binary cross entropy will lead us to just classify everything as in class zero, which will give around 97% accuracy, with a good chance of the relative predictions within a class - on which AUC depends - being very noisy as we haven't trained our network to optimise them.
# 
# ## The solution
# 
# This notebook demonstrates a custom loss function for neural nets, that provides a differentiable approximation to AUC. AUC, in turn, has a linear relationship with Gini, hence this is very useful when we want to train a network to maximise AUC.
# 
# We set up 2 identical NNs and run them for a few epochs, to show how this approach improves convergence on AUC compared to binary crossentropy.
# 
# I've used this to get a network that has a local CV AUC around 0.642, which corresponds to Gini of 0.284. The performance on the LB test set is considerably worse (around 0.270) in the one case I tested - mainly I've used them as inputs to blends.
# 
# This is hacked together from various bits of my local code, and hasn't been thoroughly tested, so let please me know of any bugs etc.
# 
# I would have coded as a script, but I need to use the Theano backend as the AUC function uses Theano specific code. If anyone knows how to make Kaggle Kernels use the Theano backend for script, let me know, and I'll post it as a script.
# 
# This is my pretty much my first kernel so feedback welcome.
# 
# **Imports and constants**
# 
# First things first...note that we need the Theano backend. Converting to Tensorflow is on my to-do list.

# In[ ]:


import numpy as np
import pandas as pd

get_ipython().run_line_magic('env', 'KERAS_BACKEND=theano')

from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import custom_object_scope
from keras import callbacks

from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

import theano

# train and test data path
DATA_TRAIN_PATH = '../input/train.csv'
DATA_TEST_PATH = '../input/test.csv'

featuresToDrop = [
    'ps_calc_10',
    'ps_calc_01',
    'ps_calc_02',
    'ps_calc_03',
    'ps_calc_13',
    'ps_calc_08',
    'ps_calc_07',
    'ps_calc_12',
    'ps_calc_04',
    'ps_calc_17_bin',
    'ps_car_10_cat',
    'ps_car_11_cat',
    'ps_calc_14',
    'ps_calc_11',
    'ps_calc_06',
    'ps_calc_16_bin',
    'ps_calc_19_bin',
    'ps_calc_20_bin',
    'ps_calc_15_bin',
    'ps_ind_11_bin',
    'ps_ind_10_bin'
]


# **The secret sauce**
# 
# This is where the magic happens - the soft_AUC function. This 
# - Takes the predictions
# - Splits them into groups according to whether the true values are one/zero
# - Takes each pair of predictions from the one/zero groups, and subtracts the zeroes from the ones.
# - Takes the mean of the sigmoid of the result
# 
# If AUC is perfect, an (actual) one in the CV data will always have a higher pred than a zero in the CV data. Each time the prediction is wrong, and a one has a lower pred than a zero, the output loss is increased. Hence this is an suitable loss function to substitute for genuine AUC in that it decreases as AUC decreases and vice versa.
# 
# Like AUC, we only care about relative ordering of predictions between the classes. We don't care about the absolute values of the predictions. This means that your final output values from your NN will also only care about ordering, and hence you use them to blend you will need to use ranking or similar to blend.
# 
# It's important to note that you need a large enough batch size, as if you have no examples of one class you'll get no data, and ideally you want several positive cases in each batch. I used a batch size of 4096, which with a 3% approx positive class rate, gives around 100 positives per batch - enough to give a useful result but not so many as to make calculations take forever. I did try calculating on the whole training batch, but convergence was not as good.
# 
# Some useful references:
#  * http://www.ipipan.waw.pl/~sj/pdf/PKDD07web.pdf
#  * https://github.com/Lasagne/Lasagne/issues/767 - my code based heavily on this code.
#  * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.2.3727&rep=rep1&type=pdf

# In[ ]:



# An analogue to AUC which takes the differences between each pair of true/false predictions
# and takes the average sigmoid of the differences to get a differentiable loss function.
# Based on code and ideas from https://github.com/Lasagne/Lasagne/issues/767
def soft_AUC_theano(y_true, y_pred):
    # Extract 1s
    pos_pred_vr = y_pred[y_true.nonzero()]
    # Extract zeroes
    neg_pred_vr = y_pred[theano.tensor.eq(y_true, 0).nonzero()]
    # Broadcast the subtraction to give a matrix of differences  between pairs of observations.
    pred_diffs_vr = pos_pred_vr.dimshuffle(0, 'x') - neg_pred_vr.dimshuffle('x', 0)
    # Get signmoid of each pair.
    stats = theano.tensor.nnet.sigmoid(pred_diffs_vr * 2)
    # Take average and reverse sign
    return 1-theano.tensor.mean(stats) # as we want to minimise, and get this to zero


# **Callback**
# 
# Now, we define a callback to print out our SKLearn AUC, and add this to the logs so we can use it for early stopping. See https://keras.io/callbacks/
# 
# In my own version I also use this to save down best scores in csv files where I store metadata for each network I run. This makes it much easier to look at trends of hyperparameter performance with overnight runs.

# In[ ]:



# This callback records the SKLearn calculated AUC each round, for use by early stopping
# It also has slots where you can save down metadata or the model at useful points -
# for Kaggle kernel purposes I've commented these out
class AUC_SKlearn_callback(callbacks.Callback):
    def __init__(self, X_train, y_train, useCv = True):
        super(AUC_SKlearn_callback, self).__init__()
        self.bestAucCv = 0
        self.bestAucTrain = 0
        self.cvLosses = []
        self.bestCvLoss = 1,
        self.X_train = X_train
        self.y_train = y_train
        self.useCv = useCv

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        train_pred = self.model.predict(np.array(self.X_train))
        aucTrain = roc_auc_score(self.y_train, train_pred)
        print("SKLearn Train AUC score: " + str(aucTrain))

        if (self.bestAucTrain < aucTrain):
            self.bestAucTrain = aucTrain
            print ("Best SKlearn AUC training score so far")
            #**TODO: Add your own logging/saving/record keeping code here

        if (self.useCv) :
            cv_pred = self.model.predict(self.validation_data[0])
            aucCv = roc_auc_score(self.validation_data[1], cv_pred)
            print ("SKLearn CV AUC score: " +  str(aucCv))

            if (self.bestAucCv < aucCv) :
                # Great! New best *actual* CV AUC found (as opposed to the proxy AUC surface we are descending)
                print("Best SKLearn genuine AUC so far so saving model")
                self.bestAucCv = aucCv

                # **TODO: Add your own logging/model saving/record keeping code here.
                self.model.save("best_auc_model.h5", overwrite=True)

            vl = logs.get('val_loss')
            if (self.bestCvLoss < vl) :
                print("Best val loss on SoftAUC so far")
                #**TODO -  Add your own logging/saving/record keeping code here.
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        # logs include loss, and optionally acc( if accuracy monitoring is enabled).
        return


# **Model creation and training functions**
# 
# Standard model creation and training code, note that we reference the custom loss function and the callback here.

# In[ ]:


# Create the model.
def create_model_AUC(input_dim, first_layer_size, second_layer_size, third_layer_size, lr, l2reg, dropout):
    return create_model(input_dim, first_layer_size, second_layer_size, third_layer_size, lr, l2reg, dropout, "AUC")

def create_model_bce(input_dim, first_layer_size, second_layer_size, third_layer_size, lr, l2reg, dropout):
    return create_model(input_dim, first_layer_size, second_layer_size, third_layer_size, lr, l2reg, dropout, "crossentropy")


def create_model(input_dim, first_layer_size, second_layer_size, third_layer_size, lr, l2reg, dropout, mode="AUC") :
    print("Creating model with input dim ", input_dim)
    # likely to need tuning!
    reg = regularizers.l2(l2reg)

    model = Sequential()

    model.add(Dense(units=first_layer_size, kernel_initializer='lecun_normal', kernel_regularizer=reg, activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(units=second_layer_size, kernel_initializer='lecun_normal', activation='relu', kernel_regularizer=reg))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(dropout))

    model.add(Dense(units=third_layer_size, kernel_initializer='lecun_normal', activation='relu', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(1, kernel_initializer='lecun_normal', activation='sigmoid'))

    # classifier.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=['mae', 'accuracy'])
    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    if (mode == "AUC"):
        model.compile(loss=soft_AUC_theano, metrics=[soft_AUC_theano], optimizer=opt)  # not sure whether to use metrics here?
    else:
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)  # not sure whether to use metrics here?
    return model


def train_model( X_train, y_train, model, valSplit=0.15, epochs = 5, batch_size = 4096):

    callbacksList = [AUC_SKlearn_callback(X_train, y_train, useCv = (valSplit > 0))]
    if (valSplit > 0) :
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5,
                                                       verbose=0, mode='min')
        callbacksList.append( early_stopping )
    return model.fit(x=np.array(X_train), y=np.array(y_train),
                        callbacks=callbacksList, validation_split=valSplit,
                        verbose=2, batch_size=batch_size, epochs=epochs)


# **Data preparation**
# 
# We remove some of the noisier features, there was an excellent Kernel which I will try to find and cite that the list of columns to remove was taken from. This makes a big difference to effectiveness.

# In[ ]:




def scale_features(df_for_range, df_to_scale, columnsToScale) :
    # Scale columnsToScale in df_to_scale
    columnsOut = list(map( (lambda x: x + "_scaled"), columnsToScale))
    for c, co in zip(columnsToScale, columnsOut) :
        scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        print("scaling ", c ," to ",co)
        vals = df_for_range[c].values.reshape(-1, 1)
        scaler.fit(vals )
        df_to_scale[co]=scaler.transform(df_to_scale[c].values.reshape(-1,1))

    df_to_scale.drop (columnsToScale, axis=1, inplace = True)

    return df_to_scale


def one_hot (df, cols):
    # One hot cols requested, drop original cols, return df
    df = pd.concat([df, pd.get_dummies(df[cols], columns=cols)], axis=1)
    df.drop(cols, axis=1, inplace = True)
    return df

def get_data() :
    X_train = pd.read_csv(DATA_TRAIN_PATH, index_col = "id")
    X_test = pd.read_csv(DATA_TEST_PATH, index_col = "id")

    y_train = pd.DataFrame(index = X_train.index)
    y_train['target'] = X_train.loc[:,'target']
    X_train.drop ('target', axis=1, inplace = True)
    X_train.drop (featuresToDrop, axis=1, inplace = True)
    X_test.drop (featuresToDrop,axis=1, inplace = True)

    # car_11 is really a cat col
    X_train.rename(columns={'ps_car_11': 'ps_car_11a_cat'}, inplace=True)
    X_test.rename(columns={'ps_car_11': 'ps_car_11a_cat'}, inplace=True)

    cat_cols = [elem for elem in list(X_train.columns) if "cat" in elem]
    bin_cols = [elem for elem in list(X_train.columns) if "bin" in elem]
    other_cols = [elem for elem in list(X_train.columns) if elem not in bin_cols and elem not in cat_cols]

    # Scale numeric features in region of -1,1 using training set as the scaling range
    X_test = scale_features(X_train, X_test, columnsToScale=other_cols)
    X_train = scale_features(X_train, X_train, columnsToScale=other_cols)

    X_train = one_hot(X_train, cat_cols)
    X_test = one_hot(X_test, cat_cols)


    return X_train, X_test, y_train


# **Put it all together**
# 
# We set this up to run 2 comparable networks over a few epochs, to demonstrate that convergence is superior. I only run for 5 epochs to give a flavour of the comparison and avoid timing out the kernel.
# 
# I found each epoch took around 3-5 minutes on my GT 1070 based windows system, and decent results were obtained after about 30 epochs, so around 2 hours training time. I've not given away the best hyperparameters I've found; I'll leave that as an exercise for the reader.
# 
# The submission generation code is untested, and you would want to run the network for longer and tune the hyperparameters a bit anyway before using this.
# 

# In[ ]:




def makeOutputFile(pred_fun, test, subsFile) :
    df_out = pd.DataFrame(index=test.index)
    y_pred = pred_fun( test )
    df_out['target'] = y_pred
    df_out.to_csv(subsFile, index_label="id")

def main() :
    X_train, X_test, y_train = get_data()
    model = create_model( input_dim=X_train.shape[1],
                          first_layer_size=300,
                          second_layer_size=200,
                          third_layer_size=200,
                          lr=0.0001,
                          l2reg = 0.1,
                          dropout = 0.2,
                          mode="AUC")

    train_model(X_train, y_train, model)

    with custom_object_scope({'soft_AUC_theano': soft_AUC_theano}):
        pred_fun = lambda x: model.predict(np.array(x))
        makeOutputFile(pred_fun, X_test, "auc.csv")

    model = create_model_bce( input_dim=X_train.shape[1],
                          first_layer_size=300,
                          second_layer_size=200,
                          third_layer_size=200,
                          lr=0.0001,
                          l2reg = 0.1,
                          dropout = 0.2)

    train_model(X_train, y_train, model)

    pred_fun = lambda x: model.predict(np.array(x))
    makeOutputFile(pred_fun, X_test, "no_auc.csv")

main()


# **Conclusions**
# 
# We can see that the AUC (and hence Gini) rises much quicker using the custom loss function. Please upvote if you like this kernal or find it useful.
