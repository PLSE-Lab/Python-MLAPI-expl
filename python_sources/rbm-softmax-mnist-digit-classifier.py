# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import argparse
import time

from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda
from keras.utils import multi_gpu_model
from keras import optimizers
from keras import backend as K
from keras.layers import Layer, Input
from keras import initializers

class RBM(Layer):
    """Restricted Boltzmann Machine based on Keras."""
    def __init__(self, hps, output_dim, name=None, **kwargs):
        self.hps = hps
        self.output_dim = output_dim
        self.name = name
        super(RBM, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.rbm_weight = self.add_weight(name='rbm_weight'
                                 , shape=(input_shape[1], self.output_dim)
                                 , initializer='uniform' # Which initializer is optimal?
                                 , trainable=True)

        self.hidden_bias = self.add_weight(name='rbm_hidden_bias'
                                           , shape=(self.output_dim, )
                                           , initializer='uniform'
                                           , trainable=True)
        self.visible_bias = K.variable(initializers.get('uniform')((input_shape[1], ))
                            , dtype=K.floatx()
                            , name='rbm_visible_bias')
        
        # Make symbolic computation objects.
        # Transform visible units.
        self.input_visible = K.placeholder(shape=(None, input_shape[1]), name='input_visible')
        self.transform = K.sigmoid(K.dot(self.input_visible, self.rbm_weight) + self.hidden_bias)
        self.transform_func = K.function([self.input_visible], [self.transform])
  
        # Transform hidden units.      
        self.input_hidden = K.placeholder(shape=(None, self.output_dim), name='input_hidden')
        self.inv_transform = K.sigmoid(K.dot(self.input_hidden, K.transpose(self.rbm_weight)) + self.visible_bias)
        self.inv_transform_func = K.function([self.input_hidden], [self.inv_transform])
        
        # Calculate free energy.
        self.free_energy = -1 * (K.squeeze(K.dot(self.input_visible, K.expand_dims(self.visible_bias, axis=-1)), -1) +\
                                K.sum(K.log(1 + K.exp(K.dot(self.input_visible, self.rbm_weight) +\
                                                self.hidden_bias)), axis=-1))
        self.free_energy_func = K.function([self.input_visible], [self.free_energy])

        super(RBM, self).build(input_shape)
        
    def call(self, x):
        return K.sigmoid(K.dot(x, self.rbm_weight) + self.hidden_bias) # Float type?
    
    def transform(self, v):
        return self.transform_func(v)
    
    def inv_transform(self, h):
        return self.inv_transform_func(h)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def cal_free_energy(self, v):
        return self.free_energy_func(v)
    
    def fit(self, V, verbose=1):
        """Train RBM with the data V.
        
        Parameters
        ----------
        V : 2d numpy array
            Visible data (batch size x input_dim).
        verbose : integer
            Verbose mode (default, 1).
        """
        num_step = V.shape[0] // self.hps['batch_size'] \
            if V.shape[0] % self.hps['batch_size'] == 0 else V.shape[0] // self.hps['batch_size'] + 1 # Exception processing?
             
        for k in range(self.hps['epochs']):
            if verbose == 1:
                print(k + 1, '/', self.hps['epochs'], ' epochs')

            # Contrastive divergence.
            v_pos = self.input_visible
            h_pos = self.transform
            v_neg = K.cast(K.less(K.random_uniform(shape=(self.hps['batch_size'], V.shape[1]))
                    , K.sigmoid(K.dot(h_pos, K.transpose(self.rbm_weight)) + self.visible_bias))
                    , dtype=np.float32)
            h_neg = K.sigmoid(K.dot(v_neg, self.rbm_weight) + self.hidden_bias)
            update = K.transpose(K.transpose(K.dot(K.transpose(v_pos), h_pos)) \
                                 - K.dot(K.transpose(h_neg), v_neg))
            self.rbm_weight_update_func = K.function([self.input_visible]
                                            , [K.update_add(self.rbm_weight, self.hps['lr'] * update)])
            self.hidden_bias_update_func = K.function([self.input_visible]
                                            , [K.update_add(self.hidden_bias, self.hps['lr'] \
                                            * (K.sum(h_pos, axis=0) - K.sum(h_neg, axis=0)))])
            self.visible_bias_update_func = K.function([self.input_visible]
                                            , [K.update_add(self.visible_bias, self.hps['lr'] \
                                            * (K.sum(v_pos, axis=0) - K.sum(v_neg, axis=0)))])
            
            # Create the fist visible nodes sampling object.
            self.sample_first_visible = K.function([self.input_visible]
                                                , [v_neg])       
            for i in range(num_step):
                if i == (num_step - 1):
                    # Contrastive divergence.
                    v_pos = self.input_visible
                    h_pos = self.transform
                    v_neg = K.cast(K.less(K.random_uniform(shape=(V.shape[0] - int(i*self.hps['batch_size'])
                                   , V.shape[1])) #?
                                   , K.sigmoid(K.dot(h_pos, K.transpose(self.rbm_weight)) \
                                   + self.visible_bias)), dtype=np.float32)
                    h_neg = K.sigmoid(K.dot(v_neg, self.rbm_weight) + self.hidden_bias)
                    update = K.transpose(K.transpose(K.dot(K.transpose(v_pos), h_pos)) \
                                         - K.dot(K.transpose(h_neg), v_neg))
                    self.rbm_weight_update_func = K.function([self.input_visible]
                                                , [K.update_add(self.rbm_weight, self.hps['lr'] * update)])
                    self.hidden_bias_update_func = K.function([self.input_visible]
                                                 , [K.update_add(self.hidden_bias, self.hps['lr'] \
                                                 * (K.sum(h_pos, axis=0) - K.sum(h_neg, axis=0)))])
                    self.visible_bias_update_func = K.function([self.input_visible]
                                                  , [K.update_add(self.visible_bias, self.hps['lr'] \
                                                  * (K.sum(v_pos, axis=0) - K.sum(v_neg, axis=0)))])

                    # Create the fist visible nodes sampling object.
                    self.sample_first_visible = K.function([self.input_visible]
                                                , [v_neg])

                    V_batch = [V[int(i*self.hps['batch_size']):V.shape[0]]]
                    
                    # Train.
                    self.rbm_weight_update_func(V_batch)
                    self.hidden_bias_update_func(V_batch)
                    self.visible_bias_update_func(V_batch)
                else:
                    V_batch = [V[int(i*self.hps['batch_size']):int((i+1)*self.hps['batch_size'])]]
                    
                    # Train.
                    self.rbm_weight_update_func(V_batch)
                    self.hidden_bias_update_func(V_batch)
                    self.visible_bias_update_func(V_batch)
            
                # Calculate a training score by each step.
                # Free energy of the input visible nodes.
                fe = self.cal_free_energy(V_batch)
                
                # Free energy of the first sampled visible nodes.
                V_p_batch = self.sample_first_visible(V_batch)
                fe_p = self.cal_free_energy(V_p_batch)
                
                score = np.mean(np.abs(fe[0] - fe_p[0])) # Scale?
                print('{0:d}/{1:d}, score: {2:f}'.format(i + 1, num_step, score))

# Constants.
DEBUG = True
MULTI_GPU = False
NUM_GPUS = 4

class MNISTClassifier(object):
    """MNIST digit classifier using the RBM + Softmax model."""
    # Constants.
    MODEL_PATH = 'digit_classificaton_model.h5'
    IMAGE_SIZE = 784
    
    def __init__(self, hps, nn_arch_info, model_loading=False):
        self.hps = hps
        self.nn_arch_info = nn_arch_info

        if model_loading: 
            if MULTI_GPU:
                self.digit_classificaton_model = load_model(self.MODEL_PATH, custom_objects={'RBM': RBM}) # Custom layer loading problem?
                self.rbm = self.digit_classificaton_model.get_layer('rbm')
                
                self.digit_classificaton_parallel_model = multi_gpu_model(self.model, gpus = NUM_GPUS)
                opt = optimizers.Adam(lr=self.hps['lr']
                                        , beta_1=self.hps['beta_1']
                                        , beta_2=self.hps['beta_2']
                                        , decay=self.hps['decay']) 
                self.digit_classificaton_parallel_model.compile(optimizer=opt, loss='mse') 
            else:
                self.digit_classificaton_model = load_model(self.MODEL_PATH, custom_objects={'RBM': RBM})
                self.rbm = self.digit_classificaton_model.get_layer('rbm')
        else:        
            # Design the model.
            input_image = Input(shape=(self.IMAGE_SIZE,))
            x = Lambda(lambda x: x/255)(input_image)
            
            # RBM layer.
            self.rbm = RBM(self.hps['rbm_hps'], self.nn_arch_info['output_dim'], name='rbm')
            x = self.rbm(x) #?
            
            # Softmax layer.
            output = Dense(10, activation='softmax')(x)
            
            # Create a model.
            self.digit_classificaton_model = Model(inputs=[input_image], outputs=[output])
            
            opt = optimizers.Adam(lr=self.hps['lr']
                                    , beta_1=self.hps['beta_1']
                                    , beta_2=self.hps['beta_2']
                                    , decay=self.hps['decay'])
            
            self.digit_classificaton_model.compile(optimizer=opt, loss='categorical_crossentropy')
            self.digit_classificaton_model.summary() 

    def train(self):
        """Train."""
        # Load training data.
        V, gt = self._load_training_data()
        
        # Semi-supervised learning.
        # Unsupervised learning.
        # RBM training.
        print('Train the RBM model.')
        self.rbm.fit(V)
        
        # Supervised learning.
        print('Train the NN model.')
        if MULTI_GPU:
            self.digit_classificaton_parallel_model.fit(V
                                           , gt
                                           , batch_size=self.hps['batch_size']
                                           , epochs=self.hps['epochs']
                                           , verbose=1)        
        else:
            self.digit_classificaton_model.fit(V
                                           , gt
                                           , batch_size=self.hps['batch_size']
                                           , epochs=self.hps['epochs']
                                           , verbose=1)

        print('Save the model.')            
        self.digit_classificaton_model.save(self.MODEL_PATH)
    
    def _load_training_data(self):
        """Load training data."""
        train_df = pd.read_csv('../input/train.csv')
        V = []
        gt = []
        
        for i in range(train_df.shape[0]):
            V.append(train_df.iloc[i, 1:].values/255)
            t_gt = np.zeros(shape=(10,))
            t_gt[train_df.iloc[i,0]] = 1.
            gt.append(t_gt)
        
        V = np.asarray(V, dtype=np.float32)
        gt = np.asarray(gt, dtype=np.float32)
        
        return V, gt
    
    def test(self):
        """Test."""
        # Load test data.
        V = self._load_test_data()
        
        # Predict digits.
        res = self.digit_classificaton_model.predict(V
                                                     , verbose=1)
        
        # Record results into a file.
        with open('output', 'w') as f:
            f.write('ImageId,Label\n')
            
            for i, v in enumerate(res):
                f.write(str(i + 1) + ',' + str(np.argmax(v)) + '\n') 
        
    def _load_test_data(self):
        """Load test data."""
        test_df = pd.read_csv('../input/test.csv')
        V = []
        
        for i in range(test_df.shape[0]):
            V.append(test_df.iloc[i, :].values/255)
        
        V = np.asarray(V, dtype=np.float32)
        
        return V       

def main():
    """Main."""
    hps = {}
    nn_arch_info = {}

    # Get arguments.      
    nn_arch_info['output_dim'] = 128   
        
    hps['lr'] = 0.001
    hps['beta_1'] = 0.99
    hps['beta_2'] = 0.99
    hps['decay'] = 0.0
    hps['batch_size'] = 128
    hps['epochs'] = 1 # 100: Saturation #epochs
        
    rbm_hps = {}
    rbm_hps['lr'] = 0.001
    rbm_hps['batch_size'] = 128
    rbm_hps['epochs'] = 1 # 100: Saturation #epochs
    hps['rbm_hps'] = rbm_hps        
        
    model_loading = False        
        
    # Train.
    mc = MNISTClassifier(hps, nn_arch_info, model_loading)
        
    ts = time.time()
    mc.train()
    mc.test()
    te = time.time()
        
    print('Elasped time: {0:f}s'.format(te-ts))

main()