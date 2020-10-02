'''
Original code for LR_Updater and LR_Cycle from this repo:
https://github.com/gunchagarg/learning-rate-techniques-keras

Based on the Cyclic LR strategy described here:
http://course18.fast.ai/lessons/lesson2.html
which inspired the fast.ai implementation described here:
https://docs.fast.ai/callbacks.one_cycle.html#OneCycleScheduler
'''

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import numpy as np

class LR_Updater(Callback):
    '''This callback is utilized to log learning rates every iteration (batch cycle)
    it is not meant to be directly used as a callback but extended by other callbacks
    ie. LR_Cycle
    '''
    def __init__(self, epoch_iterations):
        '''
        iterations = training batches
        epoch_iterations = number of batches in one full training cycle
        '''
        self.epoch_iterations = epoch_iterations
        self.trn_iterations = 0.
        self.history = {}
    def on_train_begin(self, logs={}):
        self.trn_iterations = 0.
        logs = logs or {}
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        K.set_value(self.model.optimizer.lr, self.setRate())
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class LR_Cycle(LR_Updater):
    '''This callback is utilized to implement cyclical learning rates
    it is based on this pytorch implementation https://github.com/fastai/fastai/blob/master/fastai
    and adopted from this keras implementation https://github.com/bckenstler/CLR
    '''
    
    def __init__(self, iterations, cycle_mult=1, max_lr_decay=1):
        '''
        iterations: initial number of iterations in one annealing cycle
        cycle_mult: used to increase the cycle length cycle_mult times after every cycle
            For example: cycle_mult = 2 will make it so that the length of the
            second cycle is twice the initial cycle length, the length of the
            third cycle is 2*2 = 4 times the initial cycle length, etc.
        max_lr_decay: scale factor by which to multiply the max learning rate in
            successive cycles
            For example: max_lr_decay = 0.5 will make it so that the max learning
            rate at the start of the second cycle is half as large as the initial
            max learning rate, the max learning rate at the start of the third
            cycle is 0.5*0.5 = 0.25 times as large, etc.
        '''
        self.min_lr = 0
        self.cycle_mult = cycle_mult
        self.max_lr_decay = max_lr_decay
        self.cycle_iterations = 0.
        super().__init__(iterations)
    
    def setRate(self):
        self.cycle_iterations += 1
        if self.cycle_iterations == self.epoch_iterations:
            self.cycle_iterations = 0.
            self.epoch_iterations *= self.cycle_mult
            self.max_lr *= self.max_lr_decay
        decay_phase = np.pi*self.cycle_iterations/self.epoch_iterations
        decay = (np.cos(decay_phase) + 1.) / 2.
        return self.max_lr * decay
    
    def on_train_begin(self, logs={}):
        super().on_train_begin(logs={})
        self.cycle_iterations = 0.
        self.max_lr = K.get_value(self.model.optimizer.lr)




'''
Original code for LearningRateMultiplier from this repo:
https://github.com/stante/keras-contrib/blob/feature-lr-multiplier/keras_contrib/optimizers/lr_multiplier.py

Significant changes have been made to update the code to work with the version
of keras that comes with tensorflow v1.14.

Based on the discriminative layer training strategy implemented in the fast.ai
library, as described here:
https://docs.fast.ai/basic_train.html#Discriminative-layer-training
'''

from tensorflow.keras.optimizers import Adam

class Adam_DiscLR(Adam):
    '''Adam wrapper implementing "discriminative" i.e. per-layer learning rates.
    # Arguments
        layers: List of layer names, ordered from lowest to highest.
        lorate: Rate to apply to lower layers. If None given, this will be set
            to hirate. Base lr is set to lorate - this overrides the
            learning_rate kwarg.
        hirate: Rate to apply to higher layers. If None given, this will be set
            to lorate. Base lr is set to lorate - this overrides the
            learning_rate kwarg.
        lr_multipliers: Dictionary of the per layer factors. For
            example `lr_multipliers={'conv2d_1':0.5}`. Overrides hirate and
            lorate when given; base lr is set according to the learning_rate
            kwarg.
        **kwargs: The arguments for instantiating the wrapped Adam optimizer.
    '''
    def __init__(self, layers=None, lorate=None, hirate=None, lr_multipliers=None, **kwargs):
        super().__init__(**kwargs)
        self._lr_multipliers = lr_multipliers or {}

        if lr_multipliers is None and layers is not None:
            lr_multipliers = {}
            if hirate is not None and lorate is None:
                lorate = hirate
            if lorate is not None and hirate is None:
                hirate = lorate
            if lorate is not None:
                print('Setting learning rate based on lorate,hirate')
                K.set_value(self.lr, lorate)
                n_layers = len(layers)
                for L in range(n_layers):
                    layer = layers[L]
                    multiplier = (float(hirate)/float(lorate))**(float(L)/float(n_layers-1.))
                    lr_multipliers[layer] = multiplier
            self._lr_multipliers = lr_multipliers

    def get_updates(self, loss, params):
        mult_lr_params = {p: self._lr_multipliers[p.name.split('/')[0]] for p in params
                          if p.name.split('/')[0] in self._lr_multipliers}
        base_lr_params = [p for p in params if p.name.split('/')[0] not in self._lr_multipliers]

        updates = []
        base_lr = K.get_value(self.lr)
        for param, multiplier in mult_lr_params.items():
            K.set_value(self.lr, base_lr * multiplier)
            updates.extend(super().get_updates(loss, [param]))

        K.set_value(self.lr, base_lr)
        updates.extend(super().get_updates(loss, base_lr_params))

        return updates

    def get_config(self):
        config = super().get_config()
        config.update({'lr_multipliers': self._lr_multipliers})
        return config

