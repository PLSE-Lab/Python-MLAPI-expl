#!/usr/bin/env python
# coding: utf-8

# For those who uses TensorFlow 2.1.0 I want to share the Quadratic Weighted Kappa metric implementation used in the [Prostate cANcer graDe Assessment (PANDA) Challenge](https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview/evaluation).

# In[ ]:


import tensorflow as tf
import numpy as np

class QuadraticWeightedKappa(tf.keras.metrics.Metric):
    def __init__(self, maxClassesCount=6, name='Kappa', **kwargs):        
        super(QuadraticWeightedKappa, self).__init__(name=name, **kwargs)
        self.M = maxClassesCount

        self.O = self.add_weight(name='O', initializer='zeros',shape=(self.M,self.M,), dtype=tf.int64)
        self.W = self.add_weight(name='W', initializer='zeros',shape=(self.M,self.M,), dtype=tf.float32)
        self.actualHist = self.add_weight(name='actHist', initializer='zeros',shape=(self.M,), dtype=tf.int64)
        self.predictedHist = self.add_weight(name='predHist', initializer='zeros',shape=(self.M,), dtype=tf.int64)
        
        # filling up the content of W once
        w = np.zeros((self.M,self.M),dtype=np.float32)
        for i in range(0,self.M):
            for j in range(0,self.M):
                w[i,j] = (i-j)*(i-j) / ((self.M - 1)*(self.M - 1))
        self.W.assign(w)
    
    def reset_states(self):
        """Resets all of the metric state variables.
        This function is called between epochs/steps,
        when a metric is evaluated during training.
        """
        # value should be a Numpy array
        zeros1D = np.zeros(self.M)
        zeros2D = np.zeros((self.M,self.M))
        tf.keras.backend.batch_set_value([
            (self.O, zeros2D),
            (self.actualHist, zeros1D),
            (self.predictedHist,zeros1D)
        ])



    def update_state(self, y_true, y_pred, sample_weight=None):
        # shape is: Batch x 1
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        y_true_int = tf.cast(tf.math.round(y_true), dtype=tf.int64)
        y_pred_int = tf.cast(tf.math.round(y_pred), dtype=tf.int64)

        confM = tf.math.confusion_matrix(y_true_int, y_pred_int, dtype=tf.int64, num_classes=self.M)

        # incremeting confusion matrix and standalone histograms
        self.O.assign_add(confM)

        cur_act_hist = tf.math.reduce_sum(confM, 0)
        self.actualHist.assign_add(cur_act_hist)

        cur_pred_hist = tf.math.reduce_sum(confM, 1)
        self.predictedHist.assign_add(cur_pred_hist)

    def result(self):
        EFloat = tf.cast(tf.tensordot(self.actualHist,self.predictedHist, axes=0),dtype=tf.float32)
        OFloat = tf.cast(self.O,dtype=tf.float32)
        
        # E must be normalized "such that E and O have the same sum"
        ENormalizedFloat = EFloat / tf.math.reduce_sum(EFloat) * tf.math.reduce_sum(OFloat)

        
        return 1.0 - tf.math.reduce_sum(tf.math.multiply(self.W, OFloat))/tf.math.reduce_sum(tf.multiply(self.W, ENormalizedFloat))


# To use the metric in TF Keras API use it as follows in the model compile call:
# ```
# model.compile(
#           optimizer= ... ,
#           loss= ... ,
#           metrics=[QuadraticWeightedKappa()]
#           )
# ```
# 
# I suppose it can be easily adobted for pure Keras.
# 
# Hope, it will be helpful :-)
