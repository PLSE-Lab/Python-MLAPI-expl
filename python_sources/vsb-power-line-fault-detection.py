#!/usr/bin/env python
# coding: utf-8

# ****-This kernel is summary of kernesl I have seen for the dataset as well as some modification.****

# In[ ]:


import keras
import keras.backend as K
from keras.layers import LSTM,Dropout,Dense,TimeDistributed,Conv1D,MaxPooling1D,Flatten
from keras.models import Sequential
import tensorflow as tf
import pandas as pd
# import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Both numpy and scipy has utilities for FFT which is an endlessly useful algorithm
from numpy.fft import *
from scipy import fftpack


# **Signals:**
# 
# - 800.000 measurement points for 8712 signals.
# - The signals are **three-phased** so there are 2904 distinct signaling instances.
# - Three phase signals:
#     - Sums to zero.
#     - When one fails other continue to carry the current.
#     - Can be rectified to be converted to DC current.
#     - Ripples in rectification can be seen on failure.
#     
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/3-phase_flow.gif/357px-3-phase_flow.gif)

# ### What is partial discharge?
# 
#  - Typical situation of PD: imagine there is an internal cavity/void or **impurity in insulation**. 
#  - When **High Voltage** is applied on conductor, a field is also induced on the cavity. Further, when the field increases, this **defect breaks down** and **discharges** different forms of energy which result in partial discharge.
#  - This phenomenon is damaging over a long period of time. It is not event that occurs suddenly. 

# ### Classical Modes of Detection
# - Partial Discharges can be detected by **measuring the emissions** they give off: Ultrasonic Sound, Transient Earth Voltages (TEV and UHF energy).
# - Is it possible to enhance the modes of detection by **better feature extraction** for the classifiers?
# - **Intel Mobile ODT** challenge on 2017 was about topping **classical image processing** methods by automatic feature extaction using pre-trained CNN models and **transfer learning**.
# - **Two possible approaches**:
#     - FE on signals and feeding them into NNs for classification.
#     - Using NNs further as feature extractors and then use shallow classifiers (XGBoost) for binary classification
# 

# ### **TASK:** Classify long-term failure of covered conductors based on signal characteristics:
# - Extract features from time series data for classification.
# - Use **CNN** for further FE and **LSTM** to get temporal dependencies and perform time series classification on the top layer.

# In[ ]:


signals = pq.read_table('../input/train.parquet', columns=[str(i) for i in range(999)]).to_pandas()
print('signals shape is: ',signals.shape)
#Since data size is big we just load one third of it for now
signals = np.array(signals).T.reshape((999//3, 3, 800000))
print('signals shape after reshaping is: ', signals.shape)
train_df = pd.read_csv('../input/metadata_train.csv')
print('metadata shape is: ',train_df.shape)


# In[ ]:


fig, axs = plt.subplots(2, 2, constrained_layout=True,figsize=(15, 10))

axs[0,0].set_title('normal wires')
axs[0,0].plot(signals[0, 0, :], label='Phase 0')
axs[0,0].plot(signals[0, 1, :], label='Phase 1')
axs[0,0].plot(signals[0, 2, :], label='Phase 2')
axs[0,0].legend()

axs[1,0].set_title('damaged wires')
axs[1,0].plot(signals[1, 0, :], label='Phase 0')
axs[1,0].plot(signals[1, 1, :], label='Phase 1')
axs[1,0].plot(signals[1, 2, :], label='Phase 2')
axs[1,0].legend()

axs[0,1].set_title('normal wires')
axs[0,1].plot(signals[77, 0, :], label='Phase 0')
axs[0,1].plot(signals[77, 1, :], label='Phase 1')
axs[0,1].plot(signals[77, 2, :], label='Phase 2')
axs[0,1].legend()

axs[1,1].set_title('damaged wires')
axs[1,1].plot(signals[76, 0, :], label='Phase 0')
axs[1,1].plot(signals[76, 1, :], label='Phase 1')
axs[1,1].plot(signals[76, 2, :], label='Phase 2')
axs[1,1].legend()


plt.show()


# In[ ]:


target = train_df['target']
plt.figure(figsize=(15, 10))
sns.countplot(target)
plt.show()

print('number of damaged samples', sum(target))
print('number of normal sampes', target.shape[0]-sum(target))


# In[ ]:


#FFT to filter out HF components and get main signal profile
def low_pass(s, threshold=1e4):
    fourier = rfft(s)
    frequencies = rfftfreq(s.size, d=2e-2/s.size)
    fourier[frequencies > threshold] = 0
    return irfft(fourier)


# In[ ]:


#normal one
lf_normal1_1 = low_pass(signals[0, 0, :])
lf_normal1_2 = low_pass(signals[0, 1, :])
lf_normal1_3 = low_pass(signals[0, 2, :])
#normal two
lf_normal2_1 = low_pass(signals[77, 0, :])
lf_normal2_2 = low_pass(signals[77, 1, :])
lf_normal2_3 = low_pass(signals[77, 2, :])
#damaged one
lf_damaged1_1 = low_pass(signals[1, 0, :])
lf_damaged1_2 = low_pass(signals[1, 1, :])
lf_damaged1_3 = low_pass(signals[1, 2, :])
#damaged two
lf_damaged2_1 = low_pass(signals[76, 0, :])
lf_damaged2_2 = low_pass(signals[76, 1, :])
lf_damaged2_3 = low_pass(signals[76, 2, :])


# In[ ]:


fig, axs = plt.subplots(2, 2, constrained_layout=True,figsize=(15, 10))

axs[0,0].set_title('normal wires')
axs[0,0].plot(lf_normal1_1, label='Phase 0')
axs[0,0].plot(lf_normal1_2, label='Phase 1')
axs[0,0].plot(lf_normal1_3, label='Phase 2')
axs[0,0].legend()

axs[1,0].set_title('damaged wires')
axs[1,0].plot(lf_damaged1_1, label='Phase 0')
axs[1,0].plot(lf_damaged1_2, label='Phase 1')
axs[1,0].plot(lf_damaged1_3, label='Phase 2')
axs[1,0].legend()

axs[0,1].set_title('normal wires')
axs[0,1].plot(lf_normal2_1, label='Phase 0')
axs[0,1].plot(lf_normal2_2, label='Phase 1')
axs[0,1].plot(lf_normal2_3, label='Phase 2')
axs[0,1].legend()

axs[1,1].set_title('damaged wires')
axs[1,1].plot(lf_damaged2_1, label='Phase 0')
axs[1,1].plot(lf_damaged2_2, label='Phase 1')
axs[1,1].plot(lf_damaged2_3, label='Phase 2')
axs[1,1].legend()


plt.show()


# In[ ]:


fig, axs = plt.subplots(2, 2, constrained_layout=True,figsize=(15, 10))

axs[0,0].set_title('normal wires')
axs[0,0].plot((np.abs(lf_normal1_1)+np.abs(lf_normal1_2)+np.abs(lf_normal1_3)))
axs[0,0].plot(lf_normal1_1, label='Phase 0')
axs[0,0].plot(lf_normal1_2, label='Phase 1')
axs[0,0].plot(lf_normal1_3, label='Phase 2')
axs[0,0].legend()

axs[1,0].set_title('damaged wires')
axs[1,0].plot((np.abs(lf_damaged1_1)+np.abs(lf_damaged1_2)+np.abs(lf_damaged1_3)))
axs[1,0].plot(lf_damaged1_1, label='Phase 0')
axs[1,0].plot(lf_damaged1_2, label='Phase 1')
axs[1,0].plot(lf_damaged1_3, label='Phase 2')
axs[1,0].legend()

axs[0,1].set_title('normal wires')
axs[0,1].plot((np.abs(lf_normal2_1)+np.abs(lf_normal2_2)+np.abs(lf_normal2_3)))
axs[0,1].plot(lf_normal2_1, label='Phase 0')
axs[0,1].plot(lf_normal2_2, label='Phase 1')
axs[0,1].plot(lf_normal2_3, label='Phase 2')
axs[0,1].legend()

axs[1,1].set_title('damaged wires')
axs[1,1].plot((np.abs(lf_damaged2_1)+np.abs(lf_damaged2_2)+np.abs(lf_damaged2_3)))
axs[1,1].plot(lf_damaged2_1, label='Phase 0')
axs[1,1].plot(lf_damaged2_2, label='Phase 1')
axs[1,1].plot(lf_damaged2_3, label='Phase 2')
axs[1,1].legend()


plt.show()


# In[ ]:


###Filter out low frequencies from the signal to get HF characteristics
def high_pass(s, threshold=1e7):
    fourier = rfft(s)
    frequencies = rfftfreq(s.size, d=2e-2/s.size)
    fourier[frequencies < threshold] = 0
    return irfft(fourier)


# In[ ]:


#normal one
hf_normal1_1 = high_pass(signals[0, 0, :])
hf_normal1_2 = high_pass(signals[0, 1, :])
hf_normal1_3 = high_pass(signals[0, 2, :])
#normal two
hf_normal2_1 = high_pass(signals[77, 0, :])
hf_normal2_2 = high_pass(signals[77, 1, :])
hf_normal2_3 = high_pass(signals[77, 2, :])
#damaged one
hf_damaged1_1 = high_pass(signals[1, 0, :])
hf_damaged1_2 = high_pass(signals[1, 1, :])
hf_damaged1_3 = high_pass(signals[1, 2, :])
#damaged two
hf_damaged2_1 = high_pass(signals[76, 0, :])
hf_damaged2_2 = high_pass(signals[76, 1, :])
hf_damaged2_3 = high_pass(signals[76, 2, :])


# In[ ]:


fig, axs = plt.subplots(2, 2, constrained_layout=True,figsize=(15, 10))

axs[0,0].set_title('normal wires')
axs[0,0].plot(hf_normal1_1, label='Phase 0')
# axs[0,0].plot(hf_normal1_2, label='Phase 1')
# axs[0,0].plot(hf_normal1_3, label='Phase 2')
axs[0,0].legend()

axs[1,0].set_title('damaged wires')
axs[1,0].plot(hf_damaged1_1, label='Phase 0')
axs[1,0].plot(hf_damaged1_2, label='Phase 1')
axs[1,0].plot(hf_damaged1_3, label='Phase 2')
axs[1,0].legend()

axs[0,1].set_title('normal wires')
axs[0,1].plot(hf_normal2_1, label='Phase 0')
axs[0,1].plot(hf_normal2_2, label='Phase 1')
axs[0,1].plot(hf_normal2_3, label='Phase 2')
axs[0,1].legend()

axs[1,1].set_title('damaged wires')
axs[1,1].plot(hf_damaged2_1, label='Phase 0')
axs[1,1].plot(hf_damaged2_2, label='Phase 1')
axs[1,1].plot(hf_damaged2_3, label='Phase 2')
axs[1,1].legend()


plt.show()

