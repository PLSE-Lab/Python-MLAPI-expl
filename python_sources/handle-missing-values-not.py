#!/usr/bin/env python
# coding: utf-8

# How about handling missing data by **not handling them**!
# Missing Values are slapped into the complex plane at 0+1i rather than trying to give them a "good" real value.

# In[ ]:





# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


ls ../input


# In[ ]:


df_train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')
df_train.shape


# In[ ]:


df_train.target.mean()


# In[ ]:


df_test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')
df_test.shape


# In[ ]:


df_train.head()


# In[ ]:


for c in df_train.columns[1:-1]:
    print(c,len(df_train[c].unique()),len(df_test[c].unique()))


# In[ ]:


for c in df_train.columns[1:-1]:
    print(c,
          df_train.loc[~df_train[c].isin(df_test[c]),c].shape[0],
          df_test.loc[~df_test[c].isin(df_train[c]),c].shape[0])


# In[ ]:


for c in df_train.columns[1:-1]:
    print(c)
    le = LabelEncoder()
    le.fit(list(df_train.loc[~df_train[c].isnull(),c])+list(df_test.loc[~df_test[c].isnull(),c]))
    df_train.loc[~df_train[c].isnull(),c] = le.transform(df_train.loc[~df_train[c].isnull(),c])
    df_test.loc[~df_test[c].isnull(),c] = le.transform(df_test.loc[~df_test[c].isnull(),c])
    


# In[ ]:


for c in df_train.columns[1:-1]:
    print(c)
    x = df_train.groupby(c).target.agg(['mean','std']).reset_index(drop=False).rename(columns={'mean':'mean_'+c,'std':'std_'+c})
    df_train = df_train.merge(x,on=c,how='left')
    df_test = df_test.merge(x,on=c,how='left')
    del df_train[c]
    del df_test[c]


# In[ ]:


targets = df_train.target.values
del df_train['target']
df_train['target'] = targets


# In[ ]:


df_train[df_train.columns[1:-1]] = df_train[df_train.columns[1:-1]].astype(complex)
df_test[df_train.columns[1:-1]] = df_test[df_train.columns[1:-1]].astype(complex)
df_train = df_train.fillna(complex(0,1))
df_test = df_test.fillna(complex(0,1))


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


def Output(p):
    return 1.0/(1.0+np.exp(-p))

def GPRealI(data):
    return (-1.468275 +
            0.059312*np.tanh(np.real(((((data["mean_ord_2"]) + (np.cos((complex(0,1)*np.conjugate(((((data["std_nom_8"]) * 2.0)) + (((data["std_ord_0"]) / (np.cos((np.cos((complex(0,1)*np.conjugate(((data["std_nom_8"]) * 2.0)))))))))))))))) + (np.conjugate(np.sin((((data["std_ord_3"]) / (((data["mean_ord_3"]) * (np.sqrt((complex(0,1)*np.conjugate(data["mean_ord_2"])))))))))))))) +
            0.086455*np.tanh(np.real(((np.cos((((((((data["mean_nom_8"]) - (data["mean_ord_2"]))) - (data["mean_ord_2"]))) - (data["mean_ord_2"]))))) / ((((-((np.sin((np.conjugate(data["std_bin_1"]))))))) + ((((((data["mean_ord_5"]) + (np.sin((data["mean_nom_8"]))))) + (((data["mean_ord_3"]) + (data["mean_ord_2"]))))/2.0))))))) +
            0.082338*np.tanh(np.real(((np.cos((data["mean_ord_3"]))) / (np.conjugate((((np.tanh((((np.conjugate((((data["mean_ord_0"]) + ((((data["mean_ord_3"]) + (data["mean_nom_9"]))/2.0)))/2.0))) * 2.0)))) + (((data["std_nom_7"]) - (np.tanh((np.cos((((((((data["mean_ord_3"]) + (data["mean_nom_9"]))/2.0)) + (np.conjugate(data["mean_ord_3"])))/2.0)))))))))/2.0)))))) +
            0.000043*np.tanh(np.real(((np.cos((np.sqrt((data["std_month"]))))) / (((data["mean_ord_3"]) + (np.sin((((data["mean_ord_5"]) - ((-((((data["std_ord_2"]) - (np.sqrt(((-((((data["std_month"]) - (np.sqrt((np.cos((data["mean_ord_5"])))))))))))))))))))))))))) +
            0.069560*np.tanh(np.real(((data["mean_ord_3"]) + (((np.sin((((data["mean_ord_3"]) + (((((data["mean_ord_5"]) / 2.0)) + ((((data["mean_month"]) + (np.conjugate((-((((data["std_ord_3"]) * 2.0)))))))/2.0)))))))) / (((data["mean_ord_5"]) * (((data["mean_ord_3"]) * (data["mean_ord_3"])))))))))) +
            0.099862*np.tanh(np.real(((((data["std_ord_4"]) / ((((((data["std_ord_4"]) + (((((((data["mean_month"]) + (data["mean_ord_2"]))) * (((((data["mean_ord_5"]) * 2.0)) + (data["mean_ord_3"]))))) + (np.conjugate(((data["mean_nom_7"]) * (((data["std_nom_9"]) + (((data["std_ord_0"]) + (data["mean_nom_5"])))))))))))/2.0)) - (data["std_bin_4"]))))) * 2.0))) +
            0.086979*np.tanh(np.real(((((((((data["mean_ord_5"]) + (np.sin((np.sin((np.conjugate(((complex(0.636620)) / (((((((data["mean_ord_3"]) + (data["mean_ord_2"]))/2.0)) + (np.sin((np.sin((data["mean_nom_9"]))))))/2.0))))))))))) * 2.0)) * 2.0)) * 2.0))) +
            0.085667*np.tanh(np.real(((((np.cos((data["std_nom_3"]))) / 2.0)) / ((((((data["std_nom_3"]) + ((((data["std_nom_8"]) + ((((complex(-1.0)) + (data["mean_ord_4"]))/2.0)))/2.0)))) + ((((complex(-1.0)) + ((((((data["mean_day"]) + (((data["mean_nom_8"]) * (((np.cos((data["std_nom_3"]))) / 2.0)))))) + (data["mean_ord_3"]))/2.0)))/2.0)))/2.0))))) +
            0.075546*np.tanh(np.real(((((((data["mean_nom_6"]) + (np.sin((np.sin((data["mean_ord_3"]))))))) * 2.0)) / (((data["std_nom_8"]) - (((np.cos((((((((((data["std_ord_0"]) - (((np.cos((np.sqrt((((data["mean_ord_3"]) * 2.0)))))) / 2.0)))) + (np.sin((np.sin((data["mean_nom_9"]))))))) * 2.0)) * 2.0)))) / 2.0))))))) +
            0.002114*np.tanh(np.real(((((complex(13.09953975677490234)) - (np.cos((((data["mean_ord_5"]) / (data["mean_ord_5"]))))))) / (np.cos((((data["std_ord_2"]) * (((((data["std_ord_4"]) + (data["mean_ord_5"]))) - (((((((np.cos((((complex(3.0)) / (((data["std_ord_3"]) + (data["mean_ord_5"]))))))) * 2.0)) * 2.0)) - (complex(13.09953975677490234))))))))))))) +
            0.066201*np.tanh(np.real(((data["mean_ord_3"]) / ((((((data["mean_ord_2"]) + (((data["mean_ord_5"]) - (np.tanh((np.conjugate(np.cos((((np.sqrt((((data["std_ord_3"]) / (np.cos((data["std_ord_3"]))))))) - (((data["mean_ord_4"]) / ((-((np.tanh((np.cos((((complex(10.0)) * 2.0)))))))))))))))))))))/2.0)) * (data["std_ord_3"])))))) +
            0.097861*np.tanh(np.real(((((np.cos((((complex(1.0)) / (((((data["mean_month"]) / (np.cos((((((((data["mean_ord_0"]) / (np.cos((np.sin((((complex(1.0)) / (((data["mean_ord_3"]) + (((data["mean_ord_2"]) + (data["mean_ord_3"]))))))))))))) + (data["mean_ord_2"]))) * 2.0)))))) * 2.0)))))) / (data["mean_ord_2"]))) * 2.0))) +
            0.100000*np.tanh(np.real(((complex(3.0)) / (np.cos((((data["std_bin_3"]) / (((((((data["mean_nom_5"]) + ((((data["mean_ord_1"]) + (((data["mean_ord_5"]) + (data["mean_nom_7"]))))/2.0)))/2.0)) + ((((data["mean_nom_8"]) + ((((data["std_ord_1"]) + ((((data["mean_ord_0"]) + (((data["mean_ord_3"]) + (data["mean_ord_2"]))))/2.0)))/2.0)))/2.0)))/2.0))))))))) +
            0.099775*np.tanh(np.real(((((np.sqrt((np.sqrt((np.sin((np.sqrt((np.sin((data["mean_nom_7"]))))))))))) / (((data["mean_nom_7"]) + (((np.sqrt((data["mean_nom_9"]))) - (np.sin((np.cos((((np.sqrt((data["mean_month"]))) + (data["std_ord_2"]))))))))))))) + (((data["mean_month"]) * 2.0))))) +
            0.099586*np.tanh(np.real(((np.sin((((complex(0.636620)) / (data["mean_ord_4"]))))) + (((((((complex(3.141593)) / (data["mean_bin_2"]))) + (data["mean_ord_3"]))) + (((complex(3.141593)) / (((((((data["mean_ord_5"]) + (data["mean_ord_3"]))) * (data["mean_ord_0"]))) / (np.sin((((complex(3.141593)) / (data["mean_bin_2"])))))))))))))) +
            0.099891*np.tanh(np.real(((((complex(3.141593)) / (data["std_bin_3"]))) / (((data["std_ord_1"]) - (((data["std_bin_3"]) - (((data["mean_nom_1"]) - (((data["std_bin_3"]) - (((((((data["mean_ord_5"]) + (data["mean_nom_8"]))/2.0)) + (data["mean_nom_7"]))/2.0))))))))))))) +
            0.099160*np.tanh(np.real(((np.cos((((((data["mean_month"]) * 2.0)) - (((data["mean_ord_5"]) * 2.0)))))) / (((data["mean_ord_2"]) + (((data["mean_month"]) - (np.sqrt((((np.tanh((np.cos((((data["mean_ord_5"]) * 2.0)))))) - (((data["mean_day"]) + (data["std_bin_0"])))))))))))))) +
            0.099064*np.tanh(np.real(((((data["std_nom_9"]) + (((np.conjugate(complex(0,1)*np.conjugate(np.tanh((((np.sqrt((((data["mean_nom_8"]) + (((data["std_nom_5"]) - (np.cos((((data["mean_nom_9"]) + (data["mean_nom_2"]))))))))))) / (np.tanh(((((data["std_ord_4"]) + ((((data["std_ord_3"]) + (data["std_nom_9"]))/2.0)))/2.0)))))))))) / 2.0)))) * 2.0))) +
            0.099880*np.tanh(np.real(((np.sqrt((np.sqrt((data["mean_ord_0"]))))) / (np.sin(((((np.sin(((((data["mean_ord_0"]) + (data["mean_ord_4"]))/2.0)))) + (((complex(-1.0)) + (((data["std_nom_1"]) + (np.sqrt((np.sin((((((((data["mean_month"]) + ((((data["mean_ord_3"]) + (data["mean_ord_2"]))/2.0)))/2.0)) + (data["mean_bin_0"]))/2.0)))))))))))/2.0))))))) +
            0.090643*np.tanh(np.real(((((data["mean_nom_7"]) + ((((((data["mean_ord_3"]) + (((data["mean_nom_1"]) + (data["mean_nom_3"]))))/2.0)) - (np.cos((((np.sqrt(((((data["mean_nom_9"]) + (((data["mean_nom_4"]) + (data["mean_ord_5"]))))/2.0)))) * 2.0)))))))) / (((data["mean_nom_7"]) * ((((data["mean_nom_1"]) + (data["mean_nom_1"]))/2.0))))))) +
            0.100000*np.tanh(np.real(((((data["mean_ord_2"]) * 2.0)) + (((((data["mean_nom_8"]) + (((((((data["mean_nom_9"]) - (np.cos((((data["mean_ord_1"]) / ((((data["mean_nom_4"]) + (complex(0,1)*np.conjugate(((((((complex(1.0)) / 2.0)) - (data["mean_bin_2"]))) - (data["mean_nom_5"])))))/2.0)))))))) * 2.0)) * 2.0)))) * 2.0))))) +
            0.099488*np.tanh(np.real(((((data["std_nom_6"]) + (np.cos((data["mean_bin_0"]))))) / (((data["mean_ord_5"]) - (((((((((np.cos((((data["std_nom_6"]) + (np.cos((np.cos((((data["std_bin_0"]) + (np.cos((data["mean_bin_0"]))))))))))))) * 2.0)) - (data["mean_nom_3"]))) * 2.0)) - (data["mean_ord_2"])))))))) +
            0.099780*np.tanh(np.real(((np.sin((((((complex(0.636620)) / (data["mean_month"]))) - (((data["mean_ord_0"]) + (data["mean_ord_4"]))))))) - (np.cos((((((((((complex(0.636620)) / (data["mean_nom_2"]))) - (data["mean_ord_3"]))) - (data["mean_ord_4"]))) / (((data["mean_ord_0"]) + (((data["std_nom_5"]) + (complex(0,1)*np.conjugate(data["mean_ord_0"]))))))))))))) +
            0.099990*np.tanh(np.real(np.conjugate(((((data["std_ord_3"]) + ((-((((np.cos((np.cos((data["std_day"]))))) / (np.cos((((((np.cos((data["std_day"]))) - (((data["std_ord_4"]) * ((-((((np.tanh((((data["mean_day"]) * 2.0)))) * 2.0))))))))) + (np.tanh((((data["mean_nom_4"]) * 2.0))))))))))))))) * 2.0)))) +
            0.099379*np.tanh(np.real(((np.sin((((((complex(1.0)) / (((data["mean_nom_8"]) * 2.0)))) + (data["mean_ord_3"]))))) + (((((np.sin((((complex(1.0)) / (((data["mean_month"]) * 2.0)))))) + (np.sin((((complex(-1.0)) / (data["mean_bin_0"]))))))) - (((((data["std_bin_2"]) / (np.tanh((data["mean_ord_3"]))))) / 2.0))))))) +
            0.088375*np.tanh(np.real(((((((np.cos((((complex(5.95223093032836914)) / ((-(((((data["mean_nom_2"]) + (data["std_bin_2"]))/2.0))))))))) + (((((((np.sin((data["mean_nom_7"]))) + ((((data["std_nom_9"]) + (((data["mean_ord_2"]) - (np.cos((data["std_nom_9"]))))))/2.0)))) * 2.0)) * (complex(5.95223093032836914)))))) * 2.0)) * 2.0))) +
            0.092000*np.tanh(np.real(((data["mean_ord_5"]) - ((((((np.sqrt((((data["mean_nom_3"]) * 2.0)))) + (data["std_nom_6"]))/2.0)) / (np.cos((((((((((((data["std_nom_6"]) + ((((((data["mean_nom_3"]) * 2.0)) + (np.sqrt((data["mean_ord_5"]))))/2.0)))/2.0)) + ((((data["std_nom_1"]) + (data["std_nom_8"]))/2.0)))/2.0)) * 2.0)) * 2.0))))))))) +
            0.098638*np.tanh(np.real((((-((np.cos(((((((complex(0.0)) + (((complex(3.141593)) / (((complex(0.318310)) - (((data["std_day"]) + (data["mean_ord_1"]))))))))/2.0)) + (((data["mean_ord_3"]) + (((((data["mean_ord_4"]) + (((np.sin((data["mean_nom_9"]))) + (np.conjugate(data["mean_ord_2"])))))) * 2.0))))))))))) * 2.0))) +
            0.057095*np.tanh(np.real(((((data["mean_ord_3"]) * (complex(3.141593)))) - ((((complex(1.0)) + ((((((((complex(0,1)*np.conjugate(data["mean_ord_3"])) + (((data["std_ord_4"]) / 2.0)))) / (((np.tanh((np.tanh((((((data["std_ord_4"]) / 2.0)) - (data["mean_bin_2"]))))))) * 2.0)))) + ((-((((data["mean_ord_3"]) * (complex(3.141593))))))))/2.0)))/2.0))))))

def GPComplexI(data):
    return (0.059312*np.tanh(np.imag(((((data["mean_ord_2"]) + (np.cos((complex(0,1)*np.conjugate(((((data["std_nom_8"]) * 2.0)) + (((data["std_ord_0"]) / (np.cos((np.cos((complex(0,1)*np.conjugate(((data["std_nom_8"]) * 2.0)))))))))))))))) + (np.conjugate(np.sin((((data["std_ord_3"]) / (((data["mean_ord_3"]) * (np.sqrt((complex(0,1)*np.conjugate(data["mean_ord_2"])))))))))))))) +
            0.086455*np.tanh(np.imag(((np.cos((((((((data["mean_nom_8"]) - (data["mean_ord_2"]))) - (data["mean_ord_2"]))) - (data["mean_ord_2"]))))) / ((((-((np.sin((np.conjugate(data["std_bin_1"]))))))) + ((((((data["mean_ord_5"]) + (np.sin((data["mean_nom_8"]))))) + (((data["mean_ord_3"]) + (data["mean_ord_2"]))))/2.0))))))) +
            0.082338*np.tanh(np.imag(((np.cos((data["mean_ord_3"]))) / (np.conjugate((((np.tanh((((np.conjugate((((data["mean_ord_0"]) + ((((data["mean_ord_3"]) + (data["mean_nom_9"]))/2.0)))/2.0))) * 2.0)))) + (((data["std_nom_7"]) - (np.tanh((np.cos((((((((data["mean_ord_3"]) + (data["mean_nom_9"]))/2.0)) + (np.conjugate(data["mean_ord_3"])))/2.0)))))))))/2.0)))))) +
            0.000043*np.tanh(np.imag(((np.cos((np.sqrt((data["std_month"]))))) / (((data["mean_ord_3"]) + (np.sin((((data["mean_ord_5"]) - ((-((((data["std_ord_2"]) - (np.sqrt(((-((((data["std_month"]) - (np.sqrt((np.cos((data["mean_ord_5"])))))))))))))))))))))))))) +
            0.069560*np.tanh(np.imag(((data["mean_ord_3"]) + (((np.sin((((data["mean_ord_3"]) + (((((data["mean_ord_5"]) / 2.0)) + ((((data["mean_month"]) + (np.conjugate((-((((data["std_ord_3"]) * 2.0)))))))/2.0)))))))) / (((data["mean_ord_5"]) * (((data["mean_ord_3"]) * (data["mean_ord_3"])))))))))) +
            0.099862*np.tanh(np.imag(((((data["std_ord_4"]) / ((((((data["std_ord_4"]) + (((((((data["mean_month"]) + (data["mean_ord_2"]))) * (((((data["mean_ord_5"]) * 2.0)) + (data["mean_ord_3"]))))) + (np.conjugate(((data["mean_nom_7"]) * (((data["std_nom_9"]) + (((data["std_ord_0"]) + (data["mean_nom_5"])))))))))))/2.0)) - (data["std_bin_4"]))))) * 2.0))) +
            0.086979*np.tanh(np.imag(((((((((data["mean_ord_5"]) + (np.sin((np.sin((np.conjugate(((complex(0.636620)) / (((((((data["mean_ord_3"]) + (data["mean_ord_2"]))/2.0)) + (np.sin((np.sin((data["mean_nom_9"]))))))/2.0))))))))))) * 2.0)) * 2.0)) * 2.0))) +
            0.085667*np.tanh(np.imag(((((np.cos((data["std_nom_3"]))) / 2.0)) / ((((((data["std_nom_3"]) + ((((data["std_nom_8"]) + ((((complex(-1.0)) + (data["mean_ord_4"]))/2.0)))/2.0)))) + ((((complex(-1.0)) + ((((((data["mean_day"]) + (((data["mean_nom_8"]) * (((np.cos((data["std_nom_3"]))) / 2.0)))))) + (data["mean_ord_3"]))/2.0)))/2.0)))/2.0))))) +
            0.075546*np.tanh(np.imag(((((((data["mean_nom_6"]) + (np.sin((np.sin((data["mean_ord_3"]))))))) * 2.0)) / (((data["std_nom_8"]) - (((np.cos((((((((((data["std_ord_0"]) - (((np.cos((np.sqrt((((data["mean_ord_3"]) * 2.0)))))) / 2.0)))) + (np.sin((np.sin((data["mean_nom_9"]))))))) * 2.0)) * 2.0)))) / 2.0))))))) +
            0.002114*np.tanh(np.imag(((((complex(13.09953975677490234)) - (np.cos((((data["mean_ord_5"]) / (data["mean_ord_5"]))))))) / (np.cos((((data["std_ord_2"]) * (((((data["std_ord_4"]) + (data["mean_ord_5"]))) - (((((((np.cos((((complex(3.0)) / (((data["std_ord_3"]) + (data["mean_ord_5"]))))))) * 2.0)) * 2.0)) - (complex(13.09953975677490234))))))))))))) +
            0.066201*np.tanh(np.imag(((data["mean_ord_3"]) / ((((((data["mean_ord_2"]) + (((data["mean_ord_5"]) - (np.tanh((np.conjugate(np.cos((((np.sqrt((((data["std_ord_3"]) / (np.cos((data["std_ord_3"]))))))) - (((data["mean_ord_4"]) / ((-((np.tanh((np.cos((((complex(10.0)) * 2.0)))))))))))))))))))))/2.0)) * (data["std_ord_3"])))))) +
            0.097861*np.tanh(np.imag(((((np.cos((((complex(1.0)) / (((((data["mean_month"]) / (np.cos((((((((data["mean_ord_0"]) / (np.cos((np.sin((((complex(1.0)) / (((data["mean_ord_3"]) + (((data["mean_ord_2"]) + (data["mean_ord_3"]))))))))))))) + (data["mean_ord_2"]))) * 2.0)))))) * 2.0)))))) / (data["mean_ord_2"]))) * 2.0))) +
            0.100000*np.tanh(np.imag(((complex(3.0)) / (np.cos((((data["std_bin_3"]) / (((((((data["mean_nom_5"]) + ((((data["mean_ord_1"]) + (((data["mean_ord_5"]) + (data["mean_nom_7"]))))/2.0)))/2.0)) + ((((data["mean_nom_8"]) + ((((data["std_ord_1"]) + ((((data["mean_ord_0"]) + (((data["mean_ord_3"]) + (data["mean_ord_2"]))))/2.0)))/2.0)))/2.0)))/2.0))))))))) +
            0.099775*np.tanh(np.imag(((((np.sqrt((np.sqrt((np.sin((np.sqrt((np.sin((data["mean_nom_7"]))))))))))) / (((data["mean_nom_7"]) + (((np.sqrt((data["mean_nom_9"]))) - (np.sin((np.cos((((np.sqrt((data["mean_month"]))) + (data["std_ord_2"]))))))))))))) + (((data["mean_month"]) * 2.0))))) +
            0.099586*np.tanh(np.imag(((np.sin((((complex(0.636620)) / (data["mean_ord_4"]))))) + (((((((complex(3.141593)) / (data["mean_bin_2"]))) + (data["mean_ord_3"]))) + (((complex(3.141593)) / (((((((data["mean_ord_5"]) + (data["mean_ord_3"]))) * (data["mean_ord_0"]))) / (np.sin((((complex(3.141593)) / (data["mean_bin_2"])))))))))))))) +
            0.099891*np.tanh(np.imag(((((complex(3.141593)) / (data["std_bin_3"]))) / (((data["std_ord_1"]) - (((data["std_bin_3"]) - (((data["mean_nom_1"]) - (((data["std_bin_3"]) - (((((((data["mean_ord_5"]) + (data["mean_nom_8"]))/2.0)) + (data["mean_nom_7"]))/2.0))))))))))))) +
            0.099160*np.tanh(np.imag(((np.cos((((((data["mean_month"]) * 2.0)) - (((data["mean_ord_5"]) * 2.0)))))) / (((data["mean_ord_2"]) + (((data["mean_month"]) - (np.sqrt((((np.tanh((np.cos((((data["mean_ord_5"]) * 2.0)))))) - (((data["mean_day"]) + (data["std_bin_0"])))))))))))))) +
            0.099064*np.tanh(np.imag(((((data["std_nom_9"]) + (((np.conjugate(complex(0,1)*np.conjugate(np.tanh((((np.sqrt((((data["mean_nom_8"]) + (((data["std_nom_5"]) - (np.cos((((data["mean_nom_9"]) + (data["mean_nom_2"]))))))))))) / (np.tanh(((((data["std_ord_4"]) + ((((data["std_ord_3"]) + (data["std_nom_9"]))/2.0)))/2.0)))))))))) / 2.0)))) * 2.0))) +
            0.099880*np.tanh(np.imag(((np.sqrt((np.sqrt((data["mean_ord_0"]))))) / (np.sin(((((np.sin(((((data["mean_ord_0"]) + (data["mean_ord_4"]))/2.0)))) + (((complex(-1.0)) + (((data["std_nom_1"]) + (np.sqrt((np.sin((((((((data["mean_month"]) + ((((data["mean_ord_3"]) + (data["mean_ord_2"]))/2.0)))/2.0)) + (data["mean_bin_0"]))/2.0)))))))))))/2.0))))))) +
            0.090643*np.tanh(np.imag(((((data["mean_nom_7"]) + ((((((data["mean_ord_3"]) + (((data["mean_nom_1"]) + (data["mean_nom_3"]))))/2.0)) - (np.cos((((np.sqrt(((((data["mean_nom_9"]) + (((data["mean_nom_4"]) + (data["mean_ord_5"]))))/2.0)))) * 2.0)))))))) / (((data["mean_nom_7"]) * ((((data["mean_nom_1"]) + (data["mean_nom_1"]))/2.0))))))) +
            0.100000*np.tanh(np.imag(((((data["mean_ord_2"]) * 2.0)) + (((((data["mean_nom_8"]) + (((((((data["mean_nom_9"]) - (np.cos((((data["mean_ord_1"]) / ((((data["mean_nom_4"]) + (complex(0,1)*np.conjugate(((((((complex(1.0)) / 2.0)) - (data["mean_bin_2"]))) - (data["mean_nom_5"])))))/2.0)))))))) * 2.0)) * 2.0)))) * 2.0))))) +
            0.099488*np.tanh(np.imag(((((data["std_nom_6"]) + (np.cos((data["mean_bin_0"]))))) / (((data["mean_ord_5"]) - (((((((((np.cos((((data["std_nom_6"]) + (np.cos((np.cos((((data["std_bin_0"]) + (np.cos((data["mean_bin_0"]))))))))))))) * 2.0)) - (data["mean_nom_3"]))) * 2.0)) - (data["mean_ord_2"])))))))) +
            0.099780*np.tanh(np.imag(((np.sin((((((complex(0.636620)) / (data["mean_month"]))) - (((data["mean_ord_0"]) + (data["mean_ord_4"]))))))) - (np.cos((((((((((complex(0.636620)) / (data["mean_nom_2"]))) - (data["mean_ord_3"]))) - (data["mean_ord_4"]))) / (((data["mean_ord_0"]) + (((data["std_nom_5"]) + (complex(0,1)*np.conjugate(data["mean_ord_0"]))))))))))))) +
            0.099990*np.tanh(np.imag(np.conjugate(((((data["std_ord_3"]) + ((-((((np.cos((np.cos((data["std_day"]))))) / (np.cos((((((np.cos((data["std_day"]))) - (((data["std_ord_4"]) * ((-((((np.tanh((((data["mean_day"]) * 2.0)))) * 2.0))))))))) + (np.tanh((((data["mean_nom_4"]) * 2.0))))))))))))))) * 2.0)))) +
            0.099379*np.tanh(np.imag(((np.sin((((((complex(1.0)) / (((data["mean_nom_8"]) * 2.0)))) + (data["mean_ord_3"]))))) + (((((np.sin((((complex(1.0)) / (((data["mean_month"]) * 2.0)))))) + (np.sin((((complex(-1.0)) / (data["mean_bin_0"]))))))) - (((((data["std_bin_2"]) / (np.tanh((data["mean_ord_3"]))))) / 2.0))))))) +
            0.088375*np.tanh(np.imag(((((((np.cos((((complex(5.95223093032836914)) / ((-(((((data["mean_nom_2"]) + (data["std_bin_2"]))/2.0))))))))) + (((((((np.sin((data["mean_nom_7"]))) + ((((data["std_nom_9"]) + (((data["mean_ord_2"]) - (np.cos((data["std_nom_9"]))))))/2.0)))) * 2.0)) * (complex(5.95223093032836914)))))) * 2.0)) * 2.0))) +
            0.092000*np.tanh(np.imag(((data["mean_ord_5"]) - ((((((np.sqrt((((data["mean_nom_3"]) * 2.0)))) + (data["std_nom_6"]))/2.0)) / (np.cos((((((((((((data["std_nom_6"]) + ((((((data["mean_nom_3"]) * 2.0)) + (np.sqrt((data["mean_ord_5"]))))/2.0)))/2.0)) + ((((data["std_nom_1"]) + (data["std_nom_8"]))/2.0)))/2.0)) * 2.0)) * 2.0))))))))) +
            0.098638*np.tanh(np.imag((((-((np.cos(((((((complex(0.0)) + (((complex(3.141593)) / (((complex(0.318310)) - (((data["std_day"]) + (data["mean_ord_1"]))))))))/2.0)) + (((data["mean_ord_3"]) + (((((data["mean_ord_4"]) + (((np.sin((data["mean_nom_9"]))) + (np.conjugate(data["mean_ord_2"])))))) * 2.0))))))))))) * 2.0))) +
            0.057095*np.tanh(np.imag(((((data["mean_ord_3"]) * (complex(3.141593)))) - ((((complex(1.0)) + ((((((((complex(0,1)*np.conjugate(data["mean_ord_3"])) + (((data["std_ord_4"]) / 2.0)))) / (((np.tanh((np.tanh((((((data["std_ord_4"]) / 2.0)) - (data["mean_bin_2"]))))))) * 2.0)))) + ((-((((data["mean_ord_3"]) * (complex(3.141593))))))))/2.0)))/2.0))))))

def GPRealII(data):
    return (-1.468275 +
            0.089800*np.tanh(np.real(((((((((data["std_nom_7"]) + (np.sin((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(((complex(10.0)) * ((-((((data["std_nom_7"]) + (((((data["mean_ord_3"]) + (np.sqrt((data["std_ord_5"]))))) - (np.cos((np.conjugate(data["std_ord_1"]))))))))))))))))))) * 2.0)) * 2.0)) / (data["std_ord_3"])))) +
            0.092783*np.tanh(np.real(np.sin((((np.sqrt((data["std_ord_3"]))) + (((complex(11.97040462493896484)) / (((data["std_ord_2"]) + ((((((np.conjugate(np.sin((((np.sqrt((data["std_ord_0"]))) / 2.0))))) + (complex(0,1)*np.conjugate(((data["mean_ord_3"]) * 2.0))))/2.0)) + (((data["std_month"]) / 2.0))))))))))))) +
            0.083000*np.tanh(np.real(((((data["mean_nom_9"]) - (np.cos((((((data["mean_ord_2"]) + (np.tanh((((data["mean_ord_3"]) - (((((complex(0,1)*np.conjugate(data["std_ord_5"])) / 2.0)) / (((data["std_bin_1"]) / 2.0)))))))))) * 2.0)))))) + (np.sin((((((((data["std_bin_1"]) / 2.0)) / (((data["mean_ord_2"]) * (data["mean_nom_8"]))))) / 2.0))))))) +
            0.099855*np.tanh(np.real(((((data["std_bin_1"]) - (complex(-1.0)))) / (np.sin((((data["mean_nom_7"]) - (((((((data["std_bin_1"]) - ((((data["mean_nom_8"]) + (((data["mean_ord_5"]) - (data["std_bin_1"]))))/2.0)))) - ((((data["mean_nom_8"]) + (((data["mean_ord_3"]) - (data["std_bin_1"]))))/2.0)))) - (data["mean_ord_2"])))))))))) +
            0.098275*np.tanh(np.real(((np.cos((((((((data["mean_nom_9"]) / 2.0)) - (data["mean_ord_3"]))) - (((data["mean_nom_9"]) - (((np.conjugate(data["mean_nom_9"])) - (((data["mean_ord_3"]) - (data["mean_month"]))))))))))) / (((np.conjugate(data["mean_nom_9"])) - (((data["std_bin_3"]) - ((((data["mean_month"]) + (data["mean_ord_3"]))/2.0))))))))) +
            0.097944*np.tanh(np.real(((((((np.cos((((np.tanh((data["mean_ord_5"]))) + (((((data["mean_ord_0"]) - (data["mean_ord_3"]))) - (data["mean_ord_3"]))))))) * 2.0)) / (((data["mean_ord_0"]) - ((((((np.tanh((data["std_bin_3"]))) - (data["mean_ord_3"]))) + (((data["std_bin_3"]) - (data["mean_ord_5"]))))/2.0)))))) + (((data["mean_ord_3"]) * 2.0))))) +
            0.097081*np.tanh(np.real(((np.sqrt((np.sqrt((data["std_ord_4"]))))) / (np.tanh((np.tanh(((((((data["mean_nom_7"]) + (np.sqrt((data["std_nom_8"]))))/2.0)) + ((((-((np.cos((np.sqrt(((((data["std_ord_4"]) + (np.sin((((data["mean_ord_3"]) * 2.0)))))/2.0))))))))) / 2.0))))))))))) +
            0.099980*np.tanh(np.real(((((data["mean_ord_2"]) / ((((((data["mean_month"]) - (((np.tanh((data["mean_ord_2"]))) / (((data["mean_ord_0"]) - (np.cos((np.sqrt((data["mean_nom_9"]))))))))))) + (complex(0,1)*np.conjugate((-((np.sin((complex(0,1)*np.conjugate(np.cos((((complex(13.24914264678955078)) + (np.sqrt((data["mean_nom_9"])))))))))))))))/2.0)))) * 2.0))) +
            0.100000*np.tanh(np.real(((((((np.sin((((data["std_bin_1"]) / ((((data["mean_nom_8"]) + (((data["mean_ord_5"]) / 2.0)))/2.0)))))) * 2.0)) * 2.0)) - ((-((np.sin((((data["std_bin_1"]) / ((((data["mean_ord_2"]) + (((complex(0,1)*np.conjugate(data["mean_ord_0"])) / 2.0)))/2.0)))))))))))) +
            0.099956*np.tanh(np.real(((((((np.sin((((complex(9.0)) * ((((data["mean_ord_3"]) + (((((((data["mean_ord_4"]) + (data["mean_ord_5"]))) + (data["mean_month"]))) * 2.0)))/2.0)))))) / (data["mean_ord_4"]))) - (np.cos((complex(9.0)))))) / ((((data["mean_ord_4"]) + (complex(0,1)*np.conjugate(data["mean_ord_4"])))/2.0))))) +
            0.098886*np.tanh(np.real(((((complex(3.14884257316589355)) / (np.cos((((((((-((data["mean_ord_3"])))) + (((np.cos((np.sqrt((np.sin((((((data["mean_nom_9"]) * 2.0)) * 2.0)))))))) / (data["std_ord_0"]))))/2.0)) * 2.0)))))) / ((((np.cos((np.sqrt((data["mean_ord_3"]))))) + (data["mean_nom_9"]))/2.0))))) +
            0.099988*np.tanh(np.real((((((((-((((((data["std_ord_1"]) + (data["mean_nom_8"]))) - (((((data["mean_month"]) + (complex(5.0)))) / 2.0))))))) + (data["mean_month"]))) / ((-((np.sin((((complex(0.636620)) - (np.tanh((((data["mean_month"]) + (((data["std_ord_1"]) + (data["mean_nom_8"])))))))))))))))) * 2.0))) +
            0.100000*np.tanh(np.real(np.sin((((complex(1.570796)) / ((((((((((data["mean_nom_7"]) + (data["mean_nom_5"]))/2.0)) + (data["mean_nom_1"]))/2.0)) + ((((data["mean_day"]) + (((((np.sin((((data["mean_ord_5"]) - (((((data["mean_nom_1"]) - (complex(0,1)*np.conjugate(np.sqrt((np.cos((data["std_ord_5"])))))))) / 2.0)))))) * (data["mean_ord_4"]))) * 2.0)))/2.0)))/2.0))))))) +
            0.099788*np.tanh(np.real(((((data["std_nom_7"]) * 2.0)) / (((data["std_nom_1"]) - ((((np.cos((data["std_nom_7"]))) + ((-((((data["mean_ord_2"]) * (((np.sqrt(((((((((data["mean_month"]) * (((data["std_nom_7"]) * 2.0)))) + (data["mean_month"]))/2.0)) * (((np.conjugate(data["std_ord_3"])) * 2.0)))))) * 2.0))))))))/2.0))))))) +
            0.100000*np.tanh(np.real(((((np.sqrt((data["std_bin_1"]))) * 2.0)) / (((((((((((((data["mean_month"]) * (data["mean_ord_3"]))) + (((data["mean_nom_3"]) + (data["mean_ord_4"]))))/2.0)) + (((data["mean_bin_0"]) + (np.conjugate(((data["mean_bin_2"]) + (((data["mean_nom_3"]) - (data["std_bin_1"])))))))))/2.0)) * 2.0)) + ((-((data["std_bin_1"]))))))))) +
            0.099954*np.tanh(np.real(((data["mean_nom_8"]) - (np.cos((((complex(0.636620)) / ((((((((data["mean_ord_0"]) * (((((data["mean_ord_5"]) * 2.0)) + (((data["mean_ord_5"]) + (data["mean_ord_5"]))))))) + ((((((((data["mean_nom_9"]) + (data["mean_nom_8"]))) - (np.sqrt(((-((data["mean_nom_7"])))))))) + (data["mean_ord_1"]))/2.0)))/2.0)) / 2.0))))))))) +
            0.099881*np.tanh(np.real(((((data["std_nom_9"]) / 2.0)) / ((((((((np.tanh((data["mean_ord_1"]))) + (np.tanh((((data["std_ord_4"]) * (((np.cos((data["std_nom_9"]))) * 2.0)))))))/2.0)) - (np.sin((np.sin((((complex(1.0)) - (((data["std_nom_9"]) + (data["mean_nom_5"]))))))))))) / 2.0))))) +
            0.099920*np.tanh(np.real(((((np.sin((((complex(9.66714859008789062)) / (np.sqrt(((((data["mean_nom_1"]) + (((data["std_bin_0"]) / 2.0)))/2.0)))))))) * 2.0)) + (((np.sin((((complex(9.66714859008789062)) / (np.sqrt(((((((data["mean_day"]) + (data["std_nom_2"]))) + (data["mean_bin_2"]))/2.0)))))))) / (data["mean_bin_2"])))))) +
            0.097186*np.tanh(np.real(np.tanh((((np.tanh((data["mean_ord_2"]))) + (np.sin((np.sin((((data["mean_nom_6"]) + (((((complex(0,1)*np.conjugate(np.conjugate((-((np.cos((data["mean_nom_8"])))))))) + (((((data["mean_ord_3"]) * (data["mean_nom_8"]))) + (data["mean_ord_1"]))))) - (((np.cos((data["mean_nom_8"]))) / 2.0))))))))))))))) +
            0.099973*np.tanh(np.real(((data["mean_nom_5"]) / ((((((data["std_ord_3"]) * ((((data["mean_ord_0"]) + (data["mean_ord_5"]))/2.0)))) + (((data["mean_nom_5"]) + (((np.conjugate((((np.tanh((data["mean_bin_0"]))) + (((((data["mean_month"]) / 2.0)) + (data["mean_nom_8"]))))/2.0))) - ((-((((complex(-1.0)) / 2.0))))))))))/2.0))))) +
            0.099094*np.tanh(np.real(((((complex(0.318310)) + (data["mean_ord_5"]))) / (((((data["std_month"]) + (data["std_day"]))) * (((data["mean_nom_7"]) + (((((((data["mean_bin_0"]) - (complex(0.318310)))) / (((data["mean_ord_5"]) + (np.tanh((data["mean_month"]))))))) / 2.0))))))))) +
            0.099629*np.tanh(np.real((((((complex(4.0)) + (((data["mean_nom_4"]) / (((np.conjugate(data["std_ord_0"])) - (np.conjugate(((data["mean_nom_4"]) * 2.0))))))))/2.0)) - (((((data["mean_bin_2"]) * 2.0)) / (((np.conjugate(data["std_ord_0"])) - (((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(data["mean_bin_2"]))) * 2.0))))))))) +
            0.099769*np.tanh(np.real(((np.conjugate(data["mean_nom_9"])) + (((((data["mean_ord_4"]) / ((((((data["mean_ord_4"]) + (data["mean_bin_1"]))/2.0)) + ((((((((data["std_nom_6"]) + (data["std_nom_3"]))/2.0)) * 2.0)) - (np.cos((data["mean_ord_2"]))))))))) - (np.cos((((np.cos((data["std_month"]))) / (data["mean_ord_4"])))))))))) +
            0.100000*np.tanh(np.real(((np.cos((((data["mean_nom_2"]) + (data["std_ord_5"]))))) / (((((np.conjugate(((((data["mean_nom_5"]) + (np.tanh((((data["mean_ord_1"]) + (data["mean_nom_9"]))))))) * (((data["mean_nom_2"]) + (data["mean_bin_0"])))))) * 2.0)) + (np.conjugate(((data["std_day"]) - (np.cos((np.sqrt((data["std_ord_5"]))))))))))))) +
            0.099211*np.tanh(np.real(((((((np.cos((((complex(8.0)) / (data["std_nom_3"]))))) - ((-((np.cos((((complex(8.0)) / (data["std_nom_2"])))))))))) - (np.sin((((complex(8.0)) * (np.sqrt((np.sin((data["mean_ord_3"]))))))))))) + (np.sin((np.cos((((complex(8.0)) / (data["std_nom_1"])))))))))) +
            0.091995*np.tanh(np.real(((((data["mean_ord_3"]) + (((((data["mean_ord_3"]) + (((((data["mean_ord_0"]) + (((((data["mean_ord_2"]) + (((((((data["mean_day"]) * 2.0)) + (np.conjugate(((((data["mean_nom_7"]) + (data["mean_month"]))) - (np.cos((((data["mean_nom_6"]) * 2.0))))))))) * 2.0)))) * 2.0)))) * 2.0)))) * 2.0)))) * 2.0))) +
            0.099072*np.tanh(np.real(((np.sin((np.sqrt((((np.sqrt((((data["std_bin_3"]) * 2.0)))) * 2.0)))))) / ((((data["mean_nom_9"]) + (np.sin((((((((data["std_bin_3"]) * 2.0)) / ((((data["std_nom_4"]) + (((np.sqrt((data["mean_nom_8"]))) * (data["mean_ord_4"]))))/2.0)))) + (complex(0,1)*np.conjugate(data["std_bin_3"])))))))/2.0))))) +
            0.100000*np.tanh(np.real(((((((data["mean_ord_5"]) * 2.0)) - (np.sin((((((complex(7.0)) / (data["mean_bin_4"]))) / 2.0)))))) - ((((-(((-((((data["std_ord_1"]) / (data["mean_ord_5"]))))))))) + ((-((((((data["mean_ord_2"]) + (data["mean_bin_2"]))) / (((complex(0.636620)) - (((data["mean_ord_1"]) * 2.0)))))))))))))) +
            0.097776*np.tanh(np.real(((np.sqrt((((((data["mean_ord_2"]) * 2.0)) - (((data["mean_bin_2"]) * 2.0)))))) + (((np.cos(((-((((data["mean_bin_2"]) / (np.sqrt((((data["std_ord_0"]) - (((data["mean_bin_2"]) * 2.0))))))))))))) + ((-((((data["mean_bin_2"]) / (data["mean_ord_3"]))))))))))))

def GPComplexII(data):
    return (0.089800*np.tanh(np.imag(((((((((data["std_nom_7"]) + (np.sin((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(((complex(10.0)) * ((-((((data["std_nom_7"]) + (((((data["mean_ord_3"]) + (np.sqrt((data["std_ord_5"]))))) - (np.cos((np.conjugate(data["std_ord_1"]))))))))))))))))))) * 2.0)) * 2.0)) / (data["std_ord_3"])))) +
            0.092783*np.tanh(np.imag(np.sin((((np.sqrt((data["std_ord_3"]))) + (((complex(11.97040462493896484)) / (((data["std_ord_2"]) + ((((((np.conjugate(np.sin((((np.sqrt((data["std_ord_0"]))) / 2.0))))) + (complex(0,1)*np.conjugate(((data["mean_ord_3"]) * 2.0))))/2.0)) + (((data["std_month"]) / 2.0))))))))))))) +
            0.083000*np.tanh(np.imag(((((data["mean_nom_9"]) - (np.cos((((((data["mean_ord_2"]) + (np.tanh((((data["mean_ord_3"]) - (((((complex(0,1)*np.conjugate(data["std_ord_5"])) / 2.0)) / (((data["std_bin_1"]) / 2.0)))))))))) * 2.0)))))) + (np.sin((((((((data["std_bin_1"]) / 2.0)) / (((data["mean_ord_2"]) * (data["mean_nom_8"]))))) / 2.0))))))) +
            0.099855*np.tanh(np.imag(((((data["std_bin_1"]) - (complex(-1.0)))) / (np.sin((((data["mean_nom_7"]) - (((((((data["std_bin_1"]) - ((((data["mean_nom_8"]) + (((data["mean_ord_5"]) - (data["std_bin_1"]))))/2.0)))) - ((((data["mean_nom_8"]) + (((data["mean_ord_3"]) - (data["std_bin_1"]))))/2.0)))) - (data["mean_ord_2"])))))))))) +
            0.098275*np.tanh(np.imag(((np.cos((((((((data["mean_nom_9"]) / 2.0)) - (data["mean_ord_3"]))) - (((data["mean_nom_9"]) - (((np.conjugate(data["mean_nom_9"])) - (((data["mean_ord_3"]) - (data["mean_month"]))))))))))) / (((np.conjugate(data["mean_nom_9"])) - (((data["std_bin_3"]) - ((((data["mean_month"]) + (data["mean_ord_3"]))/2.0))))))))) +
            0.097944*np.tanh(np.imag(((((((np.cos((((np.tanh((data["mean_ord_5"]))) + (((((data["mean_ord_0"]) - (data["mean_ord_3"]))) - (data["mean_ord_3"]))))))) * 2.0)) / (((data["mean_ord_0"]) - ((((((np.tanh((data["std_bin_3"]))) - (data["mean_ord_3"]))) + (((data["std_bin_3"]) - (data["mean_ord_5"]))))/2.0)))))) + (((data["mean_ord_3"]) * 2.0))))) +
            0.097081*np.tanh(np.imag(((np.sqrt((np.sqrt((data["std_ord_4"]))))) / (np.tanh((np.tanh(((((((data["mean_nom_7"]) + (np.sqrt((data["std_nom_8"]))))/2.0)) + ((((-((np.cos((np.sqrt(((((data["std_ord_4"]) + (np.sin((((data["mean_ord_3"]) * 2.0)))))/2.0))))))))) / 2.0))))))))))) +
            0.099980*np.tanh(np.imag(((((data["mean_ord_2"]) / ((((((data["mean_month"]) - (((np.tanh((data["mean_ord_2"]))) / (((data["mean_ord_0"]) - (np.cos((np.sqrt((data["mean_nom_9"]))))))))))) + (complex(0,1)*np.conjugate((-((np.sin((complex(0,1)*np.conjugate(np.cos((((complex(13.24914264678955078)) + (np.sqrt((data["mean_nom_9"])))))))))))))))/2.0)))) * 2.0))) +
            0.100000*np.tanh(np.imag(((((((np.sin((((data["std_bin_1"]) / ((((data["mean_nom_8"]) + (((data["mean_ord_5"]) / 2.0)))/2.0)))))) * 2.0)) * 2.0)) - ((-((np.sin((((data["std_bin_1"]) / ((((data["mean_ord_2"]) + (((complex(0,1)*np.conjugate(data["mean_ord_0"])) / 2.0)))/2.0)))))))))))) +
            0.099956*np.tanh(np.imag(((((((np.sin((((complex(9.0)) * ((((data["mean_ord_3"]) + (((((((data["mean_ord_4"]) + (data["mean_ord_5"]))) + (data["mean_month"]))) * 2.0)))/2.0)))))) / (data["mean_ord_4"]))) - (np.cos((complex(9.0)))))) / ((((data["mean_ord_4"]) + (complex(0,1)*np.conjugate(data["mean_ord_4"])))/2.0))))) +
            0.098886*np.tanh(np.imag(((((complex(3.14884257316589355)) / (np.cos((((((((-((data["mean_ord_3"])))) + (((np.cos((np.sqrt((np.sin((((((data["mean_nom_9"]) * 2.0)) * 2.0)))))))) / (data["std_ord_0"]))))/2.0)) * 2.0)))))) / ((((np.cos((np.sqrt((data["mean_ord_3"]))))) + (data["mean_nom_9"]))/2.0))))) +
            0.099988*np.tanh(np.imag((((((((-((((((data["std_ord_1"]) + (data["mean_nom_8"]))) - (((((data["mean_month"]) + (complex(5.0)))) / 2.0))))))) + (data["mean_month"]))) / ((-((np.sin((((complex(0.636620)) - (np.tanh((((data["mean_month"]) + (((data["std_ord_1"]) + (data["mean_nom_8"])))))))))))))))) * 2.0))) +
            0.100000*np.tanh(np.imag(np.sin((((complex(1.570796)) / ((((((((((data["mean_nom_7"]) + (data["mean_nom_5"]))/2.0)) + (data["mean_nom_1"]))/2.0)) + ((((data["mean_day"]) + (((((np.sin((((data["mean_ord_5"]) - (((((data["mean_nom_1"]) - (complex(0,1)*np.conjugate(np.sqrt((np.cos((data["std_ord_5"])))))))) / 2.0)))))) * (data["mean_ord_4"]))) * 2.0)))/2.0)))/2.0))))))) +
            0.099788*np.tanh(np.imag(((((data["std_nom_7"]) * 2.0)) / (((data["std_nom_1"]) - ((((np.cos((data["std_nom_7"]))) + ((-((((data["mean_ord_2"]) * (((np.sqrt(((((((((data["mean_month"]) * (((data["std_nom_7"]) * 2.0)))) + (data["mean_month"]))/2.0)) * (((np.conjugate(data["std_ord_3"])) * 2.0)))))) * 2.0))))))))/2.0))))))) +
            0.100000*np.tanh(np.imag(((((np.sqrt((data["std_bin_1"]))) * 2.0)) / (((((((((((((data["mean_month"]) * (data["mean_ord_3"]))) + (((data["mean_nom_3"]) + (data["mean_ord_4"]))))/2.0)) + (((data["mean_bin_0"]) + (np.conjugate(((data["mean_bin_2"]) + (((data["mean_nom_3"]) - (data["std_bin_1"])))))))))/2.0)) * 2.0)) + ((-((data["std_bin_1"]))))))))) +
            0.099954*np.tanh(np.imag(((data["mean_nom_8"]) - (np.cos((((complex(0.636620)) / ((((((((data["mean_ord_0"]) * (((((data["mean_ord_5"]) * 2.0)) + (((data["mean_ord_5"]) + (data["mean_ord_5"]))))))) + ((((((((data["mean_nom_9"]) + (data["mean_nom_8"]))) - (np.sqrt(((-((data["mean_nom_7"])))))))) + (data["mean_ord_1"]))/2.0)))/2.0)) / 2.0))))))))) +
            0.099881*np.tanh(np.imag(((((data["std_nom_9"]) / 2.0)) / ((((((((np.tanh((data["mean_ord_1"]))) + (np.tanh((((data["std_ord_4"]) * (((np.cos((data["std_nom_9"]))) * 2.0)))))))/2.0)) - (np.sin((np.sin((((complex(1.0)) - (((data["std_nom_9"]) + (data["mean_nom_5"]))))))))))) / 2.0))))) +
            0.099920*np.tanh(np.imag(((((np.sin((((complex(9.66714859008789062)) / (np.sqrt(((((data["mean_nom_1"]) + (((data["std_bin_0"]) / 2.0)))/2.0)))))))) * 2.0)) + (((np.sin((((complex(9.66714859008789062)) / (np.sqrt(((((((data["mean_day"]) + (data["std_nom_2"]))) + (data["mean_bin_2"]))/2.0)))))))) / (data["mean_bin_2"])))))) +
            0.097186*np.tanh(np.imag(np.tanh((((np.tanh((data["mean_ord_2"]))) + (np.sin((np.sin((((data["mean_nom_6"]) + (((((complex(0,1)*np.conjugate(np.conjugate((-((np.cos((data["mean_nom_8"])))))))) + (((((data["mean_ord_3"]) * (data["mean_nom_8"]))) + (data["mean_ord_1"]))))) - (((np.cos((data["mean_nom_8"]))) / 2.0))))))))))))))) +
            0.099973*np.tanh(np.imag(((data["mean_nom_5"]) / ((((((data["std_ord_3"]) * ((((data["mean_ord_0"]) + (data["mean_ord_5"]))/2.0)))) + (((data["mean_nom_5"]) + (((np.conjugate((((np.tanh((data["mean_bin_0"]))) + (((((data["mean_month"]) / 2.0)) + (data["mean_nom_8"]))))/2.0))) - ((-((((complex(-1.0)) / 2.0))))))))))/2.0))))) +
            0.099094*np.tanh(np.imag(((((complex(0.318310)) + (data["mean_ord_5"]))) / (((((data["std_month"]) + (data["std_day"]))) * (((data["mean_nom_7"]) + (((((((data["mean_bin_0"]) - (complex(0.318310)))) / (((data["mean_ord_5"]) + (np.tanh((data["mean_month"]))))))) / 2.0))))))))) +
            0.099629*np.tanh(np.imag((((((complex(4.0)) + (((data["mean_nom_4"]) / (((np.conjugate(data["std_ord_0"])) - (np.conjugate(((data["mean_nom_4"]) * 2.0))))))))/2.0)) - (((((data["mean_bin_2"]) * 2.0)) / (((np.conjugate(data["std_ord_0"])) - (((complex(0,1)*np.conjugate(complex(0,1)*np.conjugate(data["mean_bin_2"]))) * 2.0))))))))) +
            0.099769*np.tanh(np.imag(((np.conjugate(data["mean_nom_9"])) + (((((data["mean_ord_4"]) / ((((((data["mean_ord_4"]) + (data["mean_bin_1"]))/2.0)) + ((((((((data["std_nom_6"]) + (data["std_nom_3"]))/2.0)) * 2.0)) - (np.cos((data["mean_ord_2"]))))))))) - (np.cos((((np.cos((data["std_month"]))) / (data["mean_ord_4"])))))))))) +
            0.100000*np.tanh(np.imag(((np.cos((((data["mean_nom_2"]) + (data["std_ord_5"]))))) / (((((np.conjugate(((((data["mean_nom_5"]) + (np.tanh((((data["mean_ord_1"]) + (data["mean_nom_9"]))))))) * (((data["mean_nom_2"]) + (data["mean_bin_0"])))))) * 2.0)) + (np.conjugate(((data["std_day"]) - (np.cos((np.sqrt((data["std_ord_5"]))))))))))))) +
            0.099211*np.tanh(np.imag(((((((np.cos((((complex(8.0)) / (data["std_nom_3"]))))) - ((-((np.cos((((complex(8.0)) / (data["std_nom_2"])))))))))) - (np.sin((((complex(8.0)) * (np.sqrt((np.sin((data["mean_ord_3"]))))))))))) + (np.sin((np.cos((((complex(8.0)) / (data["std_nom_1"])))))))))) +
            0.091995*np.tanh(np.imag(((((data["mean_ord_3"]) + (((((data["mean_ord_3"]) + (((((data["mean_ord_0"]) + (((((data["mean_ord_2"]) + (((((((data["mean_day"]) * 2.0)) + (np.conjugate(((((data["mean_nom_7"]) + (data["mean_month"]))) - (np.cos((((data["mean_nom_6"]) * 2.0))))))))) * 2.0)))) * 2.0)))) * 2.0)))) * 2.0)))) * 2.0))) +
            0.099072*np.tanh(np.imag(((np.sin((np.sqrt((((np.sqrt((((data["std_bin_3"]) * 2.0)))) * 2.0)))))) / ((((data["mean_nom_9"]) + (np.sin((((((((data["std_bin_3"]) * 2.0)) / ((((data["std_nom_4"]) + (((np.sqrt((data["mean_nom_8"]))) * (data["mean_ord_4"]))))/2.0)))) + (complex(0,1)*np.conjugate(data["std_bin_3"])))))))/2.0))))) +
            0.100000*np.tanh(np.imag(((((((data["mean_ord_5"]) * 2.0)) - (np.sin((((((complex(7.0)) / (data["mean_bin_4"]))) / 2.0)))))) - ((((-(((-((((data["std_ord_1"]) / (data["mean_ord_5"]))))))))) + ((-((((((data["mean_ord_2"]) + (data["mean_bin_2"]))) / (((complex(0.636620)) - (((data["mean_ord_1"]) * 2.0)))))))))))))) +
            0.097776*np.tanh(np.imag(((np.sqrt((((((data["mean_ord_2"]) * 2.0)) - (((data["mean_bin_2"]) * 2.0)))))) + (((np.cos(((-((((data["mean_bin_2"]) / (np.sqrt((((data["std_ord_0"]) - (((data["mean_bin_2"]) * 2.0))))))))))))) + ((-((((data["mean_bin_2"]) / (data["mean_ord_3"]))))))))))))

def GPReal(data):
    return (GPRealI(data)+GPRealII(data))/2

def GPComplex(data):
    return (GPComplexI(data)+GPComplexII(data))/2


# In[ ]:


roc_auc_score(np.real(df_train.target),Output(GPReal(df_train)))


# In[ ]:


log_loss(np.real(df_train.target),Output(GPReal(df_train)))


# In[ ]:


colors = ['r','b']
plt.figure(figsize=(15,15))
plt.scatter(GPReal(df_train),GPComplex(df_train),s=1,color=[colors[int(i)] for i in df_train.target])


# In[ ]:


submission = pd.DataFrame({'id':df_test.id.values,'target':Output(GPReal(df_test))})


# In[ ]:


submission.to_csv('gpsubmission.csv',index=False)

