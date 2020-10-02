#!/usr/bin/env python
# coding: utf-8

# * **kernel still in progress....**

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import *


# In[ ]:


#Loading the data in pandas DataFrame
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
structures = pd.read_csv('../input/structures.csv')
#mc = pd.read_csv('../input/mulliken_charges.csv')
#scc = pd.read_csv('../input/scalar_coupling_contributions.csv')

print(train.shape, test.shape)
print(train.head())
print(test.head())


# In[ ]:


#To convert the datatypes of the columns in the dataframe to reduce memeory usage

def type_conversion(pds_obj):
    
    optimized_obj = pds_obj.copy()
    
    #selecting all columns with integer datatype
    int_col = pds_obj.select_dtypes(include=['int64'])
    if len(int_col.columns)==0:
        pass
    else:
        new_int = int_col.apply(pd.to_numeric,downcast='unsigned')
        optimized_obj[int_col.columns] = new_int
    
    #selecting all columns with float datatype
    float_col = pds_obj.select_dtypes('float')
    if len(float_col.columns)==0:
        pass
    else:
        new_float = float_col.apply(pd.to_numeric,downcast='float')
        optimized_obj[float_col.columns] = new_float
    #selecting all columns with object datatype
    objects = pds_obj.select_dtypes('object').copy()
    if len(objects.columns)==0:
        pass
    else:
        obj =objects.astype('category')
        optimized_obj[objects.columns]=obj
    
    

    
    
    
    return optimized_obj


# In[ ]:


#function to print the memory usage by the dataframe
def mem_info(df):
    print(df.info(memory_usage='deep'))


# In[ ]:


mem_info(train)
mem_info(test)
mem_info(structures)


# In[ ]:


# reducing the memory usage
train=type_conversion(train)
test = type_conversion(test)
structures=type_conversion(structures)


# In[ ]:


mem_info(train)
mem_info(test)
mem_info(structures)


# In[ ]:


#Selecting each atom from molecule type and encoding it to numeric datatype
lbl = preprocessing.LabelEncoder()
for i in range(4):
    train['type'+str(i)] = lbl.fit_transform(train['type'].map(lambda x: str(x)[i])).astype('uint8')
    test['type'+str(i)] = lbl.transform(test['type'].map(lambda x: str(x)[i])).astype('uint8')



print(len(train))
print("\n\n",train.head())


# Now we need to merge all the x y z values in structures DataFrame to the train and test set with respect to their molecule name and atom index to later use these features to calculate more complex features.

# In[ ]:


def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

train = map_atom_info(train, 0)
train = map_atom_info(train, 1)

test = map_atom_info(test, 0)
test = map_atom_info(test, 1)


# In[ ]:


print(train.shape)
print(test.shape)


# Droping the unnecessary columns.

# In[ ]:


train.drop(columns=['id', 'molecule_name'], inplace=True)
test.drop(columns=['id','molecule_name'], inplace=True)
print(train.shape, test.shape)


# Now we calculate the euclidean distance between the x0,y0,z0 and x1,y1,z1 for each molecule and store it in the train and test dataframe.

# In[ ]:



train_p0 = train[['x_0', 'y_0', 'z_0']].values
train_p1 = train[['x_1', 'y_1', 'z_1']].values
test_p0 = test[['x_0', 'y_0', 'z_0']].values
test_p1 = test[['x_1', 'y_1', 'z_1']].values

train['dist'] = np.linalg.norm(train_p0 - train_p1, axis=1)
test['dist'] = np.linalg.norm(test_p0 - test_p1, axis=1)


# now lets calculate the distance between each co-ordinate with atom_idx 0,1

# In[ ]:




train['dist_x'] = (train['x_0'] - train['x_1']) ** 2
test['dist_x'] = (test['x_0'] - test['x_1']) ** 2
train['dist_y'] = (train['y_0'] - train['y_1']) ** 2
test['dist_y'] = (test['y_0'] - test['y_1']) ** 2
train['dist_z'] = (train['z_0'] - train['z_1']) ** 2
test['dist_z'] = (test['z_0'] - test['z_1']) ** 2


# In[ ]:


#Here take the mean, min and max distance for each molecule type and each atom in a type to get more specific distance features.

#mean
train['dist_to_type_mean'] = train['dist'] / train.groupby('type')['dist'].transform('mean')
test['dist_to_type_mean'] = test['dist'] / test.groupby('type')['dist'].transform('mean')

train['dist_to_type_0_mean'] = train['dist'] / train.groupby('type0')['dist'].transform('mean')
test['dist_to_type_0_mean'] = test['dist'] / test.groupby('type0')['dist'].transform('mean')

train['dist_to_type_1_mean'] = train['dist'] / train.groupby('type1')['dist'].transform('mean')
test['dist_to_type_1_mean'] = test['dist'] / test.groupby('type1')['dist'].transform('mean')

train['dist_to_type_2_mean'] = train['dist'] / train.groupby('type2')['dist'].transform('mean')
test['dist_to_type_2_mean'] = test['dist'] / test.groupby('type2')['dist'].transform('mean')

train['dist_to_type_3_mean'] = train['dist'] / train.groupby('type3')['dist'].transform('mean')
test['dist_to_type_3_mean'] = test['dist'] / test.groupby('type3')['dist'].transform('mean')

#min
train['dist_to_type_min'] = train['dist'] / train.groupby('type')['dist'].transform('min')
test['dist_to_type_min'] = test['dist'] / test.groupby('type')['dist'].transform('min')

train['dist_to_type_0_min'] = train['dist'] / train.groupby('type0')['dist'].transform('min')
test['dist_to_type_0_min'] = test['dist'] / test.groupby('type0')['dist'].transform('min')

train['dist_to_type_1_min'] = train['dist'] / train.groupby('type1')['dist'].transform('min')
test['dist_to_type_1_min'] = test['dist'] / test.groupby('type1')['dist'].transform('min')

train['dist_to_type_2_min'] = train['dist'] / train.groupby('type2')['dist'].transform('min')
test['dist_to_type_2_min'] = test['dist'] / test.groupby('type2')['dist'].transform('min')

train['dist_to_type_3_min'] = train['dist'] / train.groupby('type3')['dist'].transform('min')
test['dist_to_type_3_min'] = test['dist'] / test.groupby('type3')['dist'].transform('min')

#max
train['dist_to_type_max'] = train['dist'] / train.groupby('type')['dist'].transform('max')
test['dist_to_type_max'] = test['dist'] / test.groupby('type')['dist'].transform('max')

train['dist_to_type_0_max'] = train['dist'] / train.groupby('type0')['dist'].transform('max')
test['dist_to_type_0_max'] = test['dist'] / test.groupby('type0')['dist'].transform('mean')

train['dist_to_type_1_max'] = train['dist'] / train.groupby('type1')['dist'].transform('max')
test['dist_to_type_1_max'] = test['dist'] / test.groupby('type1')['dist'].transform('max')

train['dist_to_type_2_max'] = train['dist'] / train.groupby('type2')['dist'].transform('max')
test['dist_to_type_2_max'] = test['dist'] / test.groupby('type2')['dist'].transform('max')

train['dist_to_type_3_max'] = train['dist'] / train.groupby('type3')['dist'].transform('max')
test['dist_to_type_3_max'] = test['dist'] / test.groupby('type3')['dist'].transform('max')

print(train.shape, test.shape)


# More complex features

# In[ ]:


# This function calculates some aggregate features for each x,y,z co-ordinate with respect to it's atom_index


def features(df):
   for c in ['0', '1']:
       col = [c1 + c  for c1 in ['x_','y_','z_']]
       for agg in ['min', 'max', 'sum', 'mean', 'std']:
           df[c+agg] = eval('df[col].' + agg + '(axis=1)')
           df[c+'a'+agg] = eval('df[col].abs().' + agg + '(axis=1)')
   return df

train = features(train)
test = features(test)
print(train.shape, test.shape)


# In[ ]:


train.head()


# In[ ]:


# Encoding the string type data to numeric datatype so that it can be used as traning data for the deep learning model

from sklearn.preprocessing import LabelEncoder


for f in ['atom_0', 'atom_1','type']:
    lbl = LabelEncoder()
    lbl.fit(list(train[f].values) + list(test[f].values))
    train[f] = lbl.transform(train[f])
    test[f] = lbl.transform(test[f])
    
train.head()


# In[ ]:


train = type_conversion(train)
test=type_conversion(test)


# Building the deep learning model

# In[ ]:


#importing required libraries for building our deep learning model

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Flatten
from tensorflow.python.keras.layers import BatchNormalization,Add,Dropout
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import plotly.graph_objs as go
import plotly
from plotly.offline import iplot
print(plotly.__version__)           # version 1.9.4 required
plotly.offline.init_notebook_mode(connected=True)


# In[ ]:


# create dataframes to store loss and validation for each molecule type
losses =pd.DataFrame()
val_losses=pd.DataFrame()


# In[ ]:


def plot_history(history, label):
    plt.figure()
    plt.figure(figsize=(15,7))
    plt.plot(history.history['loss'])  
    plt.plot(history.history['val_loss'])
    plt.title('Loss for %s' % label)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    _= plt.legend(['Train','Validation'], loc='upper left')
    plt.show()


# In[ ]:


#deep learning model
def create_model():
    
    model = Sequential()

    model.add(Dense(128, kernel_initializer='normal',input_dim = 54, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(512, kernel_initializer='normal',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1024,kernel_initializer='normal',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1024,kernel_initializer='normal',activation='relu'))



    model.add(Dense(1, kernel_initializer='normal',activation='linear'))


    model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])


    return model


# In[ ]:


test_prediction=np.zeros(len(test))


config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 2} ) 
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.Session(config=config) 
K.set_session(sess)

types = train['type'].unique()
for t in types:
    print('Training %s' %t, 'out of', types, '\n')
    x_train=train[train["type"]==t]
    y_train=x_train.loc[:,"scalar_coupling_constant"].values
    x_train=x_train.drop(['scalar_coupling_constant'],axis=1)
    x_test=test[test["type"]==t]
    
    
    
    model = create_model() 
    
    # setting up callbacks to stop traning if loss is not going down for specified number of epochs
    early_stopping = callbacks.EarlyStopping(monitor='loss',patience=7,mode='auto', restore_best_weights=True)
    
    # fitting the model
    history = model.fit(x_train,y_train,epochs=200,batch_size=2000,validation_split=0.2,callbacks=[early_stopping])
    
    plot_history(history, t)
    
    model.save_weights("./type_{}_model.h5".format(t))
    print("{} model saved".format(t))
    
    losses[t] = pd.Series(history.history['loss'])
    val_losses[t]= pd.Series(history.history['val_loss'])
    

    #getting predictions for the test set
    test_predict=model.predict(x_test)
    test_predict = test_predict.reshape(-1)
    
    test_prediction[test["type"]==t]=test_predict
    
    K.clear_session()
    
    


# In[ ]:


#Decoding the molecule types which were encoded using the labelencoder

columns = lbl.inverse_transform(losses.columns)
losses.columns= columns
val_losses.columns = columns


# In[ ]:


#plotting graph of stored validation loss for each molecule type

def traces(a,name):
    trace= go.Scatter(
        x=losses.index.values,
        y=a,
        name = name
    )
    return trace

t1= traces(val_losses.iloc[:,0],columns[0])
t2= traces(val_losses.iloc[:,1],columns[1])
t3= traces(val_losses.iloc[:,2],columns[2])
t4= traces(val_losses.iloc[:,3],columns[3])
t5= traces(val_losses.iloc[:,4],columns[4])
t6= traces(val_losses.iloc[:,5],columns[5])
t7= traces(val_losses.iloc[:,6],columns[6])
t8= traces(val_losses.iloc[:,7],columns[7])


data= [t1,t2,t3,t4,t5,t6,t7,t8]

layout = go.Layout(
    title='Validation loss over the epochs',
    xaxis=dict(
        title='Epochs',
        titlefont=dict(
            family='Courier New, monospace',
            size=15,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Validation loss',
        titlefont=dict(
            family='Courier New, monospace',
            size=15,
            color='#7f7f7f'
        )
    )
)

fig = go.Figure(data=data, layout=layout)

iplot(fig)


# In[ ]:


#writing out the submission to the submissions file
sample_sub = pd.read_csv('../input/sample_submission.csv')
sample_sub['scalar_coupling_constant']= test_prediction
print(sample_sub.head())


# In[ ]:


sample_sub.to_csv('submission.csv',index=False)

