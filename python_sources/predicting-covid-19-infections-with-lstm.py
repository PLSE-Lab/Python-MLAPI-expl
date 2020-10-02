#!/usr/bin/env python
# coding: utf-8

# # Predicting COVID-19 Infections with LSTM
# 
# > *by Yohan Chung*
# 
# The pandemic of Coronavirus (COVID-19) became reality and many countries around the globe strive to contain further spread of the virus with social distancing as well as qurantining of those who contact the infected. The project aims to predict future of the infected by country and identify which country needs more attention. The prediction of the newly infected starts with reading data from the files train.csv and test.csv as below. 
# 
# The datatsets are provided by Johns Hopkins University and include the COVID-19 confirmed cases & fatalities by country.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf

# pd.set_option('display.float_format', lambda x: '%.20f' % x)

train_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv') # historical data
test_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv') # predictions to be filled


# *Cleansing & Transforming Data*
# 
# I processed data before using them for modelling and details are below.
# * Filling blank cells with default value
# * Removing special char which can malfunction computation
# * Converting datetime to day count from the starting date of data
# * Scaling data to lower values

# In[ ]:


train_df = train_df.fillna({'Province_State': 'Unknown'})
test_df = test_df.fillna({'Province_State': 'Unknown'})
train_df['Country_Region']= train_df['Country_Region'].str.replace("'", "")
train_df['Province_State']= train_df['Province_State'].str.replace("'", "")
test_df['Country_Region']= test_df['Country_Region'].str.replace("'", "")
test_df['Province_State']= test_df['Province_State'].str.replace("'", "")
train_df.isna().sum()


# In[ ]:


def to_datetime(dt):
    return datetime.datetime.strptime(dt, '%Y-%m-%d')

def count_days(dt):
    return (dt - datetime.datetime.strptime('2020-01-22', "%Y-%m-%d")).days

train_df['Date_dt'] = train_df['Date'].map(to_datetime)
train_df['Day'] = train_df['Date_dt'].map(count_days)
test_df['Date_dt'] = test_df['Date'].map(to_datetime)
test_df['Day'] = test_df['Date_dt'].map(count_days)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.utils import to_categorical

# Min-Max Scaler
scaler_c = MinMaxScaler(feature_range=(0, 100))
train_df['ConfirmedCases_scaled'] = None
train_df[['ConfirmedCases_scaled']] = scaler_c.fit_transform(train_df[['ConfirmedCases']])

scaler_f = MinMaxScaler(feature_range=(0, 100))
train_df['Fatalities_scaled'] = None
train_df[['Fatalities_scaled']] = scaler_f.fit_transform(train_df[['Fatalities']])

# Get dummy columns for geo location
geo_columns = []
for i in range(294):
    geo_columns.append('Geo_{}'.format(i))
train_df.drop(columns=geo_columns, inplace=True, errors='ignore')

lbl_encoder = LabelEncoder()
scaler_g = MinMaxScaler(feature_range=(0, 1))
hot_encoder = OneHotEncoder(sparse=False)
train_df['Geo'] = train_df['Country_Region'].astype(str) + '_' + train_df['Province_State'].astype(str)
train_df[['Geo']] = lbl_encoder.fit_transform(train_df[['Geo']])
train_df = pd.get_dummies(train_df, prefix_sep="_", columns=['Geo'])

train_df[['ConfirmedCases', 'ConfirmedCases_scaled', 'Fatalities', 'Fatalities_scaled',  'Geo_0']].head()    


# *Preparing Model Input & Target*
# 
# In this step, I prepare senquential input data with the historical steps of 7, which will be time series to predict the confirmed cases on the next day.

# In[ ]:


historical_steps = 7
n_output_node = 1

def make_sequential_input(df):
    
    inputs_c, inputs_f, inputs_geo, targets_c, targets_f = [], [], [], [], []
    
    for i in range(len(df) - historical_steps - 1):
        
        if df.iloc[i]['Country_Region'] == df.iloc[i + historical_steps]['Country_Region'] and             df.iloc[i]['Province_State'] == df.iloc[i + historical_steps]['Province_State']:
            
            # iloc[a:b] startnig from index 'a' and ending before b
            inputs_c.append(np.array(df.iloc[i : i + historical_steps][['ConfirmedCases_scaled']]).tolist()) # time seires until t-1
            inputs_f.append(np.array(df.iloc[i : i + historical_steps][['Fatalities_scaled']]).tolist()) # time seires until t-1
            inputs_geo.append(np.array(df.iloc[i + historical_steps][geo_columns]).tolist())
            targets_c.append(np.array(df.iloc[i + historical_steps][['ConfirmedCases_scaled']]).tolist()) # result data at time t
            targets_f.append(np.array(df.iloc[i + historical_steps][['Fatalities_scaled']]).tolist()) # result data at time t
              
    return inputs_c, inputs_f, inputs_geo, targets_c, targets_f

# Make sequential input for training and validation
train_inputs, train_inputs_f, train_inputs_geo, train_targets_c, train_targets_f = make_sequential_input(train_df)

print('Train input shape: {}'.format(np.shape(train_inputs)))
print('Train input geo shape: {}'.format(np.shape(train_inputs_geo)))


# Here I extract validataion dataset out of the prepared for modelling.

# In[ ]:


import random

max_index = np.array(train_inputs).shape[0] - 1
indices = []

for i in range(int(max_index*0.20)):
    indices.append(random.randint(0, max_index))

val_inputs = [ train_inputs[i] for i in indices ]
val_inputs_f = [ train_inputs_f[i] for i in indices ]
val_inputs_geo = [ train_inputs_geo[i] for i in indices  ] 
val_targets_c = [ train_targets_c[i] for i in indices ]
val_targets_f = [ train_targets_f[i] for i in indices ]

train_inputs = [ elem for i, elem in enumerate(train_inputs) if i not in indices ] 
train_inputs_f = [ elem for i, elem in enumerate(train_inputs_f) if i not in indices ] 
train_inputs_geo = [ elem for i, elem in enumerate(train_inputs_geo) if i not in indices ] 
train_targets_c = [ elem for i, elem in enumerate(train_targets_c) if i not in indices ] 
train_targets_f = [ elem for i, elem in enumerate(train_targets_f) if i not in indices ] 

pd.set_option('display.max_colwidth', -1)
print('No. train data: {}'.format(len(train_inputs)))
print('No. validation data: {}'.format(len(val_inputs)))


# # LSTM Model 
# > Baseline
# 
# Note that I use mean squared log error which give more penalty on underestimation over overestimation, since the trend is likely to grow up.

# In[ ]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

#### Train the model ####
n_output_node = 1
input_shape=np.array(train_inputs).shape[-2:]

batch_size = 64
epochs = 200
lr = 0.001

def create_model(inputs, targets, val_inputs, val_targets):
    
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.05))
    model.add(LSTM(32))
    model.add(Dropout(0.05))
    model.add(Dense(n_output_node, activation='relu'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.MSLE, metrics=[ tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError() ])

    history = model.fit(inputs, targets,               epochs=epochs,               batch_size=batch_size,               validation_data=(val_inputs, val_targets))

    scores = model.evaluate(inputs, targets)
    print("Model Accuracy: {}".format(scores))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss over epochs')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
    
    return model

print('Tarining model for ConfirmedCases')
model_cases = create_model(train_inputs, train_targets_c, val_inputs, val_targets_c)
print('Tarining model for Fatalities')
model_fatality = create_model(train_inputs_f, train_targets_f, val_inputs_f, val_targets_f)

model_cases.save('model_cases')
model_fatality.save('model_fatality')


# # LSTM with Initial State
# > Advanced

# The following codes were used when creating LSTM model with time series data for confirmed cases, as well as geo location. However, it did not give me higher accuracy versus the aformentioned model.

# In[ ]:


from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

n_output_node = 1
input_shape=np.array(train_inputs).shape[-2:]
input_shape_geo=np.array(train_inputs_geo).shape

batch_size = 64
epochs = 200
lr = 0.001


def create_model(inputs, inputs_geo, targets, v_inputs, v_inputs_geo, v_targets):
    
    geo_input = Input(shape=(294,), name='input_geo')
    h_state = Dense(64, activation='relu')(geo_input)
    h_state = Dense(64, activation='relu')(h_state)
    c_state = Dense(64, activation='relu')(geo_input)
    c_state = Dense(64, activation='relu')(c_state)

    ts_input = Input(shape=input_shape, name='input_ts')
    lstm = LSTM(64, return_sequences=True)(ts_input, initial_state=[ h_state, c_state ])
    lstm = Dropout(0.05)(lstm)
    lstm = LSTM(32)(lstm)
    lstm = Dropout(0.05)(lstm)
    main_output = Dense(n_output_node, activation='relu', name='output_main')(lstm)
    
    model = Model(inputs=[ geo_input, ts_input ], outputs=main_output)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.MSLE, metrics=[ tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError() ])
    
    history = model.fit([ inputs_geo, inputs ],  targets,               epochs=epochs,               batch_size=batch_size,               validation_data=({ 'input_geo': v_inputs_geo, 'input_ts': v_inputs },{ 'output_main': v_targets}))

    scores = model.evaluate({ 'input_geo': inputs_geo, 'input_ts': inputs }, targets)
    print("Model Accuracy: {}".format(scores))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss over epochs')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
    
    return model
                        
    

model_cases = create_model(np.array(train_inputs), np.array(train_inputs_geo), np.array(train_targets_c), np.array(val_inputs), np.array(val_inputs_geo), np.array(val_targets_c))
model_fatality = create_model(np.array(train_inputs_f), np.array(train_inputs_geo), np.array(train_targets_f), np.array(val_inputs_f), np.array(val_inputs_geo), np.array(val_targets_f))

model_cases.save('model_c_with_state')
model_fatality.save('model_f_with_state')


# # Interpreting Data

# In[ ]:


model_cases = tf.keras.models.load_model('model_c_with_state')
model_fatality = tf.keras.models.load_model('model_f_with_state')


# In[ ]:


def predict_cases(country, state):
    
    df = train_df[(train_df['Country_Region'] == country) & (train_df['Province_State'] == state) ]

    inputs = np.array(df[['ConfirmedCases_scaled']][-historical_steps-1:-1])
    inputs_geo = np.array(df.iloc[-1][geo_columns])
    actuals = np.array(df.iloc[-1][['ConfirmedCases']])
    
    predictions = model_cases.predict([  np.array(inputs_geo).astype(np.float32).reshape(1, len(geo_columns)), np.array(inputs).reshape(1, input_shape[0], input_shape[1]) ]).reshape(-1).tolist()
    
    print('Inputs: {}, Pred: {}, Expected: {}'.format(         np.array(df[['ConfirmedCases']][-historical_steps-1:-1])[:,0].tolist(),         scaler_c.inverse_transform(np.array(predictions).reshape(-1,1)),         actuals))
    
def predict_fatality(country, state):
    
    df = train_df[(train_df['Country_Region'] == country) & (train_df['Province_State'] == state) ]

    inputs = np.array(df[['Fatalities_scaled']][-historical_steps-1:-1])
    inputs_geo = np.array(df.iloc[-1][geo_columns])
    actuals = np.array(df.iloc[-1][['Fatalities']])
    
    predictions = model_fatality.predict([  np.array(inputs_geo).astype(np.float32).reshape(1, len(geo_columns)), np.array(inputs).reshape(1, input_shape[0], input_shape[1]) ]).reshape(-1).tolist()
    
    print('Inputs: {}, Pred: {}, Expected: {}'.format(         np.array(df[['Fatalities']][-historical_steps-1:-1])[:,0].tolist(),         scaler_f.inverse_transform(np.array(predictions).reshape(-1,1)),         actuals))
      

predict_cases('Australia', 'Victoria')
predict_cases('Australia', 'New South Wales')
predict_cases('Korea, South', 'Unknown')
predict_cases('Iran', 'Unknown')
predict_cases('Italy', 'Unknown')

predict_fatality('Australia', 'Victoria')
predict_fatality('Australia', 'New South Wales')
predict_fatality('Korea, South', 'Unknown')
predict_fatality('Iran', 'Unknown')
predict_fatality('Italy', 'Unknown')


# In[ ]:


import time

test_df['ConfirmedCases'] = None
test_df['Fatalities'] = None
geo_df = None

tic = time.perf_counter()
temp_df = train_df

# For each test row
for i in range(len(test_df)):
    
    if i%1000 == 0:
        toc = time.perf_counter()
        print('Looping throught the index {} - {:.2f} sec(s) taken...'.format(i, (toc-tic)))
   
    current = test_df.iloc[i]         
    geo_df = temp_df[(temp_df.Country_Region == current.Country_Region) & (temp_df.Province_State == current.Province_State) & (temp_df.Day >= (current.Day - historical_steps)) & (temp_df.Day <= (current.Day))]
        
     # Find historical steps in train data
    if not geo_df.empty and geo_df.shape[0] >= 7:                 
        
        
        if geo_df.shape[0] == 8:
            input_geo = np.array(geo_df.iloc[-historical_steps-1][geo_columns]).astype(np.float32).reshape(1, len(geo_columns))
            pred = model_cases.predict([ input_geo, np.array(geo_df.iloc[-historical_steps-1:-1][['ConfirmedCases_scaled']]).reshape(1, input_shape[0], input_shape[1]) ])
            pred_f = model_fatality.predict([ input_geo, np.array(geo_df.iloc[-historical_steps-1:-1,][['Fatalities_scaled']]).reshape(1, input_shape[0], input_shape[1]) ])
        else:
            input_geo = np.array(geo_df.iloc[-historical_steps][geo_columns]).astype(np.float32).reshape(1, len(geo_columns))
            pred = model_cases.predict([ input_geo, np.array(geo_df.iloc[-historical_steps:,][['ConfirmedCases_scaled']]).reshape(1, input_shape[0], input_shape[1]) ])
            pred_f = model_fatality.predict([ input_geo, np.array(geo_df.iloc[-historical_steps:,][['Fatalities_scaled']]).reshape(1, input_shape[0], input_shape[1]) ])
              
        test_df.loc[i, 'ConfirmedCases_scaled'] = pred[0][0]
        test_df.loc[i, 'Fatalities_scaled'] = pred_f[0][0]
        
        # Save current data in train_df for next if empty
        if geo_df.iloc[-1:,].Day.values[0] != current.Day:  
            
            new_item = { 'ConfirmedCases_scaled': pred[0][0], 'Fatalities_scaled': pred_f[0][0], 'Day': current.Day,                          'Country_Region': current.Country_Region, 'Province_State': current.Province_State }
            
            for j in range(len(geo_columns)):
                new_item['Geo_' + str(j)] = 1 if geo_df.iloc[-1:,]['Geo_' + str(j)].values[0] == 1 else 0
            
            temp_df = temp_df.append(new_item, ignore_index=True)

test_df[['ConfirmedCases']] = scaler_c.inverse_transform(test_df[['ConfirmedCases_scaled']])
test_df[['Fatalities']] = scaler_f.inverse_transform(test_df[['Fatalities_scaled']]) 


# In[ ]:


test_df[(test_df.Country_Region=='Australia') & (test_df.Province_State=='New South Wales')].iloc[-35:,:]
# temp_df[(temp_df.Country_Region=='Australia') & (temp_df.Province_State=='New South Wales')].iloc[-35:,:]


# In[ ]:


def plot_by_country(country, state):

    from_day_predicting = 57
    hist_df = train_df[(train_df.Country_Region == country) & (train_df.Province_State == state)].groupby(['Country_Region', 'Province_State', 'Day', 'Date']).agg({'ConfirmedCases': 'sum'}).reset_index()
    pred_df = test_df[(test_df.Country_Region == country) & (test_df.Province_State == state)].groupby(['Country_Region', 'Province_State', 'Day', 'Date']).agg({'ConfirmedCases': 'sum'}).reset_index()

    plt.title('{}, {}'.format(state, country))
    plt.plot(hist_df.Day, hist_df.ConfirmedCases, label='Historical')
    plt.plot(pred_df.Day, pred_df.ConfirmedCases, label='Predictive')
    plt.axvline(x=57, color='r', linestyle='--', linewidth=1, label='2019-03-31')
    plt.xlabel('Day')
    plt.ylabel('Cases')
    plt.legend()
    plt.show()
    
[ plot_by_country(country, state) for country, state in [     ('Australia', 'New South Wales'), ('Australia', 'Victoria'), ('Korea, South', 'Unknown'), ('China', 'Hubei'), ('Italy', 'Unknown')]]


# In[ ]:


test_df[['ForecastId', 'ConfirmedCases', 'Fatalities']].to_csv('submission.csv', index=False)

