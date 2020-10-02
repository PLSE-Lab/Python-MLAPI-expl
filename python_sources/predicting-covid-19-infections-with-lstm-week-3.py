#!/usr/bin/env python
# coding: utf-8

# # Predicting COVID-19 Infections with LSTM
# > by Yohan Chung
# 
# The pandemic of Coronavirus (COVID-19) became reality and many countries around the globe strive to contain further spread of the virus with social distancing as well as qurantining of those who contact the infected. The project aims to predict future of the infected by country and identify which country needs more attention. The prediction of the newly infected starts with reading data from the files train.csv and test.csv as below.
# 
# The datatsets are provided by Johns Hopkins University and include the COVID-19 confirmed cases & fatalities by country.
# 
# 1. Preprocess Data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf

train_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv') # historical data 
test_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv') # predictions to be filled


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

# Min-Max Scaler
scaler_c = MinMaxScaler(feature_range=(0, 100))
train_df['ConfirmedCases_scaled'] = None
train_df[['ConfirmedCases_scaled']] = scaler_c.fit_transform(train_df[['ConfirmedCases']])

scaler_f = MinMaxScaler(feature_range=(0, 100))
train_df['Fatalities_scaled'] = None
train_df[['Fatalities_scaled']] = scaler_f.fit_transform(train_df[['Fatalities']])

# Get dummy columns for geo location
geo_columns = []
n_geo_columns = 306
for i in range(n_geo_columns):
    geo_columns.append('Geo_{}'.format(i))
train_df.drop(columns=geo_columns, inplace=True, errors='ignore')

lbl_encoder = LabelEncoder()
scaler_g = MinMaxScaler(feature_range=(0, 1))
hot_encoder = OneHotEncoder(sparse=False)
train_df['Geo'] = train_df['Country_Region'].astype(str) + '_' + train_df['Province_State'].astype(str)
train_df[['Geo']] = lbl_encoder.fit_transform(train_df[['Geo']])
train_df = pd.get_dummies(train_df, prefix_sep="_", columns=['Geo'])


print(train_df.columns)
train_df[['ConfirmedCases', 'ConfirmedCases_scaled', 'Fatalities', 'Fatalities_scaled',  'Geo_0']].head()    


# In[ ]:


historical_steps = 14
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


# In[ ]:


import random

max_index = np.array(train_inputs).shape[0] - 1
indices = []

for i in range(int(max_index*0.2)):
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


# 2. Train LSTM Model with initial state

# In[ ]:


from tensorflow.keras import Model, Sequential, backend as K
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout

n_output_node = 1
input_shape=np.array(train_inputs).shape[-2:]
input_shape_geo=np.array(train_inputs_geo).shape

batch_size = 128
epochs = 150
lr = 0.001

def create_model(model_type_name, load_weight, inputs=None, inputs_geo=None, targets=None, v_inputs=None, v_inputs_geo=None, v_targets=None):
    
    geo_input = Input(shape=(n_geo_columns,), name='input_geo')
    
    h_state = Dense(256)(geo_input)
    h_state = Dropout(0.1)(h_state)
    h_state = Dense(256, activation='relu')(h_state)
    c_state = Dense(256)(geo_input)
    c_state = Dropout(0.1)(c_state)
    c_state = Dense(256, activation='relu')(c_state)

    ts_input = Input(shape=input_shape, name='input_ts')
    lstm = LSTM(256, return_sequences=True)(ts_input, initial_state=[ h_state, c_state ])
    lstm = Dropout(0.1)(lstm)
    lstm = Bidirectional(LSTM(128))(lstm)
    lstm = Dropout(0.1)(lstm)
    main_output = Dense(n_output_node, activation='relu', name='output_main')(lstm)
    
    model = Model(inputs=[ geo_input, ts_input ], outputs=main_output)    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.MSLE, metrics=[ tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError() ])
    
    model_path = 'model_{}.h5'.format(model_type_name)
    
    if load_weight:
        
        model = tf.keras.models.load_model(model_path)
    else:
    
        history = model.fit([ inputs_geo, inputs ],  targets,                     epochs=epochs,                     batch_size=batch_size,                     verbose=1,                     validation_data=({ 'input_geo': v_inputs_geo, 'input_ts': v_inputs },{ 'output_main': v_targets}))

        scores = model.evaluate({ 'input_geo': inputs_geo, 'input_ts': inputs }, targets)
        print("Model Accuracy: {}".format(scores))
        
        plt.plot(history.history['loss']) 
        plt.plot(history.history['val_loss'])
        plt.title('Loss over epochs')
        plt.legend(['Train', 'Validation'], loc='best')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()
        
        model.save(model_path) 
        
    return model
                        

model_cases = create_model('case', False, np.array(train_inputs), np.array(train_inputs_geo), np.array(train_targets_c), np.array(val_inputs), np.array(val_inputs_geo), np.array(val_targets_c))
model_fatality = create_model('fatality', False, np.array(train_inputs_f), np.array(train_inputs_geo), np.array(train_targets_f), np.array(val_inputs_f), np.array(val_inputs_geo), np.array(val_targets_f))


# 3. Analyse Results

# In[ ]:


model_cases = create_model('case', load_weight=True)
model_fatality = create_model('fatality', load_weight=True)


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

temp_df = train_df.copy()
temp_df = pd.concat([ temp_df, test_df[test_df.Day > train_df.iloc[-1].Day] ], ignore_index=True)
temp_df['ConfirmedCases_scaled_predicted'] = None
temp_df['Fatalities_scaled_predicted'] = None

day_predicting_from = 64
day_predicting_to = 106

current_row = None
hist_rows = None

tic = time.perf_counter()
counter = 0

# For each country and state
# for c, s in test_df[(test_df.Country_Region=='Australia') & (test_df.Province_State=='New South Wales')].groupby(['Country_Region', 'Province_State']): 
# for c, s in test_df[(test_df.Country_Region=='Australia') & (test_df.Province_State=='Victoria')].groupby(['Country_Region', 'Province_State']): 
# for c, s in test_df[(test_df.Country_Region=='Korea, South') & (test_df.Province_State=='Unknown')].groupby(['Country_Region', 'Province_State']):  
# for c, s in test_df[(test_df.Country_Region=='Italy') & (test_df.Province_State=='Unknown')].groupby(['Country_Region', 'Province_State']):  
# for c, s in test_df[(test_df.Country_Region=='France') & (test_df.Province_State=='Unknown')].groupby(['Country_Region', 'Province_State']): 
# for c, s in test_df[(test_df.Country_Region=='China') & (test_df.Province_State=='Hubei')].groupby(['Country_Region', 'Province_State']):  
for c, s in test_df.groupby(['Country_Region', 'Province_State']):  
    
    toc = time.perf_counter()
    
    country = c[0]
    state = c[1]
    
    # Traverse from Day 64 to the end
    for day in range(day_predicting_from, day_predicting_to + 1):   
        current_row = temp_df[ (temp_df.Country_Region == country) & (temp_df.Province_State == state) & (temp_df.Day == day)]
        hist_rows = temp_df[(temp_df.Country_Region == country) & (temp_df.Province_State == state) & (temp_df.Day >= (day - historical_steps)) & (temp_df.Day < day)]
        
        # Only predict when historical steps exist
        if not current_row.empty and not hist_rows.empty and hist_rows.shape[0] == historical_steps:    
            input_geo = np.array(hist_rows.iloc[-historical_steps][geo_columns]).reshape(1, len(geo_columns))
            input_c = np.array(hist_rows.iloc[-historical_steps:,][['ConfirmedCases_scaled']]).reshape(1, input_shape[0], input_shape[1])
            input_f = np.array(hist_rows.iloc[-historical_steps:,][['Fatalities_scaled']]).reshape(1, input_shape[0], input_shape[1])
            pred = model_cases.predict([ tf.convert_to_tensor(input_geo, np.float64), tf.convert_to_tensor(input_c, np.float64) ])
            pred_f = model_fatality.predict([ tf.convert_to_tensor(input_geo, np.float64), tf.convert_to_tensor(input_f, np.float64) ])
  
            # Update the predict fields
            current_idx = current_row.index.values[0]
            temp_df.at[current_idx, 'ConfirmedCases_scaled_predicted'] = float(pred[0][0])
            temp_df.at[current_idx, 'Fatalities_scaled_predicted'] = float(pred_f[0][0])
            
            last = hist_rows.iloc[-1,]
            for col_name in geo_columns:
                temp_df.at[current_idx, col_name] = last[col_name]
            
            # Update the existing fields if empty
            if (current_row['ConfirmedCases'].values[0] == None) and (current_row['Fatalities'].values[0] == None):
                temp_df.at[current_idx, 'ConfirmedCases_scaled'] = float(pred[0][0])
                temp_df.at[current_idx, 'Fatalities_scaled'] = float(pred_f[0][0])
    
    toc = time.perf_counter()
    counter = counter + 1
    print('{:.2f} sec(s) taken - Geo: {},{}, Count: {}'.format((toc-tic), country, state, counter))
        

temp_df['ConfirmedCases_inversed_predicted'] = None
temp_df['Fatalities_inversed_predicted'] = None
temp_df[['ConfirmedCases_inversed_predicted']] = scaler_c.inverse_transform(temp_df[['ConfirmedCases_scaled_predicted']])
temp_df[['Fatalities_inversed_predicted']] = scaler_f.inverse_transform(temp_df[['Fatalities_scaled_predicted']]) 


# In[ ]:


temp_df[(temp_df.Country_Region=='Australia') & (temp_df.Province_State=='New South Wales')] [['ForecastId', 'Day', 'Date', 'Country_Region', 'Province_State', 'ConfirmedCases', 'ConfirmedCases_scaled', 'ConfirmedCases_scaled_predicted', 'ConfirmedCases_inversed_predicted']] .iloc[-45:,:]


# In[ ]:


def plot_by_country(country, state):

    hist_df = temp_df[(temp_df.Country_Region == country) & (temp_df.Province_State == state) & (temp_df.Day <= 74)].groupby(['Country_Region', 'Province_State', 'Day', 'Date']).agg({'ConfirmedCases': 'first'}).reset_index()
    pred_df = temp_df[(temp_df.Country_Region == country) & (temp_df.Province_State == state) & (temp_df.Day >= 64)].groupby(['Country_Region', 'Province_State', 'Day', 'Date']).agg({'ConfirmedCases_inversed_predicted': 'first'}).reset_index()

    plt.title('{}, {}'.format(state, country))
    plt.plot(hist_df.Day, hist_df.ConfirmedCases, label='Historical')
    plt.plot(pred_df.Day, pred_df.ConfirmedCases_inversed_predicted, label='Predictive')
    plt.axvline(x=day_predicting_from, color='r', linestyle='--', linewidth=1, label='2020-04-04')
    plt.xlabel('Day')
    plt.ylabel('Cases')
    plt.legend()
    plt.show()
    
[ plot_by_country(country, state) for country, state in [     ('Australia', 'New South Wales'), ('Australia', 'Victoria'), ('Korea, South', 'Unknown'), ('China', 'Hubei'), ('Italy', 'Unknown'), ('France', 'Unknown')]]


# In[ ]:


tic = time.perf_counter()

for i in range(len(test_df)):
    
    country = test_df.at[i, 'Country_Region']
    state = test_df.at[i, 'Province_State']
    day = test_df.at[i, 'Day']
    
    current_df = temp_df[(temp_df.Country_Region == country) & (temp_df.Province_State == state) & (temp_df.Day == day)]
    test_df.at[i, 'ConfirmedCases'] = current_df['ConfirmedCases_inversed_predicted'].values[0]
    test_df.at[i, 'Fatalities'] = current_df['Fatalities_inversed_predicted'].values[0]
    
    if i%1000 == 0:
        toc = time.perf_counter()
        print('{:.2f} sec(s) taken'.format((toc-tic)))

test_df[['ForecastId', 'ConfirmedCases', 'Fatalities']].to_csv('submission.csv', index=False)

