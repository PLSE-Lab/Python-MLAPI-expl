''' FIFA 18
Multiple Machine Learning applications. Split the dataframe by some age (veterans/youth). 
Train on veterans' physical attributes to create a neural net that predicts position preferences.
Predict youth position preferences, find top 5 positions, then group everyone into 6 categories:
-strikers
-wingers
-atk midfielders
-def midfielders
-defenders
-wingbacks

Within each category, train network on veteran attributes to predict youth Overall.
Rank results by change (improvement) in Overall score.
The idea here is that it will find underrated youth players who are maybe even playing out of position.
'''
# LIBRARIES
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

# IMPORT THE DF
df = pd.read_csv('../input/CompleteDataset.csv', index_col=None)

# DATA PREPROCESSING FUNCTIONS
def convert_value(money_str):
    notes = ''
    # Find the numbers and append
    for letter in money_str:
        if letter in '1234567890.':
            notes = notes + letter
        else:
            pass
    # Divide by 1000 to convert K to M for value
    if 'K' in money_str:
        return (float(notes)/1000)    
    else:
        return float(notes)

def convert_wage(money_str):
    notes = ''
    # Find the numbers and append
    for letter in money_str:
        if letter in '1234567890.':
            notes = notes + letter
        else:
            pass
    
    return float(notes)

def convert_attributes(number_str):
    if type(number_str) == str:
        if '+' in number_str:
            return float(number_str.split('+')[0])
        elif '-' in number_str:
            return float(number_str.split('-')[0])
        else:
            return float(number_str)

def find_GK(preferences):
    if 'GK' in preferences:
        return 1
    else:
        return 0


# CONVERT THE DATA USING METHODS ABOVE
df['Wage'] = df['Wage'].apply(convert_wage) # Units = K
print(df['Wage'][-10:].dtype)
df['Value'] = df['Value'].apply(convert_value) # Units = M
print(df['Value'][-10:].dtype)

# Grab the attributes columns and position preferences
attributes = df.columns[13: 47]
x = df.columns[47:75]
position_preferences = x[:5].append(x[6:16]).append(x[17:]) 
# To take out "ID" and "Preferred Positions"
# We will create a model in KERAS that predicts this

# Convert the attributes because some of them are in a weird format (eg. '72+3')
for skill in df[attributes]:
    df[skill] = df[skill].apply(convert_attributes)
df[attributes].info() # All should be float


# FILTER OUT THE GOALKEEPERS
df['GK'] = df['Preferred Positions'].apply(find_GK)
df_out_field = df[df['GK'] == 0]
df_gk = df[df['GK'] == 1]
# We dont want them because they do no have fUll attributes

# Fix the NANs
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(df_out_field[attributes])
df_out_field[attributes] = imputer.transform(df_out_field[attributes])
print('ENSURE THERE ARE NO NULL ENTRIES \n' + str(df_out_field[attributes].isnull().any()))

# Quick PCA to see which attributes are the most closely correlated
def PCA_analysis(input_values, n_components, column_names):
    
    scaler = StandardScaler()
    scaler.fit(input_values) #Fit to data
    x_scaled = scaler.transform(input_values)
    
    # SCREE PLOT to find the best # of PCAs to guess
    U, S, V = np.linalg.svd(x_scaled)
    eigvals = S**2 / np.cumsum(S)[-1]

    fig = plt.figure(figsize=(8,5))
    sing_vals = np.arange(len(eigvals)) + 1
    plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')

    # PCA on newly scaled data
    from sklearn.decomposition import PCA
    pca = PCA(n_components= n_components) # Found by Scree plot elbow
    pca.fit(x_scaled) 
    x_pca = pca.transform(x_scaled)

    # Heat map to visualize the breakdown of PCA
    pca.components_
    df_comp = pd.DataFrame(pca.components_, columns= range(1,len(column_names)+1))
    plt.figure(figsize=(12,6))
    sns.heatmap(df_comp, cmap='plasma')
    
    stats_table = []
    for i,name in enumerate(attributes, 1):
        stats_table.append([i,name])
    print(stats_table)


x = df_out_field[attributes].values # Values only (NUMPY)
PCA_analysis(x, 6, attributes)

# No surprise, the GK stats are dead weight
attributes = attributes[:11].append(attributes[16:])
x = df_out_field[attributes].values # Values only (NUMPY)
PCA_analysis(x, 5, attributes)
# Scree plot says the elbow is now at 5
# Heatmap is mildly interesting, depicts a correlation betweem STRENGTH and HEAD ACCURACY
# also FREE KICK ACCURACY and LONG PASSING


''' POSITION PLACER
The idea here is that young players might not be in their optimal spot'''
# Make a new position preferences list that grabs the top 5 for each
y = df_out_field[position_preferences]
x=[]
for index, row in y.iterrows():
    x.append(row.sort_values(axis=0, ascending=False)[:5].index.values)
df_out_field['Preferred Positions'] = x


# SPLIT at whatever age you want
young = df_out_field[df_out_field['Age'] <= 23]
veteran = df_out_field[df_out_field['Age'] > 23]

# TRAIN/TEST on veterans, PREDICT youth
x = veteran[attributes].values
x_youth = young[attributes].values
y = veteran[position_preferences].values

scaler = StandardScaler()
scaler.fit(x) #Fit to veteran data
x_scaled = scaler.transform(x)
x_youth_scaled = scaler.transform(x_youth)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

ANN = Sequential()
# Adding the INPUT LAYER and the first HIDDEN layer WITH DROPOUT
ANN.add(Dense(units=27, activation='relu', input_dim=29, kernel_initializer='uniform'))
ANN.add(Dropout(rate=0.1))
# Rule of thumb: #Nodes = (#Xs + #ys)/2
ANN.add(Dense(units=27, activation='relu', kernel_initializer='normal'))
ANN.add(Dropout(rate=0.1))
# Output layer
ANN.add(Dense(units=26, activation='relu', kernel_initializer='normal'))
ANN.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy']) 

ANN.fit(x_train, y_train, batch_size=100, nb_epoch=100)
y_pred = ANN.predict(x_test)

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, y_pred))
print('ROOT MEAN SQUARED ERROR: %.3f / 100' %(rms))
# Move on if the error seems reasonable

# MAKE PREDICTIONS
young[position_preferences]= ANN.predict(x_youth_scaled)

# UPDATE the preffered position column like before
y = young[position_preferences]
x=[]
for index, row in y.iterrows():
    x.append(row.sort_values(axis=0, ascending=False)[:5].index.values)
young['Preferred Positions'] = x




# GROUP THE YOUNG GUYS
young_str_i = []
young_wing_i = []
young_atk_mid_i = []
young_def_mid_i = []
young_def_i = []
young_wb_i = []

for index, row in young.iterrows():
    for position in row['Preferred Positions']:
        if position in ['ST','RS','LS','CF','RF','LF'] and index not in young_str_i:
            young_str_i.append(index)
        elif position in ['RW','LW','RF','LF'] and index not in young_wing_i:
            young_wing_i.append(index)
        elif position in ['RAM','LAM','CAM','CM','RM','LM','RCM','LCM'] and index not in young_atk_mid_i:
            young_atk_mid_i.append(index)
        elif position in ['RDM','LDM','CDM','CM','RCM','LCM'] and index not in young_def_mid_i:
            young_def_mid_i.append(index)
        elif position in ['RCB','LCB','RB','LB','CB'] and index not in young_def_i:
            young_def_i.append(index)
        elif position in ['RWB','LWB'] and index not in young_wb_i:
            young_wb_i.append(index)
# Sorry, couldn't figure out a clever way to append directly from the dataframe
# Would have used their names but there are repeats

young_str = young.loc[young_str_i]
young_wing = young.loc[young_wing_i]
young_atk_mid = young.loc[young_atk_mid_i]
young_def_mid = young.loc[young_def_mid_i]
young_def = young.loc[young_def_i]
young_wb = young.loc[young_wb_i]


# GROUP THE OLD GUYS
veteran_str_i = []
veteran_wing_i = []
veteran_atk_mid_i = []
veteran_def_mid_i = []
veteran_def_i = []
veteran_wb_i = []

for index, row in veteran.iterrows():
    for position in row['Preferred Positions']:
        if position in ['ST','RS','LS','CF','RF','LF'] and index not in veteran_str_i:
            veteran_str_i.append(index)
        elif position in ['RW','LW','RF','LF'] and index not in veteran_wing_i:
            veteran_wing_i.append(index)
        elif position in ['RAM','LAM','CAM','CM','RM','LM','RCM','LCM'] and index not in veteran_atk_mid_i:
            veteran_atk_mid_i.append(index)
        elif position in ['RDM','LDM','CDM','CM','RCM','LCM'] and index not in veteran_def_mid_i:
            veteran_def_mid_i.append(index)
        elif position in ['RCB','LCB','RB','LB','CB'] and index not in veteran_def_i:
            veteran_def_i.append(index)
        elif position in ['RWB','LWB'] and index not in veteran_wb_i:
            veteran_wb_i.append(index)

veteran_str = veteran.loc[veteran_str_i]
veteran_wing = veteran.loc[veteran_wing_i]
veteran_atk_mid = veteran.loc[veteran_atk_mid_i]
veteran_def_mid = veteran.loc[veteran_def_mid_i]
veteran_def = veteran.loc[veteran_def_i]
veteran_wb = veteran.loc[veteran_wb_i]


# Create NEURAL NET that predicts overall based on corresponding veteral category
def predict_overall(df_old, df_young):
    
    x_train = df_old[attributes].values
    y_train = df_old['Overall'].values
    
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    
    ANN = Sequential()
    ANN.add(Dense(units=15, activation='relu', input_dim=29, kernel_initializer='uniform'))
    ANN.add(Dropout(rate=0.1))
    ANN.add(Dense(units=15, activation='relu', kernel_initializer='normal'))
    ANN.add(Dropout(rate=0.1))
    ANN.add(Dense(units=1, activation='relu', kernel_initializer='normal'))
    ANN.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy']) 
    
    ANN.fit(x_train, y_train, batch_size=100, nb_epoch=100)
    
    x_input = df_young[attributes].values
    
    scaler.fit(x_input)
    x_input = scaler.transform(x_input)
    
    y_pred = ANN.predict(x_input)
    return y_pred


# PREDICT YOUTH OVERALL
young_str['Predicted Overall'] = predict_overall(veteran_str, young_str)
young_wing['Predicted Overall'] = predict_overall(veteran_wing, young_wing)
young_atk_mid['Predicted Overall'] = predict_overall(veteran_atk_mid, young_atk_mid)
young_def_mid['Predicted Overall'] = predict_overall(veteran_def_mid, young_def_mid)
young_def['Predicted Overall'] = predict_overall(veteran_def, young_def)
young_wb['Predicted Overall'] = predict_overall(veteran_wb, young_wb)

young_str['Overall Change'] = young_str['Predicted Overall'] - young_str['Overall']
young_wing['Overall Change'] = young_wing['Predicted Overall'] - young_wing['Overall'] 
young_atk_mid['Overall Change'] = young_atk_mid['Predicted Overall'] - young_atk_mid['Overall']
young_def_mid['Overall Change'] = young_def_mid['Predicted Overall'] - young_def_mid['Overall']
young_def['Overall Change'] = young_def['Predicted Overall'] - young_def['Overall']
young_wb['Overall Change'] = young_wb['Predicted Overall'] - young_wb['Overall']


# Display the top 10 underrated youngsters in each category
print('UNDERRATED YOUNG STRIKERS')
print(young_str[young_str['Overall']>75][['Name','Age','Club','Overall','Overall Change']].sort_values('Overall Change', ascending=False)[:10])
print('UNDERRATED YOUNG WINGERS')
print(young_wing[young_wing['Overall']>75][['Name','Age','Club','Overall','Overall Change']].sort_values('Overall Change', ascending=False)[:10])
print('UNDERRATED YOUNG ATK MIDS')
print(young_atk_mid[young_atk_mid['Overall']>75][['Name','Age','Club','Overall','Overall Change']].sort_values('Overall Change', ascending=False)[:10])
print('UNDERRATED YOUNG DEF MIDS')
print(young_def_mid[young_def_mid['Overall']>75][['Name','Age','Club','Overall','Overall Change']].sort_values('Overall Change', ascending=False)[:10])
print('UNDERRATED YOUNG DEFENDERS')
print(young_def[young_def['Overall']>75][['Name','Age','Club','Overall','Overall Change']].sort_values('Overall Change', ascending=False)[:10])
print('UNDERRATED YOUNG WINGBACKS')
print(young_wb[young_wb['Overall']>75][['Name','Overall','Age','Club','Overall Change']].sort_values('Overall Change', ascending=False)[:10])
