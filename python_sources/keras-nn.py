#!/usr/bin/env python
# coding: utf-8

# This is my first experiment in this competition. Whereas XGBoost is highly recommended I rather tried to see how far I can go with an NN (using Keras).
# This is the basic model and with 250 epochs has an accuracy of 80% (really poor).
# I'll continue for a few days researching how much I can optimize this model.
# 
# Now switching to using forests in this new kernel https://www.kaggle.com/mulargui/xgboost
# You can find all my notes and versions at https://github.com/mulargui/kaggle-Classify-forest-types

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#load data
dftrain=pd.read_csv('/kaggle/input/learn-together/train.csv')
dftest=pd.read_csv('/kaggle/input/learn-together/test.csv')


# In[ ]:


####### FEATURE ENGINEERING #####
#taking most of this from my work in feature engineering at https://www.kaggle.com/mulargui/xgboost
#https://www.kaggle.com/arateris/2-layer-k-fold-learning-forest-cover 
#Fixing Hillshade_3pm
#replacing the zeros for better guess, mainly to avoid zeros in the feature engineering and fake outliers. 
num_train = len(dftrain)
tmp = dftrain.drop('Cover_Type', axis = 1)
all_data = tmp.append(dftest)

cols_for_HS = ['Aspect','Slope', 'Hillshade_9am','Hillshade_Noon']
HS_zero = all_data[all_data.Hillshade_3pm==0]
HS_train = all_data[all_data.Hillshade_3pm!=0]

from sklearn.ensemble import RandomForestRegressor
rf_hs = RandomForestRegressor(n_estimators=100).fit(HS_train[cols_for_HS], HS_train.Hillshade_3pm)
out = rf_hs.predict(HS_zero[cols_for_HS]).astype(int)

#I couldn't make this line work, feature not used
#all_data.loc[HS_zero.index,'Hillshade_3pm'] = out
#dftrain['Hillshade_3pm']= all_data.loc[:num_train,'Hillshade_3pm']
#dftest['Hillshade_3pm']= all_data.loc[num_train:,'Hillshade_3pm']

# Add PCA features
from sklearn.decomposition import PCA
pca = PCA(n_components=0.99).fit(all_data)
trans = pca.transform(all_data)

for i in range(trans.shape[1]):
    col_name= 'pca'+str(i+1)
    dftrain[col_name] = trans[:num_train, i]
    dftest[col_name] = trans[num_train:, i]

#https://www.kaggle.com/evimarp/top-6-roosevelt-national-forest-competition
def euclidean(df):
    df['Euclidean_distance_to_hydro'] = (df.Vertical_Distance_To_Hydrology**2 
                                         + df.Horizontal_Distance_To_Hydrology**2)**.5
    return df

dftrain = euclidean(dftrain)
dftest = euclidean(dftest)

from itertools import combinations
def distances(df):
    cols = [
        'Horizontal_Distance_To_Roadways',
        'Horizontal_Distance_To_Fire_Points',
        'Horizontal_Distance_To_Hydrology',
    ]
    df['distance_mean'] = df[cols].mean(axis=1)
    df['distance_sum'] = df[cols].sum(axis=1)
    df['distance_road_fire'] = df[cols[:2]].mean(axis=1)
    df['distance_hydro_fire'] = df[cols[1:]].mean(axis=1)
    df['distance_road_hydro'] = df[[cols[0], cols[2]]].mean(axis=1)
    
    df['distance_sum_road_fire'] = df[cols[:2]].sum(axis=1)
    df['distance_sum_hydro_fire'] = df[cols[1:]].sum(axis=1)
    df['distance_sum_road_hydro'] = df[[cols[0], cols[2]]].sum(axis=1)
    
    df['distance_dif_road_fire'] = df[cols[0]] - df[cols[1]]
    df['distance_dif_hydro_road'] = df[cols[2]] - df[cols[0]]
    df['distance_dif_hydro_fire'] = df[cols[2]] - df[cols[1]]
    
    # Vertical distances measures
    colv = ['Elevation', 'Vertical_Distance_To_Hydrology']
    df['Vertical_dif'] = df[colv[0]] - df[colv[1]]
    df['Vertical_sum'] = df[colv].sum(axis=1)
    
    return df
  
dftrain = distances(dftrain)
dftest = distances(dftest)
    
def shade(df):
    SHADES = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
    
    df['shade_noon_diff'] = df['Hillshade_9am'] - df['Hillshade_Noon']
    df['shade_3pm_diff'] = df['Hillshade_Noon'] - df['Hillshade_3pm']
    df['shade_all_diff'] = df['Hillshade_9am'] - df['Hillshade_3pm']
    df['shade_sum'] = df[SHADES].sum(axis=1)
    df['shade_mean'] = df[SHADES].mean(axis=1)
    
    return df

dftrain = shade(dftrain)
dftest = shade(dftest)

def elevation(df):
    df['ElevationHydro'] = df['Elevation'] - 0.25 * df['Euclidean_distance_to_hydro']
    return df

dftrain = elevation(dftrain)
dftest = elevation(dftest)

def elevationV(df):
    df['ElevationV'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']
    return df

dftrain = elevationV(dftrain)
dftest = elevationV(dftest)

def elevationH(df):
    df['ElevationH'] = df['Elevation'] - 0.19 * df['Horizontal_Distance_To_Hydrology']
    return df

dftrain = elevationH(dftrain)
dftest = elevationH(dftest)

def kernel_features(df):
    df['Elevation2'] = df['Elevation']**2
    df['ElevationLog'] = np.log1p(df['Elevation'])
    return df

dftrain = kernel_features(dftrain)
dftest = kernel_features(dftest)

def degree(df):
    df['Aspect_cos'] = np.cos(np.radians(df.Aspect))
    df['Aspect_sin'] = np.sin(np.radians(df.Aspect))
    #df['Slope_sin'] = np.sin(np.radians(df.Slope))
    df['Aspectcos_Slope'] = df.Slope * df.Aspect_cos
    #df['Aspectsin_Slope'] = df.Slope * df.Aspect_sin
    
    return df

dftrain = degree(dftrain)
dftest = degree(dftest)

from bisect import bisect
cardinals = [i for i in range(45, 361, 90)]
points = ['N', 'E', 'S', 'W']

def cardinal(df):
    df['Cardinal'] = df.Aspect.apply(lambda x: points[bisect(cardinals, x) % 4])
    return df

dftrain = cardinal(dftrain)
dftest = cardinal(dftest)

def cardinal_num(df):
    d = {'N': 0, 'E': 1, 'S': 0, 'W':-1}
    df['Cardinal'] = df.Cardinal.apply(lambda x: d[x])
    return df

dftrain = cardinal_num(dftrain)
dftest = cardinal_num(dftest)

#adding features based on https://douglas-fraser.com/forest_cover_management.pdf pages 21,22
#note: not all climatic and geologic codes have a soil type

def Climatic2(row): 
    if (row['Soil_Type1'] == 1) or (row['Soil_Type2'] == 1) or (row['Soil_Type3'] == 1) or (row['Soil_Type4'] == 1)         or (row['Soil_Type5'] == 1) or (row['Soil_Type6'] == 1) :
        return 1 
    return 0

dftrain['Climatic2'] = dftrain.apply (lambda row: Climatic2(row), axis=1)
dftest['Climatic2'] = dftest.apply (lambda row: Climatic2(row), axis=1)

def Climatic3(row): 
    if (row['Soil_Type7'] == 1) or (row['Soil_Type8'] == 1) :
        return 1 
    return 0

dftrain['Climatic3'] = dftrain.apply (lambda row: Climatic3(row), axis=1)
dftest['Climatic3'] = dftest.apply (lambda row: Climatic3(row), axis=1)

def Climatic4(row): 
    if (row['Soil_Type9'] == 1) or (row['Soil_Type10'] == 1) or (row['Soil_Type11'] == 1) or (row['Soil_Type12'] == 1)         or (row['Soil_Type13'] == 1) :
        return 1 
    return 0

dftrain['Climatic4'] = dftrain.apply (lambda row: Climatic4(row), axis=1)
dftest['Climatic4'] = dftest.apply (lambda row: Climatic4(row), axis=1)

def Climatic5(row): 
    if (row['Soil_Type14'] == 1) or (row['Soil_Type15'] == 1) :
        return 1 
    return 0

dftrain['Climatic5'] = dftrain.apply (lambda row: Climatic5(row), axis=1)
dftest['Climatic5'] = dftest.apply (lambda row: Climatic5(row), axis=1)

def Climatic6(row): 
    if (row['Soil_Type16'] == 1) or (row['Soil_Type17'] == 1) or (row['Soil_Type18'] == 1) :
        return 1 
    return 0

dftrain['Climatic6'] = dftrain.apply (lambda row: Climatic6(row), axis=1)
dftest['Climatic6'] = dftest.apply (lambda row: Climatic6(row), axis=1)

def Climatic7(row): 
    if (row['Soil_Type19'] == 1) or (row['Soil_Type20'] == 1) or (row['Soil_Type21'] == 1) or (row['Soil_Type22'] == 1)         or (row['Soil_Type23'] == 1) or (row['Soil_Type24'] == 1) or (row['Soil_Type25'] == 1) or (row['Soil_Type26'] == 1)         or (row['Soil_Type27'] == 1) or (row['Soil_Type28'] == 1) or (row['Soil_Type29'] == 1) or (row['Soil_Type30'] == 1)         or (row['Soil_Type31'] == 1) or (row['Soil_Type32'] == 1) or (row['Soil_Type33'] == 1) or (row['Soil_Type34'] == 1) :
        return 1 
    return 0

dftrain['Climatic7'] = dftrain.apply (lambda row: Climatic7(row), axis=1)
dftest['Climatic7'] = dftest.apply (lambda row: Climatic7(row), axis=1)

def Climatic8(row): 
    if (row['Soil_Type35'] == 1) or (row['Soil_Type36'] == 1) or (row['Soil_Type37'] == 1) or (row['Soil_Type38'] == 1)         or (row['Soil_Type39'] == 1) or (row['Soil_Type40'] == 1) :
        return 1 
    return 0

dftrain['Climatic8'] = dftrain.apply (lambda row: Climatic8(row), axis=1)
dftest['Climatic8'] = dftest.apply (lambda row: Climatic8(row), axis=1)

def Geologic1(row): 
    if (row['Soil_Type14'] == 1) or (row['Soil_Type15'] == 1) or (row['Soil_Type16'] == 1) or (row['Soil_Type17'] == 1)         or (row['Soil_Type19'] == 1) or (row['Soil_Type20'] == 1) or (row['Soil_Type21'] == 1) :
        return 1 
    return 0

dftrain['Geologic1'] = dftrain.apply (lambda row: Geologic1(row), axis=1)
dftest['Geologic1'] = dftest.apply (lambda row: Geologic1(row), axis=1)

def Geologic2(row): 
    if (row['Soil_Type9'] == 1) or (row['Soil_Type22'] == 1) or (row['Soil_Type23'] == 1) :
        return 1 
    return 0

dftrain['Geologic2'] = dftrain.apply (lambda row: Geologic2(row), axis=1)
dftest['Geologic2'] = dftest.apply (lambda row: Geologic2(row), axis=1)

def Geologic5(row): 
    if (row['Soil_Type7'] == 1) or (row['Soil_Type8'] == 1) :
        return 1 
    return 0

dftrain['Geologic5'] = dftrain.apply (lambda row: Geologic5(row), axis=1)
dftest['Geologic5'] = dftest.apply (lambda row: Geologic5(row), axis=1)

def Geologic7(row): 
    if (row['Soil_Type1'] == 1) or (row['Soil_Type2'] == 1) or (row['Soil_Type3'] == 1) or (row['Soil_Type4'] == 1)         or (row['Soil_Type5'] == 1) or (row['Soil_Type6'] == 1) or (row['Soil_Type10'] == 1)         or (row['Soil_Type11'] == 1) or (row['Soil_Type12'] == 1) or (row['Soil_Type13'] == 1) or (row['Soil_Type18'] == 1)         or (row['Soil_Type24'] == 1) or (row['Soil_Type25'] == 1) or (row['Soil_Type26'] == 1) or (row['Soil_Type27'] == 1)         or (row['Soil_Type28'] == 1) or (row['Soil_Type29'] == 1) or (row['Soil_Type30'] == 1) or (row['Soil_Type31'] == 1)         or (row['Soil_Type32'] == 1) or (row['Soil_Type33'] == 1) or (row['Soil_Type34'] == 1) or (row['Soil_Type35'] == 1)         or (row['Soil_Type36'] == 1) or (row['Soil_Type37'] == 1) or (row['Soil_Type38'] == 1) or (row['Soil_Type39'] == 1)         or (row['Soil_Type40'] == 1) :
        return 1 
    return 0

dftrain['Geologic7'] = dftrain.apply (lambda row: Geologic7(row), axis=1)
dftest['Geologic7'] = dftest.apply (lambda row: Geologic7(row), axis=1)


# In[ ]:


####### DATA PREPARATION #####
#split train data in features and labels
y = dftrain.Cover_Type
x = dftrain.drop(['Id','Cover_Type'], axis=1)

# split test data in features and Ids
Ids = dftest.Id
x_predict = dftest.drop('Id', axis=1)

#force all types to float
x = x.astype(float)
x_predict = x_predict.astype(float)

#normalize features
def normalize(feature):
    min=x[feature].min()
    min2=x_predict[feature].min()
    if (min2 < min):
        min=min2

    max=x[feature].max()
    max2=x_predict[feature].max()
    if (max2 > max):
        max=max2
        
    x[feature]=(x[feature]-min)/(max-min)                             
    x_predict[feature]=(x_predict[feature]-min)/(max-min)  
    
    return                                

normalize("Elevation")
normalize("Aspect")
normalize("Slope")
normalize("Horizontal_Distance_To_Hydrology")
normalize("Vertical_Distance_To_Hydrology")
normalize("Horizontal_Distance_To_Roadways")
normalize("Hillshade_9am")
normalize("Hillshade_Noon")
normalize("Hillshade_3pm")
normalize("Horizontal_Distance_To_Fire_Points")
normalize("pca1")
normalize("Euclidean_distance_to_hydro")
normalize("distance_mean")
normalize("distance_sum")
normalize("distance_road_fire")
normalize("distance_hydro_fire")
normalize("distance_road_hydro")
normalize("distance_sum_road_fire")
normalize("distance_sum_hydro_fire")
normalize("distance_sum_road_hydro")
normalize("distance_dif_road_fire")
normalize("distance_dif_hydro_road")
normalize("distance_dif_hydro_fire")
normalize("Vertical_dif")
normalize("Vertical_sum")
normalize("shade_noon_diff")
normalize("shade_3pm_diff")
normalize("shade_all_diff")
normalize("shade_sum")
normalize("shade_mean")
normalize("ElevationHydro")
normalize("ElevationV")
normalize("ElevationH")
normalize("Elevation2")
normalize("ElevationLog")
normalize("Aspect_cos")
normalize("Aspect_sin")
normalize("Aspectcos_Slope")


# In[ ]:


# convert the label to One Hot Encoding
num_classes = 7

#to_categorical requires 0..6 instead of 1..7
y -=1
y = y.to_numpy()

from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes)

#validate data - no rows with all zeros
#x.index[x.eq(0).all(1)]
print(x[x.eq(0).all(1)].empty)
print(x_predict[x_predict.eq(0).all(1)].empty)

#convert the features dataframes to numpy arrays
x = x.to_numpy()
x_predict = x_predict.to_numpy()

#split in train (80%) and test (20%) sets 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, stratify=y)


# In[ ]:


#here is the NN model
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

num_features = x_train.shape[1]

model = Sequential()
model.add(Dense(units=num_features*2, activation='relu', kernel_initializer='normal', input_dim=num_features))
model.add(Dropout(0.2))
model.add(Dense(units=num_features*2, activation='relu', kernel_initializer='normal'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='Adam',
              metrics=['accuracy'])

#train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=250)

# Predict!!
y_predict = model.predict(x_predict)


# In[ ]:


for i in range(10):
	print(y_predict[i], np.argmax(y_predict[i])+1)


# In[ ]:


# Save predictions to a file for submission
#argmax give us the highest probable label
# we add one to the predictions to scale from 0..6 to 1..7
output = pd.DataFrame({'Id': Ids,
                       'Cover_Type': y_predict.argmax(axis=1)+1})
output.to_csv('submission.csv', index=False)

#create a link to download the file    
from IPython.display import FileLink
FileLink(r'submission.csv')

