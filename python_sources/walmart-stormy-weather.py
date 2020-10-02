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
import math
from datetime import timedelta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import Ridge

weatherData = pd.read_csv('../input/weather.csv')
keyData = pd.read_csv('../input/key.csv')
trainData = pd.read_csv('../input/train.csv')
testData = pd.read_csv('../input/test.csv')
NoOfStations = np.amax(keyData['station_nbr'])
predFile = open('predictions.csv','w')
predFile.write("date,store_nbr,item_nbr,units\n")
for stnNo in range(1,NoOfStations+1) :
    stationData = weatherData.loc[weatherData['station_nbr'] == stnNo]
    columns = ['date','station_nbr','tavg','wetbulb','heat','cool','snowfall','preciptotal','stnpressure','sealevel','avgspeed']
    selectedCol = pd.DataFrame(stationData,columns = columns)
    selectedCol.reset_index(drop=True,inplace=True)
    for col in columns :
        if col == 'date' :
            selectedCol.loc[:,col] = pd.to_datetime(selectedCol.loc[:,col],format='%Y-%m-%d')
        else :
            selectedCol.loc[:,col] = pd.to_numeric(selectedCol.loc[:,col],errors='coerce')
    for index, row in selectedCol.iterrows() :
        for j in columns :
            if j!= 'date' and j!= 'station_nbr' and math.isnan(row[j]) :
                prevDate = row['date'] - timedelta(days=1)
                prevDateData = selectedCol.loc[selectedCol['date'] == prevDate]
                nextDate = row['date'] + timedelta(days=1)
                nextDateData = selectedCol.loc[selectedCol['date'] == nextDate]
                prevDateCol = prevDateData[j]
                nextDateCol = nextDateData[j]
                if not(prevDateCol.empty or nextDateCol.empty or math.isnan(prevDateCol.iloc[0]) or math.isnan(nextDateCol.iloc[0])) :
                    selectedCol.loc[index,j] = (prevDateCol.iloc[0] + nextDateCol.iloc[0]) / 2
                elif not (prevDateCol.empty or math.isnan(prevDateCol.iloc[0])) :
                    selectedCol.loc[index,j] = prevDateCol.iloc[0]
                elif not (nextDateCol.empty or math.isnan(nextDateCol.iloc[0])) :
                    selectedCol.loc[index,j] = nextDateCol.iloc[0]
                elif not math.isnan(selectedCol[j].mean()) :
                    selectedCol.loc[index,j] = selectedCol[j].mean()
                else :
                    selectedCol.loc[index,j] = 0
    selectedCol['wetbulbHumidity'] = selectedCol['tavg'] - selectedCol['wetbulb']
    selectedCol['waterPrecip'] = selectedCol['preciptotal'] - selectedCol['snowfall']
    pipelineDf = selectedCol.drop(['date','station_nbr'],axis=1)
    pipeline = Pipeline([
            ('kernelpca',KernelPCA(kernel='rbf')),
            ('scaler',StandardScaler())])
    transformedDf = pd.DataFrame(pipeline.fit_transform(pipelineDf))
    transformedDf = pd.concat([transformedDf,selectedCol['date']],axis=1)
    storesList = keyData.loc[keyData['station_nbr'] == stnNo]
    for index3,row3 in storesList.iterrows():
        storeNo = row3['store_nbr']
        storeData = trainData.loc[trainData['store_nbr'] == storeNo]
        storeTestData = testData.loc[testData['store_nbr'] == storeNo]
        itemsList = np.unique(storeData['item_nbr'])
        for itemNo in np.nditer(itemsList):
            itemData = storeData.loc[storeData['item_nbr'] == itemNo]
            itemTestData = storeTestData.loc[storeTestData['item_nbr'] == itemNo]
            itemData.loc[:,'date'] = pd.to_datetime(itemData.loc[:,'date'],format='%Y-%m-%d')
            itemTestData.loc[:,'date'] = pd.to_datetime(itemTestData.loc[:,'date'],format='%Y-%m-%d') 
            trainItemData = transformedDf.merge(itemData,left_on='date',right_on='date',how='inner')
            testItemData = transformedDf.merge(itemTestData,left_on='date',right_on='date',how='inner')
            X_train = trainItemData.drop(['units','date','store_nbr','item_nbr'],axis=1)
            X_test = testItemData.drop(['date','store_nbr','item_nbr'],axis=1)
            y_train = trainItemData['units'].copy()
            lin_reg = Ridge(alpha=0.1)
            lin_reg.fit(X_train,y_train)
            if X_test.shape[0] > 0 :
                y_pred = lin_reg.predict(X_test)
                testItemData['units'] = y_pred.astype(int)
                exportDF = pd.DataFrame(testItemData,columns=['date','store_nbr','item_nbr','units'])
                predFile.write(exportDF.to_csv(header=False,index=False))
predFile.close()
predictions = pd.read_csv('predictions.csv')
idDF = pd.DataFrame(predictions,columns=['store_nbr','item_nbr','date'])
idDF['id'] = idDF.apply(lambda x: '_'.join(x.dropna().astype(str).values.tolist()),axis=1)
finalDF = pd.DataFrame(idDF,columns=['id'])
finalDF['units'] = predictions['units']
finalDF.to_csv('submission.csv',index=False)