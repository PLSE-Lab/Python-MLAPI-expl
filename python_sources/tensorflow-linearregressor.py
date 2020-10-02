import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow.contrib.learn.python.learn as learn
from sklearn import metrics, preprocessing
from sklearn.utils import shuffle
from itertools import chain
import math

SRC_FILE_NAME = '../input/autos.csv'
autos = pd.read_csv(SRC_FILE_NAME, encoding="cp858")

autos = autos[(autos.yearOfRegistration >= 1990) & (autos.yearOfRegistration < 2017)]
autos = autos[(autos.price >= 100) & (autos.price <= 100000)]
autos = autos[(autos.powerPS < 600)]              
autos = autos.dropna(subset=['vehicleType','gearbox','model','fuelType','notRepairedDamage'], how='any')

en_features = ['vehicleType', 'gearbox', 'model', 'fuelType', 'brand', 'notRepairedDamage']
for f in en_features:
    le = preprocessing.LabelEncoder()
    le.fit(list(set(autos[f])))
    autos[f] = le.transform(autos[f]) 

drop_features = ['dateCrawled','name','seller','offerType','abtest','dateCreated'\
                 ,'nrOfPictures','postalCode','lastSeen' ]
                
for f in drop_features:
    del autos[f]
    
target = [math.log10(x) for x in list(chain.from_iterable(autos.as_matrix(columns=["price"])))]

del autos['price']
autos_m = autos.as_matrix()
x = preprocessing.StandardScaler().fit_transform(autos_m)
features = learn.infer_real_valued_columns_from_input(x)
regressor = learn.LinearRegressor(feature_columns=features)
regressor.fit(x, target, steps=400, batch_size=32)
preds = list(regressor.predict(x))
score = metrics.mean_squared_error(preds, target)
print("MSE: {0}".format(score))

targetDF = pd.DataFrame(data={'Target':target, 'Predicate':preds},columns=['Target','Predicate'])
samples = shuffle(targetDF, n_samples=1000)
plot = samples.plot(kind='scatter',x='Target',y='Predicate')
fig = plot.get_figure()
fig.savefig('price_target_pred.png')

df = pd.DataFrame(data={'Name':autos.columns,'Weight':[x for x in list(chain.from_iterable(regressor.weights_))]},columns=['Name','Weight'])
df.to_csv('weight.csv')

# Any results you write to the current directory are saved as output.