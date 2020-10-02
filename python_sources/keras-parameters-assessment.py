import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing

#########################################################################
# Helpers
#########################################################################

def create_nn_model(input_dim, activation, layers, optimizer):
    model = Sequential()
    
    l_num = 0
    for l in layers:
        if l_num == 0:
            model.add( Dense(l, input_dim=input_dim, activation=activation, init='he_normal') )
        else:
            model.add( Dense(l, activation=activation, init='he_normal') )
        l_num = l_num + 1
    
    model.compile(optimizer=optimizer, loss='mse')
    
    return model


#######################################################################
# Data preparation
#######################################################################

ID     = 'Id'
TARGET = 'SalePrice'

train = pd.read_csv('../input/train.csv')

FEATURES = train.columns.drop([ID, TARGET])

f_cat = train[FEATURES].select_dtypes(include=['object']).columns
f_num = train[FEATURES].select_dtypes(exclude=['object']).columns

# Replace NAs
train[f_num] = train[f_num].fillna(train[f_num].mean())
train[f_cat] = train[f_cat].fillna('?')


dummy_cat = pd.get_dummies(train[f_cat])
scale_num = pd.DataFrame(preprocessing.scale(train[f_num]), columns=f_num)

train = train[[ID, TARGET]]
train[f_num] = scale_num
train = pd.concat([train, dummy_cat], axis=1)

FEATURES = train.columns.drop([ID, TARGET])

X = train[FEATURES].values
y = np.log(train[TARGET].values)

#########################################################################
# Main code
#########################################################################


activations = ['softplus', 'relu', 'linear']
# It seems 'softsign', 'tanh','sigmoid', 'hard_sigmoid' do not perform well on the House Prices data
#activations = ['softplus', 'relu', 'linear','softsign', 'tanh','sigmoid', 'hard_sigmoid']
#optimizers = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']
optimizers = ['adagrad', 'rmsprop','adam']

layers = [[150,1], [150, 50, 1], [200, 100, 50, 1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

nnet_params = []
nnet_score = []

for a in activations:
	for o in optimizers:
		for l in layers:
			model = create_nn_model(input_dim=X.shape[1], activation=a, layers=l, optimizer=o)
			fit = model.fit(X_train, y_train, batch_size=100, nb_epoch=100, validation_split = 0.1, verbose=0)
			score = np.sqrt(model.evaluate(X_test, y_test))
			print("\nActivation: {} Optimizer: {} Layers: {} Score: {}\n".format(a, o, l, score))
			nnet_params.append( str(a) +'-' + str(o)+ '-' + str(l) )
			nnet_score.append(score)

res = pd.DataFrame({'params':nnet_params, 'score':nnet_score})
res.sort(['score'], ascending=True, inplace=True)

print(res)

res.to_csv('./keras_params_.csv')

f, ax = plt.subplots(figsize=(20, 15))
sns.set(style="whitegrid")
ax=sns.barplot(x='score',y='params',data=res,label="Keras parameters - RMSE")
ax.set(xlabel='RMSE', ylabel='Parameters')
plt.savefig('./keras_params.png')
plt.show()