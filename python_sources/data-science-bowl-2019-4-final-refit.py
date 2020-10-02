import numpy as np
import pandas as pd
import pickle
import sys
sys.path.append( '/kaggle/input/models' )
from outliers import *
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import RandomUniform

# Importing the processed data

with open( '/kaggle/input/models/processed.pkl', 'rb' ) as file:
    d, train, val, features = pickle.load( file )
    
with open( '/kaggle/input/models/relevant.pkl', 'rb' ) as file:
    relevant = pickle.load( file )

#################### PREPROCESSING ############################################

# Censoring outliers with a custom function
# Current version of the function can be found on https://github.com/d-kulikov/preprocessing
bounds = Outliers.fit( d[ relevant ] )

Outliers.transform( d, bounds )

# Imputing missing accuracies
for name in [ 'accuracy_total', 'accuracy_assess', 'accuracy_game', 'accuracy_mushroom', 'accuracy_bird', 'accuracy_cauldron', 
             'accuracy_cart', 'accuracy_chest', 'accuracy_this' ] :
    d.loc[ d[ name ].isnull(), name ] = d[ name ].mean()

# Scaling

Scaler = StandardScaler()

Scaler.fit( d[ relevant ] )

########## FITTING THE FINAL MODEL FOR THE CURRENT FOLD #######################

x = np.float32( Scaler.transform( d[ relevant ] ) )

y = np.float32( d[ 'accuracy' ] )

batch = x.shape[ 0 ]

nnmodel = Sequential()
nnmodel.add( Dense( 10, input_shape=( len( relevant ), ),
                    kernel_initializer=RandomUniform( minval=-1, maxval=1, seed=1791095845 ), activation='tanh' ) )
nnmodel.add( Dense( 5, kernel_initializer=RandomUniform( minval=-1, maxval=1, seed=550290313 ), activation='tanh' ) )
nnmodel.add( Dense( 1, kernel_initializer=RandomUniform( minval=-1, maxval=1, seed=1013994432 ), activation='sigmoid' ) )
nnmodel.compile( optimizer='adam', loss='binary_crossentropy' )
nnmodel.fit( x, y, batch_size=batch, shuffle=False, epochs=90 )

# Saving the model

nnmodel.save( 'model_fold_1.h5' )
