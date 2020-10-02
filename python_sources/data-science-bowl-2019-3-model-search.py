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
from keras.callbacks import EarlyStopping
from time import time
from gc import collect
from scipy.stats import anderson, ttest_rel, wilcoxon

# Importing the processed data

with open( '/kaggle/input/models/processed.pkl', 'rb' ) as file:
    d, train, val, features = pickle.load( file )
    
with open( '/kaggle/input/models/relevant.pkl', 'rb' ) as file:
    relevant = pickle.load( file )

#################### PREPROCESSING ############################################

# Censoring outliers with a custom function
# Current version of the function can be found on https://github.com/d-kulikov/preprocessing
bounds = Outliers.fit( d.loc[ train, relevant ] )

Outliers.transform( d, bounds )

# Imputing missing accuracies
for name in [ 'accuracy_total', 'accuracy_assess', 'accuracy_game', 'accuracy_mushroom', 'accuracy_bird', 'accuracy_cauldron', 
             'accuracy_cart', 'accuracy_chest', 'accuracy_this' ] :
    d.loc[ d[ name ].isnull(), name ] = d.loc[ train, name ].mean()

# Scaling

Scaler = StandardScaler()

Scaler.fit( d.loc[ train, relevant ] )

# Feature arrays

xtrain = np.float32( Scaler.transform( d.loc[ train, relevant ] ) )

xtest = np.float32( Scaler.transform( d.loc[ val, relevant ] ) )

# Target Variable

ytrain = np.float32( d.loc[ train, 'accuracy' ] )

ytest = np.float32( d.loc[ val, 'accuracy' ] )

###### SEARCHING BEST CONFIGURATION OF NEURAL NETWORK #########################
 
# Parameters of training

batch = np.sum( train )

validation = ( xtest, ytest )

stop = EarlyStopping( patience=10 )

# Number of trials (set to a small value for demonstration purposes)
s = 3

# Creating a summary table
np.random.seed( seed=1 )
summary = pd.DataFrame( { 'seed1' : np.random.randint( 0, 4e9, size=s ),
                          'seed2' : np.random.randint( 0, 4e9, size=s ),
                          'seed3' : np.random.randint( 0, 4e9, size=s ) } )

# Trying various configurations (number of epochs set to a small value for demonstration purposes)
    
start = time()

for i in range( 0, s ) :
    print( round( i * 100 / s, 2 ) )
    nnmodel = Sequential()
    nnmodel.add( Dense( 10, input_shape=( len( relevant ), ),
                        kernel_initializer=RandomUniform( minval=-1, maxval=1, seed=summary.loc[ i, 'seed1' ] ),
                        activation='tanh' ) )
    nnmodel.add( Dense( 5, kernel_initializer=RandomUniform( minval=-1, maxval=1, seed=summary.loc[ i, 'seed2' ] ),
                        activation='tanh' ) )
    nnmodel.add( Dense( 1, kernel_initializer=RandomUniform( minval=-1, maxval=1, seed=summary.loc[ i, 'seed3' ] ), 
                        activation='sigmoid' ) )
    nnmodel.compile( optimizer='adam', loss='binary_crossentropy' )
    fitting = nnmodel.fit( xtrain, ytrain, batch_size=batch, validation_data=validation, shuffle=False, epochs=100,
                           callbacks=[ stop ] )
    epochs = pd.DataFrame( fitting.history )
    summary.loc[ i, 'loss' ] = epochs[ 'val_loss' ].min()
    summary.loc[ i, 'epochs' ] = epochs.shape[ 0 ] - 10
    del [ nnmodel, fitting, epochs ]
    collect()

print( round( ( time() - start ) / 60, 2 ) )

############### BEST CONFIGURATION ############################################

stop = EarlyStopping( patience=10, restore_best_weights=True )

nnmodel = Sequential()
nnmodel.add( Dense( 10, input_shape=( len( relevant ), ),
                    kernel_initializer=RandomUniform( minval=-1, maxval=1, seed=1791095845 ), activation='tanh' ) )
nnmodel.add( Dense( 5, kernel_initializer=RandomUniform( minval=-1, maxval=1, seed=550290313 ), activation='tanh' ) )
nnmodel.add( Dense( 1, kernel_initializer=RandomUniform( minval=-1, maxval=1, seed=1013994432 ), activation='sigmoid' ) )
nnmodel.compile( optimizer='adam', loss='binary_crossentropy' )
nnmodel.fit( xtrain, ytrain, batch_size=batch, validation_data=validation, shuffle=False, epochs=100, callbacks=[ stop ] ) 

######## TESTING STATISTICAL SIGNIFICANCE OF THE MODEL ########################

# Test predictions
ypred = nnmodel.predict( xtest )[ :, 0 ]

# Naive benchmark (simple average)
naive = d.loc[ train, 'accuracy' ].mean()

# Calculating crossentropy errors of the model and benchmark

errors_model = -( ytest * np.log( ypred ) + ( 1 - ytest ) * np.log( 1 - ypred ) )

errors_naive = -( ytest * np.log( naive ) + ( 1 - ytest ) * np.log( 1 - naive ) )

print( np.mean( errors_model ) )

print( np.mean( errors_naive ) )

# Performing statistical tests on the errors

print( anderson( errors_model - errors_naive ) )

print( ttest_rel( errors_model, errors_naive )[ 1 ] / 2 )

print( wilcoxon( errors_model, errors_naive )[ 1 ] / 2 )
