import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from time import time
from gc import collect

# Importing the processed data
with open( '/kaggle/input/models/processed.pkl', 'rb' ) as file:
    d, train, val, features = pickle.load( file )
    
# Imputing missing accuracies
for name in [ 'accuracy_total', 'accuracy_assess', 'accuracy_game', 'accuracy_mushroom', 'accuracy_bird', 'accuracy_cauldron', 
             'accuracy_cart', 'accuracy_chest', 'accuracy_this' ] :
    d.loc[ d[ name ].isnull(), name ] = d[ name ].mean()

# For the purpose of feature selection converting the target variable into binary

ytrain = np.int8( np.where( d.loc[ train, 'accuracy' ] > 0, 1, 0 ) )

ytest = np.int8( np.where( d.loc[ val, 'accuracy' ] > 0, 1, 0 ) )

# Feature arrays

xtrain = np.float32( d.loc[ train, features ] )

xtest = np.float32( d.loc[ val, features ] )

# Number of variables considered at each split
vars = round( len( features ) / 3 )

# Number of trials (set to a small value for demonstration purposes)
s = 30

# Creating a summary table
np.random.seed( seed=1 )
summary = pd.DataFrame( { 'seed' : np.random.randint( 0, 4e9, size=s ),
                          'leaf' : np.random.randint( 30, 886, size=s ) } )

# Trying various combinations of variables and levels of generalisation
    
start = time()

for i in range( 0, s ) :
    print( round( i * 100 / s, 2 ) )
    tree = RandomForestClassifier( n_estimators=1, min_samples_leaf=summary.iloc[ i, 1 ], max_features=vars, criterion='gini',
                bootstrap=True, n_jobs=1, random_state=summary.iloc[ i, 0 ], warm_start=False, class_weight='balanced' )
    tree.fit( xtrain, ytrain )
    summary.loc[ i, 'loss' ] = log_loss( ytest, tree.predict( xtest ) )
    summary.loc[ i, 'vars' ] = np.sum( tree.feature_importances_ > 0 )
    del tree
    collect()

print( round( ( time() - start ) / 60, 2 ) )

# Refitting the best configuration

tree = RandomForestClassifier( n_estimators=1, min_samples_leaf=346, max_features=vars, criterion='gini',
            bootstrap=True, n_jobs=1, random_state=2876537340, warm_start=False, class_weight='balanced' )

tree.fit( xtrain, ytrain )

print( log_loss( ytest, tree.predict( xtest ) ) )

# Saving the list of selected variables

relevant = list( features[ tree.feature_importances_ > 0 ] )

print( relevant )

with open( 'relevant.pkl', 'wb') as file:
    pickle.dump( d, file )