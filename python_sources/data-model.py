from time import time
tr0 = time()
import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import float32 as f32
from tflearn import variable as tflVar
import numpy as np

print("import time:", round(time()-tr0,5),"s")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pathData = "../input/"
model = pd.read_csv(pathData + "tidytitanic/Model.csv",index_col=0, na_values=[] )
model = model.fillna("")
model.loc[model['Fare']=='', 'Fare']=0
model.loc[model['lastname2']=='', 'lastname2']=0
model['Fare'] = pd.to_numeric(model['Fare'])
model['lastname2'] = pd.to_numeric(model['lastname2'])
# group by discriminant type
xd_sex = ['Sex']
xd_cae = ['Pclass','Age','Embarked']
xd_fdn = ['Fare','num_dep_rel','num_indep_rel']
xd_tln = ['TicketPrefix','lastname1','lastname2']
xd_lab = ['Survived']
labels = ['survived', 'perished']

# reorder colunms by discriminant type
model = model[xd_sex+xd_cae+xd_fdn+xd_tln+['Survived','set', 'PassengerId']] 

def begin( feature ):#find column index of feature
    return ( model.columns==feature ).nonzero()[0][0]
def xd_range( g ):
    start = begin( g[0] )
    return ( start, start + len( globals()['xd_'+ g[1] ] ) )# find coulmn range of discriminant set

sex_begin,sex_end = xd_range(['Sex','sex'])#(0,1)
cae_begin,cae_end = xd_range(['Pclass','cae'])#(1,4)
fdn_begin,fdn_end = xd_range(['Fare','fdn'])#(4,7)
tln_begin,tln_end = xd_range(['TicketPrefix','tln'])#(7,10)

OBSERVATION_STRUCT = {'sex':(xd_sex,sex_begin,sex_end),
                      'cae':(xd_cae,cae_begin,cae_end),
                      'fdn':(xd_fdn,fdn_begin,fdn_end),
                      'tln':(xd_tln,tln_begin,tln_end),
                      'lab':(xd_lab,labels),
                      }

train_model_set = model.loc[model['set'] == 'train' ]

# OBSERVATION_SIZE:num observations, by num of features:

NUM_CHANNELS = 1
NUM_CLASSES = len(labels)

def loaddata(SAMPLE_SIZE):
    """
    Make training, validation and test splits.
    Creates trainable variables of 'cut-offs' for features with contiuous values
    """
    test_model = model.loc[ model[ 'set' ] == 'test' ].drop( [ 'Survived', 'set' ], axis=1 ) 
    train_model, valid_model, train_labels, valid_labels = train_test_split( 
        train_model_set.drop([ 'Survived','set' ], axis=1),
        train_model_set['Survived'],
        test_size=SAMPLE_SIZE,#test_model.shape[0],
        random_state=42)
    def demarcation_of_data(labels, test_model, valid_model, train_model, train_model_set):

        num_cutsd = 2#tf.get_variable(initial_value=2, trainable=True, name='Num_Cuts')
        num_cutsd_ctn = 1#tf.constant(value=1, name='Num_Cuts_Continuous')
        num_cutsj = 0#tf.constant(value=0, name='Num_Classes')
        _Cuts_ = []
        def class_sep_strtg( num_cuts, xd, train=True, model_set=train_model_set ):
            """
            This function takes, num_cuts, an integer, and creates either tensorflow variables  or tensorflow constants.  
            A  Boolean, train,  decides variable or constant.
            constant => cut values are not trained
            variable => cut values are trained
            k [list]
            [ unique values ] num_cuts = 0 => count of unique elements => each unique element is counted as a group. 
            [ continuous values ] num_cuts = 1 => count all unique elements as one group.
            [ cut off values ] num_cuts =>  integer, split into quantiles based on num_cuts
            holds the values of the cuts.
            """
            def if_is_not_num(s):
                try: 
                    list( map( float, s ) )
                    return False
                except:
                    return True
            nonlocal _Cuts_
            k_ = []
            m = []
            for k, ( n, c ) in zip( num_cuts, model_set[xd].iteritems() ):
                if k == num_cutsj:
                    components = c.unique()
                    m.append( if_is_not_num( components ) )
                    k_.append( components )
                elif k == num_cutsd_ctn:
                    k_.append( n )
                    m.append( False )
                else:
                    _k_ = []
                    for i in range(1,k+1):
                        brake = i*(k+1)*.1
                        with tf.variable_scope('estimate', reuse=tf.AUTO_REUSE ):
                            _Cuts_.append( tflVar(name=n+'_Cut_estimate'+str( np.quantile( c, brake ) ),
                                                    shape=None,
                                                    dtype=f32,
                                                    initializer=tf.cast( np.quantile( c, brake ), f32 ),
                                                    regularizer='L2',
                                                    trainable=True,
                                                    collections=None,
                                                    caching_device=None,
                                                    validate_shape=True,
                                                    device=None,
                                                    restore=True) )
                        _k_.append( n+'_Cut_estimate'+str( np.quantile( c, brake ) ) )
                    k_.append( _k_ )
                    m.append( False )
            #k = [0,1,2]
            return k_, m
        cae_cuts = [ num_cutsj, num_cutsd , num_cutsj ]
        fdn_cuts = [ num_cutsd_ctn,num_cutsd_ctn,num_cutsd_ctn ]
        sex_cuts = [ num_cutsj ]
        tln_cuts = [ num_cutsj, num_cutsj, num_cutsj ]
        lab_cuts = [ num_cutsj ]
        kd_cae, cae_map_to_num = class_sep_strtg( cae_cuts, xd_cae )
        kd_fdn, fdn_map_to_num = class_sep_strtg( fdn_cuts, xd_fdn )
        kd_sex, sex_map_to_num = class_sep_strtg( sex_cuts, xd_sex )
        kd_tln, tln_map_to_num = class_sep_strtg( tln_cuts, xd_tln )
        kj, lab_map_to_num = class_sep_strtg( lab_cuts, xd_lab )
        d_cae = { 'features': xd_cae,'k': kd_cae,'componentMapLogic': cae_map_to_num }
        d_sex = { 'features': xd_sex,'k': kd_sex,'componentMapLogic': sex_map_to_num }
        d_fdn = { 'features': xd_fdn,'k': kd_fdn,'componentMapLogic': fdn_map_to_num }
        d_tln = { 'features': xd_tln,'k': kd_tln,'componentMapLogic': tln_map_to_num }
        factorables = [ d_cae, d_fdn, d_sex ]
        j = {'labels': labels,'k': kj[0] }
    
        def factor( df ):# by all rights should've been done in R
            for feat_group in factorables:
                for feature, needToMap, components in zip( feat_group[ 'features'], feat_group[ 'componentMapLogic'], feat_group[ 'k'] ):
                    feat_map = dict( zip( components, range( 1, len( components )+1 ) ) ) # map feature componenents to sequence of intergers
                    if needToMap:
                        df[ feature ] = df[ feature ].replace( feat_map )
            return df
        train_model = factor( train_model )
        test_model = factor( test_model )
        valid_model = factor( valid_model )
        return test_model, train_model, valid_model, d_tln, d_fdn, d_sex, d_cae, j

    test_model, train_model, valid_model, d_tln, d_fdn, d_sex, d_cae, j = demarcation_of_data( 
        labels, 
        test_model, 
        valid_model, 
        train_model, 
        train_model_set )

    test_labels = pd.get_dummies( model.loc[ model[ 'set' ] == 'test' ][ 'Survived' ] )
    valid_labels = pd.get_dummies( valid_labels )
    train_labels = pd.get_dummies( train_labels )

    data_model = {'train':(train_model,train_labels),
                  'valid':(valid_model,valid_labels),
                  'test' :(test_model,test_labels),
                  'maps' :(d_tln, d_fdn, d_sex, d_cae, j),
                  'train_set':train_model_set,
                  }
    return data_model


def data_metrics():
    return (NUM_CHANNELS,NUM_CLASSES,pathData,
            OBSERVATION_STRUCT,loaddata)