#!/usr/bin/env python
# coding: utf-8

# # Obtain a decent score with just 11 distances and 10 types
# 
# Inspired by the really cool kernel "Distance - is all you need" (https://www.kaggle.com/criskiev/distance-is-all-you-need-lb-1-481), I tried to devise my own system of simple, geomtric features. I started out with the simplest set of distance-features, I could think of.
# 
# ## Steps
# 
# * Calculate the center-point on the axis between the two j-coupled atoms
#   (let's refer to them as atom_0 and and atom_1)
# * Get the 10 closet atoms and rank them by their distance to the center point (from closest to farthest away)
# 
# 
# ## Features
# Now, or each of the 10-nearest neighbor atoms, we have 3 features:
# 
# * atom-type
# * distance from the center-point
#    
# Adding the distance between atom_0 and atom_1 gives at total of 21 features.
# 
#     
# ## Intuition
# 
# These features provide three sources of information:
# 
# * Which atom: H, C, N, O, or F
# * How far away from the j-coupled atoms
# 
# 
# ## Model
# 
# The same LGBM used in the above mentioned kernel.  
# Only more estimators are used for test-set prediction
# 
# ## Limitations
# 
# The decent score given the simple features suggests that geometry and atom type is all you need (it's also all you got ... :-D).
# However, this simple approach suffers from 3 main drawbacks:
# 
# #### Lack of geometric information:
# 
# Distance get's you surprisingly far, but has it's limitations. This features contain no information about whether a neighbor atom is on the side of atom_0, on the side of atom_1 or orthogonal to the atom_0-atom_1 axis. This is obviously a severe limation and it's surprising, the approch even reaches a relativels decent score.
# 
# #### Permutation
#   A small difference in distances can lead to a different order of neighbor-atoms. This way very similar geometries can end up a different feature ordering which impedes learning.
# ... that's why Message Passing Neural Networks or Graph Neural networks are used: the employ permutation invariant aggregation functions, such as $sum$ or $mean$.
#   
# #### No feature structure
# 
# The 21 features above are not simply 21 scalar features. Obviously there is some structure:
# For each atom, the features type and distance belong togheter. However, the model can't account for this structure when it's simply provided a list of features.
# 
# ## Runtime
# 
# It takes forever.  
# The computation was not optimized due to the inherent limitations of the approach explained above :-)

# In[ ]:


get_ipython().run_cell_magic('capture', '', "\nimport os\nfrom os.path import join\nimport sys\nfrom pprint import pprint\n\nimport numpy as np\nfrom numpy.linalg import norm\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nfrom scipy.spatial import distance_matrix\n\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import mean_absolute_error\n\nfrom lightgbm import LGBMRegressor\n\nimport warnings\nwarnings.filterwarnings('ignore')\n\nDATA_DIR = '../input/champs-scalar-coupling'\nATOMIC_NUMBERS = {\n    'H': 1,\n    'C': 6,\n    'N': 7,\n    'O': 8,\n    'F': 9\n}")


# # Load Data

# In[ ]:


structures_dtypes = {
    'molecule_name': 'category',
    'atom_index': 'int8',
    'atom': 'category',
    'x': 'float32',
    'y': 'float32',
    'z': 'float32'
}
structures_df = pd.read_csv(join(DATA_DIR, 'structures.csv'), dtype=structures_dtypes)

structures_df['molecule_index'] = structures_df.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
structures_df['atom'] = structures_df['atom'].replace(ATOMIC_NUMBERS).astype('int8')
structures_df.index = structures_df.molecule_index
structures_df = structures_df[['atom_index', 'atom', 'x', 'y', 'z']]

structures_df[['x', 'y', 'z']] = structures_df[['x', 'y', 'z']] / 10  # puts all distances approx. in range [0, 1]
print(structures_df.shape)
display(structures_df.head())


# In[ ]:


"""
These two functions are from this cool kernel:
https://www.kaggle.com/criskiev/distance-is-all-you-need-lb-1-481
"""

def add_coordinates(base, structures, index):
    df = pd.merge(base, structures, how='inner',
                  left_on=['molecule_index', f'atom_index_{index}'],
                  right_on=['molecule_index', 'atom_index']).drop(['atom_index'], axis=1)
    df = df.rename(columns={
        'atom': f'atom_{index}',
        'x': f'x_{index}',
        'y': f'y_{index}',
        'z': f'z_{index}'
    })
    return df


def add_center(df):
    df['x_c'] = ((df['x_1'] + df['x_0']) * np.float32(0.5))
    df['y_c'] = ((df['y_1'] + df['y_0']) * np.float32(0.5))
    df['z_c'] = ((df['z_1'] + df['z_0']) * np.float32(0.5))


# In[ ]:


def load_j_coupling_csv(file_path: str, train=True, verbose=False):
    train_dtypes = {
        'molecule_name': 'category',
        'atom_index_0': 'int8',
        'atom_index_1': 'int8',
        'type': 'category',
        'scalar_coupling_constant': 'float32'
    }
    df = pd.read_csv(file_path, dtype=train_dtypes)
    df['molecule_index'] = df.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
    
    if train:
        cols = ['id', 'molecule_index', 'atom_index_0', 'atom_index_1', 'type', 'scalar_coupling_constant']
    else: 
        cols = ['id', 'molecule_index', 'atom_index_0', 'atom_index_1', 'type']
    df = df[cols]

    df = add_coordinates(df, structures_df, 0)
    df = add_coordinates(df, structures_df, 1)
    add_center(df)
    
    if verbose:
        print(df.shape)
        display(df.head())
        
    return df


train_df = load_j_coupling_csv(join(DATA_DIR, 'train.csv'), verbose=True)

mol2distance_matrix = structures_df.groupby('molecule_index').apply(
    lambda df: distance_matrix(df[['x','y', 'z']].values, df[['x','y', 'z']].values))


# # Calculate Features

# In[ ]:


def get_knn_features_center(j_coupling: pd.Series,
                            structures=structures_df,
                            mol2dist=mol2distance_matrix,
                            k=10) -> np.array:
    
    center      = j_coupling[['x_c', 'y_c', 'z_c']].values.reshape(1, 3)
    mol_df      = structures.loc[j_coupling.molecule_index]
    coordinates = mol_df[['x','y', 'z']].values
    
    center_distances = distance_matrix(center, coordinates).ravel()
    knn = np.argsort(center_distances)[:(k + 2)]  # atom-indices of KNN-atoms
    
    a_0, a_1 = j_coupling.atom_index_0, j_coupling.atom_index_1
    knn = np.array([x for x in knn if x not in (a_0, a_1)])
    
    distances = center_distances[knn]
    types = mol_df.iloc[knn].atom
    
    distances = np.pad(distances, (0, k - len(distances)), 'constant')
    types =     np.pad(types,     (0, k - len(types)),     'constant')
    
    dist_matrix = mol2distance_matrix[j_coupling.molecule_index]
    d_a0_a1 = dist_matrix[a_0, a_1]

    knn_features = np.concatenate([[d_a0_a1], distances, types])
    return knn_features


# this may take a while...
id2center_knn = {row.id : get_knn_features_center(row) for _, row in train_df.iterrows()}


# # Make Model

# In[ ]:


def make_data(df: pd.DataFrame, id2features: dict, random_state=128, split=True):
    tmp_df = df.copy()
    tmp_df['features'] = tmp_df.id.map(id2features)
    
    X = np.stack(tmp_df.features)
    y = tmp_df.scalar_coupling_constant.values

    if split:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)
        return (X_train, y_train), (X_val, y_val)
    else:
        return X, y

    
# hyper-parameters like in this kernel:
# https://www.kaggle.com/criskiev/distance-is-all-you-need-lb-1-481
LGB_PARAMS = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.2,
        'num_leaves': 128,
        'min_child_samples': 79,
        'max_depth': 9,
        'subsample_freq': 1,
        'subsample': 0.9,
        'bagging_seed': 11,
        'reg_alpha': 0.1,
        'reg_lambda': 0.3,
        'colsample_bytree': 1.0
        }

    
def train_model(train, validation):
    
    X_train, y_train = train
    X_val,   y_val   = validation

    model = LGBMRegressor(**LGB_PARAMS, n_estimators=1500, n_jobs = -1)  # 6000 estimators would be better but take much longer
    model.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='mae',
            verbose=100, early_stopping_rounds=200)

    y_pred    = model.predict(X_val)
    score     = np.log(mean_absolute_error(y_val, y_pred))
    residuals = y_val - y_pred
    
    print(f'competition-metric score: {score}')
    return model, score, residuals


def plot_residuals(residuals: np.array):
    plt.hist(residuals, bins=50)
    plt.title('residual distribution')
    plt.show();


# In[ ]:


# test pipeline with smallest type:

sub_train_df = train_df.query('type == "1JHN"')

(X_train, y_train), (X_val, y_val) = make_data(sub_train_df, id2center_knn)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

model, score, residuals = train_model((X_train, y_train), (X_val, y_val))
plot_residuals(residuals)


# # Cross-validate

# In[ ]:


def cross_validate(df, id2features):
    type2model = {}
    scores = {}

    for type_, type_df in df.groupby('type'):
        print(f'\n\n### {type_}')
        train, validation = make_data(type_df, id2features)
        model, score, residuals = train_model(train, validation)
        type2model[type_] = model
        scores[type_] = score
        plot_residuals(residuals)

    assert len(scores) == len(type2model) == 8
    return  scores


scores = cross_validate(train_df, id2center_knn)


# In[ ]:


print(f'competition-metric: {np.mean(list(scores.values())):.2f}')
print('scores per type:')
pprint(scores, width=1)


# # Predict and submit

# In[ ]:


def make_test_data(df: pd.DataFrame, id2features: dict, random_state=128):
    tmp_df = df.copy()
    tmp_df['features'] = tmp_df.id.map(id2features)
    X = np.stack(tmp_df.features)
    return X


test_df = load_j_coupling_csv(join(DATA_DIR, 'test.csv'), train=False, verbose=True)
id2center_knn_test = {row.id : get_knn_features_center(row) for _, row in test_df.iterrows()}
prediction_df = pd.DataFrame()


for type_ in sorted(train_df.type.unique()):
    print(f'\n### {type_}')
    
    train_type_df = train_df.query('type == @type_')
    X_train, y_train = make_data(train_type_df, id2center_knn, split=False)
    model = LGBMRegressor(**LGB_PARAMS, n_estimators=2000, n_jobs = -1)  # more estimators for test-set
    model.fit(X_train, y_train, eval_metric='mae')
    
    test_type_df = test_df.query('type == @type_')
    X_test = make_test_data(test_type_df, id2center_knn_test)
    y_hat  = model.predict(X_test)
    
    type_pred_df  = pd.DataFrame({'id': test_type_df.id, 'scalar_coupling_constant': y_hat})
    prediction_df = pd.concat([prediction_df, type_pred_df], ignore_index=True)


# In[ ]:


prediction_df = prediction_df.sort_values('id')
prediction_df.to_csv('submission.csv', index=False)


# In[ ]:


assert len(prediction_df) == len(test_df)
print(prediction_df.shape)
display(prediction_df.head())

