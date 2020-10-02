import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Any results you write to the current directory are saved as output.
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import ThresholdedReLU
from keras import metrics
from sklearn.decomposition import PCA

# Importing the dataset
train = pd.read_csv('../input/train.csv')
test_x = pd.read_csv('../input/test.csv')

y_column = 'SalePrice'

# Remove outliers
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

# Split x and y
train_y = train[y_column]
train_x = train.drop([y_column], axis=1)

# Save id col from test set to append it at the end
test_id_col = test_x['Id'].values.tolist()

# Combine test and train set
df_x = pd.concat([train_x, test_x])

# Remove id columns
df_x = df_x.drop(['Id'], axis=1)

# Convert potential numerical  categorical data to numerical data
ordinal = [
    ('ExterQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
    ('ExterCond', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
    ('BsmtQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
    ('BsmtCond',  ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
    ('HeatingQC', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
    ('KitchenQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
    ('FireplaceQu', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
    ('GarageQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
    ('GarageCond', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
    ('PoolQC', ['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1, 2, 3, 4, 5], lambda x: 0),
    ('Street', ['Pave', 'Grvl'], [1, 2], lambda x: 0),
    ('Alley', ['Pave', 'Grvl'], [1, 2], lambda x: 0),
    ('LotShape', ['IR3', 'IR2', 'IR1', 'Reg'], [1, 2, 3, 4], lambda x: 0),
    ('Utilities', ['ELO', 'NoSeWa', 'NoSewr', 'AllPub'], [0, 1, 2, 3], lambda x: 0),
    ('LandSlope', ['Sev', 'Mod', 'Gtl'], [1, 2, 3], lambda x: 0),
    ('BsmtExposure', ['No', 'Mn', 'Av', 'Gd'], [1, 2, 3, 4], lambda x: 0),
    ('BsmtFinType1', ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], [1, 2, 3, 4, 5, 6], lambda x: 0),
    ('BsmtFinType2', ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], [1, 2, 3, 4, 5, 6], lambda x: 0),
    ('GarageFinish', ['Unf', 'RFn', 'Fin'], [1, 2, 3], lambda x: 0),
#    ('PavedDrive', ['N', 'P', 'Y'], [1, 2, 3], lambda x: 0),
    ('Fence', ['MnWw', 'GdWo', 'MnPrv', 'GdPrv'], [1, 2, 3, 4], lambda x: 0)
]


def convert(df, ordinal):
    for ordinalData in ordinal:
        df[ordinalData[0]] = df[ordinalData[0]].fillna(ordinalData[3](df[ordinalData[0]]))
        df[ordinalData[0]] = df[ordinalData[0]].replace(ordinalData[1], ordinalData[2])
    return df


df_x = convert(df_x, ordinal=ordinal)


# Create dummies for object data
def to_dummies(df):
    objects = list(df.select_dtypes(include=['object']))
    return pd.get_dummies(df, columns=objects, drop_first=True)


df_x = to_dummies(df_x)


# Correct missing values
def prepare_missing_values(df):
    # Remove columns with more than 60% missing value,
    df = df.dropna(axis='columns', how='any', thresh=df_x.shape[0] * 0.6, subset=None, inplace=False)
    # Fill missing values with column median
    df = df.fillna(df_x.mean())
    # Remove columns with all the same value
    df = df.drop(df_x.std()[(df_x.std() == 0)].index, axis=1)
    return df


df_x = prepare_missing_values(df_x)


# Scale Data
def scale_data(df):
    from sklearn.preprocessing import StandardScaler
    from sklearn_pandas import DataFrameMapper
    mapper = DataFrameMapper([(df.columns, StandardScaler())])
    scaled_features = mapper.fit_transform(df.copy(), 4)
    df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
    return df


df_x = scale_data(df_x)

# Split train and test set
train_prepared_x = df_x[:train.shape[0]]
test_prepared_x = df_x[train.shape[0]:]

pca = PCA()
pca.fit(train_prepared_x, train_y)

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')  # for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()

variance = 0.99

nr_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= variance)

pca = PCA(n_components=nr_components)
pca.fit(train_prepared_x, train_y)

ann_train_x = pca.transform(train_prepared_x)
ann_test_x = pca.transform(test_prepared_x)

# Set seed for randomization
seed = 7
np.random.seed(seed)

# Create model
input_layer_nodes = ann_train_x.shape[1]
output_layer_nodes = 1


def create_model():
    m = Sequential()
    m.add(Dense(int(input_layer_nodes / 2), input_dim=input_layer_nodes, activation='relu'))
    m.add(Dense(int(input_layer_nodes / 5), activation=ThresholdedReLU(theta=1)))
    m.add(Dense(output_layer_nodes))
    m.compile(optimizer=keras.optimizers.Adadelta(), loss='mean_squared_error', metrics=[metrics.mse])
    return m


model = create_model()
model.summary()

history = model.fit(ann_train_x, train_y, epochs=125, batch_size=10)

prediction = model.predict(ann_test_x)

submission = pd.DataFrame()
submission['Id'] = test_id_col
submission['SalePrice'] = prediction
submission.to_csv('submission.csv', index=False)

print('Finished :-)')