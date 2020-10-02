'''
@brief: DNNRegressor for predicting of house price
@author: Shylock Hg
@time: 2017/9/1
@email: tcath2s@icloud.com
'''

import numpy as np
import tensorflow as tf
import csv
import pandas as pd

#log opts
tf.logging.set_verbosity(tf.logging.INFO)

#data path
TEST_FILE = '../input/test.csv'
TRAIN_FILE = '../input/train.csv'

#read colums name
def read_columns(file):
    with open(file) as f:
        reader = csv.reader(f)
        return next(reader)

COLUMNS = read_columns(TRAIN_FILE)
FEATURES = COLUMNS[1:-1]  #eliminate 'Id' 'Price'
TARGET = COLUMNS[-1] #'Price'

#feature colums
NUMERIC_FEATURES = ['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond',
'YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath',
'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd',
'Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold']
NONUMERIC_FEATURES = [item for item in FEATURES if item not in NUMERIC_FEATURES]
FEATURES = NUMERIC_FEATURES + NONUMERIC_FEATURES  #resort
numeric_cols = [tf.feature_column.numeric_column(k) for k in NUMERIC_FEATURES]
nonumeric_cols = [tf.feature_column.categorical_column_with_hash_bucket(k,10000) for k in NONUMERIC_FEATURES]
nonumeric_cols = [tf.feature_column.embedding_column(nonumeric_cols[k],10) for k in range(len(nonumeric_cols))]
cols = numeric_cols + nonumeric_cols

#visualization
#print('FEATURES:{}'.format(FEATURES))
#print('CLOS:{}'.format(cols))

#train input_fn
def get_train_input_fn(file):
    df = pd.read_csv(file)
    TRAIN_SIZE = (len(df)//10)*8
    train_set = df.iloc[:TRAIN_SIZE]
    eval_set = df.iloc[TRAIN_SIZE:]
    return (tf.estimator.inputs.pandas_input_fn(  #trian_input_fn
                x= pd.DataFrame({k:train_set[k].values for k in FEATURES}), #reserve features colums
                y= pd.Series(train_set[TARGET].values), #target column
                num_epochs=None,
                shuffle=True
            ),
            tf.estimator.inputs.pandas_input_fn(  #eval_input_fn
                x=pd.DataFrame({k:eval_set[k].values for k in FEATURES}),
                y=pd.Series(eval_set[TARGET].values),
                num_epochs=1,
                shuffle=False
            )
        )

#predict input_fn
def get_predict_input_fn(file):
    df = pd.read_csv(file)
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k:df[k].values for k in FEATURES}),
        num_epochs=1,
        shuffle=False
    )


HIDDEN_UNITS = [1024,256,64,16]

def train():
    DNNRegressor = tf.estimator.DNNRegressor(
        feature_columns=cols,
        hidden_units=HIDDEN_UNITS,
        model_dir='./tmp/model'
        #dropout=True
    )

    #input_fn
    train_input_fn,eval_input_fn = get_train_input_fn(TRAIN_FILE)

    while True:
        DNNRegressor.train(input_fn=train_input_fn,steps=100)
        if DNNRegressor.evaluate(input_fn=eval_input_fn)['accuracy'] > 0.90:
            break

def predict():
    DNNRegressor = tf.estimator.DNNRegressor(
        feature_columns=cols,
        hidden_units=HIDDEN_UNITS,
        model_dir='./tmp/model',
    )

    #input_fn
    predict_input_fn = get_predict_input_fn(TEST_FILE)

    predictions = DNNRegressor.predict(predict_input_fn)

    #output predictions to .csv file as 'Id,SalePrice' format
    with open('predictions.csv','w') as f:
        f.write('Id,SalePrice\n')
        for i,v in enumerate(predictions):
            f.write(str(i)+','+str(v)+'\n')


def main(unused_argv):
    train()
    predict()

if __name__ == '__main__':
    tf.app.run()
