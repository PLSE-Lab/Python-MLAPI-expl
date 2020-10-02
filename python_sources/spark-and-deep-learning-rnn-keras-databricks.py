#!/usr/bin/env python
# coding: utf-8

# # Store Item Demand Forecasting Challenge - Spark and deep learning
# ### link for the github repository (spark part is on ipynb): https://github.com/dimitreOliveira/StoreItemDemand

# In[ ]:


# this code get the resulting dataset that i uploaded from my databricks code and commits.
import pandas as pd
import os


submission25 = pd.read_csv('../input/test_data/model25.csv')
submission25.to_csv('submission25.csv', index=False)


# In[ ]:


from pyspark.sql import Window
from pyspark.ml import Pipeline
from pyspark.sql.types import *
from pyspark.sql import types as T
from pyspark.sql.functions import udf
from pyspark.sql import functions as F
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol

days = lambda i: i * 86400
get_weekday = udf(lambda x: x.weekday())
serie_has_null = F.udf(lambda x: reduce((lambda x, y: x and y), x))


# In[ ]:


import json
import numpy as np
from keras.models import model_from_json
unlist = lambda x: [float(i[0]) for i in x]


# In[ ]:


def prepare_data(data):
    list_result = []
    for i in range(len(data)):
        list_result.append(np.asarray(data[i]))
    return np.asarray(list_result)

def prepare_collected_data(data):
    list_features = []
    list_labels = []
    for i in range(len(data)):
        list_features.append(np.asarray(data[i][0]))
        list_labels.append(data[i][1])
    return np.asarray(list_features), np.asarray(list_labels)

def prepare_collected_data_test(data):
    list_features = []
    for i in range(len(data)):
        list_features.append(np.asarray(data[i][0]))
    return np.asarray(list_features)


def save_model(model_path, weights_path, model):
    """
    Save model.
    """
    np.save(weights_path, model.get_weights())
    with open(model_path, 'w') as f:
        json.dump(model.to_json(), f)
    
def load_model(model_path, weights_path):
    """
    Load model.
    """
    with open(model_path, 'r') as f:
        data = json.load(f)

    model = model_from_json(data)
    weights = np.load(weights_path)
    model.set_weights(weights)

    return model


# In[ ]:


class DateConverter(Transformer):
    def __init__(self, inputCol, outputCol):
        self.inputCol = inputCol
        self.outputCol = outputCol
    
    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != TimestampType()):
        raise Exception('Input type %s did not match input type TimestampType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, df.date.cast(self.inputCol))
    
    
class DayExtractor(Transformer):
    def __init__(self, inputCol, outputCol='day'):
        self.inputCol = inputCol
        self.outputCol = outputCol
    
    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != DateType()):
            raise Exception('DayExtractor input type %s did not match input type DateType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.dayofmonth(df[self.inputCol]))
    
    
class MonthExtractor(Transformer):
    def __init__(self, inputCol, outputCol='month'):
        self.inputCol = inputCol
        self.outputCol = outputCol
    
    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != DateType()):
            raise Exception('MonthExtractor input type %s did not match input type DateType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.month(df[self.inputCol]))
    
    
class YearExtractor(Transformer):
    def __init__(self, inputCol, outputCol='year'):
        self.inputCol = inputCol
        self.outputCol = outputCol
    
    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != DateType()):
            raise Exception('YearExtractor input type %s did not match input type DateType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.year(df[self.inputCol]))
    
    
class WeekDayExtractor(Transformer):
    def __init__(self, inputCol, outputCol='weekday'):
        self.inputCol = inputCol
        self.outputCol = outputCol
    
    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != DateType()):
            raise Exception('WeekDayExtractor input type %s did not match input type DateType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, get_weekday(df[self.inputCol]).cast('int'))
    
    
class WeekendExtractor(Transformer):
    def __init__(self, inputCol='weekday', outputCol='weekend'):
        self.inputCol = inputCol
        self.outputCol = outputCol
    
    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != IntegerType()):
            raise Exception('WeekendExtractor input type %s did not match input type IntegerType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.when(((df[self.inputCol] == 5) | (df[self.inputCol] == 6)), 1).otherwise(0))
    
    
class SerieMaker(Transformer):
    def __init__(self, inputCol='scaledFeatures', outputCol='serie', dateCol='date', idCol=['store', 'item'], serieSize=30):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.dateCol = dateCol
        self.serieSize = serieSize
        self.idCol = idCol

    def _transform(self, df):
        window = Window.partitionBy(self.idCol).orderBy(self.dateCol)
        series = []   
        
    df = df.withColumn('filled_serie', F.lit(0))
    
    for index in reversed(range(0, self.serieSize)):
        window2 = Window.partitionBy(self.idCol).orderBy(self.dateCol).rowsBetween((30 - index), 30)
        col_name = (self.outputCol + '%s' % index)
        series.append(col_name)
        df = df.withColumn(col_name, F.when(F.isnull(F.lag(F.col(self.inputCol), index).over(window)), F.first(F.col(self.inputCol), ignorenulls=True).over(window2)).otherwise(F.lag(F.col(self.inputCol), index).over(window)))
        df = df.withColumn('filled_serie', F.when(F.isnull(F.lag(F.col(self.inputCol), index).over(window)), (F.col('filled_serie') + 1)).otherwise(F.col('filled_serie')))

    df = df.withColumn('rank', F.rank().over(window))
    df = df.withColumn(self.outputCol, F.array(*series))
    
    return df.drop(*series)


class MonthBeginExtractor(Transformer):
    def __init__(self, inputCol='day', outputCol='monthbegin'):
        self.inputCol = inputCol
        self.outputCol = outputCol
    
    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != IntegerType()):
            raise Exception('MonthBeginExtractor input type %s did not match input type IntegerType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.when((df[self.inputCol] <= 7), 1).otherwise(0))
    
    
class MonthEndExtractor(Transformer):
    def __init__(self, inputCol='day', outputCol='monthend'):
        self.inputCol = inputCol
        self.outputCol = outputCol
    
    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != IntegerType()):
            raise Exception('MonthEndExtractor input type %s did not match input type IntegerType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.when((df[self.inputCol] >= 24), 1).otherwise(0))
    
    
class YearQuarterExtractor(Transformer):
    def __init__(self, inputCol='month', outputCol='yearquarter'):
        self.inputCol = inputCol
        self.outputCol = outputCol
    
    def check_input_type(self, schema):
        field = schema[self.inputCol]
        if (field.dataType != IntegerType()):
            raise Exception('YearQuarterExtractor input type %s did not match input type IntegerType' % field.dataType)

    def _transform(self, df):
        self.check_input_type(df.schema)
        return df.withColumn(self.outputCol, F.when((df[self.inputCol] <= 3), 0)
                               .otherwise(F.when((df[self.inputCol] <= 6), 1)
                                .otherwise(F.when((df[self.inputCol] <= 9), 2)
                                 .otherwise(3))))


# In[ ]:


from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler

train_data = spark.sql("select * from store_item_demand_train_csv")

train, validation = train_data.randomSplit([0.8,0.2], seed=1234)


# In[ ]:


# Feature extraction
dc = DateConverter(inputCol='date', outputCol='dateFormated')
dex = DayExtractor(inputCol='dateFormated')
mex = MonthExtractor(inputCol='dateFormated')
yex = YearExtractor(inputCol='dateFormated')
wdex = WeekDayExtractor(inputCol='dateFormated')
wex = WeekendExtractor()
mbex = MonthBeginExtractor()
meex = MonthEndExtractor()
yqex = YearQuarterExtractor()

# Data process
va = VectorAssembler(inputCols=['store', 'item', 'day', 'month', 'year', 'weekday', 'weekend', 'monthbegin', 'monthend', 'yearquarter'], outputCol="features")
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

# Serialize data
sm = SerieMaker(inputCol='scaledFeatures', dateCol='date', idCol=['store', 'item'], serieSize=15)

pipeline = Pipeline(stages=[dc, dex, mex, yex, wdex, wex, mbex, meex, yqex, va, scaler, sm])


# In[ ]:


pipiline_model = pipeline.fit(train)

train_transformed = pipiline_model.transform(train)
validation_transformed = pipiline_model.transform(validation)

train_transformed.write.saveAsTable('train_transformed_15', mode='overwrite')
validation_transformed.write.saveAsTable('validation_transformed_15', mode='overwrite')


# In[ ]:


test_data = spark.sql("select * from store_item_demand_test_csv")
test_transformed = pipiline_model.transform(test_data)
test_transformed.write.saveAsTable('test_transformed_15', mode='overwrite')


# In[ ]:


from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from pyspark.ml.evaluation import RegressionEvaluator

train_transformed = spark.sql("select * from train_transformed")
validation_transformed = spark.sql("select * from validation_transformed")

train_x, train_y = prepare_collected_data(train_transformed.select('serie', 'sales').collect())
validation_x, validation_y = prepare_collected_data(validation_transformed.select('serie', 'sales').collect())

n_label = 1
serie_size = len(train_x[0])
n_features = len(train_x[0][0])


# In[ ]:


# hyperparameters
epochs = 80
batch = 512
lr = 0.001

# design network
model = Sequential()
model.add(GRU(40, input_shape=(serie_size, n_features)))
model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
model.add(Dense(n_label))
model.summary()

adam = optimizers.Adam(lr)
model.compile(loss='mae', optimizer=adam, metrics=['mse', 'msle'])

history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch, validation_data=(validation_x, validation_y), verbose=2, shuffle=False)


# In[ ]:


model_path = '/dbfs/user/model1.json'
weights_path = '/dbfs/user/weights1.npy'
save_model(model_path, weights_path, model)

predictions = model.predict(validation_x)

import pandas as pd
ids = validation_y
df = pd.DataFrame(ids, columns=['label'])
df['sales'] = predictions
df_predictions = spark.createDataFrame(df)

rmse_evaluator = RegressionEvaluator(labelCol="label", predictionCol="sales", metricName="rmse")
mse_evaluator = RegressionEvaluator(labelCol="label", predictionCol="sales", metricName="mse")
mae_evaluator = RegressionEvaluator(labelCol="label", predictionCol="sales", metricName="mae")

validation_rmse = rmse_evaluator.evaluate(df_predictions)
validation_mse = mse_evaluator.evaluate(df_predictions)
validation_mae = mae_evaluator.evaluate(df_predictions)
print("RMSE: %f, MSE: %f, MAE: %f" % (validation_rmse, validation_mse, validation_mae))


# In[ ]:


model_path = '/dbfs/user/model1.json'
weights_path = '/dbfs/user/weights1.npy'
model = load_model(model_path, weights_path)

test_transformed = spark.sql("select * from test_transformed_15")

test = prepare_collected_data_test(test_transformed.select('serie').collect())

ids = test_transformed.select('id').collect()

predictions = model.predict(test)

import pandas as pd
df = pd.DataFrame(ids, columns=['id'])
df['sales'] = predictions
df_predictions = spark.createDataFrame(df)

df_predictions = df_predictions.withColumn('sales', df_predictions['sales'].cast('int'))
display(df_predictions)

