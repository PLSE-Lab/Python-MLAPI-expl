import numpy as np
import pandas as pd
import types

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())

#linear model

nIterations = 40
negativeFeedBack = 0.5
nTrainSamples = len(train.index)
nColumns = len(train.columns)

train.convert_objects(convert_numeric=True)
parameters =train.iloc[:,2:]

# converting text fields into numeric vectors

#isStringColumn = parameters.applymap(lambda x: type(x) is not int or type(x) is not float).all(0)
isStringColumn = parameters.applymap(lambda x: type(x) is str).any(0)
max_classifier_size = 4

stringClassifiers = {};
for column in parameters.columns[isStringColumn]:
    variables = parameters[column].unique()
    if (len(variables)<=max_classifier_size):
        stringClassifiers[column] = variables

def prettifyData(data, stringClassifiers):
    for column, unique_values in stringClassifiers.items():
        nTrainSamples = len(data.index)
        if (len(unique_values)==2):
            nClassifiers = 1
            thisClassifier = pd.Series(np.zeros(nTrainSamples), index=data.index)
            thisClassifier[data[column] == unique_values[1]] = 1
            newColumnName = '{columnName}_{valueStr}'.format(columnName = column, valueStr = unique_values[1])
            data.loc[:,newColumnName] = thisClassifier
        else:
            nClassifiers = len(unique_values)
            for v in unique_values:
                thisClassifier = pd.Series(np.zeros(nTrainSamples), index=data.index)
                thisClassifier[data[column] == v] = 1
                newColumnName = '{columnName}_{valueStr}'.format(columnName = column, valueStr = v)
                data.loc[:,newColumnName] = thisClassifier
                
    numericData = data.select_dtypes(['float64', 'int64'])
    return numericData

numericData = prettifyData(parameters, stringClassifiers)
normalizationData = {'_min':numericData.min(), '_max':numericData.max(), '_mean':numericData.mean()}
numericData = (numericData - normalizationData['_mean'])/(normalizationData['_max']-normalizationData['_min'])

nNumericColumns = len(numericData.columns)
linearCoefficients = np.zeros(nNumericColumns)
offset = train.iloc[:,1].mean()
normalization = numericData.mul(numericData).sum(0)
#print('Normalization:')
#print(normalization)

for iterationId in range( nIterations ):
    prediction = (numericData.mul(linearCoefficients, 1)).sum(1) + offset
    #print(prediction)
    deviation = train.iloc[:,1]-prediction
    deviation[prediction<0] = 0
    deviation[prediction>1] = 0
    deviation_baseline = deviation.mean()
    deviation = deviation - deviation_baseline
    offset = offset + deviation_baseline
    #print(deviation.mul(deviation).sum())
    #print(deviation)
    backpropagation = numericData.mul(deviation,0).divide(normalization, 1)
    linearCoefficients = (1-negativeFeedBack)*linearCoefficients + negativeFeedBack*backpropagation.sum(0)
    prediction[prediction<0.5] = 0
    prediction[prediction>0.5] = 1
    deviation = train.iloc[:,1]-prediction

print('Finished model fitting with baseline {baseline:4.2f} and coefficients:'.format(baseline=offset))
print(linearCoefficients)
print('Failure rate: {:4.2f}%'.format(deviation.mul(deviation).sum()/nTrainSamples*100))

#applying to test data
testNumericData = prettifyData(test.iloc[:,1:], stringClassifiers)
testNumericData = (testNumericData - normalizationData['_mean'])/(normalizationData['_max']-normalizationData['_min'])
#print(len())
prediction = (testNumericData.mul(linearCoefficients, 1)).sum(1) + offset
prediction[prediction<0.5] = 0
prediction[prediction>0.5] = 1
print(len(prediction))
print(len(test['PassengerId']))
results = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':prediction}).astype('int64')
results.to_csv('results.csv', index=False)
#train.to_csv('copy_of_the_training_data.csv', index=False)