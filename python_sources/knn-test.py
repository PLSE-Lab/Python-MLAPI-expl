"""Classify handwritten digits using MNIST data"""

# Load H2O library and start a cluster:
import h2o
h2o.init()
h2o.remove_all() 


# Load training data:
df = h2o.import_file("../input/train.csv")

# Prepare predictors and response columns:
X = df.col_names[1:]
y = df.col_names[0]


# Preprocess Y as a factor:
df[y] = df[y].asfactor()


# ### Deep Learning Model
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

dl = H2ODeepLearningEstimator(activation = "Tanh", hidden = [150, 150, 150], epochs = 20)
dl.train(X, y, training_frame=df, validation_frame=valid)

print ("R2:")
print (dl.r2())

# Load test file:
test = h2o.import_file("../input/test.csv")

# Make predictions:
pred = dl.predict(test)


# Make CSV for submission
import pandas as pd

p_df = pred.as_data_frame()

predictions = pd.DataFrame(p_df[0][1:])
predictions.columns = ['Label']
predictions.Label = predictions.Label.astype(int)
predictions.index = predictions.index + 1
predictions.index.name = 'ImageId'

# Write CSV file:
predictions.to_csv('predictions.csv')

# Shitdown H2O
h2o.shutdown()
