# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm 
import sys as sy
import h2o as model


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train.csv") # train set
test = pd.read_csv("../input/test.csv") # test set 


train_ab = pd.get_dummies(train)
train_ab.to_csv("h2o.csv")


test_ab = pd.get_dummies(test)
test_ab.to_csv("h2otest.csv")



# For Train

#numeric_values = train._get_numeric_data() # choose only numeric values
#remove_na = numeric_values.fillna(numeric_values.mean()) # replace values NA with mean
#remove_na.to_csv("h2o.csv")

# For Test

#numeric_values = test._get_numeric_data() # choose only numeric values
#remove_na = numeric_values.fillna(numeric_values.mean()) # replace values NA with mean
#remove_na.to_csv("h2otest.csv")

#seems like a time series problem too-years(2006-2010)
#http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-munging/uploading-data.html
model.init()
df = model.upload_file("h2o.csv")
dft = model.upload_file("h2otest.csv")
y = "SalePrice"
x = df.columns;
x.remove("SalePrice")
from h2o.estimators import H2OGeneralizedLinearEstimator


# Set Attributes for glm

data_glm = H2OGeneralizedLinearEstimator(family="gamma",
                                        link = "log",
                                       alpha=0.01,
                                        Lambda =0.0396)
                                        
# Train glm

data_glm.train(x               =x,
               y               =y,
               training_frame  =df)

#Prediction
               
predictiona = data_glm.predict(dft)

model.download_csv(predictiona,"solution.csv")
               
from h2o.estimators import H2OGradientBoostingEstimator


# Set Attributes for gbm
data_gbm = H2OGradientBoostingEstimator(distribution="gamma",
                                        ntrees = 150,
                                       max_depth=30,
                                   seed=1,nbins=40,learn_rate=0.1,min_rows=40,col_sample_rate=1)
                                        
# Train gbm

data_gbm.train(x               =x,
               y               =y,
               training_frame  =df)

#Prediction
               
predictionb = data_gbm.predict(dft)

model.download_csv(predictionb,"solution2.csv")


from h2o.estimators.deeplearning import H2ODeepLearningEstimator

deeplearning = H2ODeepLearningEstimator(activation="Tanh",epochs=15,seed=1)
                                        
# Train gbm

deeplearning.train(x               =x,
               y               =y,
               training_frame  =df)

#Prediction
               
predictionc = deeplearning.predict(dft)

model.download_csv(predictionc,"solution3.csv")

from h2o.estimators import H2ORandomForestEstimator

randomforest = H2ORandomForestEstimator(ntrees=150,max_depth=25,mtries=30,seed=1,col_sample_rate_per_tree = 0.8)

randomforest.train(x=x,y=y,training_frame=df)

predictiond = randomforest.predict(dft)

model.download_csv(predictiond,"solution4.csv")




