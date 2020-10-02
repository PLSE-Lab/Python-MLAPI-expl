try:
    import pyspark
except:
    !pip install pyspark

import os 
import shutil 
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel

conf = SparkConf().setAppName('Recommend')
sc = SparkContext(conf = conf)

rantingsRDD = sc.textFile('../input/u.data').map(lambda line: tuple(line.split('\t')[:3]))
model = ALS.train(rantingsRDD, 10, 10)

# test trained results, recommend 5 products for user id 100
print(model.recommendProducts(100, 5))

# test trained results, recommend 5 users for product id 100
print(model.recommendUsers(100, 5))

# save model
if os.path.isdir('model'): shutil.rmtree('model')
model.save(sc, 'model')

# try if model can be reloaded 
model = MatrixFactorizationModel.load(sc, 'model')
print(model.predict(100, 100))