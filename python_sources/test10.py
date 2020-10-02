import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SQLContext

#spark
sc = SparkContext('local','example') 
data = SQLContext(sc)

# util
def createTable(path, table):
    Spark_Full = sc.emptyRDD()
    chunk_100k = pd.read_csv(path, chunksize=100000)
    headers = list(pd.read_csv(path, nrows=0).columns)
    for chunky in chunk_100k:  #help was taken from stackeaxhange.com
        Spark_Full +=  sc.parallelize(chunky.values.tolist())
    data = Spark_Full.toDF(headers)
    data_new = data.fillna(-1)
    data_new.registerTempTable(table)
    data_new.cache()
    
#tables
createTable('../input/test.csv', 'test')
createTable('../input/train.csv', 'train')

#queries
data.sql("SELECT * FROM test LIMIT 100").show()
data.sql("SELECT * FROM train LIMIT 100").show()
#data.sql("SELECT test.*, train.* FROM test RIGHT JOIN train ON test.PassengerId = train.PassengerId").show()



    
    
    
# data.show()
# ddata.printSchema()
# ddata_new.select('Name').show()

# Any results you write to the current directory are saved as output.