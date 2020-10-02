# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
os.system("clear")

# Any results you write to the current directory are saved as output.

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import HashingTF,Tokenizer,IDF
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
from pyspark.ml.feature import VectorAssembler

spark = (SparkSession.builder
                  .appName('Toxic Comment Classification')
                  .enableHiveSupport()
                  .config("spark.executor.memory", "4G")
                  .config("spark.driver.memory","18G")
                  .config("spark.executor.cores","7")
                  .config("spark.python.worker.memory","4G")
                  .config("spark.driver.maxResultSize","0")
                  .config("spark.sql.crossJoin.enabled", "true")
                  .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")
                  .config("spark.default.parallelism","2")
                  .getOrCreate())
                  
spark.sparkContext.setLogLevel('INFO')
df=spark.read.csv("../input/creditcard.csv",header=True)
df.printSchema()
#help(VectorAssembler)

from pyspark.sql.functions import col

for col_name in df.columns[1:-1]+["Class"]:
    df= df.withColumn(col_name,col(col_name).cast("float"))
    
df=df.withColumnRenamed("Class","label")

Vector=VectorAssembler(inputCols=df.columns[1:-1],outputCol="features")
df_tr=Vector.transform(df)
df_tr=df_tr.select(['features','label'])
df_tr.show(3)
df_tr.show(3,truncate=False)
lr=LogisticRegression(maxIter=10,featuresCol="features",labelCol="label")
paramGrid=ParamGridBuilder().addGrid(lr.regParam,[0.1,0.01]) \
                            .addGrid(lr.fitIntercept,[False,True]) \
                            .addGrid(lr.elasticNetParam,[0.0,0.1,1.0]) \
                            .build()
crossVal=CrossValidator(estimator=lr, \
                        estimatorParamMaps=paramGrid, \
                        evaluator=BinaryClassificationEvaluator(),
                        numFolds=2)
cvModel=crossVal.fit(df_tr)
lst = list(cvModel.avgMetrics)
print(lst)
print(df.count())
df_pd=df.toPandas()
print(df_pd.shape[0])
df_pd.columns
df_pd.groupby('label')['Time'].count()
print(df.select('Class').groupby('Class').count())
#help(LogisticRegression)
#dir(lr)
# TODO: USE THE BEST MODEL FROM CROSS VALIDATION AND PREDICT THE DEFAULTER
#dir(cvModel)
best_model_summary= cvModel.bestModel.summary
rc_df= best_model_summary.roc.toPandas()
res=cvModel.bestModel.transform(df_tr)
res.show()
res_df=res.toPandas()
res_df.groupby('prediction')['label'].count()
res_df_main=res_df[['label','prediction']]
rc_df.plot()

from sklearn.metrics import confusion_matrix,accuracy_score,auc,precision_score,recall_score,roc_auc_score
print("ROC AUC Scoer of the model",roc_auc_score(res_df_main['label'],res_df_main['prediction']))
print("Accuracy Score of the model",accuracy_score(res_df_main['label'],res_df_main['prediction']))
print("Confusion Matrix of the model",confusion_matrix(res_df_main['label'],res_df_main['prediction']))
print("Precision Score of the model",precision_score(res_df_main['label'],res_df_main['prediction']))
print("Precision Score of the model",recall_score(res_df_main['label'],res_df_main['prediction']))

extract_prob = F.udf(lambda x: float(x[1]), T.FloatType())
res.printSchema()

df.printSchema()
df.show()
res.show()
df_res.printSchema()
from pyspark.sql.window import Window
w=Window().orderBy(F.lit("A"))
df =df.withColumn('id',F.row_number().over(w))
res=res.withColumn('id',F.row_number().over(w))

lr_res=res_df.to_csv(".../output/result.csv",index=False)
df_pd['prediction']=res_df['prediction']
df_res=df.join(res.select('id','prediction'),on='id')
#df_res=df_res.withColumn(col,extract_prob('probability')).drop('probability')
df_res.printSchema()

df_res.coalesce(1).write.csv('./results/spark_lr.csv', mode='overwrite', header=True)



