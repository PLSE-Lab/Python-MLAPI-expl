#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import mean,col,split, col, regexp_extract, when, lit
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer


# ### Spark Session

# In[ ]:


spark = SparkSession     .builder     .appName("Spark ML example on titanic data ")     .getOrCreate()


# ### Data and basic EDA

# In[ ]:


titanic_df = spark.read.csv('../input/train.csv',header = 'True',inferSchema='True')


# In[ ]:


display(titanic_df)


# In[ ]:


titanic_df.printSchema()


# In[ ]:


passengers_count = titanic_df.count()


# In[ ]:


print(passengers_count)


# In[ ]:


titanic_df.show(5)


# In[ ]:


titanic_df.describe().show()


# In[ ]:


titanic_df.printSchema()


# In[ ]:


titanic_df.select("Survived","Pclass","Embarked").show()


# In[ ]:


titanic_df.groupBy("Survived").count().show()


# In[ ]:


gropuBy_output = titanic_df.groupBy("Survived").count()


# In[ ]:


display(gropuBy_output)


# In[ ]:


titanic_df.groupBy("Sex","Survived").count().show()


# In[ ]:


titanic_df.groupBy("Pclass","Survived").count().show()


# In[ ]:


# This function use to print feature with null values and null count 
def null_value_count(df):
  null_columns_counts = []
  numRows = df.count()
  for k in df.columns:
    nullRows = df.where(col(k).isNull()).count()
    if(nullRows > 0):
      temp = k,nullRows
      null_columns_counts.append(temp)
  return(null_columns_counts)


# In[ ]:


null_columns_count_list = null_value_count(titanic_df)


# In[ ]:


spark.createDataFrame(null_columns_count_list, ['Column_With_Null_Value', 'Null_Values_Count']).show()


# In[ ]:


mean_age = titanic_df.select(mean('Age')).collect()[0][0]
print(mean_age)


# In[ ]:


titanic_df.select("Name").show()


# In[ ]:


titanic_df = titanic_df.withColumn("Initial",regexp_extract(col("Name"),"([A-Za-z]+)\.",1))


# In[ ]:


titanic_df.show()


# In[ ]:


titanic_df.select("Initial").distinct().show()


# In[ ]:


# Replacing initials with Mr, Miss, Mrs, etc
titanic_df = titanic_df.replace(['Mlle','Mme', 'Ms', 'Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
               ['Miss','Miss','Miss','Mr','Mr',  'Mrs',  'Mrs',  'Other',  'Other','Other','Mr','Mr','Mr'])


# In[ ]:


titanic_df.select("Initial").distinct().show()


# In[ ]:


titanic_df.groupby('Initial').avg('Age').collect()


# In[ ]:


titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Miss") & (titanic_df["Age"].isNull()), 22).otherwise(titanic_df["Age"]))
titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Other") & (titanic_df["Age"].isNull()), 46).otherwise(titanic_df["Age"]))
titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Master") & (titanic_df["Age"].isNull()), 5).otherwise(titanic_df["Age"]))
titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Mr") & (titanic_df["Age"].isNull()), 33).otherwise(titanic_df["Age"]))
titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Mrs") & (titanic_df["Age"].isNull()), 36).otherwise(titanic_df["Age"]))


# In[ ]:


titanic_df.filter(titanic_df.Age==46).select("Initial").show()


# In[ ]:


titanic_df.select("Age").show()


# In[ ]:


titanic_df.groupBy("Embarked").count().show()


# In[ ]:


titanic_df = titanic_df.na.fill({"Embarked" : 'S'})


# In[ ]:


titanic_df = titanic_df.drop("Cabin")


# In[ ]:


titanic_df.printSchema()


# In[ ]:


titanic_df = titanic_df.withColumn("Family_Size",col('SibSp')+col('Parch'))


# In[ ]:


titanic_df.groupBy("Family_Size").count().show()


# In[ ]:


titanic_df = titanic_df.withColumn('Alone',lit(0))


# In[ ]:


titanic_df = titanic_df.withColumn("Alone",when(titanic_df["Family_Size"] == 0, 1).otherwise(titanic_df["Alone"]))


# In[ ]:


titanic_df.columns


# In[ ]:


indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(titanic_df) for column in ["Sex","Embarked","Initial"]]
pipeline = Pipeline(stages=indexers)
titanic_df = pipeline.fit(titanic_df).transform(titanic_df)


# In[ ]:


titanic_df.show()


# In[ ]:


titanic_df.printSchema()


# In[ ]:


titanic_df = titanic_df.drop("PassengerId","Name","Ticket","Cabin","Embarked","Sex","Initial")  # drop columns which are not required


# In[ ]:


titanic_df.show()


# In[ ]:


feature = VectorAssembler(inputCols=titanic_df.columns[1:],outputCol="features")   #vectorizing remaining features
feature_vector= feature.transform(titanic_df)


# In[ ]:


feature_vector.show()


# ### Splitting Data in Train and Test

# In[ ]:


(trainingData, testData) = feature_vector.randomSplit([0.8, 0.2],seed = 11)   #train, test split


# ## Logistic Regression

# In[ ]:


from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol="Survived", featuresCol="features")
#Training algo
lrModel = lr.fit(trainingData)
lr_prediction = lrModel.transform(testData)
lr_prediction.select("prediction", "Survived", "features").show()
evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")


# In[ ]:


lr_accuracy = evaluator.evaluate(lr_prediction)
print("Accuracy of LogisticRegression is = %g"% (lr_accuracy))
print("Test Error of LogisticRegression = %g " % (1.0 - lr_accuracy))


# ## Decision Tree Classifier

# In[ ]:


from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="Survived", featuresCol="features")
dt_model = dt.fit(trainingData)
dt_prediction = dt_model.transform(testData)
dt_prediction.select("prediction", "Survived", "features").show()


# In[ ]:


dt_accuracy = evaluator.evaluate(dt_prediction)
print("Accuracy of DecisionTreeClassifier is = %g"% (dt_accuracy))
print("Test Error of DecisionTreeClassifier = %g " % (1.0 - dt_accuracy))


# ## Random Forest

# In[ ]:


from pyspark.ml.classification import RandomForestClassifier
rf = DecisionTreeClassifier(labelCol="Survived", featuresCol="features")
rf_model = rf.fit(trainingData)
rf_prediction = rf_model.transform(testData)
rf_prediction.select("prediction", "Survived", "features").show()


# In[ ]:


rf_accuracy = evaluator.evaluate(rf_prediction)
print("Accuracy of RandomForestClassifier is = %g"% (rf_accuracy))
print("Test Error of RandomForestClassifier  = %g " % (1.0 - rf_accuracy))


# ## Gradient Boosted Classifier

# In[ ]:


from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(labelCol="Survived", featuresCol="features",maxIter=10)
gbt_model = gbt.fit(trainingData)
gbt_prediction = gbt_model.transform(testData)
gbt_prediction.select("prediction", "Survived", "features").show()


# In[ ]:


gbt_accuracy = evaluator.evaluate(gbt_prediction)
print("Accuracy of Gradient-boosted tree classifie is = %g"% (gbt_accuracy))
print("Test Error of Gradient-boosted tree classifie %g"% (1.0 - gbt_accuracy))


# ## Naive Bayes

# In[ ]:


from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes(labelCol="Survived", featuresCol="features")
nb_model = nb.fit(trainingData)
nb_prediction = nb_model.transform(testData)
nb_prediction.select("prediction", "Survived", "features").show()


# In[ ]:


nb_accuracy = evaluator.evaluate(nb_prediction)
print("Accuracy of NaiveBayes is  = %g"% (nb_accuracy))
print("Test Error of NaiveBayes  = %g " % (1.0 - nb_accuracy))


# ## Linear SVC

# In[ ]:


from pyspark.ml.classification import LinearSVC
svm = LinearSVC(labelCol="Survived", featuresCol="features")
svm_model = svm.fit(trainingData)
svm_prediction = svm_model.transform(testData)
svm_prediction.select("prediction", "Survived", "features").show()


# In[ ]:


svm_accuracy = evaluator.evaluate(svm_prediction)
print("Accuracy of Support Vector Machine is = %g"% (svm_accuracy))
print("Test Error of Support Vector Machine = %g " % (1.0 - svm_accuracy))

