#!/usr/bin/env python
# coding: utf-8

# # Earnings classification example
# 
# **Dataset:** Adult Census Income <br>
# **Source:** https://archive.ics.uci.edu/ml/index.php <br>
# **Goal:** Determine whether a person makes over $50K a year. 
# <br>
# <br>
# <br>
# 
# **Models tested:**
# * Logistic Regression
# * SVM
# * Desicion Tree
# * Random Forest
# * Gradient-Boosted Tree
# * Naive Bayes Classifier
# * Multilayer Perceptron Classifier

# In[ ]:


get_ipython().system('pip install pyspark')


# In[ ]:


from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.ml import classification
from pyspark.ml import evaluation
from pyspark.ml.linalg import Vectors
from pyspark.ml import feature
from pyspark.ml import Pipeline


# In[ ]:


spark = SparkSession.builder.appName('ml_app').master("local[*]").getOrCreate()


# In[ ]:


col_names = ["age", "workclass", "fnlwgt", "education", 
             "education-num", "marital-status", "occupation", 
             "relationship", "race", "sex", "capital-gain", 
             "capital-loss", "hours-per-week", "native-country", 
             "earnings"]


# In[ ]:


# Import data
df = spark.read.csv("../input/adult-census-income/adult.csv", header=False, inferSchema=True, ignoreLeadingWhiteSpace=True)


# In[ ]:


df = df.filter("_c0 != 'age'")
df = df.select(*[f.col(old).alias(new) for old, new in zip(df.columns, col_names)]).drop("fnlwgt").dropna("any")


# In[ ]:


# Replace "?" with None. Credit: Amit Rawat
# def questionmark_as_null(x):
#     return f.when(f.col(x) != "?", f.col(x)).otherwise(None)

# exprs = [questionmark_as_null(x).alias(x) if x in col_names else x for x in df.columns]
# df = df.select(*exprs).dropna("any")


# In[ ]:


# Exemplary record
df.show(1, vertical=True)


# In[ ]:


# Split data to training and evaluation set
df_t, df_e = df.randomSplit([0.7, 0.3], 1990)


# In[ ]:


# One way with RFormula
rf = feature.RFormula(formula="earnings ~ .", featuresCol='featuresRaw')
rfModel = rf.fit(df_t)

df_train = rfModel.transform(df_t)
df_eval = rfModel.transform(df_e)

scaler = feature.StandardScaler(inputCol="featuresRaw", outputCol="features")
scal_mod = scaler.fit(df_train)
df_train = scal_mod.transform(df_train)
df_eval = scal_mod.transform(df_eval)


# In[ ]:


# Second option - pipeline
num_cols = [c for c, t in df_t.dtypes if t != "string"]
categ_cols = [c for c, t in df_t.dtypes if t == "string" and c != "earnings"]
categ_cols_idx = [c + "Idx" for c in categ_cols]
categ_cols_vect = [c + "Vect" for c in categ_cols]

indexer = feature.StringIndexer(inputCol="earnings", outputCol="label")
indexers = [feature.StringIndexer(inputCol=o, outputCol=n).setHandleInvalid("skip") for o, n in zip(categ_cols, categ_cols_idx)]
OHencoder = feature.OneHotEncoderEstimator(inputCols=categ_cols_idx, outputCols=categ_cols_vect)
vectAssembler = feature.VectorAssembler(inputCols = num_cols + categ_cols_vect, outputCol = "featuresRaw")
scaler = feature.StandardScaler(inputCol="featuresRaw", outputCol="features")

pipe = Pipeline(stages=[indexer] + indexers + [OHencoder, vectAssembler, scaler])
pipeModel = pipe.fit(df_t)

df_train = pipeModel.transform(df_t)
df_eval = pipeModel.transform(df_e)


# In[ ]:


df_train = df_train.select("label", "features")
df_eval = df_eval.select("label", "features")

# df_train.cache()
# df_eval.cache()


# In[ ]:


print("Train:")
df_train.groupBy("label").count().show()
print("Eval:")
df_eval.groupBy("label").count().show()


# Logistic Regression

# In[ ]:


lr = classification.LogisticRegression(maxIter=1000)
lrModel = lr.fit(df_train)

lrModel.coefficients


# In[ ]:


lrModel.intercept


# In[ ]:


trainingSummary = lrModel.summary

# FPR: False Positive Rate / TPR: True Posite Rate
trainingSummary.roc.show(120)


# In[ ]:


trainingSummary.roc.toPandas()


# In[ ]:


trainingSummary.pr.show(120)


# In[ ]:


trainingSummary.areaUnderROC


# In[ ]:


trainingSummary.accuracy


# In[ ]:


trainingSummary.predictions.show()


# In[ ]:


# Predict on evaluation set
lrModel.transform(df_eval).show()


# Support Vector Machine

# In[ ]:


svm = classification.LinearSVC(maxIter=1000)
svmModel = svm.fit(df_train)

svmModel.coefficients


# In[ ]:


svmModel.intercept


# In[ ]:


svmModel.transform(df_eval).show()


# Desicion Tree Classifier

# In[ ]:


tree = classification.DecisionTreeClassifier()
treeModel = tree.fit(df_train)

treeModel.depth


# In[ ]:


treeModel.numNodes


# In[ ]:


print(treeModel.toDebugString)


# In[ ]:


treeModel.transform(df_eval).show()


# Random Forest

# In[ ]:


forest = classification.RandomForestClassifier()
forestModel = forest.fit(df_train)

forestModel.featureImportances


# In[ ]:


print(forestModel.toDebugString)


# In[ ]:


forestModel.transform(df_eval).show()


# Gradient-Boosted Trees

# In[ ]:


gbt = classification.GBTClassifier()
gbtModel = gbt.fit(df_train)

gbtModel.featureImportances


# In[ ]:


print(gbtModel.toDebugString)


# In[ ]:


gbtModel.transform(df_eval).show()


# Naive Bayes Classifier

# In[ ]:


bayes = classification.NaiveBayes()
bayesModel = bayes.fit(df_train)

bayesModel.transform(df_eval).show()


# Multilayer perceptron classifier

# In[ ]:


mlp = classification.MultilayerPerceptronClassifier(maxIter=1000, layers=[475,40,2])
mlpModel = mlp.fit(df_train)

mlpModel.layers


# In[ ]:


mlpModel.weights


# In[ ]:


mlpModel.transform(df_eval).show()


# <br>
# # Evaluation

# In[ ]:


models = [(lrModel, "logistic regression"), 
          (svmModel, "svm"), 
          (treeModel, "desicion tree"), 
          (forestModel, "random forest"), 
          (gbtModel, "gradient boost"), 
          (bayesModel, "naive bayes"), 
          (mlpModel, "mlp")]


# In[ ]:


evaluator = evaluation.BinaryClassificationEvaluator()

for model, name in models:
    print(f"AUC of {name}: {evaluator.evaluate(model.transform(df_eval))}")


# Calculate model accuracy

# In[ ]:


def calculate_acc(df, label="label", prediction="prediction"):
    temp = df.select(f.when(df[label] == df[prediction], 1).otherwise(0).alias("same"))
    return temp.select(f.avg("same")).collect()[0][0]

for model, name in models:
    print(f"Accuracy of {name}: {calculate_acc(model.transform(df_eval))}")


# In[ ]:


spark.stop()


# That's it. If you like it, please upvote. Thank you
