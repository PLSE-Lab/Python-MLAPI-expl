# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures

tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph()

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_df.Pclass = train_df.Pclass.astype('str')

cat_list = ["Pclass","Name","Sex","Ticket","Cabin","Embarked"]
drop_cols = ["Name","Ticket", "Cabin"]
train_df.drop(drop_cols, axis=1, inplace=True)
test_df.drop(drop_cols, axis=1, inplace=True)

print(train_df)
print(train_df.columns)

train_y = train_df.Survived
test_id = test_df.PassengerId

train_X = train_df.drop(["PassengerId"], axis=1)
test_X = test_df.drop(["PassengerId"], axis=1)
train_X = train_X.drop(["Survived"], axis=1)

num_cols = ["Age","SibSp","Parch","Fare"]
cat_cols = ["Sex","Cabin","Embarked","Pclass"]

num_imputer = SimpleImputer()
train_X[num_cols] = num_imputer.fit_transform(train_X[num_cols])
test_X[num_cols] = num_imputer.fit_transform(test_X[num_cols])

train_X = pd.get_dummies(train_X, columns = ["Sex", "Pclass", "Embarked"])
test_X = pd.get_dummies(test_X, columns = ["Sex", "Pclass", "Embarked"])

cols = ['Age','Fare']
scaler  =StandardScaler()
train_X[cols] =scaler.fit_transform(train_X[cols])
test_X[cols] =scaler.fit_transform(test_X[cols])

#print(train_X.head())

#polyf = PolynomialFeatures() #interaction_only=True)
#train_X['Age','Fare'] = polyf.fit_transform(train_X['Age','Fare'])
#test_X['Age','Fare'] = polyf.fit_transform(test_X['Age','Fare'])

#column_to_normalize = ['Age','Fare']
#train_X['Age', 'Fare'] = train_X['Age','Fare'].apply(
#                lambda x: (x - x.min()) / (x.max() - x.min()))
#test_X['Age','Fare'] = test_X['Age', 'Fare'].apply(
#                lambda x: (x - x.min()) / (x.max() - x.min())) 
print(train_X.columns)
age = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Age'), (0, 1))
sibsp = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'SibSp'), (0,1))
parch = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Parch'), (0, 1,))
fare = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Fare'), (0, 1))

sex_male = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Sex_male'), (0,1))
sex_female = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Sex_female'), (0,1))
#cabin = tf.feature_column.bucketized_column(
#    tf.feature_column.numeric_column(key = 'Cabin'), (0,1))
embarked_q = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Embarked_Q'), (0,1))
embarked_s = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Embarked_S'), (0,1))
embarked_c = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Embarked_C'), (0,1))
pclass_1 = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Pclass_1'), (0, 1))
pclass_2 = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Pclass_2'), (0, 1))
pclass_3 = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Pclass_3'), (0, 1))

feature_columns = [age, sibsp, parch, fare, sex_male, sex_female, 
                            embarked_c, embarked_s, embarked_q, pclass_1, pclass_2, pclass_3]

classifier = tf.estimator.BoostedTreesClassifier(
                feature_columns = feature_columns,
                model_dir = "./model_dnn",
                n_batches_per_layer = 40,
                n_trees = 25,
                max_depth=5,
                #tree_complexity = 0.2,
                l2_regularization=0.01)
                #pruning_mode = 'none')
                
classifier.train(input_fn = tf.estimator.inputs.pandas_input_fn(
        train_X, train_y,
        batch_size=16, num_epochs=500, shuffle=True), steps= 25000)
        
classifier.evaluate(input_fn = tf.estimator.inputs.pandas_input_fn(
        train_X, train_y, shuffle=True), steps= 1)
    
pred = classifier.predict(input_fn = tf.estimator.inputs.pandas_input_fn(
        test_X, batch_size=10, shuffle=False))

predictions=list()
for i, prd in enumerate(pred):
    predictions.append(prd["classes"])
sub_df = pd.DataFrame({"ImageId":test_id, "Label":predictions})
sub_df.to_csv("kaggle_kernel_sub.csv", index=False)
