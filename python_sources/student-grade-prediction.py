# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
from pandas.tools.plotting import andrews_curves
from pandas.tools.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug


df = pd.read_csv("../input/xAPI-Edu-Data.csv")

#print(df.dtypes)

continuous = df._get_numeric_data()
continuous = continuous.dtypes.index.values
categorical = [x for x in df.columns[:-1] if x not in continuous]
classifier = df.columns[-1]
#tmp = pd.Series(df['Class']).astype('category')
#print (tmp.cat.categories)
#print (tmp.cat.codes)
#df['Class'] = tmp.cat.codes
#df['Class'] = df['Class'].astype(np.int64)
map_dict = {'L': 0, 'M':1,'H':2}
df['Class'] = df['Class'].map(map_dict)
df['Class'] = df['Class'].astype(np.int64)

sns.pairplot(df,hue='Class',size=2)
plt.figure()
sns.heatmap(df.corr(),vmax=.8,square=True)
plt.figure()
ax = sns.boxplot(x='Class', y='Discussion', data=df)
ax = sns.swarmplot(x='Class', y = 'Discussion', data=df, color='.25')
plt.figure()
ax = sns.boxplot(x='Class', y='VisITedResources', data=df)
ax = sns.swarmplot(x='Class', y = 'VisITedResources', data=df, color='.25')
plt.show()

train = df.sample(frac=.8,random_state=20)
test = df.drop(train.index)

print(df.dtypes)
print('_'*40)
print (continuous)
print('_'*40)
print ( categorical)
print('_'*40)
print (classifier)
print('_'*40)

features = []

for x in categorical:
	features += [tf.contrib.layers.sparse_column_with_hash_bucket(column_name=x,hash_bucket_size=1e3)]
for x in continuous:
	avg = np.average(df[x])
	std = np.std(df[x])

	features+= [tf.contrib.layers.sparse_column_with_integerized_feature(x,bucket_size=1000)]

dnn_features=[tf.contrib.layers.embedding_column(x,dimension=8) for x in features]

lin = tf.contrib.learn.LinearClassifier(feature_columns=features,
											n_classes=3,
											model_dir="model/lin/")
dnn = tf.contrib.learn.DNNClassifier(feature_columns=dnn_features,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="model/dnn/")

def input_fn(d=train):
	cols = {k: tf.constant(d[k].values)
        	for k in d.dtypes.index.values}
	label = tf.constant(d[classifier].values)
	return cols, label

def my_input_test(dt=test):
	cols = {k: tf.constant(dt[k].values)
		for k in dt.dtypes.index.values}
	return cols

def input_eval():
	return input_fn(test)

lin.fit(input_fn=input_fn,steps=1000)
prd = lin.predict_classes(input_fn=my_input_test)

#input()
lacc=lin.evaluate(input_fn=input_eval,steps=1)['accuracy']
print("\n ACCURACY: {0:f}\n".format(lacc))

#input()

dnn.fit(input_fn=input_fn,steps=1000)
acc=dnn.evaluate(input_fn=input_eval,steps=1)['accuracy']

prd = dnn.predict_classes(input_fn=my_input_test)



print("\n lin ACCURACY: {0:f}\n".format(lacc))
print("\n dnn ACCURACY: {0:f}\n".format(acc))