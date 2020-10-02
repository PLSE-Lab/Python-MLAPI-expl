# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from pandas.tools.plotting import andrews_curves
from pandas.tools.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug

if False:
	hooks = [tf_debug.LocalCLIDebugHook()]
	def has_inf_or_nan(datum, tensor):
		  return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))
else:
	hooks = 0

	

tf.logging.set_verbosity(tf.logging.INFO)

tmp = pd.read_csv("../input/Iris.csv")
df = tmp.sample(frac=.8,random_state=20)
test= tmp.drop(df.index)
#print(df.head())

#print(df['Species'].value_counts())




# sns.pairplot(df.drop("Id", axis=1), hue="Species", size=3)
# plt.draw()
# plt.figure(2)
# andrews_curves(df.drop("Id", axis=1), "Species")
# plt.draw()
# plt.figure(3)
# parallel_coordinates(df.drop("Id", axis=1), "Species")
# plt.show()
def my_input_fn():
	return input_fn(df)

def my_input_test():
	return input_fn(test)


def input_fn(d=df):
	cols = {k: tf.constant(d[k].values)
        	for k in d.dtypes.index.values}
    	

	label = tf.constant(d[clf].values)
	return cols, label

df = df.drop(['Id'],axis=1)
tmp = pd.Series(df['Species']).astype('category')
df['Species'] = tmp.cat.codes
df['Species'] = df['Species'].astype(np.int64)
print(df.dtypes)
data = [x for x in df.columns[:-1]]
clf = df.columns[-1]
features = []
for x in data:
	maximum = np.max(df[x])
	minimum = np.min(df[x])
	features += [tf.contrib.layers.real_valued_column(x,normalizer = lambda m:(m-minimum)/(maximum - minimum))]

test = test.drop(['Id'],axis=1)
tmp = pd.Series(test['Species']).astype('category')
test['Species'] = tmp.cat.codes
test['Species'] = test['Species'].astype(np.int64)
print(test.dtypes)
test_data = [x for x in test.columns[:-1]]
test_clf = test.columns[-1]


lin = tf.contrib.learn.LinearClassifier(feature_columns=features,
											n_classes=3,
											model_dir="model/")
dnn = tf.contrib.learn.DNNClassifier(feature_columns=features,
                                            hidden_units=[10, 20, 5],
                                            n_classes=3,
                                            model_dir="dmodel/")
hparams = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
        num_trees=5, max_nodes=1000, num_classes=3, num_features=4)
rforest = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(hparams,
                                                                                report_feature_importances=True,
                                                                                model_dir="rfmodel/")

iris = tf.contrib.learn.datasets.load_iris()
data = iris.data.astype(np.float32)
target = iris.target.astype(np.float32)
hook = [tf.contrib.tensor_forest.client.random_forest.TensorForestLossHook(10)]

sess = tf.Session()

sess.run(tf.global_variables_initializer())


lin.fit(input_fn=my_input_fn,steps=1000,monitors=hooks)
prd = lin.evaluate(input_fn=my_input_test,steps=1)["accuracy"]
print("\n ACCURACY: {0:f}\n".format(prd))

dnn.fit(input_fn=my_input_fn,steps=1000)
prd2 = dnn.evaluate(input_fn=my_input_test,steps=1)["accuracy"]
print("\n ACCURACY: {0:f}\n".format(prd2))

rforest.fit(x=data,y=target,steps=1000,monitors=hook)
rforest.evaluate(x=data,y=target,steps=1)
