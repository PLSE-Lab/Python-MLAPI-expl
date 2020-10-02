import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
# train = pd.read_csv("../input/train.csv")
# test  = pd.read_csv("../input/test.csv")

# Write to the log:
# print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
# print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import csv
import time

threadhold = 125

print ('Start',time.clock())
# list for feature
# train_feature = []
# list for labels
# train_label = []
ds = SupervisedDataSet(784, 1)
train_file = open("../input/train.csv",'r')
reader = csv.reader(train_file)
for row in reader:
    if reader.line_num == 1:
        # skip metadata
        continue
    # train_label = [1 if int(row[0]) == i else 0 for i in range(0, 10)]
    train_label = [int(row[0]), ]
    features_norm = [1 if int(x) > threadhold else 0 for x in row[1:]]
    ds.addSample(features_norm, train_label)
train_file.close()
# print (ds)
# list for feature
# test_feature = []
ds_test = SupervisedDataSet(784, 1)
# get the test set
test_file = open("../input/test.csv",'r')
reader = csv.reader(test_file)
for row in reader:
    if reader.line_num == 1:
        # skip metadata
        continue
    # test_lable = [0 for i in range(0, 10)]
    test_lable = [0, ]
    features_norm = [1 if int(x) > threadhold else 0 for x in row[0:]]
    ds_test.addSample(features_norm, test_lable)
test_file.close()
# print (ds_test)
print ('Done pre-process & Build NN!',time.clock())

# train_feature = np.asarray(train_feature, 'float64')
# train_label = np.asarray(train_label)
# test_feature = np.asarray(test_feature, 'float64')
# pca = pca.CovEigPCA(num_components=10)
# pca.train(train_feature)
# print train_feature
# train_feature = pca.reconstruct(train_feature)
# print train_feature
# test_feature = pca.reconstruct(test_feature)
# print 'Done PCA!',time.clock()

# make a nerual net
net = buildNetwork(784, 1, bias=True)
print ('Done Build!',time.clock())
trainer = BackpropTrainer(net, ds, verbose=True)
print ('Done Back!',time.clock())
trainer.train()
print ('Done fit!',time.clock())

test_label = net.activateOnDataset(ds_test)
print (test_label)
print ('Done predict!',time.clock())

output_file = open("output_pybrain.csv",'w+')
output_file.write("ImageId,Label\n")
i=1
for every in test_label:
    # k = 0
    # for j in every:
    #     if j == 1:
    #         break
    #     else:
    #         k=k+1
    output_file.write("%d,%d\n"%(i, every))
    i = i + 1
output_file.close()
print ('All Done',time.clock())