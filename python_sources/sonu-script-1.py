import pandas as pd
import h2o
# The competition datafiles , are in the directory ../input
# Read competition data files:
#h2o.connect(ip = "h2o.127.0.0.1", port=54321)
h2o.init(ip="127.0.0.1", port=54321)
 
#train = pd.read_csv("../input/train.csv")
#test  = pd.read_csv("../input/test.csv")
#print(train[1:])
#X = train[:-1]

# Write to the log:
#print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
#print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs