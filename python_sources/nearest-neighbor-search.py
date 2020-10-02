# Nearest Neighbor Search with Cosine Similarity
import pandas as pd
import numpy as np
import time

def l2_normalize(x):
    norms = np.apply_along_axis(np.linalg.norm, 1, x) + 1.0e-7
    return x / np.expand_dims(norms, -1)

def nearest_neighbor(train_x, train_y, test_x):
    train_x = l2_normalize(train_x)
    test_x = l2_normalize(test_x)
    
    cosine = np.dot(test_x, np.transpose(train_x))
    argmax = np.argmax(cosine, axis=1)
    preds = train_y[argmax]

    return preds

def validate(train_x, train_y, test_x, test_y):
    preds = nearest_neighbor(train_x, train_y, test_x)
    count = len(preds)
    correct = (preds == test_y).sum()
    return float(correct) / count

def make_submission(train_x, train_y, test_x, fname):
    preds = nearest_neighbor(train_x, train_y, test_x)
    pd.DataFrame({"ImageId": list(range(1, len(preds) + 1)),
                  "Label": preds}
                 ).to_csv(fname, index = False, header = True)

# Read data
train_x = pd.read_csv('../input/train.csv')
train_y = train_x.ix[:,0].values.astype('int32')
train_x = (train_x.ix[:,1:].values).astype('float32')
test_x = (pd.read_csv('../input/test.csv').values).astype('float32')

# Run validation
train_size = int(len(train_x) * 0.8)
vt_train_x = train_x[:train_size]
vt_train_y = train_y[:train_size]
vt_test_x = train_x[train_size:]
vt_test_y = train_y[train_size:]
t = time.time()
acc = validate(vt_train_x, vt_train_y, vt_test_x, vt_test_y)
print("* Validation Accuracy: %f, %.2fs" % (acc, time.time() - t))

# Make submission # out of memory on kaggle script

print("* Make submission...")
t = time.time()
make_submission(train_x, train_y, test_x, "nearest_neighbor.csv")
print("* Done %.2fs" % (time.time() - t))
