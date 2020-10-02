import gzip, pickle, sys
f = gzip.open('../input/mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    (X_train, y_train), (X_test, y_test) = pickle.load(f)
else:
    (X_train, y_train), (X_test, y_test) = pickle.load(f, encoding="bytes")
    
print(X_train.shape)
print(y_train.shape)