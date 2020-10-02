import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

dfs_train = [train.values[img] for img in range(0,train.shape[0])]
dfs_test = [test.values[img] for img in range(0,test.shape[0])]

dfs_train = np.asarray(dfs_train)
dfs_test = np.asarray(dfs_test)

linsvc = LinearSVC(C=1)
linsvc = linsvc.fit(dfs_train[0::,1::], dfs_train[0::,0])

output_svc = linsvc.predict(dfs_test).astype(int)

np.savetxt("out_svc.csv", output_svc, delimiter=",")

'''
def draw_img(row,c):
    pix = row.reshape((28, 28)).astype('uint8')
    im = Image.fromarray(pix)
    im.save('out%d.png' % c)
    
for i in range(0,100): draw_img(dfs_test[i],i)
'''