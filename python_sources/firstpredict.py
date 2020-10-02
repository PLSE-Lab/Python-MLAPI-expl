import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
import pandas as pd


from sklearn import linear_model
from numpy import genfromtxt,savetxt


dataset = pd.read_csv("../input/train.csv")
train_Y = dataset.iloc[:,0]
train_X = dataset.iloc[:,1:]

test_X = pd.read_csv("../input/test.csv")


#creat and train the model
model = linear_model.SGDClassifier()


model.fit(train_X,train_Y)


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



rs = model.predict(test_X)

sm = pd.DataFrame({ 'ImageId': range(1,len(rs) + 1), 'Label':rs})

sm.to_csv('mine_submisson.csv', index = False)
