
import  numpy
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# parse train data
with open("../input/train.csv",'r') as reader:
    reader.readline()
    train_label = []
    train_data = []
    for line in reader.readlines():
        data = list(map(int, line.rstrip().split(',')))
        train_label.append(data[0])
        train_data.append(data[1:])

# train data PCA process
train_label = numpy.array(train_label)
train_data = numpy.array(train_data)
pca = PCA(n_components=28,whiten=True)
pca.fit(train_data)
train_data = pca.transform(train_data)

# train data SVM model
svc = SVC()
svc.fit(train_data, train_label)

# parse test data
with open("../input/test.csv",'r') as reader:
    reader.readline()
    test_data = []
    for line in reader.readlines():
        pixels = list(map(int,line.rstrip().split(',')))
        test_data.append(pixels)

# test data PCA process
test_data = numpy.array(test_data)
test_data = pca.transform(test_data)

# test data SVM predict
predict = svc.predict(test_data)

# output result
with open('predict.csv','w') as writer:
    writer.write('"ImageId","Label"\n')
    count = 0
    for p in predict:
        count+=1
        writer.write(str(count)+',"'+str(p)+'"\n')

