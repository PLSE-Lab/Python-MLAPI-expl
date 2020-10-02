import numpy
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
COMPONENT_NUM = 35

print('Read training data...')
with open('../input/train.csv', 'r') as reader:
    reader.readline()
    train_label = []
    train_data = []
    for line in reader.readlines():
        data = list(map(int, line.rstrip().split(',')))
        train_label.append(data[0])
        train_data.append(data[1:])
print('Loaded ' + str(len(train_label)))

train,test=numpy.split(numpy.array(train_data),2)
trainl,testl=numpy.split(numpy.array(train_label),2)

print('Reduction...')
trainl = numpy.array(trainl)
train = numpy.array(train)
pca = PCA(n_components=COMPONENT_NUM, whiten=True)
pca.fit(train)
traint = pca.transform(train)

print('Train SVM...')
svc = SVC()
svc.fit(traint, trainl)

#print('Read testing data...')
#with open('../input/test.csv', 'r') as reader:
#    reader.readline()
#    test_data = []
#    for line in reader.readlines():
#        pixels = list(map(int, line.rstrip().split(',')))
#        test_data.append(pixels)
#print('Loaded ' + str(len(test_data)))

print('Predicting...')
testt=pca.transform(test)
predict=svc.predict(testt)
count=0
score=0
print(len(test[0:784]))
fig = plt.figure(figsize=(28,28))
plt.show(fig)
img=numpy.reshape(test[0:784],(28,-1))
print(img)
plt.imshow(img)
plt.show()
for p in predict:
    if p==testl[count]:
        score=score+1
    else: 
        print(p,testl[count]) 
    count +=1
    
print(score/float(count))


#test_data = numpy.array(test_data)
#test_data = pca.transform(test_data)
#predict = svc.predict(test_data)




#print('Saving...')
#with open('predict.csv', 'w') as writer:
#    writer.write('"ImageId","Label"\n')
#    count = 0
#    for p in predict:
#        count += 1
#        writer.write(str(count) + ',"' + str(p) + '"\n')

