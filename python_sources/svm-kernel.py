import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
from multiprocessing.dummy import Pool as ThreadPool
#%matplotlib inline


#labeled_images = pd.read_csv('C:/Users/Harshit/Desktop/MNIST/Kaggel/train.csv')
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:42000,1:]
labels = labeled_images.iloc[0:42000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.9,test_size=0.1, random_state=21)#0.945238095238
#train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.9,test_size=0.1, random_state=21)#0.94130952381
#train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.6,test_size=0.4, random_state=21)#0.936011904762
i=1
'''img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))'''

test_images[test_images>0]=1
train_images[train_images>0]=1

img=train_images.iloc[i].as_matrix().reshape((28,28))
#plt.imshow(img,cmap='binary')


'''plt.axis("off")
plt.imshow(img,cmap='gray')'''

#plt.show()

plt.title(train_labels.iloc[i,0])

plt.hist(train_images.iloc[i])

#plt.show()

clf = svm.SVC()
#pool = ThreadPool(128)
#raveled_val= train_labels.values.ravel()
#pool.starmap(clf.fit,zip([train_images],[raveled_val]))
clf.fit(train_images,train_labels.values.ravel())
# close the pool and wait for the work to finish 
#pool.close() 
pool.join() 

#clf.fit(train_images, train_labels.values.ravel())
sc= clf.score(test_images,test_labels)
print(sc)
'''
#test_data=pd.read_csv('C:/Users/Harshit/Desktop/MNIST/Kaggel/test.csv')
test_data=pd.read_csv('../input/test.csv')
test_data[test_data>0]=1
#results=clf.predict(test_data[0:5000])
results=clf.predict(test_data)

df = pd.DataFrame(results)
df.index+=1
df.index.name='ImageId'
df.columns=['Label']
df.to_csv('results.csv', header=True)

'''