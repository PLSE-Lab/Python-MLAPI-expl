import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.decomposition import PCA

print('Read data...')
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:,1:]
labels = labeled_images.iloc[0:,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

pca = PCA(n_components=0.8, whiten=True)
train_images = pca.fit_transform(train_images)
test_images = pca.transform(test_images)

print('Train SVM...')
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
print("score: " + str(clf.score(test_images,test_labels)))

print('Predicting...')
test_data = pd.read_csv('../input/test.csv')
test_data = pca.transform(test_data)
results = clf.predict(test_data)

print('Saving...')
pd.DataFrame({"ImageId": list(range(1, len(results) + 1)), "Label": results}).to_csv("result.csv", index=False, header=True)