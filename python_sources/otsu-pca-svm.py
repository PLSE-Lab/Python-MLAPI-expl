import pandas as pd
import skimage
from numpy.matlib import zeros, int16
from skimage import filters
from sklearn import svm
from sklearn.decomposition import PCA

def otsu(img):
    # thresh = filters.threshold_otsu(img)
    thresh = 100
    img[img < thresh] = 0
    return img

def otsu_transform(imgs):
    imgs[imgs < 100] = 0
    return imgs
    # rows = len(imgs.index)
    # columns = len(imgs.columns)
    # i = 0
    # arr = zeros([rows, columns], dtype=int16)
    # while i < rows:
    #     img = imgs.iloc[i].values.reshape(28, 28)
    #     dst = otsu(img)
    #     arr[i] = dst.ravel()
    #     i += 1
    # return pd.DataFrame(arr, columns=imgs.columns, index=imgs.index, dtype=int16)


print('Read data...')
labeled_images = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

train_images = labeled_images.iloc[:,1:]
train_labels = labeled_images.iloc[:,:1]

print('Otsu thresholding...')
train_images = otsu_transform(train_images)
test_data = otsu_transform(test_data)

print('PCA...')
pca = PCA(n_components=0.8, whiten=True)
train_images = pca.fit_transform(train_images)
test_data = pca.transform(test_data)

print('Train SVM...')
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())

print('Predicting...')
results = clf.predict(test_data)

print('Saving...')
output = pd.DataFrame({"ImageId": list(range(1, len(results) + 1)), "Label": results})
output.to_csv("result.csv", index=False, header=True)
