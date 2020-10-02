from sklearn.datasets import load_digits
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
import tensorflow as tf
from tensorflow.contrib import learn
def LoadData(test_size=0.25,is_images=True,one_hot=True):
    digits=load_digits()
    images=digits['images']
    data=digits['data']
    labels=digits['target']
    
    n_folds=int(1/test_size)
    skf=StratifiedKFold(labels,n_folds=n_folds)
    train_index,test_index=list(skf)[0]
    
    lb=preprocessing.LabelBinarizer()
    transformed_labels=lb.fit_transform(labels)
    if(is_images == True):
        if(one_hot ==True):
            return (images[train_index,:,:],images[test_index,:,:],transformed_labels[train_index,:],transformed_labels[test_index,:])
        else:
            return (images[train_index,:,:],images[test_index,:,:],labels[train_index],labels[test_index])
    else:
        if(one_hot == True):
            return (data[train_index,:],data[test_index,:],transformed_labels[train_index,:],transformed_labels[test_index,:])
        else:
            return (data[train_index,:],data[test_index,:],labels[train_index],labels[test_index])

train_x,test_x,train_y,test_y = LoadData(is_images=False,one_hot=False)

dnn=learn.DNNClassifier(hidden_units=[10,20],n_classes=10)
dnn.fit(x=train_x,y=train_y,steps=50)
print(dnn.evaluate(x=test_x,y=test_y)["accuracy"])
