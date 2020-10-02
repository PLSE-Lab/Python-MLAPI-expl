import numpy as np
import pandas as pd
import imblearn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
from tensorflow import keras


#get Data
filename = "train_set.csv"
data_train = pd.read_csv(filename)
filename = "test_set.csv"
data_test = pd.read_csv(filename)
fea_col = data_train.columns[2:]

data_Y = data_train['target']
data_X = data_train[fea_col]
data_test_X = data_test.drop(columns=['id'])


#impute
imputer = SimpleImputer(missing_values=-1, strategy="mean")
imputer.fit(data_X)
features = imputer.transform(data_X)

#get best features
#selector=SelectKBest(score_func=f_classif, k=20)
selector=SelectKBest(score_func=chi2, k=20)
#selector=SelectKBest(score_func=f_regression, k=20)
fit = selector.fit(features, data_Y)
features = fit.transform(features)

#sample
under_over_sampler = SMOTE(sampling_strategy="minority")
X_smt, y_smt = under_over_sampler.fit_sample(features, data_Y)

#train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_smt, y_smt, test_size = 0.3, shuffle = True)
#x_train, x_test, y_train, y_test = train_test_split(features, data_Y, test_size = 0.3, shuffle = True)
#x_train, y_train = under_over_sampler.fit_sample(x_train, y_train)

#indices for test feature
indices = selector.get_support()




#neuronal network
model = keras.Sequential([
  keras.layers.Flatten(input_shape=(20,)),
  keras.layers.Dense(4, activation=tf.nn.relu),
  keras.layers.Dense(4, activation=tf.nn.relu),
  keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=10)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)



#cast to int
y_pred = model.predict(x_test)
y_pred = y_pred.flatten()
y_pred = (y_pred>0.5).astype(int)


def get_total_accuracy(y_pred,y_test):
    acc = sum(y_pred==y_test)/len(y_test)
    print("total accuracy: ",acc)
    return acc

def extract_one_label(x_test, y_test, class_label):
    indices = y_test == class_label
    X_pos = x_test[indices]
    y_pos = y_test[indices]
    return X_pos, y_pos, indices

def get_one_class_accuracy(x_test, y_test, y_pred,class_label):  
    X_pos, y_pos, indices = extract_one_label(x_test, y_test, class_label)
    y_pospred = y_pred[indices]
    acc=sum(y_pospred==y_pos)/len(y_pos)
    print("accuracy of label ",class_label,"is: ",acc)
    return acc
def get_score(y_test,y_pred):
    score = metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary', zero_division='warn')
    print("binary f1 score is: ",score)
    score = metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='weighted', zero_division='warn')
    print("weighted f1 score is: ",score)
    score = metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='micro', zero_division='warn')
    print("micro f1 score is: ",score)
    return score


get_total_accuracy(y_pred,y_test)
get_one_class_accuracy(x_test,y_test,y_pred,class_label=0)
get_one_class_accuracy(x_test,y_test,y_pred,class_label=1)
get_score(y_test,y_pred)
print(metrics.confusion_matrix(y_test,y_pred))
print(metrics.classification_report(y_test,y_pred))


#Submission
data_test_X = data_test.drop(columns=['id'])

imputer=SimpleImputer(missing_values=-1, strategy='mean')
fit = imputer.fit(data_test_X)
features = fit.transform(data_test_X)
y_target = model.predict(features[:,indices])
y_target = (y_target>0.5).astype(int)
y_target = y_target.flatten()
#y_target = clf.predict(features[:,indices])
sum(y_target==0)
print (y_target, sum(y_target==1), sum(y_target==0))




data_out = pd.DataFrame(data_test['id'].copy())
data_out.insert(1, "target", y_target, True) 
data_out.to_csv('submission.csv',index=False)

