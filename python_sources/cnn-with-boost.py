import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


seed = 2
np.random.seed(seed)

# Load the data
train = pd.read_csv("../input/train.csv")
print(train.shape)

y = train["label"]
X = train.drop("label", axis = 1) 
print(y.value_counts().to_dict())
y = to_categorical(y, num_classes = 10)
del train

X = X / 255.0
X = X.values.reshape(-1,28,28,1)

train_index, valid_index = ShuffleSplit(n_splits=1, train_size=0.9, test_size=None, random_state=seed).split(X).__next__()
train_x = X[train_index]
train_y = y[train_index]
valid_x = X[valid_index]
valid_y = y[valid_index]
print(train_x.shape, valid_x.shape)


model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Valid', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same',  activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-3), metrics=["accuracy"])
annealer = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

epochs = 20
batch_size = 64

# spot_shape = (2, 2)
# def random_add_or_erase_spot(img):
#     img = img.copy()
        
#     cand_shape0 = img.shape[0] - spot_shape[0] + 1
#     cand_shape1 = img.shape[1] - spot_shape[1] + 1
#     cand_point_num = cand_shape0 * cand_shape1
#     chosen_point_index = np.random.randint(0, cand_point_num)
#     chosen_index00 = chosen_point_index // cand_shape1
#     chosen_index01 = chosen_index00 + spot_shape[0]
#     chosen_index10 = chosen_point_index % cand_shape1
#     chosen_index11 = chosen_index10 + spot_shape[1]
    
#     most_black = np.max(img)
#     most_white = np.min(img)
#     gray = np.mean(img)
#     if not np.random.randint(0, 2): # add
#         img[chosen_index00:chosen_index01, chosen_index10:chosen_index11, 0] = 0.5*most_black + 0.5*gray
#     else: # erase
#         img[chosen_index00:chosen_index01, chosen_index10:chosen_index11, 0] = 0.9*most_white + 0.1*gray
        
#     return img


datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
        samplewise_std_normalization=False, zca_whitening=False, rotation_range=10, zoom_range = 0.1, width_shift_range=0.1,
        height_shift_range=0.1, horizontal_flip=False, vertical_flip=False)#, preprocessing_function=random_add_or_erase_spot)

model.fit_generator(datagen.flow(train_x,train_y,batch_size=batch_size), epochs=epochs, validation_data=(valid_x[:600,:],valid_y[:600,:]),
                              verbose = 2, steps_per_epoch=train_x.shape[0]//batch_size, callbacks=[annealer])

 
tr_ps =  model.predict(train_x)
tr_p = np.max(tr_ps, axis=1)
tr_pl = np.argmax(tr_ps, axis=1)
vl_ps = model.predict(valid_x)
vl_p = np.max(vl_ps, axis=1)
vl_pl = np.argmax(vl_ps, axis=1)


tr_y = np.argmax(train_y, axis=1)
tr_x_err = train_x[tr_y != tr_pl]
tr_y_err = tr_y[tr_y != tr_pl]
tr_pl_err = tr_pl[tr_y != tr_pl]
vl_y = np.argmax(valid_y, axis=1)
vl_x_err = valid_x[vl_y != vl_pl]
vl_y_err = vl_y[vl_y != vl_pl]
vl_pl_err = vl_pl[vl_y != vl_pl]
for i in range(tr_y_err.shape[0]):
    plt.imsave(fname='train_err_%d_%d_%d.csv'%(tr_y_err[i], tr_pl_err[i], i), arr=tr_x_err[i].reshape(28,28),format='png')
for i in range(vl_y_err.shape[0]):
    plt.imsave(fname='valid_err_%d_%d_%d.csv'%(vl_y_err[i], vl_pl_err[i], i), arr=vl_x_err[i].reshape(28,28),format='png')


# epochs = 30
# p_group_bound = 0.95

# model.save('model.h5')
# print("Saved model to disk")

# sub_models = []
# for i in range(10):
#     tr_x = train_x[(tr_pl == i) & (tr_p < p_group_bound)]
#     tr_y = train_y[(tr_pl == i) & (tr_p < p_group_bound)]
#     vl_x = valid_x[(vl_pl == i) & (vl_p < p_group_bound)]
#     vl_y = valid_y[(vl_pl == i) & (vl_p < p_group_bound)]
#     print(i, tr_x.shape, tr_y.shape, vl_x.shape, vl_y.shape)
    
#     sub_model = load_model('model.h5')
#     sub_model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-3), metrics=["accuracy"])
#     print("Loaded model from disk")
    
#     sub_model.fit_generator(datagen.flow(tr_x,tr_y,batch_size=tr_x.shape[0]//2), epochs=epochs, validation_data=(vl_x,vl_y),
#                               verbose=2, steps_per_epoch=30, callbacks=[annealer])
#     sub_models.append(sub_model)


# def predict_label(px):
#     pps =  model.predict(px)
#     pp = np.max(pps, axis=1)
#     ppl = np.argmax(pps, axis=1)
#     res_ppl = np.argmax(pps, axis=1)
    
#     for i in range(10):
#         sub_model = sub_models[i]
#         sub_px = px[(ppl == i) & (pp < p_group_bound)]
#         sub_pl = np.argmax(sub_model.predict(sub_px), axis=1)
#         res_ppl[(ppl == i) & (pp < p_group_bound)] = sub_pl
        
#     return res_ppl


# def predict_label(px):
#     pps =  model.predict(px)
    
#     for i in range(10):
#         sub_ps = sub_models[i].predict(px)
#         pps += sub_ps
        
#     return np.argmax(pps, axis=1)


# valid_p = predict_label(valid_x)
# target = np.argmax(valid_y, axis=1)
# print("valid accuracy: {0:.4f}".format(accuracy_score(target, valid_p)))
# cm = confusion_matrix(target, valid_p)
# print(cm)


print('Base model scores:')
valid_loss, valid_acc = model.evaluate(valid_x, valid_y, verbose=0)
print("model valid loss: {0:.4f}, valid accuracy: {1:.4f}".format(valid_loss, valid_acc))

valid_p = np.argmax(model.predict(valid_x), axis=1)
target = np.argmax(valid_y, axis=1)
cm = confusion_matrix(target, valid_p)
print(cm)


test = pd.read_csv("../input/test.csv")
print(test.shape)
test = test / 255.0
test = test.values.reshape(-1,28,28,1)
# p = predict_label(test)
p = np.argmax(model.predict(test), axis = 1)

submission = pd.DataFrame(pd.Series(range(1, p.shape[0]+1), name='ImageId'))
submission['Label'] = p
submission.to_csv("cnn.csv", index=False)
