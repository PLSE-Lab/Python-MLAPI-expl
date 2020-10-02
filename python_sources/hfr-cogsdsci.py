import os
import re
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from scipy import ndimage, misc
from scipy.optimize import curve_fit
import random
import pandas as pd
from operator import methodcaller
import multiprocessing as mp
import threading 
from threading import Thread
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.utils import to_categorical, np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
from skimage.feature import texture
from skimage import measure
from mpl_toolkits import mplot3d

device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))
config = tf.ConfigProto()


def get_image_order(folder):
    files = os.listdir(folder)
    order = pd.DataFrame([re.split("[-.]", x) for x in files], columns = ["Subject", "RImage.Idx", "Ext"])
    order = order.drop("Ext", axis = 1)
    order["Subject"] = files
    order.columns = ["Files", "RImage.Idx"]
    order["RImage.Idx"] = [(int(x) - 1) for x in order["RImage.Idx"]]
    return(order.sort_values(["RImage.Idx"], axis = 0))
    
    
def train_test(eye_data, mouth_data, meta, f):
    rands = random.sample(range(meta.shape[0]), int(f * meta.shape[0]))
    Training = meta.iloc[rands]
    Testing = meta.iloc[np.setdiff1d(range(meta.shape[0]), rands)]
    
    x_train_eye = eye_data[Training["Image.Idx"]]
    x_test_eye = eye_data[Testing["Image.Idx"]]
    
    x_train_mouth = mouth_data[Training["Image.Idx"]]
    x_test_mouth = mouth_data[Testing["Image.Idx"]]
    
    x_train_face = mouth_data[Training["Image.Idx"]]
    x_test_face = mouth_data[Testing["Image.Idx"]]
    
    y_train = meta["Subject.ID"].iloc[Training["Image.Idx"]]
    y_test = meta["Subject.ID"].iloc[Testing["Image.Idx"]]
    
    return((x_train_eye, x_test_eye, x_train_mouth, x_test_mouth, x_train_face, x_test_face, y_train, y_test))

def create_model(x_train, y_train, e, r, c1):
    model = Sequential()
    
    model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape= x_train.shape[1:]))
    model.add(Conv2D(32, kernel_size=3, activation="relu"))

    model.add(MaxPooling2D((r, r)))
    model.add(Conv2D(c1, kernel_size = 3, activation="relu"))
    
    model.add(Flatten())
    model.add(Dense(38, activation = "softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(x_train, y_train, batch_size = 32, epochs=e)
    
    return(model)

def vote(votes):
    unique_votes, count = np.unique(votes, return_counts=True)
    mcount = np.max(count)
    idx = np.where(count == mcount)[0]
    l = len(idx)
    if(l > 1):
        ridx = random.randint(0, len(unique_votes) - 1)
        return(unique_votes[ridx])
    else:
        return(unique_votes[idx][0])

def accuracy(guessed, known):
    return(np.sum([guessed[i] == known[i] for i in range(len(guessed))]) / len(guessed))
    
def evaluate_to_n(r, c1):
    epochs = np.arange(0, N, 1)
    accuracies = np.zeros([4, len(epochs)])

    for j in range(len(epochs)):
        e = epochs[j]
        
        x_train_eye, x_test_eye, x_train_mouth, x_test_mouth, x_train_face, x_test_face, y_train, y_test = train_test(eye_im, mouth_im, meta, f)
        y_train = to_categorical(np.asarray(y_train) - 1, num_classes = 38)
        y_test_indices = np.asarray(y_test) - 1
        y_test = to_categorical(y_test_indices, num_classes = 38)
        
        with tf.device('/gpu:0'):
            eye_model = create_model(x_train_eye, y_train, e + 1, r, c1)
            mouth_model = create_model(x_train_mouth, y_train, e + 1, r, c1)
            face_model = create_model(x_train_face, y_train, e + 1, r, c1)
            
            y_pred_eye = eye_model.predict_classes(x_test_eye)
            y_pred_mouth = mouth_model.predict_classes(x_test_mouth)
            y_pred_face = face_model.predict_classes(x_test_face)
        
        y_pred_total = [vote([y_pred_eye[i], y_pred_mouth[i]]) for i in range(len(y_pred_eye))]
        
        accuracies[0, j] = accuracy(y_pred_eye, y_test_indices)
        accuracies[1, j] = accuracy(y_pred_mouth, y_test_indices)
        accuracies[2, j] = accuracy(y_pred_total, y_test_indices)
        accuracies[3, j] = accuracy(y_pred_face, y_test_indices)
        
    return(accuracies)
    

meta = pd.read_csv("../input/meta-data/meta.csv").drop(["Unnamed: 0"], axis = 1)
meta["Image.Idx"] = meta["Image.Idx"] - 1

file_order = get_image_order("../input/cropped/cropped/cropped/eyes")

eye_im_master = io.ImageCollection([os.path.join("../input/cropped/cropped/cropped/eyes/", x) for x in file_order["Files"]], load_func = misc.imread)
mouth_im_master = io.ImageCollection([os.path.join("../input/cropped/cropped/cropped/mouth/", x) for x in file_order["Files"]], load_func = misc.imread)
face_im_master = io.ImageCollection([os.path.join("../input/cropped/cropped/cropped/face/", x) for x in file_order["Files"]], load_func = misc.imread)

eye_im = np.asarray(eye_im_master).reshape(len(eye_im_master), eye_im_master[0].shape[0], eye_im_master[0].shape[1], 1)
mouth_im = np.asarray(mouth_im_master).reshape(len(mouth_im_master), mouth_im_master[0].shape[0], mouth_im_master[0].shape[1], 1)
face_im = np.asarray(face_im_master).reshape(len(face_im_master), face_im_master[0].shape[0], face_im_master[0].shape[1], 1)


N = 10
r = 4
c1 = 32
M = 1
f = 0.5
accuracies = np.zeros([4, N])
for i in range(M):
    accuracies = accuracies + evaluate_to_n(r, c1)
accuracies = accuracies / M


x = np.arange(1, len(accuracies[0, :]) + 1, 1)
plt.figure(facecolor="white")
plt.plot(x, accuracies[0, :], 'r')
plt.plot(x, accuracies[1, :], 'b')
# plt.plot(x, accuracies[2, :], 'k')
plt.plot(x, accuracies[3, :], 'm')
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy')
plt.title('CNN')
plt.ylim([0, 1])
# plt.legend(['Eye', 'Mouth', 'Voted', 'Face'], loc = 'best')
plt.legend(['Eye', 'Mouth', 'Face'], loc = 'best')
plt.show()



##################
def extract_features(feature_temp, im_set, objects, properties, return_dict):
    for i in range(len(im_set)):
        P = texture.greycomatrix(im_set[i], [1], [0, np.pi / 4], levels=256, symmetric=False, normed=False)
        contrast = texture.greycoprops(P, properties[0])
        dissimilarity = texture.greycoprops(P, properties[1])
        homogeneity = texture.greycoprops(P, properties[2])
        energy = texture.greycoprops(P, properties[3])
        correlation = texture.greycoprops(P, properties[4])
        ASM = texture.greycoprops(P, properties[5])
        
        feature_temp[objects + 'Contrast'][i] = np.mean(contrast)
        feature_temp[objects + 'Dissimilarity'][i] = np.mean(dissimilarity)
        feature_temp[objects + 'Homogeneity'][i] = np.mean(homogeneity)
        feature_temp[objects + 'Energy'][i] = np.mean(energy)
        feature_temp[objects + 'Correlation'][i] = np.mean(correlation)
        feature_temp[objects + 'ASM'][i] = np.mean(ASM)
        feature_temp[objects + 'Contrast_sd'][i] = np.std(contrast)
        feature_temp[objects + 'Dissimilarity_sd'][i] = np.std(dissimilarity)
        feature_temp[objects + 'Homogeneity_sd'][i] = np.std(homogeneity)
        feature_temp[objects + 'Energy_sd'][i] = np.std(energy)
        feature_temp[objects + 'Correlation_sd'][i] = np.std(correlation)
        feature_temp[objects + 'ASM_sd'][i] = np.std(ASM)
        feature_temp[objects + 'Entropy_Base'][i] = measure.shannon_entropy(im_set[i])
        feature_temp[objects + 'Entropy_GLCM'][i] = measure.shannon_entropy(P)
        feature_temp[objects + 'Max_Intensity'][i] = im_set[i].max()
        feature_temp[objects + 'Min_Intensity'][i] = im_set[i].min()
        feature_temp[objects + 'Mean_Intensity'][i] = np.mean(im_set[i])
        feature_temp[objects + 'Median_Intensity'][i] = np.median(im_set[i])
        feature_temp[objects + 'sd_Intensity'][i] = np.std(im_set[i])
        feature_temp[objects + 'Max_GLCM'][i] = P.max()
        feature_temp[objects + 'Min_GLCM'][i] = P.min()
        feature_temp[objects + 'Mean_GLCM'][i] = np.mean(P)
        feature_temp[objects + 'Median_GLCM'][i] = np.median(P)
        feature_temp[objects + 'sd_GLCM'][i] = np.std(P)
        
        laplacian = cv2.Laplacian(im_set[i],cv2.CV_64F)
        feature_temp[objects + 'Laplacian_Max_Intensity'][i] = laplacian.max()
        feature_temp[objects + 'Laplacian_Min_Intensity'][i] = laplacian.min()
        feature_temp[objects + 'Laplacian_Mean_Intensity'][i] = np.mean(laplacian)
        feature_temp[objects + 'Laplacian_Median_Intensity'][i] = np.median(laplacian)
        
        glcm_laplacian = cv2.Laplacian(np.mean(P, axis = 3), cv2.CV_64F)
        feature_temp[objects + 'Laplacian_Max_GLCM'][i] = glcm_laplacian.max()
        feature_temp[objects + 'Laplacian_Min_GLCM'][i] = glcm_laplacian.min()
        feature_temp[objects + 'Laplacian_Mean_GLCM'][i] = np.mean(glcm_laplacian)
        feature_temp[objects + 'Laplacian_Median_GLCM'][i] = np.median(glcm_laplacian)
    
    return_dict.append(feature_temp)
    

eye_features = pd.DataFrame(meta, columns = ['Subject.ID', 'Deg.Azimuth', 'Deg.Elevation'])
mouth_features = pd.DataFrame(meta, columns = ['Subject.ID', 'Deg.Azimuth', 'Deg.Elevation'])
face_features = pd.DataFrame(meta, columns = ['Subject.ID', 'Deg.Azimuth', 'Deg.Elevation'])
objects = ["Eye_", "Mouth_", "Face_"]
feature_names = ["Contrast", "Contrast_sd", "Dissimilarity", "Dissimilarity_sd", "Homogeneity", 
                "Homogeneity_sd", "Energy", "Energy_sd", "Correlation", "Correlation_sd", 
                "ASM", "ASM_sd", "Max_Intensity", "Min_Intensity", "Mean_Intensity", "Median_Intensity",
                "sd_Intensity", "Entropy_Base", "Entropy_GLCM", "Max_GLCM", "Min_GLCM", "Mean_GLCM", "Median_GLCM",
                "sd_GLCM", "Laplacian_Max_Intensity", "Laplacian_Min_Intensity", "Laplacian_Mean_Intensity", 
                "Laplacian_Median_Intensity", "Laplacian_Max_GLCM", "Laplacian_Min_GLCM", "Laplacian_Mean_GLCM", 
                "Laplacian_Median_GLCM"]
properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
for i in range(len(feature_names)):
    eye_features.insert(i + 3, objects[0] + feature_names[i], np.arange(0, len(eye_features)))
    mouth_features.insert(i + 3, objects[1] + feature_names[i], np.arange(0, len(mouth_features)))
    face_features.insert(i + 3, objects[2] + feature_names[i], np.arange(0, len(face_features)))

manager = mp.Manager()
eye_dict = manager.list()
mouth_dict = manager.list()
face_dict = manager.list()
p_eye = mp.Process(target = extract_features, args = (eye_features, eye_im_master, objects[0], properties, eye_dict))    
p_mouth = mp.Process(target = extract_features, args = (mouth_features, mouth_im_master, objects[1], properties, mouth_dict)) 
p_face = mp.Process(target = extract_features, args = (face_features, face_im_master, objects[2], properties, face_dict)) 
jobs = [p_eye, p_mouth, p_face]

for i in range(3):
    jobs[i].start()

for i in range(3):
    jobs[i].join()

eye_features = eye_dict[0]
mouth_features = mouth_dict[0]
face_features = face_dict[0]

eye_features = eye_features.loc[:, (eye_features != 0).any(axis=0)]
mouth_features = mouth_features.loc[:, (mouth_features != 0).any(axis=0)]
face_features = face_features.loc[:, (face_features != 0).any(axis=0)]



f = 0.8
rands = random.sample(range(eye_features.shape[0]), int(f * eye_features.shape[0]))
train = rands
test = np.setdiff1d(range(eye_features.shape[0]), rands)
Training_eye = eye_features.iloc[train]
Training_mouth = mouth_features.iloc[train]
Training_face = face_features.iloc[train]
Testing_eye = eye_features.iloc[test]
Testing_mouth = mouth_features.iloc[test]
Testing_face = face_features.iloc[test]

Y_train = np.array(Training_eye["Subject.ID"])
X_train_eye = pd.DataFrame(preprocessing.scale(Training_eye.drop("Subject.ID", axis = 1)), columns = Training_eye.drop("Subject.ID", axis = 1).columns)
X_train_mouth = pd.DataFrame(preprocessing.scale(Training_mouth.drop("Subject.ID", axis = 1)), columns = Training_mouth.drop("Subject.ID", axis = 1).columns)
X_train_face = pd.DataFrame(preprocessing.scale(Training_face.drop("Subject.ID", axis = 1)), columns = Training_face.drop("Subject.ID", axis = 1).columns)

Y_test = np.array(Testing_eye['Subject.ID'])
X_test_eye = pd.DataFrame(preprocessing.scale(Testing_eye.drop("Subject.ID", axis = 1)), columns = Testing_eye.drop("Subject.ID", axis = 1).columns)
X_test_mouth = pd.DataFrame(preprocessing.scale(Testing_mouth.drop("Subject.ID", axis = 1)), columns = Testing_mouth.drop("Subject.ID", axis = 1).columns)
X_test_face = pd.DataFrame(preprocessing.scale(Testing_face.drop("Subject.ID", axis = 1)), columns = Testing_face.drop("Subject.ID", axis = 1).columns)

net_score_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
net_score_mat = np.zeros([len(net_score_x), 3])

def MLPclass(net_score_x, i, X_train_eye, X_train_mouth, X_train_face, Y_train, Y_test, l):
    global net_score_mat
    temp = np.zeros([len(net_score_x), 3])
    for k in range(len(net_score_x)):
        net = MLPClassifier(hidden_layer_sizes=(net_score_x[k], ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)
        if i == 0:
            net.fit(X_train_eye, Y_train)
            y_pred = net.predict(X_test_eye)
            temp[k, 0] = accuracy_score(Y_test, y_pred)
        elif i == 1:
            net.fit(X_train_mouth, Y_train)
            y_pred = net.predict(X_test_mouth)
            temp[k, i] = accuracy_score(Y_test, y_pred)
        elif i == 2:
            net.fit(X_train_face, Y_train)
            y_pred = net.predict(X_test_face)
            temp[k, i] = accuracy_score(Y_test, y_pred)
    
    l.acquire()
    net_score_mat += temp
    l.release()

lock = threading.Semaphore()
eye_t = Thread(target=MLPclass, args=(net_score_x, 0, X_train_eye, X_train_mouth, X_train_face, Y_train, Y_test, lock))
mouth_t = Thread(target=MLPclass, args=(net_score_x, 1, X_train_eye, X_train_mouth, X_train_face, Y_train, Y_test, lock))
face_t = Thread(target=MLPclass, args=(net_score_x, 2, X_train_eye, X_train_mouth, X_train_face, Y_train, Y_test, lock))
jobs = [eye_t, mouth_t, face_t]

for i in range(3):
    jobs[i].start()
    
for i in range(3):
    jobs[i].join()   
    
    
plt.figure(facecolor="white")
plt.plot(net_score_x, net_score_mat[:, 0], 'r-')
plt.plot(net_score_x, net_score_mat[:, 1], 'b-')
plt.plot(net_score_x, net_score_mat[:, 2], 'k-')
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Test Accuracy')
plt.title('Neural Network')
plt.legend(['Eye', 'Mouth', 'Face'], loc = 'best')
plt.ylim([0, 1])
plt.show()

eye_features.to_csv('eye_features.csv',index=False)
mouth_features.to_csv('mouth_features.csv', index=False)
face_features.to_csv('face_features.csv', index=False)
    
#PCA 
pca_eye = PCA(n_components = 3)
pca_eye.fit(eye_features)
eye_features_pca = pca_eye.transform(eye_features)

pca_mouth = PCA(n_components = 3)
pca_mouth.fit(mouth_features)
mouth_features_pca = pca_mouth.transform(mouth_features)

pca_face = PCA(n_components = 3)
pca_face.fit(face_features)
face_features_pca = pca_face.transform(face_features)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(eye_features_pca[:,0], eye_features_pca[:,1], eye_features_pca[:,2], c=eye_features_pca[:,2], cmap='Greens')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(mouth_features_pca[:,0], mouth_features_pca[:,1], mouth_features_pca[:,2], c=mouth_features_pca[:,2], cmap='Greens')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(face_features_pca[:,0], face_features_pca[:,1], face_features_pca[:,2], c=face_features_pca[:,2], cmap='Greens')

