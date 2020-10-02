from os import listdir
from PIL import Image
import numpy as np
import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.estimator import regression

# [1,0] - cat
# [0,1] - dog
data = []
label = []
lab_name = ["cat", "dog"]


def mov_file():
    i = 0
    for file in listdir("train/"):
        if not file.startswith('.') and file != 'Thumbs.db':
            i = i + 1
            file_name_split = file.split('.')
            file_name =  file_name_split[0]
            file_lab =  file_name_split[1]
            print(file_name, file_lab, i)
            if file_name == 'cat':
                save_dir = "cat/"+file_lab+".jpg"
            elif file_name == 'dog':
                save_dir = "dog/"+file_lab+".jpg"
            images = Image.open("test/" + file)
            imResize = images.resize((200, 200) , Image.ANTIALIAS)
            imResize.save("datasets/test/"+file ,'JPEG',quality=90)



convnet = input_data(shape=[None, 200, 200, 3], name='input')

convnet = conv_2d(convnet, 32, 2, strides=[1, 2, 2, 1], activation='relu')
convnet = max_pool_2d(convnet, [1, 2, 2, 1])

convnet = conv_2d(convnet, 64, 2, strides=[1, 2, 2, 1], activation='relu')
convnet = avg_pool_2d(convnet, [1, 2, 2, 1])

convnet = conv_2d(convnet, 128, 2, strides=[1, 2, 2, 1], activation='relu')
convnet = max_pool_2d(convnet, [1, 2, 2, 1])

convnet = conv_2d(convnet, 256, 2, strides=[1, 2, 2, 1], activation='relu')
convnet = max_pool_2d(convnet, [1, 2, 2, 1])

convnet = conv_2d(convnet, 128, 2, strides=[1, 2, 2, 1], activation='relu')
convnet = max_pool_2d(convnet, [1, 2, 2, 1])

convnet = conv_2d(convnet, 64, 2, strides=[1, 2, 2, 1], activation='relu')
convnet = avg_pool_2d(convnet, [1, 2, 2, 1])

convnet = conv_2d(convnet, 32, 2, strides=[1, 2, 2, 1], activation='relu')
convnet = max_pool_2d(convnet, [1, 2, 2, 1])
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, learning_rate=0.001, loss='categorical_crossentropy', optimizer='adam')

model = tflearn.DNN(convnet)

def read(name):
    images = []
    print("Reading " + name + "...")
    i = 0
    for file in listdir("datasets/" + name):
        if not file.startswith('.') and file != 'Thumbs.db':
            i = i+1
            print(i, ":"+file)
            images = Image.open("datasets/" + name + "/" + file)
            images = np.asarray(images)
            data.append(images)
            if name == "cat":
                label.append([1, 0])
            else:
                label.append([0, 1])

def learn():
    # model learn
    read("cat")
    read("dog")
    model.fit(data, label,
              n_epoch=10, show_metric=True,
              snapshot_step=20, batch_size=100, run_id='cat_dog')
    model.save("cat_dog")


def predict():
    model.load('cat_dog')
    with open('submission_file.csv', 'w') as f:
        f.write('id,label\n')
    with open('submission_file.csv', 'a') as f:
        for file in listdir("datasets/test"):
            image = Image.open("datasets/test/"+file)
            image = np.asarray(image)
            image = image.reshape(-1, 200, 200, 3)
            #p = np.argmax(model.predict(image))
            mod = model.predict(image)[0][1]
            f.write('{},{}\n'.format(file.split(".")[0], mod))
            print(file+":", mod)








#learn()

predict()