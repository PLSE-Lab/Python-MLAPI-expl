import numpy as np
import h5py

def group_datasets(dataset):
    set_size = len(dataset)
    #print("dataset: " + str(dataset) + " set_size: " + str(set_size))
    
    image_set = []
    label_set = []
    
    stop_point = 0
    
    for i in range(set_size):
        current_index = "X"+str(i)
        if(current_index in dataset):
            #print("current itteration: " + str(i)) this was just to make sure everything was working correctly
            current_img = dataset["X"+str(i)]
            current_label = dataset["Y"+str(i)]
            image_set.append(current_img)
            label_set.append(current_label)
            stop_point += 1
    
    image_array = np.array(image_set)
    label_array = np.array(label_set)
    
    return image_array, label_array

#This function was pulled from Andrew Ng's course series on deep learning and slightly modified
def load_dataset():
    train_dataset = h5py.File("../input/apples-h5py/testSet.h5", "r")
    train_set_x_orig, train_set_y_orig = group_datasets(train_dataset) # your train set features

    test_dataset = h5py.File("../input/apples-h5py/testSet.h5", "r")
    test_set_x_orig, test_set_y_orig = group_datasets(test_dataset) # your test set features

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig