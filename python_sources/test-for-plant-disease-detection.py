import io, os, pickle, string
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array

default_image_size = tuple((256, 256))
saved_classifier_model = pickle.load(open("../input/plant-disease-detection-using-keras/cnn_model.pkl",'rb'))
label_binarizer = pickle.load(open("../input/plant-disease-detection-using-keras/label_transform.pkl",'rb'))
    
def convert_image(path):
    try:
        image = Image.open(path)
        if image is not None :
            image = image.resize(default_image_size, Image.ANTIALIAS)   
            image_array = img_to_array(image)
            return np.expand_dims(image_array, axis=0), None
        else :
            return None, "Error loading image file"
    except Exception as e:
        return None, str(e)

def classify(image_data):
    image_array, err_msg = convert_image(image_data)
    if err_msg is None :
        result = label_binarizer.inverse_transform(saved_classifier_model.predict(image_array))[0]
        if result == "Corn_healthy":
            print("No diseases detected")
        else:
            print(result[5:].replace("_", " ") + " detected")
    else:
        print("Error")
        
for i in range(5):
    classify("../input/cropscornnew/crop/" + string.ascii_lowercase[i] + ".jpg")